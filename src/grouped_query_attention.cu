#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "cuda_utils.cuh"
#include <float.h>
#include "rope.cuh"
#include "grouped_query_attention.cuh"

__global__ void attention_query_key_kernel1(float* Q, float* K, float* output, int B, int T, int C, int head_dim, int num_heads) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int num_elements = B * num_heads * T * T;
    if (idx >= num_elements) return;

    // Compute indices
    int b = idx / (num_heads * T * T);       // Batch index
    int h = (idx / (T * T)) % num_heads;     // Head index
    int t1 = (idx / T) % T;                  // Query timestep
    int t2 = idx % T;                        // Key timestep

    // Mask upper triangle
    if (t2 > t1) {
        output[idx] = -INFINITY;
        return;
    }

    // Compute Q and K offsets
    Q += b * T * num_heads * head_dim + t1 * num_heads * head_dim + h * head_dim;  // Query vector
    K += b * T * num_heads * head_dim + t2 * num_heads * head_dim + (h/4) * head_dim;  // Key vector

    // Dot product: Q · K
    float val = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        val += Q[i] * K[i];
    }
    val *= 1.0f / sqrtf(static_cast<float>(head_dim));  // Scale

    // Store the result
    output[idx] = val;
}

__global__ void softmax_query_key_kernel3(float *input, float *output, int B, int T, int C, int head_dim, int num_heads, int block_size) {
    int idx = blockIdx.x; // B NH T
    int tid = threadIdx.x; // [0, block_size)
    if (idx >= B * num_heads * T) {
        return;
    }
    extern __shared__ float shared[];
    float max_val = -INFINITY;
    // thread coarsing.
    for(int i = tid; i < T; i+=block_size){
        max_val = fmaxf(max_val, input[idx * T + tid]);
    }
    shared[tid] = max_val;
    __syncthreads();
    for(int stride = block_size / 2; stride > 0; stride /= 2){

        if(tid < stride){
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    max_val = shared[0];
    for(int i=tid; i < T; i++){
        output[idx * T + i] = expf(input[idx * T + i] - max_val);
    }
    __syncthreads();
    float sum_val = 0;
    for(int i=tid; i < T; i+=block_size){
        sum_val += input[idx * T + i];
    }
    shared[tid] = sum_val;
    __syncthreads();
    for(int i=block_size / 2; i>0; i=i/2){
        if(tid < i){
            shared[tid] = shared[tid] + shared[tid + i];
        }
        __syncthreads();
    }
    float sum = shared[0];
    for(int i=tid; i < T; i++){
        output[idx * T + i] /= sum;
    }


}

void matmul_forward(float* output, float* input, float* weight, int B, int T, int C, int OC){
    cublasHandle_t handle = createCublasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, input, C, &beta, output, OC));
    cublasDestroy(handle);

}

__global__ void permute_V(const float* __restrict__ V_in,
    float* __restrict__ V_out,
    int B, int T, int NH, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * NH * D;

    if (idx >= total_elements) return;

    int d = idx % D;
    int h = (idx / D) % NH;
    int t = (idx / (D * NH)) % T;
    int b = idx / (D * NH * T);

    int in_idx  = b * T * NH * D + t * NH * D + h * D + d;
    int out_idx = b * NH * T * D + h * T * D + t * D + d;

    V_out[out_idx] = V_in[in_idx];
}

__global__ void permute_kernel(const float* matrix, float* out_matrix,
    int dim1, int dim2, int dim3, int dim4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim1 * dim2 * dim3 * dim4;

    if (idx >= total) return;

    // Original indices (dim1, dim2, dim3, dim4)
    int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
    int i2 = (idx / (dim3 * dim4)) % dim2;
    int i3 = (idx / dim4) % dim3;
    int i4 = idx % dim4;

    // For attention_output: [B, NH, T, D]
    // Permute to:           [B, T, NH, D]
    // Map: (b, h, t, d) → (b, t, h, d)

    int permuted_idx = 
    i1 * (dim3 * dim2 * dim4) +  // B
    i3 * (dim2 * dim4) +         // T
    i2 * dim4 +                  // NH
    i4;                          // D

    out_matrix[permuted_idx] = matrix[idx];
}

void multi_head_attention_forward_gpu1(
    float* input, float* weight_q, float* weight_k, float* weight_v, float* weight_o,
    float* output, int B, int T, int C, int head_dim, int num_heads, int block_size
) {

    int qkv_size = B * T * C; 
    float *Q, *K, *V;
    cudaCheck(cudaMalloc(&Q, qkv_size * sizeof(float)));
    cudaCheck(cudaMalloc(&K, qkv_size * sizeof(float)));
    cudaCheck(cudaMalloc(&V, qkv_size * sizeof(float)));

    float* freq_inv = (float*)malloc((C / 2) * sizeof(float));
    for (int i = 0; i < C / 2; i++) {
        freq_inv[i] = 1.0f / powf(10000.0f, 2.0f * i / C);
    }
    float *d_freq_inv;
    cudaCheck(cudaMalloc((void**)&d_freq_inv, (C / 2) * sizeof(float)));
    cudaCheck(cudaMemcpy(d_freq_inv, freq_inv, (C / 2) * sizeof(float), cudaMemcpyHostToDevice));

    matmul_forward(Q, input, weight_q, B, T, C, num_heads*head_dim);
    matmul_forward(K, input, weight_k, B, T, C, num_heads*head_dim);
    matmul_forward(V, input, weight_v, B, T, C, num_heads*head_dim);

    rope_forward(2, Q, Q, B, T, num_heads * head_dim, d_freq_inv, block_size); // Apply RoPE to Q
    rope_forward(2, K, K, B, T, num_heads * head_dim, d_freq_inv, block_size); // Apply RoPE to K

    float* attention_scores;
    float* softmax_output;
    cudaCheck(cudaMalloc(&attention_scores, B * num_heads * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&softmax_output, B * num_heads * T * T * sizeof(float)));

    int total_elements = B * num_heads * T * T;
    dim3 attention_blocks((total_elements + block_size - 1) / block_size);
    attention_query_key_kernel1<<<attention_blocks, block_size>>>(
        Q, K, attention_scores, B, T, C, head_dim, num_heads);
    int softmax_shared_memory_size = block_size * sizeof(float);
    softmax_query_key_kernel3<<<B * T * num_heads, block_size, softmax_shared_memory_size>>>(
        attention_scores, softmax_output, B, T, C, head_dim, num_heads, block_size);
    float* attention_output;
    cudaMalloc(&attention_output, B * num_heads * T * head_dim * sizeof(float));

    float* V_reordered;
    float* attention_output_permuted;
    cudaCheck(cudaMalloc(&V_reordered, B * num_heads * T * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&attention_output_permuted, B * num_heads * T * head_dim * sizeof(float)));
    int total = B * T * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_V<<<blocks, threads>>>(V, V_reordered, B, T, num_heads, head_dim);
    
    // softmax_output (B NH T T) V_reordered (B NH_kv T head_dim)
    matmul_forward(attention_output, softmax_output, V_reordered, B * num_heads, T, T, head_dim);
    
    permute_kernel<<<blocks, threads>>>(
        attention_output,             // [B, NH, T, D]
        attention_output_permuted,    // [B, T, NH, D]
        B, num_heads, T, head_dim
    );
    // attention_output [B T NH head_dim] weight_o [NH * head_dim C]
    matmul_forward(output, attention_output_permuted, weight_o, B, T, num_heads * head_dim, C);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attention_scores);
    cudaFree(softmax_output);
    cudaFree(attention_output);
    cudaFree(attention_output_permuted);
    cudaFree(attention_output_permuted);
    cudaFree(V_reordered);
    cudaFree(d_freq_inv);

    free(freq_inv);

}