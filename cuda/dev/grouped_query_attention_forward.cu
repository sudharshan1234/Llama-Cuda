#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "../../include/cuda_utils.cuh"
#include <float.h>

void matrix_multiply(float* a, float* b, float* result, int m, int n, int p) {
    // Multiply matrix `a` of size m x n with matrix `b` of size n x p
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i * p + j] = 0;
            for (int k = 0; k < n; k++) {
                result[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}

void softmax(float* x, int length) {
    float max_val = x[0];
    for (int i = 1; i < length; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < length; i++) {
        x[i] /= sum;
    }
}

void multi_head_attention_forward_cpu(
    float* inputs, float* weight_q, float* weight_k, float* weight_v, float* weight_o,
    float* output, int B, int T, int C, int head_dim, int num_heads
) {
    int total_dim = num_heads * head_dim;
    
    // Allocate memory for Q, K, V, and attention matrices
    float* Q = (float*)malloc(B * T * total_dim * sizeof(float));
    float* K = (float*)malloc(B * T * total_dim * sizeof(float));
    float* V = (float*)malloc(B * T * total_dim * sizeof(float));
    float* attention_scores = (float*)malloc(B * num_heads * T * T * sizeof(float));
    float* attention_output = (float*)malloc(B * T * total_dim * sizeof(float));
    
    // Step 1: Linear transformations for Q, K, V for each head
    for (int b = 0; b < B; b++) {
        matrix_multiply(&inputs[b * T * C], weight_q, &Q[b * T * total_dim], T, C, total_dim);
        matrix_multiply(&inputs[b * T * C], weight_k, &K[b * T * total_dim], T, C, total_dim);
        matrix_multiply(&inputs[b * T * C], weight_v, &V[b * T * total_dim], T, C, total_dim);
    }

    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, num_heads, T, T)
    // output is (B, T, C)
    float scale = 1.0 / sqrtf(head_dim);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < num_heads; h++) {
                const float* query_t = Q + b * T * C + t * C + h * head_dim;
                float* preatt_bth = attention_scores + b*num_heads*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = K + b * T * C + t2 * C + h * head_dim; // +C because it's key

                    // (Q) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    preatt_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        preatt_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        preatt_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = attention_output + b * T * C + t * C + h * head_dim;
                for (int i = 0; i < head_dim; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = V + b * T * C + t2 * C + h * head_dim;
                    float att_btht2 = preatt_bth[t2];
                    for (int i = 0; i < head_dim; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }

    // Step 5: Apply final linear projection by multiplying with weight_o
    for (int b = 0; b < B; b++) {
        matrix_multiply(&attention_output[b * T * total_dim], weight_o, &output[b * T * C], T, total_dim, C);
    }

    // Free allocated memory
    free(Q);
    free(K);
    free(V);
    free(attention_scores);
    free(attention_output);
}

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
    K += b * T * num_heads * head_dim + t2 * num_heads * head_dim + h * head_dim;  // Key vector

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
        max_val = fmaxf(max_val, input[idx * T + i]);
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
        sum_val += output[idx * T + i];
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
    for(int i=tid; i < T; i+=block_size){
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

    matmul_forward(Q, input, weight_q, B, T, C, num_heads*head_dim);
    matmul_forward(K, input, weight_k, B, T, C, num_heads*head_dim);
    matmul_forward(V, input, weight_v, B, T, C, num_heads*head_dim);


    float* attention_scores;
    float* softmax_output;
    cudaCheck(cudaMalloc(&attention_scores, B * num_heads * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&softmax_output, B * num_heads * T * T * sizeof(float)));

    int total_elements = B * num_heads * T * T;
    dim3 attention_blocks((total_elements + block_size - 1) / block_size);
    attention_query_key_kernel1<<<attention_blocks, block_size>>>(
        Q, K, attention_scores, B, T, C, head_dim, num_heads);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    int softmax_shared_memory_size = block_size * sizeof(float);
    softmax_query_key_kernel3<<<B * num_heads * T, block_size, softmax_shared_memory_size>>>(
        attention_scores, softmax_output, B, T, C, head_dim, num_heads, block_size);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
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
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    
    // softmax_output (B NH T T) V_reordered (B NH_kv T head_dim)
    matmul_forward(attention_output, softmax_output, V_reordered, B * num_heads, T, T, head_dim);
    
    permute_kernel<<<blocks, threads>>>(
        attention_output,             // [B, NH, T, D]
        attention_output_permuted,    // [B, T, NH, D]
        B, num_heads, T, head_dim
    );
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    // attention_output [B T NH head_dim] weight_o [NH * head_dim C]
    matmul_forward(output, attention_output_permuted, weight_o, B, T, num_heads * head_dim, C);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attention_scores);
    cudaFree(softmax_output);
    cudaFree(attention_output);

}

int main() {
    srand(0);

    int B = 1;
    int T = 2048;
    int C = 768;
    int head_dim = 64;
    int num_heads = 12;
    int total_dim = num_heads * head_dim;
    int deviceIdx = 0;
    float eps = 1e-6;

    cudaCheck(cudaSetDevice(deviceIdx));

    // Allocate host memory
    float* input = make_random_float(B * T * C);
    float* weight_q = make_random_float(C * total_dim);
    float* weight_k = make_random_float(C * total_dim);
    float* weight_v = make_random_float(C * total_dim);
    float* weight_o = make_random_float(total_dim * C);
    float* out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* out_gpu = (float*)malloc(B * T * C * sizeof(float));

    // Allocate GPU memory
    float *d_input, *d_weight_q, *d_weight_k, *d_weight_v, *d_weight_o, *d_out;
    cudaCheck(cudaMalloc(&d_input, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_q, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_k, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_v, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_o, total_dim * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));

    // Copy data to GPU
    cudaCheck(cudaMemcpy(d_input, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_q, weight_q, C * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_k, weight_k, C * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_v, weight_v, C * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_o, weight_o, total_dim * C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // CPU validation
    multi_head_attention_forward_cpu(input, weight_q, weight_k, weight_v, weight_o,
                                     out_cpu, B, T, C, head_dim, num_heads);

    // Validate kernel correctness at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        multi_head_attention_forward_gpu1(d_input, d_weight_q, d_weight_k, d_weight_v, d_weight_o,
                                          d_out, B, T, C, head_dim, num_heads, block_size);
        cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

        validate_result(d_out, out_cpu, "multi_head_attention_output", B * T * C, eps);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // Benchmark kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 200;
        float elapsed_time = benchmark_kernel(repeat_times, multi_head_attention_forward_gpu1,
                                              d_input, d_weight_q, d_weight_k, d_weight_v, d_weight_o,
                                              d_out, B, T, C, head_dim, num_heads, block_size);

        // Napkin math: estimate the memory bandwidth achieved
        long memory_ops = (
            4 * B * T * C + 
            4 * C * (num_heads * head_dim) + 
            8 * B * T * (num_heads * head_dim) + 
            4 * B * num_heads * T * T
        ) * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // Free memory
    free(input);
    free(weight_q);
    free(weight_k);
    free(weight_v);
    free(weight_o);
    free(out_cpu);
    free(out_gpu);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_weight_q));
    cudaCheck(cudaFree(d_weight_k));
    cudaCheck(cudaFree(d_weight_v));
    cudaCheck(cudaFree(d_weight_o));
    cudaCheck(cudaFree(d_out));

    return 0;
}