#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "cuda_utils.h"

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
    float* attention_scores = (float*)malloc(T * T * sizeof(float));
    float* attention_output = (float*)malloc(B * T * total_dim * sizeof(float));
    
    // Step 1: Linear transformations for Q, K, V for each head
    for (int b = 0; b < B; b++) {
        matrix_multiply(&inputs[b * T * C], weight_q, &Q[b * T * total_dim], T, C, total_dim);
        matrix_multiply(&inputs[b * T * C], weight_k, &K[b * T * total_dim], T, C, total_dim);
        matrix_multiply(&inputs[b * T * C], weight_v, &V[b * T * total_dim], T, C, total_dim);
    }

    // Process each batch and head separately
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < num_heads; h++) {
            int head_offset = h * head_dim;

            // Step 2: Compute attention scores (Q * K^T / sqrt(d_k))
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    attention_scores[i * T + j] = 0.0;
                    for (int k = 0; k < head_dim; k++) {
                        attention_scores[i * T + j] += Q[b * T * total_dim + i * total_dim + head_offset + k] *
                                                       K[b * T * total_dim + j * total_dim + head_offset + k];
                    }
                    attention_scores[i * T + j] /= sqrtf((float)head_dim);
                }
            }

            // Step 3: Apply softmax to each row of the attention scores
            for (int i = 0; i < T; i++) {
                softmax(&attention_scores[i * T], T);
            }

            // Step 4: Compute attention output (softmax(QK^T) * V)
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < head_dim; j++) {
                    attention_output[b * T * total_dim + i * total_dim + head_offset + j] = 0.0;
                    for (int k = 0; k < T; k++) {
                        attention_output[b * T * total_dim + i * total_dim + head_offset + j] +=
                            attention_scores[i * T + k] * V[b * T * total_dim + k * total_dim + head_offset + j];
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

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void attention_query_key_kernel1(float* Q, float* K, float* output, int B, int T, int C, int head_dim, int num_heads){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx > B * num_heads * T * T){
        return;
    }
    int b = idx / (num_heads * T * T);
    int h = (idx / (T * T)) % num_heads;
    int t1 = (idx / T) % T;
    int t2 = idx % T;

    if(t2 > t1){
        output[idx] = -INFINITY;
        return;
    }

    // h * head_dim = c
    Q += b * T * C + t*C + h * head_dim;
    K += b * T * C + t2*C + h * head_dim;

    // (q) dot (k)
    float val = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        val += Q[i] * K[i];
    }
    val *= 1.0 / sqrtf(head_dim);

    output[idx] = val;
}

__global__ void softmax_query_key_kernel(float *input, float *output, int B, int T, int C, int head_dim, int num_heads, int block_size){
    extern __shared__ float shared[]; // 2 * block_size/32
    int tid = threadIdx.x; //0 - block_size
    int idx = blockIdx.x; // b*NH*T + n*T + t1
    int warpId = tid / 32;
    int laneId = tid % 32;
    input += idx * T;
    output += idx * T;
    float *max_val = &shared[0];
    float *sum_val = &shared[block_size/32];
    int max_num = 0;
    for(int i=tid; i<T, i += block_size){
        max_num = fmaxf(max_num, input[i]);
    }

    max_num = warpReduceMax(max_num);

    if (laneId == 0){
        max_val[warpId] = max_num;
    }
    __syncthreads();

    // Now each warp has stored its maximum in `max_val[warpId]`
    // Perform another reduction across the warps
    if (tid == 0) {  // Only the first thread in the first warp participates
        max_num = max_val[0];
        for (int i = 1; i < block_size / 32; i++) {  // Loop over all warp results
            max_num = fmaxf(max_num, max_val[i]);
        }
        max_val[0] = max_num;  // Store the final block-wide maximum in max_val[0]
    }
    __syncthreads();

    // Now max_val[0] contains the maximum value across the block
    max_num = max_val[0];  // Broadcast the result to all threads in the block
    
    
    float sum = 0.0f;
    for(int i=tid; i<T, i += block_size){
        int val = exp(input[i] - max_num);
        output[i] = val;
        sum += val;
    }
    sum = warpReduceSum(sum);

    if (laneId == 0){
        sum_val[warpId] = sum;
    }
    __syncthreads();

    // Now each warp has stored its sum in `sum_val[warpId]`
    // Perform another reduction across the warps
    if (tid == 0) {  // Only the first thread in the first warp participates
        sum = sum_val[0];
        for (int i = 1; i < block_size / 32; i++) {  // Loop over all warp results
            sum += sum_val[i];
        }
        sum_val[0] = sum;  // Store the final block-wide maximum in max_val[0]
    }
    __syncthreads();

    sum = sumvals[0];

    for (int i = tid; i < T; i += block_size) {
        output[i] = output[i] / sum;
    }

}

void multi_head_attention_forward_gpu1(
    float* input, float* weight_q, float* weight_k, float* weight_v, float* weight_o,
    float* output, int B, int T, int C, int head_dim, int num_heads, int block_size
) {

    int qkv_size = B * T * C; 
    int HxD = num_heads * head_dim;
    float alpha = 1.0f, beta = 0.0f;
    float *Q, *K, *V;
    cudaMalloc(&Q, qkv_size * sizeof(float));
    cudaMalloc(&K, qkv_size * sizeof(float));
    cudaMalloc(&V, qkv_size * sizeof(float));

    cublasHandle_t handle = createCublasHandle();
    // Using batched gemm for Q, K, V computation
    int batch_count = B * num_heads;
    int m = T;
    int n = head_dim;
    int k = C;
    int lda = C, ldb = C, ldc = head_dim;
    float alpha = 1.0f;
    float beta = 0.0f;
    size_t strideA = T * C;
    size_t strideB = C * head_dim;
    size_t strideC = T * head_dim;

    // Q = input * weight_q
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              weight_q, lda, strideB,
                              input, lda, strideA,
                              &beta, Q, ldc, strideC, batch_count);

    // K = input * weight_k
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              weight_k, lda, strideB,
                              input, lda, strideA,
                              &beta, K, ldc, strideC, batch_count);

    // V = input * weight_v
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              weight_v, lda, strideB,
                              input, lda, strideA,
                              &beta, V, ldc, strideC, batch_count);

    float* attention_scores;
    float* softmax_output;
    cudaMalloc(&attention_scores, B * num_heads * T * T * sizeof(float));
    cudaMalloc(&softmax_output, B * num_heads * T * T * sizeof(float));

    int total_elements = B * num_heads * T * T;
    dim3 attention_blocks((total_elements + block_size - 1) / block_size);
    attention_query_key_kernel1<<<attention_blocks, block_size>>>(
        Q, K, attention_scores, B, T, C, head_dim, num_heads);

    int softmax_shared_memory_size = 2 * (block_size / 32) * sizeof(float);
    softmax_query_key_kernel<<<attention_blocks, block_size, softmax_shared_memory_size>>>(
        attention_scores, softmax_output, B, T, C, head_dim, num_heads, block_size);

    float* attention_output;
    cudaMalloc(&attention_output, B * num_heads * T * head_dim * sizeof(float));
    
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, T, T, &alpha,
                              V, ldc, strideC,
                              softmax_output, T, T * T,
                              &beta, attention_output, ldc, strideC, batch_count);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, T, num_heads * head_dim, &alpha,
                              weight_o, C, 0,
                              attention_output, num_heads * head_dim, strideC,
                              &beta, output, C, strideA, B);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attention_scores);
    cudaFree(softmax_output);
    cudaFree(attention_output);

    cublasDestroy(handle);

}

int main(){
    // inputs -> B, T, C
    // Wq -> B, C, (H X D)
    // Wk -> B, C, (H X D)
    // Wv -> B, C, (H X D)
    // Q, K, V -> B, H, T, D
    // att -> B, H, T, T
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int total_dim = num_heads * head_dim;
    // Allocate memory for Q, K, V, and attention matrices
    int deviceIdx = 0;
    int kernel_num = 3;
    cudaCheck(cudaSetDevice(deviceIdx));
    // create host memory of random numbers
    float *input = (float*)malloc(B * T * C * sizeof(float));
    input = make_random_float(B * T * C);
    float *out_gpu = (float*)malloc(B * T * C * sizeof(float));
    destroyCublasHandle(handle);
    
}