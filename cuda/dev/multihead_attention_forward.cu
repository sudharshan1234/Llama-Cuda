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
    if (idx == 0) {
        printf("Number of elements in Q: %d\n", B * T * C);
        printf("Number of elements in K: %d\n", B * T * C);
        printf("Number of elements in output: %d\n", B * num_heads * T * T);
        printf("B: %d, T: %d, C: %d, head_dim: %d, num_heads: %d\n", B, T, C, head_dim, num_heads);
    }
    if(idx >= B * num_heads * T * T){
        return;
    }
    int b = idx / (num_heads * T * T);
    int h = (idx / (T * T)) % num_heads;
    int t1 = (idx / T) % T;
    int t2 = idx % T;

    if (idx < 10) { // Print for a few threads
        printf("idx: %d, b: %d, h: %d, t1: %d, t2: %d\n", idx, b, h, t1, t2);
    }
    if(t2 > t1){
        output[idx] = -INFINITY;
        return;
    }

    // h * head_dim = c
    Q += b * T * C + t1*C + h * head_dim;
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
    for(int i=tid; i<T; i += block_size){
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
    for(int i=tid; i<T; i += block_size){
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

    sum = sum_val[0];

    for (int i = tid; i < T; i += block_size) {
        output[i] = output[i] / sum;
    }

}

void multi_head_attention_forward_gpu1(
    float* input, float* weight_q, float* weight_k, float* weight_v, float* weight_o,
    float* output, int B, int T, int C, int head_dim, int num_heads, int block_size
) {

    int qkv_size = B * T * C; 
    int batch_count = B * num_heads;
    float *Q, *K, *V;
    cudaCheck(cudaMalloc(&Q, qkv_size * sizeof(float)));
    cudaCheck(cudaMalloc(&K, qkv_size * sizeof(float)));
    cudaCheck(cudaMalloc(&V, qkv_size * sizeof(float)));

    cublasHandle_t handle = createCublasHandle();
    cudaCheck(cudaDeviceSynchronize());

    // Leading dimensions and strides
    int lda = T;                   // Leading dimension of input matrices
    int ldb = C;                   // Leading dimension of weight_q
    int ldc = T;                   // Leading dimension of output matrices
    long long int strideA = T * C; // Stride between batches for input
    long long int strideB = 0;     // Stride for shared weight_q (constant across batches)
    long long int strideC = T * head_dim; // Stride between batches for output

    // GEMM parameters
    int m = T;               // Number of rows of the output matrix
    int n = head_dim;        // Number of columns of the output matrix
    int k = C;               // Inner dimension of the matrix multiplication
    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute Q
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_q, ldb, strideB, // Weight matrix Q C Num_head*head dim
        input, lda, strideA,    // Input matrix B T C
        &beta,                  // Beta
        Q, ldc, strideC,        // Output Q
        B                       // Batch count
    ));
    cudaCheck(cudaDeviceSynchronize());

    // Compute K = input * weight_k
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_k, ldb, strideB, // Weight matrix K
        input, lda, strideA,    // Input matrix
        &beta,                  // Beta
        K, ldc, strideC,        // Output K
        B                       // Batch count
    ));

    // Compute V = input * weight_v
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_v, ldb, strideB, // Weight matrix V
        input, lda, strideA,    // Input matrix
        &beta,                  // Beta
        V, ldc, strideC,        // Output V
        B                       // Batch count
    ));

    float* attention_scores;
    float* softmax_output;
    cudaCheck(cudaMalloc(&attention_scores, B * num_heads * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&softmax_output, B * num_heads * T * T * sizeof(float)));

    int total_elements = B * num_heads * T * T;
    dim3 attention_blocks((total_elements + block_size - 1) / block_size);
    attention_query_key_kernel1<<<attention_blocks, block_size>>>(
        Q, K, attention_scores, B, T, C, head_dim, num_heads);
    cudaCheck(cudaDeviceSynchronize());
    int softmax_shared_memory_size = 2 * (block_size / 32) * sizeof(float);
    softmax_query_key_kernel<<<B * num_heads, block_size, softmax_shared_memory_size>>>(
        attention_scores, softmax_output, B, T, C, head_dim, num_heads, block_size);
    cudaCheck(cudaDeviceSynchronize());
    float* attention_output;
    cudaMalloc(&attention_output, B * num_heads * T * head_dim * sizeof(float));

    // Attention output (B * num_head, T, head_dim) = softmax(scores) (B * num_head, T, T) * V (B * num_head, T, head_dim)
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,               // No transpose for softmax(scores)
        CUBLAS_OP_N,               // No transpose for V
        head_dim, T, T,                   // Dimensions of GEMM (head_dim, T, T)
        &alpha,                    // Alpha
        V, head_dim, strideC,           // V matrix
        softmax_output, T, T * T, // Softmax scores matrix
        &beta,                     // Beta
        attention_output, head_dim, strideC, // Attention output matrix
        batch_count                       // Batch count
    ));

    cudaCheck(cudaDeviceSynchronize());
    // Concatenate heads and project back to output space: output (B , T, C)  = attention_output (B * num_head, T, head_dim) * weight_o (head_dim * num_head, C)
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        C, T, num_heads * head_dim,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_o, C, 0, // Weight matrix V
        attention_output, head_dim, T * head_dim,    // Input matrix
        &beta,                  // Beta
        output, C, T * head_dim,        // Output V
        B                       // Batch count
    ));
    cudaCheck(cudaDeviceSynchronize());

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attention_scores);
    cudaFree(softmax_output);
    cudaFree(attention_output);

    cublasDestroy(handle);

}

int main() {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int head_dim = 64;
    int num_heads = 12;
    int total_dim = num_heads * head_dim;
    float eps = 1e-6;
    int deviceIdx = 0;

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
    // multi_head_attention_forward_cpu(input, weight_q, weight_k, weight_v, weight_o,
    //                                  out_cpu, B, T, C, head_dim, num_heads);

    // Validate kernel correctness at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        multi_head_attention_forward_gpu1(d_input, d_weight_q, d_weight_k, d_weight_v, d_weight_o,
                                          d_out, B, T, C, head_dim, num_heads, block_size);
        cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

        validate_result(out_cpu, out_gpu, "multi_head_attention_output", B * T * C, eps);
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
        long memory_ops = (2 * B * T * C + 2 * C * total_dim) * sizeof(float);
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