#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "cuda_utils.cuh"
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

__global__ void softmax_query_key_kernel1(float *input, float *output, int B, int T, int C, int head_dim, int num_heads, int block_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * num_heads;

    if (idx < total_threads) {
        int h = idx % num_heads;
        int t = (idx / num_heads) % T;
        int b = idx / (num_heads * T);

        const float* input_temp = input + b*num_heads*T*T + h*T*T + t*T;
        float* output_temp = output + b*num_heads*T*T + h*T*T + t*T;

        // find maxval
        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++) {
            if (input_temp[t2] > maxval) {
                maxval = input_temp[t2];
            }
        }

        // calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = expf(input_temp[t2] - maxval);
            expsum += expv;
            output_temp[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                output_temp[t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                output_temp[t2] = 0.0f;
            }
        }
    }
}

__global__ void softmax_query_key_kernel2(float *input, float *output, int B, int T, int C, int head_dim, int num_heads, int block_size){
    extern __shared__ float shared[]; // 2 * block_size/32
    int tid = threadIdx.x; //0 - block_size
    int idx = blockIdx.x; // b*num_heads*T + n*T + t1
    int warpId = tid / 32;
    int laneId = tid % 32;
    input += idx * T;
    output += idx * T;
    float *max_val = &shared[0];
    float *sum_val = &shared[block_size/32];
    float max_num = -INFINITY;
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
        float val = expf(input[i] - max_num);
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

    long long int strideA = T * C; // Stride between batches for input
    long long int strideB = 0;     // Stride for shared weight_q (constant across batches)
    long long int strideC = T * num_heads * head_dim; // Stride between batches for output

    // GEMM parameters
    int m = T;               // Number of rows of the output matrix
    int n = num_heads * head_dim;        // Number of columns of the output matrix
    int k = C;               // Inner dimension of the matrix multiplication
    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute Q (B, T, num_heads * head_dim) = Input (B, T, C) * Wq (C, num_heads * head_dim)
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_q, n, strideB, // Weight matrix C num_heads*head dim
        input, k, strideA,    // Input matrix B T C
        &beta,                  // Beta
        Q, n, strideC,        // Output Q
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
        weight_k, n, strideB, // Weight matrix C num_heads*head dim
        input, k, strideA,    // Input matrix B T C
        &beta,                  // Beta
        K, n, strideC,        // Output Q
        B                       // Batch count
    ));

    // Compute V = input * weight_v
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, m, k,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_v, n, strideB, // Weight matrix C num_heads*head dim
        input, k, strideA,    // Input matrix B T C
        &beta,                  // Beta
        V, n, strideC,        // Output Q
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
    softmax_query_key_kernel2<<<B * T * num_heads, block_size, softmax_shared_memory_size>>>(
        attention_scores, softmax_output, B, T, C, head_dim, num_heads, block_size);
    cudaCheck(cudaDeviceSynchronize());
    float* attention_output;
    cudaMalloc(&attention_output, B * num_heads * T * head_dim * sizeof(float));

    // Attention output (B * num_heads, T, head_dim) = softmax(scores) (B * num_heads, T, T) * V (B * num_heads, T, head_dim)
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,               // No transpose for softmax(scores)
        CUBLAS_OP_N,               // No transpose for V
        head_dim, T, T,                   // Dimensions of GEMM (head_dim, T, T)
        &alpha,                    // Alpha
        V, head_dim, T * head_dim,           // V matrix
        softmax_output, T, T * T, // Softmax scores matrix
        &beta,                     // Beta
        attention_output, head_dim, T * head_dim, // Attention output matrix
        batch_count                       // Batch count
    ));

    cudaCheck(cudaDeviceSynchronize());
    // Concatenate heads and project back to output space: output (B , T, C)  = attention_output (B * num_heads, T, head_dim) * weight_o (head_dim * num_heads, C)
    cublasCheck(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        C, T, num_heads * head_dim,                // Dimensions of GEMM
        &alpha,                 // Alpha
        weight_o, C, strideB, // Weight matrix V
        attention_output, num_heads * head_dim, T * num_heads * head_dim,    // Input matrix
        &beta,                  // Beta
        output, C, T * C,        // Output V
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

    int B = 2;
    int T = 4;
    int C = 1;
    int head_dim = 1;
    int num_heads = 1;
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