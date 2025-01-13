#ifndef MULTI_HEAD_ATTENTION_CUH
#define MULTI_HEAD_ATTENTION_CUH

#include <cuda_runtime.h>

// Kernels
__global__ void attention_query_key_kernel1(float* Q, float* K, float* output, int B, int T, int C, int head_dim, int num_heads);
__global__ void softmax_query_key_kernel1(float* input, float* output, int B, int T, int C, int head_dim, int num_heads, int block_size);
__global__ void softmax_query_key_kernel2(float* input, float* output, int B, int T, int C, int head_dim, int num_heads, int block_size);

// Kernel Caller Functions
void multi_head_attention_forward_gpu1(
    float* input, float* weight_q, float* weight_k, float* weight_v, float* weight_o,
    float* output, int B, int T, int C, int head_dim, int num_heads, int block_size
);

#endif