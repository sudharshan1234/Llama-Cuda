#ifndef TRANSFORMER_BLOCK_CUH
#define TRANSFORMER_BLOCK_CUH

#include <cuda_runtime.h>
#include "multi_head_attention.cuh"
#include "rms_norm.cuh"
#include "rope.cuh"

// Kernels
__global__ void ffn_kernel(float* output, float* input, float* weight1, float* weight2, float* bias1, float* bias2, int B, int T, int C);

// Kernel Caller Functions
void transformer_block_forward(
    float* output, float* input, 
    float* attn_weight_q, float* attn_weight_k, float* attn_weight_v, float* attn_weight_o, 
    float* ffn_weight1, float* ffn_weight2, float* ffn_bias1, float* ffn_bias2, 
    float* rms_norm_weight, float* rms_norm_bias, 
    int B, int T, int C, int head_dim, int num_heads, int block_size
);

#endif