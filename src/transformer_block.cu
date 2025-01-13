#include "../include/transformer_block.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include "cuda_utils.cuh"

// GELU Activation Function
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

// FFN Kernel
__global__ void ffn_kernel(float* output, float* input, float* weight1, float* weight2, float* bias1, float* bias2, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;

    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;

    // First Linear Layer
    float val = 0.0f;
    for (int i = 0; i < C; i++) {
        val += input[b * T * C + t * C + i] * weight1[i * C + c];
    }
    val += bias1[c];
    val = gelu(val); // GELU Activation

    // Second Linear Layer
    float out_val = 0.0f;
    for (int i = 0; i < C; i++) {
        out_val += val * weight2[i * C + c];
    }
    out_val += bias2[c];

    output[idx] = out_val;
}

// Transformer Block Forward Pass
void transformer_block_forward(
    float* output, float* input, 
    float* attn_weight_q, float* attn_weight_k, float* attn_weight_v, float* attn_weight_o, 
    float* ffn_weight1, float* ffn_weight2, float* ffn_bias1, float* ffn_bias2, 
    float rms_norm_weight, float rms_norm_bias, 
    int B, int T, int C, int head_dim, int num_heads, int block_size
) {
    // Allocate memory for intermediate outputs
    float *attn_output, *norm_output1, *ffn_output, *norm_output2;
    cudaCheck(cudaMalloc(&attn_output, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&norm_output1, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&ffn_output, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&norm_output2, B * T * C * sizeof(float)));

    // Multi-Head Attention
    multi_head_attention_forward_gpu1(
        input, attn_weight_q, attn_weight_k, attn_weight_v, attn_weight_o, 
        attn_output, B, T, C, head_dim, num_heads, block_size
    );

    float eps = 1e-6f;
    float scale = 1.0f;
    float shift = 0.0f;

    // Add & Normalize (Residual Connection + RMS Norm)
    rms_norm_forward(3, norm_output1, attn_output, B, T, C, eps, scale, shift, block_size);

    // Feed-Forward Network
    ffn_kernel<<<(B * T * C + block_size - 1) / block_size, block_size>>>(
        ffn_output, norm_output1, ffn_weight1, ffn_weight2, ffn_bias1, ffn_bias2, B, T, C
    );

    // Add & Normalize (Residual Connection + RMS Norm)
    rms_norm_forward3(norm_output2, ffn_output, B, T, C, 1e-5f, rms_norm_weight, rms_norm_bias, block_size);

    // Copy final output
    cudaMemcpy(output, norm_output2, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free intermediate memory
    cudaFree(attn_output);
    cudaFree(norm_output1);
    cudaFree(ffn_output);
    cudaFree(norm_output2);
}