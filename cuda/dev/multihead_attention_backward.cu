#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

// Backward implementation for multi-head attention
void multi_head_attention_backward_gpu(
    float* dOutput, float* input, float* Q, float* K, float* V, float* softmax_output, float* attention_scores,
    float* weight_q, float* weight_k, float* weight_v, float* weight_o,
    float* dInput, float* dWeight_q, float* dWeight_k, float* dWeight_v, float* dWeight_o,
    int B, int T, int C, int head_dim, int num_heads, int block_size
) {
    cublasHandle_t handle = createCublasHandle();
    
    int qkv_size = B * T * C;
    int HxD = num_heads * head_dim;

    float alpha = 1.0f;
    float beta = 0.0f;

    float *dQ, *dK, *dV, *dAttention, *dAttentionScores;
    cudaMalloc(&dQ, qkv_size * sizeof(float));
    cudaMalloc(&dK, qkv_size * sizeof(float));
    cudaMalloc(&dV, qkv_size * sizeof(float));
    cudaMalloc(&dAttention, B * num_heads * T * head_dim * sizeof(float));
    cudaMalloc(&dAttentionScores, B * num_heads * T * T * sizeof(float));

    // Compute gradient of output projection (weight_o)
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, C, HxD, T, &alpha,
                              dOutput, C, T * C,
                              V, HxD, T * HxD,
                              &beta, dWeight_o, C, HxD * C, B);

    // Backpropagate through output projection to dAttention
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, HxD, T, C, &alpha,
                              weight_o, C, HxD * C,
                              dOutput, C, T * C,
                              &beta, dAttention, HxD, T * HxD, B);

    // Backpropagate through attention mechanism to compute dV
    int num_blocks_dV = (B * num_heads * T * head_dim + block_size - 1) / block_size;
    compute_dV<<<num_blocks_dV, block_size>>>(dV, dAttention, softmax_output, B, T, head_dim, num_heads);

    cudaDeviceSynchronize();

    // Compute gradient of attention scores
    int num_blocks_dAttentionScores = (B * num_heads * T * T + block_size - 1) / block_size;
    compute_dAttentionScores<<<num_blocks_dAttentionScores, block_size>>>(dAttentionScores, dAttention, V, B, T, head_dim, num_heads);

    cudaDeviceSynchronize();

    // Backpropagate through scaled dot-product attention to compute dQ and dK
    int num_blocks_dQK = (B * num_heads * T * head_dim + block_size - 1) / block_size;
    compute_dQK<<<num_blocks_dQK, block_size>>>(dQ, dK, dAttentionScores, Q, K, B, T, head_dim, num_heads);

    cudaDeviceSynchronize();

    // Backpropagate through linear transformations for Q, K, V to compute dWeight_q, dWeight_k, dWeight_v
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, C, T, head_dim, &alpha,
                              input, C, T * C,
                              dQ, head_dim, T * head_dim,
                              &beta, dWeight_q, C, head_dim * C, B * num_heads);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, C, T, head_dim, &alpha,
                              input, C, T * C,
                              dK, head_dim, T * head_dim,
                              &beta, dWeight_k, C, head_dim * C, B * num_heads);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, C, T, head_dim, &alpha,
                              input, C, T * C,
                              dV, head_dim, T * head_dim,
                              &beta, dWeight_v, C, head_dim * C, B * num_heads);

    // Backpropagate gradients to input
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, T, head_dim, &alpha,
                              weight_q, C, head_dim * C,
                              dQ, head_dim, T * head_dim,
                              &beta, dInput, C, T * C, B * num_heads);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, T, head_dim, &alpha,
                              weight_k, C, head_dim * C,
                              dK, head_dim, T * head_dim,
                              &beta, dInput, C, T * C, B * num_heads);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, T, head_dim, &alpha,
                              weight_v, C, head_dim * C,
                              dV, head_dim, T * head_dim,
                              &beta, dInput, C, T * C, B * num_heads);

    // Cleanup
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dAttention);
    cudaFree(dAttentionScores);
    destroyCublasHandle(handle);
}

int main() {
    srand(0);

    // Define dimensions
    int B = 8;       // Batch size
    int T = 1024;    // Sequence length
    int C = 768;     // Input dimension
    int head_dim = 64;
    int num_heads = 12;
    int total_dim = num_heads * head_dim;
    int block_size = 256;  // Chosen block size for GPU kernels

    // Allocate host memory
    float* input = make_random_float(B * T * C);
    float* weight_q = make_random_float(C * total_dim);
    float* weight_k = make_random_float(C * total_dim);
    float* weight_v = make_random_float(C * total_dim);
    float* weight_o = make_random_float(total_dim * C);
    float* dOutput = make_random_float(B * T * C);
    float* Q = make_random_float(B * T * total_dim);
    float* K = make_random_float(B * T * total_dim);
    float* V = make_random_float(B * T * total_dim);
    float* softmax_output = make_random_float(B * num_heads * T * T);
    float* attention_scores = make_random_float(B * num_heads * T * T);

    float *dInput_cpu = (float*)malloc(B * T * C * sizeof(float));
    float *dWeight_q_cpu = (float*)malloc(C * total_dim * sizeof(float));
    float *dWeight_k_cpu = (float*)malloc(C * total_dim * sizeof(float));
    float *dWeight_v_cpu = (float*)malloc(C * total_dim * sizeof(float));
    float *dWeight_o_cpu = (float*)malloc(total_dim * C * sizeof(float));

    float *dInput_gpu = (float*)malloc(B * T * C * sizeof(float));
    float *dWeight_q_gpu = (float*)malloc(C * total_dim * sizeof(float));
    float *dWeight_k_gpu = (float*)malloc(C * total_dim * sizeof(float));
    float *dWeight_v_gpu = (float*)malloc(C * total_dim * sizeof(float));
    float *dWeight_o_gpu = (float*)malloc(total_dim * C * sizeof(float));

    // Allocate GPU memory
    float *d_input, *d_weight_q, *d_weight_k, *d_weight_v, *d_weight_o;
    float *d_Q, *d_K, *d_V, *d_softmax_output, *d_attention_scores, *d_dOutput;
    cudaCheck(cudaMalloc(&d_input, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_q, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_k, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_v, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight_o, total_dim * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_Q, B * T * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_K, B * T * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_V, B * T * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_softmax_output, B * num_heads * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_attention_scores, B * num_heads * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dOutput, B * T * C * sizeof(float)));

    float *d_dInput, *d_dWeight_q, *d_dWeight_k, *d_dWeight_v, *d_dWeight_o;
    cudaCheck(cudaMalloc(&d_dInput, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dWeight_q, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dWeight_k, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dWeight_v, C * total_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dWeight_o, total_dim * C * sizeof(float)));

    // Copy data to GPU
    cudaCheck(cudaMemcpy(d_input, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_q, weight_q, C * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_k, weight_k, C * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_v, weight_v, C * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight_o, weight_o, total_dim * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dOutput, dOutput, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_Q, Q, B * T * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_K, K, B * T * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_V, V, B * T * total_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_softmax_output, softmax_output, B * num_heads * T * T * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_attention_scores, attention_scores, B * num_heads * T * T * sizeof(float), cudaMemcpyHostToDevice));

    // CPU Backward Pass
    multi_head_attention_backward_cpu(dOutput, input, Q, K, V, softmax_output, attention_scores,
                                      weight_q, weight_k, weight_v, weight_o,
                                      dInput_cpu, dWeight_q_cpu, dWeight_k_cpu, dWeight_v_cpu, dWeight_o_cpu,
                                      B, T, C, head_dim, num_heads);

    // GPU Backward Pass
    multi_head_attention_backward_gpu(d_dOutput, d_input, d_Q, d_K, d_V, d_softmax_output, d_attention_scores,
                                       d_weight_q, d_weight_k, d_weight_v, d_weight_o,
                                       d_dInput, d_dWeight_q, d_dWeight_k, d_dWeight_v, d_dWeight_o,
                                       B, T, C, head_dim, num_heads, block_size);

    // Copy results back to host
    cudaCheck(cudaMemcpy(dInput_gpu, d_dInput, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dWeight_q_gpu, d_dWeight_q, C * total_dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dWeight_k_gpu, d_dWeight_k, C * total_dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dWeight_v_gpu, d_dWeight_v, C * total_dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dWeight_o_gpu, d_dWeight_o, total_dim * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate results
    validate_result(dInput_cpu, dInput_gpu, "dInput", B * T * C, 1e-5f);
    validate_result(dWeight_q_cpu, dWeight_q_gpu, "dWeight_q", C * total_dim, 1e-5f);
    validate_result(dWeight_k_cpu, dWeight_k_gpu, "dWeight_k", C * total_dim, 1e-5f);
    validate_result(dWeight_v_cpu, dWeight_v_gpu, "dWeight_v", C * total_dim, 1e-5f);
    validate_result(dWeight_o_cpu, dWeight_o_gpu, "dWeight_o", total_dim * C, 1e-5f);

    printf("All results match between CPU and GPU implementations.\n");

    // Free memory
    free(input);
    free(weight_q);
    free(weight_k);
    free(weight_v);
    free(weight_o);
    free(dOutput);
    free(Q);
    free(K);
    free(V);
    free(softmax_output);
    free(attention_scores);
    free(dInput_cpu);
    free(dWeight_q_cpu);
    free(dWeight_k_cpu);
    free(dWeight_v_cpu);
    free(dWeight_o_cpu);
    free(dInput_gpu);
    free(dWeight_q_gpu);
    free(dWeight_k_gpu);
    free(dWeight_v_gpu);
    free(dWeight_o_gpu);
    cudaFree(d_input);
    cudaFree(d_weight_q);
    cudaFree(d_weight_k);
    cudaFree(d_weight_v);
    cudaFree(d_weight_o);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_softmax_output);
    cudaFree(d_attention_scores);
    cudaFree(d_dOutput);
    cudaFree(d_dInput);
    cudaFree(d_dWeight_q);
    cudaFree(d_dWeight_k);
    cudaFree(d_dWeight_v);
    cudaFree(d_dWeight_o);

    return 0;
}
