#include "../include/transformer_block.cuh"
#include "cuda_utils.cuh"
#include <iostream>
#include <cstdlib>

int main() {
    // Set random seed for reproducibility
    srand(0);

    // Define dimensions
    int B = 8;  // Batch size
    int T = 4;  // Sequence length
    int C = 8;  // Embedding dimension
    int head_dim = 4;  // Dimension of each attention head
    int num_heads = 2; // Number of attention heads
    int block_size = 256; // CUDA block size

    // Allocate and initialize host memory
    float *input = make_random_float(B * T * C); // Use utility function
    float *output = (float*)malloc(B * T * C * sizeof(float));

    // Allocate and initialize device memory
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize weights (for simplicity, use random values)
    float *attn_weight_q = make_random_float(C * num_heads * head_dim);
    float *attn_weight_k = make_random_float(C * num_heads * head_dim);
    float *attn_weight_v = make_random_float(C * num_heads * head_dim);
    float *attn_weight_o = make_random_float(num_heads * head_dim * C);

    float *ffn_weight1 = make_random_float(C * 4 * C); // FFN hidden layer size = 4 * C
    float *ffn_weight2 = make_random_float(4 * C * C);
    float *ffn_bias1 = make_random_float(4 * C);
    float *ffn_bias2 = make_random_float(C);

    float rms_norm_weight = 1.0;
    float rms_norm_bias = 0.0;

    // Call the Transformer block
    transformer_block_forward(
        d_output, d_input, 
        attn_weight_q, attn_weight_k, attn_weight_v, attn_weight_o, 
        ffn_weight1, ffn_weight2, ffn_bias1, ffn_bias2, 
        rms_norm_weight, rms_norm_bias, 
        B, T, C, head_dim, num_heads, block_size
    );

    // Copy output back to host
    cudaCheck(cudaMemcpy(output, d_output, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    free(input);
    free(output);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));

    std::cout << "Transformer block executed successfully!" << std::endl;
    return 0;
}