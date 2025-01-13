#include "../include/transformer_block.cuh"
#include "cuda_utils.cuh"
#include <iostream>
#include <cstdlib>

// Wrapper function to execute all transformer blocks
void execute_transformer_blocks(
    float* d_input, float* d_output, float* d_output_temp,
    float** d_attn_weight_q, float** d_attn_weight_k, float** d_attn_weight_v, float** d_attn_weight_o,
    float** d_ffn_weight1, float** d_ffn_weight2, float** d_ffn_bias1, float** d_ffn_bias2,
    float rms_norm_weight, float rms_norm_bias,
    int B, int T, int C, int head_dim, int num_heads, int block_size, int num_blocks
) {
    for (int block_id = 0; block_id < num_blocks; ++block_id) {
        transformer_block_forward(
            (block_id == 0) ? d_input : d_output_temp, d_output,
            d_attn_weight_q[block_id], d_attn_weight_k[block_id], d_attn_weight_v[block_id], d_attn_weight_o[block_id],
            d_ffn_weight1[block_id], d_ffn_weight2[block_id], d_ffn_bias1[block_id], d_ffn_bias2[block_id],
            rms_norm_weight, rms_norm_bias, B, T, C, head_dim, num_heads, block_size
        );

        if (block_id < num_blocks - 1) {
            // Swap buffers for the next block
            float* temp = d_output_temp;
            d_output_temp = d_output;
            d_output = temp;
        }
    }
}

int main() {
    // Set random seed for reproducibility
    srand(0);

    // Define dimensions
    int B = 8;               // Smaller batch size
    int T = 1024;            // Shorter context length
    int C = 2048;            // Smaller hidden size
    int head_dim = 128;      // Dimension of each attention head
    int num_heads = 16;      // Number of attention heads (C / head_dim = 2048 / 128 = 16)
    int block_size = 256;    // CUDA block size
    int num_blocks = 12;     // Fewer transformer blocks

    // Allocate and initialize host memory
    float* input = make_random_float(B * T * C); // Use utility function
    float* output = (float*)malloc(B * T * C * sizeof(float));

    // Allocate and initialize device memory
    float *d_input, *d_output, *d_output_temp;
    cudaCheck(cudaMalloc(&d_input, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output_temp, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate weights for each block
    float **d_attn_weight_q = (float**)malloc(num_blocks * sizeof(float*));
    float **d_attn_weight_k = (float**)malloc(num_blocks * sizeof(float*));
    float **d_attn_weight_v = (float**)malloc(num_blocks * sizeof(float*));
    float **d_attn_weight_o = (float**)malloc(num_blocks * sizeof(float*));
    float **d_ffn_weight1 = (float**)malloc(num_blocks * sizeof(float*));
    float **d_ffn_weight2 = (float**)malloc(num_blocks * sizeof(float*));
    float **d_ffn_bias1 = (float**)malloc(num_blocks * sizeof(float*));
    float **d_ffn_bias2 = (float**)malloc(num_blocks * sizeof(float*));

    for (int block_id = 0; block_id < num_blocks; ++block_id) {
        cudaCheck(cudaMalloc(&d_attn_weight_q[block_id], C * num_heads * head_dim * sizeof(float)));
        cudaCheck(cudaMalloc(&d_attn_weight_k[block_id], C * num_heads * head_dim * sizeof(float)));
        cudaCheck(cudaMalloc(&d_attn_weight_v[block_id], C * num_heads * head_dim * sizeof(float)));
        cudaCheck(cudaMalloc(&d_attn_weight_o[block_id], num_heads * head_dim * C * sizeof(float)));

        cudaCheck(cudaMalloc(&d_ffn_weight1[block_id], C * 4 * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_ffn_weight2[block_id], 4 * C * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_ffn_bias1[block_id], 4 * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_ffn_bias2[block_id], C * sizeof(float)));

        // Initialize weights (e.g., copy random values to device)
        cudaCheck(cudaMemcpy(d_attn_weight_q[block_id], make_random_float(C * num_heads * head_dim), C * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_attn_weight_k[block_id], make_random_float(C * num_heads * head_dim), C * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_attn_weight_v[block_id], make_random_float(C * num_heads * head_dim), C * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_attn_weight_o[block_id], make_random_float(num_heads * head_dim * C), num_heads * head_dim * C * sizeof(float), cudaMemcpyHostToDevice));

        cudaCheck(cudaMemcpy(d_ffn_weight1[block_id], make_random_float(C * 4 * C), C * 4 * C * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_ffn_weight2[block_id], make_random_float(4 * C * C), 4 * C * C * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_ffn_bias1[block_id], make_random_float(4 * C), 4 * C * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_ffn_bias2[block_id], make_random_float(C), C * sizeof(float), cudaMemcpyHostToDevice));
    }

    float rms_norm_weight = 1.0;
    float rms_norm_bias = 0.0;

    // Benchmark the execution of all transformer blocks
    int repeat_times = 200;
    float avg_elapsed_time = benchmark_kernel(
        repeat_times,
        execute_transformer_blocks,
        d_input, d_output, d_output_temp,
        d_attn_weight_q, d_attn_weight_k, d_attn_weight_v, d_attn_weight_o,
        d_ffn_weight1, d_ffn_weight2, d_ffn_bias1, d_ffn_bias2,
        rms_norm_weight, rms_norm_bias,
        B, T, C, head_dim, num_heads, block_size, num_blocks
    );

    // Napkin math: estimate the memory bandwidth achieved
    long memory_ops = (2 * B * T * C + 2 * C * num_heads * head_dim) * num_blocks * sizeof(float);
    float memory_bandwidth = memory_ops / (avg_elapsed_time / 1000.0f) / 1e9; // Convert to GB/s

    printf("Number of blocks: %d | Average time per iteration: %.4f ms | Memory bandwidth: %.2f GB/s\n",
           num_blocks, avg_elapsed_time, memory_bandwidth);

    // Free memory
    free(input);
    free(output);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    cudaCheck(cudaFree(d_output_temp));

    for (int block_id = 0; block_id < num_blocks; ++block_id) {
        cudaCheck(cudaFree(d_attn_weight_q[block_id]));
        cudaCheck(cudaFree(d_attn_weight_k[block_id]));
        cudaCheck(cudaFree(d_attn_weight_v[block_id]));
        cudaCheck(cudaFree(d_attn_weight_o[block_id]));
        cudaCheck(cudaFree(d_ffn_weight1[block_id]));
        cudaCheck(cudaFree(d_ffn_weight2[block_id]));
        cudaCheck(cudaFree(d_ffn_bias1[block_id]));
        cudaCheck(cudaFree(d_ffn_bias2[block_id]));
    }

    free(d_attn_weight_q);
    free(d_attn_weight_k);
    free(d_attn_weight_v);
    free(d_attn_weight_o);
    free(d_ffn_weight1);
    free(d_ffn_weight2);
    free(d_ffn_bias1);
    free(d_ffn_bias2);

    std::cout << "Transformer blocks executed successfully!" << std::endl;

    return 0;
}