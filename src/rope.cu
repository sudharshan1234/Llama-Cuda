#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include "cuda_utils.cuh"
#include "../include/rope.cuh"

__global__ void rope_forward_gpu_kernel1(float* output, float* input, int B, int T, int C, float* freq_inv){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = B * T * C;
    if (idx >= total_size) return;

    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;

    int pair_index = c / 2;
    bool is_odd = (c % 2 != 0);
    float angle = t * freq_inv[pair_index];

    if(is_odd){
        output[idx] = input[b * T * C + t * C + c - 1] * sinf(angle) + input[idx] * cosf(angle);
    } else {
        output[idx] = input[idx] * cosf(angle) - input[b * T * C + t * C + c + 1] * sinf(angle);
    }
}

__global__ void rope_forward_gpu_kernel2(float* output, float* input, int B, int T, int C, float* freq_inv){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = B * T * (C/2);
    if (idx >= total_size) return;

    int b = idx / (T * (C/2));
    int t = (idx / (C/2)) % T;
    int c = idx % (C/2);

    int even_idx = c * 2;
    int odd_idx = even_idx + 1;

    float angle = t * freq_inv[c];

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float* input_ptr = input + b * T * C + t * C;
    float* output_ptr = output + b * T * C + t * C;

    float even_val = input_ptr[even_idx];
    float odd_val = input_ptr[odd_idx];

    output_ptr[even_idx] = even_val * cos_val - odd_val * sin_val;
    output_ptr[odd_idx] = even_val * sin_val + odd_val * cos_val;
}

void rope_forward1( float* output, float* input, int B, int T, int C, float* freq_inv, int block_size) {
    if (C % 2 != 0) {
        fprintf(stderr, "Embedding dimension C must be even for RoPE.\n");
        return;
    }

    int total_size = B * T * C;
    int num_blocks = (total_size + block_size - 1) / block_size; // Number of blocks

    rope_forward_gpu_kernel1<<<num_blocks, block_size>>>(output, input, B, T, C, freq_inv);

    cudaDeviceSynchronize();
}

void rope_forward2(float* output, float* input, int B, int T, int C, float* freq_inv, int block_size) {
    if (C % 2 != 0) {
        fprintf(stderr, "Embedding dimension C must be even for RoPE.\n");
        return;
    }

    int total_pairs = B * T * (C / 2);
    int num_blocks = (total_pairs + block_size - 1) / block_size;

    rope_forward_gpu_kernel2<<<num_blocks, block_size>>>(output, input, B, T, C, freq_inv);

    cudaDeviceSynchronize();
}

void rope_forward(int kernel_num, float* output, float* input, int B, int T, int C, float* freq_inv, int block_size) {
    switch (kernel_num) {
        case 1:
            rope_forward1(output, input, B, T, C, freq_inv, block_size);
            break;
        case 2:
            rope_forward2(output, input, B, T, C, freq_inv, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}