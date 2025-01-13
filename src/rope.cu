#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include "cuda_utils.h"
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

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 4;
    int C = 8;

    int deviceIdx = 0;
    int kernel_num = 2;
    cudaCheck(cudaSetDevice(deviceIdx));
    // create host memory of random numbers
    float *out = (float*)malloc(B * T * C * sizeof(float));
    float *out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float *X = make_random_float(B * T * C);
    float* freq_inv = (float*)malloc((C / 2) * sizeof(float));
    for (int i = 0; i < C / 2; i++) {
        freq_inv[i] = 1.0f / powf(10000.0f, 2.0f * i / C);
    }
    // move to GPU
    float *d_out, *d_X, *d_freq_inv;

    cudaCheck(cudaMalloc((void**)&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_X, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_freq_inv, (C / 2) * sizeof(float)));
    cudaCheck(cudaMemcpy(d_X, X, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freq_inv, freq_inv, (C / 2) * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rope_forward_cpu(out, X, B, T, C, freq_inv);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        rope_forward(kernel_num, d_out, d_X, B, T, C, d_freq_inv, block_size);
        cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 200;
        float elapsed_time = benchmark_kernel(repeat_times, rope_forward,
                                              kernel_num, d_out, d_X, B, T, C, d_freq_inv, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(out_gpu);
    free(X);
    free(freq_inv);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_X));

    return 0;
}