#ifndef ROPE_CUH
#define ROPE_CUH

#include <cuda_runtime.h>

// Kernels
__global__ void rope_forward_gpu_kernel1(float* output, float* input, int B, int T, int C, float* freq_inv);
__global__ void rope_forward_gpu_kernel2(float* output, float* input, int B, int T, int C, float* freq_inv);

// Kernel Caller Functions
void rope_forward1(float* output, float* input, int B, int T, int C, float* freq_inv, int block_size);
void rope_forward2(float* output, float* input, int B, int T, int C, float* freq_inv, int block_size);
void rope_forward(int kernel_num, float* output, float* input, int B, int T, int C, float* freq_inv, int block_size);

#endif