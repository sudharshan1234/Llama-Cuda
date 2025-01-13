#ifndef RMS_NORM_CUH
#define RMS_NORM_CUH

#include <cuda_runtime.h>

// Kernels
__global__ void rms_norm_forward_kernel1(float* out, float* X, int B, int T, int C, float eps, float scale, float shift);
__global__ void rms_norm_forward_kernel2(float* out, float* X, float* sum, int B, int T, int C, float eps, float scale, float shift);
__global__ void rms_norm_forward_kernel3(float* out, float* X, float* rms, int B, int T, int C, float eps, float scale, float shift);

// Kernel Caller Functions
void rms_norm_forward1(float* out, float* X, int B, int T, int C, float eps, float scale, float shift, int block_size);
void rms_norm_forward2(float* out, float* X, int B, int T, int C, float eps, float scale, float shift, int block_size);
void rms_norm_forward3(float* out, float* X, int B, int T, int C, float eps, float scale, float shift, int block_size);
void rms_norm_forward(int kernel_num, float* out, float* X, int B, int T, int C, float eps, float scale, float shift, int block_size);

#endif