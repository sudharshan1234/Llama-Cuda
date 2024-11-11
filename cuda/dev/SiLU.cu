#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "cuda_utils.h"


__global__ void silu_forward_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val)); // SiLU(x) = x * sigmoid(x)
    }
}

__global__ void silu_backward_kernel(const float* x, const float* grad_output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        grad_input[idx] = grad_output[idx] * (sigmoid_val * (1 + val * (1 - sigmoid_val)));
    }
}

void silu_forward(const float* x, float* y, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_forward_kernel<<<blocks, threads>>>(x, y, n);
    cudaDeviceSynchronize(); // Ensure computation is complete
}

void silu_backward(const float* x, const float* grad_output, float* grad_input, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_backward_kernel<<<blocks, threads>>>(x, grad_output, grad_input, n);
    cudaDeviceSynchronize(); // Ensure computation is complete
}