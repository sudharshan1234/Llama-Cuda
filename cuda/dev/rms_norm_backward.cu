#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "cuda_utils.cuh"

// Function to compute the backward pass of RMS normalization on the CPU
void rms_norm_backward_cpu(float *d_input,  // Gradient w.r.t input (output we want to compute)
                           float *d_out,    // Gradient w.r.t output (input from the next layer)
                           float *X,        // Input tensor X
                           int B,           // Batch size
                           int T,           // Sequence length (time steps)
                           int C,           // Number of channels (feature dimension)
                           float eps,       // Epsilon for numerical stability
                           float scale)     // Scale factor used during forward pass
{
    // Iterate over each batch and each time step
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // Compute the starting index for the (b, t) slice
            int index = (b * T + t) * C;

            // Step 1: Calculate sum of squares of input X for the current slice (b, t)
            float sum_sq = 0.0f;
            for (int i = 0; i < C; ++i) {
                float tempX = X[index + i];
                sum_sq += tempX * tempX;
            }

            // Step 2: Calculate RMS: sqrt(mean of squared values) + epsilon
            float rms = sqrtf(sum_sq / C + eps);

            // Step 3: Compute sum of (X_j * d_out_j) for the channel dimension
            float sum_dout_x = 0.0f;
            for (int i = 0; i < C; ++i) {
                sum_dout_x += X[index + i] * d_out[index + i];
            }

            // Step 4: Compute the gradient w.r.t. each element in the input (b, t) slice
            for (int i = 0; i < C; ++i) {
                float x_val = X[index + i];
                float d_out_val = d_out[index + i];

                // Apply the final gradient formula
                d_input[index + i] = scale * (d_out_val / rms - (x_val * sum_dout_x) / (C * rms * rms * rms));
            }
        }
    }
}

__global__ void rms_kernel(float *rms, float *X, int N, int C, float eps, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    X += idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float tmp_X = X[i];
        sum += tmp_X * tmp_X;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rms[idx] = sqrt(shared[0]/C + eps);
        // printf("Block %d: sum = %f, mean_square = %f, rms = %f\n", idx, shared[0], shared[0]/C, rms[idx]);
    }
}

__global__ void rms_norm_backward_kernel1(float *d_input, float *d_output, float *X, float *rms, int C, float scale, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    X += idx * C;
    d_output += idx * C;
    d_input += idx * C;
    rms += idx;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum +=  X[i] * d_output[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    for (int i = tid; i < C; i += block_size) {
        float x_val = X[i];
        float dout_val = d_output[i];
        d_input[i] = scale * (dout_val / (*rms) - (x_val * shared[0]) / (C * (*rms) * (*rms) * (*rms)));
    }

}

// Host code to launch the kernel
void rms_norm_backward1(float *d_input, float *d_output, float *X, int B, int T, int C, float eps, float scale, int block_size) {
    int N = B * T;
    float *d_rms;
    cudaMalloc(&d_rms, B * T * sizeof(float));
    rms_kernel<<<N, block_size, block_size * sizeof(float)>>>(d_rms, X, B*T, C, eps, block_size);
    cudaCheck(cudaGetLastError());
    const int grid_size2 = ceil_div(B * T * C, 256);
    rms_norm_backward_kernel1<<<N, block_size, block_size * sizeof(float)>>>(d_input, d_output, X, d_rms, C, scale, block_size);
    cudaCheck(cudaGetLastError());
}

void rms_norm_backward(int kernel_num, float *d_input, float *d_output, float *X, int B, int T, int C, float eps, float scale, int block_size) {
    switch (kernel_num) {
        case 1:
            rms_norm_backward1(d_input, d_output, X, B, T, C, eps, scale, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    float eps = 1e-6;
    float scale = 1.0f;

    int deviceIdx = 0;
    int kernel_num = 1;
    cudaCheck(cudaSetDevice(deviceIdx));
    // create host memory of random numbers
    float *grad_input = (float*)malloc(B * T * C * sizeof(float));
    float *grad_output = make_random_float(B * T * C);
    float *X = make_random_float(B * T * C);
    // move to GPU
    float *d_input, *d_out, *d_X;

    cudaCheck(cudaMalloc((void**)&d_input, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_X, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_X, X, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_out, grad_output, B * T * C * sizeof(float), cudaMemcpyHostToDevice));


    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rms_norm_backward_cpu(grad_input, grad_output, X, B, T, C, eps, scale);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
// float *d_input, float *d_output, float *X, int B, int T, int C, float eps, float scale, int block_size
        rms_norm_backward(kernel_num, d_input, d_out, d_X, B, T, C,  eps, scale, block_size);

        validate_result(d_input, grad_input, "d_input", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 200;
        float elapsed_time = benchmark_kernel(repeat_times, rms_norm_backward,
                                              kernel_num, d_input, d_out, d_X, B, T, C,  eps, scale, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(grad_input);
    free(grad_output);
    free(X);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_X));

    return 0;
}