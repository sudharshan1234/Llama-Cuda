#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include "cuda_utils.cuh"

// Function for RMS normalization on CPU
void rms_norm_forward_cpu(float *out, float *X, int B, int T, int C, float eps, float scale, float shift) {
    // Iterate over each batch (B) and each time step (T)
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // Compute the starting index for the (b, t) slice
            int index = (b * T + t) * C;

            // Calculate sum of squares for the C elements in the (b, t) slice
            float sum = 0.0f;
            for (int i = 0; i < C; ++i) {
                float tempX = X[index + i];
                sum += tempX * tempX;
            }

            // Calculate RMS: sqrt(mean of squared values) + epsilon
            float rms = std::sqrt(sum / C + eps);

            // Normalize and scale each element in the (b, t) slice
            for (int i = 0; i < C; ++i) {
                out[index + i] = (X[index + i] / rms) * scale + shift;
            }
        }
    }
}

// Kernel 1: Naive Implementation parallelizes over B,T, loops over C
__global__ void rms_norm_forward_kernel1(float *out, float *X, int B, int T, int C, float eps, float scale, float shift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= B * T) {
        return;
    }
    
    float sum = 0.0f;
    X += idx * C;
    out += idx * C;

    for(int i=0; i < C; i++){
        float tempX = X[i];
        sum += tempX * tempX;
    }

    float rms = sqrt(sum/C + eps);
    for(int i=0; i<C; i++){
        out[i] = (X[i]/rms) * scale + shift;
    }
}

// Kernel 2: Implementation parallelizes over B,T and C
template <const int BLOCK_SIZE>
__global__ void rms_norm_forward_kernel2(float *out, float *X, float *sum, int B, int T, int C, float eps, float scale, float shift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= B * T * C){
        return;
    }
    
    X += idx;
    out += idx;
    float tmp_X = *X;
    float finalSum = atomicAdd(sum, tmp_X * tmp_X);
    __syncthreads();

    __shared__ float rms;
    if(threadIdx.x == 0){
        rms = sqrt(finalSum/C + eps);
    }

    __syncthreads();
    *out = (tmp_X/rms) * scale + shift;
}
__global__ void rms_kernel(float* rms, float* X, int N, int C, float eps, int block_size) {
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

__global__ void rms_norm_forward_kernel3(float *out, float *X, float *rms, int B, int T, int C, float eps, float scale, float shift){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = idx / C;
    if(idx >= B * T * C){
        return;
    }
    // if(rms[n]==0.0){
    //     printf("RMS for %d is %f\n", idx, rms[n]);
    // }
    // printf("idx: %d, n: %d, rms[n]: %f\n", idx, n, rms[n]);
    X += idx;
    out += idx;
    rms += n;
    
    *out = (*X / *rms) * scale + shift;

}

// Kernel Callers

void rms_norm_forward1(float *out, float *X, int B, int T, int C, float eps, float scale, float shift, int block_size){
    const int grid_size = ceil_div(B * T, block_size);
    rms_norm_forward_kernel1<<<grid_size, block_size>>>(out, X, B, T, C, eps, scale, shift);
    cudaCheck(cudaGetLastError());
}
void rms_norm_forward2(float *out, float *X, int B, int T, int C, float eps, float scale, float shift, int block_size){
    const int grid_size = ceil_div(B * T * C, 256);
    float *d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    rms_norm_forward_kernel2<256><<<grid_size, 256>>>(out, X, d_sum, B, T, C, eps, scale, shift);
    cudaCheck(cudaGetLastError());
}

// Host code to launch the kernel
void rms_norm_forward3(float* out, float* X, int B, int T, int C, float eps, float scale, float shift, int block_size) {
    int N = B * T;
    float *d_rms;
    cudaMalloc(&d_rms, B * T * sizeof(float));
    rms_kernel<<<N, block_size, block_size * sizeof(float)>>>(d_rms, X, B*T, C, eps, block_size);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError());
    const int grid_size2 = ceil_div(B * T * C, 256);
    rms_norm_forward_kernel3<<<grid_size2, 256>>>(out, X, d_rms, B, T, C, eps, scale, shift);
    cudaCheck(cudaGetLastError());
}


void rms_norm_forward(int kernel_num, float *out, float *X, int B, int T, int C, float eps, float scale, float shift, int block_size) {
    switch (kernel_num) {
        case 1:
            rms_norm_forward1(out, X, B, T, C, eps, scale, shift, block_size);
            break;
        case 2:
            rms_norm_forward2(out, X, B, T, C, eps, scale, shift, block_size);
            break;
        case 3:
            rms_norm_forward3(out, X, B, T, C, eps, scale, shift, block_size);
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
    float shift = 0.0f;

    int deviceIdx = 0;
    int kernel_num = 3;
    cudaCheck(cudaSetDevice(deviceIdx));
    // create host memory of random numbers
    float *out = (float*)malloc(B * T * C * sizeof(float));
    float *out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float *X = make_random_float(B * T * C);
    // move to GPU
    float *d_out, *d_X;

    cudaCheck(cudaMalloc((void**)&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_X, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_X, X, B * T * C * sizeof(float), cudaMemcpyHostToDevice));


    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rms_norm_forward_cpu(out, X, B, T, C, eps, scale, shift);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        rms_norm_forward(kernel_num, d_out, d_X, B, T, C,  eps, scale, shift, block_size);
        cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 200;
        float elapsed_time = benchmark_kernel(repeat_times, rms_norm_forward,
                                              kernel_num, d_out, d_X, B, T, C,  eps, scale, shift, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(X);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_X));

    return 0;
}