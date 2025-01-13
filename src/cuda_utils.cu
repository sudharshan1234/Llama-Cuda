
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include<iostream>
#include "cuda_utils.cuh"

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}


void cuda_check(cudaError_t error, const char *file, int line, const char *func) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d in function %s:\n%s\n", file, line, func, cudaGetErrorString(error));
        printf("[CUDA ERROR CODE] %d\n", error);
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__, __func__))

#define cublasCheck(status) do { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error at %s:%d - %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Function to create a cuBLAS handle
cublasHandle_t createCublasHandle() {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return handle;
}

// Function to destroy a cuBLAS handle
void destroyCublasHandle(cublasHandle_t handle) {
    cublasStatus_t status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to destroy cuBLAS handle!" << std::endl;
        exit(EXIT_FAILURE);
    }
}
