#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include "cuda_utils.cuh"

void softmax_cpu(float* x, int length) {
    float max_val = x[0];
    for (int i = 1; i < length; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < length; i++) {
        x[i] /= sum;
    }
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_query_key_kernel(float *input, float *output, int B, int T, int C, int head_dim, int num_heads, int block_size){
    extern __shared__ float shared[]; // 2 * block_size/32
    int tid = threadIdx.x; //0 - block_size
    int idx = blockIdx.x; // b*NH*T + n*T + t1
    int warpId = tid / 32;
    int laneId = tid % 32;
    input += idx * T;
    output += idx * T;
    float *max_val = &shared[0];
    float *sum_val = &shared[block_size/32];
    int max_num = 0;
    for(int i=tid; i<T; i += block_size){
        max_num = fmaxf(max_num, input[i]);
    }

    max_num = warpReduceMax(max_num);

    if (laneId == 0){
        max_val[warpId] = max_num;
    }
    __syncthreads();

    // Now each warp has stored its maximum in `max_val[warpId]`
    // Perform another reduction across the warps
    if (tid == 0) {  // Only the first thread in the first warp participates
        max_num = max_val[0];
        for (int i = 1; i < block_size / 32; i++) {  // Loop over all warp results
            max_num = fmaxf(max_num, max_val[i]);
        }
        max_val[0] = max_num;  // Store the final block-wide maximum in max_val[0]
    }
    __syncthreads();

    // Now max_val[0] contains the maximum value across the block
    max_num = max_val[0];  // Broadcast the result to all threads in the block
    
    
    float sum = 0.0f;
    for(int i=tid; i<T; i += block_size){
        int val = exp(input[i] - max_num);
        output[i] = val;
        sum += val;
    }
    sum = warpReduceSum(sum);

    if (laneId == 0){
        sum_val[warpId] = sum;
    }
    __syncthreads();

    // Now each warp has stored its sum in `sum_val[warpId]`
    // Perform another reduction across the warps
    if (tid == 0) {  // Only the first thread in the first warp participates
        sum = sum_val[0];
        for (int i = 1; i < block_size / 32; i++) {  // Loop over all warp results
            sum += sum_val[i];
        }
        sum_val[0] = sum;  // Store the final block-wide maximum in max_val[0]
    }
    __syncthreads();

    sum = sum_val[0];

    for (int i = tid; i < T; i += block_size) {
        output[i] = output[i] / sum;
    }

}