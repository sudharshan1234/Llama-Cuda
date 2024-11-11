#include<stdio.h>
#include<cuda_runtime.h>

__global__ multihead_attention_backward_kernel1(float *d_Q, float *d_K, float* Q, float* K, float* d_output, int B, int T, int C, int head_dim, int num_heads){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >  B * num_heads * T * T){
        return;
    }
    int b = idx / (num_heads * T * T);
    int h = (idx / (T * T)) % num_heads;
    int t1 = (idx / T) % T;
    int t2 = idx % T;

    if(t2 > t1){
        output[idx] = -INFINITY;
        return;
    }

    // h * head_dim = c
    Q += b * T * C + t*C + h * head_dim;
    K += b * T * C + t2*C + h * head_dim;
    d_Q += b * T * C + t*C + h * head_dim;
    d_K += b * T * C + t2*C + h * head_dim;
}