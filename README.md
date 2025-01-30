# Llama Implementation with Custom CUDA Kernels

This project implements a Llama architecture using custom CUDA kernels for various operations, including RMS Norm, Softmax, Multihead Attention Forward, RoPE (Rotary Positional Embedding), and FFN (Feed-Forward Network). The transformer block is repeated `block_num` times in the main program to simulate a multi-layer transformer model.

## Key Components

### 1. Custom CUDA Kernels

- **RMS Norm**: Implements Root Mean Square Normalization.
- **Softmax**: Implements the Softmax function.
- **Multihead Attention Forward**: Implements the forward pass of multi-head attention.
- **RoPE**: Implements Rotary Positional Embedding.
- **FFN**: Implements the Feed-Forward Network.

### 2. Transformer Block

The transformer block integrates the above kernels to perform the following operations:

1. Multihead Attention with RoPE.
2. RMS Norm.
3. Feed-Forward Network.
4. Residual connections.

### 3. Main Program

The main program (`main.cu`) repeats the transformer block `block_num` times to simulate a multi-layer transformer model. It also includes benchmarking to measure the performance of the kernels.

## Performance Benchmarks

### Machine Specifications

- **GPU Specifications:**

  - **Model:** NVIDIA GeForce RTX 3080 Ti
  - **Architecture:** Ampere
  - **CUDA Cores:** 10240
  - **Memory:** GDDR6, 12GB
  - **Memory Bandwidth:** 912.4 GB/s
  - **Core Clock Speed:** 1.37 GHz (Base), 1.67 GHz (Boost)
  - **Compute Capability:** 8.6

- **CUDA Version:**

  - **CUDA Toolkit:** 12.6

- **Driver Version:**

  - **NVIDIA Driver:** 560.35.03

- **Environment:**
  - Benchmarks were run in a clean environment with no other resource-intensive processes.

---

The project includes performance benchmarks for different block sizes. Below are some results on the kernels:

### RMS Norm

| Block Size | Time (ms) | Bandwidth (GB/s) |
| ---------- | --------- | ---------------- |
| 32         | 0.0471    | 400.87           |
| 64         | 0.0467    | 404.13           |
| 128        | 0.0467    | 404.07           |
| 256        | 0.0499    | 378.18           |
| 512        | 0.0630    | 299.57           |
| 1024       | 0.1236    | 152.71           |

### Multihead attention

| Block Size | Time (ms) | Bandwidth (GB/s) |
| ---------- | --------- | ---------------- |
| 32         | 145.2367  | N/A              |
| 64         | 145.7929  | N/A              |
| 128        | 145.3862  | N/A              |

### ROPE

| Block Size | Time (ms) | Bandwidth (GB/s) |
| ---------- | --------- | ---------------- |
| 32         | 0.0361    | 522.94           |
| 64         | 0.0316    | 597.73           |
| 128        | 0.0314    | 600.34           |
| 256        | 0.0316    | 597.99           |
| 512        | 0.0318    | 594.32           |
| 1024       | 0.0335    | 562.63           |

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.

### Building the Project

To build the project, use the following command:

```bash
nvcc -I ./include -lcublas -o ./build/transformer_block ./src/*.cu main.cu
```

### This command:

- Includes the ./include directory for header files.

- Links the cublas library.

- Compiles all .cu files in the ./src directory and main.cu.

- Outputs the executable to ./build/transformer_block.

## Running the Program

./build/transformer_block

### To test out the kernels separately

`nvcc -I ./include -lcublas -o ./build/attn_f ./cuda/dev/program.cu ./src/cuda_utils.cu`

And run:

`./build/program`
