# Llama Implementation with Custom CUDA Kernels

This project implements a Llama using custom CUDA kernels for various operations, including RMS Norm, Softmax, Multihead Attention Forward, RoPE (Rotary Positional Embedding), and FFN (Feed-Forward Network). The transformer block is repeated `block_num` times in the main program to simulate a multi-layer transformer model.

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

  - **Model:** NVIDIA GeForce RTX 3070
  - **Architecture:** Ampere
  - **CUDA Cores:** 5888
  - **Memory:** GDDR6, 8GB
  - **Memory Bandwidth:** 448 GB/s
  - **Core Clock Speed:** 1.50 GHz (Base), 1.73 GHz (Boost)
  - **Compute Capability:** 8.6

- **CPU Specifications:**

  - **Model:** Intel Core i7-10700K (or equivalent, depending on your system)
  - **Architecture:** 10th Gen
  - **Cores/Threads:** 8 cores / 16 threads
  - **Clock Speed:** 3.8 GHz (Base), 5.1 GHz (Turbo)

- **Operating System:**

  - **Name:** Ubuntu 20.04 LTS (or equivalent, depending on your system)

- **CUDA Version:**

  - **CUDA Toolkit:** 11.2 (or later)

- **Driver Version:**

  - **NVIDIA Driver:** 460.32.03 (or later)

- **Environment:**
  - Benchmarks were run in a clean environment with no other resource-intensive processes.

---

The project includes performance benchmarks for different block sizes. Below are some results on the kernels:

### RMS Norm

| Block Size | Time (ms) | Bandwidth (GB/s) |
| ---------- | --------- | ---------------- |
| 32         | 0.2287    | 220.09           |
| 64         | 0.2266    | 222.14           |
| 128        | 0.2206    | 228.20           |
| 256        | 0.2816    | 178.71           |
| 512        | 0.4014    | 125.38           |
| 1024       | 0.7179    | 70.11            |

### Multihead attention

| Block Size | Time (ms) | Bandwidth (GB/s) |
| ---------- | --------- | ---------------- |
| 32         | 0.3215    | N/A              |
| 64         | 0.3381    | N/A              |
| 128        | 0.3256    | N/A              |
| 256        | 0.3165    | N/A              |
| 512        | 0.3514    | N/A              |
| 1024       | 0.3460    | N/A              |

### ROPE

| Block Size | Time (ms) | Bandwidth (GB/s) |
| ---------- | --------- | ---------------- |
| 32         | 0.0076    | 477.90           |
| 64         | 0.0071    | 508.79           |
| 128        | 0.0070    | 506.08           |
| 256        | 0.0071    | 506.41           |
| 512        | 0.0069    | 506.33           |
| 1024       | 0.0071    | 502.28           |

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.

### Building the Project

To build the project, use the following command:

```bash
nvcc -I ./include -lcublas -o ./build/transformer_block ./src/*.cu main.cu
```

# This command:

# - Includes the ./include directory for header files.

# - Links the cublas library.

# - Compiles all .cu files in the ./src directory and main.cu.

# - Outputs the executable to ./build/transformer_block.

# Running the Program

# After building the project, you can run the main program:

./build/transformer_block

### To test out the kernels separately

`nvcc -I ./include -lcublas -o ./build/attn_f ./cuda/dev/program.cu ./src/cuda_utils.cu`

### After building the project, you can run the main program:

./build/program
