# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a CUDA learning tutorial repository containing progressive examples for learning GPU programming with NVIDIA CUDA. The examples are designed for Chinese language learners and progress from basic concepts to practical applications.

## Build System

All CUDA programs use `nvcc` (NVIDIA CUDA Compiler) directly:

```bash
# Compile a CUDA program
nvcc <filename>.cu -o <output_name>

# Compile and run in one step
nvcc <filename>.cu -o <output_name> && ./<output_name>
```

No build system (Make, CMake) is used. Each `.cu` file is standalone and self-contained.

## Code Architecture

The tutorial follows a progressive learning structure:

### 基础篇 (01-04)

1. **hello_cuda.cu** - Introduction to CUDA kernels and thread organization
   - Demonstrates `__global__` kernel functions
   - Shows basic thread indexing (threadIdx, blockIdx, blockDim)
   - Introduces execution configuration syntax `<<<blocks, threads>>>`

2. **kernel_basics.cu** - Core CUDA function types and multi-dimensional grids
   - `__global__` vs `__device__` function qualifiers
   - 1D and 2D thread organization using `dim3`
   - Passing parameters to kernels

3. **memory_management.cu** - CPU-GPU memory model
   - Host (CPU) and device (GPU) memory separation
   - Memory allocation: `cudaMalloc()`, `free()`, `cudaFree()`
   - Data transfer: `cudaMemcpy()` with HostToDevice/DeviceToHost
   - Error checking pattern with `CHECK_CUDA()` macro

4. **vector_add.cu** - Complete parallel computing workflow
   - Full CUDA programming pipeline demonstration
   - CPU vs GPU performance comparison
   - Result verification and error handling
   - Event-based timing with `cudaEvent_t`

### 进阶篇 (05-07)

5. **05_shared_memory.cu** - Shared memory optimization
   - `__shared__` memory declaration
   - Thread synchronization with `__syncthreads()`
   - Bank conflicts and optimization strategies

6. **06_sync_atomic.cu** - Synchronization and atomic operations
   - `atomicAdd()`, `atomicMax()`, `atomicCAS()`
   - Thread block synchronization patterns
   - Race condition handling

7. **07_cuda_streams.cu** - CUDA Streams and asynchronous execution
   - `cudaStreamCreate()`, `cudaStreamDestroy()`
   - Async memory operations with `cudaMemcpyAsync()`
   - Overlapping computation and data transfer

### 内存优化篇 (08-10)

8. **08_unified_memory.cu** - Unified Memory
   - `cudaMallocManaged()` for simplified memory management
   - Memory prefetching with `cudaMemPrefetchAsync()`
   - Memory advise hints

9. **09_texture_memory.cu** - Texture Memory
   - Texture objects and descriptors
   - Hardware interpolation and filtering
   - Address modes and boundary handling

10. **10_constant_reduction.cu** - Constant memory and reduction operations
    - `__constant__` memory declaration
    - Parallel reduction algorithms
    - Warp shuffle operations (`__shfl_down_sync`)

### 实战篇 (11-15)

11. **11_matrix_multiply.cu** - Matrix multiplication optimization
    - Naive vs tiled implementations
    - Bank conflict avoidance
    - Comparison with cuBLAS
    - Compile: `nvcc -lcublas 11_matrix_multiply.cu -o 11_matrix_multiply`

12. **12_profiling_debug.cu** - Performance profiling and debugging
    - Error checking patterns
    - CUDA Events timing
    - Nsight Systems/Compute usage
    - Common performance issues

13. **13_dynamic_parallelism.cu** - Dynamic parallelism and cooperative groups
    - Launching kernels from kernels
    - `cooperative_groups` namespace
    - Warp-level primitives
    - Compile: `nvcc -rdc=true -lcudadevrt 13_dynamic_parallelism.cu -o 13_dynamic_parallelism`

14. **14_multi_gpu.cu** - Multi-GPU programming
    - Device management APIs
    - P2P (Peer-to-Peer) access
    - Work distribution strategies
    - Cross-device synchronization

15. **15_thrust_practical.cu** - Thrust library and practical examples
    - `thrust::device_vector`, `thrust::host_vector`
    - Algorithms: sort, reduce, transform, scan
    - Advanced iterators
    - Integration with native CUDA

### 库应用篇 (16-20)

16. **16_cudnn_deeplearning.cu** - cuDNN deep learning primitives
    - Convolution operations
    - Activation functions
    - Pooling layers
    - Compile: `nvcc -lcudnn 16_cudnn_deeplearning.cu -o 16_cudnn_deeplearning`

17. **17_cufft.cu** - cuFFT for Fast Fourier Transform
    - 1D/2D/3D FFT operations
    - Batch processing
    - Real-to-complex and complex-to-complex transforms

18. **18_cusparse.cu** - cuSPARSE for sparse matrix operations
    - Sparse matrix formats (CSR, COO)
    - SpMV (Sparse Matrix-Vector multiplication)
    - Sparse matrix-matrix operations

19. **19_curand.cu** - cuRAND random number generation
    - Host and device API
    - Various distributions
    - Quasi-random sequences

20. **20_cuda_graphs.cu** - CUDA Graphs for optimized execution
    - Stream capture
    - Explicit graph construction
    - Graph update and execution

### 高级篇 (21-25)

21. **21_interop_graphics.cu** - CUDA-Graphics API interoperability
    - OpenGL buffer/texture sharing
    - VBO particle systems
    - Vulkan interop basics
    - Compile: `nvcc 21_interop_graphics.cu -o 21_interop_graphics -lGL -lGLU -lglut`

22. **22_memory_pools.cu** - Memory pools and virtual memory
    - `cudaMallocAsync` / `cudaFreeAsync`
    - Memory pool configuration
    - Virtual memory management (CUDA 10.2+)
    - Stream-ordered allocation

23. **23_cooperative_groups_advanced.cu** - Advanced cooperative groups
    - Thread block tiles
    - Coalesced groups
    - Grid-level synchronization
    - Warp-level primitives (shfl, ballot, match)
    - Compile: `nvcc -rdc=true -lcudadevrt 23_cooperative_groups_advanced.cu -o 23_cooperative_groups_advanced`

24. **24_optimization_workshop.cu** - Performance optimization workshop
    - Matrix transpose optimization case study
    - Memory coalescing
    - Bank conflict avoidance
    - Occupancy optimization
    - Branch divergence
    - Instruction-level optimization

25. **25_deep_learning_integration.cu** - Deep learning framework integration
    - Custom activation functions (ReLU, GELU, SiLU)
    - LayerNorm and Softmax implementation
    - FP16 mixed precision
    - Operator fusion
    - PyTorch/TensorFlow integration patterns

### 专题篇 (26-30)

26. **26_ptx_inline_assembly.cu** - PTX inline assembly
    - PTX instruction basics
    - Inline assembly syntax
    - Performance optimization with PTX

27. **27_warp_matrix_tensor_cores.cu** - Warp matrix and Tensor Cores
    - WMMA API usage
    - FP16/INT8 matrix multiplication
    - Tensor Core optimization strategies

28. **28_async_copy_pipeline.cu** - Asynchronous copy and pipelines
    - `cp.async` instructions
    - Multi-stage software pipelining
    - Hiding memory latency

29. **29_debugging_best_practices.cu** - Debugging best practices
    - Error checking patterns
    - compute-sanitizer usage
    - Common bugs and fixes
    - Performance debugging

30. **30_image_processing_project.cu** - Image processing project
    - Color conversion (RGB/YUV/HSV)
    - Gaussian blur and edge detection
    - Histogram equalization
    - Image resizing

### 前沿应用篇 (31-35)

31. **31_neural_network_inference.cu** - Neural network inference engine
    - Activation functions (ReLU, GELU, Softmax)
    - Dense/Convolution/Pooling layers
    - BatchNorm implementation
    - Simple CNN inference pipeline
    - Optimization techniques

32. **32_realtime_video_processing.cu** - Real-time video/audio processing
    - Video frame color correction and filtering
    - Gaussian blur, sharpening, bilateral filter
    - Audio FFT and spectrum analysis
    - Double-buffering pipeline design
    - Hardware codec integration concepts
    - Compile: `nvcc -lcufft 32_realtime_video_processing.cu -o 32_realtime`

33. **33_scientific_computing.cu** - Scientific computing and numerical methods
    - Monte Carlo integration
    - Conjugate Gradient (CG) solver
    - N-body simulation
    - Heat equation (finite difference)
    - ODE solving (Lorenz system, RK4)
    - Compile: `nvcc -lcurand 33_scientific_computing.cu -o 33_scientific`

34. **34_jetson_embedded.cu** - Jetson embedded and robotics
    - Point cloud processing (LiDAR)
    - Depth image processing
    - Path planning (wavefront)
    - Sensor fusion (IMU integration)
    - Power optimization strategies

35. **35_hpc_future.cu** - HPC applications and future outlook
    - Real-world HPC application cases
    - Latest CUDA features (CUDA 12+)
    - Multi-GPU and cluster computing
    - AI acceleration and large models
    - Advanced optimization techniques
    - Future technology trends

## Development Conventions

### Error Handling

All programs after `memory_management.cu` use a standard error-checking macro:

```cuda
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}
```

Always check CUDA API calls and kernel launches for errors.

### Thread Configuration

Standard pattern for determining grid size:

```cuda
int threadsPerBlock = 256;  // Typical: 128, 256, or 512
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(args);
```

### Kernel Boundary Checking

Always include bounds checking in kernels to handle grid size != problem size:

```cuda
__global__ void kernel(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // Safe to access data[tid]
    }
}
```

## CUDA Environment

The system uses:
- CUDA 13.0
- NVIDIA Thor GPU
- Driver Version 580.00

All examples are compatible with CUDA 10.0+.

## Language and Comments

Code comments and output messages are in Chinese (中文). When modifying or adding examples, maintain this convention for consistency with the learning audience.
