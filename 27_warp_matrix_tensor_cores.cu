/**
 * =============================================================================
 * CUDA 教程 27: Warp 级矩阵操作与 Tensor Cores
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 WMMA (Warp Matrix Multiply-Accumulate) API
 * 2. 学会使用 Tensor Cores 加速矩阵运算
 * 3. 掌握混合精度计算 (FP16/FP32)
 * 4. 理解 Tensor Cores 的性能特点
 *
 * 关键概念：
 * - Tensor Cores 架构
 * - WMMA fragment 类型
 * - 矩阵布局 (row_major / col_major)
 * - 混合精度计算
 *
 * 编译命令：
 *   nvcc -arch=sm_70 27_warp_matrix_tensor_cores.cu -o 27_warp_matrix_tensor_cores
 *
 * 注意：需要 Volta (sm_70) 或更新的 GPU 架构
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：Tensor Cores 概述
// ============================================================================

void demoTensorCoresOverview() {
    printf("=== 第一部分：Tensor Cores 概述 ===\n\n");

    printf("什么是 Tensor Cores:\n");
    printf("  专用于矩阵乘法累加 (MMA) 的硬件单元\n");
    printf("  每个时钟周期可执行 4x4x4 矩阵运算\n\n");

    printf("Tensor Cores 演进:\n");
    printf("  ┌────────────┬──────────────────────────────────────┐\n");
    printf("  │ 架构       │ Tensor Core 能力                     │\n");
    printf("  ├────────────┼──────────────────────────────────────┤\n");
    printf("  │ Volta      │ FP16 → FP16/FP32                     │\n");
    printf("  │ Turing     │ + INT8, INT4, INT1                   │\n");
    printf("  │ Ampere     │ + TF32, BF16, FP64                   │\n");
    printf("  │ Hopper     │ + FP8, 更大矩阵尺寸                  │\n");
    printf("  └────────────┴──────────────────────────────────────┘\n\n");

    printf("WMMA 支持的矩阵尺寸 (M x N x K):\n");
    printf("  FP16: 16x16x16, 32x8x16, 8x32x16\n");
    printf("  INT8: 16x16x16, 32x8x16, 8x32x16\n");
    printf("  TF32 (Ampere+): 16x16x8\n\n");

    printf("编程模型:\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  D = A × B + C                                          │\n");
    printf("  │                                                         │\n");
    printf("  │  A: M × K (输入矩阵)                                    │\n");
    printf("  │  B: K × N (输入矩阵)                                    │\n");
    printf("  │  C: M × N (累加矩阵)                                    │\n");
    printf("  │  D: M × N (输出矩阵)                                    │\n");
    printf("  │                                                         │\n");
    printf("  │  所有操作由一个 Warp (32 线程) 协作完成                 │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第二部分：WMMA API 基础
// ============================================================================

// WMMA 矩阵尺寸
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// 简单的 WMMA 矩阵乘法示例
__global__ void wmmaSimpleGemm(half *a, half *b, float *c, float *d,
                                int M, int N, int K) {
    // 计算 warp 处理的 tile 位置
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 检查边界
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // 声明 fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(acc_frag, 0.0f);

    // 循环遍历 K 维度
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // 检查边界
        if (aCol + WMMA_K <= K) {
            // 加载 A 和 B 的 tile
            wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);

            // 执行矩阵乘法累加
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 加载 C (如果需要)
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::load_matrix_sync(c_frag, c + cRow * N + cCol, N, wmma::mem_row_major);

    // D = A * B + C
    for (int i = 0; i < c_frag.num_elements; i++) {
        acc_frag.x[i] += c_frag.x[i];
    }

    // 存储结果
    wmma::store_matrix_sync(d + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}

void demoWMMABasics() {
    printf("=== 第二部分：WMMA API 基础 ===\n\n");

    printf("Fragment 类型:\n");
    printf("  wmma::fragment<type, M, N, K, element_type, layout>\n\n");

    printf("Fragment 种类:\n");
    printf("  ┌──────────────────┬────────────────────────────────────┐\n");
    printf("  │ 类型             │ 说明                               │\n");
    printf("  ├──────────────────┼────────────────────────────────────┤\n");
    printf("  │ wmma::matrix_a   │ A 矩阵片段 (M × K)                 │\n");
    printf("  │ wmma::matrix_b   │ B 矩阵片段 (K × N)                 │\n");
    printf("  │ wmma::accumulator│ 累加器/输出 (M × N)                │\n");
    printf("  └──────────────────┴────────────────────────────────────┘\n\n");

    printf("布局选项:\n");
    printf("  wmma::row_major - 行优先\n");
    printf("  wmma::col_major - 列优先\n\n");

    printf("核心 API:\n");
    printf("  // 加载矩阵片段\n");
    printf("  wmma::load_matrix_sync(frag, ptr, stride);\n\n");

    printf("  // 执行矩阵乘加\n");
    printf("  wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);\n\n");

    printf("  // 存储结果\n");
    printf("  wmma::store_matrix_sync(ptr, frag, stride, layout);\n\n");

    printf("  // 填充片段\n");
    printf("  wmma::fill_fragment(frag, value);\n\n");

    // 运行简单测试
    const int M = 16;
    const int N = 16;
    const int K = 16;

    // 分配内存
    half *d_a, *d_b;
    float *d_c, *d_d;

    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, M * N * sizeof(float)));

    // 初始化
    half *h_a = (half*)malloc(M * K * sizeof(half));
    half *h_b = (half*)malloc(K * N * sizeof(half));
    float *h_c = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) h_c[i] = 0.0f;

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // 启动内核 (1 个 warp)
    dim3 grid(1, 1);
    dim3 block(32, 1);  // 1 warp

    wmmaSimpleGemm<<<grid, block>>>(d_a, d_b, d_c, d_d, M, N, K);

    // 验证结果
    float *h_d = (float*)malloc(M * N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_d, d_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("简单测试 (16x16 @ 16x16 全1矩阵):\n");
    printf("  结果 D[0] = %.1f (期望 = 16.0)\n", h_d[0]);
    printf("  结果 D[255] = %.1f (期望 = 16.0)\n\n", h_d[M*N-1]);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
}

// ============================================================================
// 第三部分：优化的 WMMA GEMM
// ============================================================================

// Tile 尺寸
const int BLOCK_SIZE = 16;
const int TILE_M = 64;
const int TILE_N = 64;
const int TILE_K = 16;

// 优化的 WMMA GEMM 内核
__global__ void wmmaOptimizedGemm(const half *__restrict__ A,
                                   const half *__restrict__ B,
                                   float *__restrict__ C,
                                   int M, int N, int K) {
    // 共享内存 tiles
    __shared__ half smemA[TILE_M][TILE_K];
    __shared__ half smemB[TILE_K][TILE_N];

    // 计算块处理的位置
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Warp 在块内的位置
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int warpRow = warpId / 4;  // 0-3
    int warpCol = warpId % 4;  // 0-3

    // 每个 warp 处理 16x16 的 tile
    // 一个块有 4x4 = 16 个 warp，处理 64x64 的输出

    // 声明 fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 循环遍历 K 维度的 tiles
    for (int tileK = 0; tileK < K; tileK += TILE_K) {
        // 协作加载 A 和 B 到共享内存
        // 每个线程加载多个元素
        int numLoadsA = (TILE_M * TILE_K) / (blockDim.x);
        int numLoadsB = (TILE_K * TILE_N) / (blockDim.x);

        for (int i = 0; i < numLoadsA; i++) {
            int idx = threadIdx.x + i * blockDim.x;
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int globalRow = blockRow * TILE_M + row;
            int globalCol = tileK + col;

            if (globalRow < M && globalCol < K) {
                smemA[row][col] = A[globalRow * K + globalCol];
            } else {
                smemA[row][col] = __float2half(0.0f);
            }
        }

        for (int i = 0; i < numLoadsB; i++) {
            int idx = threadIdx.x + i * blockDim.x;
            int row = idx / TILE_N;
            int col = idx % TILE_N;
            int globalRow = tileK + row;
            int globalCol = blockCol * TILE_N + col;

            if (globalRow < K && globalCol < N) {
                smemB[row][col] = B[globalRow * N + globalCol];
            } else {
                smemB[row][col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // 从共享内存加载并计算
        int aRow = warpRow * WMMA_M;
        int bCol = warpCol * WMMA_N;

        wmma::load_matrix_sync(a_frag, &smemA[aRow][0], TILE_K);
        wmma::load_matrix_sync(b_frag, &smemB[0][bCol], TILE_N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // 写回结果
    int cRow = blockRow * TILE_M + warpRow * WMMA_M;
    int cCol = blockCol * TILE_N + warpCol * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

void demoOptimizedWMMA() {
    printf("=== 第三部分：优化的 WMMA GEMM ===\n\n");

    printf("优化策略:\n");
    printf("  1. 使用共享内存缓存 tiles\n");
    printf("  2. 多个 warp 协作处理大 tile\n");
    printf("  3. 双缓冲隐藏延迟\n");
    printf("  4. 优化内存访问模式\n\n");

    printf("Tile 配置:\n");
    printf("  块处理: %d x %d\n", TILE_M, TILE_N);
    printf("  Warp 处理: %d x %d\n", WMMA_M, WMMA_N);
    printf("  每块 Warp 数: %d x %d = %d\n\n",
           TILE_M/WMMA_M, TILE_N/WMMA_N, (TILE_M/WMMA_M)*(TILE_N/WMMA_N));

    // 较大矩阵测试
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    half *d_a, *d_b;
    float *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    // 初始化
    half *h_a = (half*)malloc(M * K * sizeof(half));
    half *h_b = (half*)malloc(K * N * sizeof(half));

    for (int i = 0; i < M * K; i++) h_a[i] = __float2half((float)(rand() % 10) / 10.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half((float)(rand() % 10) / 10.0f);

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // 计算网格配置
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(256);  // 8 warps

    // 预热
    wmmaOptimizedGemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        wmmaOptimizedGemm<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;

    // 计算 TFLOPS
    double flops = 2.0 * M * N * K;  // GEMM FLOPs
    double tflops = flops / (elapsed / 1000.0) / 1e12;

    printf("性能测试 (%d x %d x %d):\n", M, N, K);
    printf("  时间: %.3f ms\n", elapsed);
    printf("  性能: %.2f TFLOPS\n\n", tflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// ============================================================================
// 第四部分：混合精度计算
// ============================================================================

// FP16 输入，FP32 累加，FP16 输出
__global__ void wmmaMixedPrecision(const half *A, const half *B,
                                    half *D, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;  // FP32 累加
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out_frag;   // FP16 输出

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int bCol = warpN * WMMA_N;

        if (k + WMMA_K <= K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 转换 FP32 累加器到 FP16 输出
    for (int i = 0; i < acc_frag.num_elements; i++) {
        out_frag.x[i] = __float2half(acc_frag.x[i]);
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::store_matrix_sync(D + cRow * N + cCol, out_frag, N, wmma::mem_row_major);
}

void demoMixedPrecision() {
    printf("=== 第四部分：混合精度计算 ===\n\n");

    printf("混合精度模式:\n");
    printf("  ┌─────────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("  │ 输入 A/B    │ 累加器 C    │ 输出 D      │ 说明        │\n");
    printf("  ├─────────────┼─────────────┼─────────────┼─────────────┤\n");
    printf("  │ FP16        │ FP16        │ FP16        │ 全 FP16     │\n");
    printf("  │ FP16        │ FP32        │ FP16        │ 推荐        │\n");
    printf("  │ FP16        │ FP32        │ FP32        │ 最高精度    │\n");
    printf("  │ TF32*       │ FP32        │ FP32        │ Ampere+     │\n");
    printf("  │ BF16*       │ FP32        │ FP32/BF16   │ Ampere+     │\n");
    printf("  └─────────────┴─────────────┴─────────────┴─────────────┘\n\n");

    printf("为什么使用混合精度:\n");
    printf("  1. 内存带宽: FP16 是 FP32 的一半\n");
    printf("  2. 计算吞吐: Tensor Cores 原生支持 FP16\n");
    printf("  3. 精度保留: FP32 累加防止精度损失\n\n");

    printf("精度注意事项:\n");
    printf("  - FP16 范围: ±65504, 精度约 3 位有效数字\n");
    printf("  - 大矩阵乘法可能累加超出范围\n");
    printf("  - 使用 Loss Scaling 防止梯度下溢\n\n");
}

// ============================================================================
// 第五部分：与 CUDA Cores 对比
// ============================================================================

// CUDA Cores 实现的 GEMM
__global__ void cudaCoreGemm(const half *A, const half *B, float *C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// CUDA Cores with shared memory
__global__ void cudaCoreGemmShared(const half *A, const half *B, float *C,
                                    int M, int N, int K) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // 加载 A tile
        if (row < M && tile * BLOCK_SIZE + tx < K) {
            sA[ty][tx] = __half2float(A[row * K + tile * BLOCK_SIZE + tx]);
        } else {
            sA[ty][tx] = 0.0f;
        }

        // 加载 B tile
        if (tile * BLOCK_SIZE + ty < K && col < N) {
            sB[ty][tx] = __half2float(B[(tile * BLOCK_SIZE + ty) * N + col]);
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void demoTensorCoreComparison() {
    printf("=== 第五部分：Tensor Cores vs CUDA Cores ===\n\n");

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    half *d_a, *d_b;
    float *d_c_tensor, *d_c_cuda;

    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c_tensor, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c_cuda, M * N * sizeof(float)));

    // 初始化
    half *h_a = (half*)malloc(M * K * sizeof(half));
    half *h_b = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(0.1f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(0.1f);
    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 50;

    // CUDA Cores (with shared memory)
    dim3 cudaBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 cudaGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 预热
    cudaCoreGemmShared<<<cudaGrid, cudaBlock>>>(d_a, d_b, d_c_cuda, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        cudaCoreGemmShared<<<cudaGrid, cudaBlock>>>(d_a, d_b, d_c_cuda, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cudaTime;
    CHECK_CUDA(cudaEventElapsedTime(&cudaTime, start, stop));
    cudaTime /= NUM_RUNS;

    // Tensor Cores
    dim3 tensorGrid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 tensorBlock(256);

    // 预热
    wmmaOptimizedGemm<<<tensorGrid, tensorBlock>>>(d_a, d_b, d_c_tensor, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        wmmaOptimizedGemm<<<tensorGrid, tensorBlock>>>(d_a, d_b, d_c_tensor, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float tensorTime;
    CHECK_CUDA(cudaEventElapsedTime(&tensorTime, start, stop));
    tensorTime /= NUM_RUNS;

    // 计算 TFLOPS
    double flops = 2.0 * M * N * K;
    double cudaTflops = flops / (cudaTime / 1000.0) / 1e12;
    double tensorTflops = flops / (tensorTime / 1000.0) / 1e12;

    printf("性能对比 (%d x %d x %d):\n", M, N, K);
    printf("  ┌─────────────────────┬───────────┬───────────┐\n");
    printf("  │ 方法                │ 时间 (ms) │ TFLOPS    │\n");
    printf("  ├─────────────────────┼───────────┼───────────┤\n");
    printf("  │ CUDA Cores (共享)   │ %9.3f │ %9.2f │\n", cudaTime, cudaTflops);
    printf("  │ Tensor Cores (WMMA) │ %9.3f │ %9.2f │\n", tensorTime, tensorTflops);
    printf("  └─────────────────────┴───────────┴───────────┘\n\n");
    printf("  加速比: %.2fx\n\n", cudaTime / tensorTime);

    // 验证结果
    float *h_c_tensor = (float*)malloc(M * N * sizeof(float));
    float *h_c_cuda = (float*)malloc(M * N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_c_tensor, d_c_tensor, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c_cuda, d_c_cuda, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_c_tensor[i] - h_c_cuda[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("结果验证:\n");
    printf("  最大差异: %.6f\n", maxDiff);
    printf("  (FP16 精度限制，差异正常)\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_a);
    free(h_b);
    free(h_c_tensor);
    free(h_c_cuda);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_tensor));
    CHECK_CUDA(cudaFree(d_c_cuda));
}

// ============================================================================
// 第六部分：实际应用场景
// ============================================================================

void demoApplications() {
    printf("=== 第六部分：实际应用场景 ===\n\n");

    printf("1. 深度学习训练:\n");
    printf("   - 全连接层 (Linear/Dense)\n");
    printf("   - 卷积层 (通过 im2col 转换)\n");
    printf("   - Attention 机制 (Q*K^T, Attention*V)\n\n");

    printf("2. 深度学习推理:\n");
    printf("   - TensorRT 自动使用 Tensor Cores\n");
    printf("   - INT8 量化推理\n");
    printf("   - FP8 推理 (Hopper+)\n\n");

    printf("3. 科学计算:\n");
    printf("   - 大规模线性代数\n");
    printf("   - 稀疏矩阵运算\n");
    printf("   - 信号处理\n\n");

    printf("使用 Tensor Cores 的库:\n");
    printf("  ┌────────────────┬────────────────────────────────────┐\n");
    printf("  │ 库             │ 说明                               │\n");
    printf("  ├────────────────┼────────────────────────────────────┤\n");
    printf("  │ cuBLAS         │ 自动使用 (cublasGemmEx)            │\n");
    printf("  │ cuDNN          │ 卷积和 GEMM                        │\n");
    printf("  │ CUTLASS        │ 可定制的 GEMM 模板                 │\n");
    printf("  │ TensorRT       │ 推理优化                           │\n");
    printf("  │ PyTorch        │ torch.matmul (自动)                │\n");
    printf("  │ TensorFlow     │ tf.matmul (自动)                   │\n");
    printf("  └────────────────┴────────────────────────────────────┘\n\n");

    printf("数据布局要求:\n");
    printf("  - 矩阵维度应是 8 或 16 的倍数\n");
    printf("  - 内存地址需要 16 字节对齐\n");
    printf("  - 使用正确的 leading dimension\n\n");
}

// ============================================================================
// 第七部分：最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第七部分：最佳实践 ===\n\n");

    printf("1. 选择正确的精度:\n");
    printf("   □ 训练: FP16 输入 + FP32 累加\n");
    printf("   □ 推理: INT8/FP16 取决于精度需求\n");
    printf("   □ 科学计算: TF32 或 FP64 (如可用)\n\n");

    printf("2. 优化内存访问:\n");
    printf("   □ 确保矩阵维度是 16 的倍数\n");
    printf("   □ 使用共享内存缓存\n");
    printf("   □ 避免 bank conflicts\n");
    printf("   □ 预取下一个 tile\n\n");

    printf("3. 最大化并行度:\n");
    printf("   □ 多个 warp 协作\n");
    printf("   □ 流水线化 (双缓冲)\n");
    printf("   □ 重叠内存访问和计算\n\n");

    printf("4. 调试技巧:\n");
    printf("   □ 先用小矩阵验证正确性\n");
    printf("   □ 对比 cuBLAS 结果\n");
    printf("   □ 检查 NaN/Inf\n");
    printf("   □ 使用 Nsight Compute 分析\n\n");

    printf("5. 性能调优:\n");
    printf("   □ 调整 tile 大小\n");
    printf("   □ 优化 warp 数量\n");
    printf("   □ 平衡计算和访存\n");
    printf("   □ 考虑使用 CUTLASS 库\n\n");

    printf("常见陷阱:\n");
    printf("  - 忘记检查架构支持 (需要 sm_70+)\n");
    printf("  - 矩阵尺寸不满足 WMMA 要求\n");
    printf("  - Fragment 生命周期管理错误\n");
    printf("  - 混淆 row_major 和 col_major\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 27: Warp 级矩阵操作与 Tensor Cores                  ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);

    // 检查 Tensor Core 支持
    bool hasTensorCores = prop.major >= 7;
    printf("Tensor Cores: %s\n\n", hasTensorCores ? "支持" : "不支持");

    if (!hasTensorCores) {
        printf("警告: 此设备不支持 Tensor Cores (需要 sm_70 或更高)\n");
        printf("部分示例将无法正确运行\n\n");
    }

    demoTensorCoresOverview();

    if (hasTensorCores) {
        demoWMMABasics();
        demoOptimizedWMMA();
        demoTensorCoreComparison();
    } else {
        printf("=== 跳过 WMMA 示例 (设备不支持) ===\n\n");
    }

    demoMixedPrecision();
    demoApplications();
    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. Tensor Cores 基础:\n");
    printf("   - 专用矩阵乘法硬件\n");
    printf("   - 支持多种精度 (FP16, INT8, TF32, BF16)\n");
    printf("   - 比 CUDA Cores 快数倍\n\n");

    printf("2. WMMA API:\n");
    printf("   - fragment: 矩阵片段类型\n");
    printf("   - load_matrix_sync: 加载矩阵\n");
    printf("   - mma_sync: 执行乘加\n");
    printf("   - store_matrix_sync: 存储结果\n\n");

    printf("3. 混合精度:\n");
    printf("   - FP16 输入减少带宽\n");
    printf("   - FP32 累加保持精度\n");
    printf("   - 根据需求选择输出精度\n\n");

    printf("4. 优化策略:\n");
    printf("   - 共享内存缓存\n");
    printf("   - 双缓冲流水线\n");
    printf("   - 正确的数据布局\n\n");

    printf("5. 实际使用:\n");
    printf("   - 优先使用 cuBLAS/CUTLASS\n");
    printf("   - 自定义内核需要仔细优化\n");
    printf("   - 注意架构兼容性\n\n");

    return 0;
}
