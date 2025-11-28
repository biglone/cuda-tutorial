/**
 * =============================================================================
 * CUDA 教程 28: 异步内存拷贝与流水线 (Async Copy & Pipeline)
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 CUDA 异步内存拷贝机制
 * 2. 学会使用 cuda::memcpy_async 和 cuda::pipeline
 * 3. 掌握多阶段流水线设计
 * 4. 理解内存拷贝与计算的重叠
 *
 * 关键概念：
 * - Asynchronous Copy (异步拷贝)
 * - Pipeline (流水线)
 * - Commit/Wait 模型
 * - 多缓冲 (Multi-buffering)
 *
 * 编译命令：
 *   nvcc -arch=sm_80 28_async_copy_pipeline.cu -o 28_async_copy_pipeline
 *
 * 注意：异步拷贝需要 Ampere (sm_80) 或更新架构
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：异步拷贝概述
// ============================================================================

void demoAsyncCopyOverview() {
    printf("=== 第一部分：异步拷贝概述 ===\n\n");

    printf("传统同步拷贝:\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │ Thread: Load ──→ Store ──→ Compute ──→ ...     │\n");
    printf("  │              ↑          ↑                       │\n");
    printf("  │           等待       等待                       │\n");
    printf("  └─────────────────────────────────────────────────┘\n\n");

    printf("异步拷贝 (Ampere+):\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │ Copy Unit:  Load[0] ─→ Load[1] ─→ Load[2] ──→   │\n");
    printf("  │ Thread:          Compute[0] ─→ Compute[1] ──→   │\n");
    printf("  │                  (重叠执行)                      │\n");
    printf("  └─────────────────────────────────────────────────┘\n\n");

    printf("异步拷贝优势:\n");
    printf("  1. 绕过寄存器，直接 Global → Shared\n");
    printf("  2. 不占用计算资源\n");
    printf("  3. 支持流水线化\n");
    printf("  4. 减少寄存器压力\n\n");

    printf("支持的架构:\n");
    printf("  - Ampere (sm_80): 引入 cp.async\n");
    printf("  - Hopper (sm_90): 增强的异步操作\n\n");

    printf("API 选项:\n");
    printf("  1. PTX: cp.async 指令\n");
    printf("  2. C++: cuda::memcpy_async\n");
    printf("  3. C++: cuda::pipeline\n\n");
}

// ============================================================================
// 第二部分：基本异步拷贝
// ============================================================================

// 传统同步拷贝
__global__ void syncCopyKernel(float *output, const float *input, int n) {
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 同步加载: Global → Register → Shared
    if (gid < n) {
        smem[tid] = input[gid];  // 经过寄存器
    }
    __syncthreads();

    // 计算
    if (gid < n) {
        output[gid] = smem[tid] * 2.0f;
    }
}

// 异步拷贝版本 (使用 cooperative_groups)
__global__ void asyncCopyKernel(float *output, const float *input, int n) {
    __shared__ float smem[256];

    auto block = cg::this_thread_block();
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 异步拷贝: Global → Shared (绕过寄存器)
    if (gid < n) {
        // 使用 memcpy_async 进行异步拷贝
        cg::memcpy_async(block, smem, input + blockIdx.x * blockDim.x,
                         sizeof(float) * blockDim.x);
    }

    // 等待异步拷贝完成
    cg::wait(block);

    // 计算
    if (gid < n) {
        output[gid] = smem[tid] * 2.0f;
    }
}

void demoBasicAsyncCopy() {
    printf("=== 第二部分：基本异步拷贝 ===\n\n");

    printf("memcpy_async 使用:\n");
    printf("  // 1. 包含头文件\n");
    printf("  #include <cooperative_groups.h>\n\n");

    printf("  // 2. 获取线程块\n");
    printf("  auto block = cg::this_thread_block();\n\n");

    printf("  // 3. 异步拷贝\n");
    printf("  cg::memcpy_async(block, dst_shared, src_global, size);\n\n");

    printf("  // 4. 等待完成\n");
    printf("  cg::wait(block);\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));

    // 初始化
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_input[i] = (float)i;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 同步版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        syncCopyKernel<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float syncTime;
    CHECK_CUDA(cudaEventElapsedTime(&syncTime, start, stop));
    syncTime /= NUM_RUNS;

    // 异步版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        asyncCopyKernel<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float asyncTime;
    CHECK_CUDA(cudaEventElapsedTime(&asyncTime, start, stop));
    asyncTime /= NUM_RUNS;

    printf("性能对比 (N = %d):\n", N);
    printf("  同步拷贝: %.3f ms\n", syncTime);
    printf("  异步拷贝: %.3f ms\n", asyncTime);
    printf("  加速比: %.2fx\n\n", syncTime / asyncTime);

    // 验证
    float *h_output = (float*)malloc(size);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (fabs(h_output[i] - h_input[i] * 2.0f) > 1e-5) correct = false;
    }
    printf("验证: %s\n\n", correct ? "通过" : "失败");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第三部分：cuda::pipeline API
// ============================================================================

// Pipeline 常量
constexpr int STAGES = 2;  // 流水线阶段数
constexpr int TILE_SIZE = 256;

// 使用 pipeline 的内核
__global__ void pipelineKernel(float *output, const float *input, int n) {
    __shared__ float smem[STAGES][TILE_SIZE];

    // 创建 pipeline
    auto pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int numTiles = (n + blockDim.x - 1) / blockDim.x;
    int tileStart = blockIdx.x * numTiles;
    int tileEnd = min(tileStart + numTiles, (n + blockDim.x - 1) / blockDim.x);

    // 预取第一个 tile
    for (int stage = 0; stage < STAGES - 1 && tileStart + stage < tileEnd; stage++) {
        int gid = (tileStart + stage) * blockDim.x + tid;
        if (gid < n) {
            pipe.producer_acquire();
            cuda::memcpy_async(&smem[stage][tid], &input[gid], sizeof(float), pipe);
            pipe.producer_commit();
        }
    }

    // 主循环
    for (int tile = tileStart; tile < tileEnd; tile++) {
        int stage = tile % STAGES;
        int nextStage = (tile + STAGES - 1) % STAGES;

        // 等待当前 tile 数据就绪
        pipe.consumer_wait();

        // 预取下一个 tile
        int nextTile = tile + STAGES - 1;
        if (nextTile < tileEnd) {
            int nextGid = nextTile * blockDim.x + tid;
            if (nextGid < n) {
                pipe.producer_acquire();
                cuda::memcpy_async(&smem[nextStage][tid], &input[nextGid], sizeof(float), pipe);
                pipe.producer_commit();
            }
        }

        // 计算当前 tile
        int gid = tile * blockDim.x + tid;
        if (gid < n) {
            output[gid] = smem[stage][tid] * 2.0f;
        }

        pipe.consumer_release();
    }
}

void demoPipelineAPI() {
    printf("=== 第三部分：cuda::pipeline API ===\n\n");

    printf("Pipeline 概念:\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │ Stage 0: [Load Tile 0] [Compute Tile 0]                 │\n");
    printf("  │ Stage 1:              [Load Tile 1] [Compute Tile 1]    │\n");
    printf("  │ Stage 0:                           [Load Tile 2] ...    │\n");
    printf("  │                                                         │\n");
    printf("  │ 计算和加载重叠执行！                                     │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");

    printf("Pipeline API:\n");
    printf("  // 创建 pipeline\n");
    printf("  auto pipe = cuda::make_pipeline();\n\n");

    printf("  // 生产者端 (加载数据)\n");
    printf("  pipe.producer_acquire();  // 获取槽位\n");
    printf("  cuda::memcpy_async(dst, src, size, pipe);  // 异步拷贝\n");
    printf("  pipe.producer_commit();   // 提交\n\n");

    printf("  // 消费者端 (使用数据)\n");
    printf("  pipe.consumer_wait();     // 等待数据就绪\n");
    printf("  // ... 使用数据 ...\n");
    printf("  pipe.consumer_release();  // 释放槽位\n\n");

    printf("流水线阶段数选择:\n");
    printf("  - 2 阶段: 最小内存，基本重叠\n");
    printf("  - 3-4 阶段: 更好隐藏延迟\n");
    printf("  - 更多阶段: 更多共享内存占用\n\n");
}

// ============================================================================
// 第四部分：多阶段流水线 GEMM
// ============================================================================

constexpr int GEMM_TILE_M = 64;
constexpr int GEMM_TILE_N = 64;
constexpr int GEMM_TILE_K = 16;
constexpr int GEMM_STAGES = 3;

// 简化的流水线 GEMM
__global__ void pipelineGemm(float *C, const float *A, const float *B,
                              int M, int N, int K) {
    __shared__ float smemA[GEMM_STAGES][GEMM_TILE_M][GEMM_TILE_K];
    __shared__ float smemB[GEMM_STAGES][GEMM_TILE_K][GEMM_TILE_N];

    auto pipe = cuda::make_pipeline();

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 每个线程负责计算 C 的一部分
    float acc = 0.0f;

    int numKTiles = (K + GEMM_TILE_K - 1) / GEMM_TILE_K;

    // 预取前几个阶段
    for (int stage = 0; stage < GEMM_STAGES - 1 && stage < numKTiles; stage++) {
        int kTile = stage;

        // 加载 A tile
        int aRow = by * GEMM_TILE_M + ty;
        int aCol = kTile * GEMM_TILE_K + tx;
        if (aRow < M && aCol < K && tx < GEMM_TILE_K && ty < GEMM_TILE_M) {
            pipe.producer_acquire();
            cuda::memcpy_async(&smemA[stage][ty][tx], &A[aRow * K + aCol], sizeof(float), pipe);
            pipe.producer_commit();
        }

        // 加载 B tile
        int bRow = kTile * GEMM_TILE_K + ty;
        int bCol = bx * GEMM_TILE_N + tx;
        if (bRow < K && bCol < N && ty < GEMM_TILE_K && tx < GEMM_TILE_N) {
            pipe.producer_acquire();
            cuda::memcpy_async(&smemB[stage][ty][tx], &B[bRow * N + bCol], sizeof(float), pipe);
            pipe.producer_commit();
        }
    }

    // 主循环
    for (int kTile = 0; kTile < numKTiles; kTile++) {
        int stage = kTile % GEMM_STAGES;

        // 等待当前阶段数据
        pipe.consumer_wait();

        // 预取下一个 tile
        int nextKTile = kTile + GEMM_STAGES - 1;
        if (nextKTile < numKTiles) {
            int nextStage = nextKTile % GEMM_STAGES;

            int aRow = by * GEMM_TILE_M + ty;
            int aCol = nextKTile * GEMM_TILE_K + tx;
            if (aRow < M && aCol < K && tx < GEMM_TILE_K && ty < GEMM_TILE_M) {
                pipe.producer_acquire();
                cuda::memcpy_async(&smemA[nextStage][ty][tx], &A[aRow * K + aCol], sizeof(float), pipe);
                pipe.producer_commit();
            }

            int bRow = nextKTile * GEMM_TILE_K + ty;
            int bCol = bx * GEMM_TILE_N + tx;
            if (bRow < K && bCol < N && ty < GEMM_TILE_K && tx < GEMM_TILE_N) {
                pipe.producer_acquire();
                cuda::memcpy_async(&smemB[nextStage][ty][tx], &B[bRow * N + bCol], sizeof(float), pipe);
                pipe.producer_commit();
            }
        }

        __syncthreads();

        // 计算
        if (ty < GEMM_TILE_M && tx < GEMM_TILE_N) {
            for (int k = 0; k < GEMM_TILE_K; k++) {
                acc += smemA[stage][ty][k] * smemB[stage][k][tx];
            }
        }

        pipe.consumer_release();
        __syncthreads();
    }

    // 写回结果
    int cRow = by * GEMM_TILE_M + ty;
    int cCol = bx * GEMM_TILE_N + tx;
    if (cRow < M && cCol < N && ty < GEMM_TILE_M && tx < GEMM_TILE_N) {
        C[cRow * N + cCol] = acc;
    }
}

void demoMultiStagePipeline() {
    printf("=== 第四部分：多阶段流水线 GEMM ===\n\n");

    printf("3 阶段流水线时序:\n");
    printf("  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │ Time →                                                       │\n");
    printf("  │                                                              │\n");
    printf("  │ Stage 0: Load[0]           Load[3]           Load[6]        │\n");
    printf("  │ Stage 1:       Load[1]           Load[4]           ...      │\n");
    printf("  │ Stage 2:             Load[2]           Load[5]              │\n");
    printf("  │                                                              │\n");
    printf("  │ Compute:       Comp[0] Comp[1] Comp[2] Comp[3] ...          │\n");
    printf("  │                                                              │\n");
    printf("  │ 每个计算都与后续加载重叠！                                    │\n");
    printf("  └──────────────────────────────────────────────────────────────┘\n\n");

    printf("内存布局:\n");
    printf("  __shared__ float smemA[STAGES][TILE_M][TILE_K];\n");
    printf("  __shared__ float smemB[STAGES][TILE_K][TILE_N];\n\n");

    printf("阶段数与延迟隐藏:\n");
    printf("  - 2 阶段: 隐藏 1 个 tile 加载延迟\n");
    printf("  - 3 阶段: 隐藏 2 个 tile 加载延迟\n");
    printf("  - 4 阶段: 隐藏 3 个 tile 加载延迟\n\n");

    // 简单测试
    const int M = 512;
    const int N = 512;
    const int K = 512;

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M * N * sizeof(float)));

    // 初始化
    float *h_a = (float*)malloc(M * K * sizeof(float));
    float *h_b = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_a[i] = 0.01f;
    for (int i = 0; i < K * N; i++) h_b[i] = 0.01f;
    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(GEMM_TILE_N, GEMM_TILE_M);
    dim3 grid((N + GEMM_TILE_N - 1) / GEMM_TILE_N, (M + GEMM_TILE_M - 1) / GEMM_TILE_M);

    // 由于线程块配置较大，这里简化测试
    printf("流水线 GEMM 配置:\n");
    printf("  矩阵大小: %d x %d x %d\n", M, N, K);
    printf("  Tile 大小: %d x %d x %d\n", GEMM_TILE_M, GEMM_TILE_N, GEMM_TILE_K);
    printf("  流水线阶段: %d\n\n", GEMM_STAGES);

    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// ============================================================================
// 第五部分：cp.async PTX 指令
// ============================================================================

// 使用 PTX cp.async 的内核
__global__ void cpAsyncPTX(float *output, const float *input, int n) {
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用 PTX cp.async 进行异步拷贝
    if (gid < n) {
        // cp.async.ca.shared.global [dst], [src], size;
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 4;"
            :: "l"(&smem[tid]), "l"(&input[gid])
        );
    }

    // 提交并等待
    asm volatile ("cp.async.commit_group;");
    asm volatile ("cp.async.wait_group 0;");

    __syncthreads();

    // 计算
    if (gid < n) {
        output[gid] = smem[tid] * 2.0f;
    }
}

// 多组 cp.async
__global__ void cpAsyncMultiGroup(float *output, const float *input, int n) {
    __shared__ float smem[2][256];  // 双缓冲

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 第一组异步拷贝
    if (gid < n) {
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 4;"
            :: "l"(&smem[0][tid]), "l"(&input[gid])
        );
    }
    asm volatile ("cp.async.commit_group;");  // 提交组 0

    // 第二组异步拷贝 (如果有)
    int gid2 = gid + gridDim.x * blockDim.x;
    if (gid2 < n) {
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 4;"
            :: "l"(&smem[1][tid]), "l"(&input[gid2])
        );
    }
    asm volatile ("cp.async.commit_group;");  // 提交组 1

    // 等待组 0 完成 (保留组 1 在飞行中)
    asm volatile ("cp.async.wait_group 1;");
    __syncthreads();

    // 处理第一组数据
    if (gid < n) {
        output[gid] = smem[0][tid] * 2.0f;
    }

    // 等待所有组完成
    asm volatile ("cp.async.wait_group 0;");
    __syncthreads();

    // 处理第二组数据
    if (gid2 < n) {
        output[gid2] = smem[1][tid] * 2.0f;
    }
}

void demoCpAsyncPTX() {
    printf("=== 第五部分：cp.async PTX 指令 ===\n\n");

    printf("cp.async 指令格式:\n");
    printf("  cp.async.{cache}.shared.global [dst], [src], size;\n\n");

    printf("缓存修饰符:\n");
    printf("  .ca - 缓存到所有级别\n");
    printf("  .cg - 只缓存到 L2\n");
    printf("  .cs - 流式 (可能不缓存)\n\n");

    printf("大小选项:\n");
    printf("  4  - 4 字节\n");
    printf("  8  - 8 字节\n");
    printf("  16 - 16 字节 (推荐，最高效)\n\n");

    printf("组管理:\n");
    printf("  cp.async.commit_group;     // 提交当前组\n");
    printf("  cp.async.wait_group N;     // 等待直到最多 N 组在飞行中\n");
    printf("  cp.async.wait_all;         // 等待所有组完成\n\n");

    printf("示例 (双缓冲):\n");
    printf("  // 提交组 0\n");
    printf("  cp.async ... // 拷贝到 buffer[0]\n");
    printf("  cp.async.commit_group;\n\n");

    printf("  // 提交组 1\n");
    printf("  cp.async ... // 拷贝到 buffer[1]\n");
    printf("  cp.async.commit_group;\n\n");

    printf("  // 等待组 0，组 1 继续在飞行中\n");
    printf("  cp.async.wait_group 1;\n");
    printf("  // 使用 buffer[0]\n\n");

    printf("  // 等待所有完成\n");
    printf("  cp.async.wait_group 0;\n");
    printf("  // 使用 buffer[1]\n\n");
}

// ============================================================================
// 第六部分：实际优化案例 - 矩阵转置
// ============================================================================

constexpr int TRANSPOSE_TILE = 32;

// 传统共享内存转置
__global__ void transposeSync(float *output, const float *input, int width, int height) {
    __shared__ float tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    int x = blockIdx.x * TRANSPOSE_TILE + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE + threadIdx.y;

    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 异步拷贝转置
__global__ void transposeAsync(float *output, const float *input, int width, int height) {
    __shared__ float tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    auto block = cg::this_thread_block();

    int x = blockIdx.x * TRANSPOSE_TILE + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE + threadIdx.y;
    int linearIdx = threadIdx.y * TRANSPOSE_TILE + threadIdx.x;

    // 异步拷贝整个 tile
    if (x < width && y < height) {
        // 使用 cp.async 进行 16 字节拷贝 (如果对齐)
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 4;"
            :: "l"(&tile[threadIdx.y][threadIdx.x]), "l"(&input[y * width + x])
        );
    }

    asm volatile ("cp.async.commit_group;");
    asm volatile ("cp.async.wait_group 0;");
    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE + threadIdx.y;

    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void demoTransposeOptimization() {
    printf("=== 第六部分：矩阵转置优化案例 ===\n\n");

    const int WIDTH = 4096;
    const int HEIGHT = 4096;
    const size_t size = WIDTH * HEIGHT * sizeof(float);

    float *d_input, *d_output_sync, *d_output_async;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output_sync, size));
    CHECK_CUDA(cudaMalloc(&d_output_async, size));

    // 初始化
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_input[i] = (float)i;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 block(TRANSPOSE_TILE, TRANSPOSE_TILE);
    dim3 grid((WIDTH + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
              (HEIGHT + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 同步版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transposeSync<<<grid, block>>>(d_output_sync, d_input, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float syncTime;
    CHECK_CUDA(cudaEventElapsedTime(&syncTime, start, stop));
    syncTime /= NUM_RUNS;

    // 异步版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transposeAsync<<<grid, block>>>(d_output_async, d_input, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float asyncTime;
    CHECK_CUDA(cudaEventElapsedTime(&asyncTime, start, stop));
    asyncTime /= NUM_RUNS;

    // 计算带宽
    float bandwidth_sync = 2.0f * size / (syncTime / 1000.0f) / 1e9;
    float bandwidth_async = 2.0f * size / (asyncTime / 1000.0f) / 1e9;

    printf("矩阵转置性能 (%d x %d):\n", WIDTH, HEIGHT);
    printf("  ┌────────────────────┬───────────┬─────────────┐\n");
    printf("  │ 方法               │ 时间 (ms) │ 带宽 (GB/s) │\n");
    printf("  ├────────────────────┼───────────┼─────────────┤\n");
    printf("  │ 同步 (传统)        │ %9.3f │ %11.2f │\n", syncTime, bandwidth_sync);
    printf("  │ 异步 (cp.async)    │ %9.3f │ %11.2f │\n", asyncTime, bandwidth_async);
    printf("  └────────────────────┴───────────┴─────────────┘\n\n");

    // 验证
    float *h_sync = (float*)malloc(size);
    float *h_async = (float*)malloc(size);
    CHECK_CUDA(cudaMemcpy(h_sync, d_output_sync, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_async, d_output_async, size, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < WIDTH * HEIGHT && correct; i++) {
        if (h_sync[i] != h_async[i]) correct = false;
    }
    printf("结果验证: %s\n\n", correct ? "通过" : "失败");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_sync);
    free(h_async);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_sync));
    CHECK_CUDA(cudaFree(d_output_async));
}

// ============================================================================
// 第七部分：最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第七部分：最佳实践 ===\n\n");

    printf("1. 选择正确的拷贝大小:\n");
    printf("   □ 优先使用 16 字节 (128 位) 拷贝\n");
    printf("   □ 确保源地址 16 字节对齐\n");
    printf("   □ 使用向量类型 (float4) 提高效率\n\n");

    printf("2. 流水线阶段数:\n");
    printf("   □ 从 2 阶段开始\n");
    printf("   □ 根据延迟增加阶段数\n");
    printf("   □ 考虑共享内存容量限制\n\n");

    printf("3. 同步策略:\n");
    printf("   □ 使用 wait_group N 而非 wait_all\n");
    printf("   □ 尽早提交，尽晚等待\n");
    printf("   □ 避免不必要的 __syncthreads()\n\n");

    printf("4. 数据对齐:\n");
    printf("   □ 确保 16 字节对齐以获得最佳性能\n");
    printf("   □ 使用 __align__(16) 属性\n");
    printf("   □ 对齐共享内存分配\n\n");

    printf("5. 调试技巧:\n");
    printf("   □ 先用同步版本验证正确性\n");
    printf("   □ 使用 Nsight Compute 分析\n");
    printf("   □ 检查异步操作是否实际发生\n\n");

    printf("6. 何时使用异步拷贝:\n");
    printf("   ✓ 内存带宽受限的内核\n");
    printf("   ✓ 可以流水线化的算法\n");
    printf("   ✓ 大量数据加载到共享内存\n");
    printf("   ✗ 小数据量操作\n");
    printf("   ✗ 计算密集型内核\n\n");

    printf("7. 架构兼容性:\n");
    printf("   - sm_70 (Volta): 基本异步支持\n");
    printf("   - sm_80 (Ampere): 完整 cp.async\n");
    printf("   - sm_90 (Hopper): TMA (更强大)\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 28: 异步内存拷贝与流水线                             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);

    bool supportsAsync = prop.major >= 8;
    printf("异步拷贝支持: %s\n\n", supportsAsync ? "完整支持" : "有限支持");

    demoAsyncCopyOverview();

    if (prop.major >= 7) {
        demoBasicAsyncCopy();
    } else {
        printf("=== 跳过异步拷贝示例 (需要 sm_70+) ===\n\n");
    }

    demoPipelineAPI();
    demoMultiStagePipeline();
    demoCpAsyncPTX();

    if (prop.major >= 8) {
        demoTransposeOptimization();
    } else {
        printf("=== 跳过转置优化示例 (需要 sm_80+) ===\n\n");
    }

    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 异步拷贝基础:\n");
    printf("   - 绕过寄存器直接拷贝\n");
    printf("   - 不占用计算资源\n");
    printf("   - 支持与计算重叠\n\n");

    printf("2. API 选项:\n");
    printf("   - cg::memcpy_async (简单场景)\n");
    printf("   - cuda::pipeline (复杂流水线)\n");
    printf("   - PTX cp.async (最大控制)\n\n");

    printf("3. 流水线设计:\n");
    printf("   - 双缓冲/多缓冲\n");
    printf("   - Commit/Wait 模型\n");
    printf("   - 平衡阶段数和内存\n\n");

    printf("4. 优化要点:\n");
    printf("   - 16 字节对齐拷贝\n");
    printf("   - 尽早提交，尽晚等待\n");
    printf("   - 适当的阶段数\n\n");

    printf("5. 应用场景:\n");
    printf("   - GEMM 流水线\n");
    printf("   - 卷积运算\n");
    printf("   - 内存带宽受限的内核\n\n");

    return 0;
}
