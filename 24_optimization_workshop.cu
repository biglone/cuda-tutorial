/**
 * =============================================================================
 * CUDA 教程 24: 性能优化综合实战
 * =============================================================================
 *
 * 学习目标：
 * 1. 掌握系统性的 CUDA 性能优化方法
 * 2. 学会识别和解决常见性能瓶颈
 * 3. 通过矩阵转置案例实践各种优化技术
 * 4. 了解性能分析工具的使用
 *
 * 关键概念：
 * - 内存合并 (Coalesced Memory Access)
 * - Bank Conflict 避免
 * - 占用率优化 (Occupancy)
 * - 指令级优化
 *
 * 编译命令：
 *   nvcc 24_optimization_workshop.cu -o 24_optimization_workshop
 *
 * 使用 Nsight 分析:
 *   nsys profile ./24_optimization_workshop
 *   ncu ./24_optimization_workshop
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 矩阵大小
#define MATRIX_SIZE 4096
#define TILE_DIM 32
#define BLOCK_ROWS 8

// ============================================================================
// 第一部分：性能优化概述
// ============================================================================

void demoOptimizationOverview() {
    printf("=== 第一部分：性能优化概述 ===\n\n");

    printf("CUDA 性能优化金字塔:\n");
    printf("  ┌─────────────────────────────────────────┐\n");
    printf("  │          算法级优化                     │ ← 最重要\n");
    printf("  │     (选择正确的并行算法)                │\n");
    printf("  ├─────────────────────────────────────────┤\n");
    printf("  │         内存级优化                      │\n");
    printf("  │  (合并访问、减少传输、缓存利用)         │\n");
    printf("  ├─────────────────────────────────────────┤\n");
    printf("  │        执行级优化                       │\n");
    printf("  │   (占用率、分支、指令级并行)            │\n");
    printf("  ├─────────────────────────────────────────┤\n");
    printf("  │        指令级优化                       │\n");
    printf("  │    (快速数学、内置函数)                 │ ← 微调\n");
    printf("  └─────────────────────────────────────────┘\n\n");

    printf("性能瓶颈类型:\n");
    printf("  1. 内存带宽限制 (Memory Bound)\n");
    printf("     - 全局内存带宽未充分利用\n");
    printf("     - 非合并访问\n");
    printf("     - 过多的内存传输\n\n");

    printf("  2. 计算限制 (Compute Bound)\n");
    printf("     - GPU 计算资源成为瓶颈\n");
    printf("     - 通常是好事 (充分利用 GPU)\n\n");

    printf("  3. 延迟限制 (Latency Bound)\n");
    printf("     - 启动开销\n");
    printf("     - 同步等待\n");
    printf("     - 分支发散\n\n");

    printf("优化工作流:\n");
    printf("  1. 建立基准 (Baseline)\n");
    printf("  2. 分析瓶颈 (Profile)\n");
    printf("  3. 应用优化 (Optimize)\n");
    printf("  4. 验证结果 (Validate)\n");
    printf("  5. 迭代改进\n\n");
}

// ============================================================================
// 第二部分：矩阵转置 - 基准版本
// ============================================================================

__global__ void transposeNaive(float *odata, float *idata, int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + i) < height) {
            odata[index_out + i] = idata[index_in + i * width];
        }
    }
}

// ============================================================================
// 第三部分：矩阵转置 - 合并读取版本
// ============================================================================

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + width * yIndex;

    int xIndexOut = blockIdx.y * TILE_DIM + threadIdx.x;
    int yIndexOut = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndexOut + height * yIndexOut;

    // 合并读取到共享内存
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
    }

    __syncthreads();

    // 合并写入 (但有 bank conflict!)
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndexOut < height && (yIndexOut + i) < width) {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// ============================================================================
// 第四部分：矩阵转置 - 无 Bank Conflict 版本
// ============================================================================

__global__ void transposeNoBankConflict(float *odata, float *idata, int width, int height) {
    // 关键优化: +1 避免 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + width * yIndex;

    int xIndexOut = blockIdx.y * TILE_DIM + threadIdx.x;
    int yIndexOut = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndexOut + height * yIndexOut;

    // 合并读取
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
    }

    __syncthreads();

    // 合并写入，无 bank conflict
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndexOut < height && (yIndexOut + i) < width) {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// ============================================================================
// 第五部分：矩阵转置 - 进一步优化版本
// ============================================================================

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height) {
    // 对角线重排序以减少分区冲突
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 使用对角线坐标
    int blockIdx_y = blockIdx.x;
    int blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + width * yIndex;

    int xIndexOut = blockIdx_y * TILE_DIM + threadIdx.x;
    int yIndexOut = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndexOut + height * yIndexOut;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
    }

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndexOut < height && (yIndexOut + i) < width) {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// 性能测试函数
void runTransposeBenchmark() {
    printf("=== 第二至五部分：矩阵转置优化实战 ===\n\n");

    const int width = MATRIX_SIZE;
    const int height = MATRIX_SIZE;
    const size_t size = width * height * sizeof(float);

    printf("矩阵大小: %d x %d\n", width, height);
    printf("数据量: %.2f MB\n\n", size / (1024.0f * 1024.0f));

    // 分配内存
    float *h_idata = (float*)malloc(size);
    float *h_odata = (float*)malloc(size);
    float *h_gold = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < width * height; i++) {
        h_idata[i] = (float)i;
    }

    // CPU 参考结果
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_gold[x * height + y] = h_idata[y * width + x];
        }
    }

    float *d_idata, *d_odata;
    CHECK_CUDA(cudaMalloc(&d_idata, size));
    CHECK_CUDA(cudaMalloc(&d_odata, size));
    CHECK_CUDA(cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice));

    dim3 grid(width / TILE_DIM, height / TILE_DIM);
    dim3 threads(TILE_DIM, BLOCK_ROWS);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 带宽计算辅助
    auto printBandwidth = [&](float time_ms, const char* name) {
        float bandwidth = 2.0f * size / (time_ms / 1000.0f) / 1e9;
        printf("  %-30s: %7.3f ms, 带宽: %6.2f GB/s\n", name, time_ms, bandwidth);
    };

    printf("性能对比 (运行 %d 次平均):\n\n", NUM_RUNS);

    // 1. 朴素版本
    CHECK_CUDA(cudaMemset(d_odata, 0, size));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transposeNaive<<<grid, threads>>>(d_odata, d_idata, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float naiveTime;
    CHECK_CUDA(cudaEventElapsedTime(&naiveTime, start, stop));
    naiveTime /= NUM_RUNS;
    printBandwidth(naiveTime, "朴素版本 (非合并)");

    // 验证
    CHECK_CUDA(cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < width * height && correct; i++) {
        if (h_odata[i] != h_gold[i]) correct = false;
    }
    printf("    验证: %s\n", correct ? "通过" : "失败");

    // 2. 合并读取版本
    CHECK_CUDA(cudaMemset(d_odata, 0, size));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transposeCoalesced<<<grid, threads>>>(d_odata, d_idata, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float coalescedTime;
    CHECK_CUDA(cudaEventElapsedTime(&coalescedTime, start, stop));
    coalescedTime /= NUM_RUNS;
    printBandwidth(coalescedTime, "合并读取 (有 bank conflict)");

    CHECK_CUDA(cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost));
    correct = true;
    for (int i = 0; i < width * height && correct; i++) {
        if (h_odata[i] != h_gold[i]) correct = false;
    }
    printf("    验证: %s, 加速比: %.2fx\n", correct ? "通过" : "失败", naiveTime / coalescedTime);

    // 3. 无 Bank Conflict 版本
    CHECK_CUDA(cudaMemset(d_odata, 0, size));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transposeNoBankConflict<<<grid, threads>>>(d_odata, d_idata, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float noBankConflictTime;
    CHECK_CUDA(cudaEventElapsedTime(&noBankConflictTime, start, stop));
    noBankConflictTime /= NUM_RUNS;
    printBandwidth(noBankConflictTime, "无 Bank Conflict");

    CHECK_CUDA(cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost));
    correct = true;
    for (int i = 0; i < width * height && correct; i++) {
        if (h_odata[i] != h_gold[i]) correct = false;
    }
    printf("    验证: %s, 加速比: %.2fx\n", correct ? "通过" : "失败", naiveTime / noBankConflictTime);

    // 4. 对角线优化版本
    CHECK_CUDA(cudaMemset(d_odata, 0, size));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transposeDiagonal<<<grid, threads>>>(d_odata, d_idata, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float diagonalTime;
    CHECK_CUDA(cudaEventElapsedTime(&diagonalTime, start, stop));
    diagonalTime /= NUM_RUNS;
    printBandwidth(diagonalTime, "对角线优化");

    CHECK_CUDA(cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost));
    correct = true;
    for (int i = 0; i < width * height && correct; i++) {
        if (h_odata[i] != h_gold[i]) correct = false;
    }
    printf("    验证: %s, 加速比: %.2fx\n\n", correct ? "通过" : "失败", naiveTime / diagonalTime);

    // 理论带宽
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    // 注意: memoryClockRate 在 CUDA 12+ 已弃用
#if CUDART_VERSION < 12000
    float theoreticalBandwidth = prop.memoryClockRate * 1e3 *
                                  (prop.memoryBusWidth / 8) * 2 / 1e9;
    printf("理论峰值带宽: %.2f GB/s\n\n", theoreticalBandwidth);
#else
    printf("内存总线宽度: %d bits\n\n", prop.memoryBusWidth);
#endif

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_idata);
    free(h_odata);
    free(h_gold);
    CHECK_CUDA(cudaFree(d_idata));
    CHECK_CUDA(cudaFree(d_odata));
}

// ============================================================================
// 第六部分：内存优化详解
// ============================================================================

void demoMemoryOptimization() {
    printf("=== 第六部分：内存优化详解 ===\n\n");

    printf("1. 内存合并 (Coalesced Access):\n");
    printf("   好的访问模式:\n");
    printf("   ┌───┬───┬───┬───┬───┬───┬───┬───┐\n");
    printf("   │ T0│ T1│ T2│ T3│ T4│ T5│ T6│ T7│  → 连续地址\n");
    printf("   └───┴───┴───┴───┴───┴───┴───┴───┘\n");
    printf("   = 1 次内存事务\n\n");

    printf("   差的访问模式 (strided):\n");
    printf("   ┌───┬───┬───┬───┬───┬───┬───┬───┐\n");
    printf("   │ T0│   │ T1│   │ T2│   │ T3│   │  → 间隔访问\n");
    printf("   └───┴───┴───┴───┴───┴───┴───┴───┘\n");
    printf("   = 多次内存事务\n\n");

    printf("2. Shared Memory Bank Conflict:\n");
    printf("   32 个 bank，每个 4 字节宽\n");
    printf("   \n");
    printf("   无冲突:               有冲突:\n");
    printf("   T0→Bank0             T0→Bank0\n");
    printf("   T1→Bank1             T1→Bank0  ← 冲突!\n");
    printf("   T2→Bank2             T2→Bank0  ← 冲突!\n");
    printf("   ...\n\n");

    printf("   解决方案: padding\n");
    printf("   __shared__ float tile[32][32];     // 有 conflict\n");
    printf("   __shared__ float tile[32][32+1];   // 无 conflict\n\n");

    printf("3. 内存访问模式检查清单:\n");
    printf("   □ 全局内存读写是否合并?\n");
    printf("   □ 共享内存是否有 bank conflict?\n");
    printf("   □ 是否可以使用只读缓存 (__ldg)?\n");
    printf("   □ 是否可以使用纹理内存?\n");
    printf("   □ 是否有不必要的内存传输?\n\n");
}

// ============================================================================
// 第七部分：占用率优化
// ============================================================================

__global__ void occupancyDemo(float *output, float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid] * 2.0f;
    }
}

void demoOccupancyOptimization() {
    printf("=== 第七部分：占用率优化 ===\n\n");

    printf("占用率 (Occupancy) 定义:\n");
    printf("  实际活跃 warp 数 / SM 最大支持 warp 数\n\n");

    printf("影响占用率的因素:\n");
    printf("  1. 每块线程数 (Threads per Block)\n");
    printf("  2. 每线程寄存器数 (Registers per Thread)\n");
    printf("  3. 每块共享内存 (Shared Memory per Block)\n\n");

    // 使用占用率 API
    int blockSize;
    int minGridSize;

    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize,
        occupancyDemo, 0, 0));

    printf("推荐配置 (occupancyDemo):\n");
    printf("  建议块大小: %d\n", blockSize);
    printf("  最小网格大小: %d\n\n", minGridSize);

    // 计算占用率
    int maxActiveBlocks;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, occupancyDemo, blockSize, 0));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    int actualWarps = maxActiveBlocks * (blockSize / 32);
    float occupancy = (float)actualWarps / maxWarpsPerSM * 100;

    printf("占用率计算:\n");
    printf("  每 SM 最大 warp: %d\n", maxWarpsPerSM);
    printf("  每 SM 最大活跃块: %d\n", maxActiveBlocks);
    printf("  实际活跃 warp: %d\n", actualWarps);
    printf("  理论占用率: %.1f%%\n\n", occupancy);

    printf("优化建议:\n");
    printf("  - 占用率 > 50%% 通常足够\n");
    printf("  - 过高占用率可能增加寄存器压力\n");
    printf("  - 使用 launch_bounds 控制编译器\n\n");

    printf("示例: 使用 launch_bounds\n");
    printf("  __global__ void __launch_bounds__(256, 2) kernel() {\n");
    printf("      // 最大 256 线程/块, 至少 2 块/SM\n");
    printf("  }\n\n");
}

// ============================================================================
// 第八部分：分支优化
// ============================================================================

__global__ void branchDivergentKernel(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 分支发散: warp 内不同线程走不同路径
    if (tid % 2 == 0) {
        data[tid] = sinf(data[tid]);
    } else {
        data[tid] = cosf(data[tid]);
    }
}

__global__ void branchOptimizedKernel(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 优化: 使用无分支计算
    float val = data[tid];
    float sinVal = sinf(val);
    float cosVal = cosf(val);

    // 选择 (无分支)
    float mask = (float)(tid % 2 == 0);
    data[tid] = mask * sinVal + (1.0f - mask) * cosVal;
}

__global__ void branchWarpAlignedKernel(float *evenData, float *oddData, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 优化: 按 warp 对齐处理
    // 前半部分 warp 处理偶数，后半部分处理奇数
    int halfN = n / 2;

    if (tid < halfN) {
        evenData[tid] = sinf(evenData[tid]);
    }

    if (tid < halfN) {
        oddData[tid] = cosf(oddData[tid]);
    }
}

void demoBranchOptimization() {
    printf("=== 第八部分：分支优化 ===\n\n");

    printf("Warp 分支发散问题:\n");
    printf("  ┌─────────────────────────────────────────────┐\n");
    printf("  │  if (tid %% 2 == 0) {                       │\n");
    printf("  │      // 偶数线程执行 (奇数线程等待)        │\n");
    printf("  │  } else {                                   │\n");
    printf("  │      // 奇数线程执行 (偶数线程等待)        │\n");
    printf("  │  }                                          │\n");
    printf("  │  // Warp 串行执行两个分支!                  │\n");
    printf("  └─────────────────────────────────────────────┘\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 初始化
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = (float)i * 0.01f;
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 分支发散版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        branchDivergentKernel<<<gridSize, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float divergentTime;
    CHECK_CUDA(cudaEventElapsedTime(&divergentTime, start, stop));
    divergentTime /= NUM_RUNS;

    // 重置数据
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // 无分支版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        branchOptimizedKernel<<<gridSize, blockSize>>>(d_data, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float optimizedTime;
    CHECK_CUDA(cudaEventElapsedTime(&optimizedTime, start, stop));
    optimizedTime /= NUM_RUNS;

    printf("性能对比:\n");
    printf("  分支发散版本: %.3f ms\n", divergentTime);
    printf("  无分支版本:   %.3f ms\n", optimizedTime);
    printf("  加速比: %.2fx\n\n", divergentTime / optimizedTime);

    printf("优化策略:\n");
    printf("  1. 使用谓词执行代替分支\n");
    printf("  2. 重组数据使相同操作的线程在同一 warp\n");
    printf("  3. 使用 warp 投票函数统一分支\n");
    printf("  4. 循环展开减少分支\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 第九部分：指令级优化
// ============================================================================

__global__ void mathSlowKernel(float *output, float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = input[tid];
    // 精确但慢的数学函数
    output[tid] = sinf(x) * cosf(x) + expf(x) / (1.0f + x * x);
}

__global__ void mathFastKernel(float *output, float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = input[tid];
    // 快速数学函数 (精度略低)
    output[tid] = __sinf(x) * __cosf(x) + __expf(x) / (1.0f + x * x);
}

__global__ void mathFusedKernel(float *output, float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = input[tid];
    // 使用融合乘加 (FMA)
    output[tid] = fmaf(__sinf(x), __cosf(x), __expf(x) * __frcp_rn(fmaf(x, x, 1.0f)));
}

void demoInstructionOptimization() {
    printf("=== 第九部分：指令级优化 ===\n\n");

    printf("1. 快速数学函数:\n");
    printf("   ┌──────────────┬──────────────┬─────────────────┐\n");
    printf("   │ 标准函数     │ 快速函数     │ 精度            │\n");
    printf("   ├──────────────┼──────────────┼─────────────────┤\n");
    printf("   │ sinf(x)      │ __sinf(x)    │ ~2 ulp          │\n");
    printf("   │ cosf(x)      │ __cosf(x)    │ ~2 ulp          │\n");
    printf("   │ expf(x)      │ __expf(x)    │ ~2 ulp          │\n");
    printf("   │ logf(x)      │ __logf(x)    │ ~3 ulp          │\n");
    printf("   │ powf(x,y)    │ __powf(x,y)  │ ~8 ulp          │\n");
    printf("   │ 1/x          │ __frcp_rn(x) │ IEEE 舍入       │\n");
    printf("   │ 1/sqrt(x)    │ __frsqrt_rn  │ IEEE 舍入       │\n");
    printf("   └──────────────┴──────────────┴─────────────────┘\n\n");

    printf("2. 编译器选项:\n");
    printf("   -use_fast_math : 启用所有快速数学\n");
    printf("   -fmad=true     : 启用融合乘加\n");
    printf("   -prec-div=false: 快速除法\n");
    printf("   -prec-sqrt=false: 快速开方\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_input[i] = (float)i * 0.001f;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 标准版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        mathSlowKernel<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float slowTime;
    CHECK_CUDA(cudaEventElapsedTime(&slowTime, start, stop));
    slowTime /= NUM_RUNS;

    // 快速版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        mathFastKernel<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float fastTime;
    CHECK_CUDA(cudaEventElapsedTime(&fastTime, start, stop));
    fastTime /= NUM_RUNS;

    // 融合版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        mathFusedKernel<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float fusedTime;
    CHECK_CUDA(cudaEventElapsedTime(&fusedTime, start, stop));
    fusedTime /= NUM_RUNS;

    printf("3. 性能对比:\n");
    printf("   标准数学函数: %.3f ms\n", slowTime);
    printf("   快速数学函数: %.3f ms (%.2fx)\n", fastTime, slowTime / fastTime);
    printf("   融合优化版本: %.3f ms (%.2fx)\n\n", fusedTime, slowTime / fusedTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第十部分：性能分析工具使用
// ============================================================================

void demoProfilingTools() {
    printf("=== 第十部分：性能分析工具 ===\n\n");

    printf("1. Nsight Systems (系统级分析):\n");
    printf("   用途: 整体时间线、CPU-GPU 交互、内存传输\n");
    printf("   命令: nsys profile ./my_cuda_app\n");
    printf("   输出: timeline.qdrep (用 Nsight Systems 打开)\n\n");

    printf("2. Nsight Compute (内核级分析):\n");
    printf("   用途: 详细的内核性能分析\n");
    printf("   命令: ncu --set full ./my_cuda_app\n");
    printf("   关键指标:\n");
    printf("     - SM 效率\n");
    printf("     - 内存带宽利用率\n");
    printf("     - 占用率\n");
    printf("     - L1/L2 缓存命中率\n\n");

    printf("3. CUDA 内置分析:\n");
    printf("   // 使用 CUDA Events\n");
    printf("   cudaEventRecord(start);\n");
    printf("   kernel<<<...>>>();\n");
    printf("   cudaEventRecord(stop);\n");
    printf("   cudaEventSynchronize(stop);\n");
    printf("   cudaEventElapsedTime(&ms, start, stop);\n\n");

    printf("4. 关键性能指标:\n");
    printf("   ┌────────────────────────┬──────────────────────────┐\n");
    printf("   │ 指标                   │ 目标                     │\n");
    printf("   ├────────────────────────┼──────────────────────────┤\n");
    printf("   │ 占用率                 │ > 50%%                    │\n");
    printf("   │ 内存带宽效率           │ > 60%% 理论峰值          │\n");
    printf("   │ SM 活跃周期            │ > 80%%                    │\n");
    printf("   │ L2 缓存命中率          │ 取决于访问模式           │\n");
    printf("   │ 分支效率               │ > 90%%                    │\n");
    printf("   └────────────────────────┴──────────────────────────┘\n\n");

    printf("5. 常见性能问题诊断:\n");
    printf("   问题                          可能原因\n");
    printf("   ─────────────────────────────────────────────────\n");
    printf("   内存带宽低                    非合并访问\n");
    printf("   SM 利用率低                   块太少/占用率低\n");
    printf("   L1 命中率低                   随机访问/工作集大\n");
    printf("   指令吞吐低                    寄存器压力/依赖\n");
    printf("   同步开销大                    过多 __syncthreads\n\n");
}

// ============================================================================
// 第十一部分：优化清单
// ============================================================================

void demoOptimizationChecklist() {
    printf("=== 第十一部分：优化清单 ===\n\n");

    printf("□ 内存优化:\n");
    printf("  □ 全局内存访问是否合并\n");
    printf("  □ 是否使用共享内存缓存重复访问的数据\n");
    printf("  □ 共享内存是否有 bank conflict\n");
    printf("  □ 是否使用常量内存存储只读数据\n");
    printf("  □ 是否使用纹理内存 (空间局部性)\n");
    printf("  □ 是否最小化 CPU-GPU 数据传输\n");
    printf("  □ 是否使用 pinned memory 加速传输\n");
    printf("  □ 是否使用异步传输重叠计算\n\n");

    printf("□ 执行优化:\n");
    printf("  □ 占用率是否足够 (>50%%)\n");
    printf("  □ 块大小是否合适 (通常 128-512)\n");
    printf("  □ 是否有分支发散\n");
    printf("  □ 是否有不必要的同步\n");
    printf("  □ 循环是否展开\n\n");

    printf("□ 算法优化:\n");
    printf("  □ 是否选择了适合 GPU 的算法\n");
    printf("  □ 是否有足够的并行度\n");
    printf("  □ 工作负载是否均衡\n");
    printf("  □ 是否减少了冗余计算\n\n");

    printf("□ 指令优化:\n");
    printf("  □ 是否使用快速数学函数 (可接受精度损失时)\n");
    printf("  □ 是否使用 FMA 指令\n");
    printf("  □ 是否使用 __ldg 进行只读访问\n");
    printf("  □ 是否使用内置函数 (min, max, etc.)\n\n");

    printf("□ 高级优化:\n");
    printf("  □ 是否使用 CUDA Streams 并发执行\n");
    printf("  □ 是否使用 CUDA Graphs 减少启动开销\n");
    printf("  □ 是否考虑多 GPU 扩展\n");
    printf("  □ 是否使用库函数 (cuBLAS, cuFFT 等)\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 24: 性能优化综合实战                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("SM 数量: %d\n", prop.multiProcessorCount);
    printf("显存大小: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    // 注意: memoryClockRate 在 CUDA 12+ 已弃用
#if CUDART_VERSION < 12000
    printf("显存带宽: %.2f GB/s (理论)\n\n",
           prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9);
#else
    printf("内存总线宽度: %d bits\n\n", prop.memoryBusWidth);
#endif

    demoOptimizationOverview();
    runTransposeBenchmark();
    demoMemoryOptimization();
    demoOccupancyOptimization();
    demoBranchOptimization();
    demoInstructionOptimization();
    demoProfilingTools();
    demoOptimizationChecklist();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 优化优先级:\n");
    printf("   算法 > 内存 > 执行 > 指令\n\n");

    printf("2. 内存关键点:\n");
    printf("   - 合并访问\n");
    printf("   - 避免 bank conflict\n");
    printf("   - 使用共享内存缓存\n\n");

    printf("3. 执行关键点:\n");
    printf("   - 保持足够占用率\n");
    printf("   - 减少分支发散\n");
    printf("   - 平衡工作负载\n\n");

    printf("4. 分析工具:\n");
    printf("   - Nsight Systems: 系统级\n");
    printf("   - Nsight Compute: 内核级\n\n");

    printf("5. 迭代优化:\n");
    printf("   测量 → 分析 → 优化 → 验证 → 重复\n\n");

    return 0;
}
