/**
 * =============================================================================
 * CUDA 教程 12: 性能分析与调试
 * =============================================================================
 *
 * 学习目标：
 * 1. 掌握 CUDA 错误检查和调试技术
 * 2. 学会使用计时方法分析性能
 * 3. 了解 Nsight Systems 和 Nsight Compute 工具
 * 4. 学习常见性能问题的诊断方法
 *
 * 关键概念：
 * - 错误处理：cudaGetLastError, cudaPeekAtLastError
 * - 计时：cudaEvent, CPU 计时器
 * - 性能指标：占用率、带宽、FLOPS
 * - 分析工具：nsys, ncu, cuda-memcheck
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"
#include <chrono>

// ============================================================================
// 第一部分：错误检查
// ============================================================================

// 基础错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 带变量名的详细错误检查
#define CHECK_CUDA_VERBOSE(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("错误: %s\n", #call); \
        printf("  文件: %s, 行: %d\n", __FILE__, __LINE__); \
        printf("  错误码: %d\n", err); \
        printf("  错误名: %s\n", cudaGetErrorName(err)); \
        printf("  错误描述: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 检查内核启动错误
#define CHECK_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("内核启动错误: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        printf("内核执行错误: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 故意制造错误的核函数
__global__ void buggyKernel(int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 可能的越界访问
    data[tid] = tid;  // 如果 tid >= n 则越界
}

void demoErrorChecking() {
    printf("=== 第一部分：错误检查 ===\n\n");

    printf("1. 常见 CUDA 错误类型:\n");
    printf("   - cudaErrorInvalidValue: 无效参数\n");
    printf("   - cudaErrorMemoryAllocation: 内存分配失败\n");
    printf("   - cudaErrorInvalidDevice: 无效设备\n");
    printf("   - cudaErrorInvalidConfiguration: 无效内核配置\n");
    printf("   - cudaErrorIllegalAddress: 非法内存访问\n\n");

    // 演示错误检查
    printf("2. 错误检查示例:\n");

    // 尝试分配过大的内存
    float *d_huge;
    cudaError_t err = cudaMalloc(&d_huge, (size_t)1024 * 1024 * 1024 * 100);  // 100GB
    if (err != cudaSuccess) {
        printf("   预期的错误 - 分配 100GB: %s\n", cudaGetErrorString(err));
    }
    // 重置错误状态
    cudaGetLastError();

    // 尝试无效的内核配置
    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, 100 * sizeof(int)));

    // 尝试启动过大的线程块
    printf("   测试无效线程块大小...\n");
    buggyKernel<<<1, 2048>>>(d_data, 100);  // 大多数GPU不支持2048线程
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("   预期的错误 - 线程块过大: %s\n", cudaGetErrorString(err));
    }

    CHECK_CUDA(cudaFree(d_data));

    printf("\n3. 错误检查最佳实践:\n");
    printf("   - 每个 CUDA API 调用后检查错误\n");
    printf("   - 内核启动后使用 cudaGetLastError()\n");
    printf("   - 开发阶段使用 cudaDeviceSynchronize() 捕获异步错误\n");
    printf("   - 生产代码中权衡错误检查的开销\n\n");
}

// ============================================================================
// 第二部分：计时方法
// ============================================================================

__global__ void computeKernel(float *data, int n, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) * cosf(val) + 0.1f;
        }
        data[tid] = val;
    }
}

void demoTiming() {
    printf("=== 第二部分：计时方法 ===\n\n");

    const int N = 1 << 20;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(float)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 方法 1: CUDA Events（推荐）
    printf("方法 1: CUDA Events\n");
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // 预热
        computeKernel<<<gridSize, blockSize>>>(d_data, N, 100);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            computeKernel<<<gridSize, blockSize>>>(d_data, N, 100);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("   总时间: %.3f ms, 平均: %.3f ms\n", milliseconds, milliseconds / 10);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // 方法 2: CPU 计时器（需要同步）
    printf("\n方法 2: CPU 计时器 (std::chrono)\n");
    {
        CHECK_CUDA(cudaDeviceSynchronize());  // 确保之前操作完成

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            computeKernel<<<gridSize, blockSize>>>(d_data, N, 100);
        }
        CHECK_CUDA(cudaDeviceSynchronize());  // 必须同步!
        auto end = std::chrono::high_resolution_clock::now();

        double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
        printf("   总时间: %.3f ms, 平均: %.3f ms\n", milliseconds, milliseconds / 10);
    }

    // 方法 3: 使用 cudaEventQuery 非阻塞检查
    printf("\n方法 3: 非阻塞计时\n");
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        computeKernel<<<gridSize, blockSize>>>(d_data, N, 100);
        CHECK_CUDA(cudaEventRecord(stop));

        // 非阻塞等待
        while (cudaEventQuery(stop) == cudaErrorNotReady) {
            // 可以在此期间做其他工作
        }

        float milliseconds;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("   时间: %.3f ms\n", milliseconds);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    printf("\n计时注意事项:\n");
    printf("  - CUDA Events 精度更高，推荐使用\n");
    printf("  - CPU 计时必须在适当位置调用同步\n");
    printf("  - 第一次调用通常较慢（预热）\n");
    printf("  - 多次运行取平均值\n\n");

    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 第三部分：性能指标计算
// ============================================================================

__global__ void bandwidthTestKernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void computeBoundKernel(float *data, int n, int ops) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];
        for (int i = 0; i < ops; i++) {
            val = fmaf(val, 1.001f, 0.001f);  // FMA 指令
        }
        data[tid] = val;
    }
}

void demoPerformanceMetrics() {
    printf("=== 第三部分：性能指标 ===\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // 理论峰值 - 使用版本兼容性宏自动处理 CUDA 12+ API 弃用问题
    float peakBandwidth = GET_MEMORY_BANDWIDTH_GBPS(prop);
    float peakFLOPS = GET_CLOCK_RATE_MHZ(prop) > 0
        ? prop.multiProcessorCount * GET_CLOCK_RATE_MHZ(prop) * 2.0f / 1000.0f  // 从 MHz 转换
        : prop.multiProcessorCount * 128.0f * 2.0f;  // CUDA 12+ 估算值（每 SM 128 核心）

    printf("设备理论峰值:\n");
    printf("  内存带宽: %.1f GB/s (估算)\n", peakBandwidth);
    printf("  计算能力: ~%.0f GFLOPS (估算)\n\n", peakFLOPS);

    const int N = 1 << 24;  // 16M 元素
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测试内存带宽受限的内核
    printf("1. 内存带宽测试 (向量加法):\n");
    {
        // 预热
        bandwidthTestKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            bandwidthTestKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= 10;

        // 读取 a, b，写入 c = 3N 次内存访问
        float bandwidth = 3.0f * N * sizeof(float) / (ms * 1e6);  // GB/s
        float efficiency = bandwidth / peakBandwidth * 100;

        printf("   数据量: %.0f MB\n", 3.0f * N * sizeof(float) / (1024 * 1024));
        printf("   时间: %.3f ms\n", ms);
        printf("   实际带宽: %.1f GB/s\n", bandwidth);
        printf("   带宽效率: %.1f%%\n\n", efficiency);
    }

    // 测试计算受限的内核
    printf("2. 计算吞吐量测试:\n");
    {
        const int OPS = 1000;  // 每个元素 1000 次 FMA

        // 预热
        computeBoundKernel<<<gridSize, blockSize>>>(d_a, N, OPS);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            computeBoundKernel<<<gridSize, blockSize>>>(d_a, N, OPS);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= 10;

        // FMA = 2 FLOP
        float flops = (float)N * OPS * 2 / (ms * 1e6);  // GFLOPS

        printf("   每元素操作数: %d FMA = %d FLOP\n", OPS, OPS * 2);
        printf("   时间: %.3f ms\n", ms);
        printf("   计算吞吐: %.1f GFLOPS\n\n", flops);
    }

    // 占用率计算
    printf("3. 占用率分析:\n");
    {
        int maxActiveBlocks;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, computeBoundKernel, blockSize, 0));

        int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        float occupancy = (float)(maxActiveBlocks * blockSize) / maxThreadsPerSM;

        printf("   线程块大小: %d\n", blockSize);
        printf("   每 SM 最大活跃块: %d\n", maxActiveBlocks);
        printf("   每 SM 活跃线程: %d\n", maxActiveBlocks * blockSize);
        printf("   每 SM 最大线程: %d\n", maxThreadsPerSM);
        printf("   理论占用率: %.1f%%\n\n", occupancy * 100);

        // 建议最佳块大小
        int minGridSize, optBlockSize;
        CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &optBlockSize, computeBoundKernel, 0, 0));
        printf("   建议块大小: %d (自动优化)\n", optBlockSize);
        printf("   建议最小网格: %d\n\n", minGridSize);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// ============================================================================
// 第四部分：分析工具使用指南
// ============================================================================

void demoProfilingTools() {
    printf("=== 第四部分：分析工具 ===\n\n");

    printf("1. Nsight Systems (nsys) - 系统级分析:\n");
    printf("   用途: 分析整体应用性能，包括 CPU 和 GPU 活动\n");
    printf("   命令:\n");
    printf("     nsys profile ./your_program\n");
    printf("     nsys profile -o report ./your_program\n");
    printf("   输出: 时间线、API 调用、内核执行、内存传输\n\n");

    printf("2. Nsight Compute (ncu) - 内核级分析:\n");
    printf("   用途: 深入分析单个内核的性能\n");
    printf("   命令:\n");
    printf("     ncu ./your_program\n");
    printf("     ncu --set full ./your_program\n");
    printf("     ncu --kernel-name computeKernel ./your_program\n");
    printf("   输出: 性能计数器、占用率、内存访问模式\n\n");

    printf("3. cuda-memcheck - 内存检查:\n");
    printf("   用途: 检测内存访问错误\n");
    printf("   命令:\n");
    printf("     cuda-memcheck ./your_program\n");
    printf("     cuda-memcheck --tool memcheck ./your_program\n");
    printf("     cuda-memcheck --tool racecheck ./your_program\n");
    printf("   检测: 越界访问、未初始化内存、竞争条件\n\n");

    printf("4. compute-sanitizer (CUDA 11+):\n");
    printf("   命令:\n");
    printf("     compute-sanitizer ./your_program\n");
    printf("     compute-sanitizer --tool memcheck ./your_program\n");
    printf("     compute-sanitizer --tool racecheck ./your_program\n");
    printf("     compute-sanitizer --tool initcheck ./your_program\n\n");

    printf("5. NVTX 标记 (代码中添加):\n");
    printf("   #include <nvtx3/nvToolsExt.h>\n");
    printf("   nvtxRangePush(\"MySection\");\n");
    printf("   // ... 代码 ...\n");
    printf("   nvtxRangePop();\n\n");
}

// ============================================================================
// 第五部分：常见性能问题诊断
// ============================================================================

// 低占用率示例
__global__ void lowOccupancyKernel(float *data, int n) {
    // 使用大量寄存器
    float r[64];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        for (int i = 0; i < 64; i++) {
            r[i] = data[tid] * (i + 1);
        }
        float sum = 0;
        for (int i = 0; i < 64; i++) {
            sum += r[i];
        }
        data[tid] = sum;
    }
}

// 非合并访问示例
__global__ void nonCoalescedKernel(float *data, int n, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * stride;  // 非连续访问
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 合并访问示例
__global__ void coalescedKernel(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;  // 连续访问
    }
}

void demoCommonIssues() {
    printf("=== 第五部分：常见性能问题 ===\n\n");

    const int N = 1 << 20;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 问题 1: 内存合并
    printf("1. 内存合并问题:\n");
    {
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        // 合并访问
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            coalescedKernel<<<gridSize, blockSize>>>(d_data, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float coalescedTime;
        CHECK_CUDA(cudaEventElapsedTime(&coalescedTime, start, stop));

        // 非合并访问 (stride = 32)
        gridSize = (N / 32 + blockSize - 1) / blockSize;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            nonCoalescedKernel<<<gridSize, blockSize>>>(d_data, N, 32);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float nonCoalescedTime;
        CHECK_CUDA(cudaEventElapsedTime(&nonCoalescedTime, start, stop));

        printf("   合并访问: %.3f ms\n", coalescedTime);
        printf("   非合并访问 (stride=32): %.3f ms\n", nonCoalescedTime);
        printf("   性能差异: %.2fx\n\n", nonCoalescedTime / coalescedTime);
    }

    // 问题 2: 占用率
    printf("2. 占用率问题:\n");
    {
        int maxActiveBlocks1, maxActiveBlocks2;

        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks1, coalescedKernel, 256, 0));

        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks2, lowOccupancyKernel, 256, 0));

        printf("   普通内核占用率: %d 块/SM\n", maxActiveBlocks1);
        printf("   高寄存器内核占用率: %d 块/SM\n\n", maxActiveBlocks2);
    }

    // 问题 3: 分支分歧
    printf("3. 分支分歧问题:\n");
    printf("   同一 warp 内线程走不同分支会串行执行\n");
    printf("   解决方案:\n");
    printf("   - 重组数据使相邻线程走相同分支\n");
    printf("   - 使用谓词执行代替分支\n\n");

    printf("4. 共享内存 bank 冲突:\n");
    printf("   32 个 bank，stride = 1 为最优\n");
    printf("   检测: ncu 中查看 'shared_load_transactions'\n");
    printf("   解决: 添加填充或调整访问模式\n\n");

    printf("5. 诊断清单:\n");
    printf("   □ 内存带宽利用率是否达到峰值?\n");
    printf("   □ 内存访问是否合并?\n");
    printf("   □ 占用率是否足够?\n");
    printf("   □ 是否存在分支分歧?\n");
    printf("   □ 共享内存是否有 bank 冲突?\n");
    printf("   □ 是否存在不必要的同步?\n");
    printf("   □ 数据传输是否与计算重叠?\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 第六部分：调试技巧
// ============================================================================

__global__ void debugKernel(int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 调试打印（仅限少量线程）
    #ifdef DEBUG_PRINT
    if (tid < 5) {
        printf("线程 %d: blockIdx=%d, threadIdx=%d\n",
               tid, blockIdx.x, threadIdx.x);
    }
    #endif

    if (tid < n) {
        data[tid] = tid * 2;
    }
}

void demoDebuggingTips() {
    printf("=== 第六部分：调试技巧 ===\n\n");

    printf("1. printf 调试:\n");
    printf("   - 内核中可使用 printf\n");
    printf("   - 限制打印线程数避免输出过多\n");
    printf("   - 编译时使用 -DDEBUG_PRINT 启用\n");
    printf("   示例: if (tid == 0) printf(\"val = %%f\\n\", val);\n\n");

    printf("2. assert 断言:\n");
    printf("   #include <assert.h>\n");
    printf("   assert(tid < n);  // 失败时终止\n\n");

    printf("3. cuda-gdb 调试器:\n");
    printf("   编译: nvcc -g -G program.cu\n");
    printf("   运行: cuda-gdb ./program\n");
    printf("   命令:\n");
    printf("     (cuda-gdb) break myKernel\n");
    printf("     (cuda-gdb) run\n");
    printf("     (cuda-gdb) cuda thread\n");
    printf("     (cuda-gdb) print threadIdx\n\n");

    printf("4. 渐进式调试:\n");
    printf("   - 从小规模数据开始\n");
    printf("   - 单线程块 -> 多线程块\n");
    printf("   - 与 CPU 结果对比\n");
    printf("   - 逐步增加复杂度\n\n");

    printf("5. 边界检查:\n");
    printf("   __global__ void kernel(int *data, int n) {\n");
    printf("       int tid = threadIdx.x + blockIdx.x * blockDim.x;\n");
    printf("       if (tid >= n) return;  // 关键！\n");
    printf("       data[tid] = ...;\n");
    printf("   }\n\n");

    // 演示 printf
    printf("6. 演示内核 printf:\n");
    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, 100 * sizeof(int)));

    printf("   执行带 printf 的内核...\n");
    debugKernel<<<1, 32>>>(d_data, 100);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   (使用 -DDEBUG_PRINT 编译以查看输出)\n\n");

    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 12: 性能分析与调试                            ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    demoErrorChecking();
    demoTiming();
    demoPerformanceMetrics();
    demoProfilingTools();
    demoCommonIssues();
    demoDebuggingTips();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 错误处理:\n");
    printf("   - 每个 API 调用检查返回值\n");
    printf("   - 内核后使用 cudaGetLastError()\n");
    printf("   - 同步后检查执行错误\n\n");

    printf("2. 计时方法:\n");
    printf("   - 优先使用 cudaEvent\n");
    printf("   - CPU 计时需要适当同步\n");
    printf("   - 预热后多次测量取平均\n\n");

    printf("3. 性能指标:\n");
    printf("   - 带宽利用率 = 实际/理论\n");
    printf("   - FLOPS = 操作数/时间\n");
    printf("   - 占用率 = 活跃线程/最大线程\n\n");

    printf("4. 分析工具:\n");
    printf("   - nsys: 系统级时间线分析\n");
    printf("   - ncu:  内核级详细分析\n");
    printf("   - compute-sanitizer: 内存错误检测\n\n");

    printf("5. 常见问题:\n");
    printf("   - 非合并内存访问\n");
    printf("   - 低占用率\n");
    printf("   - 分支分歧\n");
    printf("   - Bank 冲突\n\n");

    printf("推荐的分析流程:\n");
    printf("┌────────────────────────────────────────────────────────┐\n");
    printf("│ 1. 功能正确性 (cuda-memcheck/compute-sanitizer)       │\n");
    printf("│        ↓                                              │\n");
    printf("│ 2. 系统级分析 (nsys) - 找到热点内核                   │\n");
    printf("│        ↓                                              │\n");
    printf("│ 3. 内核级分析 (ncu) - 分析性能瓶颈                    │\n");
    printf("│        ↓                                              │\n");
    printf("│ 4. 优化并重复测量                                     │\n");
    printf("└────────────────────────────────────────────────────────┘\n\n");

    return 0;
}
