/**
 * =============================================================================
 * CUDA 教程 29: 调试与错误处理最佳实践
 * =============================================================================
 *
 * 学习目标：
 * 1. 掌握 CUDA 错误检查的完整方法
 * 2. 学会使用各种调试工具
 * 3. 理解常见 CUDA 错误及解决方案
 * 4. 掌握内存调试和竞态检测技术
 *
 * 关键概念：
 * - 错误检查宏
 * - cuda-memcheck / compute-sanitizer
 * - printf 调试
 * - Nsight 调试器
 *
 * 编译命令 (调试版):
 *   nvcc -g -G -lineinfo 29_debugging_best_practices.cu -o 29_debugging
 *
 * 运行内存检查:
 *   compute-sanitizer ./29_debugging
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

// ============================================================================
// 第一部分：错误检查宏
// ============================================================================

// 基本错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// 带返回值的错误检查
#define CHECK_CUDA_RETURN(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return err; \
    } \
}

// 内核启动后的错误检查
#define CHECK_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "内核启动错误 %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "内核执行错误 %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// 调试模式下的断言
#ifdef DEBUG
#define CUDA_ASSERT(cond) { \
    if (!(cond)) { \
        fprintf(stderr, "CUDA 断言失败 %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(EXIT_FAILURE); \
    } \
}
#else
#define CUDA_ASSERT(cond) ((void)0)
#endif

// 详细错误信息
void printDetailedError(cudaError_t err, const char *file, int line) {
    fprintf(stderr, "═══════════════════════════════════════════════════════════\n");
    fprintf(stderr, "CUDA 错误详情:\n");
    fprintf(stderr, "  文件: %s\n", file);
    fprintf(stderr, "  行号: %d\n", line);
    fprintf(stderr, "  错误码: %d\n", err);
    fprintf(stderr, "  错误名: %s\n", cudaGetErrorName(err));
    fprintf(stderr, "  错误描述: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "═══════════════════════════════════════════════════════════\n");
}

#define CHECK_CUDA_DETAILED(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printDetailedError(err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void demoErrorChecking() {
    printf("=== 第一部分：错误检查宏 ===\n\n");

    printf("基本错误检查:\n");
    printf("  #define CHECK_CUDA(call) { \\\n");
    printf("      cudaError_t err = call; \\\n");
    printf("      if (err != cudaSuccess) { \\\n");
    printf("          fprintf(stderr, \"CUDA 错误 %%s:%%d: %%s\\n\", \\\n");
    printf("                  __FILE__, __LINE__, cudaGetErrorString(err)); \\\n");
    printf("          exit(EXIT_FAILURE); \\\n");
    printf("      } \\\n");
    printf("  }\n\n");

    printf("使用示例:\n");
    printf("  CHECK_CUDA(cudaMalloc(&ptr, size));\n");
    printf("  CHECK_CUDA(cudaMemcpy(dst, src, size, kind));\n\n");

    printf("内核启动检查:\n");
    printf("  myKernel<<<grid, block>>>(args);\n");
    printf("  CHECK_CUDA(cudaGetLastError());      // 检查启动错误\n");
    printf("  CHECK_CUDA(cudaDeviceSynchronize()); // 检查执行错误\n\n");

    printf("错误类型:\n");
    printf("  ┌─────────────────────────────┬────────────────────────────────┐\n");
    printf("  │ 错误                        │ 说明                           │\n");
    printf("  ├─────────────────────────────┼────────────────────────────────┤\n");
    printf("  │ cudaErrorInvalidValue       │ 无效参数                       │\n");
    printf("  │ cudaErrorMemoryAllocation   │ 内存分配失败                   │\n");
    printf("  │ cudaErrorInvalidDevicePtr   │ 无效设备指针                   │\n");
    printf("  │ cudaErrorInvalidConfiguration│ 无效启动配置                  │\n");
    printf("  │ cudaErrorIllegalAddress     │ 非法内存访问                   │\n");
    printf("  │ cudaErrorLaunchTimeout      │ 内核执行超时                   │\n");
    printf("  └─────────────────────────────┴────────────────────────────────┘\n\n");
}

// ============================================================================
// 第二部分：内核中的 printf 调试
// ============================================================================

__global__ void debugKernelPrintf(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 只打印前几个线程的信息
    if (tid < 5) {
        printf("[Thread %d] Block=%d, Thread=%d, Value=%.2f\n",
               tid, blockIdx.x, threadIdx.x, data[tid]);
    }

    // 条件打印
    if (tid < n && data[tid] < 0) {
        printf("警告: data[%d] = %.2f 是负数!\n", tid, data[tid]);
    }
}

__global__ void debugKernelDetailed(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    // Warp 级别调试
    if (laneId == 0 && warpId < 3) {
        printf("Warp %d (线程 %d-%d): 活跃\n",
               warpId, warpId * warpSize, (warpId + 1) * warpSize - 1);
    }

    // 共享内存调试
    __shared__ float smem[256];
    if (tid < n && threadIdx.x < 256) {
        smem[threadIdx.x] = data[tid];
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Block 0 共享内存前 4 个值: %.2f, %.2f, %.2f, %.2f\n",
               smem[0], smem[1], smem[2], smem[3]);
    }
}

void demoKernelPrintf() {
    printf("=== 第二部分：内核 printf 调试 ===\n\n");

    printf("printf 注意事项:\n");
    printf("  1. 输出缓冲有限 (默认 1MB)\n");
    printf("  2. 调用 cudaDeviceSynchronize() 刷新缓冲\n");
    printf("  3. 性能影响大，仅用于调试\n");
    printf("  4. 限制打印线程数\n\n");

    printf("增加 printf 缓冲:\n");
    printf("  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size);\n\n");

    // 设置较大的 printf 缓冲
    size_t printfSize = 10 * 1024 * 1024;  // 10 MB
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfSize));

    const int N = 256;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i - 5);  // 有些负数
    }
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    printf("运行调试内核:\n");
    printf("─────────────────────────────────────\n");
    debugKernelPrintf<<<1, 256>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());  // 刷新 printf 缓冲
    printf("─────────────────────────────────────\n\n");

    printf("详细调试输出:\n");
    printf("─────────────────────────────────────\n");
    debugKernelDetailed<<<2, 128>>>(d_data, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("─────────────────────────────────────\n\n");

    free(h_data);
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 第三部分：常见错误示例与修复
// ============================================================================

// 错误示例 1: 越界访问
__global__ void buggyKernelOOB(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // BUG: 没有边界检查
    // data[tid] = data[tid] * 2.0f;

    // 修复: 添加边界检查
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}

// 错误示例 2: 未初始化共享内存
__global__ void buggyKernelUninitialized(float *output, int n) {
    __shared__ float smem[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // BUG: 部分线程没有初始化共享内存
    // if (tid < n && threadIdx.x < 128) {
    //     smem[threadIdx.x] = 1.0f;
    // }

    // 修复: 所有需要的位置都要初始化
    if (threadIdx.x < 256) {
        smem[threadIdx.x] = (tid < n) ? 1.0f : 0.0f;
    }
    __syncthreads();

    if (tid < n) {
        output[tid] = smem[threadIdx.x];
    }
}

// 错误示例 3: 竞态条件
__global__ void buggyKernelRace(int *counter) {
    // BUG: 非原子更新
    // (*counter)++;

    // 修复: 使用原子操作
    atomicAdd(counter, 1);
}

// 错误示例 4: 死锁
__global__ void buggyKernelDeadlock(int *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // BUG: 条件 syncthreads 导致死锁
    // if (tid < n / 2) {
    //     __syncthreads();  // 只有部分线程到达
    // }

    // 修复: 所有线程必须到达 syncthreads
    __syncthreads();
    if (tid < n / 2) {
        data[tid] = data[tid] + 1;
    }
}

// 错误示例 5: Bank Conflict (性能问题)
__global__ void buggyKernelBankConflict(float *output, float *input, int n) {
    __shared__ float smem[32][32];  // BUG: 列访问有 bank conflict

    int tid = threadIdx.x;
    int row = threadIdx.y;

    // Bank conflict 访问
    // smem[row][tid] = input[row * 32 + tid];
    // output[tid * 32 + row] = smem[tid][row];  // 列访问

    // 修复: 使用 padding
    __shared__ float smem_fixed[32][33];  // +1 避免 bank conflict
    smem_fixed[row][tid] = input[row * 32 + tid];
    __syncthreads();
    output[tid * 32 + row] = smem_fixed[tid][row];
}

void demoCommonBugs() {
    printf("=== 第三部分：常见错误与修复 ===\n\n");

    printf("1. 越界访问:\n");
    printf("   错误: 没有边界检查\n");
    printf("   修复: if (tid < n) { ... }\n\n");

    printf("2. 未初始化共享内存:\n");
    printf("   错误: 部分线程没有写入共享内存\n");
    printf("   修复: 确保所有位置都被初始化\n\n");

    printf("3. 竞态条件:\n");
    printf("   错误: 多线程非原子更新同一变量\n");
    printf("   修复: 使用 atomicAdd 等原子操作\n\n");

    printf("4. __syncthreads 死锁:\n");
    printf("   错误: 条件分支中使用 __syncthreads\n");
    printf("   修复: 确保所有线程到达同步点\n\n");

    printf("5. Bank Conflict:\n");
    printf("   错误: 共享内存列访问\n");
    printf("   修复: 使用 padding (如 [32][33])\n\n");

    // 演示竞态检测
    printf("竞态条件测试:\n");
    int *d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

    // 使用原子操作
    buggyKernelRace<<<100, 256>>>(d_counter);
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_counter;
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  原子计数结果: %d (期望: %d)\n\n", h_counter, 100 * 256);

    CHECK_CUDA(cudaFree(d_counter));
}

// ============================================================================
// 第四部分：compute-sanitizer 工具
// ============================================================================

// 用于测试内存错误的内核
__global__ void memoryErrorKernel(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 这会导致越界访问 (用 compute-sanitizer 检测)
    #ifdef ENABLE_MEMORY_ERROR
    data[tid + n] = 0.0f;  // 故意越界
    #else
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
    #endif
}

// 未对齐访问测试
__global__ void misalignedAccessKernel(char *data) {
    // 未对齐的 float 访问
    #ifdef ENABLE_MISALIGNED
    float *fptr = (float*)(data + 1);  // 未对齐
    *fptr = 1.0f;
    #else
    float *fptr = (float*)(data);  // 对齐
    *fptr = 1.0f;
    #endif
}

void demoComputeSanitizer() {
    printf("=== 第四部分：compute-sanitizer 工具 ===\n\n");

    printf("compute-sanitizer 是 CUDA 内存调试工具:\n\n");

    printf("基本用法:\n");
    printf("  compute-sanitizer ./my_cuda_program\n\n");

    printf("检查类型:\n");
    printf("  ┌────────────────────────┬────────────────────────────────────┐\n");
    printf("  │ 工具                   │ 检查内容                           │\n");
    printf("  ├────────────────────────┼────────────────────────────────────┤\n");
    printf("  │ --tool memcheck        │ 内存访问错误 (默认)                │\n");
    printf("  │ --tool racecheck       │ 共享内存竞态条件                   │\n");
    printf("  │ --tool initcheck       │ 未初始化内存访问                   │\n");
    printf("  │ --tool synccheck       │ 同步原语错误                       │\n");
    printf("  └────────────────────────┴────────────────────────────────────┘\n\n");

    printf("常用选项:\n");
    printf("  --leak-check full       # 检查内存泄漏\n");
    printf("  --show-backtrace yes    # 显示调用栈\n");
    printf("  --print-level info      # 详细输出\n");
    printf("  --error-exitcode 1      # 有错误时返回非零\n\n");

    printf("示例输出:\n");
    printf("  ========= Invalid __global__ read of size 4\n");
    printf("  =========     at 0x00000148 in myKernel(float*, int)\n");
    printf("  =========     by thread (256,0,0) in block (0,0,0)\n");
    printf("  =========     Address 0x7f1234567890 is out of bounds\n\n");

    printf("集成到 CI/CD:\n");
    printf("  #!/bin/bash\n");
    printf("  compute-sanitizer --error-exitcode 1 ./test_program\n");
    printf("  if [ $? -ne 0 ]; then\n");
    printf("      echo \"内存错误检测到!\"\n");
    printf("      exit 1\n");
    printf("  fi\n\n");
}

// ============================================================================
// 第五部分：Nsight 调试器使用
// ============================================================================

void demoNsightDebugger() {
    printf("=== 第五部分：Nsight 调试器 ===\n\n");

    printf("编译调试版本:\n");
    printf("  nvcc -g -G -lineinfo program.cu -o program_debug\n\n");

    printf("编译选项:\n");
    printf("  -g        : 主机代码调试信息\n");
    printf("  -G        : 设备代码调试信息 (禁用优化)\n");
    printf("  -lineinfo : 保留行号信息 (可与优化共用)\n\n");

    printf("Nsight Visual Studio Edition:\n");
    printf("  1. 设置 CUDA 断点\n");
    printf("  2. 单步执行内核\n");
    printf("  3. 查看线程状态\n");
    printf("  4. 检查共享内存/寄存器\n\n");

    printf("Nsight Eclipse Edition / VSCode:\n");
    printf("  1. 启动 cuda-gdb\n");
    printf("  2. 设置条件断点\n");
    printf("  3. 选择特定线程调试\n\n");

    printf("cuda-gdb 命令:\n");
    printf("  ┌─────────────────────────┬────────────────────────────────────┐\n");
    printf("  │ 命令                    │ 说明                               │\n");
    printf("  ├─────────────────────────┼────────────────────────────────────┤\n");
    printf("  │ cuda-gdb ./program      │ 启动调试器                         │\n");
    printf("  │ break myKernel          │ 在内核入口设断点                   │\n");
    printf("  │ cuda thread (1,0,0)     │ 切换到指定线程                     │\n");
    printf("  │ cuda block (0,0,0)      │ 切换到指定块                       │\n");
    printf("  │ info cuda threads       │ 显示所有线程                       │\n");
    printf("  │ print threadIdx         │ 打印线程索引                       │\n");
    printf("  │ print @shared(var)      │ 打印共享内存变量                   │\n");
    printf("  │ x/10f $ptr              │ 检查内存                           │\n");
    printf("  └─────────────────────────┴────────────────────────────────────┘\n\n");

    printf("条件断点示例:\n");
    printf("  break myKernel if threadIdx.x == 0 && blockIdx.x == 0\n\n");
}

// ============================================================================
// 第六部分：性能调试
// ============================================================================

__global__ void performanceTestKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // 模拟一些计算
        float val = input[tid];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) + cosf(val);
        }
        output[tid] = val;
    }
}

void demoPerformanceDebugging() {
    printf("=== 第六部分：性能调试 ===\n\n");

    printf("Nsight Systems (系统级分析):\n");
    printf("  nsys profile ./program\n");
    printf("  nsys profile -o report ./program\n\n");

    printf("Nsight Compute (内核级分析):\n");
    printf("  ncu ./program\n");
    printf("  ncu --set full ./program\n");
    printf("  ncu --section MemoryWorkloadAnalysis ./program\n\n");

    printf("关键性能指标:\n");
    printf("  ┌──────────────────────┬──────────────────────────────────────┐\n");
    printf("  │ 指标                 │ 理想值                               │\n");
    printf("  ├──────────────────────┼──────────────────────────────────────┤\n");
    printf("  │ SM 占用率           │ > 50%%                                │\n");
    printf("  │ 内存吞吐效率         │ > 60%% 峰值                          │\n");
    printf("  │ L1 缓存命中率       │ 取决于访问模式                       │\n");
    printf("  │ 分支效率             │ > 90%%                                │\n");
    printf("  │ 活跃 warp 比例      │ > 75%%                                │\n");
    printf("  └──────────────────────┴──────────────────────────────────────┘\n\n");

    // 使用 CUDA Events 计时
    const int N = 1 << 20;
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));

    float *h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = (float)i;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 计时
    CHECK_CUDA(cudaEventRecord(start));
    performanceTestKernel<<<(N+255)/256, 256>>>(d_output, d_input, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    printf("CUDA Events 计时示例:\n");
    printf("  内核执行时间: %.3f ms\n", elapsed);
    printf("  处理元素: %d\n", N);
    printf("  吞吐量: %.2f M元素/秒\n\n", N / elapsed / 1000.0f);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第七部分：断言与运行时检查
// ============================================================================

// 设备端断言
__global__ void assertKernel(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // 设备端断言 (调试模式)
        assert(data[tid] >= 0.0f && "数据必须非负!");

        data[tid] = sqrtf(data[tid]);
    }
}

// 自定义错误检查
__device__ bool g_errorFlag = false;
__device__ int g_errorThread = -1;

__global__ void customErrorCheckKernel(float *data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        if (data[tid] < 0.0f) {
            // 记录错误但继续执行
            g_errorFlag = true;
            atomicMin(&g_errorThread, tid);
        } else {
            data[tid] = sqrtf(data[tid]);
        }
    }
}

// 边界检查包装
template<typename T>
__device__ T safeLoad(const T *data, int idx, int n, T defaultVal = T(0)) {
    if (idx >= 0 && idx < n) {
        return data[idx];
    }
    printf("警告: 越界访问 idx=%d, n=%d\n", idx, n);
    return defaultVal;
}

template<typename T>
__device__ void safeStore(T *data, int idx, int n, T val) {
    if (idx >= 0 && idx < n) {
        data[idx] = val;
    } else {
        printf("警告: 越界写入 idx=%d, n=%d\n", idx, n);
    }
}

void demoAssertions() {
    printf("=== 第七部分：断言与运行时检查 ===\n\n");

    printf("设备端断言:\n");
    printf("  #include <assert.h>\n");
    printf("  assert(condition && \"错误消息\");\n\n");

    printf("断言注意事项:\n");
    printf("  1. 需要 -G 编译选项才能启用\n");
    printf("  2. 断言失败会停止内核执行\n");
    printf("  3. 生产代码应禁用断言\n\n");

    printf("安全访问包装:\n");
    printf("  template<typename T>\n");
    printf("  __device__ T safeLoad(const T *data, int idx, int n) {\n");
    printf("      if (idx >= 0 && idx < n) return data[idx];\n");
    printf("      return T(0);  // 默认值\n");
    printf("  }\n\n");

    printf("自定义错误标志:\n");
    printf("  __device__ bool g_errorFlag = false;\n");
    printf("  // 内核中设置错误标志\n");
    printf("  // 主机端检查错误标志\n\n");
}

// ============================================================================
// 第八部分：内存调试
// ============================================================================

void demoMemoryDebugging() {
    printf("=== 第八部分：内存调试 ===\n\n");

    printf("内存泄漏检查:\n");
    printf("  compute-sanitizer --leak-check full ./program\n\n");

    printf("内存使用追踪:\n");

    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));

    printf("  当前 GPU 内存状态:\n");
    printf("    总内存: %.2f GB\n", totalMem / (1024.0 * 1024.0 * 1024.0));
    printf("    可用: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));
    printf("    已用: %.2f GB\n\n", (totalMem - freeMem) / (1024.0 * 1024.0 * 1024.0));

    printf("内存分配包装 (调试版):\n");
    printf("  void* debugMalloc(size_t size, const char* file, int line) {\n");
    printf("      void *ptr;\n");
    printf("      cudaError_t err = cudaMalloc(&ptr, size);\n");
    printf("      if (err == cudaSuccess) {\n");
    printf("          printf(\"分配 %%zu 字节 @ %%s:%%d\\n\", size, file, line);\n");
    printf("          // 记录分配信息用于泄漏检测\n");
    printf("      }\n");
    printf("      return ptr;\n");
    printf("  }\n");
    printf("  #define CUDA_MALLOC(ptr, size) ptr = debugMalloc(size, __FILE__, __LINE__)\n\n");

    printf("Unified Memory 调试:\n");
    printf("  - 设置 CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 检查访问模式\n");
    printf("  - 使用 cudaMemPrefetchAsync 优化访问\n");
    printf("  - compute-sanitizer 可检测无效 UM 访问\n\n");
}

// ============================================================================
// 第九部分：调试清单
// ============================================================================

void demoDebuggingChecklist() {
    printf("=== 第九部分：调试清单 ===\n\n");

    printf("□ 编译检查:\n");
    printf("  □ 使用 -Werror 将警告视为错误\n");
    printf("  □ 检查计算能力是否匹配\n");
    printf("  □ 调试版使用 -g -G\n\n");

    printf("□ 运行时检查:\n");
    printf("  □ 检查所有 CUDA API 返回值\n");
    printf("  □ 检查内核启动参数\n");
    printf("  □ 使用 cudaDeviceSynchronize 捕获异步错误\n\n");

    printf("□ 内存检查:\n");
    printf("  □ 运行 compute-sanitizer --tool memcheck\n");
    printf("  □ 检查内存泄漏\n");
    printf("  □ 验证指针有效性\n");
    printf("  □ 检查缓冲区大小\n\n");

    printf("□ 逻辑检查:\n");
    printf("  □ 边界条件 (tid < n)\n");
    printf("  □ __syncthreads 位置\n");
    printf("  □ 原子操作使用\n");
    printf("  □ 共享内存初始化\n\n");

    printf("□ 性能检查:\n");
    printf("  □ 内存合并访问\n");
    printf("  □ Bank conflict\n");
    printf("  □ 分支发散\n");
    printf("  □ 占用率\n\n");

    printf("□ 工具使用:\n");
    printf("  □ printf 快速定位\n");
    printf("  □ cuda-gdb 详细调试\n");
    printf("  □ Nsight Systems 时间线\n");
    printf("  □ Nsight Compute 内核分析\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 29: 调试与错误处理最佳实践                           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    demoErrorChecking();
    demoKernelPrintf();
    demoCommonBugs();
    demoComputeSanitizer();
    demoNsightDebugger();
    demoPerformanceDebugging();
    demoAssertions();
    demoMemoryDebugging();
    demoDebuggingChecklist();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 错误检查:\n");
    printf("   - 始终检查 API 返回值\n");
    printf("   - 使用 CHECK_CUDA 宏\n");
    printf("   - 同步后检查内核错误\n\n");

    printf("2. 调试工具:\n");
    printf("   - printf: 快速定位\n");
    printf("   - compute-sanitizer: 内存错误\n");
    printf("   - cuda-gdb: 交互式调试\n");
    printf("   - Nsight: 可视化分析\n\n");

    printf("3. 常见错误:\n");
    printf("   - 越界访问\n");
    printf("   - 竞态条件\n");
    printf("   - 未初始化内存\n");
    printf("   - syncthreads 死锁\n\n");

    printf("4. 最佳实践:\n");
    printf("   - 调试版本单独编译\n");
    printf("   - CI 集成内存检查\n");
    printf("   - 保持代码简洁可调试\n\n");

    printf("5. 调试流程:\n");
    printf("   定位问题 → 最小复现 → 工具分析 → 修复验证\n\n");

    return 0;
}
