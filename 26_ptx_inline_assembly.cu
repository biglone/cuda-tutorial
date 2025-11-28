/**
 * =============================================================================
 * CUDA 教程 26: PTX 汇编与 Inline PTX
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 PTX (Parallel Thread Execution) 汇编语言
 * 2. 学会使用 Inline PTX 优化关键代码
 * 3. 掌握 PTX 寄存器和指令
 * 4. 了解 PTX 与 SASS 的关系
 *
 * 关键概念：
 * - PTX 中间表示
 * - Inline PTX 语法 (asm volatile)
 * - 特殊寄存器访问
 * - 原子操作的 PTX 实现
 *
 * 编译命令：
 *   nvcc 26_ptx_inline_assembly.cu -o 26_ptx_inline_assembly
 *
 * 查看 PTX:
 *   nvcc -ptx 26_ptx_inline_assembly.cu
 *
 * 查看 SASS:
 *   cuobjdump -sass 26_ptx_inline_assembly
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

// ============================================================================
// 第一部分：PTX 概述
// ============================================================================

void demoPTXOverview() {
    printf("=== 第一部分：PTX 概述 ===\n\n");

    printf("编译流程:\n");
    printf("  ┌─────────────────┐\n");
    printf("  │   CUDA C/C++    │\n");
    printf("  └────────┬────────┘\n");
    printf("           │ nvcc 前端\n");
    printf("           ▼\n");
    printf("  ┌─────────────────┐\n");
    printf("  │      PTX        │ ← 中间表示 (可移植)\n");
    printf("  │ (虚拟指令集)    │\n");
    printf("  └────────┬────────┘\n");
    printf("           │ ptxas\n");
    printf("           ▼\n");
    printf("  ┌─────────────────┐\n");
    printf("  │      SASS       │ ← 实际机器码 (GPU 特定)\n");
    printf("  │ (原生指令集)    │\n");
    printf("  └─────────────────┘\n\n");

    printf("PTX 特点:\n");
    printf("  - 类似汇编的低级语言\n");
    printf("  - GPU 架构无关 (向前兼容)\n");
    printf("  - 支持虚拟寄存器 (无限)\n");
    printf("  - 单指令多线程 (SIMT) 模型\n\n");

    printf("PTX 数据类型:\n");
    printf("  ┌──────────┬──────────────────────────────────┐\n");
    printf("  │ 类型     │ 说明                             │\n");
    printf("  ├──────────┼──────────────────────────────────┤\n");
    printf("  │ .b8-b64  │ 无类型位串                       │\n");
    printf("  │ .u8-u64  │ 无符号整数                       │\n");
    printf("  │ .s8-s64  │ 有符号整数                       │\n");
    printf("  │ .f16     │ 半精度浮点                       │\n");
    printf("  │ .f32     │ 单精度浮点                       │\n");
    printf("  │ .f64     │ 双精度浮点                       │\n");
    printf("  │ .pred    │ 谓词 (布尔)                      │\n");
    printf("  └──────────┴──────────────────────────────────┘\n\n");

    printf("PTX 寄存器类型:\n");
    printf("  - %%r: 32 位整数寄存器\n");
    printf("  - %%rd: 64 位整数寄存器\n");
    printf("  - %%f: 32 位浮点寄存器\n");
    printf("  - %%fd: 64 位浮点寄存器\n");
    printf("  - %%p: 谓词寄存器\n\n");
}

// ============================================================================
// 第二部分：Inline PTX 基础
// ============================================================================

// 基本 Inline PTX 语法
__device__ __forceinline__ int ptx_add(int a, int b) {
    int result;
    // asm volatile ("指令" : "输出约束"(变量) : "输入约束"(变量));
    asm volatile ("add.s32 %0, %1, %2;"
                  : "=r"(result)      // 输出: r = 32位寄存器
                  : "r"(a), "r"(b));  // 输入
    return result;
}

// 浮点加法
__device__ __forceinline__ float ptx_fadd(float a, float b) {
    float result;
    asm volatile ("add.f32 %0, %1, %2;"
                  : "=f"(result)      // f = 浮点寄存器
                  : "f"(a), "f"(b));
    return result;
}

// 融合乘加 (FMA)
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float result;
    // result = a * b + c (单精度)
    asm volatile ("fma.rn.f32 %0, %1, %2, %3;"
                  : "=f"(result)
                  : "f"(a), "f"(b), "f"(c));
    return result;
}

// 测试基本 PTX 操作
__global__ void testBasicPTX(float *output, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 使用 PTX FMA
        output[tid] = ptx_fma(a[tid], b[tid], 1.0f);
    }
}

void demoBasicInlinePTX() {
    printf("=== 第二部分：Inline PTX 基础 ===\n\n");

    printf("Inline PTX 语法:\n");
    printf("  asm volatile (\"指令模板\"\n");
    printf("                : \"输出约束\"(变量)     // 输出操作数\n");
    printf("                : \"输入约束\"(变量)     // 输入操作数\n");
    printf("                : \"clobber\");          // 副作用 (可选)\n\n");

    printf("约束符号:\n");
    printf("  ┌───────┬──────────────────────────────────┐\n");
    printf("  │ 符号  │ 含义                             │\n");
    printf("  ├───────┼──────────────────────────────────┤\n");
    printf("  │ r     │ 32位整数寄存器                   │\n");
    printf("  │ l     │ 64位整数寄存器                   │\n");
    printf("  │ f     │ 32位浮点寄存器                   │\n");
    printf("  │ d     │ 64位浮点寄存器                   │\n");
    printf("  │ n     │ 立即数                           │\n");
    printf("  │ =     │ 只写输出                         │\n");
    printf("  │ +     │ 读写输出                         │\n");
    printf("  └───────┴──────────────────────────────────┘\n\n");

    printf("示例代码:\n");
    printf("  // 整数加法\n");
    printf("  __device__ int ptx_add(int a, int b) {\n");
    printf("      int result;\n");
    printf("      asm (\"add.s32 %%0, %%1, %%2;\" \n");
    printf("           : \"=r\"(result) : \"r\"(a), \"r\"(b));\n");
    printf("      return result;\n");
    printf("  }\n\n");

    // 运行测试
    const int N = 1024;
    float *d_a, *d_b, *d_out;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = 2.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    testBasicPTX<<<(N+255)/256, 256>>>(d_out, d_a, d_b, N);

    float *h_out = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("验证 FMA (a * b + 1.0):\n");
    printf("  h_a[5] = %.1f, h_b[5] = %.1f\n", h_a[5], h_b[5]);
    printf("  结果 = %.1f (期望 = %.1f)\n\n", h_out[5], h_a[5] * h_b[5] + 1.0f);

    free(h_a);
    free(h_b);
    free(h_out);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

// ============================================================================
// 第三部分：特殊寄存器访问
// ============================================================================

// 读取线程 ID (更直接)
__device__ __forceinline__ unsigned int ptx_laneid() {
    unsigned int laneid;
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

// 读取 warp ID
__device__ __forceinline__ unsigned int ptx_warpid() {
    unsigned int warpid;
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}

// 读取 SM ID
__device__ __forceinline__ unsigned int ptx_smid() {
    unsigned int smid;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// 读取时钟计数器
__device__ __forceinline__ unsigned long long ptx_clock64() {
    unsigned long long clock;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(clock));
    return clock;
}

// 读取全局计时器
__device__ __forceinline__ unsigned long long ptx_globaltimer() {
    unsigned long long timer;
    asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(timer));
    return timer;
}

__global__ void testSpecialRegisters(int *laneid, int *warpid, int *smid, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        laneid[tid] = ptx_laneid();
        warpid[tid] = ptx_warpid();
        smid[tid] = ptx_smid();
    }
}

void demoSpecialRegisters() {
    printf("=== 第三部分：特殊寄存器访问 ===\n\n");

    printf("PTX 特殊寄存器:\n");
    printf("  ┌─────────────────┬────────────────────────────────┐\n");
    printf("  │ 寄存器          │ 说明                           │\n");
    printf("  ├─────────────────┼────────────────────────────────┤\n");
    printf("  │ %%tid.x/y/z     │ 线程在块内的索引               │\n");
    printf("  │ %%ntid.x/y/z    │ 块的维度                       │\n");
    printf("  │ %%ctaid.x/y/z   │ 块在网格内的索引               │\n");
    printf("  │ %%nctaid.x/y/z  │ 网格的维度                     │\n");
    printf("  │ %%laneid        │ 线程在 warp 内的位置 (0-31)    │\n");
    printf("  │ %%warpid        │ warp 在块内的 ID               │\n");
    printf("  │ %%smid          │ SM 的 ID                       │\n");
    printf("  │ %%clock         │ 32位时钟计数器                 │\n");
    printf("  │ %%clock64       │ 64位时钟计数器                 │\n");
    printf("  │ %%globaltimer   │ 全局纳秒计时器                 │\n");
    printf("  └─────────────────┴────────────────────────────────┘\n\n");

    const int N = 256;  // 8 个 warp
    int *d_laneid, *d_warpid, *d_smid;
    CHECK_CUDA(cudaMalloc(&d_laneid, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_warpid, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_smid, N * sizeof(int)));

    testSpecialRegisters<<<1, N>>>(d_laneid, d_warpid, d_smid, N);

    int *h_laneid = (int*)malloc(N * sizeof(int));
    int *h_warpid = (int*)malloc(N * sizeof(int));
    int *h_smid = (int*)malloc(N * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_laneid, d_laneid, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_warpid, d_warpid, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_smid, d_smid, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("示例输出 (前 64 个线程):\n");
    printf("  线程 │ laneid │ warpid │ smid\n");
    printf("  ─────┼────────┼────────┼─────\n");
    for (int i = 0; i < 64; i += 16) {
        printf("  %4d │   %4d │   %4d │ %4d\n",
               i, h_laneid[i], h_warpid[i], h_smid[i]);
    }
    printf("\n");

    free(h_laneid);
    free(h_warpid);
    free(h_smid);
    CHECK_CUDA(cudaFree(d_laneid));
    CHECK_CUDA(cudaFree(d_warpid));
    CHECK_CUDA(cudaFree(d_smid));
}

// ============================================================================
// 第四部分：内存操作 PTX
// ============================================================================

// 使用缓存修饰符的加载
__device__ __forceinline__ float ptx_load_ca(const float *addr) {
    float result;
    // .ca = cache all (缓存到所有级别)
    asm volatile ("ld.global.ca.f32 %0, [%1];"
                  : "=f"(result)
                  : "l"(addr));
    return result;
}

// 不缓存的加载 (流式数据)
__device__ __forceinline__ float ptx_load_cs(const float *addr) {
    float result;
    // .cs = cache streaming (流式，可能不缓存)
    asm volatile ("ld.global.cs.f32 %0, [%1];"
                  : "=f"(result)
                  : "l"(addr));
    return result;
}

// 只读数据缓存加载 (通过纹理缓存)
__device__ __forceinline__ float ptx_load_ldg(const float *addr) {
    float result;
    // .nc = non-coherent (只读缓存路径)
    asm volatile ("ld.global.nc.f32 %0, [%1];"
                  : "=f"(result)
                  : "l"(addr));
    return result;
}

// 向量加载 (float4)
__device__ __forceinline__ float4 ptx_load_vec4(const float4 *addr) {
    float4 result;
    asm volatile ("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                  : "=f"(result.x), "=f"(result.y),
                    "=f"(result.z), "=f"(result.w)
                  : "l"(addr));
    return result;
}

// volatile 存储 (绕过缓存)
__device__ __forceinline__ void ptx_store_wt(float *addr, float val) {
    // .wt = write-through
    asm volatile ("st.global.wt.f32 [%0], %1;"
                  :: "l"(addr), "f"(val));
}

__global__ void testMemoryPTX(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 使用只读缓存加载
        float val = ptx_load_ldg(&input[tid]);
        output[tid] = val * 2.0f;
    }
}

void demoMemoryPTX() {
    printf("=== 第四部分：内存操作 PTX ===\n\n");

    printf("加载指令缓存修饰符:\n");
    printf("  ┌──────┬───────────────────────────────────────────┐\n");
    printf("  │ 修饰 │ 说明                                      │\n");
    printf("  ├──────┼───────────────────────────────────────────┤\n");
    printf("  │ .ca  │ Cache All - 缓存到所有级别 (默认)         │\n");
    printf("  │ .cg  │ Cache Global - 只缓存到 L2                │\n");
    printf("  │ .cs  │ Cache Streaming - 流式，可能绕过缓存      │\n");
    printf("  │ .lu  │ Last Use - 数据最后一次使用后可逐出       │\n");
    printf("  │ .cv  │ Cache Volatile - 不缓存，每次都从内存读   │\n");
    printf("  │ .nc  │ Non-Coherent - 只读缓存路径 (如 __ldg)    │\n");
    printf("  └──────┴───────────────────────────────────────────┘\n\n");

    printf("存储指令修饰符:\n");
    printf("  ┌──────┬───────────────────────────────────────────┐\n");
    printf("  │ 修饰 │ 说明                                      │\n");
    printf("  ├──────┼───────────────────────────────────────────┤\n");
    printf("  │ .wb  │ Write-Back - 先写缓存 (默认)              │\n");
    printf("  │ .cg  │ Cache Global - 绕过 L1，写到 L2           │\n");
    printf("  │ .cs  │ Cache Streaming - 流式写入                │\n");
    printf("  │ .wt  │ Write-Through - 同时写缓存和内存          │\n");
    printf("  └──────┴───────────────────────────────────────────┘\n\n");

    printf("示例: 使用 __ldg 等效的 PTX\n");
    printf("  // C++ 内置函数\n");
    printf("  float val = __ldg(&ptr[i]);\n\n");
    printf("  // 等效 PTX\n");
    printf("  asm (\"ld.global.nc.f32 %%0, [%%1];\" \n");
    printf("       : \"=f\"(val) : \"l\"(&ptr[i]));\n\n");

    // 测试
    const int N = 1024;
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    testMemoryPTX<<<(N+255)/256, 256>>>(d_out, d_in, N);

    float *h_out = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("验证 __ldg 加载:\n");
    printf("  input[10] = %.1f → output[10] = %.1f (期望 = %.1f)\n\n",
           h_in[10], h_out[10], h_in[10] * 2.0f);

    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

// ============================================================================
// 第五部分：原子操作 PTX
// ============================================================================

// PTX 原子加法
__device__ __forceinline__ int ptx_atomicAdd(int *addr, int val) {
    int old;
    asm volatile ("atom.global.add.s32 %0, [%1], %2;"
                  : "=r"(old)
                  : "l"(addr), "r"(val)
                  : "memory");
    return old;
}

// PTX 原子 CAS (Compare-And-Swap)
__device__ __forceinline__ int ptx_atomicCAS(int *addr, int compare, int val) {
    int old;
    asm volatile ("atom.global.cas.b32 %0, [%1], %2, %3;"
                  : "=r"(old)
                  : "l"(addr), "r"(compare), "r"(val)
                  : "memory");
    return old;
}

// 使用 CAS 实现原子最大值 (float)
__device__ __forceinline__ float ptx_atomicMaxFloat(float *addr, float val) {
    int *addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;

    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (old_val >= val) break;

        int new_val = __float_as_int(val);
        asm volatile ("atom.global.cas.b32 %0, [%1], %2, %3;"
                      : "=r"(old)
                      : "l"(addr_as_int), "r"(assumed), "r"(new_val)
                      : "memory");
    } while (assumed != old);

    return __int_as_float(old);
}

// 原子操作作用域 (GPU 级别 vs 系统级别)
__device__ __forceinline__ int ptx_atomicAdd_gpu(int *addr, int val) {
    int old;
    // .gpu = GPU 内所有线程可见
    asm volatile ("atom.add.gpu.s32 %0, [%1], %2;"
                  : "=r"(old)
                  : "l"(addr), "r"(val)
                  : "memory");
    return old;
}

__global__ void testAtomicPTX(int *counter, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        ptx_atomicAdd(counter, 1);
    }
}

void demoAtomicPTX() {
    printf("=== 第五部分：原子操作 PTX ===\n\n");

    printf("原子操作指令格式:\n");
    printf("  atom{.space}.op{.scope}.type  dest, [addr], operand\n\n");

    printf("空间 (.space):\n");
    printf("  .global - 全局内存 (默认)\n");
    printf("  .shared - 共享内存\n\n");

    printf("操作 (.op):\n");
    printf("  ┌───────┬──────────────────────────────────────┐\n");
    printf("  │ 操作  │ 说明                                 │\n");
    printf("  ├───────┼──────────────────────────────────────┤\n");
    printf("  │ .add  │ 原子加法                             │\n");
    printf("  │ .min  │ 原子最小值                           │\n");
    printf("  │ .max  │ 原子最大值                           │\n");
    printf("  │ .inc  │ 原子递增 (带上限回绕)                │\n");
    printf("  │ .dec  │ 原子递减 (带上限回绕)                │\n");
    printf("  │ .cas  │ 比较并交换                           │\n");
    printf("  │ .exch │ 交换                                 │\n");
    printf("  │ .and  │ 按位与                               │\n");
    printf("  │ .or   │ 按位或                               │\n");
    printf("  │ .xor  │ 按位异或                             │\n");
    printf("  └───────┴──────────────────────────────────────┘\n\n");

    printf("作用域 (.scope) [Compute Capability 6.0+]:\n");
    printf("  .cta  - 线程块内可见\n");
    printf("  .gpu  - GPU 内所有线程可见\n");
    printf("  .sys  - 系统内所有线程可见 (含 CPU)\n\n");

    // 测试
    int *d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(int)));

    const int N = 10000;
    testAtomicPTX<<<(N+255)/256, 256>>>(d_counter, N);

    int h_counter;
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    printf("原子计数测试:\n");
    printf("  线程数: %d\n", N);
    printf("  计数结果: %d (期望 = %d)\n\n", h_counter, N);

    CHECK_CUDA(cudaFree(d_counter));
}

// ============================================================================
// 第六部分：Warp 级 PTX 指令
// ============================================================================

// PTX warp shuffle
__device__ __forceinline__ int ptx_shfl_sync(unsigned mask, int val, int src_lane) {
    int result;
    asm volatile ("shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;"
                  : "=r"(result)
                  : "r"(val), "r"(src_lane), "r"(mask));
    return result;
}

// PTX warp shuffle down
__device__ __forceinline__ int ptx_shfl_down_sync(unsigned mask, int val, int delta) {
    int result;
    asm volatile ("shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;"
                  : "=r"(result)
                  : "r"(val), "r"(delta), "r"(mask));
    return result;
}

// PTX warp vote - ballot
__device__ __forceinline__ unsigned ptx_ballot_sync(unsigned mask, int pred) {
    unsigned result;
    asm volatile ("vote.sync.ballot.b32 %0, %1, %2;"
                  : "=r"(result)
                  : "r"(pred), "r"(mask));
    return result;
}

// PTX warp vote - all
__device__ __forceinline__ int ptx_all_sync(unsigned mask, int pred) {
    int result;
    asm volatile ("vote.sync.all.pred %0, %1, %2;"
                  : "=r"(result)
                  : "r"(pred), "r"(mask));
    return result;
}

// PTX warp match
__device__ __forceinline__ unsigned ptx_match_any_sync(unsigned mask, int val) {
    unsigned result;
    asm volatile ("match.any.sync.b32 %0, %1, %2;"
                  : "=r"(result)
                  : "r"(val), "r"(mask));
    return result;
}

// 使用 PTX shuffle 实现 warp 归约
__device__ __forceinline__ int ptx_warp_reduce_sum(int val) {
    unsigned mask = 0xFFFFFFFF;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        int temp;
        asm volatile ("shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;"
                      : "=r"(temp)
                      : "r"(val), "r"(offset), "r"(mask));
        val += temp;
    }
    return val;
}

__global__ void testWarpPTX(int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 每个线程的值是其 lane id
    int val = ptx_laneid();

    // Warp 内归约求和
    int sum = ptx_warp_reduce_sum(val);

    // 只有 lane 0 写结果
    if (ptx_laneid() == 0) {
        output[tid / 32] = sum;
    }
}

void demoWarpPTX() {
    printf("=== 第六部分：Warp 级 PTX 指令 ===\n\n");

    printf("Shuffle 指令:\n");
    printf("  shfl.sync.idx   - 从任意 lane 读取\n");
    printf("  shfl.sync.up    - 从较低 lane 读取\n");
    printf("  shfl.sync.down  - 从较高 lane 读取\n");
    printf("  shfl.sync.bfly  - 蝴蝶模式 (XOR)\n\n");

    printf("Vote 指令:\n");
    printf("  vote.sync.all   - 所有线程谓词为真?\n");
    printf("  vote.sync.any   - 任一线程谓词为真?\n");
    printf("  vote.sync.uni   - 所有线程谓词相同?\n");
    printf("  vote.sync.ballot - 返回谓词的位图\n\n");

    printf("Match 指令 (Volta+):\n");
    printf("  match.any.sync  - 找到具有相同值的线程\n");
    printf("  match.all.sync  - 所有活跃线程值相同?\n\n");

    // 测试 warp 归约
    const int N = 128;  // 4 个 warp
    const int numWarps = N / 32;

    int *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, numWarps * sizeof(int)));

    testWarpPTX<<<1, N>>>(d_output, N);

    int *h_output = (int*)malloc(numWarps * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numWarps * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Warp 归约测试 (sum of lane IDs 0-31 = %d):\n", 31*32/2);
    for (int i = 0; i < numWarps; i++) {
        printf("  Warp %d 归约结果: %d\n", i, h_output[i]);
    }
    printf("\n");

    free(h_output);
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第七部分：控制流 PTX
// ============================================================================

// PTX 条件执行
__device__ __forceinline__ float ptx_conditional_add(float a, float b, int pred) {
    float result;
    // 使用谓词进行条件执行
    asm volatile (
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ne.s32 p, %3, 0;\n\t"       // 设置谓词
        "  @p add.f32 %0, %1, %2;\n\t"       // 条件执行
        "  @!p mov.f32 %0, %1;\n\t"          // 否则复制
        "}"
        : "=f"(result)
        : "f"(a), "f"(b), "r"(pred)
    );
    return result;
}

// PTX 内存栅栏
__device__ __forceinline__ void ptx_membar_cta() {
    asm volatile ("membar.cta;" ::: "memory");
}

__device__ __forceinline__ void ptx_membar_gpu() {
    asm volatile ("membar.gpu;" ::: "memory");
}

__device__ __forceinline__ void ptx_membar_sys() {
    asm volatile ("membar.sys;" ::: "memory");
}

// PTX barrier (同步)
__device__ __forceinline__ void ptx_bar_sync(int barrier_id) {
    asm volatile ("bar.sync %0;" :: "r"(barrier_id));
}

void demoControlFlowPTX() {
    printf("=== 第七部分：控制流 PTX ===\n\n");

    printf("谓词执行:\n");
    printf("  PTX 支持条件执行，使用谓词寄存器 (%%p)\n\n");

    printf("示例:\n");
    printf("  // 设置谓词\n");
    printf("  setp.lt.s32 p, %%r1, %%r2;  // p = (r1 < r2)\n\n");

    printf("  // 条件执行\n");
    printf("  @p add.s32 %%r3, %%r1, %%r2;  // 如果 p 为真，执行加法\n");
    printf("  @!p mov.s32 %%r3, 0;          // 如果 p 为假，赋值 0\n\n");

    printf("内存栅栏:\n");
    printf("  ┌─────────────┬────────────────────────────────────┐\n");
    printf("  │ 指令        │ 作用范围                           │\n");
    printf("  ├─────────────┼────────────────────────────────────┤\n");
    printf("  │ membar.cta  │ 线程块内                           │\n");
    printf("  │ membar.gpu  │ GPU 内所有线程                     │\n");
    printf("  │ membar.sys  │ 整个系统 (含 CPU)                  │\n");
    printf("  └─────────────┴────────────────────────────────────┘\n\n");

    printf("同步指令:\n");
    printf("  bar.sync N   - 等待块内所有线程到达 barrier N\n");
    printf("  bar.arrive N - 到达但不等待\n");
    printf("  bar.red      - 带归约的 barrier\n\n");
}

// ============================================================================
// 第八部分：性能优化案例
// ============================================================================

// 标准版本
__global__ void reduceStandard(int *output, const int *input, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// PTX 优化版本 (使用 warp shuffle)
__global__ void reducePTXOptimized(int *output, const int *input, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据
    int val = (gid < n) ? input[gid] : 0;

    // Warp 内使用 PTX shuffle 归约
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        int temp;
        asm volatile ("shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;"
                      : "=r"(temp) : "r"(val), "r"(offset), "r"(mask));
        val += temp;
    }

    // 每个 warp 的结果写入共享内存
    int lane = tid % 32;
    int warp = tid / 32;

    if (lane == 0) {
        sdata[warp] = val;
    }
    __syncthreads();

    // 第一个 warp 归约最终结果
    if (warp == 0) {
        val = (lane < (blockDim.x / 32)) ? sdata[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            int temp;
            asm volatile ("shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;"
                          : "=r"(temp) : "r"(val), "r"(offset), "r"(mask));
            val += temp;
        }

        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}

void demoPTXOptimization() {
    printf("=== 第八部分：PTX 性能优化案例 ===\n\n");

    const int N = 1 << 22;  // 4M 元素
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, gridSize * sizeof(int)));

    // 初始化
    int *h_input = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_input[i] = 1;  // 全 1
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 标准版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        reduceStandard<<<gridSize, blockSize, blockSize * sizeof(int)>>>(
            d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float standardTime;
    CHECK_CUDA(cudaEventElapsedTime(&standardTime, start, stop));
    standardTime /= NUM_RUNS;

    // PTX 优化版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        reducePTXOptimized<<<gridSize, blockSize, (blockSize/32) * sizeof(int)>>>(
            d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ptxTime;
    CHECK_CUDA(cudaEventElapsedTime(&ptxTime, start, stop));
    ptxTime /= NUM_RUNS;

    // 验证
    int *h_output = (int*)malloc(gridSize * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    int sum = 0;
    for (int i = 0; i < gridSize; i++) sum += h_output[i];

    printf("归约性能对比 (N = %d):\n", N);
    printf("  标准版本 (__syncthreads): %.3f ms\n", standardTime);
    printf("  PTX 优化版本 (shuffle):   %.3f ms\n", ptxTime);
    printf("  加速比: %.2fx\n\n", standardTime / ptxTime);
    printf("验证: sum = %d (期望 = %d)\n\n", sum, N);

    printf("优化要点:\n");
    printf("  1. Warp shuffle 避免共享内存使用\n");
    printf("  2. 减少 __syncthreads 调用\n");
    printf("  3. 更少的内存操作\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第九部分：PTX 调试与查看
// ============================================================================

void demoPTXDebugging() {
    printf("=== 第九部分：PTX 调试与查看 ===\n\n");

    printf("1. 生成 PTX 文件:\n");
    printf("   nvcc -ptx mykernel.cu -o mykernel.ptx\n\n");

    printf("2. 查看二进制中的 PTX:\n");
    printf("   cuobjdump -ptx myprogram\n\n");

    printf("3. 查看 SASS (原生机器码):\n");
    printf("   cuobjdump -sass myprogram\n\n");

    printf("4. 在代码中嵌入调试打印:\n");
    printf("   asm volatile (\"// DEBUG: value = %%0\" :: \"r\"(val));\n\n");

    printf("5. 使用 Nsight 查看:\n");
    printf("   - Nsight Compute 可以显示源码、PTX、SASS 对应关系\n");
    printf("   - 使用 -lineinfo 编译以保留行号信息\n");
    printf("   - nvcc -lineinfo mykernel.cu\n\n");

    printf("6. PTX 调试技巧:\n");
    printf("   - 使用 printf 调试 (compute_20+)\n");
    printf("   - 检查寄存器使用: nvcc --ptxas-options=-v\n");
    printf("   - 查看编译器决策: nvcc -Xptxas -v\n\n");

    printf("7. 常见 PTX 性能问题:\n");
    printf("   - 寄存器溢出 (spilling) - 查看 local 内存使用\n");
    printf("   - 分支发散 - 检查谓词使用\n");
    printf("   - 内存访问模式 - 查看 ld/st 指令修饰符\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 26: PTX 汇编与 Inline PTX                         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("SM 数量: %d\n\n", prop.multiProcessorCount);

    demoPTXOverview();
    demoBasicInlinePTX();
    demoSpecialRegisters();
    demoMemoryPTX();
    demoAtomicPTX();
    demoWarpPTX();
    demoControlFlowPTX();
    demoPTXOptimization();
    demoPTXDebugging();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. PTX 基础:\n");
    printf("   - PTX 是 CUDA 的中间表示\n");
    printf("   - 使用 asm volatile 嵌入 PTX\n");
    printf("   - 了解约束符号 (r, f, l, d)\n\n");

    printf("2. 特殊寄存器:\n");
    printf("   - %%laneid, %%warpid, %%smid\n");
    printf("   - %%clock64, %%globaltimer\n");
    printf("   - 用于精确控制和调试\n\n");

    printf("3. 内存操作:\n");
    printf("   - 缓存修饰符: .ca, .cg, .cs, .nc\n");
    printf("   - 向量加载/存储\n");
    printf("   - 原子操作作用域\n\n");

    printf("4. Warp 级操作:\n");
    printf("   - shfl.sync 族指令\n");
    printf("   - vote.sync 族指令\n");
    printf("   - match.sync 族指令\n\n");

    printf("5. 何时使用 Inline PTX:\n");
    printf("   - 访问 CUDA C 不支持的功能\n");
    printf("   - 极致性能优化\n");
    printf("   - 精确控制指令生成\n");
    printf("   - 注意: 大多数情况编译器做得更好!\n\n");

    return 0;
}
