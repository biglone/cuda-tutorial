/**
 * =============================================================================
 * CUDA 教程 08: 统一内存 (Unified Memory)
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解统一内存的概念和优势
 * 2. 学会使用 cudaMallocManaged() 分配托管内存
 * 3. 了解内存预取 (Prefetch) 优化
 * 4. 掌握内存使用建议 (Memory Hints)
 *
 * 关键概念：
 * - 统一内存创建单一内存空间，CPU 和 GPU 都可以访问
 * - 系统自动在 CPU 和 GPU 之间迁移数据
 * - 简化代码，无需手动 cudaMemcpy
 * - 可通过预取和提示优化性能
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
// 核函数：向量加法
// ============================================================================

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// ============================================================================
// 核函数：数组初始化（在 GPU 上初始化）
// ============================================================================

__global__ void initArray(float *arr, float value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        arr[tid] = value + tid * 0.001f;
    }
}

// ============================================================================
// 示例 1: 传统方式 vs 统一内存
// ============================================================================

void demoTraditionalVsUnified() {
    printf("=== 示例 1: 传统方式 vs 统一内存 ===\n\n");

    const int N = 1 << 20;  // 1M 元素
    const int size = N * sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // === 传统方式 ===
    printf("【传统方式】\n");
    {
        // 分配主机内存
        float *h_a = (float*)malloc(size);
        float *h_b = (float*)malloc(size);
        float *h_c = (float*)malloc(size);

        // 分配设备内存
        float *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, size));
        CHECK_CUDA(cudaMalloc(&d_b, size));
        CHECK_CUDA(cudaMalloc(&d_c, size));

        // CPU 初始化
        for (int i = 0; i < N; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
        }

        CHECK_CUDA(cudaEventRecord(start));

        // 复制到 GPU
        CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

        // 执行核函数
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

        // 复制回 CPU
        CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time;
        CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
        printf("  耗时: %.3f ms\n", time);
        printf("  结果验证: c[0] = %.1f (应为 3.0)\n", h_c[0]);

        // 清理
        free(h_a); free(h_b); free(h_c);
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    // === 统一内存方式 ===
    printf("\n【统一内存方式】\n");
    {
        // 分配托管内存 - CPU 和 GPU 都能访问
        float *a, *b, *c;
        CHECK_CUDA(cudaMallocManaged(&a, size));
        CHECK_CUDA(cudaMallocManaged(&b, size));
        CHECK_CUDA(cudaMallocManaged(&c, size));

        // CPU 直接初始化
        for (int i = 0; i < N; i++) {
            a[i] = 1.0f;
            b[i] = 2.0f;
        }

        CHECK_CUDA(cudaEventRecord(start));

        // 直接执行核函数，无需手动复制
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time;
        CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
        printf("  耗时: %.3f ms\n", time);
        printf("  结果验证: c[0] = %.1f (应为 3.0)\n", c[0]);

        // 清理 - 使用 cudaFree
        CHECK_CUDA(cudaFree(a));
        CHECK_CUDA(cudaFree(b));
        CHECK_CUDA(cudaFree(c));
    }

    printf("\n代码对比:\n");
    printf("  传统: malloc + cudaMalloc + cudaMemcpy x2 + cudaFree + free\n");
    printf("  统一: cudaMallocManaged + cudaFree\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 示例 2: 内存预取优化
// ============================================================================

void demoPrefetch() {
    printf("=== 示例 2: 内存预取 (Prefetch) ===\n\n");

    const int N = 1 << 22;  // 4M 元素
    const int size = N * sizeof(float);

    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    float *a, *b, *c;
    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    // CPU 初始化
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // === 无预取 ===
    printf("【无预取】\n");
    CHECK_CUDA(cudaEventRecord(start));

    vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float timeNoPrefetch;
    CHECK_CUDA(cudaEventElapsedTime(&timeNoPrefetch, start, stop));
    printf("  耗时: %.3f ms\n", timeNoPrefetch);

    // 重置数据
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // === 带预取 ===
    printf("\n【带预取】\n");
    CHECK_CUDA(cudaEventRecord(start));

    // 预取数据到 GPU
    CHECK_CUDA(cudaMemPrefetchAsync(a, size, device, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(b, size, device, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(c, size, device, 0));

    vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float timePrefetch;
    CHECK_CUDA(cudaEventElapsedTime(&timePrefetch, start, stop));
    printf("  耗时: %.3f ms\n", timePrefetch);
    printf("  加速比: %.2fx\n\n", timeNoPrefetch / timePrefetch);

    /*
     * 预取的作用：
     * - 无预取：GPU 访问时按需迁移页面，产生页面故障
     * - 有预取：提前批量迁移，减少页面故障开销
     */

    printf("说明:\n");
    printf("  cudaMemPrefetchAsync(ptr, size, device, stream)\n");
    printf("  - device = GPU设备ID: 预取到 GPU\n");
    printf("  - device = cudaCpuDeviceId: 预取到 CPU\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
}

// ============================================================================
// 示例 3: 内存使用建议 (Memory Advise)
// ============================================================================

void demoMemAdvise() {
    printf("=== 示例 3: 内存使用建议 ===\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);

    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    float *readOnlyData;
    float *writeData;
    float *result;

    CHECK_CUDA(cudaMallocManaged(&readOnlyData, size));
    CHECK_CUDA(cudaMallocManaged(&writeData, size));
    CHECK_CUDA(cudaMallocManaged(&result, size));

    // 初始化
    for (int i = 0; i < N; i++) {
        readOnlyData[i] = 1.0f;
        writeData[i] = 0.0f;
    }

    printf("可用的内存建议:\n\n");

    // 1. 设置首选位置
    printf("1. cudaMemAdviseSetPreferredLocation\n");
    printf("   设置数据的首选存放位置\n");
    CHECK_CUDA(cudaMemAdvise(readOnlyData, size,
        cudaMemAdviseSetPreferredLocation, device));
    printf("   -> readOnlyData 首选位置设为 GPU\n\n");

    // 2. 设置只读
    printf("2. cudaMemAdviseSetReadMostly\n");
    printf("   标记数据为只读，允许创建副本提高访问效率\n");
    CHECK_CUDA(cudaMemAdvise(readOnlyData, size,
        cudaMemAdviseSetReadMostly, device));
    printf("   -> readOnlyData 标记为只读\n\n");

    // 3. 设置访问者
    printf("3. cudaMemAdviseSetAccessedBy\n");
    printf("   提示哪个处理器会访问该数据\n");
    CHECK_CUDA(cudaMemAdvise(writeData, size,
        cudaMemAdviseSetAccessedBy, device));
    printf("   -> writeData 将被 GPU 访问\n\n");

    // 预取并执行
    CHECK_CUDA(cudaMemPrefetchAsync(readOnlyData, size, device, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(writeData, size, device, 0));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(readOnlyData, writeData, result, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 取消建议
    printf("取消建议示例:\n");
    CHECK_CUDA(cudaMemAdvise(readOnlyData, size,
        cudaMemAdviseUnsetReadMostly, device));
    printf("   -> 取消 readOnlyData 的只读标记\n\n");

    CHECK_CUDA(cudaFree(readOnlyData));
    CHECK_CUDA(cudaFree(writeData));
    CHECK_CUDA(cudaFree(result));
}

// ============================================================================
// 示例 4: GPU 上初始化数据
// ============================================================================

void demoGPUInit() {
    printf("=== 示例 4: GPU 上初始化数据 ===\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);

    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    float *a, *b, *c;
    CHECK_CUDA(cudaMallocManaged(&a, size));
    CHECK_CUDA(cudaMallocManaged(&b, size));
    CHECK_CUDA(cudaMallocManaged(&c, size));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // === CPU 初始化 ===
    printf("【CPU 初始化后 GPU 计算】\n");
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    CHECK_CUDA(cudaEventRecord(start));
    vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float timeCPUInit;
    CHECK_CUDA(cudaEventElapsedTime(&timeCPUInit, start, stop));
    printf("  耗时: %.3f ms\n", timeCPUInit);

    // === GPU 初始化 ===
    printf("\n【GPU 初始化后 GPU 计算】\n");

    // 预取空数据到 GPU，确保分配在 GPU 上
    CHECK_CUDA(cudaMemPrefetchAsync(a, size, device, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(b, size, device, 0));
    CHECK_CUDA(cudaMemPrefetchAsync(c, size, device, 0));

    CHECK_CUDA(cudaEventRecord(start));

    // 在 GPU 上初始化
    initArray<<<gridSize, blockSize>>>(a, 1.0f, N);
    initArray<<<gridSize, blockSize>>>(b, 2.0f, N);

    // 计算
    vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float timeGPUInit;
    CHECK_CUDA(cudaEventElapsedTime(&timeGPUInit, start, stop));
    printf("  耗时: %.3f ms\n", timeGPUInit);
    printf("  加速比: %.2fx\n\n", timeCPUInit / timeGPUInit);

    printf("说明:\n");
    printf("  - CPU 初始化导致数据先在 CPU 内存\n");
    printf("  - GPU 访问时需要页面迁移\n");
    printf("  - GPU 初始化避免了跨设备迁移\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
}

// ============================================================================
// 示例 5: 统一内存与 Streams 结合
// ============================================================================

__global__ void scaleArray(float *arr, float scale, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        arr[tid] *= scale;
    }
}

void demoUnifiedWithStreams() {
    printf("=== 示例 5: 统一内存与 Streams 结合 ===\n\n");

    const int NUM_STREAMS = 4;
    const int N = 1 << 22;
    const int CHUNK_SIZE = N / NUM_STREAMS;

    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    // 分配统一内存
    float *data;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(float)));

    // CPU 初始化
    for (int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    // 创建 streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (CHUNK_SIZE + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaEventRecord(start));

    // 每个 stream 处理一个数据块
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * CHUNK_SIZE;
        int chunkBytes = CHUNK_SIZE * sizeof(float);

        // 预取该块到 GPU
        CHECK_CUDA(cudaMemPrefetchAsync(&data[offset], chunkBytes,
                                        device, streams[i]));

        // 在该 stream 中处理
        scaleArray<<<gridSize, blockSize, 0, streams[i]>>>(
            &data[offset], 2.0f, CHUNK_SIZE);
    }

    // 等待所有完成
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    printf("  处理 %d 个元素，使用 %d 个 streams\n", N, NUM_STREAMS);
    printf("  耗时: %.3f ms\n", time);
    printf("  结果验证: data[0] = %.1f (应为 2.0)\n\n", data[0]);

    /*
     * 每个 stream 独立工作：
     *
     * Stream 0: [预取块0][计算块0]
     * Stream 1:    [预取块1][计算块1]
     * Stream 2:       [预取块2][计算块2]
     * Stream 3:          [预取块3][计算块3]
     */

    // 清理
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(data));
}

// ============================================================================
// 示例 6: 查询内存属性
// ============================================================================

void demoMemoryAttributes() {
    printf("=== 示例 6: 查询内存属性 ===\n\n");

    const int size = 1024 * sizeof(float);

    // 分配不同类型的内存
    float *managed, *deviceOnly, *hostPinned;

    CHECK_CUDA(cudaMallocManaged(&managed, size));
    CHECK_CUDA(cudaMalloc(&deviceOnly, size));
    CHECK_CUDA(cudaMallocHost(&hostPinned, size));

    // 查询托管内存属性
    cudaPointerAttributes attr;

    CHECK_CUDA(cudaPointerGetAttributes(&attr, managed));
    printf("托管内存 (cudaMallocManaged):\n");
    printf("  类型: %s\n",
        attr.type == cudaMemoryTypeManaged ? "Managed" :
        attr.type == cudaMemoryTypeDevice ? "Device" :
        attr.type == cudaMemoryTypeHost ? "Host" : "Unknown");
    printf("  设备指针: %p\n", attr.devicePointer);
    printf("  主机指针: %p\n", attr.hostPointer);
    printf("  设备: %d\n\n", attr.device);

    CHECK_CUDA(cudaPointerGetAttributes(&attr, deviceOnly));
    printf("设备内存 (cudaMalloc):\n");
    printf("  类型: %s\n",
        attr.type == cudaMemoryTypeDevice ? "Device" : "Other");
    printf("  设备指针: %p\n", attr.devicePointer);
    printf("  主机指针: %p\n\n", attr.hostPointer);

    CHECK_CUDA(cudaPointerGetAttributes(&attr, hostPinned));
    printf("固定内存 (cudaMallocHost):\n");
    printf("  类型: %s\n",
        attr.type == cudaMemoryTypeHost ? "Host" : "Other");
    printf("  设备指针: %p\n", attr.devicePointer);
    printf("  主机指针: %p\n\n", attr.hostPointer);

    CHECK_CUDA(cudaFree(managed));
    CHECK_CUDA(cudaFree(deviceOnly));
    CHECK_CUDA(cudaFreeHost(hostPinned));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 08: 统一内存 (Unified Memory)                ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    // 检查设备支持
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("统一内存支持: %s\n", prop.managedMemory ? "是" : "否");
    printf("并发托管内存访问: %s\n",
        prop.concurrentManagedAccess ? "是" : "否");
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    if (!prop.managedMemory) {
        printf("警告: 设备不支持统一内存！\n");
        return 1;
    }

    demoTraditionalVsUnified();
    demoPrefetch();
    demoMemAdvise();
    demoGPUInit();
    demoUnifiedWithStreams();
    demoMemoryAttributes();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 统一内存基础:\n");
    printf("   - cudaMallocManaged() 分配 CPU/GPU 都能访问的内存\n");
    printf("   - 系统自动处理数据迁移\n");
    printf("   - 简化代码，无需手动 cudaMemcpy\n\n");

    printf("2. 性能优化:\n");
    printf("   - cudaMemPrefetchAsync() 预取数据\n");
    printf("   - cudaMemAdvise() 提供内存使用提示\n");
    printf("   - 尽量在 GPU 上初始化数据\n\n");

    printf("3. 内存建议类型:\n");
    printf("   - SetPreferredLocation: 首选存放位置\n");
    printf("   - SetReadMostly: 只读数据\n");
    printf("   - SetAccessedBy: 访问者提示\n\n");

    printf("4. 最佳实践:\n");
    printf("   - 原型开发阶段使用统一内存简化代码\n");
    printf("   - 性能关键代码考虑显式内存管理\n");
    printf("   - 使用预取和建议优化统一内存性能\n");
    printf("   - 结合 Streams 实现并发预取和计算\n\n");

    printf("5. 内存类型对比:\n");
    printf("   ┌─────────────────┬───────────────┬───────────────┐\n");
    printf("   │     类型        │   CPU 访问    │   GPU 访问    │\n");
    printf("   ├─────────────────┼───────────────┼───────────────┤\n");
    printf("   │ malloc          │      √        │      ×        │\n");
    printf("   │ cudaMalloc      │      ×        │      √        │\n");
    printf("   │ cudaMallocHost  │      √        │     映射      │\n");
    printf("   │ cudaMallocManaged│     √        │      √        │\n");
    printf("   └─────────────────┴───────────────┴───────────────┘\n\n");

    return 0;
}
