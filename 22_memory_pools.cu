/**
 * =============================================================================
 * CUDA 教程 22: CUDA 内存池与虚拟内存管理
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 CUDA 内存池 (Memory Pool) 的概念和优势
 * 2. 学会使用 cudaMallocAsync 进行异步内存分配
 * 3. 掌握 CUDA 虚拟内存管理 API
 * 4. 了解内存池的配置和调优
 *
 * 关键概念：
 * - Memory Pool: 预分配内存池，减少分配开销
 * - Stream-Ordered Memory: 与流操作顺序一致的内存管理
 * - Virtual Memory: 灵活的内存映射和管理
 *
 * 编译命令：
 *   nvcc 22_memory_pools.cu -o 22_memory_pools
 *
 * 需要: CUDA 11.2+ (内存池), CUDA 10.2+ (虚拟内存)
 */

#include <stdio.h>
#include <stdint.h>  // for UINT64_MAX
#include <cuda_runtime.h>
#include "cuda_version_compat.h"
#include <cuda.h>    // for CUDA Driver API (virtual memory)
#include <vector>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 简单测试内核
// ============================================================================

__global__ void initKernel(float *data, int n, float value) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = value;
    }
}

__global__ void addKernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// ============================================================================
// 第一部分：内存池概念介绍
// ============================================================================

void demoMemoryPoolConcepts() {
    printf("=== 第一部分：内存池概念 ===\n\n");

    printf("传统内存分配问题:\n");
    printf("  - cudaMalloc/cudaFree 有显著开销\n");
    printf("  - 每次分配可能触发 GPU 同步\n");
    printf("  - 频繁分配/释放导致内存碎片\n");
    printf("  - 不适合动态工作负载\n\n");

    printf("内存池解决方案:\n");
    printf("  - 预分配内存池，重用内存块\n");
    printf("  - 异步分配，无需同步\n");
    printf("  - 流顺序语义，自动管理依赖\n");
    printf("  - 减少碎片和开销\n\n");

    printf("CUDA 内存池 API (CUDA 11.2+):\n");
    printf("  - cudaMallocAsync: 异步分配\n");
    printf("  - cudaFreeAsync: 异步释放\n");
    printf("  - cudaMemPoolCreate: 创建自定义池\n");
    printf("  - cudaDeviceGetDefaultMemPool: 获取默认池\n\n");

    printf("Stream-Ordered Memory Allocation:\n");
    printf("  ┌─────────────────────────────────────────┐\n");
    printf("  │  Stream A                               │\n");
    printf("  │  cudaMallocAsync(&ptr, size, streamA)   │\n");
    printf("  │       │                                 │\n");
    printf("  │       ▼                                 │\n");
    printf("  │  kernel<<<..., streamA>>>(ptr)          │\n");
    printf("  │       │                                 │\n");
    printf("  │       ▼                                 │\n");
    printf("  │  cudaFreeAsync(ptr, streamA)            │\n");
    printf("  │       │                                 │\n");
    printf("  │       ▼                                 │\n");
    printf("  │  (内存返回池，可被复用)                 │\n");
    printf("  └─────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第二部分：异步内存分配 (cudaMallocAsync)
// ============================================================================

void demoAsyncAllocation() {
    printf("=== 第二部分：异步内存分配 ===\n\n");

    // 检查设备是否支持内存池
    int device = 0;
    int supportsMemoryPools = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device));

    if (!supportsMemoryPools) {
        printf("警告: 当前设备不支持内存池，跳过此演示\n\n");
        return;
    }

    printf("设备支持内存池: 是\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 异步分配
    float *d_a, *d_b, *d_c;

    printf("1. 异步分配内存:\n");
    CHECK_CUDA(cudaMallocAsync(&d_a, size, stream));
    CHECK_CUDA(cudaMallocAsync(&d_b, size, stream));
    CHECK_CUDA(cudaMallocAsync(&d_c, size, stream));
    printf("   分配了 3 个 %.2f MB 的数组\n\n", size / (1024.0f * 1024.0f));

    // 初始化
    printf("2. 初始化数据:\n");
    initKernel<<<gridSize, blockSize, 0, stream>>>(d_a, N, 1.0f);
    initKernel<<<gridSize, blockSize, 0, stream>>>(d_b, N, 2.0f);

    // 计算
    printf("3. 执行计算:\n");
    addKernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);

    // 验证
    float result;
    CHECK_CUDA(cudaMemcpyAsync(&result, d_c, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("   c[0] = %.1f (期望: 3.0)\n\n", result);

    // 异步释放
    printf("4. 异步释放内存:\n");
    CHECK_CUDA(cudaFreeAsync(d_a, stream));
    CHECK_CUDA(cudaFreeAsync(d_b, stream));
    CHECK_CUDA(cudaFreeAsync(d_c, stream));
    printf("   内存返回到池中\n\n");

    // 性能对比
    printf("5. 性能对比 (1000 次分配/释放):\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 异步分配性能
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < 1000; i++) {
        float *temp;
        CHECK_CUDA(cudaMallocAsync(&temp, size, stream));
        initKernel<<<gridSize, blockSize, 0, stream>>>(temp, N, 1.0f);
        CHECK_CUDA(cudaFreeAsync(temp, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float asyncTime;
    CHECK_CUDA(cudaEventElapsedTime(&asyncTime, start, stop));
    printf("   cudaMallocAsync: %.2f ms\n", asyncTime);

    // 同步分配性能
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < 1000; i++) {
        float *temp;
        CHECK_CUDA(cudaMalloc(&temp, size));
        initKernel<<<gridSize, blockSize>>>(temp, N, 1.0f);
        CHECK_CUDA(cudaFree(temp));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    float syncTime;
    CHECK_CUDA(cudaEventElapsedTime(&syncTime, start, stop));
    printf("   cudaMalloc:      %.2f ms\n", syncTime);
    printf("   加速比: %.2fx\n\n", syncTime / asyncTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

// ============================================================================
// 第三部分：内存池配置
// ============================================================================

void demoMemoryPoolConfig() {
    printf("=== 第三部分：内存池配置 ===\n\n");

    int device = 0;
    int supportsMemoryPools = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device));

    if (!supportsMemoryPools) {
        printf("警告: 当前设备不支持内存池，跳过此演示\n\n");
        return;
    }

    // 获取默认内存池
    cudaMemPool_t pool;
    CHECK_CUDA(cudaDeviceGetDefaultMemPool(&pool, device));

    printf("1. 默认内存池属性:\n");

    // 获取当前释放阈值
    size_t threshold;
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrReleaseThreshold, &threshold));
    printf("   释放阈值: %zu bytes\n", threshold);

    // 获取保留内存
    size_t reserved, used;
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrReservedMemCurrent, &reserved));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrUsedMemCurrent, &used));
    printf("   当前保留: %zu bytes\n", reserved);
    printf("   当前使用: %zu bytes\n\n", used);

    // 配置内存池
    printf("2. 配置内存池:\n");

    // 设置释放阈值 (不自动释放)
    size_t newThreshold = UINT64_MAX;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool,
        cudaMemPoolAttrReleaseThreshold, &newThreshold));
    printf("   设置释放阈值为 UINT64_MAX (不自动释放)\n");

    // 分配一些内存来测试
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int size = 100 * 1024 * 1024;  // 100 MB
    float *d_data;
    CHECK_CUDA(cudaMallocAsync(&d_data, size, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrReservedMemCurrent, &reserved));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrUsedMemCurrent, &used));
    printf("\n   分配后:\n");
    printf("     保留: %.2f MB\n", reserved / (1024.0f * 1024.0f));
    printf("     使用: %.2f MB\n", used / (1024.0f * 1024.0f));

    CHECK_CUDA(cudaFreeAsync(d_data, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrReservedMemCurrent, &reserved));
    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrUsedMemCurrent, &used));
    printf("\n   释放后:\n");
    printf("     保留: %.2f MB (内存仍在池中)\n", reserved / (1024.0f * 1024.0f));
    printf("     使用: %.2f MB\n", used / (1024.0f * 1024.0f));

    // 手动释放池中未使用的内存
    printf("\n3. 手动释放池内存:\n");
    CHECK_CUDA(cudaMemPoolTrimTo(pool, 0));

    CHECK_CUDA(cudaMemPoolGetAttribute(pool,
        cudaMemPoolAttrReservedMemCurrent, &reserved));
    printf("   TrimTo(0) 后保留: %.2f MB\n\n", reserved / (1024.0f * 1024.0f));

    CHECK_CUDA(cudaStreamDestroy(stream));
}

// ============================================================================
// 第四部分：自定义内存池
// ============================================================================

void demoCustomMemoryPool() {
    printf("=== 第四部分：自定义内存池 ===\n\n");

    int device = 0;
    int supportsMemoryPools = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device));

    if (!supportsMemoryPools) {
        printf("警告: 当前设备不支持内存池，跳过此演示\n\n");
        return;
    }

    printf("1. 创建自定义内存池:\n");

    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = device;

    cudaMemPool_t customPool;
    CHECK_CUDA(cudaMemPoolCreate(&customPool, &poolProps));
    printf("   创建成功\n\n");

    // 使用自定义池
    printf("2. 使用自定义池分配:\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int size = 10 * 1024 * 1024;  // 10 MB
    float *d_data;

    // 从自定义池分配
    CHECK_CUDA(cudaMallocFromPoolAsync(&d_data, size, customPool, stream));
    printf("   从自定义池分配了 10 MB\n");

    // 检查池状态
    size_t reserved, used;
    CHECK_CUDA(cudaMemPoolGetAttribute(customPool,
        cudaMemPoolAttrReservedMemCurrent, &reserved));
    CHECK_CUDA(cudaMemPoolGetAttribute(customPool,
        cudaMemPoolAttrUsedMemCurrent, &used));
    printf("   池保留: %.2f MB\n", reserved / (1024.0f * 1024.0f));
    printf("   池使用: %.2f MB\n\n", used / (1024.0f * 1024.0f));

    // 释放
    CHECK_CUDA(cudaFreeAsync(d_data, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 将流关联到自定义池
    printf("3. 设置流的默认内存池:\n");

    cudaMemPool_t oldPool;
    CHECK_CUDA(cudaDeviceGetMemPool(&oldPool, device));

    CHECK_CUDA(cudaDeviceSetMemPool(device, customPool));
    printf("   设备默认池已更改为自定义池\n");

    // 现在 cudaMallocAsync 会使用自定义池
    CHECK_CUDA(cudaMallocAsync(&d_data, size, stream));
    CHECK_CUDA(cudaMemPoolGetAttribute(customPool,
        cudaMemPoolAttrUsedMemCurrent, &used));
    printf("   通过 cudaMallocAsync 分配后池使用: %.2f MB\n\n", used / (1024.0f * 1024.0f));

    // 恢复默认池
    CHECK_CUDA(cudaDeviceSetMemPool(device, oldPool));

    // 清理
    CHECK_CUDA(cudaFreeAsync(d_data, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemPoolDestroy(customPool));
    CHECK_CUDA(cudaStreamDestroy(stream));

    printf("4. 已销毁自定义池\n\n");
}

// ============================================================================
// 第五部分：跨流内存共享
// ============================================================================

void demoCrossStreamSharing() {
    printf("=== 第五部分：跨流内存共享 ===\n\n");

    int device = 0;
    int supportsMemoryPools = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device));

    if (!supportsMemoryPools) {
        printf("警告: 当前设备不支持内存池，跳过此演示\n\n");
        return;
    }

    printf("跨流共享异步分配的内存:\n\n");

    cudaStream_t streamA, streamB;
    CHECK_CUDA(cudaStreamCreate(&streamA));
    CHECK_CUDA(cudaStreamCreate(&streamB));

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 在 streamA 中分配
    float *d_data;
    CHECK_CUDA(cudaMallocAsync(&d_data, size, streamA));
    printf("1. 在 streamA 中分配内存\n");

    // 在 streamA 中初始化
    initKernel<<<gridSize, blockSize, 0, streamA>>>(d_data, N, 5.0f);
    printf("2. 在 streamA 中初始化\n");

    // 让 streamB 等待 streamA
    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));
    CHECK_CUDA(cudaEventRecord(event, streamA));
    CHECK_CUDA(cudaStreamWaitEvent(streamB, event, 0));
    printf("3. streamB 等待 streamA (通过事件)\n");

    // 在 streamB 中使用数据
    initKernel<<<gridSize, blockSize, 0, streamB>>>(d_data, N, 10.0f);
    printf("4. 在 streamB 中修改数据\n");

    // 让 streamA 等待 streamB 完成后再释放
    CHECK_CUDA(cudaEventRecord(event, streamB));
    CHECK_CUDA(cudaStreamWaitEvent(streamA, event, 0));
    CHECK_CUDA(cudaFreeAsync(d_data, streamA));
    printf("5. streamA 等待 streamB 后释放内存\n\n");

    // 验证
    CHECK_CUDA(cudaStreamSynchronize(streamA));
    CHECK_CUDA(cudaStreamSynchronize(streamB));
    printf("   所有流已同步完成\n\n");

    printf("重要: 跨流共享时必须使用事件同步!\n");
    printf("  否则可能出现数据竞争或过早释放\n\n");

    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaStreamDestroy(streamA));
    CHECK_CUDA(cudaStreamDestroy(streamB));
}

// ============================================================================
// 第六部分：虚拟内存管理
// ============================================================================

void demoVirtualMemory() {
    printf("=== 第六部分：虚拟内存管理 ===\n\n");

    // 检查虚拟内存支持
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 7) {
        printf("警告: 虚拟内存需要计算能力 7.0+，跳过此演示\n\n");
        return;
    }

    printf("虚拟内存管理 (CUDA 10.2+):\n\n");

    printf("概念:\n");
    printf("  - 分离地址空间和物理内存\n");
    printf("  - 可以预留大地址范围\n");
    printf("  - 按需映射物理内存\n");
    printf("  - 支持动态增长的数据结构\n\n");

    // 初始化 CUDA Driver API
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) {
        printf("警告: cuInit 失败，跳过虚拟内存演示\n\n");
        return;
    }

    // 获取分配粒度（使用 Driver API）
    size_t granularity;
    CUmemAllocationProp allocProp = {};
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = device;

    cuErr = cuMemGetAllocationGranularity(&granularity, &allocProp,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (cuErr != CUDA_SUCCESS) {
        printf("警告: cuMemGetAllocationGranularity 不支持，跳过虚拟内存演示\n\n");
        return;
    }

    printf("1. 分配粒度: %zu bytes (%.2f KB)\n\n", granularity, granularity / 1024.0f);

    // 计算对齐大小
    size_t requestedSize = 100 * 1024 * 1024;  // 100 MB
    size_t alignedSize = ((requestedSize + granularity - 1) / granularity) * granularity;

    printf("2. 预留虚拟地址空间:\n");
    printf("   请求大小: %zu MB\n", requestedSize / (1024 * 1024));
    printf("   对齐大小: %zu MB\n\n", alignedSize / (1024 * 1024));

    // 预留虚拟地址空间
    CUdeviceptr dptr = 0;
    cuErr = cuMemAddressReserve(&dptr, alignedSize, granularity, 0, 0);
    if (cuErr != CUDA_SUCCESS) {
        printf("警告: cuMemAddressReserve 失败，跳过虚拟内存演示\n\n");
        return;
    }
    printf("   预留地址: 0x%llx\n\n", (unsigned long long)dptr);

    // 创建物理内存句柄
    printf("3. 创建物理内存:\n");
    CUmemGenericAllocationHandle allocHandle;

    CUmemAllocationProp cuAllocProp = {};
    cuAllocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    cuAllocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    cuAllocProp.location.id = device;

    cuErr = cuMemCreate(&allocHandle, alignedSize, &cuAllocProp, 0);
    if (cuErr != CUDA_SUCCESS) {
        cuMemAddressFree(dptr, alignedSize);
        printf("警告: cuMemCreate 失败，跳过虚拟内存演示\n\n");
        return;
    }
    printf("   创建 %zu MB 物理内存\n\n", alignedSize / (1024 * 1024));

    // 映射物理内存到虚拟地址
    printf("4. 映射内存:\n");
    cuErr = cuMemMap(dptr, alignedSize, 0, allocHandle, 0);
    if (cuErr != CUDA_SUCCESS) {
        cuMemRelease(allocHandle);
        cuMemAddressFree(dptr, alignedSize);
        printf("警告: cuMemMap 失败\n\n");
        return;
    }
    printf("   已映射到虚拟地址\n\n");

    // 设置访问权限
    printf("5. 设置访问权限:\n");
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    cuErr = cuMemSetAccess(dptr, alignedSize, &accessDesc, 1);
    if (cuErr != CUDA_SUCCESS) {
        cuMemUnmap(dptr, alignedSize);
        cuMemRelease(allocHandle);
        cuMemAddressFree(dptr, alignedSize);
        printf("警告: cuMemSetAccess 失败\n\n");
        return;
    }
    printf("   设置为读写权限\n\n");

    // 使用内存
    printf("6. 使用虚拟内存:\n");
    float *d_data = (float*)dptr;
    const int N = alignedSize / sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    initKernel<<<gridSize, blockSize>>>(d_data, N, 3.14f);
    CHECK_CUDA(cudaDeviceSynchronize());

    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    printf("   data[0] = %.2f\n\n", result);

    // 清理
    printf("7. 清理:\n");
    cuMemUnmap(dptr, alignedSize);
    printf("   取消映射\n");
    cuMemRelease(allocHandle);
    printf("   释放物理内存\n");
    cuMemAddressFree(dptr, alignedSize);
    printf("   释放虚拟地址\n\n");
}

// ============================================================================
// 第七部分：动态增长数组示例
// ============================================================================

void demoGrowableArray() {
    printf("=== 第七部分：动态增长数组 ===\n\n");

    int device = 0;
    int supportsMemoryPools = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device));

    if (!supportsMemoryPools) {
        printf("警告: 当前设备不支持内存池，跳过此演示\n\n");
        return;
    }

    printf("使用内存池实现动态增长数组:\n\n");

    printf("// 伪代码示例\n");
    printf("class GrowableDeviceArray {\n");
    printf("    float *d_data;\n");
    printf("    size_t capacity;\n");
    printf("    size_t size;\n");
    printf("    cudaStream_t stream;\n");
    printf("    \n");
    printf("public:\n");
    printf("    void resize(size_t newSize) {\n");
    printf("        if (newSize > capacity) {\n");
    printf("            size_t newCapacity = newSize * 2;\n");
    printf("            float *newData;\n");
    printf("            \n");
    printf("            // 异步分配新内存\n");
    printf("            cudaMallocAsync(&newData, newCapacity * sizeof(float), stream);\n");
    printf("            \n");
    printf("            // 复制旧数据\n");
    printf("            if (d_data) {\n");
    printf("                cudaMemcpyAsync(newData, d_data, size * sizeof(float),\n");
    printf("                                cudaMemcpyDeviceToDevice, stream);\n");
    printf("                cudaFreeAsync(d_data, stream);\n");
    printf("            }\n");
    printf("            \n");
    printf("            d_data = newData;\n");
    printf("            capacity = newCapacity;\n");
    printf("        }\n");
    printf("        size = newSize;\n");
    printf("    }\n");
    printf("};\n\n");

    // 实际演示
    printf("实际演示:\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<float*> allocations;
    size_t totalAllocated = 0;

    printf("逐步增长分配:\n");
    for (int i = 0; i < 10; i++) {
        size_t size = (1 << (i + 16)) * sizeof(float);  // 从 256KB 到 128MB
        float *d_data;
        CHECK_CUDA(cudaMallocAsync(&d_data, size, stream));
        allocations.push_back(d_data);
        totalAllocated += size;
        printf("  分配 #%d: %.2f MB (总计: %.2f MB)\n",
               i + 1, size / (1024.0f * 1024.0f), totalAllocated / (1024.0f * 1024.0f));
    }

    printf("\n批量释放:\n");
    for (float *ptr : allocations) {
        CHECK_CUDA(cudaFreeAsync(ptr, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("  所有内存已释放回池\n\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
}

// ============================================================================
// 第八部分：最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第八部分：最佳实践 ===\n\n");

    printf("1. 何时使用内存池:\n");
    printf("   ✓ 频繁的小内存分配/释放\n");
    printf("   ✓ 动态工作负载\n");
    printf("   ✓ 图 (Graph) 中的分配\n");
    printf("   ✓ 需要低延迟的应用\n\n");

    printf("2. 内存池调优:\n");
    printf("   // 设置释放阈值\n");
    printf("   size_t threshold = 1024 * 1024 * 1024;  // 1GB\n");
    printf("   cudaMemPoolSetAttribute(pool,\n");
    printf("       cudaMemPoolAttrReleaseThreshold, &threshold);\n\n");

    printf("   // 定期修剪\n");
    printf("   cudaMemPoolTrimTo(pool, minSize);\n\n");

    printf("3. 避免的问题:\n");
    printf("   ✗ 跨流使用未同步的内存\n");
    printf("   ✗ 忘记释放异步分配的内存\n");
    printf("   ✗ 在错误的流中释放\n");
    printf("   ✗ 池内存使用后同步分配\n\n");

    printf("4. 虚拟内存使用场景:\n");
    printf("   - 大型稀疏数据结构\n");
    printf("   - 需要地址连续性\n");
    printf("   - 延迟分配物理内存\n");
    printf("   - 多 GPU 内存映射\n\n");

    printf("5. 调试技巧:\n");
    printf("   // 检查池状态\n");
    printf("   size_t reserved, used, high;\n");
    printf("   cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);\n");
    printf("   cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);\n");
    printf("   cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &high);\n");
    printf("   printf(\"保留: %%zu, 使用: %%zu, 峰值: %%zu\\n\", reserved, used, high);\n\n");

    printf("6. 与 CUDA Graph 集成:\n");
    printf("   // 在图中使用异步分配\n");
    printf("   cudaGraphCreate(&graph, 0);\n");
    printf("   \n");
    printf("   // 添加分配节点\n");
    printf("   cudaMemAllocNodeParams allocParams = {};\n");
    printf("   allocParams.poolProps.allocType = cudaMemAllocationTypePinned;\n");
    printf("   allocParams.poolProps.location.type = cudaMemLocationTypeDevice;\n");
    printf("   allocParams.poolProps.location.id = device;\n");
    printf("   allocParams.bytesize = size;\n");
    printf("   \n");
    printf("   cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &allocParams);\n");
    printf("   \n");
    printf("   // 添加释放节点\n");
    printf("   cudaGraphAddMemFreeNode(&freeNode, graph, deps, 1, allocParams.dptr);\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA 教程 22: 内存池与虚拟内存管理                             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    // 检查设备
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);

    int driverVersion, runtimeVersion;
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    printf("CUDA Driver: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime: %d.%d\n\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    demoMemoryPoolConcepts();
    demoAsyncAllocation();
    demoMemoryPoolConfig();
    demoCustomMemoryPool();
    demoCrossStreamSharing();
    demoVirtualMemory();
    demoGrowableArray();
    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 内存池核心 API:\n");
    printf("   ┌────────────────────────────────┬──────────────────────────────┐\n");
    printf("   │ 函数                           │ 功能                         │\n");
    printf("   ├────────────────────────────────┼──────────────────────────────┤\n");
    printf("   │ cudaMallocAsync                │ 异步分配                     │\n");
    printf("   │ cudaFreeAsync                  │ 异步释放                     │\n");
    printf("   │ cudaMallocFromPoolAsync        │ 从指定池分配                 │\n");
    printf("   │ cudaMemPoolCreate              │ 创建自定义池                 │\n");
    printf("   │ cudaMemPoolSetAttribute        │ 配置池属性                   │\n");
    printf("   │ cudaMemPoolTrimTo              │ 修剪池大小                   │\n");
    printf("   └────────────────────────────────┴──────────────────────────────┘\n\n");

    printf("2. 虚拟内存 API:\n");
    printf("   ┌────────────────────────────────┬──────────────────────────────┐\n");
    printf("   │ 函数                           │ 功能                         │\n");
    printf("   ├────────────────────────────────┼──────────────────────────────┤\n");
    printf("   │ cuMemAddressReserve            │ 预留虚拟地址空间             │\n");
    printf("   │ cuMemCreate                    │ 创建物理内存句柄             │\n");
    printf("   │ cuMemMap                       │ 映射物理到虚拟               │\n");
    printf("   │ cuMemSetAccess                 │ 设置访问权限                 │\n");
    printf("   │ cuMemUnmap                     │ 取消映射                     │\n");
    printf("   │ cuMemRelease                   │ 释放物理内存                 │\n");
    printf("   └────────────────────────────────┴──────────────────────────────┘\n\n");

    printf("3. 性能优势:\n");
    printf("   - 减少分配开销 (10-100x)\n");
    printf("   - 避免 GPU 同步\n");
    printf("   - 更好的内存重用\n");
    printf("   - 与流操作自然集成\n\n");

    printf("4. 注意事项:\n");
    printf("   - 需要 CUDA 11.2+ (内存池)\n");
    printf("   - 需要 CUDA 10.2+ (虚拟内存)\n");
    printf("   - 跨流使用需要显式同步\n");
    printf("   - 注意池内存的生命周期\n\n");

    return 0;
}
