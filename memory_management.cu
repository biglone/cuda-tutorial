#include <stdio.h>
#include <cuda_runtime.h>

// 核函数：将数组每个元素乘以 2
__global__ void doubleArray(int *d_array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 确保不越界
    if (tid < size) {
        d_array[tid] = d_array[tid] * 2;
    }
}

// 工具函数：检查 CUDA 错误
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

int main() {
    printf("=== CUDA 内存管理教程 ===\n\n");

    // 1. 定义数据大小
    const int N = 10;
    const int size = N * sizeof(int);

    printf("【步骤 1】在主机（CPU）上分配和初始化数据\n");
    printf("----------------------------------------\n");

    // 主机内存分配
    int *h_array = (int*)malloc(size);

    // 初始化数据
    printf("初始数组: ");
    for (int i = 0; i < N; i++) {
        h_array[i] = i + 1;
        printf("%d ", h_array[i]);
    }
    printf("\n\n");

    // 2. 在设备（GPU）上分配内存
    printf("【步骤 2】在设备（GPU）上分配内存\n");
    printf("----------------------------------------\n");
    int *d_array;
    CHECK_CUDA(cudaMalloc((void**)&d_array, size));
    printf("✓ 在 GPU 上分配了 %d 字节内存\n", size);
    printf("  主机地址: %p\n", h_array);
    printf("  设备地址: %p\n\n", d_array);

    // 3. 将数据从主机复制到设备
    printf("【步骤 3】主机 → 设备 数据传输\n");
    printf("----------------------------------------\n");
    CHECK_CUDA(cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));
    printf("✓ 已将 %d 个整数从 CPU 复制到 GPU\n\n", N);

    // 4. 在 GPU 上执行核函数
    printf("【步骤 4】在 GPU 上执行计算\n");
    printf("----------------------------------------\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("启动配置: %d 块 x %d 线程\n", blocksPerGrid, threadsPerBlock);

    doubleArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("✓ GPU 计算完成\n\n");

    // 5. 将结果从设备复制回主机
    printf("【步骤 5】设备 → 主机 数据传输\n");
    printf("----------------------------------------\n");
    CHECK_CUDA(cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost));
    printf("✓ 已将结果从 GPU 复制回 CPU\n\n");

    // 6. 显示结果
    printf("【步骤 6】查看结果\n");
    printf("----------------------------------------\n");
    printf("处理后数组: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");
    printf("✓ 每个元素都乘以了 2\n\n");

    // 7. 释放内存
    printf("【步骤 7】释放内存\n");
    printf("----------------------------------------\n");
    CHECK_CUDA(cudaFree(d_array));
    free(h_array);
    printf("✓ GPU 和 CPU 内存已释放\n\n");

    // 内存管理总结
    printf("=== 内存管理关键函数 ===\n");
    printf("1. cudaMalloc()       - 在 GPU 上分配内存\n");
    printf("2. cudaMemcpy()       - CPU ↔ GPU 数据传输\n");
    printf("   • HostToDevice     - CPU → GPU\n");
    printf("   • DeviceToHost     - GPU → CPU\n");
    printf("   • DeviceToDevice   - GPU → GPU\n");
    printf("3. cudaFree()         - 释放 GPU 内存\n");
    printf("4. cudaDeviceSynchronize() - 等待 GPU 完成\n\n");

    printf("⚠️  注意事项:\n");
    printf("   • 主机不能直接访问设备内存\n");
    printf("   • 设备不能直接访问主机内存\n");
    printf("   • 数据传输有开销，需要优化\n");

    return 0;
}
