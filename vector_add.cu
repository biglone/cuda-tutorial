#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA 错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// CPU 版本：向量加法
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU 核函数：向量加法
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    // 计算全局线程 ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 边界检查
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// 验证结果是否正确
bool verifyResults(float *cpu_result, float *gpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            printf("错误：索引 %d 处结果不匹配！CPU: %f, GPU: %f\n",
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== CUDA 向量加法：完整实践 ===\n\n");

    // 向量大小（100万个元素）
    const int N = 1000000;
    const int size = N * sizeof(float);

    printf("问题规模: %d 个元素 (%.2f MB)\n\n", N, size / 1024.0 / 1024.0);

    // ============ 1. 内存分配 ============
    printf("【1】分配内存\n");
    printf("------------------------\n");

    // 主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);  // CPU 结果
    float *h_c_gpu = (float*)malloc(size);  // GPU 结果

    // 设备内存
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    printf("✓ 主机和设备内存分配完成\n\n");

    // ============ 2. 初始化数据 ============
    printf("【2】初始化数据\n");
    printf("------------------------\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    printf("✓ 向量 A: [0, 1, 2, ..., %d]\n", N-1);
    printf("✓ 向量 B: [0, 2, 4, ..., %d]\n\n", (N-1)*2);

    // ============ 3. CPU 计算 ============
    printf("【3】CPU 计算\n");
    printf("------------------------\n");
    clock_t start_cpu = clock();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    clock_t end_cpu = clock();
    double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("✓ CPU 耗时: %.3f ms\n\n", time_cpu);

    // ============ 4. GPU 计算 ============
    printf("【4】GPU 计算\n");
    printf("------------------------\n");

    // 数据传输：主机 → 设备
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA(cudaEventCreate(&start_gpu));
    CHECK_CUDA(cudaEventCreate(&stop_gpu));

    CHECK_CUDA(cudaEventRecord(start_gpu));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 配置并启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("  启动配置: %d 块 x %d 线程 = %d 总线程\n",
           blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    // 数据传输：设备 → 主机
    CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop_gpu));
    CHECK_CUDA(cudaEventSynchronize(stop_gpu));

    float time_gpu;
    CHECK_CUDA(cudaEventElapsedTime(&time_gpu, start_gpu, stop_gpu));
    printf("✓ GPU 总耗时: %.3f ms\n\n", time_gpu);

    // ============ 5. 验证结果 ============
    printf("【5】验证结果\n");
    printf("------------------------\n");
    if (verifyResults(h_c_cpu, h_c_gpu, N)) {
        printf("✓ 验证通过！CPU 和 GPU 结果一致\n");
        printf("  示例: %g + %g = %g\n", h_a[0], h_b[0], h_c_gpu[0]);
        printf("  示例: %g + %g = %g\n", h_a[100], h_b[100], h_c_gpu[100]);
    } else {
        printf("✗ 验证失败！\n");
    }
    printf("\n");

    // ============ 6. 性能对比 ============
    printf("【6】性能对比\n");
    printf("------------------------\n");
    printf("CPU 时间: %.3f ms\n", time_cpu);
    printf("GPU 时间: %.3f ms\n", time_gpu);
    printf("加速比:   %.2fx\n\n", time_cpu / time_gpu);

    if (time_cpu > time_gpu) {
        printf("🚀 GPU 比 CPU 快 %.2fx！\n\n", time_cpu / time_gpu);
    } else {
        printf("⚠️  注意：对于小数据集，GPU 开销可能超过收益\n\n");
    }

    // ============ 7. 清理 ============
    printf("【7】清理资源\n");
    printf("------------------------\n");
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CHECK_CUDA(cudaEventDestroy(start_gpu));
    CHECK_CUDA(cudaEventDestroy(stop_gpu));
    printf("✓ 所有资源已释放\n\n");

    printf("=== 学习总结 ===\n");
    printf("✓ 理解了完整的 CUDA 编程流程\n");
    printf("✓ 掌握了内存管理和数据传输\n");
    printf("✓ 学会了配置和启动核函数\n");
    printf("✓ 了解了 GPU 并行计算的优势\n");
    printf("✓ 学会了性能测量和结果验证\n");

    return 0;
}
