#include <stdio.h>
#include <cuda_runtime.h>

// __device__ 函数：只能在 GPU 上调用，被其他 GPU 函数调用
__device__ int square(int x) {
    return x * x;
}

// 一维核函数示例
__global__ void kernel1D() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int squared = square(tid);
    printf("1D - 线程 %d: %d 的平方 = %d\n", tid, tid, squared);
}

// 二维核函数示例
__global__ void kernel2D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    printf("2D - 位置 (%d, %d)\n", x, y);
}

// 带参数的核函数
__global__ void kernelWithParams(int multiplier, float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 演示参数使用
    printf("线程 %d: multiplier = %d\n", tid, multiplier);
}

int main() {
    printf("=== CUDA 核函数基础学习 ===\n\n");

    // === 示例 1: 一维核函数 ===
    printf("【示例 1】一维核函数（1D Grid）\n");
    printf("-------------------------------\n");
    dim3 blocks1D(2);        // 2 个块
    dim3 threads1D(3);       // 每块 3 个线程
    kernel1D<<<blocks1D, threads1D>>>();
    cudaDeviceSynchronize();

    printf("\n");

    // === 示例 2: 二维核函数 ===
    printf("【示例 2】二维核函数（2D Grid）\n");
    printf("-------------------------------\n");
    dim3 blocks2D(2, 2);     // 2x2 = 4 个块
    dim3 threads2D(2, 2);    // 每块 2x2 = 4 个线程
    printf("配置: %dx%d 块, 每块 %dx%d 线程\n",
           blocks2D.x, blocks2D.y, threads2D.x, threads2D.y);
    kernel2D<<<blocks2D, threads2D>>>();
    cudaDeviceSynchronize();

    printf("\n");

    // === 示例 3: 带参数的核函数 ===
    printf("【示例 3】带参数的核函数\n");
    printf("-------------------------------\n");
    int multiplier = 10;
    kernelWithParams<<<1, 4>>>(multiplier, nullptr);
    cudaDeviceSynchronize();

    // 检查错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\nCUDA 错误: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("\n=== 学习要点总结 ===\n");
    printf("1. __global__: GPU 核函数，从 CPU 调用\n");
    printf("2. __device__: GPU 设备函数，只能从 GPU 调用\n");
    printf("3. dim3: 用于定义 1D/2D/3D 线程组织\n");
    printf("4. 核函数可以接收参数（值、指针等）\n");
    printf("5. 必须用 cudaDeviceSynchronize() 等待 GPU 完成\n");

    return 0;
}
