#include <stdio.h>
#include <cuda_runtime.h>

// 这是一个 CUDA 核函数（kernel），用 __global__ 标记
// __global__ 表示这个函数在 GPU 上运行，从 CPU 调用
__global__ void helloFromGPU() {
    // threadIdx: 当前线程在线程块中的索引
    // blockIdx: 当前线程块在网格中的索引
    // blockDim: 线程块的大小
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU! 线程 ID: %d\n", tid);
}

int main() {
    printf("=== CUDA Hello World 程序 ===\n\n");

    // 定义线程组织结构
    int numBlocks = 2;      // 2 个线程块
    int threadsPerBlock = 4; // 每个块 4 个线程

    printf("启动 GPU 核函数...\n");
    printf("- 线程块数量: %d\n", numBlocks);
    printf("- 每块线程数: %d\n", threadsPerBlock);
    printf("- 总线程数: %d\n\n", numBlocks * threadsPerBlock);

    // 启动 CUDA 核函数
    // <<<numBlocks, threadsPerBlock>>> 是 CUDA 的执行配置语法
    helloFromGPU<<<numBlocks, threadsPerBlock>>>();

    // 等待 GPU 完成所有操作
    cudaDeviceSynchronize();

    // 检查是否有错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("\n程序执行完成！\n");
    return 0;
}
