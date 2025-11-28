/**
 * =============================================================================
 * CUDA 教程 05: 共享内存 (Shared Memory)
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解 GPU 内存层次结构
 * 2. 学会使用共享内存优化性能
 * 3. 实现数组归约求和
 *
 * 内存层次（从快到慢）：
 * - 寄存器 (Registers) - 最快，每个线程私有
 * - 共享内存 (Shared Memory) - 很快，块内线程共享
 * - 全局内存 (Global Memory) - 较慢，所有线程可访问
 * - 常量内存 (Constant Memory) - 只读，有缓存
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 示例 1: 共享内存基础 - 数组反转
// ============================================================================

// 不使用共享内存的版本（效率低）
__global__ void reverseArrayGlobal(int *d_in, int *d_out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 直接从全局内存读写 - 访问模式不连续
        d_out[n - 1 - tid] = d_in[tid];
    }
}

// 使用共享内存的版本（效率高）
__global__ void reverseArrayShared(int *d_in, int *d_out, int n) {
    // 声明共享内存数组 - 块内所有线程共享
    // __shared__ 表示这块内存在共享内存中
    extern __shared__ int s_data[];  // 动态大小

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // 步骤 1: 从全局内存读取到共享内存（合并访问）
    if (tid < n) {
        s_data[localIdx] = d_in[tid];
    }

    // 步骤 2: 同步 - 确保所有线程都完成读取
    __syncthreads();

    // 步骤 3: 从共享内存反向写回全局内存
    if (tid < n) {
        // 计算块内的反向索引
        int reversedLocalIdx = blockDim.x - 1 - localIdx;
        // 计算全局的目标位置
        int targetBlock = gridDim.x - 1 - blockIdx.x;
        int targetIdx = targetBlock * blockDim.x + reversedLocalIdx;

        if (targetIdx < n) {
            d_out[targetIdx] = s_data[localIdx];
        }
    }
}

// ============================================================================
// 示例 2: 归约求和 - 共享内存的经典应用
// ============================================================================

// CPU 版本：简单求和
float sumArrayCPU(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// GPU 版本：使用共享内存的并行归约
__global__ void reduceSum(float *d_in, float *d_out, int n) {
    // 静态共享内存（大小在编译时确定）
    __shared__ float s_data[256];  // 假设每块最多 256 线程

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // 步骤 1: 加载数据到共享内存
    s_data[localIdx] = (tid < n) ? d_in[tid] : 0.0f;
    __syncthreads();

    // 步骤 2: 并行归约
    // 每次迭代，活跃线程数减半
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            s_data[localIdx] += s_data[localIdx + stride];
        }
        __syncthreads();  // 每次迭代后同步
    }

    // 步骤 3: 将每个块的结果写回全局内存
    if (localIdx == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// ============================================================================
// 示例 3: 共享内存避免 bank conflict
// ============================================================================

#define TILE_SIZE 16

// 矩阵转置 - 朴素版本（有 bank conflict）
__global__ void transposeNaive(float *d_in, float *d_out, int width, int height) {
    int x = threadIdx.x + blockIdx.x * TILE_SIZE;
    int y = threadIdx.y + blockIdx.y * TILE_SIZE;

    if (x < width && y < height) {
        d_out[x * height + y] = d_in[y * width + x];
    }
}

// 矩阵转置 - 使用共享内存优化
__global__ void transposeShared(float *d_in, float *d_out, int width, int height) {
    // 添加 padding 避免 bank conflict
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 是关键！

    int x = threadIdx.x + blockIdx.x * TILE_SIZE;
    int y = threadIdx.y + blockIdx.y * TILE_SIZE;

    // 读取到共享内存（合并访问）
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = d_in[y * width + x];
    }
    __syncthreads();

    // 计算转置后的位置
    x = threadIdx.x + blockIdx.y * TILE_SIZE;
    y = threadIdx.y + blockIdx.x * TILE_SIZE;

    // 从共享内存写回（合并访问）
    if (x < height && y < width) {
        d_out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// 主函数 - 演示所有示例
// ============================================================================

void demoArrayReverse() {
    printf("=== 示例 1: 数组反转（共享内存 vs 全局内存）===\n\n");

    const int N = 16;
    int size = N * sizeof(int);

    // 主机内存
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i;

    printf("输入数组: ");
    for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
    printf("\n");

    // 设备内存
    int *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // 启动核函数 - 使用动态共享内存
    int threadsPerBlock = N;
    int sharedMemSize = N * sizeof(int);
    reverseArrayShared<<<1, threadsPerBlock, sharedMemSize>>>(d_in, d_out, N);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    printf("反转结果: ");
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);
    printf("\n\n");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

void demoReduceSum() {
    printf("=== 示例 2: 归约求和（并行算法）===\n\n");

    const int N = 1024;
    const int size = N * sizeof(float);

    // 主机内存
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;  // 全是 1，和应该是 N

    // CPU 求和
    float cpuSum = sumArrayCPU(h_data, N);
    printf("CPU 求和结果: %.0f\n", cpuSum);

    // 设备内存
    float *d_in, *d_partial;
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice));

    // 第一轮归约
    reduceSum<<<blocks, threadsPerBlock>>>(d_in, d_partial, N);

    // 将部分和复制回主机，完成最终求和
    float *h_partial = (float*)malloc(blocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpuSum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        gpuSum += h_partial[i];
    }
    printf("GPU 求和结果: %.0f\n", gpuSum);

    printf("\n归约过程说明:\n");
    printf("- 输入 %d 个元素\n", N);
    printf("- 分成 %d 个块，每块 %d 线程\n", blocks, threadsPerBlock);
    printf("- 每个块产生 1 个部分和\n");
    printf("- CPU 对 %d 个部分和做最终求和\n\n", blocks);

    free(h_data);
    free(h_partial);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_partial));
}

void printGPUMemoryInfo() {
    printf("=== GPU 共享内存信息 ===\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("设备名称: %s\n", prop.name);
    printf("每块最大共享内存: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("每个SM共享内存: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("每块最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("Warp 大小: %d\n", prop.warpSize);
    printf("SM 数量: %d\n\n", prop.multiProcessorCount);
}

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 05: 共享内存 (Shared Memory)                 ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printGPUMemoryInfo();
    demoArrayReverse();
    demoReduceSum();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 共享内存是块内线程共享的快速内存\n");
    printf("2. 使用 __shared__ 关键字声明\n");
    printf("3. 静态大小: __shared__ float arr[256];\n");
    printf("4. 动态大小: extern __shared__ float arr[];\n");
    printf("   - 启动时指定: kernel<<<blocks, threads, sharedSize>>>()\n");
    printf("5. 使用 __syncthreads() 同步块内线程\n");
    printf("6. 避免 bank conflict: 添加 padding\n");
    printf("7. 共享内存适合:\n");
    printf("   - 数据重用\n");
    printf("   - 线程间通信\n");
    printf("   - 归约操作\n");
    printf("   - 矩阵分块计算\n\n");

    return 0;
}
