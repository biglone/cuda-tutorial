/**
 * =============================================================================
 * CUDA 教程 10: 常量内存与归约操作
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解常量内存的特性和使用方法
 * 2. 掌握并行归约（Reduction）算法
 * 3. 学习归约操作的多种优化技术
 * 4. 了解 warp 级别原语的使用
 *
 * 关键概念：
 * - 常量内存：只读、带缓存、广播到所有线程
 * - 归约：将大量数据合并为单一结果（求和、最大值等）
 * - Warp 级原语：__shfl_down_sync 等高效通信
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define WARP_SIZE 32

// ============================================================================
// 第一部分：常量内存
// ============================================================================

// 声明常量内存（必须在文件作用域）
__constant__ float d_filterKernel[25];  // 5x5 卷积核
__constant__ float d_coefficients[16];   // 多项式系数

// 使用常量内存的核函数
__global__ void applyFilter(float *input, float *output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        float sum = 0.0f;
        // 所有线程读取相同的滤波器系数 -> 常量内存广播
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int kidx = (ky + 2) * 5 + (kx + 2);
                sum += input[(y + ky) * width + (x + kx)] * d_filterKernel[kidx];
            }
        }
        output[y * width + x] = sum;
    }
}

// 使用全局内存的对比版本
__global__ void applyFilterGlobal(float *input, float *output,
                                   float *filterKernel, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int kidx = (ky + 2) * 5 + (kx + 2);
                sum += input[(y + ky) * width + (x + kx)] * filterKernel[kidx];
            }
        }
        output[y * width + x] = sum;
    }
}

void demoConstantMemory() {
    printf("=== 第一部分：常量内存 ===\n\n");

    printf("常量内存特性:\n");
    printf("  - 总大小: 64 KB\n");
    printf("  - 只读（从内核角度）\n");
    printf("  - 带缓存，同一 warp 读取相同地址时广播\n");
    printf("  - 特别适合：所有线程读取相同数据\n\n");

    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int size = WIDTH * HEIGHT * sizeof(float);

    // 准备滤波器（高斯模糊）
    float h_filter[25];
    float sigma = 1.0f;
    float sum = 0.0f;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            int idx = (y + 2) * 5 + (x + 2);
            h_filter[idx] = expf(-(x*x + y*y) / (2*sigma*sigma));
            sum += h_filter[idx];
        }
    }
    // 归一化
    for (int i = 0; i < 25; i++) h_filter[i] /= sum;

    // 复制到常量内存
    CHECK_CUDA(cudaMemcpyToSymbol(d_filterKernel, h_filter, 25 * sizeof(float)));

    // 也复制到全局内存用于对比
    float *d_filterGlobal;
    CHECK_CUDA(cudaMalloc(&d_filterGlobal, 25 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_filterGlobal, h_filter, 25 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 准备输入输出
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = (float)(rand() % 256) / 255.0f;
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // 测试常量内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        applyFilter<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float constTime;
    CHECK_CUDA(cudaEventElapsedTime(&constTime, start, stop));
    printf("常量内存版本 (100次): %.3f ms\n", constTime);

    // 测试全局内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        applyFilterGlobal<<<gridDim, blockDim>>>(d_input, d_output,
                                                  d_filterGlobal, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float globalTime;
    CHECK_CUDA(cudaEventElapsedTime(&globalTime, start, stop));
    printf("全局内存版本 (100次): %.3f ms\n", globalTime);
    printf("常量内存加速比: %.2fx\n\n", globalTime / constTime);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_filterGlobal));
    free(h_input);
}

// ============================================================================
// 第二部分：基础归约操作
// ============================================================================

// 版本1：朴素归约（有 warp 分歧）
__global__ void reduceNaive(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 朴素归约 - 有分歧问题
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 版本2：消除 warp 分歧
__global__ void reduceInterleaved(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 交错寻址 - 连续线程活跃
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 版本3：首次加载时归约
__global__ void reduceFirstAdd(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 首次加载时就做一次加法
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 版本4：使用 warp 展开
__device__ void warpReduce(volatile float *sdata, int tid) {
    // 最后一个 warp 不需要 __syncthreads()
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceWarpUnroll(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 归约到 32 个元素
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最后一个 warp 使用展开
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 版本5：使用 warp shuffle
__global__ void reduceWarpShuffle(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 先归约到每个 warp
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 使用 warp shuffle 完成最后的归约
    if (tid < 32) {
        float val = sdata[tid];
        // __shfl_down_sync 将值从高 lane 移到低 lane
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) output[blockIdx.x] = val;
    }
}

void demoReduction() {
    printf("=== 第二部分：归约操作优化 ===\n\n");

    const int N = 1 << 24;  // 16M 元素
    const int BLOCK_SIZE = 256;

    printf("问题规模: %d 元素 (%d MB)\n\n", N, N * 4 / (1024*1024));

    // 分配内存
    float *h_input = (float*)malloc(N * sizeof(float));
    double expectedSum = 0.0;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // 简单起见，全部为1
        expectedSum += h_input[i];
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&d_output, numBlocks * sizeof(float)));

    float *h_output = (float*)malloc(numBlocks * sizeof(float));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测试各版本
    struct {
        const char *name;
        void (*kernel)(float*, float*, int);
        int gridDivider;
    } versions[] = {
        {"1. 朴素归约 (有warp分歧)", reduceNaive, 1},
        {"2. 交错寻址 (消除分歧)", reduceInterleaved, 1},
        {"3. 首次加载归约", reduceFirstAdd, 2},
        {"4. Warp 展开", reduceWarpUnroll, 2},
        {"5. Warp Shuffle", reduceWarpShuffle, 2},
    };

    int numVersions = sizeof(versions) / sizeof(versions[0]);

    for (int v = 0; v < numVersions; v++) {
        int gridSize = numBlocks / versions[v].gridDivider;
        int sharedSize = BLOCK_SIZE * sizeof(float);

        // 预热
        versions[v].kernel<<<gridSize, BLOCK_SIZE, sharedSize>>>(d_input, d_output, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < 100; iter++) {
            versions[v].kernel<<<gridSize, BLOCK_SIZE, sharedSize>>>(d_input, d_output, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time;
        CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));

        // 完成 CPU 端归约
        CHECK_CUDA(cudaMemcpy(h_output, d_output, gridSize * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double gpuSum = 0.0;
        for (int i = 0; i < gridSize; i++) gpuSum += h_output[i];

        float bandwidth = (N * sizeof(float)) / (time/100 * 1e6);  // GB/s
        printf("%s\n", versions[v].name);
        printf("   时间: %.3f ms, 带宽: %.1f GB/s, 结果: %.0f\n\n",
               time/100, bandwidth, gpuSum);
    }

    printf("期望结果: %.0f\n\n", expectedSum);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

// ============================================================================
// 第三部分：其他归约操作
// ============================================================================

// 求最大值归约
__global__ void reduceMax(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float maxVal = -FLT_MAX;
    if (i < n) maxVal = fmaxf(maxVal, input[i]);
    if (i + blockDim.x < n) maxVal = fmaxf(maxVal, input[i + blockDim.x]);
    sdata[tid] = maxVal;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 求最小值归约
__global__ void reduceMin(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float minVal = FLT_MAX;
    if (i < n) minVal = fminf(minVal, input[i]);
    if (i + blockDim.x < n) minVal = fminf(minVal, input[i + blockDim.x]);
    sdata[tid] = minVal;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 点积归约
__global__ void reduceDot(float *a, float *b, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n) sum += a[i] * b[i];
    if (i + blockDim.x < n) sum += a[i + blockDim.x] * b[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void demoOtherReductions() {
    printf("=== 第三部分：其他归约操作 ===\n\n");

    const int N = 1 << 20;
    const int BLOCK_SIZE = 256;

    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));

    // 初始化测试数据
    float cpuMax = -FLT_MAX, cpuMin = FLT_MAX;
    double cpuDot = 0.0;
    for (int i = 0; i < N; i++) {
        h_a[i] = sinf((float)i * 0.001f);
        h_b[i] = cosf((float)i * 0.001f);
        cpuMax = fmaxf(cpuMax, h_a[i]);
        cpuMin = fminf(cpuMin, h_a[i]);
        cpuDot += h_a[i] * h_b[i];
    }

    float *d_a, *d_b, *d_output;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    CHECK_CUDA(cudaMalloc(&d_output, numBlocks * sizeof(float)));
    float *h_output = (float*)malloc(numBlocks * sizeof(float));

    int sharedSize = BLOCK_SIZE * sizeof(float);

    // 求最大值
    reduceMax<<<numBlocks, BLOCK_SIZE, sharedSize>>>(d_a, d_output, N);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpuMax = -FLT_MAX;
    for (int i = 0; i < numBlocks; i++) gpuMax = fmaxf(gpuMax, h_output[i]);
    printf("最大值: GPU = %.6f, CPU = %.6f\n", gpuMax, cpuMax);

    // 求最小值
    reduceMin<<<numBlocks, BLOCK_SIZE, sharedSize>>>(d_a, d_output, N);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpuMin = FLT_MAX;
    for (int i = 0; i < numBlocks; i++) gpuMin = fminf(gpuMin, h_output[i]);
    printf("最小值: GPU = %.6f, CPU = %.6f\n", gpuMin, cpuMin);

    // 点积
    reduceDot<<<numBlocks, BLOCK_SIZE, sharedSize>>>(d_a, d_b, d_output, N);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double gpuDot = 0.0;
    for (int i = 0; i < numBlocks; i++) gpuDot += h_output[i];
    printf("点积:   GPU = %.6f, CPU = %.6f\n\n", gpuDot, cpuDot);

    // 清理
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_output));
    free(h_a);
    free(h_b);
    free(h_output);
}

// ============================================================================
// 第四部分：完整的单次调用归约
// ============================================================================

// 使用原子操作的完整归约
__global__ void reduceAtomicSum(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 加载和首次归约
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 块内归约
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp shuffle 归约
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        // 使用原子加法将块结果累加
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

void demoAtomicReduction() {
    printf("=== 第四部分：原子操作归约 ===\n\n");

    const int N = 1 << 24;
    const int BLOCK_SIZE = 256;

    float *h_input = (float*)malloc(N * sizeof(float));
    double expectedSum = 0.0;
    for (int i = 0; i < N; i++) {
        h_input[i] = 0.001f;  // 小数避免溢出
        expectedSum += h_input[i];
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float zero = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start));
    reduceAtomicSum<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_input, d_output, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));

    float gpuSum;
    CHECK_CUDA(cudaMemcpy(&gpuSum, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("原子操作归约:\n");
    printf("  单次内核调用完成全部归约\n");
    printf("  GPU 结果: %.2f\n", gpuSum);
    printf("  期望结果: %.2f\n", expectedSum);
    printf("  耗时: %.3f ms\n", time);
    printf("  带宽: %.1f GB/s\n\n", (N * sizeof(float)) / (time * 1e6));

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 10: 常量内存与归约操作                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("常量内存: %zu KB\n", prop.totalConstMem / 1024);
    printf("Warp 大小: %d\n\n", prop.warpSize);

    demoConstantMemory();
    demoReduction();
    demoOtherReductions();
    demoAtomicReduction();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 常量内存:\n");
    printf("   - __constant__ 声明（文件作用域）\n");
    printf("   - cudaMemcpyToSymbol() 复制数据\n");
    printf("   - 64KB 限制，只读\n");
    printf("   - 所有线程读相同地址时最高效\n\n");

    printf("2. 归约优化技术:\n");
    printf("   ┌────────────────────────────────────────────┐\n");
    printf("   │ 优化技术              │ 效果               │\n");
    printf("   ├────────────────────────────────────────────┤\n");
    printf("   │ 交错寻址              │ 消除 warp 分歧     │\n");
    printf("   │ 首次加载归约          │ 减少全局内存访问   │\n");
    printf("   │ Warp 展开             │ 避免最后几轮同步   │\n");
    printf("   │ Warp Shuffle          │ 寄存器级通信       │\n");
    printf("   │ 原子操作              │ 单次调用完成归约   │\n");
    printf("   └────────────────────────────────────────────┘\n\n");

    printf("3. Warp Shuffle 原语:\n");
    printf("   - __shfl_sync()      - 任意 lane 交换\n");
    printf("   - __shfl_up_sync()   - 向上移动\n");
    printf("   - __shfl_down_sync() - 向下移动\n");
    printf("   - __shfl_xor_sync()  - XOR 交换\n\n");

    printf("4. 归约操作类型:\n");
    printf("   - 求和 (Sum)\n");
    printf("   - 最大/最小值 (Max/Min)\n");
    printf("   - 点积 (Dot Product)\n");
    printf("   - 计数/均值 (Count/Average)\n");
    printf("   - 自定义二元操作\n\n");

    printf("5. 性能关键点:\n");
    printf("   - 归约是内存带宽受限操作\n");
    printf("   - 理论峰值 = 数据大小 / 内存带宽\n");
    printf("   - 实际性能受同步和分歧影响\n\n");

    return 0;
}
