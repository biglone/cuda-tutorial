/**
 * =============================================================================
 * CUDA 教程 06: 线程同步与原子操作
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解为什么需要同步
 * 2. 掌握 __syncthreads() 的使用
 * 3. 学会使用原子操作
 * 4. 了解常见的并行编程陷阱
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
// 示例 1: 为什么需要同步 - 竞态条件演示
// ============================================================================

// 错误示例：没有同步的累加（结果会是错误的！）
__global__ void incrementWrong(int *counter) {
    // 多个线程同时读取、修改、写入同一个值
    // 这会导致数据竞争（race condition）
    int temp = *counter;  // 读取
    temp = temp + 1;      // 修改
    *counter = temp;      // 写入
    // 问题：线程 A 和 B 可能同时读取相同的值！
}

// 正确示例：使用原子操作
__global__ void incrementAtomic(int *counter) {
    // atomicAdd 保证读-改-写是原子的（不可分割的）
    atomicAdd(counter, 1);
}

// ============================================================================
// 示例 2: __syncthreads() 块内同步
// ============================================================================

__global__ void syncDemo(int *data) {
    __shared__ int sharedData[256];
    int tid = threadIdx.x;

    // 阶段 1: 每个线程写入共享内存
    sharedData[tid] = tid * 2;

    // ⚠️ 如果没有这行，后面的读取可能得到错误的值！
    __syncthreads();  // 等待所有线程完成写入

    // 阶段 2: 每个线程读取邻居的值
    if (tid < blockDim.x - 1) {
        data[tid] = sharedData[tid + 1];
    }
}

// ============================================================================
// 示例 3: 常用原子操作
// ============================================================================

__global__ void atomicOperationsDemo(int *results) {
    int tid = threadIdx.x;

    // atomicAdd - 原子加法
    atomicAdd(&results[0], 1);

    // atomicSub - 原子减法
    atomicSub(&results[1], 1);

    // atomicMax - 原子取最大值
    atomicMax(&results[2], tid);

    // atomicMin - 原子取最小值
    atomicMin(&results[3], 100 - tid);

    // atomicExch - 原子交换
    atomicExch(&results[4], tid);  // 最后一个线程的值

    // atomicCAS - 比较并交换 (Compare And Swap)
    // 如果 results[5] == 0，则设为 tid
    atomicCAS(&results[5], 0, tid);
}

// ============================================================================
// 示例 4: 使用原子操作实现直方图
// ============================================================================

__global__ void histogram(int *data, int *hist, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        int bin = data[tid];  // 数据值就是桶索引（简化）
        atomicAdd(&hist[bin], 1);  // 原子增加计数
    }
}

// 使用共享内存优化的直方图
__global__ void histogramShared(int *data, int *hist, int n, int numBins) {
    // 在共享内存中创建局部直方图
    extern __shared__ int localHist[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localTid = threadIdx.x;

    // 初始化局部直方图
    for (int i = localTid; i < numBins; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // 在共享内存中累加（减少全局内存原子操作）
    if (tid < n) {
        int bin = data[tid];
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    // 将局部结果合并到全局直方图
    for (int i = localTid; i < numBins; i += blockDim.x) {
        atomicAdd(&hist[i], localHist[i]);
    }
}

// ============================================================================
// 示例 5: 使用原子操作查找最大值及其索引
// ============================================================================

__global__ void findMaxWithIndex(float *data, float *maxVal, int *maxIdx, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        float val = data[tid];
        // 注意：这个实现可能有问题，因为 maxVal 和 maxIdx 的更新不是原子的
        // 这里仅作演示，实际应使用归约算法
        float old = atomicMax((int*)maxVal, __float_as_int(val));
        if (__float_as_int(val) > __int_as_float(old)) {
            *maxIdx = tid;  // 这不是原子的！
        }
    }
}

// 正确的做法：使用归约
__global__ void findMaxReduce(float *data, float *blockMax, int *blockMaxIdx, int n) {
    __shared__ float s_data[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // 加载数据
    if (tid < n) {
        s_data[localIdx] = data[tid];
        s_idx[localIdx] = tid;
    } else {
        s_data[localIdx] = -1e38f;
        s_idx[localIdx] = -1;
    }
    __syncthreads();

    // 归约找最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            if (s_data[localIdx + stride] > s_data[localIdx]) {
                s_data[localIdx] = s_data[localIdx + stride];
                s_idx[localIdx] = s_idx[localIdx + stride];
            }
        }
        __syncthreads();
    }

    // 保存块的最大值
    if (localIdx == 0) {
        blockMax[blockIdx.x] = s_data[0];
        blockMaxIdx[blockIdx.x] = s_idx[0];
    }
}

// ============================================================================
// 主函数 - 演示
// ============================================================================

void demoRaceCondition() {
    printf("=== 示例 1: 竞态条件演示 ===\n\n");

    int h_counter = 0;
    int *d_counter;
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(int)));

    const int numThreads = 10000;

    // 测试错误版本
    CHECK_CUDA(cudaMemcpy(d_counter, &h_counter, sizeof(int),
                          cudaMemcpyHostToDevice));
    incrementWrong<<<numThreads / 256, 256>>>(d_counter);
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int),
                          cudaMemcpyDeviceToHost));
    printf("错误版本结果: %d (期望: %d)\n", h_counter, numThreads);

    // 测试正确版本
    h_counter = 0;
    CHECK_CUDA(cudaMemcpy(d_counter, &h_counter, sizeof(int),
                          cudaMemcpyHostToDevice));
    incrementAtomic<<<numThreads / 256, 256>>>(d_counter);
    CHECK_CUDA(cudaMemcpy(&h_counter, d_counter, sizeof(int),
                          cudaMemcpyDeviceToHost));
    printf("原子版本结果: %d (期望: %d)\n\n", h_counter, numThreads);

    CHECK_CUDA(cudaFree(d_counter));
}

void demoAtomicOperations() {
    printf("=== 示例 3: 原子操作演示 ===\n\n");

    // results[0]: atomicAdd 起始值 0
    // results[1]: atomicSub 起始值 1000
    // results[2]: atomicMax 起始值 0
    // results[3]: atomicMin 起始值 1000
    // results[4]: atomicExch 起始值 0
    // results[5]: atomicCAS 起始值 0
    int h_results[6] = {0, 1000, 0, 1000, 0, 0};
    int *d_results;

    CHECK_CUDA(cudaMalloc(&d_results, 6 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_results, h_results, 6 * sizeof(int),
                          cudaMemcpyHostToDevice));

    int numThreads = 100;
    atomicOperationsDemo<<<1, numThreads>>>(d_results);

    CHECK_CUDA(cudaMemcpy(h_results, d_results, 6 * sizeof(int),
                          cudaMemcpyDeviceToHost));

    printf("使用 %d 个线程:\n", numThreads);
    printf("atomicAdd (0 起始):     %d (每线程+1)\n", h_results[0]);
    printf("atomicSub (1000 起始):  %d (每线程-1)\n", h_results[1]);
    printf("atomicMax (0 起始):     %d (最大线程ID)\n", h_results[2]);
    printf("atomicMin (1000 起始):  %d (最小 100-tid)\n", h_results[3]);
    printf("atomicExch:             %d (某个线程ID)\n", h_results[4]);
    printf("atomicCAS (0→tid):      %d (第一个执行的线程ID)\n\n", h_results[5]);

    CHECK_CUDA(cudaFree(d_results));
}

void demoHistogram() {
    printf("=== 示例 4: 直方图计算 ===\n\n");

    const int N = 10000;
    const int NUM_BINS = 10;

    // 创建测试数据（0-9 的随机值）
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_hist = (int*)malloc(NUM_BINS * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_data[i] = i % NUM_BINS;  // 简单起见，均匀分布
    }
    memset(h_hist, 0, NUM_BINS * sizeof(int));

    int *d_data, *d_hist;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    histogram<<<blocks, threadsPerBlock>>>(d_data, d_hist, N);

    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    printf("直方图结果 (共 %d 个元素):\n", N);
    for (int i = 0; i < NUM_BINS; i++) {
        printf("  桶 %d: %d\n", i, h_hist[i]);
    }
    printf("\n");

    free(h_data);
    free(h_hist);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_hist));
}

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 06: 线程同步与原子操作                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    demoRaceCondition();
    demoAtomicOperations();
    demoHistogram();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 同步原因:\n");
    printf("   - GPU 线程并行执行，顺序不确定\n");
    printf("   - 共享数据需要同步避免竞态条件\n\n");

    printf("2. __syncthreads():\n");
    printf("   - 只能同步同一块内的线程\n");
    printf("   - 所有线程必须都执行到这里\n");
    printf("   - 不能放在条件分支内（可能死锁）\n\n");

    printf("3. 原子操作:\n");
    printf("   - atomicAdd/Sub/Max/Min/And/Or/Xor\n");
    printf("   - atomicExch - 原子交换\n");
    printf("   - atomicCAS - 比较并交换\n");
    printf("   - 比非原子操作慢，尽量减少使用\n\n");

    printf("4. 优化策略:\n");
    printf("   - 先在共享内存中局部累加\n");
    printf("   - 再用原子操作合并到全局\n");
    printf("   - 减少全局内存的原子操作次数\n\n");

    return 0;
}
