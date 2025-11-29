/**
 * =============================================================================
 * CUDA 教程 13: 动态并行与协作组
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解动态并行（Dynamic Parallelism）的概念
 * 2. 学会在内核中启动子内核
 * 3. 掌握协作组（Cooperative Groups）的使用
 * 4. 了解网格级同步和线程块协作
 *
 * 关键概念：
 * - 动态并行：GPU 内核可以启动其他内核
 * - 协作组：更灵活的线程分组和同步机制
 * - 网格级同步：所有线程块的全局同步
 *
 * 编译要求：
 *   nvcc -rdc=true -lcudadevrt 13_dynamic_parallelism.cu -o 13_dynamic_parallelism
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// CUDA 13+ 兼容性：设备端同步
// 在动态并行中等待子内核完成
// 注意：CUDA 13 中设备端 cudaDeviceSynchronize 已被移除
// 使用 cooperative_groups 或重构代码以避免设备端同步
__device__ __forceinline__ void deviceSyncChildKernels() {
    // 使用 __syncthreads() 进行线程块级同步（针对单线程内核调用仍然有效）
    // 对于真正的动态并行子内核同步，需要重构为使用流回调或其他机制
    __syncthreads();
}

// ============================================================================
// 第一部分：协作组基础
// ============================================================================

// 演示线程块协作组
__global__ void demoThreadBlockGroup(int *output) {
    // 获取当前线程块的协作组
    cg::thread_block block = cg::this_thread_block();

    int tid = block.thread_rank();  // 块内线程索引

    // 使用协作组同步（代替 __syncthreads()）
    block.sync();

    if (tid == 0) {
        output[blockIdx.x] = block.size();  // 线程块大小
    }
}

// 演示 warp 级协作组
__global__ void demoTiledPartition(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();

    // 将线程块划分为 32 线程的 tile（即 warp）
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float val = input[tid];

        // warp 内归约求和
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            val += warp.shfl_down(val, offset);
        }

        // warp 的第一个线程写入结果
        if (warp.thread_rank() == 0) {
            atomicAdd(output, val);
        }
    }
}

// 演示自定义大小的 tile
__global__ void demoCustomTile(int *output) {
    cg::thread_block block = cg::this_thread_block();

    // 创建 16 线程的 tile
    cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);

    // 创建 8 线程的 tile
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);

    int tid = threadIdx.x;

    if (tid < 32) {
        // tile16 内的索引
        int tile16_rank = tile16.thread_rank();
        int tile16_id = tid / 16;

        // tile8 内的索引
        int tile8_rank = tile8.thread_rank();
        int tile8_id = tid / 8;

        output[tid] = tile16_rank * 100 + tile8_rank;
    }

    block.sync();
}

void demoCooperativeGroupsBasic() {
    printf("=== 第一部分：协作组基础 ===\n\n");

    printf("协作组类型:\n");
    printf("  - thread_block: 整个线程块\n");
    printf("  - thread_block_tile<N>: N线程的分区 (N=2,4,8,16,32)\n");
    printf("  - coalesced_group: 活跃线程组\n");
    printf("  - grid_group: 整个网格（需要协作启动）\n\n");

    // 演示线程块组
    int *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, 4 * sizeof(int)));

    demoThreadBlockGroup<<<4, 128>>>(d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_output[4];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost));

    printf("线程块协作组:\n");
    for (int i = 0; i < 4; i++) {
        printf("  块 %d 大小: %d\n", i, h_output[i]);
    }

    // 演示 warp 归约
    const int N = 1024;
    float *h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_sum;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));

    demoTiledPartition<<<(N + 255) / 256, 256>>>(d_input, d_sum, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_sum;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nWarp 归约求和: %.0f (应为 %d)\n\n", h_sum, N);

    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_sum));
    free(h_input);
}

// ============================================================================
// 第二部分：Warp 级原语
// ============================================================================

__global__ void demoWarpPrimitives(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = warp.thread_rank();

    if (tid < n) {
        float val = input[tid];

        // 1. shfl - 从指定 lane 获取值
        float fromLane0 = warp.shfl(val, 0);

        // 2. shfl_up - 向上移动（低 lane -> 高 lane）
        float shiftedUp = warp.shfl_up(val, 1);

        // 3. shfl_down - 向下移动（高 lane -> 低 lane）
        float shiftedDown = warp.shfl_down(val, 1);

        // 4. shfl_xor - XOR 交换
        float xorSwap = warp.shfl_xor(val, 1);  // 与相邻 lane 交换

        // 5. ballot - 投票，返回满足条件的 lane 掩码
        unsigned int mask = warp.ballot(val > 0.5f);

        // 6. any/all - 条件检测
        bool anyPositive = warp.any(val > 0);
        bool allPositive = warp.all(val > 0);

        // 只有 lane 0 输出结果
        if (lane == 0) {
            output[blockIdx.x * 5 + 0] = fromLane0;
            output[blockIdx.x * 5 + 1] = (float)__popc(mask);  // 统计满足条件的数量
            output[blockIdx.x * 5 + 2] = anyPositive ? 1.0f : 0.0f;
            output[blockIdx.x * 5 + 3] = allPositive ? 1.0f : 0.0f;
            output[blockIdx.x * 5 + 4] = xorSwap;
        }
    }
}

void demoWarpOperations() {
    printf("=== 第二部分：Warp 级原语 ===\n\n");

    printf("协作组 Warp 原语:\n");
    printf("  shfl(val, lane)     - 从指定 lane 广播\n");
    printf("  shfl_up(val, delta) - 向上移动\n");
    printf("  shfl_down(val, delta) - 向下移动\n");
    printf("  shfl_xor(val, mask) - XOR 模式交换\n");
    printf("  ballot(pred)        - 谓词投票\n");
    printf("  any(pred)           - 任一为真\n");
    printf("  all(pred)           - 全部为真\n\n");

    const int N = 128;
    float *h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = (float)i;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, 20 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    demoWarpPrimitives<<<4, 32>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_output[20];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 20 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Warp 0 结果:\n");
    printf("  从 lane 0 广播: %.0f\n", h_output[0]);
    printf("  ballot 计数 (val > 0.5): %.0f\n", h_output[1]);
    printf("  any(val > 0): %.0f\n", h_output[2]);
    printf("  all(val > 0): %.0f\n", h_output[3]);
    printf("\n");

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
}

// ============================================================================
// 第三部分：动态并行
// ============================================================================

// 子内核
__global__ void childKernel(int *data, int start, int end, int depth) {
    int tid = start + threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < end) {
        data[tid] = depth;  // 记录递归深度
    }
}

// 父内核 - 演示动态并行
__global__ void parentKernel(int *data, int n, int depth, int maxDepth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0 && depth < maxDepth) {
        // 在 GPU 上启动子内核
        int chunkSize = n / 4;
        for (int i = 0; i < 4; i++) {
            int start = i * chunkSize;
            int end = (i + 1) * chunkSize;
            if (i == 3) end = n;

            int numThreads = min(256, end - start);
            int numBlocks = (end - start + numThreads - 1) / numThreads;

            // 动态启动子内核
            childKernel<<<numBlocks, numThreads>>>(data, start, end, depth + 1);
        }

        // 等待所有子内核完成
        deviceSyncChildKernels();
    }
}

// 递归快速排序（动态并行示例）
__device__ void swap(int *a, int *b) {
    int t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

__global__ void quicksortKernel(int *arr, int low, int high, int depth) {
    if (low < high && depth < 10) {  // 限制递归深度
        int pi = partition(arr, low, high);

        // 递归排序左半部分
        if (pi - 1 > low) {
            quicksortKernel<<<1, 1>>>(arr, low, pi - 1, depth + 1);
        }

        // 递归排序右半部分
        if (pi + 1 < high) {
            quicksortKernel<<<1, 1>>>(arr, pi + 1, high, depth + 1);
        }

        deviceSyncChildKernels();
    }
}

void demoDynamicParallelism() {
    printf("=== 第三部分：动态并行 ===\n\n");

    printf("动态并行特性:\n");
    printf("  - GPU 内核可以启动其他内核\n");
    printf("  - 支持递归算法\n");
    printf("  - 需要计算能力 >= 3.5\n");
    printf("  - 编译: nvcc -rdc=true -lcudadevrt\n\n");

    // 检查设备支持
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("警告: 设备不支持动态并行 (需要 sm_35+)\n\n");
        return;
    }

    // 演示父子内核
    printf("1. 父子内核示例:\n");
    {
        const int N = 1024;
        int *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
        CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(int)));

        parentKernel<<<1, 1>>>(d_data, N, 0, 2);
        CHECK_CUDA(cudaDeviceSynchronize());

        int *h_data = (int*)malloc(N * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

        printf("   数据样本 (显示递归深度):\n");
        printf("   data[0]=%d, data[256]=%d, data[512]=%d, data[768]=%d\n",
               h_data[0], h_data[256], h_data[512], h_data[768]);

        CHECK_CUDA(cudaFree(d_data));
        free(h_data);
    }

    // 演示递归快速排序
    printf("\n2. 递归快速排序:\n");
    {
        const int N = 64;
        int h_arr[N];

        // 生成随机数组
        for (int i = 0; i < N; i++) {
            h_arr[i] = rand() % 100;
        }

        printf("   排序前: ");
        for (int i = 0; i < 10; i++) printf("%d ", h_arr[i]);
        printf("...\n");

        int *d_arr;
        CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice));

        quicksortKernel<<<1, 1>>>(d_arr, 0, N - 1, 0);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost));

        printf("   排序后: ");
        for (int i = 0; i < 10; i++) printf("%d ", h_arr[i]);
        printf("...\n");

        // 验证排序
        bool sorted = true;
        for (int i = 1; i < N; i++) {
            if (h_arr[i] < h_arr[i-1]) {
                sorted = false;
                break;
            }
        }
        printf("   排序正确: %s\n\n", sorted ? "是" : "否");

        CHECK_CUDA(cudaFree(d_arr));
    }
}

// ============================================================================
// 第四部分：协作组归约
// ============================================================================

// 使用协作组的高效归约
__global__ void cooperativeReduce(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    extern __shared__ float sdata[];

    int tid = block.thread_rank();
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 首次加载归约
    float sum = 0.0f;
    if (gid < n) sum += input[gid];
    if (gid + blockDim.x < n) sum += input[gid + blockDim.x];

    // Warp 内归约
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }

    // Warp 首线程写入共享内存
    if (warp.thread_rank() == 0) {
        sdata[tid / 32] = sum;
    }

    block.sync();

    // 第一个 warp 完成最终归约
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;

        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            sum += warp.shfl_down(sum, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

void demoCooperativeReduce() {
    printf("=== 第四部分：协作组归约 ===\n\n");

    const int N = 1 << 20;
    const int BLOCK_SIZE = 256;

    float *h_input = (float*)malloc(N * sizeof(float));
    float expectedSum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
        expectedSum += h_input[i];
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    CHECK_CUDA(cudaMalloc(&d_output, numBlocks * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    cooperativeReduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE / 32 * sizeof(float)>>>(
        d_input, d_output, N);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        cooperativeReduce<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE / 32 * sizeof(float)>>>(
            d_input, d_output, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // CPU 端完成归约
    float *h_output = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpuSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        gpuSum += h_output[i];
    }

    printf("协作组归约结果:\n");
    printf("  GPU 求和: %.0f\n", gpuSum);
    printf("  期望值: %.0f\n", expectedSum);
    printf("  平均时间: %.4f ms\n", ms / 100);
    printf("  带宽: %.1f GB/s\n\n", N * sizeof(float) / (ms / 100 * 1e6));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

// ============================================================================
// 第五部分：coalesced_group（活跃线程组）
// ============================================================================

__global__ void demoCoalescedGroup(int *input, int *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 只有满足条件的线程继续
    if (tid < n && input[tid] > 0) {
        // 获取当前活跃线程的协作组
        cg::coalesced_group active = cg::coalesced_threads();

        // active.size() 是活跃线程数
        // active.thread_rank() 是在活跃组中的索引

        // 活跃线程内的归约
        int val = input[tid];
        for (int offset = active.size() / 2; offset > 0; offset /= 2) {
            val += active.shfl_down(val, offset);
        }

        // 只有组内第一个线程写入
        if (active.thread_rank() == 0) {
            atomicAdd(output, val);
        }
    }
}

void demoCoalescedGroupUsage() {
    printf("=== 第五部分：活跃线程组 ===\n\n");

    printf("coalesced_group 用途:\n");
    printf("  - 处理分支后的活跃线程\n");
    printf("  - 动态确定参与计算的线程\n");
    printf("  - 稀疏数据处理\n\n");

    const int N = 128;
    int h_input[N];

    // 创建稀疏数据（约一半为正）
    int positiveCount = 0;
    int expectedSum = 0;
    for (int i = 0; i < N; i++) {
        h_input[i] = (rand() % 2 == 0) ? (i + 1) : 0;
        if (h_input[i] > 0) {
            positiveCount++;
            expectedSum += h_input[i];
        }
    }

    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(int)));

    demoCoalescedGroup<<<(N + 31) / 32, 32>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_output;
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    printf("稀疏数据归约:\n");
    printf("  总元素: %d\n", N);
    printf("  正数个数: %d\n", positiveCount);
    printf("  GPU 求和: %d\n", h_output);
    printf("  期望值: %d\n\n", expectedSum);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 13: 动态并行与协作组                          ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("协作组支持: %s\n", (prop.major >= 6) ? "完全支持" :
           (prop.major >= 3) ? "部分支持" : "不支持");
    printf("动态并行支持: %s\n\n",
           (prop.major > 3 || (prop.major == 3 && prop.minor >= 5)) ? "是" : "否");

    demoCooperativeGroupsBasic();
    demoWarpOperations();
    demoDynamicParallelism();
    demoCooperativeReduce();
    demoCoalescedGroupUsage();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 协作组类型:\n");
    printf("   ┌────────────────────┬────────────────────────────────┐\n");
    printf("   │ 类型               │ 用途                           │\n");
    printf("   ├────────────────────┼────────────────────────────────┤\n");
    printf("   │ thread_block       │ 整个线程块同步                 │\n");
    printf("   │ thread_block_tile  │ 固定大小分区 (2,4,8,16,32)     │\n");
    printf("   │ coalesced_group    │ 分支后的活跃线程               │\n");
    printf("   │ grid_group         │ 网格级同步（协作启动）         │\n");
    printf("   └────────────────────┴────────────────────────────────┘\n\n");

    printf("2. Warp 级操作:\n");
    printf("   - shfl 系列: 线程间数据交换\n");
    printf("   - ballot: 谓词投票\n");
    printf("   - any/all: 条件检测\n");
    printf("   - 比共享内存更快的通信方式\n\n");

    printf("3. 动态并行:\n");
    printf("   - 内核中启动子内核\n");
    printf("   - 适合递归算法\n");
    printf("   - 需要 -rdc=true -lcudadevrt\n");
    printf("   - 有一定开销，谨慎使用\n\n");

    printf("4. 协作组优势:\n");
    printf("   - 更灵活的线程分组\n");
    printf("   - 类型安全的同步\n");
    printf("   - 统一的 API 接口\n");
    printf("   - 支持新硬件特性\n\n");

    printf("5. 最佳实践:\n");
    printf("   - 使用协作组代替 __syncthreads()\n");
    printf("   - 利用 tile 进行 warp 级优化\n");
    printf("   - coalesced_group 处理分支代码\n");
    printf("   - 动态并行适合递归，但注意开销\n\n");

    printf("编译命令:\n");
    printf("  nvcc -rdc=true -lcudadevrt 13_dynamic_parallelism.cu -o 13_dynamic_parallelism\n\n");

    return 0;
}
