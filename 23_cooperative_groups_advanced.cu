/**
 * =============================================================================
 * CUDA 教程 23: 协作组 (Cooperative Groups) 高级用法
 * =============================================================================
 *
 * 学习目标：
 * 1. 深入理解协作组的各种类型
 * 2. 学会使用 Grid-level 同步
 * 3. 掌握 Warp-level 原语和集合操作
 * 4. 了解分区和瓦片化协作组
 *
 * 关键概念：
 * - Thread Block Group: 线程块级别同步
 * - Grid Group: 全网格同步
 * - Tiled Partition: 自定义大小的线程组
 * - Coalesced Group: 活跃线程组
 *
 * 编译命令：
 *   nvcc -rdc=true -lcudadevrt 23_cooperative_groups_advanced.cu -o 23_cooperative_groups_advanced
 *
 * 需要: CUDA 9.0+, 计算能力 6.0+
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：协作组概念回顾
// ============================================================================

void demoCooperativeGroupsConcepts() {
    printf("=== 第一部分：协作组概念回顾 ===\n\n");

    printf("协作组 (Cooperative Groups) 是什么？\n");
    printf("  - 灵活的线程分组和同步机制\n");
    printf("  - 替代传统的 __syncthreads()\n");
    printf("  - 支持多种粒度的同步\n\n");

    printf("协作组类型层次:\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  grid_group (全网格)                            │\n");
    printf("  │    └─ multi_grid_group (多 GPU)                 │\n");
    printf("  │                                                 │\n");
    printf("  │  thread_block (线程块)                          │\n");
    printf("  │    └─ thread_block_tile<N> (瓦片，N=2,4,8..32) │\n");
    printf("  │                                                 │\n");
    printf("  │  coalesced_group (活跃线程)                     │\n");
    printf("  │                                                 │\n");
    printf("  │  cluster_group (线程块集群, CUDA 11.8+)         │\n");
    printf("  └─────────────────────────────────────────────────┘\n\n");

    printf("关键操作:\n");
    printf("  - sync()         : 组内同步\n");
    printf("  - size()         : 组大小\n");
    printf("  - thread_rank()  : 线程在组内的序号\n");
    printf("  - shfl()         : 组内数据交换\n");
    printf("  - reduce()       : 组内归约\n\n");
}

// ============================================================================
// 第二部分：线程块组 (Thread Block Group)
// ============================================================================

__global__ void threadBlockGroupDemo(int *output) {
    // 获取线程块组
    cg::thread_block block = cg::this_thread_block();

    int tid = block.thread_rank();

    // 使用共享内存
    extern __shared__ int sdata[];
    sdata[tid] = tid;

    // 块内同步 (等同于 __syncthreads())
    block.sync();

    // 读取邻居数据
    int neighbor = (tid + 1) % block.size();
    output[tid] = sdata[neighbor];

    // 只有第一个线程打印信息
    if (block.thread_rank() == 0) {
        printf("  线程块大小: %d\n", block.size());
        printf("  块 ID: (%d, %d, %d)\n",
               block.group_index().x, block.group_index().y, block.group_index().z);
    }
}

void demoThreadBlockGroup() {
    printf("=== 第二部分：线程块组 ===\n\n");

    const int N = 256;
    int *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(int)));

    printf("启动内核 (1 块, %d 线程):\n", N);
    threadBlockGroupDemo<<<1, N, N * sizeof(int)>>>(d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *h_output = new int[N];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("  验证: output[0]=%d (期望 1), output[255]=%d (期望 0)\n\n",
           h_output[0], h_output[255]);

    delete[] h_output;
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第三部分：瓦片化分区 (Tiled Partition)
// ============================================================================

__global__ void tiledPartitionDemo(float *input, float *output, int n) {
    // 获取线程块
    cg::thread_block block = cg::this_thread_block();

    // 创建 32 线程的瓦片 (warp)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // 创建更小的瓦片
    cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = input[tid];

    // 在不同大小的瓦片内进行归约
    // 使用 32 线程瓦片
    float sum32 = val;
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum32 += warp.shfl_down(sum32, offset);
    }

    // 使用 4 线程瓦片进行局部归约
    float sum4 = val;
    for (int offset = tile4.size() / 2; offset > 0; offset /= 2) {
        sum4 += tile4.shfl_down(sum4, offset);
    }

    // 打印一些调试信息
    if (tid == 0) {
        printf("  瓦片大小: warp=%d, tile16=%d, tile8=%d, tile4=%d\n",
               warp.size(), tile16.size(), tile8.size(), tile4.size());
        printf("  32线程归约结果: %.1f\n", sum32);
        printf("  4线程归约结果 (前4个): %.1f\n", sum4);
    }

    // 输出 warp 归约结果
    if (warp.thread_rank() == 0) {
        output[tid / 32] = sum32;
    }
}

void demoTiledPartition() {
    printf("=== 第三部分：瓦片化分区 ===\n\n");

    const int N = 256;
    float *d_input, *d_output;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, (N / 32) * sizeof(float)));

    // 初始化输入
    float *h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    printf("启动瓦片化归约:\n");
    tiledPartitionDemo<<<1, N>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_output = new float[N / 32];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, (N / 32) * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  每个 warp 的归约结果:\n");
    for (int i = 0; i < N / 32; i++) {
        printf("    Warp %d: %.1f (期望 32.0)\n", i, h_output[i]);
    }
    printf("\n");

    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第四部分：合并组 (Coalesced Group)
// ============================================================================

__global__ void coalescedGroupDemo(int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 条件分支，只有偶数线程执行
    if (tid % 2 == 0 && tid < n) {
        // 获取当前活跃线程组
        cg::coalesced_group active = cg::coalesced_threads();

        // 在活跃线程间进行操作
        int rank = active.thread_rank();
        int size = active.size();

        // 存储结果
        output[tid] = rank * 1000 + size;

        // 第一个活跃线程打印
        if (rank == 0) {
            printf("  活跃线程数: %d (每个 warp 中偶数线程)\n", size);
            printf("  活跃线程掩码: 0x%x\n", active.meta_group_rank());
        }

        // 活跃线程间同步
        active.sync();
    }
}

void demoCoalescedGroup() {
    printf("=== 第四部分：合并组 ===\n\n");

    printf("合并组用于处理条件分支后的活跃线程\n");
    printf("示例: 只有偶数线程执行\n\n");

    const int N = 64;
    int *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_output, 0, N * sizeof(int)));

    coalescedGroupDemo<<<1, N>>>(d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *h_output = new int[N];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("  部分结果 (rank*1000 + group_size):\n");
    printf("    线程 0: %d\n", h_output[0]);
    printf("    线程 2: %d\n", h_output[2]);
    printf("    线程 4: %d\n", h_output[4]);
    printf("\n");

    delete[] h_output;
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第五部分：网格级同步 (Grid-level Sync)
// ============================================================================

__global__ void gridSyncDemo(int *data, int n, int iterations) {
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int iter = 0; iter < iterations; iter++) {
        // 每个线程增加其值
        if (tid < n) {
            data[tid] += 1;
        }

        // 全网格同步 - 所有块的所有线程必须到达此点
        grid.sync();

        // 现在所有线程都完成了本次迭代
        // 可以安全地读取其他线程的结果
    }

    if (tid == 0) {
        printf("  网格大小: %d 线程\n", (int)grid.size());
        printf("  迭代次数: %d\n", iterations);
    }
}

void demoGridSync() {
    printf("=== 第五部分：网格级同步 ===\n\n");

    // 检查设备是否支持协作启动
    int device = 0;
    int supportsCoopLaunch = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supportsCoopLaunch,
                                       cudaDevAttrCooperativeLaunch, device));

    if (!supportsCoopLaunch) {
        printf("警告: 设备不支持协作启动，跳过此演示\n\n");
        return;
    }

    printf("设备支持协作启动: 是\n\n");

    const int N = 1024;
    const int blockSize = 256;
    const int iterations = 5;

    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(int)));

    // 查询最大可启动的块数
    int numBlocksPerSm;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, gridSyncDemo, blockSize, 0));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int maxBlocks = numBlocksPerSm * prop.multiProcessorCount;
    int numBlocks = min(maxBlocks, (N + blockSize - 1) / blockSize);

    printf("启动参数:\n");
    printf("  SM 数量: %d\n", prop.multiProcessorCount);
    printf("  每 SM 最大块数: %d\n", numBlocksPerSm);
    printf("  实际启动块数: %d\n\n", numBlocks);

    // 使用协作启动
    void *kernelArgs[] = { &d_data, (void*)&N, (void*)&iterations };

    CHECK_CUDA(cudaLaunchCooperativeKernel(
        (void*)gridSyncDemo,
        dim3(numBlocks),
        dim3(blockSize),
        kernelArgs
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // 验证结果
    int *h_data = new int[N];
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("  结果: data[0]=%d (期望 %d)\n\n", h_data[0], iterations);

    delete[] h_data;
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 第六部分：协作组归约操作
// ============================================================================

template<int TileSize>
__global__ void reduceWithTiles(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (tid < n) ? input[tid] : 0.0f;

    // 使用内置的归约函数
    float sum = cg::reduce(tile, val, cg::plus<float>());

    // 每个 tile 的第一个线程输出结果
    if (tile.thread_rank() == 0) {
        atomicAdd(output, sum);
    }
}

__global__ void reduceWithShfl(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ float sdata[32];  // 每个 warp 一个

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = warp.thread_rank();
    int warpId = threadIdx.x / 32;

    float val = (tid < n) ? input[tid] : 0.0f;

    // Warp 内归约
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // 每个 warp 的第一个线程写入共享内存
    if (lane == 0) {
        sdata[warpId] = val;
    }

    block.sync();

    // 第一个 warp 进行最终归约
    if (warpId == 0) {
        val = (lane < blockDim.x / 32) ? sdata[lane] : 0.0f;

        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            val += warp.shfl_down(val, offset);
        }

        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}

void demoCooperativeReduce() {
    printf("=== 第六部分：协作组归约操作 ===\n\n");

    const int N = 1 << 20;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));

    // 初始化输入
    float *h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("数据量: %d 个浮点数\n\n", N);

    // 测试不同 tile 大小
    printf("1. 使用内置 cg::reduce:\n");

    // 32 线程 tile
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start));
    reduceWithTiles<32><<<gridSize, blockSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    float time32;
    CHECK_CUDA(cudaEventElapsedTime(&time32, start, stop));
    printf("   Tile<32>: 结果=%.0f, 时间=%.3f ms\n", result, time32);

    // 16 线程 tile
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start));
    reduceWithTiles<16><<<gridSize, blockSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    float time16;
    CHECK_CUDA(cudaEventElapsedTime(&time16, start, stop));
    printf("   Tile<16>: 结果=%.0f, 时间=%.3f ms\n", result, time16);

    // 使用 shfl 的实现
    printf("\n2. 使用 shfl_down 手动实现:\n");
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start));
    reduceWithShfl<<<gridSize, blockSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    float timeShfl;
    CHECK_CUDA(cudaEventElapsedTime(&timeShfl, start, stop));
    printf("   结果=%.0f, 时间=%.3f ms\n\n", result, timeShfl);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    delete[] h_input;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第七部分：Warp 级原语
// ============================================================================

__global__ void warpPrimitivesDemo() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int lane = warp.thread_rank();

    // 每个线程有不同的值
    int myVal = lane + 1;

    printf("=== Warp 级原语演示 ===\n\n");

    // 1. shfl - 从指定线程获取值
    if (lane == 0) printf("1. shfl (广播 lane 0 的值):\n");
    warp.sync();
    int broadcast = warp.shfl(myVal, 0);  // 从 lane 0 广播
    if (lane < 4) printf("   Lane %d: %d -> %d\n", lane, myVal, broadcast);
    warp.sync();

    // 2. shfl_up - 从更低的 lane 获取值
    if (lane == 0) printf("\n2. shfl_up (shift=1):\n");
    warp.sync();
    int up = warp.shfl_up(myVal, 1);
    if (lane < 4) printf("   Lane %d: %d -> %d\n", lane, myVal, up);
    warp.sync();

    // 3. shfl_down - 从更高的 lane 获取值
    if (lane == 0) printf("\n3. shfl_down (shift=1):\n");
    warp.sync();
    int down = warp.shfl_down(myVal, 1);
    if (lane < 4) printf("   Lane %d: %d -> %d\n", lane, myVal, down);
    warp.sync();

    // 4. shfl_xor - XOR 模式交换
    if (lane == 0) printf("\n4. shfl_xor (mask=1, 交换相邻):\n");
    warp.sync();
    int xored = warp.shfl_xor(myVal, 1);
    if (lane < 4) printf("   Lane %d: %d <-> %d\n", lane, myVal, xored);
    warp.sync();

    // 5. ballot - 投票
    if (lane == 0) printf("\n5. ballot (lane < 16):\n");
    warp.sync();
    unsigned int mask = warp.ballot(lane < 16);
    if (lane == 0) printf("   结果掩码: 0x%08x (期望: 0x0000ffff)\n", mask);
    warp.sync();

    // 6. any/all
    if (lane == 0) printf("\n6. any/all:\n");
    warp.sync();
    bool anyTrue = warp.any(lane == 15);
    bool allTrue = warp.all(lane < 32);
    if (lane == 0) {
        printf("   any(lane==15): %s\n", anyTrue ? "true" : "false");
        printf("   all(lane<32): %s\n", allTrue ? "true" : "false");
    }
    warp.sync();

    // 7. match_any/match_all (需要 CUDA 9+)
    if (lane == 0) printf("\n7. match_any (按值分组):\n");
    warp.sync();
    int group = lane / 8;  // 4 组
    unsigned int matchMask = warp.match_any(group);
    if (lane % 8 == 0) {
        printf("   组 %d (lane %d): 掩码=0x%08x\n", group, lane, matchMask);
    }
}

void demoWarpPrimitives() {
    printf("=== 第七部分：Warp 级原语 ===\n\n");

    printf("启动单个 warp (32 线程):\n\n");
    warpPrimitivesDemo<<<1, 32>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("\n");
}

// ============================================================================
// 第八部分：实际应用 - 并行前缀和
// ============================================================================

__global__ void parallelPrefixSum(int *input, int *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据
    sdata[tid] = (gid < n) ? input[gid] : 0;
    block.sync();

    // Warp 内前缀和
    int val = sdata[tid];
    for (int offset = 1; offset < warp.size(); offset *= 2) {
        int temp = warp.shfl_up(val, offset);
        if (warp.thread_rank() >= offset) {
            val += temp;
        }
    }
    sdata[tid] = val;
    block.sync();

    // 块内前缀和 (跨 warp)
    if (tid % 32 == 31) {
        // 每个 warp 的最后一个线程
        int warpSum = val;
        for (int warpOffset = 1; warpOffset < blockDim.x / 32; warpOffset *= 2) {
            int temp = __shfl_up_sync(0xFFFFFFFF, warpSum, warpOffset);
            if ((tid / 32) >= warpOffset) {
                warpSum += temp;
            }
        }
        // 更新每个 warp 的最后一个元素
        sdata[tid] = warpSum;
    }
    block.sync();

    // 将 warp 的和传播到组内其他线程
    if (tid / 32 > 0 && tid % 32 != 31) {
        val += sdata[(tid / 32) * 32 - 1];
    } else if (tid / 32 > 0 && tid % 32 == 31) {
        val = sdata[tid];
    }

    // 输出
    if (gid < n) {
        output[gid] = val;
    }
}

void demoPrefixSum() {
    printf("=== 第八部分：并行前缀和 ===\n\n");

    const int N = 256;
    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(int)));

    // 初始化
    int *h_input = new int[N];
    for (int i = 0; i < N; i++) h_input[i] = 1;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("输入: 全 1 数组, 长度 %d\n\n", N);

    // 运行内核
    parallelPrefixSum<<<1, N, N * sizeof(int)>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 验证
    int *h_output = new int[N];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("前缀和结果 (inclusive):\n");
    printf("  output[0] = %d (期望: 1)\n", h_output[0]);
    printf("  output[31] = %d (期望: 32)\n", h_output[31]);
    printf("  output[63] = %d (期望: 64)\n", h_output[63]);
    printf("  output[255] = %d (期望: 256)\n\n", h_output[255]);

    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第九部分：最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第九部分：最佳实践 ===\n\n");

    printf("1. 选择合适的组类型:\n");
    printf("   - thread_block: 块内同步 (替代 __syncthreads)\n");
    printf("   - thread_block_tile<32>: warp 操作\n");
    printf("   - coalesced_group: 条件分支后同步\n");
    printf("   - grid_group: 全网格同步 (谨慎使用)\n\n");

    printf("2. Tile 大小选择:\n");
    printf("   - 32: warp 级别，最高效\n");
    printf("   - 16/8/4: 更小粒度，更多并行\n");
    printf("   - 必须是 2 的幂\n\n");

    printf("3. 网格同步注意事项:\n");
    printf("   - 需要协作启动 (cudaLaunchCooperativeKernel)\n");
    printf("   - 块数受限于驻留能力\n");
    printf("   - 所有块必须同时驻留\n");
    printf("   - 使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor\n\n");

    printf("4. 避免的问题:\n");
    printf("   - 在分支中调用 sync() 导致死锁\n");
    printf("   - 不匹配的组操作\n");
    printf("   - 超出组边界的访问\n\n");

    printf("5. 性能提示:\n");
    printf("   - 优先使用 warp 级原语\n");
    printf("   - 避免不必要的同步\n");
    printf("   - 使用内置归约函数\n");
    printf("   - 考虑内存合并访问\n\n");

    printf("6. 代码示例:\n");
    printf("   // 安全的条件同步\n");
    printf("   cg::thread_block block = cg::this_thread_block();\n");
    printf("   if (condition) {\n");
    printf("       cg::coalesced_group active = cg::coalesced_threads();\n");
    printf("       // 只同步活跃线程\n");
    printf("       active.sync();\n");
    printf("   }\n");
    printf("   block.sync();  // 所有线程\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA 教程 23: 协作组 (Cooperative Groups) 高级用法             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    if (prop.major < 6) {
        printf("警告: 协作组需要计算能力 6.0+\n\n");
    }

    demoCooperativeGroupsConcepts();
    demoThreadBlockGroup();
    demoTiledPartition();
    demoCoalescedGroup();
    demoGridSync();
    demoCooperativeReduce();
    demoWarpPrimitives();
    demoPrefixSum();
    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 协作组类型:\n");
    printf("   ┌─────────────────────────┬────────────────────────────────┐\n");
    printf("   │ 类型                    │ 用途                           │\n");
    printf("   ├─────────────────────────┼────────────────────────────────┤\n");
    printf("   │ thread_block            │ 块内同步                       │\n");
    printf("   │ thread_block_tile<N>    │ warp/子warp 操作               │\n");
    printf("   │ coalesced_group         │ 活跃线程组                     │\n");
    printf("   │ grid_group              │ 全网格同步                     │\n");
    printf("   └─────────────────────────┴────────────────────────────────┘\n\n");

    printf("2. 关键操作:\n");
    printf("   - sync(): 组内同步\n");
    printf("   - shfl/shfl_up/shfl_down/shfl_xor: 数据交换\n");
    printf("   - ballot/any/all: 集体决策\n");
    printf("   - reduce: 归约操作\n\n");

    printf("3. 编译要求:\n");
    printf("   - 网格同步需要: -rdc=true -lcudadevrt\n");
    printf("   - 启动方式: cudaLaunchCooperativeKernel\n\n");

    printf("4. 实际应用:\n");
    printf("   - 并行归约\n");
    printf("   - 前缀和/扫描\n");
    printf("   - 直方图\n");
    printf("   - 排序算法\n\n");

    return 0;
}
