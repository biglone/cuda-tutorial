/**
 * =============================================================================
 * CUDA 教程 07: CUDA Streams 并发执行
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解什么是 CUDA Stream
 * 2. 学会创建和使用多个 Stream
 * 3. 实现计算与传输重叠
 * 4. 了解异步操作
 *
 * 关键概念：
 * - Stream 是一个操作队列，同一 stream 内操作按顺序执行
 * - 不同 stream 之间的操作可以并发执行
 * - 默认所有操作都在 stream 0 (默认流) 中执行
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

// 简单的核函数，做一些计算
__global__ void processData(float *data, int n, float multiplier) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 模拟一些计算工作
        float val = data[tid];
        for (int i = 0; i < 100; i++) {
            val = val * multiplier + 0.1f;
        }
        data[tid] = val;
    }
}

// ============================================================================
// 示例 1: 同步执行 vs 异步执行
// ============================================================================

void demoSyncVsAsync() {
    printf("=== 示例 1: 同步 vs 异步执行 ===\n\n");

    const int N = 1 << 20;  // 1M 元素
    const int size = N * sizeof(float);

    // 分配固定内存（用于异步传输）
    float *h_data;
    CHECK_CUDA(cudaMallocHost(&h_data, size));  // 固定内存

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 初始化数据
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // === 同步版本 ===
    CHECK_CUDA(cudaEventRecord(start));

    // 同步复制（阻塞 CPU）
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    processData<<<(N+255)/256, 256>>>(d_data, N, 1.01f);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float syncTime;
    CHECK_CUDA(cudaEventElapsedTime(&syncTime, start, stop));
    printf("同步版本耗时: %.3f ms\n", syncTime);

    // === 异步版本 ===
    CHECK_CUDA(cudaEventRecord(start));

    // 异步复制（不阻塞 CPU）
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, 0);
    processData<<<(N+255)/256, 256>>>(d_data, N, 1.01f);
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, 0);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float asyncTime;
    CHECK_CUDA(cudaEventElapsedTime(&asyncTime, start, stop));
    printf("异步版本耗时: %.3f ms\n\n", asyncTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 示例 2: 多 Stream 并发执行
// ============================================================================

void demoMultipleStreams() {
    printf("=== 示例 2: 多 Stream 并发执行 ===\n\n");

    const int NUM_STREAMS = 4;
    const int N = 1 << 20;  // 每个 stream 处理 1M 元素
    const int CHUNK_SIZE = N / NUM_STREAMS;
    const int chunkBytes = CHUNK_SIZE * sizeof(float);

    // 分配固定内存
    float *h_data;
    CHECK_CUDA(cudaMallocHost(&h_data, N * sizeof(float)));

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    // 初始化
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    // 创建多个 stream
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // === 单 Stream 版本 ===
    CHECK_CUDA(cudaEventRecord(start));

    cudaMemcpyAsync(d_data, h_data, N * sizeof(float),
                    cudaMemcpyHostToDevice, 0);
    processData<<<(N+255)/256, 256, 0, 0>>>(d_data, N, 1.01f);
    cudaMemcpyAsync(h_data, d_data, N * sizeof(float),
                    cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float singleStreamTime;
    CHECK_CUDA(cudaEventElapsedTime(&singleStreamTime, start, stop));
    printf("单 Stream 耗时: %.3f ms\n", singleStreamTime);

    // === 多 Stream 版本 ===
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * CHUNK_SIZE;

        // 每个 stream 处理自己的数据块
        cudaMemcpyAsync(&d_data[offset], &h_data[offset], chunkBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        processData<<<(CHUNK_SIZE+255)/256, 256, 0, streams[i]>>>(
            &d_data[offset], CHUNK_SIZE, 1.01f);

        cudaMemcpyAsync(&h_data[offset], &d_data[offset], chunkBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 等待所有 stream 完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float multiStreamTime;
    CHECK_CUDA(cudaEventElapsedTime(&multiStreamTime, start, stop));
    printf("多 Stream (%d) 耗时: %.3f ms\n", NUM_STREAMS, multiStreamTime);
    printf("加速比: %.2fx\n\n", singleStreamTime / multiStreamTime);

    /*
     * 时间线示意图：
     *
     * 单 Stream:
     * [===H2D===][===计算===][===D2H===]
     *
     * 多 Stream (重叠执行):
     * Stream 0: [H2D][计算][D2H]
     * Stream 1:    [H2D][计算][D2H]
     * Stream 2:       [H2D][计算][D2H]
     * Stream 3:          [H2D][计算][D2H]
     */

    // 清理
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 示例 3: Stream 事件同步
// ============================================================================

void demoStreamEvents() {
    printf("=== 示例 3: Stream 事件同步 ===\n\n");

    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreate(&event));

    const int N = 1 << 18;
    float *d_data1, *d_data2;
    CHECK_CUDA(cudaMalloc(&d_data1, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_data2, N * sizeof(float)));

    // Stream 1 先执行任务 A
    processData<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N, 1.01f);
    CHECK_CUDA(cudaEventRecord(event, stream1));  // 在 stream1 中记录事件

    // Stream 2 等待 stream1 完成后再执行任务 B
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event, 0));  // stream2 等待事件
    processData<<<(N+255)/256, 256, 0, stream2>>>(d_data2, N, 1.02f);

    // 等待所有完成
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("执行顺序:\n");
    printf("1. Stream1: 任务 A 执行\n");
    printf("2. Stream1: 记录事件 (信号)\n");
    printf("3. Stream2: 等待事件\n");
    printf("4. Stream2: 任务 B 执行\n\n");

    CHECK_CUDA(cudaEventDestroy(event));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data1));
    CHECK_CUDA(cudaFree(d_data2));
}

// ============================================================================
// 示例 4: 回调函数（可选高级特性）
// ============================================================================

void CUDART_CB myCallback(void *userData) {
    printf("  [回调] Stream 任务完成! 用户数据: %s\n", (char*)userData);
}

void demoCallback() {
    printf("=== 示例 4: Stream 回调函数 ===\n\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int N = 1 << 18;
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    // 执行任务
    processData<<<(N+255)/256, 256, 0, stream>>>(d_data, N, 1.01f);

    // 添加回调
    char message[] = "Hello from callback!";
    CHECK_CUDA(cudaLaunchHostFunc(stream, myCallback, message));

    printf("主线程继续执行...\n");

    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("所有任务完成\n\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 示例 5: 查询 Stream 状态
// ============================================================================

void demoStreamQuery() {
    printf("=== 示例 5: 查询 Stream 状态 ===\n\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int N = 1 << 22;  // 较大数据以便观察
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    // 启动任务
    processData<<<(N+255)/256, 256, 0, stream>>>(d_data, N, 1.01f);

    // 非阻塞查询
    cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaSuccess) {
        printf("Stream 已完成\n");
    } else if (status == cudaErrorNotReady) {
        printf("Stream 仍在执行中...\n");
    }

    // 等待完成
    CHECK_CUDA(cudaStreamSynchronize(stream));

    status = cudaStreamQuery(stream);
    if (status == cudaSuccess) {
        printf("Stream 已完成\n");
    }

    printf("\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 07: CUDA Streams 并发执行                    ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    // 检查设备支持
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("并发内核支持: %s\n", prop.concurrentKernels ? "是" : "否");
    printf("异步引擎数量: %d\n\n", prop.asyncEngineCount);

    demoSyncVsAsync();
    demoMultipleStreams();
    demoStreamEvents();
    demoCallback();
    demoStreamQuery();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. Stream 基础:\n");
    printf("   - cudaStreamCreate() / cudaStreamDestroy()\n");
    printf("   - 默认流: stream 0 或 NULL\n");
    printf("   - 同一 stream 内操作顺序执行\n\n");

    printf("2. 异步操作:\n");
    printf("   - cudaMemcpyAsync() - 异步内存复制\n");
    printf("   - kernel<<<..., stream>>>() - 指定 stream 执行\n");
    printf("   - 需要使用固定内存 (cudaMallocHost)\n\n");

    printf("3. 同步方法:\n");
    printf("   - cudaDeviceSynchronize() - 等待所有 stream\n");
    printf("   - cudaStreamSynchronize(stream) - 等待指定 stream\n");
    printf("   - cudaStreamQuery() - 非阻塞查询状态\n\n");

    printf("4. 事件同步:\n");
    printf("   - cudaEventRecord(event, stream) - 记录事件\n");
    printf("   - cudaStreamWaitEvent(stream, event) - 等待事件\n\n");

    printf("5. 优化策略:\n");
    printf("   - 将大任务分成小块\n");
    printf("   - 使用多个 stream 重叠执行\n");
    printf("   - 重叠数据传输和计算\n\n");

    return 0;
}
