/**
 * =============================================================================
 * CUDA 教程 14: 多 GPU 编程
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解多 GPU 系统的基本概念
 * 2. 学会在多个 GPU 之间分配工作
 * 3. 掌握 GPU 间数据传输（P2P）
 * 4. 了解多 GPU 同步机制
 *
 * 关键概念：
 * - 设备管理：cudaSetDevice, cudaGetDeviceCount
 * - P2P 访问：GPU 直接访问另一 GPU 的内存
 * - 多流并发：每个 GPU 使用独立的流
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"
#include <vector>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：设备查询和管理
// ============================================================================

void demoDeviceManagement() {
    printf("=== 第一部分：设备管理 ===\n\n");

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    printf("检测到 %d 个 CUDA 设备:\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  SM 数量: %d\n", prop.multiProcessorCount);
        // 使用版本兼容性宏自动处理 CUDA 12+ memoryClockRate 弃用问题
        printf("  内存带宽: %.0f GB/s (估算)\n", GET_MEMORY_BANDWIDTH_GBPS(prop));

        // 检查统一虚拟地址支持
        printf("  统一虚拟寻址(UVA): %s\n", prop.unifiedAddressing ? "是" : "否");

        printf("\n");
    }

    // 检查 P2P 支持
    if (deviceCount >= 2) {
        printf("P2P (Peer-to-Peer) 支持:\n");
        for (int i = 0; i < deviceCount; i++) {
            for (int j = 0; j < deviceCount; j++) {
                if (i != j) {
                    int canAccess;
                    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess, i, j));
                    printf("  设备 %d -> 设备 %d: %s\n", i, j,
                           canAccess ? "支持" : "不支持");
                }
            }
        }
        printf("\n");
    }
}

// ============================================================================
// 第二部分：多 GPU 工作分配
// ============================================================================

__global__ void vectorAddKernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void demoWorkDistribution() {
    printf("=== 第二部分：多 GPU 工作分配 ===\n\n");

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 1) {
        printf("没有可用的 GPU\n");
        return;
    }

    // 即使只有一个 GPU，也演示分配逻辑
    int numGPUs = deviceCount;
    printf("使用 %d 个 GPU 进行计算\n\n", numGPUs);

    const int N = 1 << 24;  // 16M 元素
    const int totalSize = N * sizeof(float);

    // 分配主机内存
    float *h_a, *h_b, *h_c;
    CHECK_CUDA(cudaMallocHost(&h_a, totalSize));
    CHECK_CUDA(cudaMallocHost(&h_b, totalSize));
    CHECK_CUDA(cudaMallocHost(&h_c, totalSize));

    // 初始化
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 每个 GPU 分配的数据量
    int chunkSize = N / numGPUs;

    // 为每个 GPU 分配内存和流
    std::vector<float*> d_a(numGPUs), d_b(numGPUs), d_c(numGPUs);
    std::vector<cudaStream_t> streams(numGPUs);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 在每个 GPU 上分配内存并传输数据
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        int offset = i * chunkSize;
        int size = (i == numGPUs - 1) ? (N - offset) : chunkSize;
        int bytes = size * sizeof(float);

        CHECK_CUDA(cudaMalloc(&d_a[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_b[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_c[i], bytes));

        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // 异步复制数据
        CHECK_CUDA(cudaMemcpyAsync(d_a[i], h_a + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_b[i], h_b + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[i]));
    }

    // 在每个 GPU 上启动内核
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        int offset = i * chunkSize;
        int size = (i == numGPUs - 1) ? (N - offset) : chunkSize;

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        vectorAddKernel<<<gridSize, blockSize, 0, streams[i]>>>(
            d_a[i], d_b[i], d_c[i], size);
    }

    // 复制结果回主机
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        int offset = i * chunkSize;
        int size = (i == numGPUs - 1) ? (N - offset) : chunkSize;
        int bytes = size * sizeof(float);

        CHECK_CUDA(cudaMemcpyAsync(h_c + offset, d_c[i], bytes,
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // 同步所有 GPU
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            correct = false;
            printf("错误: h_c[%d] = %f\n", i, h_c[i]);
            break;
        }
    }

    printf("结果: %s\n", correct ? "正确" : "错误");
    printf("总时间: %.3f ms\n", ms);
    printf("有效带宽: %.2f GB/s\n\n",
           3.0 * totalSize / (ms * 1e6));

    // 清理
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_a[i]));
        CHECK_CUDA(cudaFree(d_b[i]));
        CHECK_CUDA(cudaFree(d_c[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
}

// ============================================================================
// 第三部分：P2P 数据传输
// ============================================================================

void demoP2PTransfer() {
    printf("=== 第三部分：P2P 数据传输 ===\n\n");

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        printf("需要至少 2 个 GPU 进行 P2P 演示\n");
        printf("跳过此部分，仅显示理论知识...\n\n");

        printf("P2P 传输原理:\n");
        printf("  1. 启用 P2P 访问:\n");
        printf("     cudaDeviceEnablePeerAccess(peerDevice, 0);\n\n");
        printf("  2. 直接内存复制:\n");
        printf("     cudaMemcpyPeer(dst, dstDev, src, srcDev, size);\n\n");
        printf("  3. 异步 P2P 复制:\n");
        printf("     cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, size, stream);\n\n");
        printf("  4. 直接访问对方内存（内核中）:\n");
        printf("     // GPU 0 的内核直接读取 GPU 1 的内存\n\n");
        return;
    }

    // 检查并启用 P2P
    int canAccess01, canAccess10;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess01, 0, 1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&canAccess10, 1, 0));

    if (!canAccess01 || !canAccess10) {
        printf("设备之间不支持 P2P 访问\n\n");
        return;
    }

    printf("启用 P2P 访问...\n");

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(1, 0));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(0, 0));

    printf("P2P 访问已启用\n\n");

    const int N = 1 << 24;
    const int size = N * sizeof(float);

    // 在 GPU 0 上分配
    float *d0_data;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&d0_data, size));

    // 在 GPU 1 上分配
    float *d1_data;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d1_data, size));

    // 初始化 GPU 0 的数据
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(d0_data, h_data, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测试 P2P 传输带宽
    printf("P2P 传输带宽测试:\n");

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 10; i++) {
        CHECK_CUDA(cudaMemcpyPeer(d1_data, 1, d0_data, 0, size));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("  数据大小: %.0f MB\n", size / (1024.0 * 1024));
    printf("  总时间 (10次): %.3f ms\n", ms);
    printf("  P2P 带宽: %.2f GB/s\n\n", 10.0 * size / (ms * 1e6));

    // 验证数据
    float *h_verify = (float*)malloc(size);
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(h_verify, d1_data, size, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_verify[i] != 1.0f) {
            correct = false;
            break;
        }
    }
    printf("P2P 传输验证: %s\n\n", correct ? "正确" : "错误");

    // 禁用 P2P
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceDisablePeerAccess(1));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceDisablePeerAccess(0));

    // 清理
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(d0_data));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaFree(d1_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_data);
    free(h_verify);
}

// ============================================================================
// 第四部分：多 GPU 矩阵乘法
// ============================================================================

__global__ void matmulKernel(float *A, float *B, float *C,
                              int M, int N, int K,
                              int rowStart, int numRows) {
    int row = rowStart + blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowStart + numRows && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[(row - rowStart) * K + k] * B[k * N + col];
        }
        C[(row - rowStart) * N + col] = sum;
    }
}

void demoMultiGPUMatmul() {
    printf("=== 第四部分：多 GPU 矩阵乘法 ===\n\n");

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    int numGPUs = deviceCount;
    printf("使用 %d 个 GPU 进行矩阵乘法\n", numGPUs);

    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    printf("矩阵大小: A(%d×%d) × B(%d×%d) = C(%d×%d)\n\n", M, K, K, N, M, N);

    // 分配主机内存
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_C, M * N * sizeof(float)));

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = 0.001f;
    for (int i = 0; i < K * N; i++) h_B[i] = 0.001f;

    // 每个 GPU 处理的行数
    int rowsPerGPU = M / numGPUs;

    // 设备内存和流
    std::vector<float*> d_A(numGPUs), d_B(numGPUs), d_C(numGPUs);
    std::vector<cudaStream_t> streams(numGPUs);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 在每个 GPU 上分配内存
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        int rowStart = i * rowsPerGPU;
        int numRows = (i == numGPUs - 1) ? (M - rowStart) : rowsPerGPU;

        // 每个 GPU 需要: 部分 A, 完整 B, 部分 C
        CHECK_CUDA(cudaMalloc(&d_A[i], numRows * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], numRows * N * sizeof(float)));

        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // 复制数据
        CHECK_CUDA(cudaMemcpyAsync(d_A[i], h_A + rowStart * K,
                                   numRows * K * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], h_B,
                                   K * N * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));
    }

    // 启动内核
    dim3 blockDim(16, 16);

    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        int rowStart = i * rowsPerGPU;
        int numRows = (i == numGPUs - 1) ? (M - rowStart) : rowsPerGPU;

        dim3 gridDim((N + 15) / 16, (numRows + 15) / 16);

        matmulKernel<<<gridDim, blockDim, 0, streams[i]>>>(
            d_A[i], d_B[i], d_C[i], M, N, K, rowStart, numRows);
    }

    // 复制结果回主机
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        int rowStart = i * rowsPerGPU;
        int numRows = (i == numGPUs - 1) ? (M - rowStart) : rowsPerGPU;

        CHECK_CUDA(cudaMemcpyAsync(h_C + rowStart * N, d_C[i],
                                   numRows * N * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // 同步
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算 GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e6);

    printf("总时间: %.3f ms\n", ms);
    printf("性能: %.2f GFLOPS\n", gflops);

    // 验证一个元素 (C[0][0] = sum(A[0][k] * B[k][0]) for k in 0..K)
    float expected = K * 0.001f * 0.001f;
    printf("验证 C[0][0]: 计算值=%.6f, 期望值=%.6f\n\n",
           h_C[0], expected);

    // 清理
    for (int i = 0; i < numGPUs; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
}

// ============================================================================
// 第五部分：多 GPU 同步
// ============================================================================

void demoMultiGPUSync() {
    printf("=== 第五部分：多 GPU 同步 ===\n\n");

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    printf("多 GPU 同步方法:\n\n");

    printf("1. cudaDeviceSynchronize()\n");
    printf("   - 等待当前设备上所有工作完成\n");
    printf("   - 需要对每个设备调用\n");
    printf("   示例:\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       cudaDeviceSynchronize();\n");
    printf("   }\n\n");

    printf("2. cudaStreamSynchronize(stream)\n");
    printf("   - 等待指定流完成\n");
    printf("   - 更细粒度的控制\n");
    printf("   示例:\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       cudaStreamSynchronize(streams[i]);\n");
    printf("   }\n\n");

    printf("3. cudaEvent 跨设备同步\n");
    printf("   - 一个设备等待另一个设备的事件\n");
    printf("   - 支持跨设备流依赖\n");
    printf("   示例:\n");
    printf("   // GPU 0 记录事件\n");
    printf("   cudaSetDevice(0);\n");
    printf("   cudaEventRecord(event0, stream0);\n");
    printf("   \n");
    printf("   // GPU 1 等待 GPU 0 的事件\n");
    printf("   cudaSetDevice(1);\n");
    printf("   cudaStreamWaitEvent(stream1, event0, 0);\n\n");

    // 演示事件同步
    if (deviceCount >= 2) {
        printf("事件同步演示:\n");

        cudaEvent_t event0;
        cudaStream_t stream0, stream1;

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventCreate(&event0));
        CHECK_CUDA(cudaStreamCreate(&stream0));

        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaStreamCreate(&stream1));

        // GPU 0 分配内存并记录事件
        float *d0_data;
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMalloc(&d0_data, 1024 * sizeof(float)));
        CHECK_CUDA(cudaEventRecord(event0, stream0));

        // GPU 1 等待 GPU 0
        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaStreamWaitEvent(stream1, event0, 0));

        printf("  GPU 1 等待 GPU 0 完成 - 同步成功\n\n");

        // 清理
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaFree(d0_data));
        CHECK_CUDA(cudaEventDestroy(event0));
        CHECK_CUDA(cudaStreamDestroy(stream0));

        CHECK_CUDA(cudaSetDevice(1));
        CHECK_CUDA(cudaStreamDestroy(stream1));
    }
}

// ============================================================================
// 第六部分：多 GPU 编程最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第六部分：最佳实践 ===\n\n");

    printf("1. 工作分配策略:\n");
    printf("   - 数据并行：将数据分割到多个 GPU\n");
    printf("   - 模型并行：将模型分割到多个 GPU\n");
    printf("   - 流水线并行：不同阶段在不同 GPU\n\n");

    printf("2. 数据传输优化:\n");
    printf("   - 使用 cudaMallocHost 分配固定内存\n");
    printf("   - 使用异步传输重叠计算和通信\n");
    printf("   - 尽量使用 P2P 直接传输\n");
    printf("   - 批量传输减少次数\n\n");

    printf("3. 负载均衡:\n");
    printf("   - 考虑 GPU 计算能力差异\n");
    printf("   - 动态工作分配\n");
    printf("   - 监控各 GPU 利用率\n\n");

    printf("4. 内存管理:\n");
    printf("   - 复用内存减少分配次数\n");
    printf("   - 使用内存池\n");
    printf("   - 注意各 GPU 内存限制\n\n");

    printf("5. 错误处理:\n");
    printf("   - 检查每个 GPU 的操作\n");
    printf("   - 正确设置当前设备\n");
    printf("   - 处理设备间差异\n\n");

    printf("6. 代码模式:\n");
    printf("   // 典型的多 GPU 处理流程\n");
    printf("   \n");
    printf("   // 1. 初始化\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       cudaMalloc(&d_data[i], ...);\n");
    printf("       cudaStreamCreate(&streams[i]);\n");
    printf("   }\n");
    printf("   \n");
    printf("   // 2. 数据传输 (H2D)\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       cudaMemcpyAsync(..., streams[i]);\n");
    printf("   }\n");
    printf("   \n");
    printf("   // 3. 计算\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       kernel<<<..., streams[i]>>>(...);\n");
    printf("   }\n");
    printf("   \n");
    printf("   // 4. 数据传输 (D2H)\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       cudaMemcpyAsync(..., streams[i]);\n");
    printf("   }\n");
    printf("   \n");
    printf("   // 5. 同步\n");
    printf("   for (int i = 0; i < numGPUs; i++) {\n");
    printf("       cudaSetDevice(i);\n");
    printf("       cudaStreamSynchronize(streams[i]);\n");
    printf("   }\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 14: 多 GPU 编程                               ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    demoDeviceManagement();
    demoWorkDistribution();
    demoP2PTransfer();
    demoMultiGPUMatmul();
    demoMultiGPUSync();
    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 设备管理:\n");
    printf("   - cudaGetDeviceCount() - 获取 GPU 数量\n");
    printf("   - cudaSetDevice(id) - 设置当前设备\n");
    printf("   - cudaGetDevice(&id) - 获取当前设备\n\n");

    printf("2. P2P 访问:\n");
    printf("   - cudaDeviceCanAccessPeer() - 检查支持\n");
    printf("   - cudaDeviceEnablePeerAccess() - 启用\n");
    printf("   - cudaMemcpyPeer() - P2P 复制\n\n");

    printf("3. 同步方法:\n");
    printf("   - cudaDeviceSynchronize() - 设备同步\n");
    printf("   - cudaStreamSynchronize() - 流同步\n");
    printf("   - cudaStreamWaitEvent() - 跨设备事件等待\n\n");

    printf("4. 性能优化:\n");
    printf("   - 使用固定内存\n");
    printf("   - 异步操作重叠\n");
    printf("   - P2P 直接传输\n");
    printf("   - 负载均衡\n\n");

    printf("5. 注意事项:\n");
    printf("   - 始终检查当前设备\n");
    printf("   - 内存仅在分配设备有效\n");
    printf("   - P2P 不是所有配置都支持\n");
    printf("   - 考虑 PCIe 带宽限制\n\n");

    // 重置设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }

    return 0;
}
