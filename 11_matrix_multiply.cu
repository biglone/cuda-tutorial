/**
 * =============================================================================
 * CUDA 教程 11: 矩阵乘法实战
 * =============================================================================
 *
 * 学习目标：
 * 1. 实现朴素矩阵乘法并理解其性能瓶颈
 * 2. 使用共享内存优化（分块矩阵乘法）
 * 3. 进一步优化：内存合并、bank冲突避免
 * 4. 与 cuBLAS 库性能对比
 *
 * 矩阵乘法: C = A × B
 * A: M × K
 * B: K × N
 * C: M × N
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"
#include <cublas_v2.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS 错误 %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
}

// 分块大小
#define TILE_SIZE 32

// ============================================================================
// 版本 1: 朴素实现
// ============================================================================

__global__ void matmulNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 版本 2: 共享内存分块
// ============================================================================

__global__ void matmulShared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有分块
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A 的分块到共享内存
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载 B 的分块到共享内存
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算分块的部分乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 版本 3: 避免 Bank 冲突的共享内存分块
// ============================================================================

// 添加填充避免 bank 冲突
#define TILE_SIZE_PADDED (TILE_SIZE + 1)

__global__ void matmulSharedNoBankConflict(float *A, float *B, float *C,
                                            int M, int N, int K) {
    // 使用填充避免 bank 冲突
    __shared__ float As[TILE_SIZE][TILE_SIZE_PADDED];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE_PADDED];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 展开内循环
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 版本 4: 每个线程计算多个元素
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 4
#define THREAD_N 4

__global__ void matmulOptimized(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 每个线程块的线程数
    int threadId = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;

    // 计算该线程负责的 C 矩阵位置
    int rowStart = by * TILE_M + (threadId / (TILE_N / THREAD_N)) * THREAD_M;
    int colStart = bx * TILE_N + (threadId % (TILE_N / THREAD_N)) * THREAD_N;

    // 累加器
    float accum[THREAD_M][THREAD_N] = {0.0f};

    // 遍历 K 维度的分块
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // 协作加载 A 到共享内存
        for (int i = threadId; i < TILE_K * TILE_M; i += numThreads) {
            int sk = i / TILE_M;
            int sm = i % TILE_M;
            int gm = by * TILE_M + sm;
            int gk = t * TILE_K + sk;
            As[sk][sm] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }

        // 协作加载 B 到共享内存
        for (int i = threadId; i < TILE_K * TILE_N; i += numThreads) {
            int sk = i / TILE_N;
            int sn = i % TILE_N;
            int gk = t * TILE_K + sk;
            int gn = bx * TILE_N + sn;
            Bs[sk][sn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }

        __syncthreads();

        // 计算
        int localRow = (threadId / (TILE_N / THREAD_N)) * THREAD_M;
        int localCol = (threadId % (TILE_N / THREAD_N)) * THREAD_N;

        for (int k = 0; k < TILE_K; k++) {
            float aVals[THREAD_M];
            float bVals[THREAD_N];

            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                aVals[m] = As[k][localRow + m];
            }

            #pragma unroll
            for (int n = 0; n < THREAD_N; n++) {
                bVals[n] = Bs[k][localCol + n];
            }

            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_N; n++) {
                    accum[m][n] += aVals[m] * bVals[n];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
    for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
            int gRow = rowStart + m;
            int gCol = colStart + n;
            if (gRow < M && gCol < N) {
                C[gRow * N + gCol] = accum[m][n];
            }
        }
    }
}

// ============================================================================
// CPU 参考实现
// ============================================================================

void matmulCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// 验证结果
// ============================================================================

bool verifyResult(float *ref, float *test, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > tolerance) {
            printf("验证失败: 位置 %d, 参考值 = %f, 测试值 = %f, 差异 = %f\n",
                   i, ref[i], test[i], diff);
            return false;
        }
    }
    return true;
}

// ============================================================================
// 性能测试
// ============================================================================

void runBenchmark(int M, int N, int K) {
    printf("\n=== 矩阵大小: A(%d×%d) × B(%d×%d) = C(%d×%d) ===\n\n",
           M, K, K, N, M, N);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // 分配主机内存
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 计算 FLOPS
    double flops = 2.0 * M * N * K;  // 每个元素: K次乘法 + K次加法
    printf("理论计算量: %.2f GFLOP\n\n", flops / 1e9);

    // === 测试各版本 ===

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    struct KernelInfo {
        const char *name;
        int iterations;
    };

    int numIterations = 10;

    // 1. 朴素实现
    printf("1. 朴素实现:\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float naiveTime;
    CHECK_CUDA(cudaEventElapsedTime(&naiveTime, start, stop));
    naiveTime /= numIterations;
    printf("   时间: %.3f ms, 性能: %.2f GFLOPS\n",
           naiveTime, flops / (naiveTime * 1e6));

    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C, sizeC, cudaMemcpyDeviceToHost));

    // 2. 共享内存分块
    printf("2. 共享内存分块:\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        matmulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float sharedTime;
    CHECK_CUDA(cudaEventElapsedTime(&sharedTime, start, stop));
    sharedTime /= numIterations;
    printf("   时间: %.3f ms, 性能: %.2f GFLOPS, 加速比: %.2fx\n",
           sharedTime, flops / (sharedTime * 1e6), naiveTime / sharedTime);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("   验证: %s\n", verifyResult(h_C_ref, h_C, M * N) ? "通过" : "失败");

    // 3. 避免 Bank 冲突
    printf("3. 避免 Bank 冲突:\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        matmulSharedNoBankConflict<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float noBankTime;
    CHECK_CUDA(cudaEventElapsedTime(&noBankTime, start, stop));
    noBankTime /= numIterations;
    printf("   时间: %.3f ms, 性能: %.2f GFLOPS, 加速比: %.2fx\n",
           noBankTime, flops / (noBankTime * 1e6), naiveTime / noBankTime);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("   验证: %s\n", verifyResult(h_C_ref, h_C, M * N) ? "通过" : "失败");

    // 4. 每线程多元素
    printf("4. 每线程多元素:\n");
    dim3 optBlockDim(16, 16);
    dim3 optGridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        matmulOptimized<<<optGridDim, optBlockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float optTime;
    CHECK_CUDA(cudaEventElapsedTime(&optTime, start, stop));
    optTime /= numIterations;
    printf("   时间: %.3f ms, 性能: %.2f GFLOPS, 加速比: %.2fx\n",
           optTime, flops / (optTime * 1e6), naiveTime / optTime);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("   验证: %s\n", verifyResult(h_C_ref, h_C, M * N, 1e-2f) ? "通过" : "失败");

    // 5. cuBLAS
    printf("5. cuBLAS (参考):\n");
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // 预热
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cublasTime;
    CHECK_CUDA(cudaEventElapsedTime(&cublasTime, start, stop));
    cublasTime /= numIterations;
    printf("   时间: %.3f ms, 性能: %.2f GFLOPS\n",
           cublasTime, flops / (cublasTime * 1e6));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("   验证: %s\n", verifyResult(h_C_ref, h_C, M * N, 1e-2f) ? "通过" : "失败");

    CHECK_CUBLAS(cublasDestroy(handle));

    // 性能总结
    printf("\n性能总结:\n");
    printf("┌──────────────────────┬────────────┬────────────────┐\n");
    printf("│ 版本                 │ GFLOPS     │ 相对 cuBLAS    │\n");
    printf("├──────────────────────┼────────────┼────────────────┤\n");
    printf("│ 朴素实现             │ %10.2f │ %13.1f%% │\n",
           flops / (naiveTime * 1e6), cublasTime / naiveTime * 100);
    printf("│ 共享内存分块         │ %10.2f │ %13.1f%% │\n",
           flops / (sharedTime * 1e6), cublasTime / sharedTime * 100);
    printf("│ 避免 Bank 冲突       │ %10.2f │ %13.1f%% │\n",
           flops / (noBankTime * 1e6), cublasTime / noBankTime * 100);
    printf("│ 每线程多元素         │ %10.2f │ %13.1f%% │\n",
           flops / (optTime * 1e6), cublasTime / optTime * 100);
    printf("│ cuBLAS               │ %10.2f │ %13.1f%% │\n",
           flops / (cublasTime * 1e6), 100.0f);
    printf("└──────────────────────┴────────────┴────────────────┘\n");

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
}

// ============================================================================
// 原理解释
// ============================================================================

void explainOptimizations() {
    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    优化原理解释                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 朴素实现的问题:\n");
    printf("   ┌─────────────────────────────────────────────────────┐\n");
    printf("   │ for (k = 0; k < K; k++)                             │\n");
    printf("   │     C[i][j] += A[i][k] * B[k][j]                    │\n");
    printf("   └─────────────────────────────────────────────────────┘\n");
    printf("   - 每个 C 元素需要读取 A 的一整行和 B 的一整列\n");
    printf("   - 大量重复的全局内存访问\n");
    printf("   - 内存带宽成为瓶颈\n\n");

    printf("2. 分块优化原理:\n");
    printf("   ┌────────────────────────────────────────────────────────┐\n");
    printf("   │        B                                               │\n");
    printf("   │  ┌─────────────┐                                       │\n");
    printf("   │  │ B00 │ B01   │      C = A × B                        │\n");
    printf("   │  │─────│───────│                                       │\n");
    printf("   │  │ B10 │ B11   │      C00 = A00×B00 + A01×B10          │\n");
    printf("   │  └─────────────┘      C01 = A00×B01 + A01×B11          │\n");
    printf("   │                       C10 = A10×B00 + A11×B10          │\n");
    printf("   │  A      ┌─────────────┐  C11 = A10×B01 + A11×B11       │\n");
    printf("   │  ┌──────┤ C00 │ C01   │                                │\n");
    printf("   │  │ A00  │─────│───────│                                │\n");
    printf("   │  │ A01  │ C10 │ C11   │                                │\n");
    printf("   │  │──────┼─────────────┘                                │\n");
    printf("   │  │ A10  │                                              │\n");
    printf("   │  │ A11  │                                              │\n");
    printf("   │  └──────┘                                              │\n");
    printf("   └────────────────────────────────────────────────────────┘\n");
    printf("   - 将矩阵分成小块\n");
    printf("   - 每个块先加载到共享内存\n");
    printf("   - 块内计算复用共享内存数据\n");
    printf("   - 数据复用率: O(TILE_SIZE)\n\n");

    printf("3. Bank 冲突:\n");
    printf("   共享内存分为 32 个 bank，连续 4 字节分配到连续 bank\n");
    printf("   当多个线程访问同一 bank 的不同地址时产生冲突\n");
    printf("   \n");
    printf("   有冲突情况 (列访问):\n");
    printf("   线程 0 → bank 0, 线程 1 → bank 0  (冲突!)\n");
    printf("   \n");
    printf("   添加填充后 (TILE_SIZE + 1):\n");
    printf("   行 0: [0][1][2]...[31][pad]    → bank 0,1,2..31,0\n");
    printf("   行 1: [0][1][2]...[31][pad]    → bank 1,2,3..0,1\n");
    printf("   现在列访问时每个线程访问不同 bank\n\n");

    printf("4. 每线程多元素:\n");
    printf("   - 每个线程计算 THREAD_M × THREAD_N 个输出\n");
    printf("   - 更好的指令级并行 (ILP)\n");
    printf("   - 减少线程调度开销\n");
    printf("   - 提高寄存器利用率\n\n");

    printf("5. 其他高级优化 (cuBLAS 使用):\n");
    printf("   - 向量化加载 (float4)\n");
    printf("   - 双缓冲 (计算与加载重叠)\n");
    printf("   - Tensor Core (如果可用)\n");
    printf("   - 自动调优选择最佳配置\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 11: 矩阵乘法实战                              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("共享内存 (每块): %zu KB\n", prop.sharedMemPerBlock / 1024);
    // 使用版本兼容性宏自动处理 CUDA 12+ memoryClockRate 弃用问题
    printf("内存带宽: %.0f GB/s (估算)\n\n", GET_MEMORY_BANDWIDTH_GBPS(prop));

    // 测试不同矩阵大小
    runBenchmark(1024, 1024, 1024);
    runBenchmark(2048, 2048, 2048);

    explainOptimizations();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 矩阵乘法优化层次:\n");
    printf("   朴素 → 分块 → 避免冲突 → 多元素/线程 → 向量化\n\n");

    printf("2. 关键性能指标:\n");
    printf("   - GFLOPS = 2×M×N×K / 时间(秒) / 10^9\n");
    printf("   - 内存带宽利用率\n");
    printf("   - 计算密度 (FLOP/字节)\n\n");

    printf("3. 优化策略:\n");
    printf("   - 使用共享内存减少全局内存访问\n");
    printf("   - 数据复用最大化\n");
    printf("   - 避免 bank 冲突\n");
    printf("   - 循环展开和指令级并行\n\n");

    printf("4. 实际建议:\n");
    printf("   - 生产代码直接使用 cuBLAS\n");
    printf("   - 手写核函数用于学习和特殊需求\n");
    printf("   - 注意数值精度和验证\n\n");

    printf("编译命令:\n");
    printf("  nvcc -lcublas 11_matrix_multiply.cu -o 11_matrix_multiply\n\n");

    return 0;
}
