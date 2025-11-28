/**
 * =============================================================================
 * CUDA 教程 33: 科学计算与数值方法
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解 GPU 在科学计算中的应用
 * 2. 实现常用数值计算算法
 * 3. 学习稀疏矩阵和迭代求解器
 * 4. 掌握物理模拟的 GPU 实现
 *
 * 实现内容：
 * - 数值积分 (Monte Carlo, 梯形法)
 * - 线性方程组求解 (共轭梯度法)
 * - ODE/PDE 数值解 (欧拉法, 有限差分)
 * - N体模拟
 * - 热传导方程
 *
 * 编译命令：
 *   nvcc 33_scientific_computing.cu -o 33_scientific -O3 -lcurand
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CURAND(call) { \
    curandStatus_t err = call; \
    if (err != CURAND_STATUS_SUCCESS) { \
        printf("cuRAND 错误 %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：Monte Carlo 数值积分
// ============================================================================

// 初始化随机数生成器
__global__ void initRngKernel(curandState *states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Monte Carlo 计算 π
// 通过在单位正方形内随机采样，统计落在单位圆内的点
__global__ void monteCarlopiKernel(int *counts, curandState *states, int samplesPerThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];

    int localCount = 0;
    for (int i = 0; i < samplesPerThread; i++) {
        float x = curand_uniform(&localState);
        float y = curand_uniform(&localState);

        if (x * x + y * y <= 1.0f) {
            localCount++;
        }
    }

    counts[tid] = localCount;
    states[tid] = localState;
}

// Monte Carlo 积分 (通用)
// 计算 ∫f(x)dx 在 [a,b] 上的积分
__device__ float integrand(float x) {
    // 示例: sin(x)
    return sinf(x);
}

__global__ void monteCarloIntegralKernel(float *results, curandState *states,
                                          float a, float b, int samplesPerThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = states[tid];

    float sum = 0.0f;
    float range = b - a;

    for (int i = 0; i < samplesPerThread; i++) {
        float x = a + curand_uniform(&localState) * range;
        sum += integrand(x);
    }

    results[tid] = sum * range / samplesPerThread;
    states[tid] = localState;
}

// 并行规约求和
__global__ void reduceSum(float *output, const float *input, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void demoMonteCarlo() {
    printf("=== 第一部分：Monte Carlo 数值积分 ===\n\n");

    const int numThreads = 256;
    const int numBlocks = 256;
    const int totalThreads = numThreads * numBlocks;
    const int samplesPerThread = 10000;
    const long long totalSamples = (long long)totalThreads * samplesPerThread;

    curandState *d_states;
    int *d_counts;
    float *d_results, *d_partialSums;

    CHECK_CUDA(cudaMalloc(&d_states, totalThreads * sizeof(curandState)));
    CHECK_CUDA(cudaMalloc(&d_counts, totalThreads * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_results, totalThreads * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partialSums, numBlocks * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 初始化随机数生成器
    initRngKernel<<<numBlocks, numThreads>>>(d_states, time(NULL), totalThreads);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 1. Monte Carlo 计算 π
    printf("1. Monte Carlo 计算 π:\n");

    CHECK_CUDA(cudaEventRecord(start));
    monteCarlopiKernel<<<numBlocks, numThreads>>>(d_counts, d_states, samplesPerThread);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    // 统计结果
    int *h_counts = (int*)malloc(totalThreads * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_counts, d_counts, totalThreads * sizeof(int), cudaMemcpyDeviceToHost));

    long long totalInCircle = 0;
    for (int i = 0; i < totalThreads; i++) {
        totalInCircle += h_counts[i];
    }

    double piEstimate = 4.0 * totalInCircle / totalSamples;
    double piError = fabs(piEstimate - M_PI);

    printf("   采样点数: %lld\n", totalSamples);
    printf("   π 估计值: %.10f\n", piEstimate);
    printf("   真实值:   %.10f\n", M_PI);
    printf("   误差:     %.10f\n", piError);
    printf("   计算时间: %.3f ms\n", elapsed);
    printf("   吞吐量:   %.2f 亿样本/秒\n\n", totalSamples / elapsed / 1e5);

    // 2. Monte Carlo 积分
    printf("2. Monte Carlo 积分 (∫sin(x)dx, x∈[0,π]):\n");

    CHECK_CUDA(cudaEventRecord(start));
    monteCarloIntegralKernel<<<numBlocks, numThreads>>>(d_results, d_states, 0.0f, M_PI, samplesPerThread);

    // 规约求平均
    reduceSum<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_partialSums, d_results, totalThreads);
    reduceSum<<<1, numBlocks, numBlocks * sizeof(float)>>>(d_results, d_partialSums, numBlocks);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_results, sizeof(float), cudaMemcpyDeviceToHost));
    h_result /= totalThreads;

    double exactValue = 2.0;  // ∫sin(x)dx from 0 to π = 2
    printf("   估计值:   %.10f\n", h_result);
    printf("   精确值:   %.10f\n", exactValue);
    printf("   误差:     %.10f\n", fabs(h_result - exactValue));
    printf("   计算时间: %.3f ms\n\n", elapsed);

    free(h_counts);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_states));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_partialSums));
}

// ============================================================================
// 第二部分：共轭梯度法 (CG) 求解线性方程组
// ============================================================================

// 向量点积
__global__ void dotProductKernel(float *result, const float *a, const float *b, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

// 向量加法: y = a*x + y
__global__ void axpyKernel(float *y, const float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// 向量缩放和拷贝: y = a*x
__global__ void scaleKernel(float *y, const float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i];
    }
}

// 稀疏矩阵向量乘法 (CSR 格式)
__global__ void spMVKernel(float *y, const float *values, const int *colIndices,
                           const int *rowPtrs, const float *x, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        float sum = 0.0f;
        int rowStart = rowPtrs[row];
        int rowEnd = rowPtrs[row + 1];

        for (int j = rowStart; j < rowEnd; j++) {
            sum += values[j] * x[colIndices[j]];
        }
        y[row] = sum;
    }
}

// 生成测试用的对称正定矩阵 (泊松矩阵)
void generatePoissonMatrix(float **values, int **colIndices, int **rowPtrs,
                           int *nnz, int n) {
    // 1D 泊松矩阵: 三对角
    *nnz = 3 * n - 2;
    *values = (float*)malloc(*nnz * sizeof(float));
    *colIndices = (int*)malloc(*nnz * sizeof(int));
    *rowPtrs = (int*)malloc((n + 1) * sizeof(int));

    int idx = 0;
    for (int i = 0; i < n; i++) {
        (*rowPtrs)[i] = idx;

        if (i > 0) {
            (*values)[idx] = -1.0f;
            (*colIndices)[idx] = i - 1;
            idx++;
        }

        (*values)[idx] = 2.0f;
        (*colIndices)[idx] = i;
        idx++;

        if (i < n - 1) {
            (*values)[idx] = -1.0f;
            (*colIndices)[idx] = i + 1;
            idx++;
        }
    }
    (*rowPtrs)[n] = idx;
}

void demoCGSolver() {
    printf("=== 第二部分：共轭梯度法 (CG) ===\n\n");

    printf("共轭梯度法适用于:\n");
    printf("  - 大规模稀疏对称正定矩阵\n");
    printf("  - 不需要显式存储矩阵逆\n");
    printf("  - 迭代收敛\n\n");

    const int n = 10000;
    const int maxIter = 1000;
    const float tol = 1e-6f;

    // 生成测试矩阵
    float *h_values;
    int *h_colIndices, *h_rowPtrs;
    int nnz;
    generatePoissonMatrix(&h_values, &h_colIndices, &h_rowPtrs, &nnz, n);

    // 生成右端向量
    float *h_b = (float*)malloc(n * sizeof(float));
    float *h_x = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_b[i] = 1.0f;
        h_x[i] = 0.0f;
    }

    // 分配设备内存
    float *d_values, *d_x, *d_b, *d_r, *d_p, *d_Ap;
    int *d_colIndices, *d_rowPtrs;
    float *d_dot;

    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIndices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowPtrs, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dot, sizeof(float)));

    // 拷贝数据
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIndices, h_colIndices, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowPtrs, h_rowPtrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + 255) / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // CG 算法
    // r = b - A*x (初始 x=0, 所以 r=b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(float), cudaMemcpyDeviceToDevice));

    float rsold = 0.0f;
    CHECK_CUDA(cudaMemset(d_dot, 0, sizeof(float)));
    dotProductKernel<<<grid, block, block.x * sizeof(float)>>>(d_dot, d_r, d_r, n);
    CHECK_CUDA(cudaMemcpy(&rsold, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    int iter;
    for (iter = 0; iter < maxIter; iter++) {
        // Ap = A * p
        spMVKernel<<<grid, block>>>(d_Ap, d_values, d_colIndices, d_rowPtrs, d_p, n);

        // alpha = rsold / (p' * Ap)
        CHECK_CUDA(cudaMemset(d_dot, 0, sizeof(float)));
        dotProductKernel<<<grid, block, block.x * sizeof(float)>>>(d_dot, d_p, d_Ap, n);
        float pAp;
        CHECK_CUDA(cudaMemcpy(&pAp, d_dot, sizeof(float), cudaMemcpyDeviceToHost));
        float alpha = rsold / pAp;

        // x = x + alpha * p
        axpyKernel<<<grid, block>>>(d_x, d_p, alpha, n);

        // r = r - alpha * Ap
        axpyKernel<<<grid, block>>>(d_r, d_Ap, -alpha, n);

        // rsnew = r' * r
        CHECK_CUDA(cudaMemset(d_dot, 0, sizeof(float)));
        dotProductKernel<<<grid, block, block.x * sizeof(float)>>>(d_dot, d_r, d_r, n);
        float rsnew;
        CHECK_CUDA(cudaMemcpy(&rsnew, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

        if (sqrtf(rsnew) < tol) {
            iter++;
            break;
        }

        // p = r + (rsnew/rsold) * p
        float beta = rsnew / rsold;
        scaleKernel<<<grid, block>>>(d_p, d_p, beta, n);
        axpyKernel<<<grid, block>>>(d_p, d_r, 1.0f, n);

        rsold = rsnew;
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    printf("CG 求解器结果 (n=%d, nnz=%d):\n", n, nnz);
    printf("   迭代次数: %d\n", iter);
    printf("   残差:     %.2e\n", sqrtf(rsold));
    printf("   计算时间: %.3f ms\n", elapsed);
    printf("   每次迭代: %.3f ms\n\n", elapsed / iter);

    // 验证解
    CHECK_CUDA(cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("   解的前 5 个分量: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_x[i]);
    printf("\n\n");

    // 清理
    free(h_values); free(h_colIndices); free(h_rowPtrs);
    free(h_b); free(h_x);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_colIndices));
    CHECK_CUDA(cudaFree(d_rowPtrs));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_Ap));
    CHECK_CUDA(cudaFree(d_dot));
}

// ============================================================================
// 第三部分：N体模拟
// ============================================================================

struct Body {
    float x, y, z;      // 位置
    float vx, vy, vz;   // 速度
    float mass;
};

// 基本 N体力计算 (O(N²))
__global__ void nBodyForceKernel(float *ax, float *ay, float *az,
                                  const float *px, const float *py, const float *pz,
                                  const float *mass, int n, float eps2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float accX = 0.0f, accY = 0.0f, accZ = 0.0f;
        float xi = px[i], yi = py[i], zi = pz[i];

        for (int j = 0; j < n; j++) {
            float dx = px[j] - xi;
            float dy = py[j] - yi;
            float dz = pz[j] - zi;

            float dist2 = dx * dx + dy * dy + dz * dz + eps2;
            float invDist = rsqrtf(dist2);
            float invDist3 = invDist * invDist * invDist;

            float force = mass[j] * invDist3;
            accX += force * dx;
            accY += force * dy;
            accZ += force * dz;
        }

        ax[i] = accX;
        ay[i] = accY;
        az[i] = accZ;
    }
}

// 使用共享内存的分块 N体 (更高效)
#define BLOCK_SIZE 256

__global__ void nBodyForceTiledKernel(float *ax, float *ay, float *az,
                                       const float *px, const float *py, const float *pz,
                                       const float *mass, int n, float eps2) {
    __shared__ float spx[BLOCK_SIZE];
    __shared__ float spy[BLOCK_SIZE];
    __shared__ float spz[BLOCK_SIZE];
    __shared__ float smass[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float accX = 0.0f, accY = 0.0f, accZ = 0.0f;
    float xi = 0.0f, yi = 0.0f, zi = 0.0f;

    if (i < n) {
        xi = px[i]; yi = py[i]; zi = pz[i];
    }

    // 分块处理
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        int j = tile * BLOCK_SIZE + tid;

        if (j < n) {
            spx[tid] = px[j];
            spy[tid] = py[j];
            spz[tid] = pz[j];
            smass[tid] = mass[j];
        } else {
            spx[tid] = 0.0f;
            spy[tid] = 0.0f;
            spz[tid] = 0.0f;
            smass[tid] = 0.0f;
        }
        __syncthreads();

        if (i < n) {
            #pragma unroll 16
            for (int k = 0; k < BLOCK_SIZE; k++) {
                float dx = spx[k] - xi;
                float dy = spy[k] - yi;
                float dz = spz[k] - zi;

                float dist2 = dx * dx + dy * dy + dz * dz + eps2;
                float invDist = rsqrtf(dist2);
                float invDist3 = invDist * invDist * invDist;

                float force = smass[k] * invDist3;
                accX += force * dx;
                accY += force * dy;
                accZ += force * dz;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        ax[i] = accX;
        ay[i] = accY;
        az[i] = accZ;
    }
}

// 更新位置和速度 (欧拉积分)
__global__ void nBodyIntegrateKernel(float *px, float *py, float *pz,
                                      float *vx, float *vy, float *vz,
                                      const float *ax, const float *ay, const float *az,
                                      int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // 更新速度
        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        vz[i] += az[i] * dt;

        // 更新位置
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }
}

void demoNBody() {
    printf("=== 第三部分：N体模拟 ===\n\n");

    printf("N体问题:\n");
    printf("  - 计算 N 个物体之间的万有引力\n");
    printf("  - 直接方法: O(N²) 复杂度\n");
    printf("  - 使用共享内存优化访存\n\n");

    const int n = 16384;
    const float dt = 0.01f;
    const float eps2 = 0.01f;  // 软化参数

    // 分配内存
    float *h_px = (float*)malloc(n * sizeof(float));
    float *h_py = (float*)malloc(n * sizeof(float));
    float *h_pz = (float*)malloc(n * sizeof(float));
    float *h_mass = (float*)malloc(n * sizeof(float));

    // 初始化: 随机分布的粒子
    for (int i = 0; i < n; i++) {
        float r = 10.0f * powf((float)rand() / RAND_MAX, 1.0f/3.0f);
        float theta = 2 * M_PI * rand() / RAND_MAX;
        float phi = acosf(2.0f * rand() / RAND_MAX - 1.0f);

        h_px[i] = r * sinf(phi) * cosf(theta);
        h_py[i] = r * sinf(phi) * sinf(theta);
        h_pz[i] = r * cosf(phi);
        h_mass[i] = 1.0f;
    }

    float *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    float *d_ax, *d_ay, *d_az, *d_mass;

    CHECK_CUDA(cudaMalloc(&d_px, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_py, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pz, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vz, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ax, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ay, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_az, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mass, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pz, h_pz, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mass, h_mass, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_vx, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_vy, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_vz, 0, n * sizeof(float)));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 基本版本性能测试
    CHECK_CUDA(cudaEventRecord(start));
    nBodyForceKernel<<<grid, block>>>(d_ax, d_ay, d_az, d_px, d_py, d_pz, d_mass, n, eps2);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float basicTime;
    CHECK_CUDA(cudaEventElapsedTime(&basicTime, start, stop));

    // 分块版本性能测试
    CHECK_CUDA(cudaEventRecord(start));
    nBodyForceTiledKernel<<<grid, block>>>(d_ax, d_ay, d_az, d_px, d_py, d_pz, d_mass, n, eps2);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float tiledTime;
    CHECK_CUDA(cudaEventElapsedTime(&tiledTime, start, stop));

    // 完整模拟 (10 步)
    const int numSteps = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 0; step < numSteps; step++) {
        nBodyForceTiledKernel<<<grid, block>>>(d_ax, d_ay, d_az, d_px, d_py, d_pz, d_mass, n, eps2);
        nBodyIntegrateKernel<<<grid, block>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                                               d_ax, d_ay, d_az, n, dt);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float simTime;
    CHECK_CUDA(cudaEventElapsedTime(&simTime, start, stop));

    // 计算 GFLOPS
    long long interactions = (long long)n * n;
    long long flopsPerInteraction = 20;  // 大约
    float gflops = interactions * flopsPerInteraction / tiledTime / 1e6;

    printf("N体模拟性能 (N=%d):\n", n);
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 方法                    │ 时间 (ms) │ GFLOPS          │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");
    printf("  │ 基本版本                │ %9.3f │ %7.2f         │\n",
           basicTime, interactions * flopsPerInteraction / basicTime / 1e6);
    printf("  │ 分块共享内存            │ %9.3f │ %7.2f         │\n", tiledTime, gflops);
    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");
    printf("  %d 步模拟总时间: %.3f ms (每步 %.3f ms)\n\n", numSteps, simTime, simTime / numSteps);

    // 清理
    free(h_px); free(h_py); free(h_pz); free(h_mass);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_px)); CHECK_CUDA(cudaFree(d_py)); CHECK_CUDA(cudaFree(d_pz));
    CHECK_CUDA(cudaFree(d_vx)); CHECK_CUDA(cudaFree(d_vy)); CHECK_CUDA(cudaFree(d_vz));
    CHECK_CUDA(cudaFree(d_ax)); CHECK_CUDA(cudaFree(d_ay)); CHECK_CUDA(cudaFree(d_az));
    CHECK_CUDA(cudaFree(d_mass));
}

// ============================================================================
// 第四部分：热传导方程 (有限差分)
// ============================================================================

// 2D 热传导方程: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
// 使用显式欧拉法和五点差分

__global__ void heatDiffusionKernel(float *next, const float *curr,
                                     int nx, int ny, float alpha, float dt, float dx2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        int idx = y * nx + x;

        float center = curr[idx];
        float left = curr[idx - 1];
        float right = curr[idx + 1];
        float up = curr[idx - nx];
        float down = curr[idx + nx];

        // 拉普拉斯算子
        float laplacian = (left + right + up + down - 4.0f * center) / dx2;

        // 显式欧拉更新
        next[idx] = center + alpha * dt * laplacian;
    }
}

// 使用共享内存的优化版本
#define HEAT_TILE_SIZE 16

__global__ void heatDiffusionSharedKernel(float *next, const float *curr,
                                           int nx, int ny, float alpha, float dt, float dx2) {
    __shared__ float smem[HEAT_TILE_SIZE + 2][HEAT_TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * HEAT_TILE_SIZE + tx;
    int y = blockIdx.y * HEAT_TILE_SIZE + ty;

    // 加载中心区域
    if (x < nx && y < ny) {
        smem[ty + 1][tx + 1] = curr[y * nx + x];
    }

    // 加载边界
    if (tx == 0 && x > 0) {
        smem[ty + 1][0] = curr[y * nx + x - 1];
    }
    if (tx == HEAT_TILE_SIZE - 1 && x < nx - 1) {
        smem[ty + 1][HEAT_TILE_SIZE + 1] = curr[y * nx + x + 1];
    }
    if (ty == 0 && y > 0) {
        smem[0][tx + 1] = curr[(y - 1) * nx + x];
    }
    if (ty == HEAT_TILE_SIZE - 1 && y < ny - 1) {
        smem[HEAT_TILE_SIZE + 1][tx + 1] = curr[(y + 1) * nx + x];
    }

    __syncthreads();

    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        float center = smem[ty + 1][tx + 1];
        float left = smem[ty + 1][tx];
        float right = smem[ty + 1][tx + 2];
        float up = smem[ty][tx + 1];
        float down = smem[ty + 2][tx + 1];

        float laplacian = (left + right + up + down - 4.0f * center) / dx2;
        next[y * nx + x] = center + alpha * dt * laplacian;
    }
}

// 设置边界条件
__global__ void setBoundaryKernel(float *u, int nx, int ny,
                                   float top, float bottom, float left, float right) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nx) {
        u[idx] = top;                    // 上边界
        u[(ny - 1) * nx + idx] = bottom; // 下边界
    }

    if (idx < ny) {
        u[idx * nx] = left;              // 左边界
        u[idx * nx + nx - 1] = right;    // 右边界
    }
}

void demoHeatEquation() {
    printf("=== 第四部分：热传导方程 ===\n\n");

    printf("2D 热传导方程:\n");
    printf("  ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)\n");
    printf("  使用显式有限差分法求解\n\n");

    const int nx = 512;
    const int ny = 512;
    const float alpha = 0.25f;  // 热扩散系数
    const float dx = 1.0f;
    const float dx2 = dx * dx;
    const float dt = 0.1f;      // 时间步长
    const int numSteps = 1000;

    size_t size = nx * ny * sizeof(float);

    float *d_curr, *d_next;
    CHECK_CUDA(cudaMalloc(&d_curr, size));
    CHECK_CUDA(cudaMalloc(&d_next, size));

    // 初始化: 全部为零，边界条件
    CHECK_CUDA(cudaMemset(d_curr, 0, size));
    CHECK_CUDA(cudaMemset(d_next, 0, size));

    dim3 block(HEAT_TILE_SIZE, HEAT_TILE_SIZE);
    dim3 grid((nx + HEAT_TILE_SIZE - 1) / HEAT_TILE_SIZE,
              (ny + HEAT_TILE_SIZE - 1) / HEAT_TILE_SIZE);

    // 设置边界条件: 上100°, 下0°, 左右0°
    setBoundaryKernel<<<(max(nx, ny) + 255) / 256, 256>>>(d_curr, nx, ny, 100.0f, 0.0f, 0.0f, 0.0f);
    setBoundaryKernel<<<(max(nx, ny) + 255) / 256, 256>>>(d_next, nx, ny, 100.0f, 0.0f, 0.0f, 0.0f);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 基本版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 0; step < numSteps; step++) {
        heatDiffusionKernel<<<grid, block>>>(d_next, d_curr, nx, ny, alpha, dt, dx2);
        float *temp = d_curr;
        d_curr = d_next;
        d_next = temp;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float basicTime;
    CHECK_CUDA(cudaEventElapsedTime(&basicTime, start, stop));

    // 重置
    CHECK_CUDA(cudaMemset(d_curr, 0, size));
    CHECK_CUDA(cudaMemset(d_next, 0, size));
    setBoundaryKernel<<<(max(nx, ny) + 255) / 256, 256>>>(d_curr, nx, ny, 100.0f, 0.0f, 0.0f, 0.0f);
    setBoundaryKernel<<<(max(nx, ny) + 255) / 256, 256>>>(d_next, nx, ny, 100.0f, 0.0f, 0.0f, 0.0f);

    // 共享内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 0; step < numSteps; step++) {
        heatDiffusionSharedKernel<<<grid, block>>>(d_next, d_curr, nx, ny, alpha, dt, dx2);
        float *temp = d_curr;
        d_curr = d_next;
        d_next = temp;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float sharedTime;
    CHECK_CUDA(cudaEventElapsedTime(&sharedTime, start, stop));

    // 计算内存带宽
    long long bytesPerStep = 2LL * nx * ny * sizeof(float);  // 读+写
    float bandwidth = bytesPerStep * numSteps / sharedTime / 1e6;

    printf("热传导模拟 (%dx%d, %d步):\n", nx, ny, numSteps);
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 方法                    │ 时间 (ms) │ 带宽 (GB/s)     │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");
    printf("  │ 基本版本                │ %9.3f │ %7.2f         │\n",
           basicTime, bytesPerStep * numSteps / basicTime / 1e6);
    printf("  │ 共享内存                │ %9.3f │ %7.2f         │\n", sharedTime, bandwidth);
    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");

    // 获取最终温度分布的统计
    float *h_result = (float*)malloc(size);
    CHECK_CUDA(cudaMemcpy(h_result, d_curr, size, cudaMemcpyDeviceToHost));

    float minT = h_result[0], maxT = h_result[0], avgT = 0.0f;
    for (int i = 0; i < nx * ny; i++) {
        minT = fminf(minT, h_result[i]);
        maxT = fmaxf(maxT, h_result[i]);
        avgT += h_result[i];
    }
    avgT /= (nx * ny);

    printf("  最终温度分布:\n");
    printf("    最低: %.2f°C, 最高: %.2f°C, 平均: %.2f°C\n\n", minT, maxT, avgT);

    free(h_result);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_curr));
    CHECK_CUDA(cudaFree(d_next));
}

// ============================================================================
// 第五部分：常微分方程组 (ODE)
// ============================================================================

// Lorenz 系统
// dx/dt = σ(y - x)
// dy/dt = x(ρ - z) - y
// dz/dt = xy - βz

__global__ void lorenzRK4Kernel(float *x, float *y, float *z,
                                 float sigma, float rho, float beta,
                                 float dt, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float xi = x[tid], yi = y[tid], zi = z[tid];

        // RK4 method
        float k1x = sigma * (yi - xi);
        float k1y = xi * (rho - zi) - yi;
        float k1z = xi * yi - beta * zi;

        float x1 = xi + 0.5f * dt * k1x;
        float y1 = yi + 0.5f * dt * k1y;
        float z1 = zi + 0.5f * dt * k1z;

        float k2x = sigma * (y1 - x1);
        float k2y = x1 * (rho - z1) - y1;
        float k2z = x1 * y1 - beta * z1;

        float x2 = xi + 0.5f * dt * k2x;
        float y2 = yi + 0.5f * dt * k2y;
        float z2 = zi + 0.5f * dt * k2z;

        float k3x = sigma * (y2 - x2);
        float k3y = x2 * (rho - z2) - y2;
        float k3z = x2 * y2 - beta * z2;

        float x3 = xi + dt * k3x;
        float y3 = yi + dt * k3y;
        float z3 = zi + dt * k3z;

        float k4x = sigma * (y3 - x3);
        float k4y = x3 * (rho - z3) - y3;
        float k4z = x3 * y3 - beta * z3;

        x[tid] = xi + dt / 6.0f * (k1x + 2.0f * k2x + 2.0f * k3x + k4x);
        y[tid] = yi + dt / 6.0f * (k1y + 2.0f * k2y + 2.0f * k3y + k4y);
        z[tid] = zi + dt / 6.0f * (k1z + 2.0f * k2z + 2.0f * k3z + k4z);
    }
}

void demoODE() {
    printf("=== 第五部分：常微分方程组 (ODE) ===\n\n");

    printf("Lorenz 系统 (混沌吸引子):\n");
    printf("  dx/dt = σ(y - x)\n");
    printf("  dy/dt = x(ρ - z) - y\n");
    printf("  dz/dt = xy - βz\n");
    printf("  使用 RK4 方法求解\n\n");

    const int n = 100000;      // 并行求解的初值数量
    const float sigma = 10.0f;
    const float rho = 28.0f;
    const float beta = 8.0f / 3.0f;
    const float dt = 0.001f;
    const int numSteps = 10000;

    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    float *h_z = (float*)malloc(n * sizeof(float));

    // 初始化: 在吸引子附近的随机初值
    for (int i = 0; i < n; i++) {
        h_x[i] = -8.0f + 16.0f * rand() / RAND_MAX;
        h_y[i] = -8.0f + 16.0f * rand() / RAND_MAX;
        h_z[i] = 20.0f + 10.0f * rand() / RAND_MAX;
    }

    float *d_x, *d_y, *d_z;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + 255) / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int step = 0; step < numSteps; step++) {
        lorenzRK4Kernel<<<grid, block>>>(d_x, d_y, d_z, sigma, rho, beta, dt, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    // 统计结果
    CHECK_CUDA(cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost));

    float avgX = 0, avgY = 0, avgZ = 0;
    for (int i = 0; i < n; i++) {
        avgX += h_x[i];
        avgY += h_y[i];
        avgZ += h_z[i];
    }
    avgX /= n; avgY /= n; avgZ /= n;

    long long totalOps = (long long)n * numSteps;
    printf("Lorenz 系统求解 (n=%d, steps=%d):\n", n, numSteps);
    printf("  计算时间: %.3f ms\n", elapsed);
    printf("  吞吐量: %.2f 亿 ODE步/秒\n", totalOps / elapsed / 1e5);
    printf("  最终平均位置: (%.2f, %.2f, %.2f)\n\n", avgX, avgY, avgZ);

    free(h_x); free(h_y); free(h_z);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_z));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 33: 科学计算与数值方法                               ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    demoMonteCarlo();
    demoCGSolver();
    demoNBody();
    demoHeatEquation();
    demoODE();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("科学计算 GPU 加速应用:\n");
    printf("  ✓ Monte Carlo 积分 - 大规模并行随机采样\n");
    printf("  ✓ 线性代数 - 稀疏矩阵、共轭梯度法\n");
    printf("  ✓ N体模拟 - 粒子间相互作用\n");
    printf("  ✓ PDE 求解 - 有限差分、热传导\n");
    printf("  ✓ ODE 求解 - RK4、Lorenz 系统\n\n");

    printf("优化技术:\n");
    printf("  - 共享内存减少全局内存访问\n");
    printf("  - 分块算法提高数据局部性\n");
    printf("  - 向量化和循环展开\n");
    printf("  - 稀疏矩阵格式优化\n\n");

    printf("CUDA 科学计算库:\n");
    printf("  ┌───────────────┬─────────────────────────────────────────┐\n");
    printf("  │ 库            │ 用途                                    │\n");
    printf("  ├───────────────┼─────────────────────────────────────────┤\n");
    printf("  │ cuBLAS        │ 线性代数 (BLAS)                         │\n");
    printf("  │ cuSPARSE      │ 稀疏矩阵运算                            │\n");
    printf("  │ cuSOLVER      │ 稠密/稀疏线性求解                       │\n");
    printf("  │ cuFFT         │ 快速傅里叶变换                          │\n");
    printf("  │ cuRAND        │ 随机数生成                              │\n");
    printf("  │ AmgX          │ 代数多重网格求解器                      │\n");
    printf("  │ CUTLASS       │ 高性能 GEMM 模板                        │\n");
    printf("  └───────────────┴─────────────────────────────────────────┘\n\n");

    return 0;
}
