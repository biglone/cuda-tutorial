/**
 * =============================================================================
 * CUDA 教程 19: cuRAND 随机数生成
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 GPU 随机数生成的基本概念
 * 2. 学会使用 cuRAND 主机 API 和设备 API
 * 3. 掌握不同的随机数分布
 * 4. 了解蒙特卡洛模拟的实现
 *
 * 关键概念：
 * - 伪随机数生成器 (PRNG)
 * - 准随机数生成器 (QRNG)
 * - 主机 API vs 设备 API
 * - 不同的分布：均匀、正态、泊松等
 *
 * 编译命令：
 *   nvcc -lcurand 19_curand.cu -o 19_curand
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CURAND(call) { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        printf("cuRAND 错误 %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 第一部分：cuRAND 基础
// ============================================================================

void demoCurandBasics() {
    printf("=== 第一部分：cuRAND 基础 ===\n\n");

    printf("随机数生成器类型:\n\n");

    printf("1. 伪随机数生成器 (PRNG):\n");
    printf("   - XORWOW (默认): 高质量，周期 2^192\n");
    printf("   - MRG32K3A: 多递归生成器，周期 2^191\n");
    printf("   - MTGP32: Mersenne Twister，周期 2^11213\n");
    printf("   - Philox: 基于计数器，高性能\n");
    printf("   特点: 可重现，高速，统计质量好\n\n");

    printf("2. 准随机数生成器 (QRNG):\n");
    printf("   - Sobol: 低差异序列\n");
    printf("   - Scrambled Sobol: 打乱的 Sobol\n");
    printf("   特点: 更均匀的覆盖，适合蒙特卡洛积分\n\n");

    printf("两种 API:\n");
    printf("  - 主机 API: 在 CPU 上调用，生成 GPU 上的随机数\n");
    printf("  - 设备 API: 在 GPU 内核中直接生成随机数\n\n");
}

// ============================================================================
// 第二部分：主机 API
// ============================================================================

void demoHostAPI() {
    printf("=== 第二部分：主机 API ===\n\n");

    const int N = 1000000;

    // 创建生成器
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // 设置种子
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // 分配设备内存
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 生成均匀分布 [0, 1)
    printf("1. 均匀分布 U(0,1):\n");
    {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CURAND(curandGenerateUniform(gen, d_data, N));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // 计算统计量
        float *h_data = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        float sum = 0, sum_sq = 0;
        for (int i = 0; i < N; i++) {
            sum += h_data[i];
            sum_sq += h_data[i] * h_data[i];
        }
        float mean = sum / N;
        float variance = sum_sq / N - mean * mean;

        printf("   生成 %d 个随机数\n", N);
        printf("   时间: %.3f ms (%.2f GB/s)\n", ms, N * sizeof(float) / (ms * 1e6));
        printf("   均值: %.6f (期望: 0.5)\n", mean);
        printf("   方差: %.6f (期望: 0.0833)\n\n", variance);

        free(h_data);
    }

    // 生成正态分布 N(0, 1)
    printf("2. 正态分布 N(0,1):\n");
    {
        CHECK_CURAND(curandGenerateNormal(gen, d_data, N, 0.0f, 1.0f));

        float *h_data = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        float sum = 0, sum_sq = 0;
        for (int i = 0; i < N; i++) {
            sum += h_data[i];
            sum_sq += h_data[i] * h_data[i];
        }
        float mean = sum / N;
        float variance = sum_sq / N - mean * mean;

        printf("   均值: %.6f (期望: 0)\n", mean);
        printf("   方差: %.6f (期望: 1)\n\n", variance);

        free(h_data);
    }

    // 生成对数正态分布
    printf("3. 对数正态分布 LogN(0,1):\n");
    {
        CHECK_CURAND(curandGenerateLogNormal(gen, d_data, N, 0.0f, 1.0f));

        float *h_data = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

        float sum = 0;
        for (int i = 0; i < N; i++) sum += h_data[i];
        float mean = sum / N;

        // 对数正态分布的期望: exp(mu + sigma^2/2)
        float expected_mean = expf(0.0f + 1.0f / 2.0f);

        printf("   均值: %.4f (期望: %.4f)\n\n", mean, expected_mean);

        free(h_data);
    }

    // 测试不同生成器
    printf("4. 不同生成器性能比较:\n");
    {
        curandRngType_t types[] = {
            CURAND_RNG_PSEUDO_XORWOW,
            CURAND_RNG_PSEUDO_MRG32K3A,
            CURAND_RNG_PSEUDO_MTGP32,
            CURAND_RNG_PSEUDO_PHILOX4_32_10
        };
        const char* names[] = {"XORWOW", "MRG32K3A", "MTGP32", "Philox"};

        for (int i = 0; i < 4; i++) {
            curandGenerator_t g;
            CHECK_CURAND(curandCreateGenerator(&g, types[i]));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(g, 1234ULL));

            // 预热
            CHECK_CURAND(curandGenerateUniform(g, d_data, N));

            CHECK_CUDA(cudaEventRecord(start));
            for (int j = 0; j < 10; j++) {
                CHECK_CURAND(curandGenerateUniform(g, d_data, N));
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

            printf("   %s: %.3f ms (%.2f GB/s)\n",
                   names[i], ms / 10, 10.0 * N * sizeof(float) / (ms * 1e6));

            CHECK_CURAND(curandDestroyGenerator(g));
        }
        printf("\n");
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================================================
// 第三部分：设备 API
// ============================================================================

// 初始化随机数状态
__global__ void setupKernel(curandState *state, unsigned long seed, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 每个线程初始化自己的状态
        // 参数: seed, sequence, offset, state
        curand_init(seed, tid, 0, &state[tid]);
    }
}

// 生成随机数
__global__ void generateKernel(curandState *state, float *result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 从状态生成随机数
        result[tid] = curand_uniform(&state[tid]);
    }
}

// 生成正态分布
__global__ void generateNormalKernel(curandState *state, float *result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        result[tid] = curand_normal(&state[tid]);
    }
}

// 生成整数随机数
__global__ void generateIntKernel(curandState *state, unsigned int *result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        result[tid] = curand(&state[tid]);  // 返回 32 位无符号整数
    }
}

void demoDeviceAPI() {
    printf("=== 第三部分：设备 API ===\n\n");

    const int N = 100000;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 分配状态数组
    curandState *d_states;
    CHECK_CUDA(cudaMalloc(&d_states, N * sizeof(curandState)));

    float *d_result;
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 初始化状态
    printf("1. 初始化状态:\n");
    {
        CHECK_CUDA(cudaEventRecord(start));
        setupKernel<<<gridSize, blockSize>>>(d_states, 1234ULL, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("   %d 个状态初始化: %.3f ms\n", N, ms);
        printf("   状态内存: %.2f MB\n\n", N * sizeof(curandState) / (1024.0 * 1024));
    }

    // 生成均匀分布
    printf("2. 生成均匀分布:\n");
    {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            generateKernel<<<gridSize, blockSize>>>(d_states, d_result, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        float *h_result = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

        float sum = 0;
        for (int i = 0; i < N; i++) sum += h_result[i];
        float mean = sum / N;

        printf("   时间: %.4f ms/次\n", ms / 100);
        printf("   均值: %.6f\n\n", mean);

        free(h_result);
    }

    // 生成正态分布
    printf("3. 生成正态分布:\n");
    {
        generateNormalKernel<<<gridSize, blockSize>>>(d_states, d_result, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        float *h_result = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

        float sum = 0, sum_sq = 0;
        for (int i = 0; i < N; i++) {
            sum += h_result[i];
            sum_sq += h_result[i] * h_result[i];
        }
        float mean = sum / N;
        float variance = sum_sq / N - mean * mean;

        printf("   均值: %.6f, 方差: %.6f\n\n", mean, variance);

        free(h_result);
    }

    printf("设备 API 函数:\n");
    printf("  curand_init()    - 初始化状态\n");
    printf("  curand()         - 32位无符号整数\n");
    printf("  curand_uniform() - 均匀分布 (0,1]\n");
    printf("  curand_normal()  - 标准正态分布\n");
    printf("  curand_log_normal() - 对数正态分布\n");
    printf("  curand_poisson() - 泊松分布\n\n");

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_states));
    CHECK_CUDA(cudaFree(d_result));
}

// ============================================================================
// 第四部分：蒙特卡洛模拟 - 估算 π
// ============================================================================

__global__ void piMonteCarlo(curandState *state, int *count, int samples_per_thread) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 初始化状态
    curandState localState;
    curand_init(1234, tid, 0, &localState);

    int local_count = 0;

    for (int i = 0; i < samples_per_thread; i++) {
        float x = curand_uniform(&localState);
        float y = curand_uniform(&localState);

        // 检查点是否在单位圆内
        if (x * x + y * y <= 1.0f) {
            local_count++;
        }
    }

    // 原子加法累加结果
    atomicAdd(count, local_count);
}

void demoPiMonteCarlo() {
    printf("=== 第四部分：蒙特卡洛估算 π ===\n\n");

    printf("原理: 在单位正方形内随机撒点\n");
    printf("      落在内切圆内的概率 = π/4\n");
    printf("      π ≈ 4 × (圆内点数 / 总点数)\n\n");

    const int threads = 256;
    const int blocks = 256;
    const int total_threads = threads * blocks;

    int samples[] = {100, 1000, 10000};

    for (int s = 0; s < 3; s++) {
        int samples_per_thread = samples[s];
        long long total_samples = (long long)total_threads * samples_per_thread;

        int *d_count;
        CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

        curandState *d_states;
        CHECK_CUDA(cudaMalloc(&d_states, total_threads * sizeof(curandState)));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        piMonteCarlo<<<blocks, threads>>>(d_states, d_count, samples_per_thread);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        int h_count;
        CHECK_CUDA(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        double pi_estimate = 4.0 * h_count / total_samples;
        double error = fabs(pi_estimate - M_PI);

        printf("样本数: %lld\n", total_samples);
        printf("  估算 π = %.8f\n", pi_estimate);
        printf("  误差: %.2e\n", error);
        printf("  时间: %.3f ms\n\n", ms);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_count));
        CHECK_CUDA(cudaFree(d_states));
    }
}

// ============================================================================
// 第五部分：准随机数 (Quasi-Random)
// ============================================================================

void demoQuasiRandom() {
    printf("=== 第五部分：准随机数 ===\n\n");

    printf("准随机数 vs 伪随机数:\n");
    printf("  - 伪随机: 统计上随机，但可能聚集\n");
    printf("  - 准随机: 低差异序列，更均匀覆盖\n");
    printf("  准随机数在蒙特卡洛积分中收敛更快\n\n");

    const int N = 10000;

    // 创建 Sobol 生成器
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32));

    // 设置维度
    CHECK_CURAND(curandSetQuasiRandomGeneratorDimensions(gen, 2));

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * 2 * sizeof(float)));

    // 生成 2D 准随机数
    CHECK_CURAND(curandGenerateUniform(gen, d_data, N * 2));

    float *h_data = (float*)malloc(N * 2 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    // 用于估算 π
    int count = 0;
    for (int i = 0; i < N; i++) {
        float x = h_data[2 * i];
        float y = h_data[2 * i + 1];
        if (x * x + y * y <= 1.0f) {
            count++;
        }
    }

    double pi_quasi = 4.0 * count / N;
    printf("Sobol 准随机数估算 π:\n");
    printf("  样本: %d\n", N);
    printf("  估算: %.8f\n", pi_quasi);
    printf("  误差: %.2e\n\n", fabs(pi_quasi - M_PI));

    // 对比伪随机数
    curandGenerator_t genPseudo;
    CHECK_CURAND(curandCreateGenerator(&genPseudo, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(genPseudo, 1234ULL));
    CHECK_CURAND(curandGenerateUniform(genPseudo, d_data, N * 2));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, N * 2 * sizeof(float), cudaMemcpyDeviceToHost));

    count = 0;
    for (int i = 0; i < N; i++) {
        float x = h_data[2 * i];
        float y = h_data[2 * i + 1];
        if (x * x + y * y <= 1.0f) {
            count++;
        }
    }

    double pi_pseudo = 4.0 * count / N;
    printf("伪随机数估算 π:\n");
    printf("  样本: %d\n", N);
    printf("  估算: %.8f\n", pi_pseudo);
    printf("  误差: %.2e\n\n", fabs(pi_pseudo - M_PI));

    // 清理
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CURAND(curandDestroyGenerator(genPseudo));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

// ============================================================================
// 第六部分：期权定价 (Black-Scholes)
// ============================================================================

__global__ void blackScholesKernel(
    curandState *state,
    float *prices,
    float S0,      // 初始股价
    float K,       // 行权价
    float r,       // 无风险利率
    float sigma,   // 波动率
    float T,       // 期限
    int paths,
    int steps_per_path
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < paths) {
        curandState localState = state[tid];

        float dt = T / steps_per_path;
        float S = S0;

        // 模拟价格路径
        for (int i = 0; i < steps_per_path; i++) {
            float dW = curand_normal(&localState) * sqrtf(dt);
            S = S * expf((r - 0.5f * sigma * sigma) * dt + sigma * dW);
        }

        // 计算看涨期权收益
        float payoff = fmaxf(S - K, 0.0f);

        // 折现到现值
        prices[tid] = payoff * expf(-r * T);

        state[tid] = localState;
    }
}

void demoOptionPricing() {
    printf("=== 第六部分：期权定价 (蒙特卡洛) ===\n\n");

    // 期权参数
    float S0 = 100.0f;    // 初始股价
    float K = 100.0f;     // 行权价
    float r = 0.05f;      // 无风险利率 5%
    float sigma = 0.2f;   // 波动率 20%
    float T = 1.0f;       // 1年期

    printf("期权参数:\n");
    printf("  初始股价 S0 = %.0f\n", S0);
    printf("  行权价 K = %.0f\n", K);
    printf("  无风险利率 r = %.0f%%\n", r * 100);
    printf("  波动率 σ = %.0f%%\n", sigma * 100);
    printf("  期限 T = %.0f 年\n\n", T);

    const int paths = 100000;
    const int steps = 252;  // 一年交易日

    // 分配内存
    curandState *d_states;
    float *d_prices;
    CHECK_CUDA(cudaMalloc(&d_states, paths * sizeof(curandState)));
    CHECK_CUDA(cudaMalloc(&d_prices, paths * sizeof(float)));

    // 初始化状态
    int blockSize = 256;
    int gridSize = (paths + blockSize - 1) / blockSize;
    setupKernel<<<gridSize, blockSize>>>(d_states, 1234ULL, paths);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    blackScholesKernel<<<gridSize, blockSize>>>(
        d_states, d_prices, S0, K, r, sigma, T, paths, steps);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算平均价格
    float *h_prices = (float*)malloc(paths * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_prices, d_prices, paths * sizeof(float), cudaMemcpyDeviceToHost));

    double sum = 0;
    for (int i = 0; i < paths; i++) {
        sum += h_prices[i];
    }
    float option_price = sum / paths;

    // Black-Scholes 解析解（用于对比）
    float d1 = (logf(S0 / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);

    // 简化的正态分布累积函数近似
    auto N = [](float x) {
        return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    };

    float bs_price = S0 * N(d1) - K * expf(-r * T) * N(d2);

    printf("蒙特卡洛模拟:\n");
    printf("  路径数: %d\n", paths);
    printf("  每路径步数: %d\n", steps);
    printf("  时间: %.2f ms\n", ms);
    printf("  期权价格: %.4f\n\n", option_price);

    printf("Black-Scholes 解析解:\n");
    printf("  期权价格: %.4f\n", bs_price);
    printf("  误差: %.2f%%\n\n", fabs(option_price - bs_price) / bs_price * 100);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_states));
    CHECK_CUDA(cudaFree(d_prices));
    free(h_prices);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 19: cuRAND 随机数生成                         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n\n", prop.name);

    demoCurandBasics();
    demoHostAPI();
    demoDeviceAPI();
    demoPiMonteCarlo();
    demoQuasiRandom();
    demoOptionPricing();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. cuRAND 生成器类型:\n");
    printf("   ┌──────────────────┬────────────────────────────────┐\n");
    printf("   │ 类型             │ 特点                           │\n");
    printf("   ├──────────────────┼────────────────────────────────┤\n");
    printf("   │ XORWOW (默认)    │ 高质量，平衡性能               │\n");
    printf("   │ MRG32K3A         │ 长周期，高质量                 │\n");
    printf("   │ MTGP32           │ Mersenne Twister               │\n");
    printf("   │ Philox           │ 最高性能，基于计数器           │\n");
    printf("   │ Sobol            │ 准随机，低差异                 │\n");
    printf("   └──────────────────┴────────────────────────────────┘\n\n");

    printf("2. 主机 API vs 设备 API:\n");
    printf("   主机 API:\n");
    printf("     - curandCreateGenerator()\n");
    printf("     - curandGenerate*()\n");
    printf("     - 适合批量生成\n");
    printf("   设备 API:\n");
    printf("     - curand_init() 初始化状态\n");
    printf("     - curand_uniform() 等生成函数\n");
    printf("     - 适合内核中按需生成\n\n");

    printf("3. 分布类型:\n");
    printf("   - 均匀分布: curandGenerateUniform\n");
    printf("   - 正态分布: curandGenerateNormal\n");
    printf("   - 对数正态: curandGenerateLogNormal\n");
    printf("   - 泊松分布: curandGeneratePoisson\n\n");

    printf("4. 应用场景:\n");
    printf("   - 蒙特卡洛模拟\n");
    printf("   - 金融定价\n");
    printf("   - 物理模拟\n");
    printf("   - 机器学习初始化\n");
    printf("   - 随机采样\n\n");

    printf("5. 性能提示:\n");
    printf("   - 主机 API 适合大批量生成\n");
    printf("   - 设备 API 状态占用较多内存\n");
    printf("   - Philox 生成器性能最高\n");
    printf("   - 准随机数加速蒙特卡洛收敛\n\n");

    printf("编译命令:\n");
    printf("  nvcc -lcurand 19_curand.cu -o 19_curand\n\n");

    return 0;
}
