/**
 * =============================================================================
 * CUDA 教程 17: cuFFT 快速傅里叶变换
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 FFT 的基本概念和应用场景
 * 2. 学会使用 cuFFT 进行 1D/2D/3D 变换
 * 3. 掌握实数到复数、复数到复数变换
 * 4. 了解 cuFFT 的性能优化技巧
 *
 * 关键概念：
 * - FFT：将信号从时域转换到频域
 * - cuFFT：NVIDIA 高性能 FFT 库
 * - 支持单精度、双精度和半精度
 *
 * 编译命令：
 *   nvcc -lcufft 17_cufft.cu -o 17_cufft
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"
#include <cufft.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult status = call; \
    if (status != CUFFT_SUCCESS) { \
        printf("cuFFT 错误 %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 第一部分：cuFFT 基础
// ============================================================================

void demoCuFFTBasics() {
    printf("=== 第一部分：cuFFT 基础 ===\n\n");

    printf("FFT 基本概念:\n");
    printf("  - 傅里叶变换：时域 <-> 频域 转换\n");
    printf("  - DFT: 离散傅里叶变换 O(N²)\n");
    printf("  - FFT: 快速傅里叶变换 O(N log N)\n\n");

    printf("cuFFT 变换类型:\n");
    printf("  - C2C: 复数到复数 (Complex to Complex)\n");
    printf("  - R2C: 实数到复数 (Real to Complex)\n");
    printf("  - C2R: 复数到实数 (Complex to Real)\n\n");

    printf("数据类型:\n");
    printf("  - cufftComplex: float2 (单精度复数)\n");
    printf("  - cufftDoubleComplex: double2 (双精度复数)\n");
    printf("  - cufftReal: float (单精度实数)\n");
    printf("  - cufftDoubleReal: double (双精度实数)\n\n");

    printf("cuFFT 工作流程:\n");
    printf("  1. cufftPlan*d() - 创建计划\n");
    printf("  2. cufftExec*() - 执行变换\n");
    printf("  3. cufftDestroy() - 销毁计划\n\n");
}

// ============================================================================
// 第二部分：1D FFT
// ============================================================================

void demo1DFFT() {
    printf("=== 第二部分：1D FFT ===\n\n");

    const int N = 1024;

    // 分配主机内存
    cufftComplex *h_signal = (cufftComplex*)malloc(N * sizeof(cufftComplex));
    cufftComplex *h_result = (cufftComplex*)malloc(N * sizeof(cufftComplex));

    // 生成测试信号：两个正弦波叠加
    // f(t) = sin(2π * 50 * t) + 0.5 * sin(2π * 120 * t)
    float sample_rate = 1000.0f;  // 采样率
    for (int i = 0; i < N; i++) {
        float t = (float)i / sample_rate;
        h_signal[i].x = sinf(2 * M_PI * 50 * t) + 0.5f * sinf(2 * M_PI * 120 * t);
        h_signal[i].y = 0.0f;  // 虚部为0
    }

    printf("信号参数:\n");
    printf("  采样点数: %d\n", N);
    printf("  采样率: %.0f Hz\n", sample_rate);
    printf("  频率成分: 50 Hz (幅度 1.0), 120 Hz (幅度 0.5)\n\n");

    // 分配设备内存
    cufftComplex *d_signal;
    CHECK_CUDA(cudaMalloc(&d_signal, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // 创建 FFT 计划
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 正向 FFT
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        CHECK_CUFFT(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("正向 FFT 性能:\n");
    printf("  时间: %.4f ms\n", ms / 1000);
    printf("  吞吐量: %.2f GFLOPS\n\n", 5.0 * N * log2f(N) / (ms / 1000 * 1e6));

    // 复制结果
    CHECK_CUDA(cudaMemcpy(h_result, d_signal, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // 分析频谱
    printf("频谱分析 (显示主要频率成分):\n");
    float freq_resolution = sample_rate / N;
    for (int i = 0; i < N / 2; i++) {
        float magnitude = sqrtf(h_result[i].x * h_result[i].x +
                                h_result[i].y * h_result[i].y) / N;
        float freq = i * freq_resolution;

        // 只显示幅度较大的频率
        if (magnitude > 0.1f) {
            printf("  频率 %.1f Hz: 幅度 = %.3f\n", freq, magnitude * 2);
        }
    }

    // 逆向 FFT 恢复原信号
    printf("\n逆向 FFT:\n");
    CHECK_CUFFT(cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE));
    CHECK_CUDA(cudaMemcpy(h_result, d_signal, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // 归一化并验证
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float recovered = h_result[i].x / N;  // 需要除以 N 归一化
        float error = fabsf(recovered - h_signal[i].x);
        if (error > max_error) max_error = error;
    }
    printf("  恢复误差 (最大): %.2e\n\n", max_error);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_signal));
    free(h_signal);
    free(h_result);
}

// ============================================================================
// 第三部分：实数 FFT (R2C/C2R)
// ============================================================================

void demoRealFFT() {
    printf("=== 第三部分：实数 FFT ===\n\n");

    const int N = 1024;
    const int COMPLEX_SIZE = N / 2 + 1;  // 实数 FFT 的复数输出大小

    printf("实数 FFT 特性:\n");
    printf("  - 输入: N 个实数\n");
    printf("  - 输出: N/2 + 1 个复数 (Hermitian 对称)\n");
    printf("  - 节省一半内存和计算\n\n");

    // 分配内存
    cufftReal *h_input = (cufftReal*)malloc(N * sizeof(cufftReal));
    cufftComplex *h_output = (cufftComplex*)malloc(COMPLEX_SIZE * sizeof(cufftComplex));
    cufftReal *h_recovered = (cufftReal*)malloc(N * sizeof(cufftReal));

    // 生成实数信号
    for (int i = 0; i < N; i++) {
        float t = (float)i / N;
        h_input[i] = sinf(2 * M_PI * 10 * t) + 0.3f * cosf(2 * M_PI * 30 * t);
    }

    // 分配设备内存
    cufftReal *d_input;
    cufftComplex *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(cufftReal)));
    CHECK_CUDA(cudaMalloc(&d_output, COMPLEX_SIZE * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(cufftReal), cudaMemcpyHostToDevice));

    // 创建 R2C 计划
    cufftHandle planR2C, planC2R;
    CHECK_CUFFT(cufftPlan1d(&planR2C, N, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&planC2R, N, CUFFT_C2R, 1));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // R2C 变换
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        CHECK_CUFFT(cufftExecR2C(planR2C, d_input, d_output));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("R2C 变换: %.4f ms\n", ms / 1000);

    // C2R 变换
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        CHECK_CUFFT(cufftExecC2R(planC2R, d_output, d_input));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("C2R 变换: %.4f ms\n\n", ms / 1000);

    // 验证往返变换
    CHECK_CUFFT(cufftExecR2C(planR2C, d_input, d_output));
    CHECK_CUFFT(cufftExecC2R(planC2R, d_output, d_input));
    CHECK_CUDA(cudaMemcpy(h_recovered, d_input, N * sizeof(cufftReal), cudaMemcpyDeviceToHost));

    // 重新加载原始输入进行比较
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(cufftReal), cudaMemcpyHostToDevice));

    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float recovered = h_recovered[i] / N;  // 归一化
        float error = fabsf(recovered - h_input[i]);
        if (error > max_error) max_error = error;
    }
    printf("往返变换误差: %.2e\n\n", max_error);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(planR2C));
    CHECK_CUFFT(cufftDestroy(planC2R));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
    free(h_recovered);
}

// ============================================================================
// 第四部分：2D FFT
// ============================================================================

void demo2DFFT() {
    printf("=== 第四部分：2D FFT ===\n\n");

    const int NX = 512;
    const int NY = 512;
    const int SIZE = NX * NY;

    printf("2D FFT 尺寸: %d × %d\n\n", NX, NY);

    // 分配内存
    cufftComplex *h_data = (cufftComplex*)malloc(SIZE * sizeof(cufftComplex));
    cufftComplex *h_result = (cufftComplex*)malloc(SIZE * sizeof(cufftComplex));

    // 生成 2D 测试图像：带有特定频率的图案
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            // 创建一个简单的条纹图案
            float value = sinf(2 * M_PI * x * 10 / NX) * sinf(2 * M_PI * y * 5 / NY);
            h_data[y * NX + x].x = value;
            h_data[y * NX + x].y = 0.0f;
        }
    }

    // 分配设备内存
    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, SIZE * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // 创建 2D FFT 计划
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, NY, NX, CUFFT_C2C));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 正向 2D FFT
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("2D FFT 性能:\n");
    printf("  时间: %.3f ms\n", ms / 100);
    printf("  吞吐量: %.2f GFlops\n\n", 5.0 * SIZE * log2f(SIZE) / (ms / 100 * 1e6));

    // 逆向 2D FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaMemcpy(h_result, d_data, SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // 验证
    float max_error = 0.0f;
    for (int i = 0; i < SIZE; i++) {
        float recovered = h_result[i].x / SIZE;
        float error = fabsf(recovered - h_data[i].x);
        if (error > max_error) max_error = error;
    }
    printf("往返变换误差: %.2e\n\n", max_error);

    // 2D 实数 FFT
    printf("2D 实数 FFT (R2C):\n");
    {
        const int COMPLEX_NX = NX / 2 + 1;

        cufftReal *d_real;
        cufftComplex *d_complex;
        CHECK_CUDA(cudaMalloc(&d_real, SIZE * sizeof(cufftReal)));
        CHECK_CUDA(cudaMalloc(&d_complex, NY * COMPLEX_NX * sizeof(cufftComplex)));

        cufftHandle planR2C;
        CHECK_CUFFT(cufftPlan2d(&planR2C, NY, NX, CUFFT_R2C));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            CHECK_CUFFT(cufftExecR2C(planR2C, d_real, d_complex));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  时间: %.3f ms\n", ms / 100);
        printf("  输出尺寸: %d × %d (节省约一半内存)\n\n", NY, COMPLEX_NX);

        CHECK_CUFFT(cufftDestroy(planR2C));
        CHECK_CUDA(cudaFree(d_real));
        CHECK_CUDA(cudaFree(d_complex));
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    free(h_result);
}

// ============================================================================
// 第五部分：批量 FFT
// ============================================================================

void demoBatchFFT() {
    printf("=== 第五部分：批量 FFT ===\n\n");

    const int N = 1024;      // 每个 FFT 的大小
    const int BATCH = 1000;  // 批量数

    printf("批量参数: %d 个 %d 点 FFT\n\n", BATCH, N);

    // 分配内存
    size_t total_size = N * BATCH * sizeof(cufftComplex);
    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, total_size));

    // 初始化
    cufftComplex *h_data = (cufftComplex*)malloc(total_size);
    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < N; i++) {
            float t = (float)i / N;
            h_data[b * N + i].x = sinf(2 * M_PI * (b + 1) * t);
            h_data[b * N + i].y = 0.0f;
        }
    }
    CHECK_CUDA(cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 方法1：使用 cufftPlan1d 批量参数
    printf("方法1: cufftPlan1d with batch:\n");
    {
        cufftHandle plan;
        CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, BATCH));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  时间: %.3f ms (%d 批次)\n", ms / 10, BATCH);
        printf("  每个 FFT: %.4f ms\n", ms / 10 / BATCH);

        CHECK_CUFFT(cufftDestroy(plan));
    }

    // 方法2：使用 cufftPlanMany
    printf("\n方法2: cufftPlanMany (更灵活):\n");
    {
        cufftHandle plan;
        int n[1] = {N};
        int inembed[1] = {N};
        int onembed[1] = {N};
        int istride = 1, ostride = 1;
        int idist = N, odist = N;

        CHECK_CUFFT(cufftPlanMany(&plan, 1, n,
            inembed, istride, idist,
            onembed, ostride, odist,
            CUFFT_C2C, BATCH));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  时间: %.3f ms\n", ms / 10);

        CHECK_CUFFT(cufftDestroy(plan));
    }

    printf("\ncufftPlanMany 参数说明:\n");
    printf("  rank: FFT 维度 (1, 2, 3)\n");
    printf("  n: 每维的大小\n");
    printf("  inembed/onembed: 输入/输出嵌入尺寸\n");
    printf("  istride/ostride: 连续元素间的步长\n");
    printf("  idist/odist: 批次间的距离\n");
    printf("  batch: 批次数量\n\n");

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

// ============================================================================
// 第六部分：卷积应用
// ============================================================================

// 复数乘法核函数
__global__ void complexMultiply(cufftComplex *a, cufftComplex *b,
                                 cufftComplex *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // (a.x + i*a.y) * (b.x + i*b.y) = (a.x*b.x - a.y*b.y) + i*(a.x*b.y + a.y*b.x)
        c[tid].x = a[tid].x * b[tid].x - a[tid].y * b[tid].y;
        c[tid].y = a[tid].x * b[tid].y + a[tid].y * b[tid].x;
    }
}

void demoConvolution() {
    printf("=== 第六部分：FFT 卷积应用 ===\n\n");

    printf("FFT 卷积原理:\n");
    printf("  时域卷积 = 频域乘法\n");
    printf("  f * g = IFFT(FFT(f) · FFT(g))\n\n");

    const int N = 1024;
    const int KERNEL_SIZE = 64;

    // 分配内存
    cufftComplex *h_signal = (cufftComplex*)malloc(N * sizeof(cufftComplex));
    cufftComplex *h_kernel = (cufftComplex*)malloc(N * sizeof(cufftComplex));
    cufftComplex *h_result = (cufftComplex*)malloc(N * sizeof(cufftComplex));

    // 生成信号（方波）
    for (int i = 0; i < N; i++) {
        h_signal[i].x = (i >= N/4 && i < 3*N/4) ? 1.0f : 0.0f;
        h_signal[i].y = 0.0f;
    }

    // 生成卷积核（高斯）
    float sigma = KERNEL_SIZE / 6.0f;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        if (i < KERNEL_SIZE / 2 || i >= N - KERNEL_SIZE / 2) {
            int idx = (i < KERNEL_SIZE / 2) ? i : (i - N);
            h_kernel[i].x = expf(-idx * idx / (2 * sigma * sigma));
            sum += h_kernel[i].x;
        } else {
            h_kernel[i].x = 0.0f;
        }
        h_kernel[i].y = 0.0f;
    }
    // 归一化
    for (int i = 0; i < N; i++) h_kernel[i].x /= sum;

    // 分配设备内存
    cufftComplex *d_signal, *d_kernel, *d_result;
    CHECK_CUDA(cudaMalloc(&d_signal, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_kernel, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(cufftComplex)));

    CHECK_CUDA(cudaMemcpy(d_signal, h_signal, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // 创建 FFT 计划
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 1. FFT 信号
    CHECK_CUFFT(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD));

    // 2. FFT 卷积核
    CHECK_CUFFT(cufftExecC2C(plan, d_kernel, d_kernel, CUFFT_FORWARD));

    // 3. 频域乘法
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    complexMultiply<<<gridSize, blockSize>>>(d_signal, d_kernel, d_result, N);

    // 4. IFFT
    CHECK_CUFFT(cufftExecC2C(plan, d_result, d_result, CUFFT_INVERSE));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("FFT 卷积时间: %.3f ms\n", ms);

    // 复制结果
    CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // 显示结果样本
    printf("\n卷积结果样本 (高斯模糊后的方波边缘):\n");
    printf("  位置:   ");
    for (int i = N/4 - 5; i < N/4 + 5; i++) printf("%4d ", i);
    printf("\n  幅值:   ");
    for (int i = N/4 - 5; i < N/4 + 5; i++) {
        printf("%.2f ", h_result[i].x / N);
    }
    printf("\n\n");

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_signal));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_result));
    free(h_signal);
    free(h_kernel);
    free(h_result);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 17: cuFFT 快速傅里叶变换                       ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n\n", prop.name);

    demoCuFFTBasics();
    demo1DFFT();
    demoRealFFT();
    demo2DFFT();
    demoBatchFFT();
    demoConvolution();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. cuFFT 计划创建:\n");
    printf("   - cufftPlan1d(): 1D FFT\n");
    printf("   - cufftPlan2d(): 2D FFT\n");
    printf("   - cufftPlan3d(): 3D FFT\n");
    printf("   - cufftPlanMany(): 批量/高级配置\n\n");

    printf("2. 变换类型:\n");
    printf("   ┌──────────┬────────────┬─────────────────────────┐\n");
    printf("   │ 类型     │ 函数       │ 说明                    │\n");
    printf("   ├──────────┼────────────┼─────────────────────────┤\n");
    printf("   │ C2C      │ cufftExecC2C│ 复数到复数              │\n");
    printf("   │ R2C      │ cufftExecR2C│ 实数到复数 (正向)       │\n");
    printf("   │ C2R      │ cufftExecC2R│ 复数到实数 (逆向)       │\n");
    printf("   │ Z2Z      │ cufftExecZ2Z│ 双精度复数              │\n");
    printf("   │ D2Z      │ cufftExecD2Z│ 双精度实数到复数        │\n");
    printf("   └──────────┴────────────┴─────────────────────────┘\n\n");

    printf("3. 实数 FFT 优势:\n");
    printf("   - 输出大小: N/2 + 1 (利用 Hermitian 对称)\n");
    printf("   - 节省约一半内存和计算\n");
    printf("   - 适合实际信号处理\n\n");

    printf("4. 性能优化:\n");
    printf("   - 使用 2 的幂次大小获得最佳性能\n");
    printf("   - 批量处理多个 FFT\n");
    printf("   - 重用 FFT 计划\n");
    printf("   - 使用 cufftSetStream 与其他操作重叠\n\n");

    printf("5. 应用场景:\n");
    printf("   - 信号处理和频谱分析\n");
    printf("   - 图像滤波和卷积\n");
    printf("   - 音频处理\n");
    printf("   - 科学计算和模拟\n\n");

    printf("编译命令:\n");
    printf("  nvcc -lcufft 17_cufft.cu -o 17_cufft\n\n");

    return 0;
}
