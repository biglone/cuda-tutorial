/**
 * =============================================================================
 * CUDA 教程 25: CUDA 与深度学习框架集成
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 CUDA 在深度学习中的角色
 * 2. 学会编写自定义 CUDA 算子
 * 3. 掌握与 PyTorch/TensorFlow 的集成方法
 * 4. 了解深度学习专用优化技术
 *
 * 关键概念：
 * - 自定义算子 (Custom Operators)
 * - Tensor 内存布局
 * - 混合精度计算 (FP16/BF16)
 * - 算子融合 (Operator Fusion)
 *
 * 编译命令：
 *   nvcc 25_deep_learning_integration.cu -o 25_deep_learning_integration
 *
 * PyTorch 扩展编译:
 *   python setup.py install
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：深度学习与 CUDA 概述
// ============================================================================

void demoDeepLearningOverview() {
    printf("=== 第一部分：深度学习与 CUDA 概述 ===\n\n");

    printf("深度学习计算特点:\n");
    printf("  - 大规模矩阵/张量运算\n");
    printf("  - 高度并行的计算模式\n");
    printf("  - 内存带宽密集\n");
    printf("  - 需要特殊精度支持 (FP16, BF16, INT8)\n\n");

    printf("CUDA 在 DL 框架中的位置:\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  Python API (PyTorch / TensorFlow / JAX)        │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  框架 C++ Backend                               │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  CUDA Libraries (cuDNN, cuBLAS, etc.)           │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  Custom CUDA Kernels                            │ ← 本教程\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  CUDA Runtime / Driver                          │\n");
    printf("  ├─────────────────────────────────────────────────┤\n");
    printf("  │  GPU Hardware                                   │\n");
    printf("  └─────────────────────────────────────────────────┘\n\n");

    printf("为什么需要自定义 CUDA 算子:\n");
    printf("  1. 框架未提供的特殊操作\n");
    printf("  2. 多个操作融合以减少内存访问\n");
    printf("  3. 针对特定模型的优化\n");
    printf("  4. 研究新算法/新架构\n\n");

    printf("主要深度学习 CUDA 库:\n");
    printf("  - cuDNN: 深度神经网络原语\n");
    printf("  - cuBLAS: 矩阵运算\n");
    printf("  - CUTLASS: 可定制的矩阵乘法\n");
    printf("  - TensorRT: 推理优化\n");
    printf("  - Triton: 编译器生成内核\n\n");
}

// ============================================================================
// 第二部分：常见激活函数实现
// ============================================================================

// ReLU 前向
__global__ void reluForward(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = fmaxf(0.0f, input[tid]);
    }
}

// ReLU 反向
__global__ void reluBackward(float *grad_input, const float *grad_output,
                              const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        grad_input[tid] = (input[tid] > 0) ? grad_output[tid] : 0.0f;
    }
}

// GELU 前向 (Gaussian Error Linear Unit)
// GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void geluForward(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[tid] = x * cdf;
    }
}

// GELU 反向
__global__ void geluBackward(float *grad_input, const float *grad_output,
                              const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float cdf = 0.5f * (1.0f + tanh_inner);
        float pdf = 0.5f * 0.7978845608f * (1.0f + 0.134145f * x * x) *
                    (1.0f - tanh_inner * tanh_inner);
        grad_input[tid] = grad_output[tid] * (cdf + x * pdf);
    }
}

// SiLU/Swish 前向: x * sigmoid(x)
__global__ void siluForward(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[tid] = x * sigmoid_x;
    }
}

void demoActivationFunctions() {
    printf("=== 第二部分：激活函数实现 ===\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_input, *d_output, *d_grad_input, *d_grad_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMalloc(&d_grad_input, size));
    CHECK_CUDA(cudaMalloc(&d_grad_output, size));

    // 初始化
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_input[i] = (float)(i - N/2) / (N/10);
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // 初始化梯度为 1
    float *h_grad = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_grad[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("激活函数性能测试 (N=%d):\n\n", N);

    // ReLU
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        reluForward<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float reluTime;
    CHECK_CUDA(cudaEventElapsedTime(&reluTime, start, stop));
    printf("  ReLU 前向:  %.3f ms (100次)\n", reluTime);

    // GELU
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        geluForward<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float geluTime;
    CHECK_CUDA(cudaEventElapsedTime(&geluTime, start, stop));
    printf("  GELU 前向:  %.3f ms (100次)\n", geluTime);

    // SiLU
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        siluForward<<<gridSize, blockSize>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float siluTime;
    CHECK_CUDA(cudaEventElapsedTime(&siluTime, start, stop));
    printf("  SiLU 前向:  %.3f ms (100次)\n\n", siluTime);

    // 验证输出
    float *h_output = (float*)malloc(size);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    printf("验证 (x=1.0 的输出):\n");
    int testIdx = N/2 + N/10;  // x ≈ 1.0
    geluForward<<<gridSize, blockSize>>>(d_output, d_input, N);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("  GELU(1.0) ≈ %.4f (期望 ≈ 0.8413)\n\n", h_output[testIdx]);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_grad);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_grad_input));
    CHECK_CUDA(cudaFree(d_grad_output));
}

// ============================================================================
// 第三部分：LayerNorm 实现
// ============================================================================

// 简化版 LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
__global__ void layerNormForward(float *output, const float *input,
                                   const float *gamma, const float *beta,
                                   int batch_size, int hidden_size, float eps) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float *s_mean = sdata;
    float *s_var = sdata + 1;

    // 计算均值
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += input[batch_idx * hidden_size + i];
    }

    // Warp 内归约
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // 每个 warp 的结果写入共享内存
    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // 第一个 warp 进行最终归约
    if (warp_id == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) {
            *s_mean = sum / hidden_size;
        }
    }
    __syncthreads();

    float mean = *s_mean;

    // 计算方差
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = input[batch_idx * hidden_size + i] - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }

    if (lane == 0) {
        warp_sums[warp_id] = var_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
        }
        if (lane == 0) {
            *s_var = var_sum / hidden_size;
        }
    }
    __syncthreads();

    float variance = *s_var;
    float inv_std = rsqrtf(variance + eps);

    // 归一化并应用 gamma, beta
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[batch_idx * hidden_size + i];
        float normalized = (x - mean) * inv_std;
        output[batch_idx * hidden_size + i] = normalized * gamma[i] + beta[i];
    }
}

void demoLayerNorm() {
    printf("=== 第三部分：LayerNorm 实现 ===\n\n");

    const int batch_size = 32;
    const int hidden_size = 1024;
    const int total_size = batch_size * hidden_size;
    const float eps = 1e-5f;

    float *d_input, *d_output, *d_gamma, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_input, total_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, hidden_size * sizeof(float)));

    // 初始化
    float *h_input = (float*)malloc(total_size * sizeof(float));
    float *h_gamma = (float*)malloc(hidden_size * sizeof(float));
    float *h_beta = (float*)malloc(hidden_size * sizeof(float));

    for (int i = 0; i < total_size; i++) h_input[i] = (float)(rand() % 1000) / 1000.0f;
    for (int i = 0; i < hidden_size; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // 配置: 每个 batch 一个 block
    int blockSize = 256;
    size_t sharedMem = 2 * sizeof(float);  // mean 和 var

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("LayerNorm 配置:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Hidden size: %d\n\n", hidden_size);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        layerNormForward<<<batch_size, blockSize, sharedMem>>>(
            d_output, d_input, d_gamma, d_beta, batch_size, hidden_size, eps);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    printf("性能: %.3f ms (100次)\n", elapsed);
    printf("吞吐量: %.2f GB/s\n\n", 2.0f * total_size * sizeof(float) * 100 / elapsed / 1e6);

    // 验证
    float *h_output = (float*)malloc(total_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 检查第一个 batch 的统计信息
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        mean += h_output[i];
    }
    mean /= hidden_size;
    for (int i = 0; i < hidden_size; i++) {
        float diff = h_output[i] - mean;
        var += diff * diff;
    }
    var /= hidden_size;

    printf("验证 (第一个 batch):\n");
    printf("  输出均值: %.6f (期望 ≈ 0)\n", mean);
    printf("  输出方差: %.6f (期望 ≈ 1)\n\n", var);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_gamma);
    free(h_beta);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));
}

// ============================================================================
// 第四部分：Softmax 实现
// ============================================================================

// 在线 Softmax: 数值稳定的实现
__global__ void softmaxForward(float *output, const float *input,
                                int batch_size, int seq_len) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // 第一步: 找最大值
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, input[batch_idx * seq_len + i]);
    }

    // Warp 内找最大值
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    __shared__ float warp_maxes[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        warp_maxes[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_maxes[lane] : -INFINITY;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        }
        if (lane == 0) {
            sdata[0] = local_max;
        }
    }
    __syncthreads();

    float max_val = sdata[0];

    // 第二步: 计算 exp 和求和
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_sum += expf(input[batch_idx * seq_len + i] - max_val);
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (lane == 0) {
        warp_maxes[warp_id] = local_sum;  // 复用
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_maxes[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) {
            sdata[0] = local_sum;
        }
    }
    __syncthreads();

    float sum_exp = sdata[0];

    // 第三步: 归一化
    for (int i = tid; i < seq_len; i += blockDim.x) {
        output[batch_idx * seq_len + i] =
            expf(input[batch_idx * seq_len + i] - max_val) / sum_exp;
    }
}

void demoSoftmax() {
    printf("=== 第四部分：Softmax 实现 ===\n\n");

    const int batch_size = 32;
    const int seq_len = 512;
    const int total_size = batch_size * seq_len;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, total_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total_size * sizeof(float)));

    float *h_input = (float*)malloc(total_size * sizeof(float));
    for (int i = 0; i < total_size; i++) h_input[i] = (float)(rand() % 1000) / 100.0f;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    size_t sharedMem = sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("Softmax 配置:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Sequence length: %d\n\n", seq_len);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        softmaxForward<<<batch_size, blockSize, sharedMem>>>(
            d_output, d_input, batch_size, seq_len);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    printf("性能: %.3f ms (1000次)\n\n", elapsed);

    // 验证
    float *h_output = (float*)malloc(total_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += h_output[i];
    }
    printf("验证 (第一行概率和): %.6f (期望 = 1.0)\n\n", sum);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第五部分：FP16 混合精度计算
// ============================================================================

// FP16 向量加法
__global__ void vectorAddFP16(half *c, const half *a, const half *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 使用 half2 进行向量化操作 (2x 吞吐)
    int idx = tid * 2;
    if (idx + 1 < n) {
        half2 a2 = *reinterpret_cast<const half2*>(&a[idx]);
        half2 b2 = *reinterpret_cast<const half2*>(&b[idx]);
        *reinterpret_cast<half2*>(&c[idx]) = __hadd2(a2, b2);
    } else if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// FP16 GELU
__global__ void geluFP16(half *output, const half *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = __half2float(input[tid]);
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[tid] = __float2half(x * cdf);
}

void demoMixedPrecision() {
    printf("=== 第五部分：FP16 混合精度 ===\n\n");

    printf("混合精度训练优势:\n");
    printf("  - 内存占用减半\n");
    printf("  - 计算速度提升 (Tensor Cores)\n");
    printf("  - 带宽效率提高\n\n");

    printf("注意事项:\n");
    printf("  - 需要损失缩放 (Loss Scaling)\n");
    printf("  - 某些操作保持 FP32 (归一化, Softmax)\n");
    printf("  - 梯度可能溢出/下溢\n\n");

    const int N = 1 << 20;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // FP32 测试
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    CHECK_CUDA(cudaMalloc(&d_a_fp32, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_fp32, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c_fp32, N * sizeof(float)));

    // FP16 测试
    half *d_a_fp16, *d_b_fp16, *d_c_fp16;
    CHECK_CUDA(cudaMalloc(&d_a_fp16, N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b_fp16, N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_c_fp16, N * sizeof(half)));

    // 初始化
    float *h_data = (float*)malloc(N * sizeof(float));
    half *h_data_fp16 = (half*)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i / N;
        h_data_fp16[i] = __float2half(h_data[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_a_fp32, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp32, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a_fp16, h_data_fp16, N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_fp16, h_data_fp16, N * sizeof(half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("性能对比 (N=%d):\n", N);

    // FP32
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        geluForward<<<gridSize, blockSize>>>(d_c_fp32, d_a_fp32, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float fp32Time;
    CHECK_CUDA(cudaEventElapsedTime(&fp32Time, start, stop));

    // FP16
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        geluFP16<<<gridSize, blockSize>>>(d_c_fp16, d_a_fp16, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float fp16Time;
    CHECK_CUDA(cudaEventElapsedTime(&fp16Time, start, stop));

    printf("  FP32 GELU: %.3f ms (100次)\n", fp32Time);
    printf("  FP16 GELU: %.3f ms (100次)\n", fp16Time);
    printf("  加速比: %.2fx\n", fp32Time / fp16Time);
    printf("  内存节省: 50%%\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_data);
    free(h_data_fp16);
    CHECK_CUDA(cudaFree(d_a_fp32));
    CHECK_CUDA(cudaFree(d_b_fp32));
    CHECK_CUDA(cudaFree(d_c_fp32));
    CHECK_CUDA(cudaFree(d_a_fp16));
    CHECK_CUDA(cudaFree(d_b_fp16));
    CHECK_CUDA(cudaFree(d_c_fp16));
}

// ============================================================================
// 第六部分：算子融合示例
// ============================================================================

// 未融合版本: 三个独立内核
__global__ void addKernel(float *c, const float *a, const float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

__global__ void mulKernel(float *c, const float *a, const float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] * b[tid];
}

// 融合版本: bias_add + gelu
__global__ void biasGeluFused(float *output, const float *input,
                               const float *bias, int batch_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden_size;
    if (idx >= total) return;

    int hidden_idx = idx % hidden_size;
    float x = input[idx] + bias[hidden_idx];

    // GELU
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}

// 更复杂的融合: add + layernorm + dropout
__global__ void residualLayerNormDropout(
    float *output, const float *input, const float *residual,
    const float *gamma, const float *beta,
    const float *dropout_mask, float dropout_prob,
    int batch_size, int hidden_size, float eps) {

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // 1. 残差连接
    // 2. LayerNorm
    // 3. Dropout
    // 所有操作融合在一起

    // 首先计算残差和
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * hidden_size + i;
        float val = input[idx] + residual[idx];  // 残差连接
        sum += val;
    }

    // 归约求均值...
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane == 0) sdata[0] = sum / hidden_size;
    }
    __syncthreads();

    float mean = sdata[0];

    // 计算方差
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * hidden_size + i;
        float val = input[idx] + residual[idx];
        float diff = val - mean;
        var_sum += diff * diff;
    }

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    if (lane == 0) warp_sums[warp_id] = var_sum;
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
        }
        if (lane == 0) sdata[0] = var_sum / hidden_size;
    }
    __syncthreads();

    float inv_std = rsqrtf(sdata[0] + eps);

    // 归一化 + dropout
    float scale = 1.0f / (1.0f - dropout_prob);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * hidden_size + i;
        float val = input[idx] + residual[idx];
        float normalized = (val - mean) * inv_std;
        float out = normalized * gamma[i] + beta[i];

        // Dropout (inference mode: 不应用)
        // output[idx] = out * dropout_mask[idx] * scale;
        output[idx] = out;
    }
}

void demoOperatorFusion() {
    printf("=== 第六部分：算子融合 ===\n\n");

    printf("算子融合优势:\n");
    printf("  - 减少内存访问 (中间结果不写回全局内存)\n");
    printf("  - 减少内核启动开销\n");
    printf("  - 提高缓存利用率\n\n");

    const int batch_size = 32;
    const int hidden_size = 1024;
    const int total = batch_size * hidden_size;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    float *d_input, *d_bias, *d_output1, *d_output2, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_input, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output1, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output2, total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp, total * sizeof(float)));

    // 初始化
    float *h_data = (float*)malloc(total * sizeof(float));
    float *h_bias = (float*)malloc(hidden_size * sizeof(float));
    for (int i = 0; i < total; i++) h_data[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < hidden_size; i++) h_bias[i] = 0.1f;
    CHECK_CUDA(cudaMemcpy(d_input, h_data, total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("对比: Bias + GELU (batch=%d, hidden=%d)\n\n", batch_size, hidden_size);

    // 未融合版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        // 分开的 bias add
        addKernel<<<gridSize, blockSize>>>(d_temp, d_input, d_bias, total);
        // 分开的 GELU
        geluForward<<<gridSize, blockSize>>>(d_output1, d_temp, total);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float unfusedTime;
    CHECK_CUDA(cudaEventElapsedTime(&unfusedTime, start, stop));

    // 融合版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        biasGeluFused<<<gridSize, blockSize>>>(
            d_output2, d_input, d_bias, batch_size, hidden_size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float fusedTime;
    CHECK_CUDA(cudaEventElapsedTime(&fusedTime, start, stop));

    printf("  未融合 (2个内核): %.3f ms (1000次)\n", unfusedTime);
    printf("  融合 (1个内核):   %.3f ms (1000次)\n", fusedTime);
    printf("  加速比: %.2fx\n\n", unfusedTime / fusedTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_data);
    free(h_bias);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_output1));
    CHECK_CUDA(cudaFree(d_output2));
    CHECK_CUDA(cudaFree(d_temp));
}

// ============================================================================
// 第七部分：PyTorch 集成示例代码
// ============================================================================

void demoPyTorchIntegration() {
    printf("=== 第七部分：PyTorch 集成 ===\n\n");

    printf("方法 1: torch.utils.cpp_extension\n");
    printf("─────────────────────────────────\n\n");

    printf("// my_cuda_kernel.cu\n");
    printf("#include <torch/extension.h>\n");
    printf("#include <cuda.h>\n");
    printf("#include <cuda_runtime.h>\n\n");

    printf("__global__ void gelu_kernel(float *output, const float *input, int n) {\n");
    printf("    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");
    printf("    if (tid < n) {\n");
    printf("        float x = input[tid];\n");
    printf("        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));\n");
    printf("        output[tid] = x * cdf;\n");
    printf("    }\n");
    printf("}\n\n");

    printf("torch::Tensor gelu_cuda(torch::Tensor input) {\n");
    printf("    auto output = torch::empty_like(input);\n");
    printf("    int n = input.numel();\n");
    printf("    int blockSize = 256;\n");
    printf("    int gridSize = (n + blockSize - 1) / blockSize;\n");
    printf("    \n");
    printf("    gelu_kernel<<<gridSize, blockSize>>>(\n");
    printf("        output.data_ptr<float>(),\n");
    printf("        input.data_ptr<float>(),\n");
    printf("        n);\n");
    printf("    \n");
    printf("    return output;\n");
    printf("}\n\n");

    printf("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n");
    printf("    m.def(\"gelu\", &gelu_cuda, \"GELU (CUDA)\");\n");
    printf("}\n\n");

    printf("// setup.py\n");
    printf("from setuptools import setup\n");
    printf("from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n\n");

    printf("setup(\n");
    printf("    name='my_cuda_kernel',\n");
    printf("    ext_modules=[\n");
    printf("        CUDAExtension('my_cuda_kernel', [\n");
    printf("            'my_cuda_kernel.cu',\n");
    printf("        ])\n");
    printf("    ],\n");
    printf("    cmdclass={'build_ext': BuildExtension}\n");
    printf(")\n\n");

    printf("// Python 使用\n");
    printf("import torch\n");
    printf("import my_cuda_kernel\n\n");

    printf("x = torch.randn(1024, device='cuda')\n");
    printf("y = my_cuda_kernel.gelu(x)\n\n");

    printf("─────────────────────────────────\n\n");

    printf("方法 2: JIT 编译 (开发时推荐)\n");
    printf("─────────────────────────────────\n\n");

    printf("from torch.utils.cpp_extension import load\n\n");

    printf("my_kernel = load(\n");
    printf("    name='my_cuda_kernel',\n");
    printf("    sources=['my_cuda_kernel.cu'],\n");
    printf("    verbose=True\n");
    printf(")\n\n");

    printf("x = torch.randn(1024, device='cuda')\n");
    printf("y = my_kernel.gelu(x)\n\n");

    printf("─────────────────────────────────\n\n");

    printf("方法 3: torch.autograd.Function (支持反向传播)\n");
    printf("─────────────────────────────────\n\n");

    printf("class GELUFunction(torch.autograd.Function):\n");
    printf("    @staticmethod\n");
    printf("    def forward(ctx, input):\n");
    printf("        ctx.save_for_backward(input)\n");
    printf("        return my_kernel.gelu_forward(input)\n");
    printf("    \n");
    printf("    @staticmethod\n");
    printf("    def backward(ctx, grad_output):\n");
    printf("        input, = ctx.saved_tensors\n");
    printf("        return my_kernel.gelu_backward(grad_output, input)\n\n");

    printf("# 使用\n");
    printf("gelu = GELUFunction.apply\n");
    printf("y = gelu(x)\n");
    printf("y.sum().backward()\n\n");
}

// ============================================================================
// 第八部分：最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第八部分：最佳实践 ===\n\n");

    printf("1. 内存布局:\n");
    printf("   PyTorch: NCHW (默认) 或 NHWC (channels_last)\n");
    printf("   TensorFlow: NHWC (默认) 或 NCHW\n");
    printf("   cuDNN: 支持多种布局，NHWC 在新 GPU 上更快\n\n");

    printf("2. 数值稳定性:\n");
    printf("   - Softmax: 先减去最大值再 exp\n");
    printf("   - LayerNorm: 使用 Welford 算法计算方差\n");
    printf("   - 损失函数: 使用 log-sum-exp 技巧\n\n");

    printf("3. 混合精度训练:\n");
    printf("   - 前向/反向传播: FP16\n");
    printf("   - 权重主拷贝: FP32\n");
    printf("   - 损失缩放: 防止梯度下溢\n");
    printf("   - 某些操作保持 FP32: LayerNorm, Softmax\n\n");

    printf("4. 算子融合策略:\n");
    printf("   高优先级融合:\n");
    printf("   - Bias + Activation\n");
    printf("   - MatMul + Bias + Activation\n");
    printf("   - LayerNorm + Dropout\n");
    printf("   - 残差连接 + LayerNorm\n\n");

    printf("5. 调试技巧:\n");
    printf("   // CUDA 同步检查\n");
    printf("   export CUDA_LAUNCH_BLOCKING=1\n\n");

    printf("   // 检查数值\n");
    printf("   torch.autograd.set_detect_anomaly(True)\n\n");

    printf("   // 打印中间结果\n");
    printf("   if (tid == 0) printf(\"debug: %%f\\n\", value);\n\n");

    printf("6. 性能优化检查清单:\n");
    printf("   □ 使用 Tensor Cores (需要对齐到 8/16)\n");
    printf("   □ 内存访问合并\n");
    printf("   □ 算子融合\n");
    printf("   □ 异步数据加载\n");
    printf("   □ 梯度累积减少通信\n");
    printf("   □ 使用 cuDNN/cuBLAS 而非手写内核\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 25: CUDA 与深度学习框架集成                          ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);

    // 检查 Tensor Core 支持
    bool hasTensorCores = prop.major >= 7;
    printf("Tensor Cores: %s\n\n", hasTensorCores ? "支持" : "不支持");

    demoDeepLearningOverview();
    demoActivationFunctions();
    demoLayerNorm();
    demoSoftmax();
    demoMixedPrecision();
    demoOperatorFusion();
    demoPyTorchIntegration();
    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 常见 DL 算子:\n");
    printf("   - 激活函数: ReLU, GELU, SiLU\n");
    printf("   - 归一化: LayerNorm, BatchNorm\n");
    printf("   - 注意力: Softmax, Flash Attention\n\n");

    printf("2. 优化技术:\n");
    printf("   - 混合精度 (FP16/BF16)\n");
    printf("   - 算子融合\n");
    printf("   - Tensor Cores\n\n");

    printf("3. 框架集成:\n");
    printf("   - PyTorch: torch.utils.cpp_extension\n");
    printf("   - TensorFlow: tf.load_op_library\n");
    printf("   - 支持自动微分\n\n");

    printf("4. 工具:\n");
    printf("   - CUTLASS: 可定制矩阵乘法\n");
    printf("   - Triton: 高级内核编程\n");
    printf("   - TensorRT: 推理优化\n\n");

    return 0;
}
