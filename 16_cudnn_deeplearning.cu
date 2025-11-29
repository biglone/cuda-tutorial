/**
 * =============================================================================
 * CUDA 教程 16: cuDNN 深度学习加速
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 cuDNN 库的基本概念和架构
 * 2. 学会使用 cuDNN 进行卷积操作
 * 3. 掌握池化、激活函数等常用操作
 * 4. 了解 cuDNN 的性能优化技巧
 *
 * 关键概念：
 * - cuDNN 是 NVIDIA 深度学习原语库
 * - 提供高度优化的卷积、池化、归一化等操作
 * - 支持多种数据格式和算法选择
 *
 * 编译命令：
 *   nvcc -lcudnn 16_cudnn_deeplearning.cu -o 16_cudnn_deeplearning
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"
#include <cudnn.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudnnGetErrorString(status)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：cuDNN 基础
// ============================================================================

void demoCuDNNBasics() {
    printf("=== 第一部分：cuDNN 基础 ===\n\n");

    // 创建 cuDNN 句柄
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 获取版本信息
    size_t version = cudnnGetVersion();
    printf("cuDNN 版本: %zu.%zu.%zu\n",
           version / 1000, (version % 1000) / 100, version % 100);

    size_t cudartVersion = cudnnGetCudartVersion();
    printf("CUDA Runtime 版本: %zu.%zu\n\n",
           cudartVersion / 1000, (cudartVersion % 1000) / 10);

    printf("cuDNN 核心概念:\n");
    printf("  1. Handle (句柄): 管理 cuDNN 资源\n");
    printf("  2. Tensor Descriptor: 描述张量的形状和格式\n");
    printf("  3. Filter Descriptor: 描述卷积核\n");
    printf("  4. Convolution Descriptor: 描述卷积操作参数\n");
    printf("  5. Algorithm: 选择最优的计算算法\n\n");

    printf("数据格式:\n");
    printf("  - NCHW: 批量×通道×高×宽 (默认)\n");
    printf("  - NHWC: 批量×高×宽×通道 (TensorFlow 风格)\n");
    printf("  - NC/32HW32: Tensor Core 优化格式\n\n");

    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// ============================================================================
// 第二部分：卷积操作
// ============================================================================

void demoConvolution() {
    printf("=== 第二部分：卷积操作 ===\n\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 输入参数
    const int batch_size = 1;
    const int in_channels = 3;
    const int in_height = 224;
    const int in_width = 224;

    // 卷积核参数
    const int out_channels = 64;
    const int kernel_size = 3;
    const int padding = 1;
    const int stride = 1;

    // 输出尺寸
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    printf("输入: %d × %d × %d × %d (NCHW)\n",
           batch_size, in_channels, in_height, in_width);
    printf("卷积核: %d × %d × %d × %d\n",
           out_channels, in_channels, kernel_size, kernel_size);
    printf("输出: %d × %d × %d × %d\n\n",
           batch_size, out_channels, out_height, out_width);

    // 创建张量描述符
    cudnnTensorDescriptor_t input_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in_channels, in_height, in_width));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, out_channels, out_height, out_width));

    // 创建卷积核描述符
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc,
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_size, kernel_size));

    // 创建卷积描述符
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
        padding, padding,     // padding
        stride, stride,       // stride
        1, 1,                 // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // 获取最佳算法
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
        input_desc, filter_desc, conv_desc, output_desc,
        10, &returnedAlgoCount, perfResults));

    printf("可用算法 (前3个):\n");
    const char* algoNames[] = {
        "IMPLICIT_GEMM", "IMPLICIT_PRECOMP_GEMM", "GEMM",
        "DIRECT", "FFT", "FFT_TILING", "WINOGRAD", "WINOGRAD_NONFUSED"
    };
    for (int i = 0; i < 3 && i < returnedAlgoCount; i++) {
        int algoIdx = perfResults[i].algo;
        printf("  %d. %s - 时间: %.3f ms, 内存: %.2f MB\n",
               i + 1,
               algoIdx < 8 ? algoNames[algoIdx] : "UNKNOWN",
               perfResults[i].time,
               perfResults[i].memory / (1024.0 * 1024.0));
    }

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;

    // 获取工作空间大小
    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_desc, filter_desc, conv_desc, output_desc,
        algo, &workspace_size));

    printf("\n选择算法: %s\n", algo < 8 ? algoNames[algo] : "UNKNOWN");
    printf("工作空间: %.2f MB\n\n", workspace_size / (1024.0 * 1024.0));

    // 分配内存
    float *d_input, *d_filter, *d_output, *d_workspace;
    size_t input_size = batch_size * in_channels * in_height * in_width * sizeof(float);
    size_t filter_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = batch_size * out_channels * out_height * out_width * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
    } else {
        d_workspace = nullptr;
    }

    // 初始化数据
    float *h_input = (float*)malloc(input_size);
    float *h_filter = (float*)malloc(filter_size);
    for (size_t i = 0; i < input_size / sizeof(float); i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    for (size_t i = 0; i < filter_size / sizeof(float); i++) {
        h_filter[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice));

    // 执行卷积
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    CHECK_CUDNN(cudnnConvolutionForward(cudnn,
        &alpha, input_desc, d_input,
        filter_desc, d_filter,
        conv_desc, algo, d_workspace, workspace_size,
        &beta, output_desc, d_output));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn,
            &alpha, input_desc, d_input,
            filter_desc, d_filter,
            conv_desc, algo, d_workspace, workspace_size,
            &beta, output_desc, d_output));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算 FLOPS
    double flops = 2.0 * batch_size * out_channels * out_height * out_width *
                   in_channels * kernel_size * kernel_size;
    double gflops = flops / (ms / 100 * 1e6);

    printf("卷积性能:\n");
    printf("  时间: %.3f ms\n", ms / 100);
    printf("  吞吐量: %.2f GFLOPS\n\n", gflops);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    free(h_input);
    free(h_filter);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// ============================================================================
// 第三部分：池化操作
// ============================================================================

void demoPooling() {
    printf("=== 第三部分：池化操作 ===\n\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    const int batch_size = 1;
    const int channels = 64;
    const int in_height = 224;
    const int in_width = 224;
    const int pool_size = 2;
    const int pool_stride = 2;

    const int out_height = in_height / pool_stride;
    const int out_width = in_width / pool_stride;

    printf("输入: %d × %d × %d × %d\n", batch_size, channels, in_height, in_width);
    printf("池化: %d × %d, stride=%d\n", pool_size, pool_size, pool_stride);
    printf("输出: %d × %d × %d × %d\n\n", batch_size, channels, out_height, out_width);

    // 创建描述符
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pool_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, channels, in_height, in_width));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, channels, out_height, out_width));

    // 测试不同池化类型
    cudnnPoolingMode_t poolModes[] = {
        CUDNN_POOLING_MAX,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    };
    const char* poolNames[] = {"Max Pooling", "Avg Pooling (包含padding)", "Avg Pooling (排除padding)"};

    // 分配内存
    size_t input_size = batch_size * channels * in_height * in_width * sizeof(float);
    size_t output_size = batch_size * channels * out_height * out_width * sizeof(float);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    // 初始化
    float *h_input = (float*)malloc(input_size);
    for (size_t i = 0; i < input_size / sizeof(float); i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int m = 0; m < 3; m++) {
        CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool_desc,
            poolModes[m], CUDNN_NOT_PROPAGATE_NAN,
            pool_size, pool_size,
            0, 0,  // padding
            pool_stride, pool_stride));

        // 预热
        CHECK_CUDNN(cudnnPoolingForward(cudnn, pool_desc,
            &alpha, input_desc, d_input,
            &beta, output_desc, d_output));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            CHECK_CUDNN(cudnnPoolingForward(cudnn, pool_desc,
                &alpha, input_desc, d_input,
                &beta, output_desc, d_output));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("%s: %.3f ms\n", poolNames[m], ms / 100);
    }

    printf("\n");

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pool_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// ============================================================================
// 第四部分：激活函数
// ============================================================================

void demoActivation() {
    printf("=== 第四部分：激活函数 ===\n\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    const int n = 1, c = 256, h = 56, w = 56;
    const int size = n * c * h * w;

    // 创建描述符
    cudnnTensorDescriptor_t tensor_desc;
    cudnnActivationDescriptor_t activation_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor_desc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensor_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // 分配内存
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, size * sizeof(float)));

    // 初始化
    float *h_input = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 200 - 100) / 100.0f;  // [-1, 1]
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    // 测试不同激活函数
    struct {
        cudnnActivationMode_t mode;
        const char* name;
        double coef;
    } activations[] = {
        {CUDNN_ACTIVATION_RELU, "ReLU", 0.0},
        {CUDNN_ACTIVATION_SIGMOID, "Sigmoid", 0.0},
        {CUDNN_ACTIVATION_TANH, "Tanh", 0.0},
        {CUDNN_ACTIVATION_CLIPPED_RELU, "Clipped ReLU (6.0)", 6.0},
        {CUDNN_ACTIVATION_ELU, "ELU", 1.0},
    };

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("激活函数性能 (%d × %d × %d × %d = %d 元素):\n\n", n, c, h, w, size);

    for (int i = 0; i < 5; i++) {
        CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
            activations[i].mode, CUDNN_NOT_PROPAGATE_NAN, activations[i].coef));

        // 预热
        CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc,
            &alpha, tensor_desc, d_input,
            &beta, tensor_desc, d_output));

        CHECK_CUDA(cudaEventRecord(start));
        for (int j = 0; j < 1000; j++) {
            CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc,
                &alpha, tensor_desc, d_input,
                &beta, tensor_desc, d_output));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // 获取一个输出样本
        float sample;
        CHECK_CUDA(cudaMemcpy(&sample, d_output, sizeof(float), cudaMemcpyDeviceToHost));

        printf("  %s:\n", activations[i].name);
        printf("    时间: %.4f ms, 带宽: %.1f GB/s\n",
               ms / 1000, 2.0 * size * sizeof(float) / (ms / 1000 * 1e6));
        printf("    样本: input=%.3f, output=%.3f\n\n", h_input[0], sample);
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// ============================================================================
// 第五部分：批归一化
// ============================================================================

void demoBatchNorm() {
    printf("=== 第五部分：批归一化 ===\n\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    const int n = 32, c = 64, h = 56, w = 56;
    const int size = n * c * h * w;

    printf("输入: %d × %d × %d × %d\n\n", n, c, h, w);

    // 创建描述符
    cudnnTensorDescriptor_t tensor_desc, bn_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bn_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensor_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // 派生 BN 参数描述符
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(bn_desc, tensor_desc,
        CUDNN_BATCHNORM_SPATIAL));

    // 分配内存
    float *d_input, *d_output;
    float *d_scale, *d_bias;
    float *d_running_mean, *d_running_var;
    float *d_saved_mean, *d_saved_var;

    CHECK_CUDA(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scale, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_running_mean, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_running_var, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_saved_mean, c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_saved_var, c * sizeof(float)));

    // 初始化
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_scale = (float*)malloc(c * sizeof(float));
    float *h_bias = (float*)malloc(c * sizeof(float));

    for (int i = 0; i < size; i++) h_input[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < c; i++) {
        h_scale[i] = 1.0f;
        h_bias[i] = 0.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale, c * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, c * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_running_mean, 0, c * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_running_var, 0, c * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;
    double epsilon = 1e-5;
    double exponentialAverageFactor = 0.1;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 训练模式
    printf("训练模式 (计算均值和方差):\n");
    {
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(cudnn,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            tensor_desc, d_input,
            tensor_desc, d_output,
            bn_desc, d_scale, d_bias,
            exponentialAverageFactor,
            d_running_mean, d_running_var,
            epsilon,
            d_saved_mean, d_saved_var));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(cudnn,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                tensor_desc, d_input,
                tensor_desc, d_output,
                bn_desc, d_scale, d_bias,
                exponentialAverageFactor,
                d_running_mean, d_running_var,
                epsilon,
                d_saved_mean, d_saved_var));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  时间: %.3f ms\n", ms / 100);
    }

    // 推理模式
    printf("\n推理模式 (使用运行均值):\n");
    {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            CHECK_CUDNN(cudnnBatchNormalizationForwardInference(cudnn,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                tensor_desc, d_input,
                tensor_desc, d_output,
                bn_desc, d_scale, d_bias,
                d_running_mean, d_running_var,
                epsilon));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  时间: %.3f ms\n\n", ms / 100);
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_running_mean));
    CHECK_CUDA(cudaFree(d_running_var));
    CHECK_CUDA(cudaFree(d_saved_mean));
    CHECK_CUDA(cudaFree(d_saved_var));
    free(h_input);
    free(h_scale);
    free(h_bias);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bn_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// ============================================================================
// 第六部分：Softmax
// ============================================================================

void demoSoftmax() {
    printf("=== 第六部分：Softmax ===\n\n");

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    const int n = 32, c = 1000;  // batch=32, classes=1000 (ImageNet)

    printf("输入: %d × %d (batch × classes)\n\n", n, c);

    // 创建描述符 (使用 NCHW，H=W=1)
    cudnnTensorDescriptor_t tensor_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensor_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, 1, 1));

    // 分配内存
    int size = n * c;
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, size * sizeof(float)));

    // 初始化
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f - 5.0f;  // [-5, 5]
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测试不同模式
    struct {
        cudnnSoftmaxMode_t mode;
        const char* name;
    } modes[] = {
        {CUDNN_SOFTMAX_MODE_INSTANCE, "Instance (整个样本)"},
        {CUDNN_SOFTMAX_MODE_CHANNEL, "Channel (按通道)"}
    };

    for (int m = 0; m < 2; m++) {
        CHECK_CUDNN(cudnnSoftmaxForward(cudnn,
            CUDNN_SOFTMAX_ACCURATE, modes[m].mode,
            &alpha, tensor_desc, d_input,
            &beta, tensor_desc, d_output));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 1000; i++) {
            CHECK_CUDNN(cudnnSoftmaxForward(cudnn,
                CUDNN_SOFTMAX_ACCURATE, modes[m].mode,
                &alpha, tensor_desc, d_input,
                &beta, tensor_desc, d_output));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // 验证和为1
        CHECK_CUDA(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
        float sum = 0;
        for (int i = 0; i < c; i++) sum += h_output[i];

        printf("%s:\n", modes[m].name);
        printf("  时间: %.4f ms\n", ms / 1000);
        printf("  第一个样本概率和: %.6f (应为 1.0)\n\n", sum);
    }

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 16: cuDNN 深度学习加速                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    demoCuDNNBasics();
    demoConvolution();
    demoPooling();
    demoActivation();
    demoBatchNorm();
    demoSoftmax();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. cuDNN 核心组件:\n");
    printf("   - Handle: cudnnCreate/cudnnDestroy\n");
    printf("   - Tensor Descriptor: 描述数据形状\n");
    printf("   - Operation Descriptor: 描述操作参数\n\n");

    printf("2. 主要操作:\n");
    printf("   ┌────────────────┬────────────────────────────────┐\n");
    printf("   │ 操作           │ 函数                           │\n");
    printf("   ├────────────────┼────────────────────────────────┤\n");
    printf("   │ 卷积           │ cudnnConvolutionForward        │\n");
    printf("   │ 池化           │ cudnnPoolingForward            │\n");
    printf("   │ 激活           │ cudnnActivationForward         │\n");
    printf("   │ 批归一化       │ cudnnBatchNormalizationForward │\n");
    printf("   │ Softmax        │ cudnnSoftmaxForward            │\n");
    printf("   └────────────────┴────────────────────────────────┘\n\n");

    printf("3. 卷积算法选择:\n");
    printf("   - IMPLICIT_GEMM: 通用，内存效率高\n");
    printf("   - IMPLICIT_PRECOMP_GEMM: 预计算索引\n");
    printf("   - GEMM: 显式矩阵乘法\n");
    printf("   - FFT: 大卷积核效率高\n");
    printf("   - WINOGRAD: 小卷积核 (3×3) 最优\n\n");

    printf("4. 性能优化:\n");
    printf("   - 使用 cudnnGetConvolutionForwardAlgorithm_v7 自动选择\n");
    printf("   - 预分配工作空间\n");
    printf("   - 使用 Tensor Core (FP16/TF32)\n");
    printf("   - 批量处理提高吞吐\n\n");

    printf("5. 数据格式建议:\n");
    printf("   - NCHW: 默认格式，广泛兼容\n");
    printf("   - NHWC: 某些操作更高效\n");
    printf("   - NC/32HW32: Tensor Core 最优\n\n");

    printf("编译命令:\n");
    printf("  nvcc -lcudnn 16_cudnn_deeplearning.cu -o 16_cudnn_deeplearning\n\n");

    return 0;
}
