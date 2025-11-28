/**
 * =============================================================================
 * CUDA 教程 31: GPU 加速神经网络推理引擎
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解神经网络推理的 GPU 实现原理
 * 2. 实现常见神经网络层的 CUDA 内核
 * 3. 学习推理优化技术
 * 4. 构建简单的推理引擎框架
 *
 * 实现内容：
 * - 全连接层 (Fully Connected)
 * - 卷积层 (Convolution)
 * - 池化层 (Pooling)
 * - 激活函数 (ReLU, Sigmoid, Softmax)
 * - BatchNorm 层
 * - 简单 CNN 推理流程
 *
 * 编译命令：
 *   nvcc 31_neural_network_inference.cu -o 31_neural_network -O3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：基础张量类
// ============================================================================

class Tensor {
public:
    float *h_data;      // 主机数据
    float *d_data;      // 设备数据
    int dims[4];        // NCHW 格式: batch, channel, height, width
    int ndim;
    size_t size;

    Tensor(int n, int c = 1, int h = 1, int w = 1) {
        dims[0] = n; dims[1] = c; dims[2] = h; dims[3] = w;
        ndim = 4;
        size = n * c * h * w * sizeof(float);
        h_data = (float*)malloc(size);
        CHECK_CUDA(cudaMalloc(&d_data, size));
        memset(h_data, 0, size);
    }

    ~Tensor() {
        if (h_data) free(h_data);
        if (d_data) cudaFree(d_data);
    }

    int elements() const { return dims[0] * dims[1] * dims[2] * dims[3]; }
    int N() const { return dims[0]; }
    int C() const { return dims[1]; }
    int H() const { return dims[2]; }
    int W() const { return dims[3]; }

    void toDevice() {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    }

    void toHost() {
        CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }

    void fillRandom(float scale = 1.0f) {
        for (int i = 0; i < elements(); i++) {
            h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        toDevice();
    }

    void fillConstant(float val) {
        for (int i = 0; i < elements(); i++) {
            h_data[i] = val;
        }
        toDevice();
    }

    // He 初始化
    void heInit(int fanIn) {
        float std = sqrtf(2.0f / fanIn);
        for (int i = 0; i < elements(); i++) {
            // Box-Muller 变换生成正态分布
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            h_data[i] = sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(2.0f * M_PI * u2) * std;
        }
        toDevice();
    }
};

// ============================================================================
// 第二部分：激活函数
// ============================================================================

// ReLU 激活
__global__ void reluKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = fmaxf(0.0f, input[tid]);
    }
}

// Leaky ReLU
__global__ void leakyReluKernel(float *output, const float *input, int n, float alpha) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = input[tid];
        output[tid] = val > 0 ? val : alpha * val;
    }
}

// Sigmoid 激活
__global__ void sigmoidKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = 1.0f / (1.0f + expf(-input[tid]));
    }
}

// Tanh 激活
__global__ void tanhKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = tanhf(input[tid]);
    }
}

// GELU 激活 (近似版)
__global__ void geluKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[tid] = x * cdf;
    }
}

// SiLU/Swish 激活
__global__ void siluKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        output[tid] = x / (1.0f + expf(-x));
    }
}

// Softmax (单个向量)
__global__ void softmaxKernel(float *output, const float *input, int batchSize, int numClasses) {
    extern __shared__ float smem[];

    int batch = blockIdx.x;
    int tid = threadIdx.x;

    const float *in = input + batch * numClasses;
    float *out = output + batch * numClasses;

    // 找最大值 (数值稳定性)
    float maxVal = -INFINITY;
    for (int i = tid; i < numClasses; i += blockDim.x) {
        maxVal = fmaxf(maxVal, in[i]);
    }
    smem[tid] = maxVal;
    __syncthreads();

    // 规约找最大
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    maxVal = smem[0];
    __syncthreads();

    // 计算 exp(x - max) 和 sum
    float sum = 0.0f;
    for (int i = tid; i < numClasses; i += blockDim.x) {
        float val = expf(in[i] - maxVal);
        out[i] = val;
        sum += val;
    }
    smem[tid] = sum;
    __syncthreads();

    // 规约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    sum = smem[0];

    // 归一化
    for (int i = tid; i < numClasses; i += blockDim.x) {
        out[i] /= sum;
    }
}

void demoActivations() {
    printf("=== 第一部分：激活函数 ===\n\n");

    printf("支持的激活函数:\n");
    printf("  ┌───────────────┬─────────────────────────────────────────┐\n");
    printf("  │ 激活函数      │ 公式                                    │\n");
    printf("  ├───────────────┼─────────────────────────────────────────┤\n");
    printf("  │ ReLU          │ max(0, x)                               │\n");
    printf("  │ Leaky ReLU    │ x > 0 ? x : αx                          │\n");
    printf("  │ Sigmoid       │ 1 / (1 + e^(-x))                        │\n");
    printf("  │ Tanh          │ (e^x - e^(-x)) / (e^x + e^(-x))         │\n");
    printf("  │ GELU          │ x * Φ(x)                                │\n");
    printf("  │ SiLU/Swish    │ x * sigmoid(x)                          │\n");
    printf("  │ Softmax       │ e^xi / Σe^xj                            │\n");
    printf("  └───────────────┴─────────────────────────────────────────┘\n\n");

    // 测试激活函数
    const int N = 10;
    Tensor input(N);
    Tensor output(N);

    for (int i = 0; i < N; i++) {
        input.h_data[i] = (float)(i - 5);  // -5 到 4
    }
    input.toDevice();

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    // ReLU
    reluKernel<<<grid, block>>>(output.d_data, input.d_data, N);
    output.toHost();

    printf("ReLU 测试:\n");
    printf("  输入:  ");
    for (int i = 0; i < N; i++) printf("%6.2f ", input.h_data[i]);
    printf("\n  输出:  ");
    for (int i = 0; i < N; i++) printf("%6.2f ", output.h_data[i]);
    printf("\n\n");

    // GELU
    geluKernel<<<grid, block>>>(output.d_data, input.d_data, N);
    output.toHost();

    printf("GELU 测试:\n");
    printf("  输入:  ");
    for (int i = 0; i < N; i++) printf("%6.2f ", input.h_data[i]);
    printf("\n  输出:  ");
    for (int i = 0; i < N; i++) printf("%6.2f ", output.h_data[i]);
    printf("\n\n");

    // Softmax
    Tensor softIn(1, 1, 1, 5);
    Tensor softOut(1, 1, 1, 5);
    softIn.h_data[0] = 1.0f; softIn.h_data[1] = 2.0f; softIn.h_data[2] = 3.0f;
    softIn.h_data[3] = 4.0f; softIn.h_data[4] = 5.0f;
    softIn.toDevice();

    softmaxKernel<<<1, 32, 32 * sizeof(float)>>>(softOut.d_data, softIn.d_data, 1, 5);
    softOut.toHost();

    printf("Softmax 测试:\n");
    printf("  输入:  ");
    for (int i = 0; i < 5; i++) printf("%6.2f ", softIn.h_data[i]);
    printf("\n  输出:  ");
    float sum = 0;
    for (int i = 0; i < 5; i++) {
        printf("%6.4f ", softOut.h_data[i]);
        sum += softOut.h_data[i];
    }
    printf("\n  总和:  %.6f (应为 1.0)\n\n", sum);
}

// ============================================================================
// 第三部分：全连接层 (Dense/Linear)
// ============================================================================

// 基本矩阵乘法实现全连接层
// output = input * weights^T + bias
// input: [batch, inputSize], weights: [outputSize, inputSize], output: [batch, outputSize]
__global__ void denseKernelBasic(float *output, const float *input, const float *weights,
                                  const float *bias, int batch, int inputSize, int outputSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output index

    if (row < batch && col < outputSize) {
        float sum = bias ? bias[col] : 0.0f;
        for (int k = 0; k < inputSize; k++) {
            sum += input[row * inputSize + k] * weights[col * inputSize + k];
        }
        output[row * outputSize + col] = sum;
    }
}

// 使用共享内存的分块全连接层
#define TILE_DIM 16

__global__ void denseKernelTiled(float *output, const float *input, const float *weights,
                                  const float *bias, int batch, int inputSize, int outputSize) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (inputSize + TILE_DIM - 1) / TILE_DIM; t++) {
        // 加载 input tile
        int inputCol = t * TILE_DIM + threadIdx.x;
        if (row < batch && inputCol < inputSize) {
            tileA[threadIdx.y][threadIdx.x] = input[row * inputSize + inputCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 weights tile (转置)
        int weightsRow = t * TILE_DIM + threadIdx.y;
        if (col < outputSize && weightsRow < inputSize) {
            tileB[threadIdx.y][threadIdx.x] = weights[col * inputSize + weightsRow];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算
        for (int k = 0; k < TILE_DIM; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < batch && col < outputSize) {
        if (bias) sum += bias[col];
        output[row * outputSize + col] = sum;
    }
}

// Dense 层类
class DenseLayer {
public:
    int inputSize, outputSize;
    Tensor *weights;
    Tensor *bias;

    DenseLayer(int in, int out) : inputSize(in), outputSize(out) {
        weights = new Tensor(out, in);
        bias = new Tensor(out);
        weights->heInit(in);
        bias->fillConstant(0.0f);
    }

    ~DenseLayer() {
        delete weights;
        delete bias;
    }

    void forward(Tensor *output, const Tensor *input) {
        int batch = input->N();
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((outputSize + TILE_DIM - 1) / TILE_DIM,
                  (batch + TILE_DIM - 1) / TILE_DIM);

        denseKernelTiled<<<grid, block>>>(
            output->d_data, input->d_data, weights->d_data, bias->d_data,
            batch, inputSize, outputSize);
    }
};

void demoDenseLayer() {
    printf("=== 第二部分：全连接层 (Dense) ===\n\n");

    printf("全连接层计算:\n");
    printf("  Y = X × W^T + b\n");
    printf("  X: [batch, input_size]\n");
    printf("  W: [output_size, input_size]\n");
    printf("  b: [output_size]\n");
    printf("  Y: [batch, output_size]\n\n");

    const int batch = 32;
    const int inputSize = 784;   // MNIST 输入
    const int outputSize = 256;

    DenseLayer layer(inputSize, outputSize);
    Tensor input(batch, inputSize);
    Tensor output(batch, outputSize);

    input.fillRandom(0.1f);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    layer.forward(&output, &input);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int NUM_RUNS = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        layer.forward(&output, &input);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;

    // 计算 GFLOPS
    long long flops = 2LL * batch * inputSize * outputSize;  // 乘加运算
    float gflops = flops / elapsed / 1e6;

    printf("Dense 层性能 (%dx%d → %d):\n", batch, inputSize, outputSize);
    printf("  执行时间: %.3f ms\n", elapsed);
    printf("  吞吐量: %.2f GFLOPS\n\n", gflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第四部分：卷积层
// ============================================================================

// 基本 2D 卷积 (NCHW 格式)
__global__ void conv2dKernel(float *output, const float *input, const float *kernel,
                              const float *bias,
                              int batch, int inChannels, int outChannels,
                              int inH, int inW, int outH, int outW,
                              int kH, int kW, int strideH, int strideW,
                              int padH, int padW) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outC = blockIdx.z % outChannels;
    int n = blockIdx.z / outChannels;

    if (outX < outW && outY < outH && n < batch) {
        float sum = bias ? bias[outC] : 0.0f;

        for (int ic = 0; ic < inChannels; ic++) {
            for (int ky = 0; ky < kH; ky++) {
                for (int kx = 0; kx < kW; kx++) {
                    int inY = outY * strideH - padH + ky;
                    int inX = outX * strideW - padW + kx;

                    if (inY >= 0 && inY < inH && inX >= 0 && inX < inW) {
                        int inputIdx = ((n * inChannels + ic) * inH + inY) * inW + inX;
                        int kernelIdx = ((outC * inChannels + ic) * kH + ky) * kW + kx;
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }

        int outputIdx = ((n * outChannels + outC) * outH + outY) * outW + outX;
        output[outputIdx] = sum;
    }
}

// 使用共享内存的优化卷积 (针对 3x3 卷积)
#define CONV_TILE_SIZE 16
#define CONV_KERNEL_SIZE 3

__global__ void conv2d3x3Shared(float *output, const float *input, const float *kernel,
                                 const float *bias,
                                 int batch, int inChannels, int outChannels,
                                 int inH, int inW, int outH, int outW) {
    __shared__ float smem[CONV_TILE_SIZE + 2][CONV_TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outX = blockIdx.x * CONV_TILE_SIZE + tx;
    int outY = blockIdx.y * CONV_TILE_SIZE + ty;
    int outC = blockIdx.z % outChannels;
    int n = blockIdx.z / outChannels;

    float sum = bias ? bias[outC] : 0.0f;

    for (int ic = 0; ic < inChannels; ic++) {
        // 加载输入 tile 到共享内存
        int inX = outX - 1;  // padding = 1
        int inY = outY - 1;

        if (inX >= 0 && inX < inW && inY >= 0 && inY < inH) {
            smem[ty][tx] = input[((n * inChannels + ic) * inH + inY) * inW + inX];
        } else {
            smem[ty][tx] = 0.0f;
        }

        // 加载边界
        if (tx < 2) {
            int borderX = outX + CONV_TILE_SIZE - 1;
            if (borderX >= 0 && borderX < inW && inY >= 0 && inY < inH) {
                smem[ty][tx + CONV_TILE_SIZE] = input[((n * inChannels + ic) * inH + inY) * inW + borderX];
            } else {
                smem[ty][tx + CONV_TILE_SIZE] = 0.0f;
            }
        }
        if (ty < 2) {
            int borderY = outY + CONV_TILE_SIZE - 1;
            if (inX >= 0 && inX < inW && borderY >= 0 && borderY < inH) {
                smem[ty + CONV_TILE_SIZE][tx] = input[((n * inChannels + ic) * inH + borderY) * inW + inX];
            } else {
                smem[ty + CONV_TILE_SIZE][tx] = 0.0f;
            }
        }

        __syncthreads();

        // 卷积计算
        if (outX < outW && outY < outH) {
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int kernelIdx = ((outC * inChannels + ic) * 3 + ky) * 3 + kx;
                    sum += smem[ty + ky][tx + kx] * kernel[kernelIdx];
                }
            }
        }

        __syncthreads();
    }

    if (outX < outW && outY < outH && n < batch) {
        output[((n * outChannels + outC) * outH + outY) * outW + outX] = sum;
    }
}

// Conv2D 层类
class Conv2DLayer {
public:
    int inChannels, outChannels;
    int kernelSize, stride, padding;
    Tensor *kernel;
    Tensor *bias;

    Conv2DLayer(int inC, int outC, int kSize, int s = 1, int pad = 0)
        : inChannels(inC), outChannels(outC), kernelSize(kSize), stride(s), padding(pad) {
        kernel = new Tensor(outC, inC, kSize, kSize);
        bias = new Tensor(outC);
        kernel->heInit(inC * kSize * kSize);
        bias->fillConstant(0.0f);
    }

    ~Conv2DLayer() {
        delete kernel;
        delete bias;
    }

    void forward(Tensor *output, const Tensor *input) {
        int batch = input->N();
        int outH = (input->H() + 2 * padding - kernelSize) / stride + 1;
        int outW = (input->W() + 2 * padding - kernelSize) / stride + 1;

        dim3 block(16, 16);
        dim3 grid((outW + 15) / 16, (outH + 15) / 16, batch * outChannels);

        conv2dKernel<<<grid, block>>>(
            output->d_data, input->d_data, kernel->d_data, bias->d_data,
            batch, inChannels, outChannels,
            input->H(), input->W(), outH, outW,
            kernelSize, kernelSize, stride, stride, padding, padding);
    }
};

void demoConv2D() {
    printf("=== 第三部分：卷积层 (Conv2D) ===\n\n");

    printf("2D 卷积计算:\n");
    printf("  输入: [N, C_in, H, W]\n");
    printf("  卷积核: [C_out, C_in, kH, kW]\n");
    printf("  输出: [N, C_out, H_out, W_out]\n");
    printf("  H_out = (H + 2*pad - kH) / stride + 1\n\n");

    const int batch = 8;
    const int inChannels = 64;
    const int outChannels = 128;
    const int H = 56, W = 56;
    const int kernelSize = 3;
    const int padding = 1;

    Conv2DLayer layer(inChannels, outChannels, kernelSize, 1, padding);
    Tensor input(batch, inChannels, H, W);
    Tensor output(batch, outChannels, H, W);

    input.fillRandom(0.1f);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    layer.forward(&output, &input);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int NUM_RUNS = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        layer.forward(&output, &input);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;

    // 计算 GFLOPS
    long long flops = 2LL * batch * outChannels * H * W * inChannels * kernelSize * kernelSize;
    float gflops = flops / elapsed / 1e6;

    printf("Conv2D 层性能 (%dx%dx%dx%d, %dx%d kernel):\n",
           batch, inChannels, H, W, kernelSize, kernelSize);
    printf("  执行时间: %.3f ms\n", elapsed);
    printf("  吞吐量: %.2f GFLOPS\n\n", gflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第五部分：池化层
// ============================================================================

// 最大池化
__global__ void maxPool2dKernel(float *output, const float *input,
                                 int batch, int channels, int inH, int inW,
                                 int outH, int outW, int poolSize, int stride) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;

    if (outX < outW && outY < outH && n < batch) {
        float maxVal = -INFINITY;

        for (int py = 0; py < poolSize; py++) {
            for (int px = 0; px < poolSize; px++) {
                int inY = outY * stride + py;
                int inX = outX * stride + px;

                if (inY < inH && inX < inW) {
                    int idx = ((n * channels + c) * inH + inY) * inW + inX;
                    maxVal = fmaxf(maxVal, input[idx]);
                }
            }
        }

        int outIdx = ((n * channels + c) * outH + outY) * outW + outX;
        output[outIdx] = maxVal;
    }
}

// 平均池化
__global__ void avgPool2dKernel(float *output, const float *input,
                                 int batch, int channels, int inH, int inW,
                                 int outH, int outW, int poolSize, int stride) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;

    if (outX < outW && outY < outH && n < batch) {
        float sum = 0.0f;
        int count = 0;

        for (int py = 0; py < poolSize; py++) {
            for (int px = 0; px < poolSize; px++) {
                int inY = outY * stride + py;
                int inX = outX * stride + px;

                if (inY < inH && inX < inW) {
                    int idx = ((n * channels + c) * inH + inY) * inW + inX;
                    sum += input[idx];
                    count++;
                }
            }
        }

        int outIdx = ((n * channels + c) * outH + outY) * outW + outX;
        output[outIdx] = sum / count;
    }
}

// 全局平均池化
__global__ void globalAvgPoolKernel(float *output, const float *input,
                                     int batch, int channels, int H, int W) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;

    if (c < channels && n < batch) {
        float sum = 0.0f;
        const float *channelData = input + (n * channels + c) * H * W;

        for (int i = 0; i < H * W; i++) {
            sum += channelData[i];
        }

        output[n * channels + c] = sum / (H * W);
    }
}

void demoPooling() {
    printf("=== 第四部分：池化层 ===\n\n");

    printf("池化类型:\n");
    printf("  ┌───────────────┬─────────────────────────────────────────┐\n");
    printf("  │ 类型          │ 说明                                    │\n");
    printf("  ├───────────────┼─────────────────────────────────────────┤\n");
    printf("  │ MaxPool       │ 取窗口内最大值                          │\n");
    printf("  │ AvgPool       │ 取窗口内平均值                          │\n");
    printf("  │ GlobalAvgPool │ 整个特征图取平均                        │\n");
    printf("  └───────────────┴─────────────────────────────────────────┘\n\n");

    const int batch = 8;
    const int channels = 64;
    const int H = 56, W = 56;
    const int poolSize = 2;
    const int stride = 2;
    const int outH = H / stride;
    const int outW = W / stride;

    Tensor input(batch, channels, H, W);
    Tensor output(batch, channels, outH, outW);
    input.fillRandom(1.0f);

    dim3 block(16, 16);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16, batch * channels);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // MaxPool
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        maxPool2dKernel<<<grid, block>>>(output.d_data, input.d_data,
                                          batch, channels, H, W, outH, outW,
                                          poolSize, stride);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float maxPoolTime;
    CHECK_CUDA(cudaEventElapsedTime(&maxPoolTime, start, stop));
    maxPoolTime /= NUM_RUNS;

    // AvgPool
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        avgPool2dKernel<<<grid, block>>>(output.d_data, input.d_data,
                                          batch, channels, H, W, outH, outW,
                                          poolSize, stride);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float avgPoolTime;
    CHECK_CUDA(cudaEventElapsedTime(&avgPoolTime, start, stop));
    avgPoolTime /= NUM_RUNS;

    printf("池化层性能 (%dx%dx%dx%d, %dx%d pool):\n",
           batch, channels, H, W, poolSize, poolSize);
    printf("  MaxPool: %.3f ms\n", maxPoolTime);
    printf("  AvgPool: %.3f ms\n\n", avgPoolTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第六部分：BatchNorm 层
// ============================================================================

// BatchNorm 推理 (已固定参数)
__global__ void batchNormInferenceKernel(float *output, const float *input,
                                          const float *gamma, const float *beta,
                                          const float *runningMean, const float *runningVar,
                                          int batch, int channels, int H, int W, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;

    if (x < W && y < H && n < batch) {
        int idx = ((n * channels + c) * H + y) * W + x;

        float mean = runningMean[c];
        float var = runningVar[c];
        float g = gamma[c];
        float b = beta[c];

        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        output[idx] = g * normalized + b;
    }
}

// 融合 BatchNorm + ReLU
__global__ void batchNormReluKernel(float *output, const float *input,
                                     const float *gamma, const float *beta,
                                     const float *runningMean, const float *runningVar,
                                     int batch, int channels, int H, int W, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;

    if (x < W && y < H && n < batch) {
        int idx = ((n * channels + c) * H + y) * W + x;

        float mean = runningMean[c];
        float var = runningVar[c];
        float g = gamma[c];
        float b = beta[c];

        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        float val = g * normalized + b;

        // 融合 ReLU
        output[idx] = fmaxf(0.0f, val);
    }
}

void demoBatchNorm() {
    printf("=== 第五部分：BatchNorm 层 ===\n\n");

    printf("BatchNorm 推理公式:\n");
    printf("  y = γ * (x - μ) / √(σ² + ε) + β\n");
    printf("  μ, σ²: 运行时统计的均值和方差\n");
    printf("  γ, β: 学习的缩放和偏移参数\n\n");

    printf("优化策略:\n");
    printf("  1. 融合操作: BatchNorm + ReLU\n");
    printf("  2. 参数折叠: 将 BN 合并到前一层\n");
    printf("  3. 使用 half 精度\n\n");

    const int batch = 8;
    const int channels = 64;
    const int H = 56, W = 56;

    Tensor input(batch, channels, H, W);
    Tensor output(batch, channels, H, W);
    Tensor gamma(channels), beta(channels);
    Tensor runningMean(channels), runningVar(channels);

    input.fillRandom(1.0f);
    gamma.fillConstant(1.0f);
    beta.fillConstant(0.0f);
    runningMean.fillConstant(0.0f);
    runningVar.fillConstant(1.0f);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16, batch * channels);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 普通 BatchNorm
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        batchNormInferenceKernel<<<grid, block>>>(
            output.d_data, input.d_data, gamma.d_data, beta.d_data,
            runningMean.d_data, runningVar.d_data,
            batch, channels, H, W, 1e-5f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float bnTime;
    CHECK_CUDA(cudaEventElapsedTime(&bnTime, start, stop));
    bnTime /= NUM_RUNS;

    // 融合 BatchNorm + ReLU
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        batchNormReluKernel<<<grid, block>>>(
            output.d_data, input.d_data, gamma.d_data, beta.d_data,
            runningMean.d_data, runningVar.d_data,
            batch, channels, H, W, 1e-5f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float fusedTime;
    CHECK_CUDA(cudaEventElapsedTime(&fusedTime, start, stop));
    fusedTime /= NUM_RUNS;

    printf("BatchNorm 性能 (%dx%dx%dx%d):\n", batch, channels, H, W);
    printf("  BatchNorm: %.3f ms\n", bnTime);
    printf("  BatchNorm + ReLU 融合: %.3f ms\n", fusedTime);
    printf("  融合加速: %.2fx\n\n", bnTime / fusedTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第七部分：简单 CNN 推理示例
// ============================================================================

class SimpleCNN {
public:
    // 网络结构: Conv -> BN -> ReLU -> Pool -> Conv -> BN -> ReLU -> Pool -> FC -> Softmax
    Conv2DLayer *conv1, *conv2;
    DenseLayer *fc;

    Tensor *conv1_out, *conv2_out, *pool1_out, *pool2_out, *flat_out, *fc_out;
    Tensor *bn1_gamma, *bn1_beta, *bn1_mean, *bn1_var;
    Tensor *bn2_gamma, *bn2_beta, *bn2_mean, *bn2_var;

    int inputC, inputH, inputW;
    int numClasses;

    SimpleCNN(int inC, int inH, int inW, int classes)
        : inputC(inC), inputH(inH), inputW(inW), numClasses(classes) {

        // Layer 1: Conv 3x3, 32 filters
        conv1 = new Conv2DLayer(inC, 32, 3, 1, 1);
        bn1_gamma = new Tensor(32); bn1_gamma->fillConstant(1.0f);
        bn1_beta = new Tensor(32); bn1_beta->fillConstant(0.0f);
        bn1_mean = new Tensor(32); bn1_mean->fillConstant(0.0f);
        bn1_var = new Tensor(32); bn1_var->fillConstant(1.0f);

        // Layer 2: Conv 3x3, 64 filters
        conv2 = new Conv2DLayer(32, 64, 3, 1, 1);
        bn2_gamma = new Tensor(64); bn2_gamma->fillConstant(1.0f);
        bn2_beta = new Tensor(64); bn2_beta->fillConstant(0.0f);
        bn2_mean = new Tensor(64); bn2_mean->fillConstant(0.0f);
        bn2_var = new Tensor(64); bn2_var->fillConstant(1.0f);

        // FC layer
        int fcInputSize = 64 * (inH / 4) * (inW / 4);  // After 2 pooling layers
        fc = new DenseLayer(fcInputSize, classes);

        // 中间张量 (假设 batch=1)
        conv1_out = new Tensor(1, 32, inH, inW);
        pool1_out = new Tensor(1, 32, inH/2, inW/2);
        conv2_out = new Tensor(1, 64, inH/2, inW/2);
        pool2_out = new Tensor(1, 64, inH/4, inW/4);
        flat_out = new Tensor(1, fcInputSize);
        fc_out = new Tensor(1, classes);
    }

    ~SimpleCNN() {
        delete conv1; delete conv2; delete fc;
        delete conv1_out; delete conv2_out;
        delete pool1_out; delete pool2_out;
        delete flat_out; delete fc_out;
        delete bn1_gamma; delete bn1_beta; delete bn1_mean; delete bn1_var;
        delete bn2_gamma; delete bn2_beta; delete bn2_mean; delete bn2_var;
    }

    void forward(Tensor *output, const Tensor *input) {
        int batch = 1;

        // Layer 1: Conv -> BN -> ReLU -> MaxPool
        conv1->forward(conv1_out, input);

        dim3 block1(16, 16);
        dim3 grid1((inputW + 15) / 16, (inputH + 15) / 16, batch * 32);
        batchNormReluKernel<<<grid1, block1>>>(
            conv1_out->d_data, conv1_out->d_data,
            bn1_gamma->d_data, bn1_beta->d_data,
            bn1_mean->d_data, bn1_var->d_data,
            batch, 32, inputH, inputW, 1e-5f);

        int pool1H = inputH / 2, pool1W = inputW / 2;
        dim3 poolGrid1((pool1W + 15) / 16, (pool1H + 15) / 16, batch * 32);
        maxPool2dKernel<<<poolGrid1, block1>>>(
            pool1_out->d_data, conv1_out->d_data,
            batch, 32, inputH, inputW, pool1H, pool1W, 2, 2);

        // Layer 2: Conv -> BN -> ReLU -> MaxPool
        conv2->forward(conv2_out, pool1_out);

        dim3 grid2((pool1W + 15) / 16, (pool1H + 15) / 16, batch * 64);
        batchNormReluKernel<<<grid2, block1>>>(
            conv2_out->d_data, conv2_out->d_data,
            bn2_gamma->d_data, bn2_beta->d_data,
            bn2_mean->d_data, bn2_var->d_data,
            batch, 64, pool1H, pool1W, 1e-5f);

        int pool2H = pool1H / 2, pool2W = pool1W / 2;
        dim3 poolGrid2((pool2W + 15) / 16, (pool2H + 15) / 16, batch * 64);
        maxPool2dKernel<<<poolGrid2, block1>>>(
            pool2_out->d_data, conv2_out->d_data,
            batch, 64, pool1H, pool1W, pool2H, pool2W, 2, 2);

        // Flatten (数据已经在 GPU 上是连续的)

        // FC layer
        fc->forward(fc_out, pool2_out);

        // Softmax
        softmaxKernel<<<1, 128, 128 * sizeof(float)>>>(
            output->d_data, fc_out->d_data, batch, numClasses);
    }
};

void demoSimpleCNN() {
    printf("=== 第六部分：简单 CNN 推理 ===\n\n");

    printf("网络结构 (类 LeNet):\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  Input: [1, 1, 28, 28]                                  │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  Conv2D(1, 32, 3x3, pad=1) → BN → ReLU                  │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  MaxPool(2x2)                                           │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  Conv2D(32, 64, 3x3, pad=1) → BN → ReLU                 │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  MaxPool(2x2)                                           │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  Flatten                                                │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  Dense(3136, 10)                                        │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  Softmax                                                │\n");
    printf("  │      ↓                                                  │\n");
    printf("  │  Output: [1, 10]                                        │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");

    SimpleCNN model(1, 28, 28, 10);

    Tensor input(1, 1, 28, 28);
    Tensor output(1, 10);

    // 模拟输入图像
    input.fillRandom(0.5f);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    model.forward(&output, &input);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int NUM_RUNS = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        model.forward(&output, &input);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;

    output.toHost();

    printf("推理结果:\n");
    printf("  类别概率: ");
    int maxIdx = 0;
    float maxProb = output.h_data[0];
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", output.h_data[i]);
        if (output.h_data[i] > maxProb) {
            maxProb = output.h_data[i];
            maxIdx = i;
        }
    }
    printf("\n  预测类别: %d (概率: %.3f)\n\n", maxIdx, maxProb);

    printf("推理性能:\n");
    printf("  单次推理: %.3f ms\n", elapsed);
    printf("  吞吐量: %.2f 图像/秒\n\n", 1000.0f / elapsed);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第八部分：推理优化技术
// ============================================================================

void demoOptimizationTechniques() {
    printf("=== 第七部分：推理优化技术 ===\n\n");

    printf("1. 算子融合 (Operator Fusion):\n");
    printf("   ┌───────────────────────────────────────────────────────┐\n");
    printf("   │ 融合前: Conv → BN → ReLU (3 次内存访问)              │\n");
    printf("   │ 融合后: Conv_BN_ReLU (1 次内存访问)                   │\n");
    printf("   │ 收益: 减少内存带宽开销                                │\n");
    printf("   └───────────────────────────────────────────────────────┘\n\n");

    printf("2. 量化 (Quantization):\n");
    printf("   ┌───────────────────────────────────────────────────────┐\n");
    printf("   │ FP32 → INT8: 4x 内存减少, 显著加速                    │\n");
    printf("   │ FP32 → FP16: 2x 内存减少, Tensor Core 加速            │\n");
    printf("   │ 精度损失: 通常 < 1%% (需要校准)                        │\n");
    printf("   └───────────────────────────────────────────────────────┘\n\n");

    printf("3. 内存优化:\n");
    printf("   - 内存池: 避免频繁分配/释放\n");
    printf("   - 内存复用: 不同层共享中间缓冲区\n");
    printf("   - Pinned Memory: 加速 Host-Device 传输\n\n");

    printf("4. 并行策略:\n");
    printf("   - 数据并行: 批处理多个输入\n");
    printf("   - 流并行: 多 Stream 隐藏延迟\n");
    printf("   - 模型并行: 大模型跨 GPU 拆分\n\n");

    printf("5. 专用库:\n");
    printf("   ┌───────────────────┬───────────────────────────────────┐\n");
    printf("   │ 库                │ 用途                              │\n");
    printf("   ├───────────────────┼───────────────────────────────────┤\n");
    printf("   │ cuDNN             │ 深度学习原语 (Conv, BN, etc.)     │\n");
    printf("   │ cuBLAS            │ 矩阵运算 (GEMM)                   │\n");
    printf("   │ TensorRT          │ 推理优化引擎                      │\n");
    printf("   │ CUTLASS           │ 高性能 GEMM 模板                  │\n");
    printf("   └───────────────────┴───────────────────────────────────┘\n\n");

    printf("6. Tensor Core 使用:\n");
    printf("   - 混合精度: FP16 输入, FP32 累加\n");
    printf("   - 对齐要求: 维度为 8 或 16 的倍数\n");
    printf("   - 性能提升: 最高 8x (vs FP32 CUDA Core)\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 31: GPU 加速神经网络推理引擎                         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("SM 数量: %d\n\n", prop.multiProcessorCount);

    demoActivations();
    demoDenseLayer();
    demoConv2D();
    demoPooling();
    demoBatchNorm();
    demoSimpleCNN();
    demoOptimizationTechniques();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("神经网络推理关键层实现:\n");
    printf("  ✓ 激活函数: ReLU, GELU, Softmax 等\n");
    printf("  ✓ 全连接层: 分块矩阵乘法\n");
    printf("  ✓ 卷积层: 直接卷积 / 共享内存优化\n");
    printf("  ✓ 池化层: MaxPool / AvgPool\n");
    printf("  ✓ BatchNorm: 推理模式 (固定统计量)\n\n");

    printf("优化技术:\n");
    printf("  - 算子融合减少内存访问\n");
    printf("  - 量化降低计算/内存开销\n");
    printf("  - 内存池和复用\n");
    printf("  - 使用专用库 (cuDNN, TensorRT)\n");
    printf("  - Tensor Core 混合精度\n\n");

    printf("下一步:\n");
    printf("  1. 集成 cuDNN 获得更好性能\n");
    printf("  2. 添加 INT8/FP16 量化支持\n");
    printf("  3. 实现更多层类型 (Attention, RNN)\n");
    printf("  4. 使用 TensorRT 部署优化\n\n");

    return 0;
}
