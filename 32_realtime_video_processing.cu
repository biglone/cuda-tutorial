/**
 * =============================================================================
 * CUDA 教程 32: 实时视频/音频处理
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解实时处理的延迟要求和设计原则
 * 2. 实现 GPU 加速的视频处理流水线
 * 3. 学习音频信号处理的 GPU 实现
 * 4. 掌握流式处理和双缓冲技术
 *
 * 实现内容：
 * - 视频帧处理 (色彩校正、滤波)
 * - 视频编解码加速概念
 * - 音频 FFT 和频谱分析
 * - 音频效果处理
 * - 实时流水线设计
 *
 * 编译命令：
 *   nvcc 32_realtime_video_processing.cu -o 32_realtime -O3 -lcufft
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        printf("cuFFT 错误 %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：实时处理基础概念
// ============================================================================

void demoRealtimeConcepts() {
    printf("=== 第一部分：实时处理基础概念 ===\n\n");

    printf("实时处理延迟要求:\n");
    printf("  ┌───────────────────┬────────────────┬───────────────────┐\n");
    printf("  │ 应用场景          │ 延迟要求       │ 帧率              │\n");
    printf("  ├───────────────────┼────────────────┼───────────────────┤\n");
    printf("  │ 视频直播          │ < 100 ms       │ 30-60 fps         │\n");
    printf("  │ 视频会议          │ < 150 ms       │ 24-30 fps         │\n");
    printf("  │ 游戏              │ < 16 ms        │ 60+ fps           │\n");
    printf("  │ VR/AR             │ < 10 ms        │ 90+ fps           │\n");
    printf("  │ 音频处理          │ < 10 ms        │ -                 │\n");
    printf("  │ 自动驾驶          │ < 50 ms        │ 30+ fps           │\n");
    printf("  └───────────────────┴────────────────┴───────────────────┘\n\n");

    printf("GPU 实时处理优势:\n");
    printf("  1. 大规模并行处理像素/样本\n");
    printf("  2. 高内存带宽 (数百 GB/s)\n");
    printf("  3. 专用硬件单元 (NVENC/NVDEC)\n");
    printf("  4. 异步执行隐藏延迟\n\n");

    printf("关键设计原则:\n");
    printf("  ┌────────────────────────────────────────────────────────┐\n");
    printf("  │ 1. 双缓冲/三缓冲: 处理当前帧时准备下一帧              │\n");
    printf("  │ 2. 流水线: 传输、处理、输出重叠                       │\n");
    printf("  │ 3. 异步操作: cudaMemcpyAsync + Streams               │\n");
    printf("  │ 4. 内存池: 预分配避免运行时分配                       │\n");
    printf("  │ 5. 固定内存: Pinned memory 加速传输                   │\n");
    printf("  └────────────────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第二部分：视频帧处理内核
// ============================================================================

// YUV 到 RGB 转换 (NV12 格式)
__global__ void nv12ToRgbKernel(unsigned char *rgb, const unsigned char *y,
                                 const unsigned char *uv, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && iy < height) {
        int yIdx = iy * width + x;
        int uvIdx = (iy / 2) * width + (x / 2) * 2;

        float Y = y[yIdx];
        float U = uv[uvIdx] - 128.0f;
        float V = uv[uvIdx + 1] - 128.0f;

        // ITU-R BT.601 转换
        float R = Y + 1.402f * V;
        float G = Y - 0.344f * U - 0.714f * V;
        float B = Y + 1.772f * U;

        int rgbIdx = yIdx * 3;
        rgb[rgbIdx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, R));
        rgb[rgbIdx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, G));
        rgb[rgbIdx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, B));
    }
}

// RGB 到 YUV 转换
__global__ void rgbToYuvKernel(unsigned char *y, unsigned char *u, unsigned char *v,
                                const unsigned char *rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && iy < height) {
        int idx = iy * width + x;
        int rgbIdx = idx * 3;

        float R = rgb[rgbIdx];
        float G = rgb[rgbIdx + 1];
        float B = rgb[rgbIdx + 2];

        // ITU-R BT.601
        float Y = 0.299f * R + 0.587f * G + 0.114f * B;
        float U = -0.169f * R - 0.331f * G + 0.500f * B + 128.0f;
        float V = 0.500f * R - 0.419f * G - 0.081f * B + 128.0f;

        y[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, Y));

        // UV 下采样 (4:2:0)
        if ((x % 2 == 0) && (iy % 2 == 0)) {
            int uvIdx = (iy / 2) * (width / 2) + (x / 2);
            u[uvIdx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, U));
            v[uvIdx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, V));
        }
    }
}

// 亮度/对比度调整
__global__ void brightnessContrastKernel(unsigned char *output, const unsigned char *input,
                                          int width, int height, int channels,
                                          float brightness, float contrast) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;
            float val = input[idx];

            // 对比度: (val - 128) * contrast + 128
            // 亮度: val + brightness
            val = (val - 128.0f) * contrast + 128.0f + brightness;
            output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    }
}

// 色彩饱和度调整
__global__ void saturationKernel(unsigned char *output, const unsigned char *input,
                                  int width, int height, float saturation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        float r = input[idx];
        float g = input[idx + 1];
        float b = input[idx + 2];

        // 计算亮度
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;

        // 调整饱和度
        r = gray + saturation * (r - gray);
        g = gray + saturation * (g - gray);
        b = gray + saturation * (b - gray);

        output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r));
        output[idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, g));
        output[idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, b));
    }
}

// 色温调整 (简化版)
__global__ void colorTemperatureKernel(unsigned char *output, const unsigned char *input,
                                        int width, int height, float temperature) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        float r = input[idx];
        float g = input[idx + 1];
        float b = input[idx + 2];

        // 色温调整: 暖色增加红色减少蓝色，冷色反之
        r = r + temperature * 20.0f;
        b = b - temperature * 20.0f;

        output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, r));
        output[idx + 1] = (unsigned char)g;
        output[idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, b));
    }
}

// LUT (Look-Up Table) 应用
__global__ void applyLutKernel(unsigned char *output, const unsigned char *input,
                                const unsigned char *lut, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        output[idx] = lut[input[idx]];              // R
        output[idx + 1] = lut[256 + input[idx + 1]]; // G
        output[idx + 2] = lut[512 + input[idx + 2]]; // B
    }
}

void demoVideoFrameProcessing() {
    printf("=== 第二部分：视频帧处理 ===\n\n");

    const int width = 1920;
    const int height = 1080;
    const int channels = 3;
    size_t frameSize = width * height * channels;

    // 分配 pinned memory (加速传输)
    unsigned char *h_input, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_input, frameSize));
    CHECK_CUDA(cudaMallocHost(&h_output, frameSize));

    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, frameSize));
    CHECK_CUDA(cudaMalloc(&d_output, frameSize));

    // 生成测试帧
    for (int i = 0; i < width * height * channels; i++) {
        h_input[i] = rand() % 256;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, frameSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("视频帧处理性能 (1920x1080 RGB):\n");
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 操作                    │ 时间 (ms) │ 可支持帧率      │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");

    const int NUM_RUNS = 100;
    float elapsed;

    // 亮度/对比度
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        brightnessContrastKernel<<<grid, block>>>(d_output, d_input, width, height, 3, 10.0f, 1.2f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 亮度/对比度             │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    // 饱和度
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        saturationKernel<<<grid, block>>>(d_output, d_input, width, height, 1.5f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 饱和度                  │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    // 色温
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        colorTemperatureKernel<<<grid, block>>>(d_output, d_input, width, height, 0.5f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 色温                    │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第三部分：视频滤波器
// ============================================================================

// 高斯模糊 (分离式)
__constant__ float d_gaussianWeights[7] = {
    0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f
};

__global__ void gaussianBlurHorizontal(unsigned char *output, const unsigned char *input,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int k = -3; k <= 3; k++) {
            int px = min(max(x + k, 0), width - 1);
            int idx = (y * width + px) * 3;
            float w = d_gaussianWeights[k + 3];
            sum[0] += w * input[idx];
            sum[1] += w * input[idx + 1];
            sum[2] += w * input[idx + 2];
        }

        int outIdx = (y * width + x) * 3;
        output[outIdx] = (unsigned char)sum[0];
        output[outIdx + 1] = (unsigned char)sum[1];
        output[outIdx + 2] = (unsigned char)sum[2];
    }
}

__global__ void gaussianBlurVertical(unsigned char *output, const unsigned char *input,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int k = -3; k <= 3; k++) {
            int py = min(max(y + k, 0), height - 1);
            int idx = (py * width + x) * 3;
            float w = d_gaussianWeights[k + 3];
            sum[0] += w * input[idx];
            sum[1] += w * input[idx + 1];
            sum[2] += w * input[idx + 2];
        }

        int outIdx = (y * width + x) * 3;
        output[outIdx] = (unsigned char)sum[0];
        output[outIdx + 1] = (unsigned char)sum[1];
        output[outIdx + 2] = (unsigned char)sum[2];
    }
}

// 锐化滤波
__global__ void sharpenKernel(unsigned char *output, const unsigned char *input,
                               int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        for (int c = 0; c < 3; c++) {
            int idx = (y * width + x) * 3 + c;

            float center = input[idx];
            float neighbors = input[((y-1) * width + x) * 3 + c] +
                            input[((y+1) * width + x) * 3 + c] +
                            input[(y * width + x-1) * 3 + c] +
                            input[(y * width + x+1) * 3 + c];

            // Unsharp masking
            float val = center + strength * (4.0f * center - neighbors);
            output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    } else if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }
}

// 运动模糊
__global__ void motionBlurKernel(unsigned char *output, const unsigned char *input,
                                  int width, int height, int blurLength, float angle) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float cosA = cosf(angle);
        float sinA = sinf(angle);

        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int i = -blurLength/2; i <= blurLength/2; i++) {
            int px = x + (int)(i * cosA);
            int py = y + (int)(i * sinA);
            px = min(max(px, 0), width - 1);
            py = min(max(py, 0), height - 1);

            int idx = (py * width + px) * 3;
            sum[0] += input[idx];
            sum[1] += input[idx + 1];
            sum[2] += input[idx + 2];
        }

        float scale = 1.0f / (blurLength + 1);
        int outIdx = (y * width + x) * 3;
        output[outIdx] = (unsigned char)(sum[0] * scale);
        output[outIdx + 1] = (unsigned char)(sum[1] * scale);
        output[outIdx + 2] = (unsigned char)(sum[2] * scale);
    }
}

// 双边滤波 (边缘保持去噪)
__global__ void bilateralFilterKernel(unsigned char *output, const unsigned char *input,
                                       int width, int height, int radius,
                                       float sigmaSpace, float sigmaColor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int centerIdx = (y * width + x) * 3;
        float centerR = input[centerIdx];
        float centerG = input[centerIdx + 1];
        float centerB = input[centerIdx + 2];

        float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
        float sumWeight = 0.0f;

        float sigmaSpace2 = 2.0f * sigmaSpace * sigmaSpace;
        float sigmaColor2 = 2.0f * sigmaColor * sigmaColor;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int px = min(max(x + dx, 0), width - 1);
                int py = min(max(y + dy, 0), height - 1);

                int idx = (py * width + px) * 3;
                float r = input[idx];
                float g = input[idx + 1];
                float b = input[idx + 2];

                // 空间权重
                float spaceWeight = expf(-(dx*dx + dy*dy) / sigmaSpace2);

                // 颜色权重
                float colorDiff = (r - centerR) * (r - centerR) +
                                 (g - centerG) * (g - centerG) +
                                 (b - centerB) * (b - centerB);
                float colorWeight = expf(-colorDiff / sigmaColor2);

                float weight = spaceWeight * colorWeight;
                sumR += weight * r;
                sumG += weight * g;
                sumB += weight * b;
                sumWeight += weight;
            }
        }

        output[centerIdx] = (unsigned char)(sumR / sumWeight);
        output[centerIdx + 1] = (unsigned char)(sumG / sumWeight);
        output[centerIdx + 2] = (unsigned char)(sumB / sumWeight);
    }
}

void demoVideoFilters() {
    printf("=== 第三部分：视频滤波器 ===\n\n");

    const int width = 1920;
    const int height = 1080;
    size_t frameSize = width * height * 3;

    unsigned char *d_input, *d_output, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_input, frameSize));
    CHECK_CUDA(cudaMalloc(&d_output, frameSize));
    CHECK_CUDA(cudaMalloc(&d_temp, frameSize));

    // 初始化测试数据
    unsigned char *h_input = (unsigned char*)malloc(frameSize);
    for (size_t i = 0; i < frameSize; i++) h_input[i] = rand() % 256;
    CHECK_CUDA(cudaMemcpy(d_input, h_input, frameSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("视频滤波性能 (1920x1080):\n");
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 滤波器                  │ 时间 (ms) │ 可支持帧率      │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");

    const int NUM_RUNS = 50;
    float elapsed;

    // 高斯模糊
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        gaussianBlurHorizontal<<<grid, block>>>(d_temp, d_input, width, height);
        gaussianBlurVertical<<<grid, block>>>(d_output, d_temp, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 高斯模糊 (7x7)          │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    // 锐化
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        sharpenKernel<<<grid, block>>>(d_output, d_input, width, height, 0.5f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 锐化                    │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    // 运动模糊
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        motionBlurKernel<<<grid, block>>>(d_output, d_input, width, height, 15, 0.0f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 运动模糊                │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    // 双边滤波
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        bilateralFilterKernel<<<grid, block>>>(d_output, d_input, width, height, 5, 10.0f, 30.0f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 双边滤波 (r=5)          │ %9.3f │ %7.0f fps     │\n", elapsed, 1000.0f/elapsed);

    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");

    free(h_input);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp));
}

// ============================================================================
// 第四部分：音频信号处理
// ============================================================================

// 音频增益调整
__global__ void audioGainKernel(float *output, const float *input, int n, float gain) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = fminf(1.0f, fmaxf(-1.0f, input[tid] * gain));
    }
}

// 音频软限幅
__global__ void audioSoftClipKernel(float *output, const float *input, int n, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        float absX = fabsf(x);
        if (absX <= threshold) {
            output[tid] = x;
        } else {
            // 软限幅: 超过阈值后使用 tanh 压缩
            float sign = x > 0 ? 1.0f : -1.0f;
            output[tid] = sign * (threshold + (1.0f - threshold) * tanhf((absX - threshold) / (1.0f - threshold)));
        }
    }
}

// 简单低通滤波 (一阶 IIR)
__global__ void audioLowPassKernel(float *output, const float *input, int n, float alpha) {
    // 注意: 这是简化版本，实际需要处理因果性
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (tid == 0) {
            output[tid] = input[tid];
        } else {
            output[tid] = alpha * input[tid] + (1.0f - alpha) * output[tid - 1];
        }
    }
}

// FIR 滤波器
#define FIR_TAPS 64

__constant__ float d_firCoeffs[FIR_TAPS];

__global__ void audioFirFilterKernel(float *output, const float *input, int n) {
    extern __shared__ float smem[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;

    // 加载到共享内存 (包含历史样本)
    if (tid < n) {
        smem[localTid + FIR_TAPS - 1] = input[tid];
    }

    // 加载历史样本
    if (localTid < FIR_TAPS - 1) {
        int histIdx = blockIdx.x * blockDim.x + localTid - (FIR_TAPS - 1);
        smem[localTid] = (histIdx >= 0) ? input[histIdx] : 0.0f;
    }

    __syncthreads();

    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < FIR_TAPS; i++) {
            sum += d_firCoeffs[i] * smem[localTid + FIR_TAPS - 1 - i];
        }
        output[tid] = sum;
    }
}

// 频谱分析
void performFFTAnalysis(cufftComplex *d_spectrum, float *d_audio, int n) {
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_R2C, 1));

    CHECK_CUFFT(cufftExecR2C(plan, d_audio, d_spectrum));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUFFT(cufftDestroy(plan));
}

// 计算频谱幅度
__global__ void spectrumMagnitudeKernel(float *magnitude, const cufftComplex *spectrum, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float real = spectrum[tid].x;
        float imag = spectrum[tid].y;
        magnitude[tid] = sqrtf(real * real + imag * imag);
    }
}

// 频谱均衡器 (简化版)
__global__ void equalizerKernel(cufftComplex *spectrum, int n, int sampleRate,
                                 float bassGain, float midGain, float trebleGain) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 计算对应频率
        float freq = (float)tid * sampleRate / (2 * n);

        float gain;
        if (freq < 250.0f) {
            gain = bassGain;       // 低频
        } else if (freq < 4000.0f) {
            gain = midGain;        // 中频
        } else {
            gain = trebleGain;     // 高频
        }

        spectrum[tid].x *= gain;
        spectrum[tid].y *= gain;
    }
}

void demoAudioProcessing() {
    printf("=== 第四部分：音频信号处理 ===\n\n");

    const int sampleRate = 48000;
    const int duration = 1;  // 秒
    const int n = sampleRate * duration;

    float *h_audio = (float*)malloc(n * sizeof(float));
    float *d_audio, *d_output;
    CHECK_CUDA(cudaMalloc(&d_audio, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));

    // 生成测试音频 (多频率叠加)
    for (int i = 0; i < n; i++) {
        float t = (float)i / sampleRate;
        h_audio[i] = 0.3f * sinf(2 * M_PI * 440 * t) +   // A4
                     0.2f * sinf(2 * M_PI * 880 * t) +   // A5
                     0.1f * sinf(2 * M_PI * 1760 * t);   // A6
    }
    CHECK_CUDA(cudaMemcpy(d_audio, h_audio, n * sizeof(float), cudaMemcpyHostToDevice));

    // 设置 FIR 滤波器系数 (低通)
    float h_firCoeffs[FIR_TAPS];
    float fc = 2000.0f / sampleRate;  // 归一化截止频率
    for (int i = 0; i < FIR_TAPS; i++) {
        int k = i - FIR_TAPS / 2;
        if (k == 0) {
            h_firCoeffs[i] = 2 * fc;
        } else {
            h_firCoeffs[i] = sinf(2 * M_PI * fc * k) / (M_PI * k);
        }
        // Hamming 窗
        h_firCoeffs[i] *= 0.54f - 0.46f * cosf(2 * M_PI * i / (FIR_TAPS - 1));
    }
    CHECK_CUDA(cudaMemcpyToSymbol(d_firCoeffs, h_firCoeffs, FIR_TAPS * sizeof(float)));

    dim3 block(256);
    dim3 grid((n + 255) / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("音频处理性能 (%d Hz, %d 样本):\n", sampleRate, n);
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 操作                    │ 时间 (ms) │ 实时比率        │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");

    const int NUM_RUNS = 100;
    float elapsed;
    float audioDurationMs = duration * 1000.0f;

    // 增益调整
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        audioGainKernel<<<grid, block>>>(d_output, d_audio, n, 1.5f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 增益调整                │ %9.3f │ %7.0fx        │\n", elapsed, audioDurationMs/elapsed);

    // 软限幅
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        audioSoftClipKernel<<<grid, block>>>(d_output, d_audio, n, 0.8f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 软限幅                  │ %9.3f │ %7.0fx        │\n", elapsed, audioDurationMs/elapsed);

    // FIR 滤波
    size_t smemSize = (block.x + FIR_TAPS - 1) * sizeof(float);
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        audioFirFilterKernel<<<grid, block, smemSize>>>(d_output, d_audio, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ FIR 滤波 (%d taps)      │ %9.3f │ %7.0fx        │\n", FIR_TAPS, elapsed, audioDurationMs/elapsed);

    // FFT 频谱分析
    cufftComplex *d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_spectrum, (n/2 + 1) * sizeof(cufftComplex)));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_R2C, 1));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        CHECK_CUFFT(cufftExecR2C(plan, d_audio, d_spectrum));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ FFT 分析 (%d点)        │ %9.3f │ %7.0fx        │\n", n, elapsed, audioDurationMs/elapsed);

    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_spectrum));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_audio);
    CHECK_CUDA(cudaFree(d_audio));
    CHECK_CUDA(cudaFree(d_output));
}

// ============================================================================
// 第五部分：实时流水线设计
// ============================================================================

// 双缓冲视频处理类
class VideoProcessingPipeline {
public:
    static const int NUM_BUFFERS = 2;

    int width, height;
    size_t frameSize;

    unsigned char *h_input[NUM_BUFFERS];
    unsigned char *h_output[NUM_BUFFERS];
    unsigned char *d_input[NUM_BUFFERS];
    unsigned char *d_output[NUM_BUFFERS];
    unsigned char *d_temp;

    cudaStream_t streams[NUM_BUFFERS];
    cudaEvent_t events[NUM_BUFFERS];

    VideoProcessingPipeline(int w, int h) : width(w), height(h) {
        frameSize = w * h * 3;

        for (int i = 0; i < NUM_BUFFERS; i++) {
            CHECK_CUDA(cudaMallocHost(&h_input[i], frameSize));
            CHECK_CUDA(cudaMallocHost(&h_output[i], frameSize));
            CHECK_CUDA(cudaMalloc(&d_input[i], frameSize));
            CHECK_CUDA(cudaMalloc(&d_output[i], frameSize));
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
            CHECK_CUDA(cudaEventCreate(&events[i]));
        }
        CHECK_CUDA(cudaMalloc(&d_temp, frameSize));
    }

    ~VideoProcessingPipeline() {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            cudaFreeHost(h_input[i]);
            cudaFreeHost(h_output[i]);
            cudaFree(d_input[i]);
            cudaFree(d_output[i]);
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(events[i]);
        }
        cudaFree(d_temp);
    }

    void processFrameAsync(int bufferIdx) {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        cudaStream_t stream = streams[bufferIdx];

        // 1. 异步传输到 GPU
        CHECK_CUDA(cudaMemcpyAsync(d_input[bufferIdx], h_input[bufferIdx],
                                    frameSize, cudaMemcpyHostToDevice, stream));

        // 2. 处理链: 亮度对比度 → 高斯模糊 → 锐化
        brightnessContrastKernel<<<grid, block, 0, stream>>>(
            d_temp, d_input[bufferIdx], width, height, 3, 5.0f, 1.1f);

        gaussianBlurHorizontal<<<grid, block, 0, stream>>>(
            d_output[bufferIdx], d_temp, width, height);

        gaussianBlurVertical<<<grid, block, 0, stream>>>(
            d_temp, d_output[bufferIdx], width, height);

        sharpenKernel<<<grid, block, 0, stream>>>(
            d_output[bufferIdx], d_temp, width, height, 0.3f);

        // 3. 异步传输回 CPU
        CHECK_CUDA(cudaMemcpyAsync(h_output[bufferIdx], d_output[bufferIdx],
                                    frameSize, cudaMemcpyDeviceToHost, stream));

        // 4. 记录完成事件
        CHECK_CUDA(cudaEventRecord(events[bufferIdx], stream));
    }

    void waitForFrame(int bufferIdx) {
        CHECK_CUDA(cudaEventSynchronize(events[bufferIdx]));
    }
};

void demoRealtimePipeline() {
    printf("=== 第五部分：实时流水线设计 ===\n\n");

    printf("双缓冲流水线结构:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  时间 →                                                   │\n");
    printf("  │                                                           │\n");
    printf("  │  缓冲A: [采集] [处理] [显示] [采集] [处理] [显示]          │\n");
    printf("  │  缓冲B:        [采集] [处理] [显示] [采集] [处理] [显示]   │\n");
    printf("  │                                                           │\n");
    printf("  │  重叠执行提高吞吐量                                       │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    const int width = 1920;
    const int height = 1080;
    const int numFrames = 100;

    VideoProcessingPipeline pipeline(width, height);

    // 初始化测试数据
    for (int i = 0; i < VideoProcessingPipeline::NUM_BUFFERS; i++) {
        for (size_t j = 0; j < pipeline.frameSize; j++) {
            pipeline.h_input[i][j] = rand() % 256;
        }
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    pipeline.processFrameAsync(0);
    pipeline.waitForFrame(0);

    CHECK_CUDA(cudaEventRecord(start));

    // 流水线处理
    int currentBuffer = 0;
    for (int frame = 0; frame < numFrames; frame++) {
        // 提交当前帧处理
        pipeline.processFrameAsync(currentBuffer);

        // 等待上一帧完成 (如果有)
        if (frame > 0) {
            int prevBuffer = 1 - currentBuffer;
            pipeline.waitForFrame(prevBuffer);
            // 这里可以使用处理完的 h_output[prevBuffer]
        }

        currentBuffer = 1 - currentBuffer;
    }

    // 等待最后一帧
    pipeline.waitForFrame(1 - currentBuffer);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    printf("流水线性能 (%d 帧 @ %dx%d):\n", numFrames, width, height);
    printf("  总时间: %.2f ms\n", elapsed);
    printf("  平均每帧: %.3f ms\n", elapsed / numFrames);
    printf("  吞吐量: %.2f fps\n\n", numFrames / elapsed * 1000.0f);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第六部分：硬件编解码器集成概念
// ============================================================================

void demoHardwareCodec() {
    printf("=== 第六部分：硬件编解码器 ===\n\n");

    printf("NVIDIA 硬件编解码器:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ NVENC (编码器)                                            │\n");
    printf("  │   - H.264, H.265 (HEVC), AV1 编码                         │\n");
    printf("  │   - 硬件加速，不占用 CUDA 核心                            │\n");
    printf("  │   - 支持多路并行编码                                      │\n");
    printf("  │                                                           │\n");
    printf("  │ NVDEC (解码器)                                            │\n");
    printf("  │   - H.264, H.265, VP9, AV1 解码                           │\n");
    printf("  │   - 直接输出到 GPU 内存                                   │\n");
    printf("  │   - 4K/8K 实时解码                                        │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    printf("集成方式:\n");
    printf("  1. NVIDIA Video Codec SDK\n");
    printf("     - 直接 API 访问\n");
    printf("     - 最大灵活性和性能\n\n");

    printf("  2. FFmpeg (硬件加速)\n");
    printf("     - -c:v h264_nvenc (编码)\n");
    printf("     - -hwaccel cuda (解码)\n\n");

    printf("  3. GStreamer\n");
    printf("     - nvh264enc / nvh264dec 插件\n");
    printf("     - 流媒体管道集成\n\n");

    printf("典型流水线:\n");
    printf("  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐\n");
    printf("  │ NVDEC   │ →  │ CUDA    │ →  │ CUDA    │ →  │ NVENC   │\n");
    printf("  │ 解码    │    │ 处理    │    │ 推理    │    │ 编码    │\n");
    printf("  └─────────┘    └─────────┘    └─────────┘    └─────────┘\n");
    printf("       ↓              ↓              ↓              ↓\n");
    printf("    GPU内存 ────────────────────────────────────→ GPU内存\n");
    printf("                (零拷贝，全程 GPU 处理)\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 32: 实时视频/音频处理                                ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("异步引擎数: %d\n\n", prop.asyncEngineCount);

    demoRealtimeConcepts();
    demoVideoFrameProcessing();
    demoVideoFilters();
    demoAudioProcessing();
    demoRealtimePipeline();
    demoHardwareCodec();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("实时处理关键技术:\n");
    printf("  ✓ 双缓冲/三缓冲避免等待\n");
    printf("  ✓ CUDA Streams 异步执行\n");
    printf("  ✓ Pinned Memory 加速传输\n");
    printf("  ✓ 内存池预分配\n\n");

    printf("视频处理:\n");
    printf("  ✓ 色彩空间转换 (YUV ↔ RGB)\n");
    printf("  ✓ 色彩校正 (亮度、对比度、饱和度)\n");
    printf("  ✓ 滤波 (高斯、锐化、双边)\n");
    printf("  ✓ NVENC/NVDEC 硬件编解码\n\n");

    printf("音频处理:\n");
    printf("  ✓ 增益和软限幅\n");
    printf("  ✓ FIR/IIR 滤波\n");
    printf("  ✓ FFT 频谱分析\n");
    printf("  ✓ 均衡器\n\n");

    printf("性能优化:\n");
    printf("  - 流水线隐藏延迟\n");
    printf("  - 算子融合减少访存\n");
    printf("  - 共享内存减少全局访存\n");
    printf("  - 硬件编解码卸载\n\n");

    return 0;
}
