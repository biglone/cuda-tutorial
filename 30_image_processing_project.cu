/**
 * =============================================================================
 * CUDA 教程 30: 实战项目 - GPU 加速图像处理
 * =============================================================================
 *
 * 学习目标：
 * 1. 综合运用前面所学的 CUDA 技术
 * 2. 实现常见图像处理算法的 GPU 加速版本
 * 3. 学习图像处理优化策略
 * 4. 完成一个完整的 CUDA 项目
 *
 * 实现功能：
 * - 灰度转换
 * - 图像滤波 (高斯模糊、边缘检测)
 * - 直方图计算与均衡化
 * - 图像缩放
 * - 色彩空间转换
 *
 * 编译命令：
 *   nvcc 30_image_processing_project.cu -o 30_image_processing -O3
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
// 图像数据结构
// ============================================================================

// RGB 像素
struct RGB {
    unsigned char r, g, b;
};

// RGBA 像素
struct RGBA {
    unsigned char r, g, b, a;
};

// 图像类
class CudaImage {
public:
    int width, height;
    unsigned char *h_data;  // 主机数据
    unsigned char *d_data;  // 设备数据
    int channels;

    CudaImage(int w, int h, int c = 3)
        : width(w), height(h), channels(c), h_data(nullptr), d_data(nullptr) {
        size_t size = w * h * c * sizeof(unsigned char);
        h_data = (unsigned char*)malloc(size);
        CHECK_CUDA(cudaMalloc(&d_data, size));
    }

    ~CudaImage() {
        if (h_data) free(h_data);
        if (d_data) cudaFree(d_data);
    }

    size_t size() const { return width * height * channels * sizeof(unsigned char); }

    void toDevice() {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size(), cudaMemcpyHostToDevice));
    }

    void toHost() {
        CHECK_CUDA(cudaMemcpy(h_data, d_data, size(), cudaMemcpyDeviceToHost));
    }

    // 生成测试图像
    void generateTestPattern() {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels;
                // 渐变模式
                h_data[idx] = (unsigned char)(x * 255 / width);       // R
                h_data[idx + 1] = (unsigned char)(y * 255 / height);  // G
                h_data[idx + 2] = (unsigned char)((x + y) * 127 / (width + height));  // B
            }
        }
    }
};

// ============================================================================
// 第一部分：灰度转换
// ============================================================================

// 基本灰度转换内核
__global__ void rgbToGrayscaleKernel(unsigned char *output, const unsigned char *input,
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int rgbIdx = idx * 3;

        unsigned char r = input[rgbIdx];
        unsigned char g = input[rgbIdx + 1];
        unsigned char b = input[rgbIdx + 2];

        // ITU-R BT.601 标准
        // Gray = 0.299*R + 0.587*G + 0.114*B
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, gray));
    }
}

// 优化版本：使用向量加载
__global__ void rgbToGrayscaleOptimized(unsigned char *output, const uchar3 *input,
                                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = input[idx];

        // 使用整数近似避免浮点运算
        // (77*R + 150*G + 29*B) >> 8 ≈ 0.299*R + 0.587*G + 0.114*B
        int gray = (77 * pixel.x + 150 * pixel.y + 29 * pixel.z) >> 8;
        output[idx] = (unsigned char)min(255, max(0, gray));
    }
}

void demoGrayscaleConversion() {
    printf("=== 第一部分：灰度转换 ===\n\n");

    const int width = 1920;
    const int height = 1080;

    CudaImage src(width, height, 3);
    src.generateTestPattern();
    src.toDevice();

    unsigned char *d_gray;
    CHECK_CUDA(cudaMalloc(&d_gray, width * height));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 预热
    rgbToGrayscaleKernel<<<grid, block>>>(d_gray, src.d_data, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 基本版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        rgbToGrayscaleKernel<<<grid, block>>>(d_gray, src.d_data, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float basicTime;
    CHECK_CUDA(cudaEventElapsedTime(&basicTime, start, stop));
    basicTime /= NUM_RUNS;

    // 优化版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        rgbToGrayscaleOptimized<<<grid, block>>>(d_gray, (uchar3*)src.d_data, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float optTime;
    CHECK_CUDA(cudaEventElapsedTime(&optTime, start, stop));
    optTime /= NUM_RUNS;

    printf("灰度转换性能 (%dx%d):\n", width, height);
    printf("  基本版本: %.3f ms\n", basicTime);
    printf("  优化版本: %.3f ms\n", optTime);
    printf("  像素吞吐: %.2f MPixels/s\n\n", width * height / optTime / 1000.0f);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_gray));
}

// ============================================================================
// 第二部分：高斯模糊
// ============================================================================

// 高斯核 (5x5)
__constant__ float d_gaussianKernel[5][5] = {
    {1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f},
    {4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f},
    {7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f},
    {4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f},
    {1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f}
};

// 基本高斯模糊
__global__ void gaussianBlurBasic(unsigned char *output, const unsigned char *input,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                int idx = (py * width + px) * 3;

                float weight = d_gaussianKernel[ky + 2][kx + 2];
                sum[0] += weight * input[idx];
                sum[1] += weight * input[idx + 1];
                sum[2] += weight * input[idx + 2];
            }
        }

        int outIdx = (y * width + x) * 3;
        output[outIdx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum[0]));
        output[outIdx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum[1]));
        output[outIdx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum[2]));
    }
}

// 使用共享内存的高斯模糊
#define TILE_SIZE 16
#define APRON 2
#define BLOCK_SIZE (TILE_SIZE + 2 * APRON)

__global__ void gaussianBlurShared(unsigned char *output, const unsigned char *input,
                                    int width, int height) {
    __shared__ unsigned char smem[BLOCK_SIZE][BLOCK_SIZE][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx - APRON;
    int y = blockIdx.y * TILE_SIZE + ty - APRON;

    // 边界处理
    int px = min(max(x, 0), width - 1);
    int py = min(max(y, 0), height - 1);
    int idx = (py * width + px) * 3;

    // 加载到共享内存
    smem[ty][tx][0] = input[idx];
    smem[ty][tx][1] = input[idx + 1];
    smem[ty][tx][2] = input[idx + 2];

    __syncthreads();

    // 只有 tile 内的线程计算
    if (tx >= APRON && tx < TILE_SIZE + APRON &&
        ty >= APRON && ty < TILE_SIZE + APRON) {
        int outX = x;
        int outY = y;

        if (outX >= 0 && outX < width && outY >= 0 && outY < height) {
            float sum[3] = {0.0f, 0.0f, 0.0f};

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int sy = ty + ky;
                    int sx = tx + kx;
                    float weight = d_gaussianKernel[ky + 2][kx + 2];
                    sum[0] += weight * smem[sy][sx][0];
                    sum[1] += weight * smem[sy][sx][1];
                    sum[2] += weight * smem[sy][sx][2];
                }
            }

            int outIdx = (outY * width + outX) * 3;
            output[outIdx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum[0]));
            output[outIdx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum[1]));
            output[outIdx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum[2]));
        }
    }
}

// 分离式高斯模糊 (更高效)
__constant__ float d_gaussianKernel1D[5] = {
    0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f
};

__global__ void gaussianBlurHorizontal(unsigned char *output, const unsigned char *input,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int k = -2; k <= 2; k++) {
            int px = min(max(x + k, 0), width - 1);
            int idx = (y * width + px) * 3;
            float weight = d_gaussianKernel1D[k + 2];
            sum[0] += weight * input[idx];
            sum[1] += weight * input[idx + 1];
            sum[2] += weight * input[idx + 2];
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

        for (int k = -2; k <= 2; k++) {
            int py = min(max(y + k, 0), height - 1);
            int idx = (py * width + x) * 3;
            float weight = d_gaussianKernel1D[k + 2];
            sum[0] += weight * input[idx];
            sum[1] += weight * input[idx + 1];
            sum[2] += weight * input[idx + 2];
        }

        int outIdx = (y * width + x) * 3;
        output[outIdx] = (unsigned char)sum[0];
        output[outIdx + 1] = (unsigned char)sum[1];
        output[outIdx + 2] = (unsigned char)sum[2];
    }
}

void demoGaussianBlur() {
    printf("=== 第二部分：高斯模糊 ===\n\n");

    const int width = 1920;
    const int height = 1080;

    CudaImage src(width, height, 3);
    CudaImage dst(width, height, 3);
    src.generateTestPattern();
    src.toDevice();

    unsigned char *d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, width * height * 3));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    dim3 blockShared(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridShared((width + TILE_SIZE - 1) / TILE_SIZE,
                    (height + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 50;

    // 基本版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        gaussianBlurBasic<<<grid, block>>>(dst.d_data, src.d_data, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float basicTime;
    CHECK_CUDA(cudaEventElapsedTime(&basicTime, start, stop));
    basicTime /= NUM_RUNS;

    // 共享内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        gaussianBlurShared<<<gridShared, blockShared>>>(dst.d_data, src.d_data, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float sharedTime;
    CHECK_CUDA(cudaEventElapsedTime(&sharedTime, start, stop));
    sharedTime /= NUM_RUNS;

    // 分离式版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        gaussianBlurHorizontal<<<grid, block>>>(d_temp, src.d_data, width, height);
        gaussianBlurVertical<<<grid, block>>>(dst.d_data, d_temp, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float separableTime;
    CHECK_CUDA(cudaEventElapsedTime(&separableTime, start, stop));
    separableTime /= NUM_RUNS;

    printf("高斯模糊性能 (%dx%d, 5x5核):\n", width, height);
    printf("  ┌─────────────────────┬───────────┬─────────────┐\n");
    printf("  │ 方法                │ 时间 (ms) │ 加速比      │\n");
    printf("  ├─────────────────────┼───────────┼─────────────┤\n");
    printf("  │ 基本版本            │ %9.3f │     1.00x   │\n", basicTime);
    printf("  │ 共享内存            │ %9.3f │     %.2fx   │\n", sharedTime, basicTime/sharedTime);
    printf("  │ 分离式 (2-pass)     │ %9.3f │     %.2fx   │\n", separableTime, basicTime/separableTime);
    printf("  └─────────────────────┴───────────┴─────────────┘\n\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_temp));
}

// ============================================================================
// 第三部分：Sobel 边缘检测
// ============================================================================

// Sobel 算子
__constant__ int d_sobelX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int d_sobelY[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__global__ void sobelEdgeDetection(unsigned char *output, const unsigned char *input,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int idx = ((y + ky) * width + (x + kx)) * 3;
                // 转换为灰度
                float gray = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];
                gx += d_sobelX[ky + 1][kx + 1] * gray;
                gy += d_sobelY[ky + 1][kx + 1] * gray;
            }
        }

        // 梯度幅值
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = (unsigned char)fminf(255.0f, magnitude);
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

// 优化版本：使用共享内存
__global__ void sobelEdgeDetectionShared(unsigned char *output, const unsigned char *input,
                                          int width, int height) {
    __shared__ float smem[18][18];  // 16x16 tile + 1 边界

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    // 加载数据并转换为灰度
    int px = min(max(x, 0), width - 1);
    int py = min(max(y, 0), height - 1);
    int idx = (py * width + px) * 3;
    smem[ty + 1][tx + 1] = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];

    // 加载边界
    if (tx == 0) {
        px = max(x - 1, 0);
        idx = (py * width + px) * 3;
        smem[ty + 1][0] = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];
    }
    if (tx == 15) {
        px = min(x + 1, width - 1);
        idx = (py * width + px) * 3;
        smem[ty + 1][17] = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];
    }
    if (ty == 0) {
        py = max(y - 1, 0);
        px = min(max(x, 0), width - 1);
        idx = (py * width + px) * 3;
        smem[0][tx + 1] = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];
    }
    if (ty == 15) {
        py = min(y + 1, height - 1);
        px = min(max(x, 0), width - 1);
        idx = (py * width + px) * 3;
        smem[17][tx + 1] = 0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2];
    }

    __syncthreads();

    if (x < width && y < height) {
        float gx = 0.0f, gy = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                float val = smem[ty + 1 + ky][tx + 1 + kx];
                gx += d_sobelX[ky + 1][kx + 1] * val;
                gy += d_sobelY[ky + 1][kx + 1] * val;
            }
        }

        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = (unsigned char)fminf(255.0f, magnitude);
    }
}

void demoSobelEdgeDetection() {
    printf("=== 第三部分：Sobel 边缘检测 ===\n\n");

    const int width = 1920;
    const int height = 1080;

    CudaImage src(width, height, 3);
    src.generateTestPattern();
    src.toDevice();

    unsigned char *d_edges;
    CHECK_CUDA(cudaMalloc(&d_edges, width * height));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 基本版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        sobelEdgeDetection<<<grid, block>>>(d_edges, src.d_data, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float basicTime;
    CHECK_CUDA(cudaEventElapsedTime(&basicTime, start, stop));
    basicTime /= NUM_RUNS;

    // 共享内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        sobelEdgeDetectionShared<<<grid, block>>>(d_edges, src.d_data, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float sharedTime;
    CHECK_CUDA(cudaEventElapsedTime(&sharedTime, start, stop));
    sharedTime /= NUM_RUNS;

    printf("Sobel 边缘检测性能 (%dx%d):\n", width, height);
    printf("  基本版本: %.3f ms\n", basicTime);
    printf("  共享内存: %.3f ms\n", sharedTime);
    printf("  加速比: %.2fx\n\n", basicTime / sharedTime);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_edges));
}

// ============================================================================
// 第四部分：直方图计算
// ============================================================================

#define HISTOGRAM_BINS 256

// 基本直方图 (原子操作)
__global__ void histogramAtomic(unsigned int *hist, const unsigned char *input,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        atomicAdd(&hist[input[idx]], 1);
    }
}

// 使用共享内存的直方图
__global__ void histogramShared(unsigned int *hist, const unsigned char *input,
                                 int width, int height) {
    __shared__ unsigned int localHist[HISTOGRAM_BINS];

    // 初始化局部直方图
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < HISTOGRAM_BINS) {
        localHist[tid] = 0;
    }
    __syncthreads();

    // 计算局部直方图
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        atomicAdd(&localHist[input[idx]], 1);
    }
    __syncthreads();

    // 合并到全局直方图
    if (tid < HISTOGRAM_BINS) {
        atomicAdd(&hist[tid], localHist[tid]);
    }
}

// 直方图均衡化 LUT 计算
__global__ void computeEqualizationLUT(unsigned char *lut, const unsigned int *hist,
                                        int totalPixels) {
    int bin = threadIdx.x;

    // 前缀和
    __shared__ unsigned int cdf[HISTOGRAM_BINS];
    cdf[bin] = hist[bin];
    __syncthreads();

    // Hillis-Steele 前缀和
    for (int stride = 1; stride < HISTOGRAM_BINS; stride *= 2) {
        unsigned int temp = 0;
        if (bin >= stride) {
            temp = cdf[bin - stride];
        }
        __syncthreads();
        if (bin >= stride) {
            cdf[bin] += temp;
        }
        __syncthreads();
    }

    // 计算均衡化后的值
    float scale = 255.0f / totalPixels;
    lut[bin] = (unsigned char)(cdf[bin] * scale);
}

// 应用 LUT
__global__ void applyLUT(unsigned char *output, const unsigned char *input,
                          const unsigned char *lut, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = lut[input[idx]];
    }
}

void demoHistogram() {
    printf("=== 第四部分：直方图计算与均衡化 ===\n\n");

    const int width = 1920;
    const int height = 1080;

    // 生成灰度测试图像
    unsigned char *h_gray = (unsigned char*)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        // 偏暗的图像
        h_gray[i] = (unsigned char)(rand() % 128);
    }

    unsigned char *d_input, *d_output;
    unsigned int *d_hist;
    unsigned char *d_lut;

    CHECK_CUDA(cudaMalloc(&d_input, width * height));
    CHECK_CUDA(cudaMalloc(&d_output, width * height));
    CHECK_CUDA(cudaMalloc(&d_hist, HISTOGRAM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_lut, HISTOGRAM_BINS));

    CHECK_CUDA(cudaMemcpy(d_input, h_gray, width * height, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // 基本直方图
    CHECK_CUDA(cudaMemset(d_hist, 0, HISTOGRAM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        CHECK_CUDA(cudaMemset(d_hist, 0, HISTOGRAM_BINS * sizeof(unsigned int)));
        histogramAtomic<<<grid, block>>>(d_hist, d_input, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float atomicTime;
    CHECK_CUDA(cudaEventElapsedTime(&atomicTime, start, stop));
    atomicTime /= NUM_RUNS;

    // 共享内存直方图
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        CHECK_CUDA(cudaMemset(d_hist, 0, HISTOGRAM_BINS * sizeof(unsigned int)));
        histogramShared<<<grid, block>>>(d_hist, d_input, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float sharedTime;
    CHECK_CUDA(cudaEventElapsedTime(&sharedTime, start, stop));
    sharedTime /= NUM_RUNS;

    printf("直方图计算性能 (%dx%d):\n", width, height);
    printf("  原子操作版本: %.3f ms\n", atomicTime);
    printf("  共享内存版本: %.3f ms\n", sharedTime);
    printf("  加速比: %.2fx\n\n", atomicTime / sharedTime);

    // 直方图均衡化完整流程
    CHECK_CUDA(cudaMemset(d_hist, 0, HISTOGRAM_BINS * sizeof(unsigned int)));
    histogramShared<<<grid, block>>>(d_hist, d_input, width, height);
    computeEqualizationLUT<<<1, HISTOGRAM_BINS>>>(d_lut, d_hist, width * height);
    applyLUT<<<grid, block>>>(d_output, d_input, d_lut, width, height);

    // 验证结果
    unsigned int *h_hist = (unsigned int*)malloc(HISTOGRAM_BINS * sizeof(unsigned int));
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, HISTOGRAM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int sum = 0;
    for (int i = 0; i < HISTOGRAM_BINS; i++) {
        sum += h_hist[i];
    }
    printf("直方图验证:\n");
    printf("  像素总数: %d (期望: %d)\n\n", sum, width * height);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_gray);
    free(h_hist);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_hist));
    CHECK_CUDA(cudaFree(d_lut));
}

// ============================================================================
// 第五部分：图像缩放
// ============================================================================

// 最近邻插值
__global__ void resizeNearestNeighbor(unsigned char *output, const unsigned char *input,
                                       int srcWidth, int srcHeight,
                                       int dstWidth, int dstHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        float srcX = x * (float)srcWidth / dstWidth;
        float srcY = y * (float)srcHeight / dstHeight;

        int sx = (int)srcX;
        int sy = (int)srcY;

        int srcIdx = (sy * srcWidth + sx) * 3;
        int dstIdx = (y * dstWidth + x) * 3;

        output[dstIdx] = input[srcIdx];
        output[dstIdx + 1] = input[srcIdx + 1];
        output[dstIdx + 2] = input[srcIdx + 2];
    }
}

// 双线性插值
__global__ void resizeBilinear(unsigned char *output, const unsigned char *input,
                                int srcWidth, int srcHeight,
                                int dstWidth, int dstHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        float srcX = x * (float)(srcWidth - 1) / (dstWidth - 1);
        float srcY = y * (float)(srcHeight - 1) / (dstHeight - 1);

        int x0 = (int)floorf(srcX);
        int y0 = (int)floorf(srcY);
        int x1 = min(x0 + 1, srcWidth - 1);
        int y1 = min(y0 + 1, srcHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        int dstIdx = (y * dstWidth + x) * 3;

        for (int c = 0; c < 3; c++) {
            float v00 = input[(y0 * srcWidth + x0) * 3 + c];
            float v01 = input[(y0 * srcWidth + x1) * 3 + c];
            float v10 = input[(y1 * srcWidth + x0) * 3 + c];
            float v11 = input[(y1 * srcWidth + x1) * 3 + c];

            float val = v00 * (1 - fx) * (1 - fy) +
                       v01 * fx * (1 - fy) +
                       v10 * (1 - fx) * fy +
                       v11 * fx * fy;

            output[dstIdx + c] = (unsigned char)fminf(255.0f, fmaxf(0.0f, val));
        }
    }
}

void demoImageResize() {
    printf("=== 第五部分：图像缩放 ===\n\n");

    const int srcWidth = 1920;
    const int srcHeight = 1080;
    const int dstWidth = 3840;
    const int dstHeight = 2160;

    CudaImage src(srcWidth, srcHeight, 3);
    CudaImage dstNN(dstWidth, dstHeight, 3);
    CudaImage dstBL(dstWidth, dstHeight, 3);

    src.generateTestPattern();
    src.toDevice();

    dim3 block(16, 16);
    dim3 grid((dstWidth + 15) / 16, (dstHeight + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 50;

    // 最近邻
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        resizeNearestNeighbor<<<grid, block>>>(dstNN.d_data, src.d_data,
                                                srcWidth, srcHeight, dstWidth, dstHeight);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float nnTime;
    CHECK_CUDA(cudaEventElapsedTime(&nnTime, start, stop));
    nnTime /= NUM_RUNS;

    // 双线性
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        resizeBilinear<<<grid, block>>>(dstBL.d_data, src.d_data,
                                         srcWidth, srcHeight, dstWidth, dstHeight);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float blTime;
    CHECK_CUDA(cudaEventElapsedTime(&blTime, start, stop));
    blTime /= NUM_RUNS;

    printf("图像缩放性能 (%dx%d → %dx%d):\n", srcWidth, srcHeight, dstWidth, dstHeight);
    printf("  最近邻插值: %.3f ms\n", nnTime);
    printf("  双线性插值: %.3f ms\n", blTime);
    printf("  输出像素吞吐: %.2f MPixels/s (双线性)\n\n",
           dstWidth * dstHeight / blTime / 1000.0f);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 第六部分：色彩空间转换
// ============================================================================

// RGB 到 HSV
__global__ void rgbToHsv(float *h, float *s, float *v,
                          const unsigned char *rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int rgbIdx = idx * 3;

        float r = rgb[rgbIdx] / 255.0f;
        float g = rgb[rgbIdx + 1] / 255.0f;
        float b = rgb[rgbIdx + 2] / 255.0f;

        float maxVal = fmaxf(r, fmaxf(g, b));
        float minVal = fminf(r, fminf(g, b));
        float delta = maxVal - minVal;

        // Value
        v[idx] = maxVal;

        // Saturation
        s[idx] = (maxVal > 0.0f) ? (delta / maxVal) : 0.0f;

        // Hue
        float hue = 0.0f;
        if (delta > 0.0f) {
            if (maxVal == r) {
                hue = 60.0f * fmodf((g - b) / delta + 6.0f, 6.0f);
            } else if (maxVal == g) {
                hue = 60.0f * ((b - r) / delta + 2.0f);
            } else {
                hue = 60.0f * ((r - g) / delta + 4.0f);
            }
        }
        h[idx] = hue;
    }
}

// HSV 到 RGB
__global__ void hsvToRgb(unsigned char *rgb, const float *h, const float *s, const float *v,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int rgbIdx = idx * 3;

        float hue = h[idx];
        float sat = s[idx];
        float val = v[idx];

        float c = val * sat;
        float hPrime = hue / 60.0f;
        float x_ = c * (1.0f - fabsf(fmodf(hPrime, 2.0f) - 1.0f));
        float m = val - c;

        float r, g, b;
        if (hPrime < 1.0f) { r = c; g = x_; b = 0; }
        else if (hPrime < 2.0f) { r = x_; g = c; b = 0; }
        else if (hPrime < 3.0f) { r = 0; g = c; b = x_; }
        else if (hPrime < 4.0f) { r = 0; g = x_; b = c; }
        else if (hPrime < 5.0f) { r = x_; g = 0; b = c; }
        else { r = c; g = 0; b = x_; }

        rgb[rgbIdx] = (unsigned char)((r + m) * 255.0f);
        rgb[rgbIdx + 1] = (unsigned char)((g + m) * 255.0f);
        rgb[rgbIdx + 2] = (unsigned char)((b + m) * 255.0f);
    }
}

void demoColorSpaceConversion() {
    printf("=== 第六部分：色彩空间转换 ===\n\n");

    const int width = 1920;
    const int height = 1080;

    CudaImage src(width, height, 3);
    CudaImage dst(width, height, 3);
    src.generateTestPattern();
    src.toDevice();

    float *d_h, *d_s, *d_v;
    CHECK_CUDA(cudaMalloc(&d_h, width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s, width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, width * height * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int NUM_RUNS = 100;

    // RGB → HSV → RGB
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        rgbToHsv<<<grid, block>>>(d_h, d_s, d_v, src.d_data, width, height);
        hsvToRgb<<<grid, block>>>(dst.d_data, d_h, d_s, d_v, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;

    printf("RGB ↔ HSV 转换性能 (%dx%d):\n", width, height);
    printf("  往返转换时间: %.3f ms\n", elapsed);
    printf("  像素吞吐: %.2f MPixels/s\n\n", width * height / elapsed / 1000.0f);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_h));
    CHECK_CUDA(cudaFree(d_s));
    CHECK_CUDA(cudaFree(d_v));
}

// ============================================================================
// 第七部分：完整图像处理流水线
// ============================================================================

void demoPipeline() {
    printf("=== 第七部分：完整图像处理流水线 ===\n\n");

    printf("典型图像处理流水线:\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  输入图像                                                │\n");
    printf("  │      ↓                                                   │\n");
    printf("  │  去噪 (高斯模糊)                                         │\n");
    printf("  │      ↓                                                   │\n");
    printf("  │  色彩校正 (直方图均衡化 / HSV 调整)                       │\n");
    printf("  │      ↓                                                   │\n");
    printf("  │  缩放/裁剪                                               │\n");
    printf("  │      ↓                                                   │\n");
    printf("  │  特征提取 (边缘检测)                                     │\n");
    printf("  │      ↓                                                   │\n");
    printf("  │  输出结果                                                │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");

    printf("多 Stream 并行:\n");
    printf("  Stream 0: 图像 A 处理\n");
    printf("  Stream 1: 图像 B 处理 (重叠)\n");
    printf("  Stream 2: 图像 C 处理 (重叠)\n\n");

    printf("优化策略:\n");
    printf("  1. 使用 Unified Memory 简化内存管理\n");
    printf("  2. 多 Stream 隐藏传输延迟\n");
    printf("  3. 融合内核减少访存\n");
    printf("  4. 使用纹理内存加速读取\n");
    printf("  5. 异步数据传输\n\n");

    // 演示多 Stream 流水线
    const int width = 1920;
    const int height = 1080;
    const int numImages = 4;

    CudaImage *images[numImages];
    cudaStream_t streams[numImages];

    for (int i = 0; i < numImages; i++) {
        images[i] = new CudaImage(width, height, 3);
        images[i]->generateTestPattern();
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    unsigned char *d_temp[numImages];
    for (int i = 0; i < numImages; i++) {
        CHECK_CUDA(cudaMalloc(&d_temp[i], width * height * 3));
    }

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 并行处理多张图像
    for (int i = 0; i < numImages; i++) {
        // 异步传输
        CHECK_CUDA(cudaMemcpyAsync(images[i]->d_data, images[i]->h_data,
                                    images[i]->size(), cudaMemcpyHostToDevice, streams[i]));

        // 高斯模糊
        gaussianBlurHorizontal<<<grid, block, 0, streams[i]>>>(
            d_temp[i], images[i]->d_data, width, height);
        gaussianBlurVertical<<<grid, block, 0, streams[i]>>>(
            images[i]->d_data, d_temp[i], width, height);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    printf("多 Stream 流水线性能 (%d 张 %dx%d 图像):\n", numImages, width, height);
    printf("  总时间: %.3f ms\n", elapsed);
    printf("  平均每张: %.3f ms\n", elapsed / numImages);
    printf("  吞吐量: %.2f 图像/秒\n\n", numImages / elapsed * 1000.0f);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    for (int i = 0; i < numImages; i++) {
        delete images[i];
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFree(d_temp[i]));
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 30: 实战项目 - GPU 加速图像处理                      ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("SM 数量: %d\n\n", prop.multiProcessorCount);

    demoGrayscaleConversion();
    demoGaussianBlur();
    demoSobelEdgeDetection();
    demoHistogram();
    demoImageResize();
    demoColorSpaceConversion();
    demoPipeline();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       项目总结                                  ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("实现的功能:\n");
    printf("  ✓ 灰度转换 (RGB → Gray)\n");
    printf("  ✓ 高斯模糊 (5x5 核)\n");
    printf("  ✓ Sobel 边缘检测\n");
    printf("  ✓ 直方图计算与均衡化\n");
    printf("  ✓ 图像缩放 (最近邻/双线性)\n");
    printf("  ✓ 色彩空间转换 (RGB ↔ HSV)\n");
    printf("  ✓ 多 Stream 流水线\n\n");

    printf("优化技术应用:\n");
    printf("  - 共享内存减少全局内存访问\n");
    printf("  - 常量内存存储滤波器核\n");
    printf("  - 向量化加载/存储\n");
    printf("  - 分离式卷积\n");
    printf("  - 多 Stream 并行\n\n");

    printf("扩展方向:\n");
    printf("  1. 支持更多图像格式 (PNG, JPEG)\n");
    printf("  2. 添加更多滤波器 (中值、锐化)\n");
    printf("  3. 实现图像金字塔\n");
    printf("  4. 集成深度学习模型\n");
    printf("  5. GPU 视频处理\n\n");

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║             恭喜完成 CUDA 教程学习！                             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
