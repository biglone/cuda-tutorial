/**
 * =============================================================================
 * CUDA 教程 09: 纹理内存 (Texture Memory)
 * =============================================================================
 *
 * 学习目标：
 * 1. 理解纹理内存的概念和优势
 * 2. 学会使用纹理对象 (Texture Objects)
 * 3. 掌握不同的寻址模式和过滤模式
 * 4. 了解纹理内存在图像处理中的应用
 *
 * 关键概念：
 * - 纹理内存是只读缓存，针对空间局部性优化
 * - 支持硬件插值、边界处理
 * - 特别适合图像处理和需要空间访问模式的应用
 * - 使用纹理对象（Texture Objects）是现代推荐方式
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// CUDA 13+ 兼容性：cudaTextureFilterMode 类型别名
#if CUDART_VERSION >= 13000
typedef cudaTextureFilterMode cudaFilterMode;
#endif

// ============================================================================
// 示例 1: 基本纹理对象使用
// ============================================================================

// 使用全局内存读取
__global__ void readGlobalMemory(float *input, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 全局内存读取
        output[tid] = input[tid] * 2.0f;
    }
}

// 使用纹理对象读取
__global__ void readTextureObject(cudaTextureObject_t texObj,
                                   float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 使用 tex1Dfetch 从纹理读取
        output[tid] = tex1Dfetch<float>(texObj, tid) * 2.0f;
    }
}

void demoBasicTexture() {
    printf("=== 示例 1: 基本纹理对象使用 ===\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);

    // 分配和初始化主机内存
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    // 分配设备内存
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // 创建纹理对象
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_input;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;  // 32 位浮点
    resDesc.res.linear.sizeInBytes = size;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 全局内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        readGlobalMemory<<<gridSize, blockSize>>>(d_input, d_output, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float globalTime;
    CHECK_CUDA(cudaEventElapsedTime(&globalTime, start, stop));
    printf("全局内存读取 (100次): %.3f ms\n", globalTime);

    // 纹理内存版本
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        readTextureObject<<<gridSize, blockSize>>>(texObj, d_output, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float textureTime;
    CHECK_CUDA(cudaEventElapsedTime(&textureTime, start, stop));
    printf("纹理内存读取 (100次): %.3f ms\n", textureTime);
    printf("纹理内存速度提升: %.2fx\n\n", globalTime / textureTime);

    // 验证结果
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("验证: output[0] = %.1f, output[100] = %.1f\n\n",
           h_output[0], h_output[100]);

    // 清理
    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

// ============================================================================
// 示例 2: 2D 纹理和寻址模式
// ============================================================================

// 2D 纹理读取核函数
__global__ void sample2DTexture(cudaTextureObject_t texObj,
                                 float *output,
                                 int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        // 使用归一化坐标 [0, 1)
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;

        // tex2D 支持硬件插值
        float value = tex2D<float>(texObj, u, v);
        output[y * width + x] = value;
    }
}

// 测试边界处理
__global__ void testBoundaryMode(cudaTextureObject_t texObj,
                                  float *output,
                                  int outWidth, int outHeight,
                                  int texWidth, int texHeight) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < outWidth && y < outHeight) {
        // 访问超出纹理边界的坐标
        float u = (float)(x - outWidth/4) / texWidth;
        float v = (float)(y - outHeight/4) / texHeight;

        output[y * outWidth + x] = tex2D<float>(texObj, u, v);
    }
}

void demo2DTexture() {
    printf("=== 示例 2: 2D 纹理和寻址模式 ===\n\n");

    const int WIDTH = 256;
    const int HEIGHT = 256;
    const int size = WIDTH * HEIGHT * sizeof(float);

    // 创建测试图像
    float *h_input = (float*)malloc(size);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            // 创建一个渐变图案
            h_input[y * WIDTH + x] = (float)(x + y) / (WIDTH + HEIGHT);
        }
    }

    // 分配 CUDA 数组（2D 纹理推荐使用）
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, h_input,
        WIDTH * sizeof(float), WIDTH * sizeof(float), HEIGHT,
        cudaMemcpyHostToDevice));

    // 测试不同的寻址模式
    const char *addressModes[] = {"Wrap", "Clamp", "Mirror", "Border"};
    cudaTextureAddressMode modes[] = {
        cudaAddressModeWrap,
        cudaAddressModeClamp,
        cudaAddressModeMirror,
        cudaAddressModeBorder
    };

    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, size));
    float *h_output = (float*)malloc(size);

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    for (int m = 0; m < 4; m++) {
        // 创建纹理描述符
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = modes[m];  // U 方向
        texDesc.addressMode[1] = modes[m];  // V 方向
        texDesc.filterMode = cudaFilterModeLinear;  // 线性插值
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;  // 使用归一化坐标

        cudaTextureObject_t texObj;
        CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        // 采样纹理
        sample2DTexture<<<gridDim, blockDim>>>(texObj, d_output, WIDTH, HEIGHT);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 获取角落的值来展示边界处理
        CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

        printf("寻址模式: %s\n", addressModes[m]);
        printf("  中心值 [128,128]: %.4f\n", h_output[128 * WIDTH + 128]);
        printf("  角落值 [0,0]: %.4f\n", h_output[0]);

        CHECK_CUDA(cudaDestroyTextureObject(texObj));
    }

    printf("\n寻址模式说明:\n");
    printf("  Wrap:   循环重复纹理\n");
    printf("  Clamp:  超出边界使用边界值\n");
    printf("  Mirror: 镜像反射\n");
    printf("  Border: 超出边界返回 0\n\n");

    // 清理
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

// ============================================================================
// 示例 3: 硬件插值
// ============================================================================

__global__ void upsampleNearest(cudaTextureObject_t texObj,
                                 float *output,
                                 int outWidth, int outHeight,
                                 int inWidth, int inHeight) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < outWidth && y < outHeight) {
        float u = (float)x / outWidth;
        float v = (float)y / outHeight;
        output[y * outWidth + x] = tex2D<float>(texObj, u, v);
    }
}

void demoInterpolation() {
    printf("=== 示例 3: 硬件插值 (图像放大) ===\n\n");

    const int IN_WIDTH = 8;
    const int IN_HEIGHT = 8;
    const int OUT_WIDTH = 32;
    const int OUT_HEIGHT = 32;

    // 创建小图像
    float h_input[IN_WIDTH * IN_HEIGHT];
    for (int y = 0; y < IN_HEIGHT; y++) {
        for (int x = 0; x < IN_WIDTH; x++) {
            // 棋盘图案
            h_input[y * IN_WIDTH + x] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;
        }
    }

    printf("原始图像 (%dx%d 棋盘图案):\n", IN_WIDTH, IN_HEIGHT);
    for (int y = 0; y < IN_HEIGHT; y++) {
        printf("  ");
        for (int x = 0; x < IN_WIDTH; x++) {
            printf("%.0f ", h_input[y * IN_WIDTH + x]);
        }
        printf("\n");
    }

    // 创建 CUDA 数组
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, IN_WIDTH, IN_HEIGHT));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, h_input,
        IN_WIDTH * sizeof(float), IN_WIDTH * sizeof(float), IN_HEIGHT,
        cudaMemcpyHostToDevice));

    // 输出缓冲区
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, OUT_WIDTH * OUT_HEIGHT * sizeof(float)));
    float *h_output = (float*)malloc(OUT_WIDTH * OUT_HEIGHT * sizeof(float));

    // 测试两种过滤模式
    cudaFilterMode filterModes[] = {cudaFilterModePoint, cudaFilterModeLinear};
    const char *filterNames[] = {"Point (最近邻)", "Linear (双线性)"};

    for (int f = 0; f < 2; f++) {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = filterModes[f];
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t texObj;
        CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        dim3 blockDim(16, 16);
        dim3 gridDim((OUT_WIDTH + 15) / 16, (OUT_HEIGHT + 15) / 16);

        upsampleNearest<<<gridDim, blockDim>>>(texObj, d_output,
            OUT_WIDTH, OUT_HEIGHT, IN_WIDTH, IN_HEIGHT);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_output, d_output,
            OUT_WIDTH * OUT_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost));

        printf("\n放大后图像 (%s, %dx%d):\n", filterNames[f], OUT_WIDTH, OUT_HEIGHT);
        printf("  显示左上角 8x8 区域:\n");
        for (int y = 0; y < 8; y++) {
            printf("  ");
            for (int x = 0; x < 8; x++) {
                float val = h_output[y * OUT_WIDTH + x];
                if (val > 0.7f) printf("█ ");
                else if (val > 0.3f) printf("▓ ");
                else printf("░ ");
            }
            printf("\n");
        }

        CHECK_CUDA(cudaDestroyTextureObject(texObj));
    }

    printf("\n过滤模式说明:\n");
    printf("  Point:  最近邻，保持锐利边缘但有锯齿\n");
    printf("  Linear: 双线性插值，平滑但可能模糊\n\n");

    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_output));
    free(h_output);
}

// ============================================================================
// 示例 4: 图像模糊（实际应用）
// ============================================================================

__global__ void gaussianBlur(cudaTextureObject_t texObj,
                              float *output,
                              int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        // 3x3 高斯核
        float kernel[3][3] = {
            {1.0f/16, 2.0f/16, 1.0f/16},
            {2.0f/16, 4.0f/16, 2.0f/16},
            {1.0f/16, 2.0f/16, 1.0f/16}
        };

        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                float u = (x + dx + 0.5f) / width;
                float v = (y + dy + 0.5f) / height;
                sum += tex2D<float>(texObj, u, v) * kernel[dy+1][dx+1];
            }
        }

        output[y * width + x] = sum;
    }
}

void demoImageBlur() {
    printf("=== 示例 4: 图像模糊（实际应用）===\n\n");

    const int WIDTH = 512;
    const int HEIGHT = 512;
    const int size = WIDTH * HEIGHT * sizeof(float);

    // 创建带有锐利边缘的测试图像
    float *h_input = (float*)malloc(size);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            // 创建一些矩形
            if ((x > 100 && x < 200 && y > 100 && y < 200) ||
                (x > 300 && x < 400 && y > 300 && y < 400)) {
                h_input[y * WIDTH + x] = 1.0f;
            } else {
                h_input[y * WIDTH + x] = 0.0f;
            }
        }
    }

    // 创建 CUDA 数组
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, h_input,
        WIDTH * sizeof(float), WIDTH * sizeof(float), HEIGHT,
        cudaMemcpyHostToDevice));

    // 创建纹理对象
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, size));
    float *h_output = (float*)malloc(size);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // 执行模糊
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        gaussianBlur<<<gridDim, blockDim>>>(texObj, d_output, WIDTH, HEIGHT);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, start, stop));
    printf("高斯模糊 %dx%d 图像 (100次): %.3f ms\n", WIDTH, HEIGHT, time);
    printf("每帧: %.3f ms (%.1f FPS)\n\n", time/100, 100000/time);

    // 显示边缘处的值变化
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    printf("边缘模糊效果 (y=150, x 从 95 到 105):\n");
    printf("  x:    ");
    for (int x = 95; x <= 105; x++) printf("%5d ", x);
    printf("\n  原始: ");
    for (int x = 95; x <= 105; x++) printf("%5.2f ", h_input[150 * WIDTH + x]);
    printf("\n  模糊: ");
    for (int x = 95; x <= 105; x++) printf("%5.2f ", h_output[150 * WIDTH + x]);
    printf("\n\n");

    // 清理
    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

// ============================================================================
// 示例 5: 分层纹理（Layered Texture）
// ============================================================================

__global__ void sampleLayeredTexture(cudaTextureObject_t texObj,
                                      float *output,
                                      int width, int height,
                                      int numLayers) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        // 对每一层进行采样并累加
        float sum = 0.0f;
        for (int layer = 0; layer < numLayers; layer++) {
            sum += tex2DLayered<float>(texObj, x + 0.5f, y + 0.5f, layer);
        }
        output[y * width + x] = sum / numLayers;  // 平均值
    }
}

void demoLayeredTexture() {
    printf("=== 示例 5: 分层纹理 ===\n\n");

    const int WIDTH = 256;
    const int HEIGHT = 256;
    const int NUM_LAYERS = 4;

    // 创建分层数据
    float *h_input = (float*)malloc(WIDTH * HEIGHT * NUM_LAYERS * sizeof(float));
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                // 每层有不同的值
                h_input[layer * WIDTH * HEIGHT + y * WIDTH + x] =
                    (float)(layer + 1) * 0.25f;
            }
        }
    }

    // 创建分层 CUDA 数组
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaExtent extent = make_cudaExtent(WIDTH, HEIGHT, NUM_LAYERS);
    cudaArray_t cuArray;
    CHECK_CUDA(cudaMalloc3DArray(&cuArray, &channelDesc, extent,
                                  cudaArrayLayered));

    // 复制数据到分层数组
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(h_input, WIDTH * sizeof(float),
                                            WIDTH, HEIGHT);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CHECK_CUDA(cudaMemcpy3D(&copyParams));

    // 创建分层纹理对象
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // 使用非归一化坐标

    cudaTextureObject_t texObj;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // 执行采样
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float)));

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    sampleLayeredTexture<<<gridDim, blockDim>>>(texObj, d_output,
        WIDTH, HEIGHT, NUM_LAYERS);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_output = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output,
        WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost));

    printf("分层纹理: %d 层，每层 %dx%d\n", NUM_LAYERS, WIDTH, HEIGHT);
    printf("每层值: Layer0=0.25, Layer1=0.50, Layer2=0.75, Layer3=1.00\n");
    printf("平均采样结果: %.4f (应为 0.625)\n\n", h_output[0]);

    printf("应用场景:\n");
    printf("  - 纹理数组/图集\n");
    printf("  - 立方体贴图\n");
    printf("  - 体积渲染的多切片\n\n");

    // 清理
    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 09: 纹理内存 (Texture Memory)                ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    // 检查设备
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("纹理对齐要求: %zu 字节\n", prop.textureAlignment);
    printf("1D 纹理最大宽度: %d\n", prop.maxTexture1D);
    printf("2D 纹理最大尺寸: %d x %d\n",
           prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("分层纹理最大层数: %d\n\n", prop.maxTexture2DLayered[2]);

    demoBasicTexture();
    demo2DTexture();
    demoInterpolation();
    demoImageBlur();
    demoLayeredTexture();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 纹理内存优势:\n");
    printf("   - 针对空间局部性优化的缓存\n");
    printf("   - 免费的硬件插值\n");
    printf("   - 自动边界处理\n");
    printf("   - 归一化坐标支持\n\n");

    printf("2. 纹理对象创建:\n");
    printf("   cudaResourceDesc  - 描述数据源\n");
    printf("   cudaTextureDesc   - 描述采样行为\n");
    printf("   cudaCreateTextureObject() - 创建对象\n\n");

    printf("3. 数据源类型:\n");
    printf("   - Linear: 线性内存 (1D)\n");
    printf("   - Array:  CUDA 数组 (2D/3D 推荐)\n");
    printf("   - Pitch2D: 带 pitch 的 2D 内存\n\n");

    printf("4. 寻址模式:\n");
    printf("   - Wrap:   循环\n");
    printf("   - Clamp:  钳制到边界\n");
    printf("   - Mirror: 镜像\n");
    printf("   - Border: 边界返回 0\n\n");

    printf("5. 过滤模式:\n");
    printf("   - Point:  最近邻 (快速，锐利)\n");
    printf("   - Linear: 双/三线性插值 (平滑)\n\n");

    printf("6. 纹理函数:\n");
    printf("   - tex1Dfetch<T>(obj, x)           - 1D 整数坐标\n");
    printf("   - tex2D<T>(obj, u, v)             - 2D 浮点坐标\n");
    printf("   - tex3D<T>(obj, u, v, w)          - 3D\n");
    printf("   - tex2DLayered<T>(obj, u, v, l)   - 分层 2D\n\n");

    printf("7. 适用场景:\n");
    printf("   - 图像处理（滤波、缩放）\n");
    printf("   - 计算机图形学\n");
    printf("   - 需要插值的查找表\n");
    printf("   - 空间局部性强的只读数据\n\n");

    return 0;
}
