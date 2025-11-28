/**
 * =============================================================================
 * CUDA 教程 34: Jetson 嵌入式与机器人应用
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 NVIDIA Jetson 平台特性
 * 2. 学习嵌入式 GPU 编程的优化策略
 * 3. 实现机器人感知和控制算法
 * 4. 掌握功耗和性能的平衡技巧
 *
 * 实现内容：
 * - 点云处理 (LiDAR)
 * - 深度图处理
 * - 路径规划 (A*)
 * - 传感器融合
 * - 实时控制循环
 *
 * 编译命令：
 *   nvcc 34_jetson_embedded.cu -o 34_jetson -O3
 *
 * 注意: 此教程适用于 Jetson Orin/Xavier/Nano 等平台
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
// 第一部分：Jetson 平台概述
// ============================================================================

void demoJetsonOverview() {
    printf("=== 第一部分：Jetson 平台概述 ===\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("当前设备信息:\n");
    printf("  设备名称: %s\n", prop.name);
    printf("  计算能力: %d.%d\n", prop.major, prop.minor);
    printf("  SM 数量: %d\n", prop.multiProcessorCount);
    printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  共享内存/块: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  最大线程/块: %d\n", prop.maxThreadsPerBlock);
    printf("  内存带宽: %.1f GB/s (理论值)\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    printf("NVIDIA Jetson 产品线:\n");
    printf("  ┌───────────────┬─────────────┬──────────┬────────────┐\n");
    printf("  │ 平台          │ GPU         │ 功耗     │ 应用场景   │\n");
    printf("  ├───────────────┼─────────────┼──────────┼────────────┤\n");
    printf("  │ Jetson Orin   │ Ampere      │ 15-60W   │ 自动驾驶   │\n");
    printf("  │ Jetson Xavier │ Volta       │ 10-30W   │ 机器人     │\n");
    printf("  │ Jetson Nano   │ Maxwell     │ 5-10W    │ 边缘AI     │\n");
    printf("  │ Jetson TX2    │ Pascal      │ 7.5-15W  │ 无人机     │\n");
    printf("  └───────────────┴─────────────┴──────────┴────────────┘\n\n");

    printf("Jetson 软件栈:\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  应用层: ROS 2, Isaac ROS, DeepStream                   │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │  加速库: cuDNN, TensorRT, VisionWorks, NPP              │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │  框架层: CUDA, OpenGL, Vulkan                           │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │  系统层: JetPack, L4T (Linux for Tegra)                 │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n\n");

    printf("嵌入式优化关键:\n");
    printf("  1. 内存管理: 统一内存减少拷贝\n");
    printf("  2. 功耗控制: 动态调整时钟频率\n");
    printf("  3. 实时性: 确定性延迟\n");
    printf("  4. 热管理: 避免过热降频\n\n");
}

// ============================================================================
// 第二部分：点云处理 (LiDAR)
// ============================================================================

struct Point3D {
    float x, y, z;
    float intensity;
};

// 点云下采样 (体素网格滤波)
__global__ void voxelGridFilterKernel(int *voxelIndices, const Point3D *points, int numPoints,
                                       float voxelSize, float minX, float minY, float minZ,
                                       int gridX, int gridY, int gridZ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numPoints) {
        Point3D p = points[tid];

        int vx = (int)((p.x - minX) / voxelSize);
        int vy = (int)((p.y - minY) / voxelSize);
        int vz = (int)((p.z - minZ) / voxelSize);

        // 边界检查
        vx = max(0, min(vx, gridX - 1));
        vy = max(0, min(vy, gridY - 1));
        vz = max(0, min(vz, gridZ - 1));

        voxelIndices[tid] = vz * gridX * gridY + vy * gridX + vx;
    }
}

// 点云变换 (旋转+平移)
__global__ void transformPointCloudKernel(Point3D *output, const Point3D *input, int numPoints,
                                           float r00, float r01, float r02,
                                           float r10, float r11, float r12,
                                           float r20, float r21, float r22,
                                           float tx, float ty, float tz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numPoints) {
        Point3D p = input[tid];

        // 旋转
        output[tid].x = r00 * p.x + r01 * p.y + r02 * p.z + tx;
        output[tid].y = r10 * p.x + r11 * p.y + r12 * p.z + ty;
        output[tid].z = r20 * p.x + r21 * p.y + r22 * p.z + tz;
        output[tid].intensity = p.intensity;
    }
}

// 点云法向量估计 (PCA 基于 KNN)
__global__ void estimateNormalsKernel(float *normals, const Point3D *points, int numPoints,
                                       const int *neighborIndices, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numPoints) {
        // 计算质心
        float cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < k; i++) {
            int idx = neighborIndices[tid * k + i];
            if (idx >= 0 && idx < numPoints) {
                cx += points[idx].x;
                cy += points[idx].y;
                cz += points[idx].z;
            }
        }
        cx /= k; cy /= k; cz /= k;

        // 计算协方差矩阵
        float cov[9] = {0};  // 3x3
        for (int i = 0; i < k; i++) {
            int idx = neighborIndices[tid * k + i];
            if (idx >= 0 && idx < numPoints) {
                float dx = points[idx].x - cx;
                float dy = points[idx].y - cy;
                float dz = points[idx].z - cz;
                cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
                cov[3] += dy * dx; cov[4] += dy * dy; cov[5] += dy * dz;
                cov[6] += dz * dx; cov[7] += dz * dy; cov[8] += dz * dz;
            }
        }

        // 简化: 使用叉积估计法向量 (适用于近似平面)
        // 实际应用中应使用 SVD 或特征值分解
        float v1x = points[neighborIndices[tid * k]].x - cx;
        float v1y = points[neighborIndices[tid * k]].y - cy;
        float v1z = points[neighborIndices[tid * k]].z - cz;
        float v2x = points[neighborIndices[tid * k + 1]].x - cx;
        float v2y = points[neighborIndices[tid * k + 1]].y - cy;
        float v2z = points[neighborIndices[tid * k + 1]].z - cz;

        float nx = v1y * v2z - v1z * v2y;
        float ny = v1z * v2x - v1x * v2z;
        float nz = v1x * v2y - v1y * v2x;

        float norm = sqrtf(nx * nx + ny * ny + nz * nz) + 1e-7f;
        normals[tid * 3] = nx / norm;
        normals[tid * 3 + 1] = ny / norm;
        normals[tid * 3 + 2] = nz / norm;
    }
}

// 地面分割 (RANSAC 简化版)
__global__ void groundSegmentationKernel(int *isGround, const Point3D *points, int numPoints,
                                          float heightThreshold, float normalThreshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numPoints) {
        Point3D p = points[tid];

        // 简单的高度阈值检测 (假设 z 是高度)
        // 实际应用中应使用 RANSAC 或更复杂的方法
        if (p.z < heightThreshold) {
            isGround[tid] = 1;
        } else {
            isGround[tid] = 0;
        }
    }
}

void demoPointCloudProcessing() {
    printf("=== 第二部分：点云处理 (LiDAR) ===\n\n");

    printf("点云处理应用:\n");
    printf("  - 自动驾驶: 障碍物检测、定位\n");
    printf("  - 机器人: 环境建图、导航\n");
    printf("  - 工业: 质量检测、逆向工程\n\n");

    const int numPoints = 100000;

    // 生成测试点云 (模拟 LiDAR 扫描)
    Point3D *h_points = (Point3D*)malloc(numPoints * sizeof(Point3D));
    for (int i = 0; i < numPoints; i++) {
        float angle = 2 * M_PI * i / numPoints;
        float r = 10.0f + 5.0f * sinf(angle * 10);
        h_points[i].x = r * cosf(angle);
        h_points[i].y = r * sinf(angle);
        h_points[i].z = 0.5f * sinf(angle * 5) + ((float)rand() / RAND_MAX - 0.5f);
        h_points[i].intensity = 50.0f + 50.0f * (float)rand() / RAND_MAX;
    }

    Point3D *d_points, *d_transformed;
    int *d_voxelIndices, *d_isGround;

    CHECK_CUDA(cudaMalloc(&d_points, numPoints * sizeof(Point3D)));
    CHECK_CUDA(cudaMalloc(&d_transformed, numPoints * sizeof(Point3D)));
    CHECK_CUDA(cudaMalloc(&d_voxelIndices, numPoints * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_isGround, numPoints * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_points, h_points, numPoints * sizeof(Point3D), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((numPoints + 255) / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    printf("点云处理性能 (%d 点):\n", numPoints);
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 操作                    │ 时间 (ms) │ 吞吐量          │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");

    const int NUM_RUNS = 100;
    float elapsed;

    // 点云变换
    float angle = M_PI / 6;  // 30度旋转
    float r00 = cosf(angle), r01 = -sinf(angle), r02 = 0;
    float r10 = sinf(angle), r11 = cosf(angle), r12 = 0;
    float r20 = 0, r21 = 0, r22 = 1;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        transformPointCloudKernel<<<grid, block>>>(d_transformed, d_points, numPoints,
                                                    r00, r01, r02, r10, r11, r12,
                                                    r20, r21, r22, 0, 0, 0);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 点云变换                │ %9.3f │ %6.2f M点/秒   │\n", elapsed, numPoints / elapsed / 1000.0f);

    // 体素滤波
    float voxelSize = 0.1f;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        voxelGridFilterKernel<<<grid, block>>>(d_voxelIndices, d_points, numPoints,
                                                voxelSize, -20.0f, -20.0f, -5.0f,
                                                400, 400, 100);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 体素滤波                │ %9.3f │ %6.2f M点/秒   │\n", elapsed, numPoints / elapsed / 1000.0f);

    // 地面分割
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        groundSegmentationKernel<<<grid, block>>>(d_isGround, d_points, numPoints, 0.3f, 0.9f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 地面分割                │ %9.3f │ %6.2f M点/秒   │\n", elapsed, numPoints / elapsed / 1000.0f);

    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");

    // 统计地面点数
    int *h_isGround = (int*)malloc(numPoints * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_isGround, d_isGround, numPoints * sizeof(int), cudaMemcpyDeviceToHost));
    int groundCount = 0;
    for (int i = 0; i < numPoints; i++) groundCount += h_isGround[i];
    printf("  地面点数: %d (%.1f%%)\n\n", groundCount, 100.0f * groundCount / numPoints);

    free(h_points);
    free(h_isGround);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_points));
    CHECK_CUDA(cudaFree(d_transformed));
    CHECK_CUDA(cudaFree(d_voxelIndices));
    CHECK_CUDA(cudaFree(d_isGround));
}

// ============================================================================
// 第三部分：深度图处理
// ============================================================================

// 深度图转点云
__global__ void depthToPointCloudKernel(Point3D *points, const float *depth,
                                         int width, int height,
                                         float fx, float fy, float cx, float cy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float d = depth[idx];

        if (d > 0 && d < 10.0f) {  // 有效深度范围
            points[idx].x = (x - cx) * d / fx;
            points[idx].y = (y - cy) * d / fy;
            points[idx].z = d;
            points[idx].intensity = 1.0f;
        } else {
            points[idx].x = 0;
            points[idx].y = 0;
            points[idx].z = 0;
            points[idx].intensity = 0;
        }
    }
}

// 深度图滤波 (双边滤波)
__global__ void depthBilateralFilterKernel(float *output, const float *input,
                                            int width, int height,
                                            float sigmaSpace, float sigmaDepth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float centerDepth = input[idx];

        if (centerDepth <= 0) {
            output[idx] = 0;
            return;
        }

        float sumWeight = 0;
        float sumDepth = 0;
        int radius = 5;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float neighborDepth = input[ny * width + nx];

                    if (neighborDepth > 0) {
                        float spatialWeight = expf(-(dx * dx + dy * dy) / (2 * sigmaSpace * sigmaSpace));
                        float depthWeight = expf(-powf(neighborDepth - centerDepth, 2) / (2 * sigmaDepth * sigmaDepth));
                        float weight = spatialWeight * depthWeight;

                        sumWeight += weight;
                        sumDepth += weight * neighborDepth;
                    }
                }
            }
        }

        output[idx] = (sumWeight > 0) ? (sumDepth / sumWeight) : centerDepth;
    }
}

// 深度图空洞填充
__global__ void depthHoleFillingKernel(float *output, const float *input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float d = input[idx];

        if (d > 0) {
            output[idx] = d;
        } else {
            // 简单的最近邻插值
            float sum = 0;
            int count = 0;
            int searchRadius = 3;

            for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                for (int dx = -searchRadius; dx <= searchRadius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        float nd = input[ny * width + nx];
                        if (nd > 0) {
                            sum += nd;
                            count++;
                        }
                    }
                }
            }

            output[idx] = (count > 0) ? (sum / count) : 0;
        }
    }
}

void demoDepthProcessing() {
    printf("=== 第三部分：深度图处理 ===\n\n");

    printf("深度传感器类型:\n");
    printf("  - 结构光: Intel RealSense, Kinect v1\n");
    printf("  - ToF: Kinect v2, Sony DepthSense\n");
    printf("  - 双目: ZED, OAK-D\n");
    printf("  - LiDAR: Ouster, Velodyne\n\n");

    const int width = 640;
    const int height = 480;
    const int numPixels = width * height;

    // 生成测试深度图
    float *h_depth = (float*)malloc(numPixels * sizeof(float));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 模拟一个带噪声的平面
            float noise = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
            h_depth[y * width + x] = 2.0f + 0.001f * x + noise;

            // 随机添加空洞
            if (rand() % 100 < 5) {
                h_depth[y * width + x] = 0;
            }
        }
    }

    float *d_depth, *d_filtered;
    Point3D *d_points;

    CHECK_CUDA(cudaMalloc(&d_depth, numPixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filtered, numPixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_points, numPixels * sizeof(Point3D)));

    CHECK_CUDA(cudaMemcpy(d_depth, h_depth, numPixels * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 相机内参 (假设)
    float fx = 525.0f, fy = 525.0f;
    float cx = width / 2.0f, cy = height / 2.0f;

    printf("深度图处理性能 (%dx%d):\n", width, height);
    printf("  ┌─────────────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 操作                    │ 时间 (ms) │ 帧率 (fps)      │\n");
    printf("  ├─────────────────────────┼───────────┼─────────────────┤\n");

    const int NUM_RUNS = 100;
    float elapsed;

    // 深度转点云
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        depthToPointCloudKernel<<<grid, block>>>(d_points, d_depth, width, height, fx, fy, cx, cy);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 深度转点云              │ %9.3f │ %7.0f         │\n", elapsed, 1000.0f / elapsed);

    // 双边滤波
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        depthBilateralFilterKernel<<<grid, block>>>(d_filtered, d_depth, width, height, 3.0f, 0.1f);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 双边滤波                │ %9.3f │ %7.0f         │\n", elapsed, 1000.0f / elapsed);

    // 空洞填充
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_RUNS; i++) {
        depthHoleFillingKernel<<<grid, block>>>(d_filtered, d_depth, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= NUM_RUNS;
    printf("  │ 空洞填充                │ %9.3f │ %7.0f         │\n", elapsed, 1000.0f / elapsed);

    printf("  └─────────────────────────┴───────────┴─────────────────┘\n\n");

    free(h_depth);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_depth));
    CHECK_CUDA(cudaFree(d_filtered));
    CHECK_CUDA(cudaFree(d_points));
}

// ============================================================================
// 第四部分：路径规划
// ============================================================================

#define MAP_SIZE 256
#define INF_DIST 1e9f

// A* 路径规划的并行扩展
__global__ void aStarExpandKernel(float *gScore, float *fScore, int *cameFrom,
                                   const unsigned char *map, int *openSet,
                                   int width, int height,
                                   int goalX, int goalY) {
    extern __shared__ int sharedOpen[];

    int tid = threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 检查是否是当前最优节点
    if (openSet[idx] && gScore[idx] < INF_DIST) {
        // 扩展邻居 (4方向)
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};

        for (int d = 0; d < 4; d++) {
            int nx = x + dx[d];
            int ny = y + dy[d];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;

                // 检查是否可通行
                if (map[nidx] == 0) {  // 0 表示可通行
                    float tentativeG = gScore[idx] + 1.0f;

                    if (tentativeG < gScore[nidx]) {
                        // 原子更新 (简化版本)
                        float oldG = atomicExch((float*)(gScore + nidx), tentativeG);
                        if (tentativeG < oldG) {
                            cameFrom[nidx] = idx;

                            // 启发式: 曼哈顿距离
                            float h = fabsf((float)(nx - goalX)) + fabsf((float)(ny - goalY));
                            fScore[nidx] = tentativeG + h;
                            openSet[nidx] = 1;
                        }
                    }
                }
            }
        }
    }
}

// 并行 Dijkstra (类似波前扩展)
__global__ void wavefrontExpansionKernel(float *distances, const unsigned char *map,
                                          int *changed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (map[idx] != 0) return;  // 障碍物

    float currentDist = distances[idx];
    if (currentDist >= INF_DIST - 1) return;  // 未访问

    // 检查邻居
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int d = 0; d < 4; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int nidx = ny * width + nx;

            if (map[nidx] == 0) {  // 可通行
                float newDist = currentDist + 1.0f;

                // 原子最小更新
                float oldDist = atomicMin((int*)(distances + nidx), __float_as_int(newDist));
                if (__int_as_float(oldDist) > newDist) {
                    *changed = 1;
                }
            }
        }
    }
}

void demoPathPlanning() {
    printf("=== 第四部分：路径规划 ===\n\n");

    printf("路径规划算法:\n");
    printf("  - A*: 启发式搜索，适合单次查询\n");
    printf("  - Dijkstra: 最短路径，适合多目标\n");
    printf("  - RRT/RRT*: 采样式规划，高维空间\n");
    printf("  - 势场法: 实时避障\n\n");

    const int mapSize = MAP_SIZE;

    // 生成测试地图
    unsigned char *h_map = (unsigned char*)malloc(mapSize * mapSize);
    memset(h_map, 0, mapSize * mapSize);

    // 添加随机障碍物
    for (int i = 0; i < mapSize * mapSize * 0.2; i++) {
        int x = rand() % mapSize;
        int y = rand() % mapSize;
        h_map[y * mapSize + x] = 1;  // 障碍物
    }

    // 确保起点和终点可通行
    h_map[0] = 0;
    h_map[mapSize * mapSize - 1] = 0;

    unsigned char *d_map;
    float *d_distances;
    int *d_changed;

    CHECK_CUDA(cudaMalloc(&d_map, mapSize * mapSize));
    CHECK_CUDA(cudaMalloc(&d_distances, mapSize * mapSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_changed, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_map, h_map, mapSize * mapSize, cudaMemcpyHostToDevice));

    // 初始化距离
    float *h_distances = (float*)malloc(mapSize * mapSize * sizeof(float));
    for (int i = 0; i < mapSize * mapSize; i++) {
        h_distances[i] = INF_DIST;
    }
    h_distances[0] = 0;  // 起点
    CHECK_CUDA(cudaMemcpy(d_distances, h_distances, mapSize * mapSize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((mapSize + 15) / 16, (mapSize + 15) / 16);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 波前扩展
    CHECK_CUDA(cudaEventRecord(start));

    int iterations = 0;
    int h_changed;
    do {
        h_changed = 0;
        CHECK_CUDA(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        wavefrontExpansionKernel<<<grid, block>>>(d_distances, d_map, d_changed, mapSize, mapSize);
        CHECK_CUDA(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        iterations++;
    } while (h_changed && iterations < mapSize * 2);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));

    // 获取终点距离
    CHECK_CUDA(cudaMemcpy(h_distances, d_distances, mapSize * mapSize * sizeof(float), cudaMemcpyDeviceToHost));
    float goalDist = h_distances[mapSize * mapSize - 1];

    printf("波前路径规划 (%dx%d 地图):\n", mapSize, mapSize);
    printf("  迭代次数: %d\n", iterations);
    printf("  计算时间: %.3f ms\n", elapsed);
    printf("  路径长度: %.0f (如果 >= %.0f 则不可达)\n\n", goalDist, INF_DIST);

    free(h_map);
    free(h_distances);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_map));
    CHECK_CUDA(cudaFree(d_distances));
    CHECK_CUDA(cudaFree(d_changed));
}

// ============================================================================
// 第五部分：传感器融合
// ============================================================================

struct IMUData {
    float ax, ay, az;      // 加速度
    float gx, gy, gz;      // 陀螺仪
    float timestamp;
};

struct Pose {
    float x, y, z;         // 位置
    float qw, qx, qy, qz;  // 四元数姿态
    float vx, vy, vz;      // 速度
};

// 四元数乘法
__device__ void quaternionMultiply(float *qr, const float *qa, const float *qb) {
    qr[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3];
    qr[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2];
    qr[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1];
    qr[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0];
}

// 四元数归一化
__device__ void quaternionNormalize(float *q) {
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    q[0] /= norm; q[1] /= norm; q[2] /= norm; q[3] /= norm;
}

// IMU 积分 (并行处理多个 IMU 数据)
__global__ void imuIntegrationKernel(Pose *poses, const IMUData *imu,
                                      int numSamples, float dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numSamples) {
        Pose pose = poses[tid];
        IMUData data = imu[tid];

        // 更新四元数 (简化的陀螺仪积分)
        float dq[4];
        float halfDt = 0.5f * dt;
        dq[0] = 1.0f;
        dq[1] = data.gx * halfDt;
        dq[2] = data.gy * halfDt;
        dq[3] = data.gz * halfDt;

        float q[4] = {pose.qw, pose.qx, pose.qy, pose.qz};
        float qNew[4];
        quaternionMultiply(qNew, q, dq);
        quaternionNormalize(qNew);

        pose.qw = qNew[0]; pose.qx = qNew[1]; pose.qy = qNew[2]; pose.qz = qNew[3];

        // 旋转加速度到世界坐标系 (简化)
        float awx = data.ax;  // 简化: 假设小角度
        float awy = data.ay;
        float awz = data.az - 9.81f;  // 减去重力

        // 更新速度
        pose.vx += awx * dt;
        pose.vy += awy * dt;
        pose.vz += awz * dt;

        // 更新位置
        pose.x += pose.vx * dt;
        pose.y += pose.vy * dt;
        pose.z += pose.vz * dt;

        poses[tid] = pose;
    }
}

// 卡尔曼滤波更新 (简化版)
__global__ void kalmanUpdateKernel(Pose *poses, const float *measurements,
                                    float measurementNoise, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        Pose pose = poses[tid];

        // 简化的卡尔曼增益 (假设过程噪声和测量噪声已知)
        float K = 0.5f;  // 简化的卡尔曼增益

        // 更新位置 (假设测量是 x,y,z 位置)
        float mx = measurements[tid * 3];
        float my = measurements[tid * 3 + 1];
        float mz = measurements[tid * 3 + 2];

        pose.x = pose.x + K * (mx - pose.x);
        pose.y = pose.y + K * (my - pose.y);
        pose.z = pose.z + K * (mz - pose.z);

        poses[tid] = pose;
    }
}

void demoSensorFusion() {
    printf("=== 第五部分：传感器融合 ===\n\n");

    printf("常见传感器融合方案:\n");
    printf("  ┌───────────────────┬─────────────────────────────────────┐\n");
    printf("  │ 方案              │ 传感器组合                          │\n");
    printf("  ├───────────────────┼─────────────────────────────────────┤\n");
    printf("  │ VIO               │ 视觉 + IMU                          │\n");
    printf("  │ LIO               │ LiDAR + IMU                         │\n");
    printf("  │ GPS/INS           │ GPS + IMU                           │\n");
    printf("  │ 多传感器融合      │ 视觉 + LiDAR + IMU + GPS            │\n");
    printf("  └───────────────────┴─────────────────────────────────────┘\n\n");

    const int numSamples = 10000;
    const float dt = 0.01f;  // 100Hz

    // 生成测试 IMU 数据
    IMUData *h_imu = (IMUData*)malloc(numSamples * sizeof(IMUData));
    Pose *h_poses = (Pose*)malloc(numSamples * sizeof(Pose));

    for (int i = 0; i < numSamples; i++) {
        h_imu[i].ax = 0.1f * sinf(i * 0.01f);
        h_imu[i].ay = 0.1f * cosf(i * 0.01f);
        h_imu[i].az = 9.81f + 0.01f * sinf(i * 0.02f);
        h_imu[i].gx = 0.01f * sinf(i * 0.005f);
        h_imu[i].gy = 0.01f * cosf(i * 0.005f);
        h_imu[i].gz = 0.001f;
        h_imu[i].timestamp = i * dt;

        // 初始化位姿
        h_poses[i].x = 0; h_poses[i].y = 0; h_poses[i].z = 0;
        h_poses[i].qw = 1; h_poses[i].qx = 0; h_poses[i].qy = 0; h_poses[i].qz = 0;
        h_poses[i].vx = 0; h_poses[i].vy = 0; h_poses[i].vz = 0;
    }

    IMUData *d_imu;
    Pose *d_poses;

    CHECK_CUDA(cudaMalloc(&d_imu, numSamples * sizeof(IMUData)));
    CHECK_CUDA(cudaMalloc(&d_poses, numSamples * sizeof(Pose)));

    CHECK_CUDA(cudaMemcpy(d_imu, h_imu, numSamples * sizeof(IMUData), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_poses, h_poses, numSamples * sizeof(Pose), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((numSamples + 255) / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // IMU 积分
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        imuIntegrationKernel<<<grid, block>>>(d_poses, d_imu, numSamples, dt);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed /= 100;

    printf("IMU 积分性能 (%d 样本):\n", numSamples);
    printf("  计算时间: %.3f ms\n", elapsed);
    printf("  吞吐量: %.2f M样本/秒\n", numSamples / elapsed / 1000.0f);
    printf("  实时因子: %.1fx (vs 100Hz)\n\n", (1.0f / 0.01f) / (elapsed * 0.001f));

    free(h_imu);
    free(h_poses);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_imu));
    CHECK_CUDA(cudaFree(d_poses));
}

// ============================================================================
// 第六部分：功耗优化
// ============================================================================

void demoPowerOptimization() {
    printf("=== 第六部分：功耗优化策略 ===\n\n");

    printf("Jetson 功耗模式:\n");
    printf("  ┌───────────────────┬────────────┬───────────────────────┐\n");
    printf("  │ 模式              │ 功耗       │ 适用场景              │\n");
    printf("  ├───────────────────┼────────────┼───────────────────────┤\n");
    printf("  │ MAXN              │ 最大       │ 基准测试、高性能需求  │\n");
    printf("  │ 15W               │ 15W        │ 平衡性能与功耗        │\n");
    printf("  │ 10W               │ 10W        │ 移动机器人            │\n");
    printf("  │ 5W (Nano)         │ 5W         │ 电池供电设备          │\n");
    printf("  └───────────────────┴────────────┴───────────────────────┘\n\n");

    printf("功耗优化技巧:\n\n");

    printf("1. 时钟频率调整:\n");
    printf("   # 查看当前频率\n");
    printf("   sudo jetson_clocks --show\n\n");
    printf("   # 设置最大性能\n");
    printf("   sudo jetson_clocks\n\n");
    printf("   # 恢复动态调频\n");
    printf("   sudo jetson_clocks --restore\n\n");

    printf("2. 功耗模式切换:\n");
    printf("   # 查看可用模式\n");
    printf("   sudo nvpmodel -q\n\n");
    printf("   # 设置功耗模式\n");
    printf("   sudo nvpmodel -m <mode_id>\n\n");

    printf("3. 代码层面优化:\n");
    printf("   - 减少内核启动次数 (批处理)\n");
    printf("   - 使用 Unified Memory 减少显式拷贝\n");
    printf("   - 避免频繁的 GPU-CPU 同步\n");
    printf("   - 使用低精度 (FP16, INT8)\n\n");

    printf("4. 任务调度优化:\n");
    printf("   - 基于负载动态调整处理频率\n");
    printf("   - 空闲时降低 GPU 频率\n");
    printf("   - 使用 CUDA Streams 提高利用率\n\n");

    printf("5. 硬件加速器利用:\n");
    printf("   - DLA (Deep Learning Accelerator)\n");
    printf("   - PVA (Programmable Vision Accelerator)\n");
    printf("   - NVDEC/NVENC (视频编解码)\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 34: Jetson 嵌入式与机器人应用                        ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    demoJetsonOverview();
    demoPointCloudProcessing();
    demoDepthProcessing();
    demoPathPlanning();
    demoSensorFusion();
    demoPowerOptimization();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                       学习要点总结                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("机器人感知处理:\n");
    printf("  ✓ 点云处理: 变换、滤波、分割\n");
    printf("  ✓ 深度图: 转换、滤波、空洞填充\n");
    printf("  ✓ 传感器融合: IMU 积分、卡尔曼滤波\n\n");

    printf("路径规划:\n");
    printf("  ✓ 波前扩展 (并行 Dijkstra)\n");
    printf("  ✓ A* 启发式搜索\n");
    printf("  ✓ 势场法实时避障\n\n");

    printf("嵌入式优化:\n");
    printf("  ✓ Unified Memory 简化内存管理\n");
    printf("  ✓ 功耗模式选择\n");
    printf("  ✓ 低精度推理 (TensorRT)\n");
    printf("  ✓ 硬件加速器利用\n\n");

    printf("推荐工具:\n");
    printf("  - Isaac ROS: 机器人应用框架\n");
    printf("  - TensorRT: 推理优化\n");
    printf("  - DeepStream: 视频分析\n");
    printf("  - VPI: 视觉处理\n");
    printf("  - jtop: 系统监控\n\n");

    return 0;
}
