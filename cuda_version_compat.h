/**
 * =============================================================================
 * CUDA 版本兼容性头文件
 * =============================================================================
 *
 * 用途：统一管理所有教程中的CUDA版本兼容性问题
 *
 * 使用方法：
 *   #include "cuda_version_compat.h"
 *
 * 提供的功能：
 *   - CUDA版本宏定义
 *   - 已弃用API的兼容性处理
 *   - 运行时特性检测辅助函数
 */

#ifndef CUDA_VERSION_COMPAT_H
#define CUDA_VERSION_COMPAT_H

#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================================
// CUDA 版本宏定义
// ============================================================================

// CUDA 主要版本检测
#if CUDART_VERSION >= 14000
#define CUDA_14_PLUS
#define CUDA_13_PLUS
#define CUDA_12_PLUS
#define CUDA_11_PLUS
#elif CUDART_VERSION >= 13000
#define CUDA_13_PLUS
#define CUDA_12_PLUS
#define CUDA_11_PLUS
#elif CUDART_VERSION >= 12000
#define CUDA_12_PLUS
#define CUDA_11_PLUS
#elif CUDART_VERSION >= 11000
#define CUDA_11_PLUS
#endif

// ============================================================================
// 已弃用 API 的兼容性宏
// ============================================================================

/**
 * memoryClockRate 和 clockRate 在 CUDA 12+ 已弃用
 * 提供带宽估算的兼容性宏
 */
#ifdef CUDA_12_PLUS
    // CUDA 12+ 使用估算方法
    #define GET_MEMORY_BANDWIDTH_GBPS(prop) \
        ((prop).memoryBusWidth * 20.0f / 8.0f)  // 假设 20 Gbps/pin

    #define GET_CLOCK_RATE_MHZ(prop) \
        0  // 不再提供，返回 0
#else
    // CUDA 11 及以下使用实际值
    #define GET_MEMORY_BANDWIDTH_GBPS(prop) \
        (2.0f * (prop).memoryClockRate * ((prop).memoryBusWidth / 8) / 1e6)

    #define GET_CLOCK_RATE_MHZ(prop) \
        ((prop).clockRate / 1000)
#endif

/**
 * cudaGraphGetEdges API 在 CUDA 13+ 变更
 * CUDA 13 添加了 edgeData 参数
 */
#ifdef CUDA_13_PLUS
    #define GRAPH_GET_EDGES(graph, from, to, numEdges) \
        cudaGraphGetEdges(graph, from, to, NULL, numEdges)
#else
    #define GRAPH_GET_EDGES(graph, from, to, numEdges) \
        cudaGraphGetEdges(graph, from, to, numEdges)
#endif

/**
 * Texture 滤波模式类型别名（CUDA 13+ 兼容性）
 */
#ifdef CUDA_13_PLUS
    typedef enum cudaTextureFilterMode cudaTextureFilterMode_compat;
#else
    typedef cudaTextureFilterMode cudaTextureFilterMode_compat;
#endif

// ============================================================================
// 运行时特性检测辅助函数
// ============================================================================

/**
 * 检测设备是否支持统一寻址
 */
static inline bool checkUnifiedAddressing(int device) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return false;
    return prop.unifiedAddressing != 0;
}

/**
 * 检测设备是否支持 Memory Pools (CUDA 11.2+)
 */
static inline bool checkMemoryPoolsSupport(int device) {
#ifdef CUDA_11_PLUS
    int supported = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supported,
        cudaDevAttrMemoryPoolsSupported, device);
    if (err != cudaSuccess) return false;
    return supported != 0;
#else
    (void)device;  // 避免未使用警告
    return false;
#endif
}

/**
 * 检测设备是否支持协作组网格同步
 */
static inline bool checkCooperativeLaunchSupport(int device) {
    int supported = 0;
    cudaError_t err = cudaDeviceGetAttribute(&supported,
        cudaDevAttrCooperativeLaunch, device);
    if (err != cudaSuccess) return false;
    return supported != 0;
}

/**
 * 检测设备计算能力
 */
static inline bool checkComputeCapability(int device, int major, int minor) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return false;
    return (prop.major > major) || (prop.major == major && prop.minor >= minor);
}

/**
 * 打印 CUDA 运行时版本信息
 */
static inline void printCUDAVersionInfo() {
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);

    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);

    printf("CUDA 版本信息:\n");
    printf("  Runtime 版本: %d.%d\n",
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  Driver 版本: %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10);

#ifdef CUDA_14_PLUS
    printf("  编译版本: CUDA 14+\n");
#elif defined(CUDA_13_PLUS)
    printf("  编译版本: CUDA 13+\n");
#elif defined(CUDA_12_PLUS)
    printf("  编译版本: CUDA 12+\n");
#elif defined(CUDA_11_PLUS)
    printf("  编译版本: CUDA 11+\n");
#else
    printf("  编译版本: CUDA 10 或更早\n");
#endif
    printf("\n");
}

// ============================================================================
// 库版本检测宏
// ============================================================================

// cuDNN 版本检测
#ifdef CUDNN_VERSION
    #if CUDNN_VERSION >= 9000
        #define CUDNN_9_PLUS
        #define CUDNN_8_PLUS
    #elif CUDNN_VERSION >= 8000
        #define CUDNN_8_PLUS
    #endif
#endif

// cuBLAS 版本检测（通过 CUDA 版本推断）
#ifdef CUDA_12_PLUS
    #define CUBLAS_12_PLUS
    #define CUBLAS_11_PLUS
#elif defined(CUDA_11_PLUS)
    #define CUBLAS_11_PLUS
#endif

// cuFFT 版本检测
#ifdef CUFFT_VERSION
    #if CUFFT_VERSION >= 11000
        #define CUFFT_11_PLUS
    #endif
#endif

// cuSPARSE 版本检测
#ifdef CUSPARSE_VERSION
    #if CUSPARSE_VERSION >= 12000
        #define CUSPARSE_12_PLUS
    #endif
#endif

// ============================================================================
// 性能优化建议宏
// ============================================================================

/**
 * 建议的线程块大小范围
 */
#define RECOMMENDED_MIN_BLOCK_SIZE 128
#define RECOMMENDED_MAX_BLOCK_SIZE 512
#define RECOMMENDED_DEFAULT_BLOCK_SIZE 256

/**
 * Warp 大小（架构常量）
 */
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

/**
 * 最大网格维度检查（动态获取）
 */
static inline void getMaxGridDims(int device, int *x, int *y, int *z) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    *x = prop.maxGridSize[0];
    *y = prop.maxGridSize[1];
    *z = prop.maxGridSize[2];
}

// ============================================================================
// 错误检查增强宏
// ============================================================================

/**
 * 带版本信息的错误检查
 */
#define CHECK_CUDA_VERSION_COMPAT(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        printf("CUDA Runtime 版本: %d\n", CUDART_VERSION); \
        exit(1); \
    } \
}

// ============================================================================
// 架构特定特性检测
// ============================================================================

/**
 * 检测是否支持 Tensor Cores
 */
static inline bool checkTensorCoreSupport(int device) {
    // Volta (sm_70) 及以上支持 Tensor Cores
    return checkComputeCapability(device, 7, 0);
}

/**
 * 检测是否支持异步拷贝（cp.async）
 */
static inline bool checkAsyncCopySupport(int device) {
    // Ampere (sm_80) 及以上支持 cp.async
    return checkComputeCapability(device, 8, 0);
}

/**
 * 检测是否支持 TMA（Tensor Memory Accelerator）
 */
static inline bool checkTMASupport(int device) {
    // Hopper (sm_90) 及以上支持 TMA
    return checkComputeCapability(device, 9, 0);
}

// ============================================================================
// 编译时警告（可选）
// ============================================================================

#ifdef CUDA_VERSION_COMPAT_WARNINGS
    #ifdef CUDA_12_PLUS
        #warning "编译使用 CUDA 12+，某些旧 API 已弃用"
    #endif

    #ifdef CUDA_13_PLUS
        #warning "编译使用 CUDA 13+，注意 API 变更"
    #endif
#endif

#endif // CUDA_VERSION_COMPAT_H
