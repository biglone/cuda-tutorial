# CUDA 版本兼容性升级指南

本指南说明如何将新的版本兼容性系统集成到现有教程文件中。

## 已创建的文件

### cuda_version_compat.h
统一的版本兼容性头文件，提供：
- CUDA 11-14+ 版本检测宏
- 已弃用 API 的兼容性包装
- 运行时特性检测函数
- 库版本检测宏（cuDNN, cuBLAS, cuFFT, cuSPARSE）

## 如何在现有文件中使用

### 1. 基础用法（所有文件）

在文件头部添加：

```cuda
#include "cuda_version_compat.h"

// 替换原有的 CHECK_CUDA 宏为兼容版本（可选）
// 或保留原有宏，两者可以共存
```

### 2. 替换已弃用的 API 使用

#### 示例 1: memoryClockRate（CUDA 12+ 已弃用）

**旧代码：**
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
#if CUDART_VERSION < 12000
    float bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
#else
    float bandwidth = prop.memoryBusWidth * 20.0f / 8;  // 估算
#endif
printf("内存带宽: %.0f GB/s\n", bandwidth);
```

**新代码：**
```cuda
#include "cuda_version_compat.h"

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
float bandwidth = GET_MEMORY_BANDWIDTH_GBPS(prop);
printf("内存带宽: %.0f GB/s\n", bandwidth);
```

#### 示例 2: cudaGraphGetEdges（CUDA 13+ API 变更）

**旧代码：**
```cuda
#if CUDART_VERSION >= 13000
#define GRAPH_GET_EDGES(graph, from, to, numEdges) \
    cudaGraphGetEdges(graph, from, to, NULL, numEdges)
#else
#define GRAPH_GET_EDGES(graph, from, to, numEdges) \
    cudaGraphGetEdges(graph, from, to, numEdges)
#endif
```

**新代码：**
```cuda
#include "cuda_version_compat.h"

// 直接使用头文件中的宏
GRAPH_GET_EDGES(graph, fromNodes, toNodes, &numEdges);
```

### 3. 添加运行时特性检测

#### 示例：Memory Pools 支持检测

**新增代码：**
```cuda
#include "cuda_version_compat.h"

void demoMemoryPools() {
    int device = 0;

    if (!checkMemoryPoolsSupport(device)) {
        printf("警告: 设备不支持 Memory Pools (需要 CUDA 11.2+)\n");
        printf("      将使用传统的 cudaMalloc/cudaFree\n\n");
        // 降级到传统方法
        return;
    }

    // 使用 Memory Pools
    cudaMallocAsync(...);
}
```

### 4. 库版本检测

#### 示例：cuDNN 版本特定功能

**16_cudnn_deeplearning.cu 增强：**
```cuda
#include "cuda_version_compat.h"
#include <cudnn.h>

void demoCuDNNVersionCheck() {
    printf("=== cuDNN 版本检查 ===\n\n");

    size_t version = cudnnGetVersion();
    printf("cuDNN 版本: %zu.%zu.%zu\n",
           version / 1000, (version % 1000) / 100, version % 100);

#ifdef CUDNN_9_PLUS
    printf("检测到 cuDNN 9+，可使用最新特性\n");
    // 使用 cuDNN 9 新特性
#elif defined(CUDNN_8_PLUS)
    printf("检测到 cuDNN 8+\n");
    // 使用 cuDNN 8 特性
#else
    printf("警告: cuDNN 版本较旧，某些示例可能不可用\n");
#endif
    printf("\n");
}
```

#### 示例：cuFFT 版本检测

**17_cufft.cu 增强：**
```cuda
#include "cuda_version_compat.h"
#include <cufft.h>

void demoCuFFTVersionCheck() {
    printf("=== cuFFT 版本检查 ===\n\n");

    int version;
    cufftGetVersion(&version);
    printf("cuFFT 版本: %d.%d.%d\n",
           version / 1000, (version % 100) / 10, version % 10);

#ifdef CUFFT_11_PLUS
    printf("检测到 cuFFT 11+，支持最新回调特性\n");
#endif
    printf("\n");
}
```

### 5. 打印版本信息（推荐在 main 函数开始处）

**在所有教程文件的 main() 函数中添加：**
```cuda
int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 XX: ...                                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    // 添加版本信息打印
    printCUDAVersionInfo();

    // 原有代码...
}
```

## 需要更新的文件列表

### 高优先级（使用了已弃用 API）

1. ✅ **11_matrix_multiply.cu** - 已有条件编译，可简化为使用宏
2. ✅ **12_profiling_debug.cu** - 已有条件编译，可简化为使用宏
3. ✅ **14_multi_gpu.cu** - 已有条件编译，可简化为使用宏
4. ⚠️ **34_jetson_embedded.cu** - 仅注释说明，需添加代码级防护

### 中优先级（库 API 文件）

5. **16_cudnn_deeplearning.cu** - 添加 cuDNN 版本检测
6. **17_cufft.cu** - 添加 cuFFT 版本检测
7. **18_cusparse.cu** - 添加 cuSPARSE 版本检测
8. **19_curand.cu** - 添加 cuRAND 版本检测

### 低优先级（增强）

9. **20_cuda_graphs.cu** - 已有完整防护，可迁移到使用头文件
10. 其他所有文件 - 添加 `printCUDAVersionInfo()` 调用

## 具体修改步骤

### 步骤 1: 简化现有条件编译

将现有的：
```cuda
#if CUDART_VERSION < 12000
    printf("内存带宽: %.0f GB/s\n", 2.0 * prop.memoryClockRate * ...);
#else
    printf("内存总线宽度: %d bits\n", prop.memoryBusWidth);
#endif
```

替换为：
```cuda
#include "cuda_version_compat.h"

float bandwidth = GET_MEMORY_BANDWIDTH_GBPS(prop);
printf("内存带宽: %.0f GB/s\n", bandwidth);
```

### 步骤 2: 添加特性检测

在使用高级特性前添加检测：
```cuda
#include "cuda_version_compat.h"

void demoTensorCores() {
    int device = 0;
    if (!checkTensorCoreSupport(device)) {
        printf("警告: 设备不支持 Tensor Cores (需要 sm_70+)\n");
        return;
    }
    // 使用 Tensor Cores
}
```

### 步骤 3: 统一错误检查（可选）

可以将：
```cuda
#define CHECK_CUDA(call) { ... }
```

替换为或补充为：
```cuda
#include "cuda_version_compat.h"
// CHECK_CUDA_VERSION_COMPAT 包含版本信息
```

## 编译说明

无需修改编译命令，cuda_version_compat.h 会自动检测编译时的 CUDA 版本。

## 示例：完整的文件头模板

```cuda
/**
 * =============================================================================
 * CUDA 教程 XX: ...
 * =============================================================================
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"  // 新增

// 原有的 CHECK_CUDA 宏（保留或使用 CHECK_CUDA_VERSION_COMPAT）
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 其他头文件和代码...

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 XX: ...                                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    // 打印版本信息（推荐）
    printCUDAVersionInfo();

    // 原有代码...

    return 0;
}
```

## 后续工作

1. ✅ 创建 cuda_version_compat.h
2. 🔄 更新高优先级文件（11, 12, 14, 34）
3. 🔄 为库 API 文件添加版本检测（16-19）
4. ⏳ 更新其他文件添加版本信息打印
5. ⏳ 创建测试脚本验证兼容性

## 注意事项

- cuda_version_compat.h 使用 `static inline` 函数，不会增加代码大小
- 所有宏都有前缀或在头文件内，不会与现有代码冲突
- 向后兼容 CUDA 10.0+，不影响旧版本编译
- 头文件仅依赖 cuda_runtime.h，无额外依赖

---

**更新日期:** 2025-11-29
**适用 CUDA 版本:** 10.0 - 14.0+
