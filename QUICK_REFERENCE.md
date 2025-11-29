# CUDA 版本兼容性快速参考

## 🚀 快速开始

### 1. 在文件中使用

```cuda
#include <cuda_runtime.h>
#include "cuda_version_compat.h"  // 添加这一行
```

### 2. 常用宏

```cuda
// 替代 memoryClockRate（CUDA 12+ 已弃用）
float bandwidth = GET_MEMORY_BANDWIDTH_GBPS(prop);

// 替代 clockRate（CUDA 12+ 已弃用）
int clockMHz = GET_CLOCK_RATE_MHZ(prop);

// CUDA 13+ cudaGraphGetEdges API 兼容
GRAPH_GET_EDGES(graph, fromNodes, toNodes, &numEdges);
```

### 3. 运行时检测

```cuda
// 检测 Memory Pools 支持（CUDA 11.2+）
if (checkMemoryPoolsSupport(device)) {
    cudaMallocAsync(...);
} else {
    cudaMalloc(...);  // 降级
}

// 检测 Tensor Cores（sm_70+）
if (checkTensorCoreSupport(device)) {
    // 使用 WMMA API
}

// 检测 Async Copy（sm_80+）
if (checkAsyncCopySupport(device)) {
    // 使用 cp.async
}
```

### 4. 版本检测

```cuda
#ifdef CUDA_13_PLUS
    // CUDA 13+ 特定代码
#elif defined(CUDA_12_PLUS)
    // CUDA 12+ 特定代码
#else
    // 旧版本代码
#endif
```

## 📋 常见模式

### 打印设备信息（现代化）

**之前：**
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
#if CUDART_VERSION < 12000
    printf("带宽: %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
#else
    printf("总线宽度: %d bits\n", prop.memoryBusWidth);
#endif
```

**现在：**
```cuda
#include "cuda_version_compat.h"

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("带宽: %.0f GB/s\n", GET_MEMORY_BANDWIDTH_GBPS(prop));
```

### 特性检测模式

```cuda
void myAdvancedFeature() {
    int device = 0;

    // 检测计算能力
    if (!checkComputeCapability(device, 7, 0)) {
        printf("需要 sm_70+ (Volta)\n");
        return;
    }

    // 检测 Tensor Cores
    if (!checkTensorCoreSupport(device)) {
        printf("需要 Tensor Cores\n");
        return;
    }

    // 使用 Tensor Cores
    // ...
}
```

## 🔍 检测函数参考

| 函数 | 用途 | 最低要求 |
|------|------|----------|
| `checkUnifiedAddressing(dev)` | 统一虚拟寻址 | sm_20 |
| `checkMemoryPoolsSupport(dev)` | Memory Pools | CUDA 11.2 |
| `checkCooperativeLaunchSupport(dev)` | 协作组 | sm_60 |
| `checkTensorCoreSupport(dev)` | Tensor Cores | sm_70 |
| `checkAsyncCopySupport(dev)` | cp.async | sm_80 |
| `checkTMASupport(dev)` | TMA | sm_90 |

## 🛠️ 库版本检测

### cuDNN

```cuda
#ifdef CUDNN_9_PLUS
    // cuDNN 9+ 特性
#elif defined(CUDNN_8_PLUS)
    // cuDNN 8+ 特性
#endif
```

### cuBLAS

```cuda
#ifdef CUBLAS_12_PLUS
    // cuBLAS 12+ 特性
#endif
```

## 📊 版本宏

```cuda
CUDA_11_PLUS  // CUDA 11.0+
CUDA_12_PLUS  // CUDA 12.0+
CUDA_13_PLUS  // CUDA 13.0+
CUDA_14_PLUS  // CUDA 14.0+

CUDNN_8_PLUS  // cuDNN 8.0+
CUDNN_9_PLUS  // cuDNN 9.0+

CUBLAS_11_PLUS   // cuBLAS 11.0+
CUBLAS_12_PLUS   // cuBLAS 12.0+
```

## 💡 最佳实践

### ✅ 推荐

```cuda
// 1. 始终包含头文件
#include "cuda_version_compat.h"

// 2. 使用宏而非直接条件编译
float bw = GET_MEMORY_BANDWIDTH_GBPS(prop);  // 好

// 3. 检测后降级
if (!checkFeature()) {
    // 使用替代方案
}

// 4. 在 main() 打印版本
int main() {
    printCUDAVersionInfo();
    // ...
}
```

### ❌ 避免

```cuda
// 1. 避免直接使用已弃用 API
float bw = 2.0 * prop.memoryClockRate * ...;  // 不好

// 2. 避免重复条件编译
#if CUDART_VERSION < 12000
    // ...
#endif
// 使用宏代替

// 3. 避免假设特性存在
cudaMallocAsync(...);  // 不检测就使用
```

## 🔗 相关文件

- `cuda_version_compat.h` - 兼容性头文件
- `VERSION_COMPAT_GUIDE.md` - 详细指南
- `IMPLEMENTATION_SUMMARY.md` - 实施总结

## 📞 帮助

遇到问题？查看：
1. `VERSION_COMPAT_GUIDE.md` 的示例
2. 已更新的文件（11, 12, 14, 34）
3. 头文件注释

---

**快速示例：完整模板**

```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_version_compat.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

int main() {
    printf("=== CUDA 程序 ===\n\n");

    // 打印版本信息
    printCUDAVersionInfo();

    // 获取设备信息（使用兼容性宏）
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("内存带宽: %.0f GB/s\n", GET_MEMORY_BANDWIDTH_GBPS(prop));

    // 特性检测
    if (checkTensorCoreSupport(0)) {
        printf("支持 Tensor Cores\n");
    }

    if (checkMemoryPoolsSupport(0)) {
        printf("支持 Memory Pools\n");
    }

    return 0;
}
```

**编译：**
```bash
nvcc -I. my_program.cu -o my_program
```

---

✅ 简单 | 🚀 高效 | 🛡️ 可靠
