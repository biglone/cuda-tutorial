# CUDA 编程完整学习指南

> 本指南将带你从零开始学习 CUDA GPU 编程，包含详细的概念解释、代码示例和实践建议。

## 目录

1. [CUDA 基础概念](#1-cuda-基础概念)
2. [第一个程序：Hello CUDA](#2-第一个程序hello-cuda)
3. [CUDA 核函数深入](#3-cuda-核函数深入)
4. [CUDA 内存管理](#4-cuda-内存管理)
5. [完整项目：向量加法](#5-完整项目向量加法)
6. [常见问题与技巧](#6-常见问题与技巧)
7. [进阶学习路线](#7-进阶学习路线)

---

## 1. CUDA 基础概念

### 1.1 什么是 CUDA？

CUDA (Compute Unified Device Architecture) 是 NVIDIA 推出的并行计算平台和编程模型。它允许你使用 C/C++ 编写在 GPU 上运行的程序。

**为什么要用 GPU？**
- **CPU**: 少量核心（4-16核），每个核心很强大，适合串行任务
- **GPU**: 大量核心（数千个），每个核心简单，适合大规模并行任务

### 1.2 核心概念

#### 🔹 主机（Host）vs 设备（Device）
- **主机（Host）**: CPU 及其内存
- **设备（Device）**: GPU 及其内存

#### 🔹 线程组织结构

CUDA 使用层次化的线程组织：

```
Grid（网格）
  └─ Block（线程块）
       └─ Thread（线程）
```

- **Thread（线程）**: 最小执行单元，执行核函数的代码
- **Block（线程块）**: 一组线程，可以共享内存和同步
- **Grid（网格）**: 所有线程块的集合

**示例**：如果有 2 个块，每块 4 个线程，总共就有 8 个线程在并行执行。

#### 🔹 线程索引

每个线程需要知道自己是谁，通过内置变量获取：

- `threadIdx.x/y/z`: 线程在块内的索引
- `blockIdx.x/y/z`: 块在网格中的索引
- `blockDim.x/y/z`: 块的大小（每块有多少线程）
- `gridDim.x/y/z`: 网格的大小（有多少块）

**计算全局线程 ID**（一维情况）：
```cuda
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

#### 🔹 函数类型限定符

- `__global__`: 核函数，在 GPU 上执行，从 CPU 调用
- `__device__`: 设备函数，在 GPU 上执行，从 GPU 调用
- `__host__`: 主机函数，在 CPU 上执行（默认，可省略）

---

## 2. 第一个程序：Hello CUDA

### 2.1 代码文件：`hello_cuda.cu`

这个程序演示了 CUDA 的基本结构：让多个 GPU 线程并行打印消息。

### 2.2 完整代码解析

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

// ============================================
// 核函数定义
// ============================================
__global__ void helloFromGPU() {
    // 计算当前线程的全局 ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU! 线程 ID: %d\n", tid);
}

int main() {
    // ============================================
    // 1. 配置线程组织
    // ============================================
    int numBlocks = 2;           // 启动 2 个线程块
    int threadsPerBlock = 4;     // 每个块有 4 个线程
    // 总共会有 2 × 4 = 8 个线程并行执行

    // ============================================
    // 2. 启动核函数
    // ============================================
    // 语法：kernelName<<<blocks, threads>>>(参数);
    helloFromGPU<<<numBlocks, threadsPerBlock>>>();

    // ============================================
    // 3. 同步等待 GPU 完成
    // ============================================
    cudaDeviceSynchronize();
    // 为什么需要？GPU 执行是异步的，CPU 不会等待
    // 这个函数强制 CPU 等待 GPU 完成所有操作

    // ============================================
    // 4. 错误检查
    // ============================================
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 错误: %s\n", cudaGetErrorString(error));
        return 1;
    }

    return 0;
}
```

### 2.3 编译和运行

```bash
# 编译
nvcc hello_cuda.cu -o hello_cuda

# 运行
./hello_cuda
```

### 2.4 预期输出

```
Hello from GPU! 线程 ID: 0
Hello from GPU! 线程 ID: 1
Hello from GPU! 线程 ID: 2
Hello from GPU! 线程 ID: 3
Hello from GPU! 线程 ID: 4
Hello from GPU! 线程 ID: 5
Hello from GPU! 线程 ID: 6
Hello from GPU! 线程 ID: 7
```

### 2.5 关键要点

✅ `__global__` 标记核函数
✅ `<<<blocks, threads>>>` 配置并行度
✅ `cudaDeviceSynchronize()` 等待 GPU 完成
✅ 总线程数 = blocks × threads

### 2.6 练习

1. 修改 `numBlocks` 和 `threadsPerBlock`，观察线程 ID 变化
2. 尝试让每个线程打印自己的 `threadIdx.x` 和 `blockIdx.x`
3. 思考：如果有 100 个任务，如何配置线程？

---

## 3. CUDA 核函数深入

### 3.1 代码文件：`kernel_basics.cu`

这个程序展示核函数的更多特性。

### 3.2 设备函数示例

```cuda
// 设备函数：只能在 GPU 上被调用
__device__ int square(int x) {
    return x * x;
}

__global__ void kernel1D() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int squared = square(tid);  // 调用设备函数
    printf("线程 %d: %d 的平方 = %d\n", tid, tid, squared);
}
```

**要点**：
- `__device__` 函数只能被 `__global__` 或其他 `__device__` 函数调用
- 不能从主机代码直接调用

### 3.3 二维线程配置

```cuda
__global__ void kernel2D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    printf("位置 (%d, %d)\n", x, y);
}

int main() {
    // 使用 dim3 定义 2D 配置
    dim3 blocks(2, 2);      // 2×2 = 4 个块
    dim3 threads(2, 2);     // 每块 2×2 = 4 个线程
    kernel2D<<<blocks, threads>>>();
    cudaDeviceSynchronize();
}
```

**什么时候用 2D/3D？**
- 图像处理：2D（宽×高）
- 矩阵运算：2D（行×列）
- 3D 模拟：3D（x×y×z）

### 3.4 传递参数

```cuda
__global__ void addValue(int *array, int value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        array[tid] += value;
    }
}
```

核函数可以接收：
- ✅ 基本类型（int, float, etc.）
- ✅ 指针（指向 GPU 内存）
- ✅ 结构体（按值传递，不要太大）
- ❌ 不能使用 C++ STL（vector, string 等）

### 3.5 练习

1. 编写一个核函数，计算每个线程 ID 的立方
2. 尝试创建一个 3×3 的线程网格
3. 修改代码，让核函数接收一个乘数参数

---

## 4. CUDA 内存管理

### 4.1 内存模型

```
CPU 内存（主机）          GPU 内存（设备）
    ↓                         ↓
[h_data]  ────复制───→   [d_data]
           cudaMemcpy
[h_result] ←──复制───    [d_result]
```

**关键规则**：
- 主机不能直接访问设备内存
- 设备不能直接访问主机内存
- 必须显式复制数据

### 4.2 内存管理步骤

#### 步骤 1：在主机分配内存

```cuda
int N = 1000;
int size = N * sizeof(float);
float *h_data = (float*)malloc(size);  // CPU 内存
```

#### 步骤 2：在设备分配内存

```cuda
float *d_data;
cudaMalloc((void**)&d_data, size);  // GPU 内存
```

**注意**：
- 参数是指针的指针 `&d_data`
- 返回值是错误码，需要检查

#### 步骤 3：主机 → 设备 复制

```cuda
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
```

参数：
1. 目标地址（GPU）
2. 源地址（CPU）
3. 大小（字节）
4. 复制方向

#### 步骤 4：执行核函数

```cuda
kernel<<<blocks, threads>>>(d_data, N);
cudaDeviceSynchronize();
```

#### 步骤 5：设备 → 主机 复制

```cuda
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
```

#### 步骤 6：释放内存

```cuda
cudaFree(d_data);    // 释放 GPU 内存
free(h_data);        // 释放 CPU 内存
```

### 4.3 错误检查模式

```cuda
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 使用方式
CHECK_CUDA(cudaMalloc(&d_data, size));
CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### 4.4 代码文件：`memory_management.cu`

查看完整示例，它演示了：
- 完整的内存分配、复制、释放流程
- 错误检查
- 数据验证

### 4.5 练习

1. 修改程序，将数组乘以 3 而不是 2
2. 增加数组大小到 1000，观察运行情况
3. 尝试故意写一个错误（如复制大小不匹配），看错误检查如何工作

---

## 5. 完整项目：向量加法

### 5.1 问题描述

计算：`C[i] = A[i] + B[i]`，对于 i = 0 到 N-1

### 5.2 CPU vs GPU 实现对比

#### CPU 版本（串行）

```cuda
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {  // 逐个计算
        c[i] = a[i] + b[i];
    }
}
```

时间复杂度：O(n)

#### GPU 版本（并行）

```cuda
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];  // 所有线程同时计算
    }
}
```

时间复杂度：O(1) 理论上，实际受硬件限制

### 5.3 为什么需要边界检查？

```cuda
if (tid < n) {
    c[tid] = a[tid] + b[tid];
}
```

**原因**：线程数往往不能整除数据量

例如：
- 数据量 N = 1000
- threadsPerBlock = 256
- blocksPerGrid = (1000 + 256 - 1) / 256 = 4
- 实际启动线程数 = 4 × 256 = 1024

有 24 个多余线程！如果不检查，会访问越界。

### 5.4 性能测量

#### CPU 计时

```cuda
#include <time.h>

clock_t start = clock();
vectorAddCPU(a, b, c, N);
clock_t end = clock();
double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
```

#### GPU 计时（使用 CUDA Event）

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... GPU 操作 ...
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float time_ms;
cudaEventElapsedTime(&time_ms, start, stop);
```

### 5.5 性能分析

运行 `vector_add.cu` 你可能看到 CPU 更快，为什么？

#### GPU 时间分解

```
总时间 = 数据传输时间 + 计算时间
       = (H→D 传输) + 核函数执行 + (D→H 传输)
```

**对于向量加法**：
- 计算简单（一次加法）
- 数据传输开销大
- 问题规模小（100万元素）

**GPU 的优势场景**：
1. 计算密集（如矩阵乘法）
2. 数据量大（数千万、数亿）
3. 可以重用数据（多次计算，少次传输）

### 5.6 结果验证

```cuda
bool verifyResults(float *cpu_result, float *gpu_result, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            return false;  // 不匹配
        }
    }
    return true;  // 所有元素匹配
}
```

**为什么用 `fabs` 和 `1e-5`？**
- 浮点数运算有精度误差
- 不能直接用 `==` 比较
- 允许小的误差范围

### 5.7 练习

1. 修改向量大小，测试不同规模下的性能
2. 修改为向量减法或乘法
3. 尝试不同的 `threadsPerBlock` 值（128, 256, 512），比较性能

---

## 6. 常见问题与技巧

### 6.1 编译错误

**问题**：`nvcc: command not found`
**解决**：CUDA toolkit 未安装或未加入 PATH

**问题**：`undefined reference to cudaMalloc`
**解决**：文件后缀必须是 `.cu` 不是 `.c` 或 `.cpp`

### 6.2 运行时错误

**问题**：程序崩溃无输出
**解决**：
```cuda
// 检查核函数错误
kernel<<<blocks, threads>>>();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel 错误: %s\n", cudaGetErrorString(err));
}
```

**问题**：`illegal memory access`
**解决**：检查数组越界、空指针、内存未分配

### 6.3 性能优化技巧

#### 选择合适的 threadsPerBlock

```cuda
// 推荐值：128, 256, 512
// 必须是 32 的倍数（warp size）
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```

#### 减少内存传输

```cuda
// ❌ 不好：多次传输
for (int i = 0; i < 100; i++) {
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    kernel<<<blocks, threads>>>(d_data);
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
}

// ✅ 好：只传输一次
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
for (int i = 0; i < 100; i++) {
    kernel<<<blocks, threads>>>(d_data);
}
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
```

### 6.4 调试技巧

#### 使用 printf 调试

```cuda
__global__ void debugKernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {  // 只让一个线程打印
        printf("调试信息\n");
    }
}
```

#### 检查 GPU 信息

```bash
nvidia-smi          # 查看 GPU 状态
nvcc --version      # 查看 CUDA 版本
```

---

## 7. 进阶学习路线

### 7.1 下一步学习内容

#### 级别 2：优化技术
1. **共享内存（Shared Memory）**
   - 块内线程共享的快速内存
   - 减少全局内存访问
   - 矩阵乘法优化

2. **内存合并（Memory Coalescing）**
   - 优化全局内存访问模式
   - 提高带宽利用率

3. **线程同步（Synchronization）**
   - `__syncthreads()`
   - 原子操作

#### 级别 3：高级特性
1. **CUDA Streams**
   - 并发执行多个核函数
   - 重叠计算和传输

2. **Unified Memory**
   - 自动管理 CPU/GPU 内存
   - `cudaMallocManaged()`

3. **动态并行**
   - GPU 上启动核函数

#### 级别 4：实际应用
1. **矩阵乘法**
2. **图像处理**（滤波、边缘检测）
3. **深度学习**（与 PyTorch/TensorFlow 集成）
4. **科学计算**（FFT, 随机数生成）

### 7.2 学习资源

#### 官方文档
- CUDA C Programming Guide
- CUDA Best Practices Guide
- CUDA API Reference

#### 推荐书籍
- 《CUDA by Example》（入门友好）
- 《Programming Massively Parallel Processors》（深入理解）

#### 在线资源
- NVIDIA Developer Blog
- CUDA Tutorial Series on YouTube
- GitHub CUDA Samples

### 7.3 实践项目建议

1. **图像模糊**：实现高斯模糊
2. **矩阵乘法**：朴素版本 → 优化版本
3. **N-body 模拟**：粒子系统
4. **蒙特卡洛**：π 值估算

---

## 8. 进阶主题：内存优化

### 8.1 共享内存

共享内存是块内线程共享的快速内存（约 100x 快于全局内存）。

```cuda
__global__ void sharedMemExample(float *input, float *output, int n) {
    // 声明共享内存
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    if (gid < n) {
        sdata[tid] = input[gid];
    }

    // 同步：确保所有线程完成加载
    __syncthreads();

    // 现在可以访问其他线程加载的数据
    // 例如：访问邻居
    if (tid > 0 && gid < n) {
        output[gid] = sdata[tid] + sdata[tid - 1];
    }
}
```

**Bank Conflict 避免**：
```cuda
// 有 bank conflict
__shared__ float tile[32][32];

// 无 bank conflict (padding)
__shared__ float tile[32][32 + 1];
```

### 8.2 内存合并访问

**好的访问模式**（连续地址）：
```cuda
// 每个线程访问连续地址
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float val = data[tid];  // 合并访问
```

**差的访问模式**（跨步访问）：
```cuda
// 间隔访问导致多次内存事务
float val = data[tid * stride];  // 非合并访问
```

### 8.3 CUDA Streams

Streams 允许并发执行操作：

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 异步操作可以重叠
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);

kernel<<<blocks, threads, 0, stream1>>>(d_a);
kernel<<<blocks, threads, 0, stream2>>>(d_b);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

---

## 9. 高级主题概览

### 9.1 协作组 (Cooperative Groups)

更灵活的线程同步机制：

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperativeKernel(float *data) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp 级别归约
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }
}
```

### 9.2 CUDA Graphs

预定义操作序列，减少启动开销：

```cuda
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStream_t stream;

// 捕获操作序列
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel1<<<...>>>(args);
kernel2<<<...>>>(args);
cudaStreamEndCapture(stream, &graph);

// 实例化
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 多次执行
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(graphExec, stream);
}
```

### 9.3 混合精度计算

使用 FP16 提高性能：

```cuda
#include <cuda_fp16.h>

__global__ void fp16Kernel(half *output, const half *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // half2 操作获得 2x 吞吐
        half2 a = *reinterpret_cast<const half2*>(&input[tid*2]);
        half2 b = __hadd2(a, a);
        *reinterpret_cast<half2*>(&output[tid*2]) = b;
    }
}
```

### 9.4 多 GPU 编程

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    // 在每个 GPU 上分配和计算
    cudaMalloc(&d_data[i], size);
    kernel<<<blocks, threads>>>(d_data[i]);
}
```

---

## 10. 深度学习相关

### 10.1 常用操作实现

**GELU 激活函数**：
```cuda
__global__ void gelu(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f *
                    (x + 0.044715f * x * x * x)));
        output[tid] = x * cdf;
    }
}
```

**LayerNorm**：
```cuda
// 简化版：每个 batch 一个 block
__global__ void layerNorm(float *output, const float *input,
                          const float *gamma, const float *beta,
                          int hidden_size, float eps) {
    extern __shared__ float sdata[];

    // 1. 计算均值
    // 2. 计算方差
    // 3. 归一化: (x - mean) / sqrt(var + eps) * gamma + beta
}
```

### 10.2 PyTorch 集成

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_op',
    ext_modules=[
        CUDAExtension('my_cuda_op', [
            'my_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

---

## 附录：CUDA 函数速查

### 内存管理
```cuda
cudaMalloc(void **ptr, size_t size)
cudaFree(void *ptr)
cudaMemcpy(void *dst, void *src, size_t size, cudaMemcpyKind kind)
cudaMemset(void *ptr, int value, size_t size)
```

### 设备管理
```cuda
cudaDeviceSynchronize()       // 等待 GPU 完成
cudaGetLastError()            // 获取最后的错误
cudaGetDeviceCount(int *count)
cudaSetDevice(int device)
```

### 事件管理
```cuda
cudaEventCreate(cudaEvent_t *event)
cudaEventRecord(cudaEvent_t event)
cudaEventSynchronize(cudaEvent_t event)
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop)
cudaEventDestroy(cudaEvent_t event)
```

### 核函数启动
```cuda
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);
// gridDim: 块的数量（可以是 int 或 dim3）
// blockDim: 每块的线程数（可以是 int 或 dim3）
// sharedMem: 动态共享内存大小（可选，默认 0）
// stream: CUDA stream（可选，默认 0）
```

---

## 结语

恭喜你完成了 CUDA 基础学习！

**记住的关键点**：
1. ✅ 理解主机/设备内存分离
2. ✅ 掌握线程组织和索引
3. ✅ 始终检查边界和错误
4. ✅ 性能不总是越快越好，要考虑开销
5. ✅ 从简单开始，逐步优化

**下一步**：选择一个感兴趣的项目开始实践！

有问题随时提问，祝学习愉快！🚀
