# CUDA 常见问题解答（FAQ）

## 目录

- [环境配置问题](#环境配置问题)
- [编译错误](#编译错误)
- [运行时错误](#运行时错误)
- [内存问题](#内存问题)
- [性能问题](#性能问题)
- [调试技巧](#调试技巧)

---

## 环境配置问题

### Q: 如何检查 CUDA 是否正确安装？

运行以下命令：

```bash
# 检查 nvcc 编译器
nvcc --version

# 检查 GPU 驱动和设备
nvidia-smi

# 检查 CUDA 库路径
echo $LD_LIBRARY_PATH | grep cuda
```

如果 `nvcc` 找不到，需要将 CUDA 添加到 PATH：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: nvidia-smi 显示 "NVIDIA-SMI has failed"

**可能原因：**
1. 驱动未安装或版本不匹配
2. 内核更新后驱动失效
3. Secure Boot 阻止加载驱动

**解决方案：**
```bash
# 检查驱动模块
lsmod | grep nvidia

# 重新安装驱动
sudo apt-get purge nvidia*
sudo apt-get install nvidia-driver-xxx  # 替换为合适版本
sudo reboot
```

### Q: CUDA 版本和驱动版本如何匹配？

| CUDA 版本 | 最低驱动版本 |
|-----------|-------------|
| CUDA 12.x | 525.60+ |
| CUDA 11.x | 450.80+ |
| CUDA 10.x | 410.48+ |

查看兼容性：[CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

---

## 编译错误

### Q: "nvcc: command not found"

CUDA 未添加到 PATH，添加环境变量：

```bash
# 添加到 ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc
```

### Q: "unsupported GNU version"

GCC 版本过高，CUDA 不支持。

**解决方案 1**：安装兼容版本的 GCC
```bash
sudo apt-get install gcc-9 g++-9
nvcc -ccbin g++-9 your_file.cu -o output
```

**解决方案 2**：使用 `--allow-unsupported-compiler`（不推荐用于生产）
```bash
nvcc --allow-unsupported-compiler your_file.cu -o output
```

### Q: "error: identifier 'xxx' is undefined"

**常见原因：**
1. 缺少头文件
2. 使用了 CPU 函数在 GPU 代码中

**示例错误**：在核函数中使用 `malloc`
```cuda
__global__ void kernel() {
    int* p = malloc(100);  // 错误！GPU 不能用 malloc
}
```

**修复**：使用设备端分配或预分配内存
```cuda
__global__ void kernel(int* pre_allocated) {
    // 使用预分配的内存
}
```

### Q: "undefined reference to cudaXxx"

链接时找不到 CUDA 库。

```bash
# 确保链接 cudart
nvcc your_file.cu -o output -lcudart

# 如果使用其他库，也要链接
nvcc your_file.cu -o output -lcublas -lcudnn
```

### Q: 编译时警告 "deprecated"

某些 API 已弃用。查看警告信息，使用新 API 替代：

| 旧 API | 新 API |
|--------|--------|
| `cudaThreadSynchronize()` | `cudaDeviceSynchronize()` |
| `cudaThreadExit()` | `cudaDeviceReset()` |
| `cudaBindTexture()` | 纹理对象 API |

---

## 运行时错误

### Q: "CUDA error: no kernel image is available for execution"

编译时的架构与运行时 GPU 不匹配。

**解决方案**：指定正确的计算能力
```bash
# 查看你的 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 编译时指定（例如 sm_75 对应计算能力 7.5）
nvcc -arch=sm_75 your_file.cu -o output

# 或者编译多个架构
nvcc -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_75,code=sm_75 \
     your_file.cu -o output
```

### Q: "CUDA error: invalid device function"

核函数调用了不支持的特性。

**常见原因**：
- 使用了高版本特性但编译目标是低版本
- 动态并行需要 `-rdc=true`

```bash
# 动态并行必须使用
nvcc -rdc=true -lcudadevrt your_file.cu -o output
```

### Q: "CUDA error: too many resources requested for launch"

线程块请求的资源超出限制。

**检查限制**：
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("每块最大线程数: %d\n", prop.maxThreadsPerBlock);
printf("每块最大共享内存: %zu\n", prop.sharedMemPerBlock);
printf("每块最大寄存器: %d\n", prop.regsPerBlock);
```

**解决方案**：减少线程数或共享内存使用
```cuda
// 从 1024 减少到 256
kernel<<<blocks, 256>>>();
```

### Q: "CUDA error: unspecified launch failure"

核函数执行时崩溃，通常是非法内存访问。

**调试方法**：
```bash
# 使用 compute-sanitizer
compute-sanitizer ./your_program
```

### Q: "CUDA error: device-side assert triggered"

设备端断言失败。

```bash
# 查看详细信息
CUDA_LAUNCH_BLOCKING=1 ./your_program
```

---

## 内存问题

### Q: "CUDA error: out of memory"

GPU 显存不足。

**解决方案**：
1. 减小数据规模
2. 分批处理
3. 使用统一内存（自动分页）
4. 及时释放不用的内存

```cuda
// 检查可用显存
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("可用: %zu MB, 总计: %zu MB\n",
       free_mem/1024/1024, total_mem/1024/1024);
```

### Q: 结果全是 0 或垃圾值

**常见原因**：
1. 忘记拷贝数据到 GPU
2. 忘记拷贝结果回 CPU
3. 核函数没有执行（配置错误）
4. 数组越界

**检查清单**：
```cuda
// 1. 确保 H→D 拷贝
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// 2. 确保核函数执行
kernel<<<blocks, threads>>>(d_data, N);
cudaDeviceSynchronize();  // 等待完成

// 3. 确保 D→H 拷贝
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

// 4. 检查每一步的错误
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

### Q: 内存泄漏如何检测？

```bash
# 使用 compute-sanitizer 的 memcheck 工具
compute-sanitizer --tool memcheck ./your_program

# 或使用 cuda-memcheck（旧版本）
cuda-memcheck ./your_program
```

### Q: cudaMalloc 和 cudaMallocManaged 怎么选？

| 场景 | 推荐 |
|------|------|
| 数据完全在 GPU 处理 | `cudaMalloc` |
| CPU/GPU 频繁交互 | `cudaMallocManaged` |
| 追求最高性能 | `cudaMalloc` + 手动管理 |
| 快速原型开发 | `cudaMallocManaged` |

---

## 性能问题

### Q: GPU 比 CPU 还慢？

**常见原因**：

1. **数据量太小**：GPU 有启动开销，小数据不划算
   ```
   建议：N > 10000 才考虑 GPU
   ```

2. **传输时间占主导**：计算量小但数据量大
   ```cuda
   // 优化：使用 pinned memory 加速传输
   cudaMallocHost(&h_data, size);  // 而不是 malloc
   ```

3. **核函数效率低**：
   - 分支发散
   - 非合并访存
   - 占用率低

### Q: 如何测量核函数执行时间？

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<blocks, threads>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("耗时: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Q: 什么是合并访存（Coalesced Access）？

线程束（32线程）同时访问连续内存地址时，可以合并为一次事务。

**好的访问模式**：
```cuda
// 连续访问 - 合并
data[tid] = value;          // tid = 0,1,2,3...
```

**差的访问模式**：
```cuda
// 跨步访问 - 不合并
data[tid * stride] = value;  // 跳跃访问
```

### Q: 什么是 Bank 冲突？

共享内存分为 32 个 bank。同一 warp 内的线程访问同一 bank 的不同地址会串行化。

**避免方法**：
```cuda
// 有冲突：所有线程访问 bank 0
__shared__ float s[32][32];
float val = s[threadIdx.x][0];

// 无冲突：每个线程访问不同 bank
float val = s[threadIdx.x][threadIdx.x];

// 技巧：padding 避免冲突
__shared__ float s[32][33];  // 33 而不是 32
```

### Q: 如何提高占用率（Occupancy）？

```cuda
// 1. 查询占用率
int blockSize = 256;
int minGridSize, gridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// 2. 减少每线程寄存器使用
nvcc -maxrregcount=32 your_file.cu -o output

// 3. 减少共享内存使用
// 4. 调整线程块大小（通常 128-512）
```

---

## 调试技巧

### Q: 如何启用同步调试？

```bash
# 设置环境变量，所有 CUDA 调用变为同步
export CUDA_LAUNCH_BLOCKING=1
./your_program
```

### Q: 如何打印调试信息？

```cuda
__global__ void kernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 只让第一个线程打印
    if (tid == 0) {
        printf("n = %d\n", n);
    }

    // 或者限制打印范围
    if (tid < 5) {
        printf("tid %d: data = %f\n", tid, data[tid]);
    }
}
```

### Q: 推荐的错误检查宏？

```cuda
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_LAST_ERROR() CHECK_CUDA(cudaGetLastError())

// 使用
CHECK_CUDA(cudaMalloc(&d_data, size));
kernel<<<blocks, threads>>>(d_data);
CHECK_LAST_ERROR();
```

### Q: 常用的性能分析工具？

| 工具 | 用途 |
|------|------|
| `nvprof`（旧） | 基础性能分析 |
| `nsys` (Nsight Systems) | 系统级时间线分析 |
| `ncu` (Nsight Compute) | 核函数详细分析 |
| `compute-sanitizer` | 内存错误检测 |

```bash
# Nsight Systems
nsys profile ./your_program
nsys-ui report.qdrep

# Nsight Compute
ncu ./your_program
ncu --set full -o report ./your_program
```

---

## 其他常见问题

### Q: 核函数可以返回值吗？

不能。核函数必须是 `void` 返回类型。通过输出参数返回结果：

```cuda
__global__ void kernel(int* output) {
    output[0] = 42;  // 通过指针返回
}
```

### Q: 核函数可以递归吗？

可以，但需要动态并行支持：

```bash
nvcc -rdc=true -lcudadevrt your_file.cu -o output
```

### Q: 如何在核函数中使用 C++ STL？

不能直接使用。GPU 代码不支持 STL。替代方案：
- 使用 Thrust 库（CUDA 的 STL 替代）
- 使用 CUB 库（高性能原语）

### Q: 多 GPU 编程的基本模式？

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaSetDevice(i);
    // 在设备 i 上分配内存和执行核函数
}
```

---

## 获取更多帮助

1. **官方文档**：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. **开发者论坛**：[NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/)
3. **Stack Overflow**：搜索 `[cuda]` 标签
4. **本教程示例**：查看对应主题的 `.cu` 文件
