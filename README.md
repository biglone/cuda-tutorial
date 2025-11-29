# CUDA 编程教程

从零开始学习 NVIDIA CUDA GPU 并行编程，包含 35 个循序渐进的教程示例。

## 快速开始

### 方式一：直接编译（适合学习单个教程）

```bash
# 编译第一个程序
nvcc hello_cuda.cu -o hello_cuda

# 运行
./hello_cuda
```

### 方式二：CMake 构建（推荐）

```bash
# 创建构建目录
mkdir build && cd build

# 配置（自动检测 GPU 架构）
cmake ..

# 编译全部教程
make -j$(nproc)

# 或编译单个教程
make hello_cuda

# 运行（可执行文件在 bin 目录）
./bin/hello_cuda
```

**按章节编译：**

```bash
make basics          # 基础篇 (01-04)
make advanced        # 进阶篇 (05-10)
make practical       # 实战篇 (11-15)
make libraries       # 库应用篇 (16-20)
make high_level      # 高级篇 (21-25)
make special_topics  # 专题篇 (26-30)
make frontier        # 前沿应用篇 (31-35)
make all_tutorials   # 全部教程
```

### 方式三：全量编译测试

```bash
# 使用测试脚本编译所有 35 个教程
./compile_all.sh

# 查看编译统计和错误信息
# 成功的二进制文件在 build_test/ 目录
```

## 环境要求

- NVIDIA GPU（计算能力 3.0+）
- CUDA Toolkit 10.0+
- GCC/G++ 编译器
- CMake 3.18+（使用 CMake 构建时需要）

验证环境：
```bash
nvcc --version
nvidia-smi
```

## 教程目录

### 基础篇（01-04）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `hello_cuda.cu` | Hello CUDA | 核函数、线程组织、执行配置 |
| `kernel_basics.cu` | 核函数基础 | `__global__`/`__device__`、dim3、多维网格 |
| `memory_management.cu` | 内存管理 | cudaMalloc、cudaMemcpy、错误检查 |
| `vector_add.cu` | 向量加法 | 完整工作流、性能计时、结果验证 |

### 进阶篇（05-07）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `05_shared_memory.cu` | 共享内存 | `__shared__`、`__syncthreads()`、Bank 冲突 |
| `06_sync_atomic.cu` | 同步与原子操作 | atomicAdd、atomicCAS、竞态条件 |
| `07_cuda_streams.cu` | CUDA 流 | 异步执行、计算与传输重叠 |

### 内存优化篇（08-10）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `08_unified_memory.cu` | 统一内存 | cudaMallocManaged、内存预取 |
| `09_texture_memory.cu` | 纹理内存 | 纹理对象、硬件插值 |
| `10_constant_reduction.cu` | 常量内存与规约 | `__constant__`、并行规约、Warp Shuffle |

### 实战篇（11-15）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `11_matrix_multiply.cu` | 矩阵乘法 | 分块优化、cuBLAS 对比 |
| `12_profiling_debug.cu` | 性能分析 | Events 计时、Nsight 工具 |
| `13_dynamic_parallelism.cu` | 动态并行 | 核函数中启动核函数 |
| `14_multi_gpu.cu` | 多 GPU 编程 | 设备管理、P2P 访问 |
| `15_thrust_practical.cu` | Thrust 库 | device_vector、算法、迭代器 |

### 库应用篇（16-20）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `16_cudnn_deeplearning.cu` | cuDNN | 卷积、激活、池化 |
| `17_cufft.cu` | cuFFT | 快速傅里叶变换 |
| `18_cusparse.cu` | cuSPARSE | 稀疏矩阵运算 |
| `19_curand.cu` | cuRAND | 随机数生成 |
| `20_cuda_graphs.cu` | CUDA Graphs | 图捕获、优化执行 |

### 高级篇（21-25）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `21_interop_graphics.cu` | 图形互操作 | OpenGL/Vulkan 集成 |
| `22_memory_pools.cu` | 内存池 | cudaMallocAsync、虚拟内存 |
| `23_cooperative_groups_advanced.cu` | 协作组 | 线程块 Tile、网格同步 |
| `24_optimization_workshop.cu` | 优化实战 | 合并访问、占用率、分支优化 |
| `25_deep_learning_integration.cu` | 深度学习集成 | 自定义算子、FP16、算子融合 |

### 专题篇（26-30）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `26_ptx_inline_assembly.cu` | PTX 汇编 | 内联汇编、底层优化 |
| `27_warp_matrix_tensor_cores.cu` | Tensor Core | WMMA、FP16/INT8 矩阵运算 |
| `28_async_copy_pipeline.cu` | 异步拷贝 | cp.async、软件流水线 |
| `29_debugging_best_practices.cu` | 调试技巧 | compute-sanitizer、常见 Bug |
| `30_image_processing_project.cu` | 图像处理 | 滤波、边缘检测、直方图 |

### 前沿应用篇（31-35）
| 文件 | 主题 | 核心内容 |
|------|------|----------|
| `31_neural_network_inference.cu` | 神经网络推理 | CNN 推理、BatchNorm |
| `32_realtime_video_processing.cu` | 实时视频处理 | 滤镜、FFT、双缓冲 |
| `33_scientific_computing.cu` | 科学计算 | 蒙特卡洛、CG 求解器、N 体模拟 |
| `34_jetson_embedded.cu` | Jetson 嵌入式 | 点云、深度图、路径规划 |
| `35_hpc_future.cu` | HPC 与前沿 | 多 GPU 集群、大模型加速 |

## 特殊编译选项

部分教程需要额外链接库：

```bash
# 矩阵乘法（cuBLAS）
nvcc -lcublas 11_matrix_multiply.cu -o 11_matrix_multiply

# 动态并行
nvcc -rdc=true -lcudadevrt 13_dynamic_parallelism.cu -o 13_dynamic_parallelism

# cuDNN
nvcc -lcudnn 16_cudnn_deeplearning.cu -o 16_cudnn_deeplearning

# 图形互操作
nvcc -lGL -lGLU -lglut 21_interop_graphics.cu -o 21_interop_graphics

# 视频处理（cuFFT）
nvcc -lcufft 32_realtime_video_processing.cu -o 32_realtime

# 科学计算（cuRAND）
nvcc -lcurand 33_scientific_computing.cu -o 33_scientific
```

## 配套文档

### 学习文档
- [CUDA 学习指南](CUDA学习指南.md) - 基础概念详解
- [CUDA 架构与原理](CUDA架构与原理.md) - GPU 硬件架构与 CUDA 执行模型深入解析
- [性能优化原理](性能优化原理.md) - 内存访问、指令级优化等性能优化理论

### 参考文档
- [快速参考卡](快速参考.md) - 常用 API 速查
- [版本兼容性指南](VERSION_COMPAT_GUIDE.md) - CUDA 版本兼容性详解
- [版本兼容性速查](QUICK_REFERENCE.md) - 兼容性快速参考
- [常见问题 FAQ](FAQ.md) - 常见问题解答

### 练习题集
- [练习题集](练习题集.md) - 基础练习
- [进阶练习题集](进阶练习题集.md) - 高级练习

## 学习建议

1. **按顺序学习**：教程难度递进，建议从 01 开始
2. **动手实践**：每个示例都要编译运行，修改参数观察变化
3. **阅读注释**：代码注释包含详细解释
4. **做练习题**：巩固所学知识
5. **查阅文档**：遇到问题先看 FAQ，再查官方文档

## 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

## License

MIT License
