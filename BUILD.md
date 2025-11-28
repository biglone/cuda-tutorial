# CUDA 教程构建指南

本文档介绍如何编译和运行 CUDA 教程项目。

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| CUDA Toolkit | 10.0+ | 推荐 11.0+ |
| CMake | 3.18+ | 构建系统 |
| GCC/G++ | 7.0+ | C++ 编译器 |
| GPU | 计算能力 3.0+ | NVIDIA GPU |

### 可选依赖

| 库 | 用途 | 相关教程 |
|----|------|----------|
| cuBLAS | 矩阵运算 | 11_matrix_multiply |
| cuDNN | 深度学习 | 16_cudnn_deeplearning |
| cuFFT | 傅里叶变换 | 17_cufft, 32_realtime |
| cuSPARSE | 稀疏矩阵 | 18_cusparse |
| cuRAND | 随机数 | 19_curand, 33_scientific |
| OpenGL/GLUT | 图形互操作 | 21_interop_graphics |

## 快速开始

### 方式一：使用构建脚本（推荐）

```bash
# 编译全部教程
./build.sh

# 编译并运行单个程序
./build.sh run hello_cuda

# 查看帮助
./build.sh help
```

### 方式二：手动使用 CMake

```bash
# 创建构建目录
mkdir build && cd build

# 配置
cmake ..

# 编译全部
make -j$(nproc)

# 或编译单个
make hello_cuda
```

## 构建脚本用法

```bash
./build.sh [命令] [选项]
```

### 命令列表

| 命令 | 说明 | 示例 |
|------|------|------|
| (无) | 编译全部教程 | `./build.sh` |
| `basics` | 基础篇 (01-04) | `./build.sh basics` |
| `advanced` | 进阶篇 (05-10) | `./build.sh advanced` |
| `practical` | 实战篇 (11-15) | `./build.sh practical` |
| `libraries` | 库应用篇 (16-20) | `./build.sh libraries` |
| `high_level` | 高级篇 (21-25) | `./build.sh high_level` |
| `special_topics` | 专题篇 (26-30) | `./build.sh special_topics` |
| `frontier` | 前沿应用篇 (31-35) | `./build.sh frontier` |
| `<target>` | 编译单个教程 | `./build.sh 08_unified_memory` |
| `run <target>` | 编译并运行 | `./build.sh run hello_cuda` |
| `clean` | 清理构建目录 | `./build.sh clean` |
| `rebuild` | 清理后重新编译 | `./build.sh rebuild` |
| `list` | 列出所有目标 | `./build.sh list` |
| `help` | 显示帮助 | `./build.sh help` |

## 教程章节

### 基础篇 (01-04)
```bash
./build.sh basics
```
- `hello_cuda` - CUDA 入门
- `kernel_basics` - 核函数基础
- `memory_management` - 内存管理
- `vector_add` - 向量加法

### 进阶篇 (05-10)
```bash
./build.sh advanced
```
- `05_shared_memory` - 共享内存
- `06_sync_atomic` - 同步与原子操作
- `07_cuda_streams` - CUDA Streams
- `08_unified_memory` - 统一内存
- `09_texture_memory` - 纹理内存
- `10_constant_reduction` - 常量内存与归约

### 实战篇 (11-15)
```bash
./build.sh practical
```
- `11_matrix_multiply` - 矩阵乘法优化
- `12_profiling_debug` - 性能分析
- `13_dynamic_parallelism` - 动态并行
- `14_multi_gpu` - 多 GPU 编程
- `15_thrust_practical` - Thrust 库

### 库应用篇 (16-20)
```bash
./build.sh libraries
```
- `16_cudnn_deeplearning` - cuDNN 深度学习
- `17_cufft` - 快速傅里叶变换
- `18_cusparse` - 稀疏矩阵
- `19_curand` - 随机数生成
- `20_cuda_graphs` - CUDA Graphs

### 高级篇 (21-25)
```bash
./build.sh high_level
```
- `21_interop_graphics` - 图形互操作
- `22_memory_pools` - 内存池
- `23_cooperative_groups_advanced` - 协作组
- `24_optimization_workshop` - 优化实战
- `25_deep_learning_integration` - 深度学习集成

### 专题篇 (26-30)
```bash
./build.sh special_topics
```
- `26_ptx_inline_assembly` - PTX 内联汇编
- `27_warp_matrix_tensor_cores` - Tensor Cores
- `28_async_copy_pipeline` - 异步拷贝
- `29_debugging_best_practices` - 调试技巧
- `30_image_processing_project` - 图像处理

### 前沿应用篇 (31-35)
```bash
./build.sh frontier
```
- `31_neural_network_inference` - 神经网络推理
- `32_realtime_video_processing` - 实时视频处理
- `33_scientific_computing` - 科学计算
- `34_jetson_embedded` - Jetson 嵌入式
- `35_hpc_future` - HPC 与未来展望

## 运行程序

编译后的可执行文件位于 `build/bin/` 目录：

```bash
# 直接运行
./build/bin/hello_cuda

# 或使用构建脚本
./build.sh run hello_cuda
```

## 常见问题

### 1. CMake 版本过低

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake

# 或从官网下载新版本
# https://cmake.org/download/
```

### 2. 找不到 CUDA

确保 CUDA 环境变量已设置：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. 某些教程被跳过

部分教程需要可选依赖，CMake 配置时会显示：

```
-- cuDNN not found, skipping 16_cudnn_deeplearning
-- OpenGL/GLUT not found, skipping 21_interop_graphics
```

安装相应依赖后重新配置即可。

### 4. GPU 架构不匹配

CMake 会自动检测 GPU 架构，也可手动指定：

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75  # RTX 20 系列
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 30 系列
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89  # RTX 40 系列
```

### 5. 清理重建

```bash
./build.sh rebuild
# 或
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
```

## 项目结构

```
CUDA/
├── build.sh              # 构建脚本
├── BUILD.md              # 构建文档（本文件）
├── CMakeLists.txt        # CMake 配置
├── cmake/
│   └── CUDASetup.cmake   # CUDA 配置模块
├── CLAUDE.md             # 项目说明
├── README.md             # 项目简介
├── hello_cuda.cu         # 教程 01
├── kernel_basics.cu      # 教程 02
├── ...                   # 其他教程
└── build/                # 构建目录（自动生成）
    ├── bin/              # 可执行文件
    └── ...
```

## 验证安装

```bash
# 检查 CUDA
nvcc --version

# 检查 CMake
cmake --version

# 检查 GPU
nvidia-smi

# 编译并运行测试
./build.sh run hello_cuda
```
