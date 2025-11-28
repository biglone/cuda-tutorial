/**
 * =============================================================================
 * CUDA 教程 35: 高性能计算案例与未来展望
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 GPU 在 HPC 领域的应用案例
 * 2. 学习前沿技术和最新发展
 * 3. 掌握性能优化的高级技巧
 * 4. 展望 GPU 计算的未来趋势
 *
 * 内容概述：
 * - 真实 HPC 应用案例
 * - 最新 CUDA 特性 (CUDA 12+)
 * - 多 GPU 和集群计算
 * - AI 加速和大模型
 * - 未来技术展望
 *
 * 编译命令：
 *   nvcc 35_hpc_future.cu -o 35_hpc -O3
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
// 第一部分：GPU 在 HPC 中的应用案例
// ============================================================================

void demoHPCApplications() {
    printf("=== 第一部分：GPU 在 HPC 中的应用案例 ===\n\n");

    printf("1. 分子动力学模拟:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 软件: GROMACS, AMBER, NAMD, LAMMPS                        │\n");
    printf("   │ 应用: 蛋白质折叠、药物设计、材料模拟                      │\n");
    printf("   │ 加速: 相比 CPU 可达 10-100x                               │\n");
    printf("   │ 特点: N体问题，近程/远程力计算                            │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("2. 天气/气候模拟:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 软件: WRF, CESM, MPAS                                     │\n");
    printf("   │ 应用: 天气预报、气候变化研究                              │\n");
    printf("   │ 特点: 大规模有限差分、FFT、稀疏线性求解                   │\n");
    printf("   │ 挑战: 复杂边界条件、多物理耦合                            │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("3. 计算流体力学 (CFD):\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 软件: OpenFOAM (GPU), Ansys Fluent, STAR-CCM+             │\n");
    printf("   │ 应用: 航空航天、汽车、能源                                │\n");
    printf("   │ 方法: 有限体积法、有限元法、格子玻尔兹曼                  │\n");
    printf("   │ 加速: GPU 版本可达 5-20x                                  │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("4. 基因组学:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 软件: NVIDIA Parabricks, BWA-MEM2, GATK                   │\n");
    printf("   │ 应用: 基因测序分析、变异检测                              │\n");
    printf("   │ 性能: 30x 全基因组分析从 24 小时缩短到 1 小时            │\n");
    printf("   │ 特点: 字符串匹配、动态规划、排序                          │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("5. 金融计算:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 应用: 期权定价、风险分析、高频交易                        │\n");
    printf("   │ 方法: Monte Carlo、PDE 求解、机器学习                     │\n");
    printf("   │ 优势: 毫秒级延迟、大规模并行模拟                          │\n");
    printf("   │ 挑战: 低延迟要求、监管合规                                │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("6. 量子化学:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 软件: Gaussian, VASP, Quantum ESPRESSO, NWChem            │\n");
    printf("   │ 应用: 分子性质计算、材料设计                              │\n");
    printf("   │ 方法: DFT、HF、耦合簇                                     │\n");
    printf("   │ 加速: 密集矩阵运算大幅受益于 GPU                          │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第二部分：最新 CUDA 特性
// ============================================================================

void demoCUDANewFeatures() {
    printf("=== 第二部分：最新 CUDA 特性 (CUDA 11/12+) ===\n\n");

    printf("1. 异步操作增强:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ cudaMemcpyAsync 改进                                      │\n");
    printf("   │   - 更低延迟的异步拷贝                                    │\n");
    printf("   │   - 改进的 DMA 引擎利用                                   │\n");
    printf("   │                                                           │\n");
    printf("   │ CUDA Graphs 增强                                          │\n");
    printf("   │   - 条件节点 (cudaGraphAddConditionalNodes)               │\n");
    printf("   │   - 图更新优化                                            │\n");
    printf("   │   - 多设备图支持                                          │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("2. 内存管理:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 流序内存分配 (Stream-Ordered Memory)                      │\n");
    printf("   │   cudaMallocAsync / cudaFreeAsync                         │\n");
    printf("   │   - 减少内存碎片                                          │\n");
    printf("   │   - 自动内存池管理                                        │\n");
    printf("   │                                                           │\n");
    printf("   │ 虚拟内存管理 (VMM)                                        │\n");
    printf("   │   - 动态内存映射                                          │\n");
    printf("   │   - 跨 GPU 内存共享                                       │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("3. 协作组 (Cooperative Groups):\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 线程块簇 (Thread Block Clusters) - Hopper 架构            │\n");
    printf("   │   - 跨线程块的高效通信                                    │\n");
    printf("   │   - 分布式共享内存                                        │\n");
    printf("   │                                                           │\n");
    printf("   │ 新同步原语                                                │\n");
    printf("   │   - arrive/wait 屏障                                      │\n");
    printf("   │   - 异步屏障                                              │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("4. Tensor Core 增强:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ FP8 支持 (Hopper)                                         │\n");
    printf("   │   - E4M3 和 E5M2 格式                                     │\n");
    printf("   │   - 2x FP16 吞吐量                                        │\n");
    printf("   │                                                           │\n");
    printf("   │ Transformer Engine                                        │\n");
    printf("   │   - 自动混合精度                                          │\n");
    printf("   │   - 动态精度选择                                          │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("5. 编译器改进:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ NVCC 改进                                                 │\n");
    printf("   │   - C++20 支持                                            │\n");
    printf("   │   - 更快的编译速度                                        │\n");
    printf("   │                                                           │\n");
    printf("   │ NVRTC (运行时编译)                                        │\n");
    printf("   │   - JIT 编译优化                                          │\n");
    printf("   │   - 并行编译支持                                          │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    // 检查当前 CUDA 版本
    int runtimeVersion, driverVersion;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));

    printf("当前 CUDA 环境:\n");
    printf("  运行时版本: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  驱动版本: %d.%d\n\n", driverVersion / 1000, (driverVersion % 100) / 10);
}

// ============================================================================
// 第三部分：多 GPU 和集群计算
// ============================================================================

void demoMultiGPUCluster() {
    printf("=== 第三部分：多 GPU 和集群计算 ===\n\n");

    // 检测可用 GPU
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    printf("检测到 %d 个 GPU 设备\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    printf("\n");

    printf("多 GPU 通信技术:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ 1. P2P 直接访问                                           │\n");
    printf("  │    cudaDeviceCanAccessPeer()                              │\n");
    printf("  │    cudaDeviceEnablePeerAccess()                           │\n");
    printf("  │    - 同一 PCIe 交换机下的 GPU 间直接内存访问              │\n");
    printf("  │                                                           │\n");
    printf("  │ 2. NVLink                                                 │\n");
    printf("  │    - 高带宽低延迟的 GPU 间互连                            │\n");
    printf("  │    - 支持 GPU 间直接原子操作                              │\n");
    printf("  │    - NVLink 4: 900 GB/s 双向带宽                          │\n");
    printf("  │                                                           │\n");
    printf("  │ 3. NVSwitch                                               │\n");
    printf("  │    - 全连接 GPU 拓扑                                      │\n");
    printf("  │    - DGX 系统使用                                         │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    printf("分布式计算框架:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ NCCL (NVIDIA Collective Communications Library)           │\n");
    printf("  │   - AllReduce, Broadcast, AllGather 等集合操作            │\n");
    printf("  │   - 自动拓扑感知优化                                      │\n");
    printf("  │   - 多节点多 GPU 支持                                     │\n");
    printf("  │                                                           │\n");
    printf("  │ MPI + CUDA                                                │\n");
    printf("  │   - 传统 HPC 并行编程模型                                 │\n");
    printf("  │   - GPU-aware MPI (无需手动拷贝)                          │\n");
    printf("  │                                                           │\n");
    printf("  │ CUDA-aware RDMA                                           │\n");
    printf("  │   - GPUDirect RDMA: GPU 内存直接网络传输                  │\n");
    printf("  │   - 绕过 CPU 和系统内存                                   │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    printf("扩展策略:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ 数据并行 (Data Parallelism)                               │\n");
    printf("  │   - 每个 GPU 处理不同数据批次                             │\n");
    printf("  │   - 梯度同步 (AllReduce)                                  │\n");
    printf("  │   - 适合: 大批量训练                                      │\n");
    printf("  │                                                           │\n");
    printf("  │ 模型并行 (Model Parallelism)                              │\n");
    printf("  │   - 张量并行: 分割单个层                                  │\n");
    printf("  │   - 流水线并行: 分割层序列                                │\n");
    printf("  │   - 适合: 超大模型 (GPT, LLaMA)                           │\n");
    printf("  │                                                           │\n");
    printf("  │ 混合并行                                                  │\n");
    printf("  │   - 结合数据并行和模型并行                                │\n");
    printf("  │   - 3D 并行: 数据 × 张量 × 流水线                        │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第四部分：AI 加速和大模型
// ============================================================================

void demoAIAcceleration() {
    printf("=== 第四部分：AI 加速和大模型 ===\n\n");

    printf("大语言模型 (LLM) 推理优化:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ KV Cache 优化                                             │\n");
    printf("  │   - PagedAttention: 虚拟内存风格的 KV 缓存管理            │\n");
    printf("  │   - 动态批处理: 连续批处理减少碎片                        │\n");
    printf("  │                                                           │\n");
    printf("  │ 量化技术                                                  │\n");
    printf("  │   - INT8/INT4 权重量化                                    │\n");
    printf("  │   - FP8 (Hopper) 训练和推理                               │\n");
    printf("  │   - AWQ, GPTQ, SmoothQuant                                │\n");
    printf("  │                                                           │\n");
    printf("  │ 注意力优化                                                │\n");
    printf("  │   - FlashAttention: IO 感知的注意力实现                   │\n");
    printf("  │   - Multi-Query/Grouped-Query Attention                   │\n");
    printf("  │   - Tensor Core 加速                                      │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    printf("推理框架:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ TensorRT-LLM                                              │\n");
    printf("  │   - NVIDIA 官方 LLM 推理库                                │\n");
    printf("  │   - 支持主流模型 (LLaMA, GPT, BERT 等)                    │\n");
    printf("  │   - 自动优化和量化                                        │\n");
    printf("  │                                                           │\n");
    printf("  │ vLLM                                                      │\n");
    printf("  │   - PagedAttention 原始实现                               │\n");
    printf("  │   - 高吞吐量推理服务                                      │\n");
    printf("  │                                                           │\n");
    printf("  │ Triton Inference Server                                   │\n");
    printf("  │   - 生产级推理服务                                        │\n");
    printf("  │   - 动态批处理、模型集成                                  │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    printf("训练优化:\n");
    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │ 混合精度训练                                              │\n");
    printf("  │   - AMP (Automatic Mixed Precision)                       │\n");
    printf("  │   - 损失缩放 (Loss Scaling)                               │\n");
    printf("  │   - FP8 训练 (Hopper+)                                    │\n");
    printf("  │                                                           │\n");
    printf("  │ 分布式训练                                                │\n");
    printf("  │   - DeepSpeed ZeRO                                        │\n");
    printf("  │   - FSDP (Fully Sharded Data Parallel)                    │\n");
    printf("  │   - Megatron-LM                                           │\n");
    printf("  │                                                           │\n");
    printf("  │ 内存优化                                                  │\n");
    printf("  │   - 梯度检查点 (Gradient Checkpointing)                   │\n");
    printf("  │   - 激活重计算                                            │\n");
    printf("  │   - CPU Offloading                                        │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    printf("Transformer 性能估算:\n");
    printf("  参数量 P, 批次 B, 序列长度 S, 隐藏维度 H\n\n");
    printf("  前向传播 FLOPs ≈ 2 × P × B × S\n");
    printf("  后向传播 FLOPs ≈ 4 × P × B × S\n");
    printf("  注意力 FLOPs ≈ 4 × B × S² × H × L (L 层数)\n\n");

    printf("  示例: 7B 模型, B=1, S=2048\n");
    printf("    前向 FLOPs ≈ 2 × 7B × 2048 ≈ 28.7 TFLOPs\n");
    printf("    A100 (312 TFLOPS FP16): ~92ms\n");
    printf("    H100 (989 TFLOPS FP16): ~29ms\n\n");
}

// ============================================================================
// 第五部分：性能优化高级技巧
// ============================================================================

// 演示内存合并访问的重要性
__global__ void coalescedAccessKernel(float *output, const float *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid] * 2.0f;  // 合并访问
    }
}

__global__ void stridedAccessKernel(float *output, const float *input, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // 跨步访问 (低效)
    }
}

void demoAdvancedOptimization() {
    printf("=== 第五部分：性能优化高级技巧 ===\n\n");

    printf("1. 内存访问模式:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 合并访问 (Coalesced Access)                               │\n");
    printf("   │   - 连续线程访问连续内存地址                              │\n");
    printf("   │   - 理想: 一次事务读取 128 字节                          │\n");
    printf("   │                                                           │\n");
    printf("   │ 跨步访问                                                  │\n");
    printf("   │   - 避免 stride = 32 (warp 大小)                          │\n");
    printf("   │   - 使用共享内存重排                                      │\n");
    printf("   │                                                           │\n");
    printf("   │ 向量化加载                                                │\n");
    printf("   │   - float4, int4 等向量类型                               │\n");
    printf("   │   - 单次指令加载多个元素                                  │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    // 性能对比测试
    const int N = 1 << 24;  // 16M 元素
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 合并访问
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        coalescedAccessKernel<<<grid, block>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coalescedTime;
    CHECK_CUDA(cudaEventElapsedTime(&coalescedTime, start, stop));
    coalescedTime /= 100;

    // 跨步访问 (stride=32)
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        stridedAccessKernel<<<grid, block>>>(d_output, d_input, N, 32);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float stridedTime;
    CHECK_CUDA(cudaEventElapsedTime(&stridedTime, start, stop));
    stridedTime /= 100;

    float bandwidth = 2.0f * N * sizeof(float) / coalescedTime / 1e6;  // GB/s

    printf("内存访问模式对比 (%d 元素):\n", N);
    printf("  ┌───────────────────┬───────────┬─────────────────┐\n");
    printf("  │ 访问模式          │ 时间 (ms) │ 带宽 (GB/s)     │\n");
    printf("  ├───────────────────┼───────────┼─────────────────┤\n");
    printf("  │ 合并访问          │ %9.3f │ %7.2f         │\n", coalescedTime, bandwidth);
    printf("  │ 跨步访问 (32)     │ %9.3f │ %7.2f         │\n",
           stridedTime, 2.0f * N * sizeof(float) / stridedTime / 1e6 / 32);
    printf("  └───────────────────┴───────────┴─────────────────┘\n\n");

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("2. 占用率优化:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 影响因素                                                  │\n");
    printf("   │   - 每块线程数                                            │\n");
    printf("   │   - 每线程寄存器数                                        │\n");
    printf("   │   - 每块共享内存                                          │\n");
    printf("   │                                                           │\n");
    printf("   │ 优化建议                                                  │\n");
    printf("   │   - 使用 __launch_bounds__ 限制寄存器                     │\n");
    printf("   │   - 使用 cudaOccupancyMaxPotentialBlockSize               │\n");
    printf("   │   - 高占用率不总是最优                                    │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("3. 指令级优化:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 数学函数                                                  │\n");
    printf("   │   - __sinf/__cosf (快速但精度低)                          │\n");
    printf("   │   - __fdividef (快速除法)                                 │\n");
    printf("   │   - rsqrtf (快速平方根倒数)                               │\n");
    printf("   │                                                           │\n");
    printf("   │ 分支优化                                                  │\n");
    printf("   │   - 避免 warp 内分支发散                                  │\n");
    printf("   │   - 使用 predication 替代分支                             │\n");
    printf("   │   - 重排数据减少分支                                      │\n");
    printf("   │                                                           │\n");
    printf("   │ 循环优化                                                  │\n");
    printf("   │   - #pragma unroll                                        │\n");
    printf("   │   - 循环展开减少开销                                      │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 第六部分：未来展望
// ============================================================================

void demoFutureOutlook() {
    printf("=== 第六部分：未来展望 ===\n\n");

    printf("1. 硬件发展趋势:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 更大的 GPU 内存                                           │\n");
    printf("   │   - HBM3e: 更高带宽和容量                                 │\n");
    printf("   │   - 堆叠内存技术                                          │\n");
    printf("   │                                                           │\n");
    printf("   │ 更高的互连带宽                                            │\n");
    printf("   │   - NVLink 5+                                             │\n");
    printf("   │   - CXL (Compute Express Link) 集成                       │\n");
    printf("   │                                                           │\n");
    printf("   │ 专用加速器                                                │\n");
    printf("   │   - Transformer Engine 增强                               │\n");
    printf("   │   - 稀疏计算加速                                          │\n");
    printf("   │   - 量化计算专用单元                                      │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("2. 软件和编程模型:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 更高级的抽象                                              │\n");
    printf("   │   - CUDA Python 成熟                                      │\n");
    printf("   │   - 编译器自动优化增强                                    │\n");
    printf("   │   - AI 辅助代码优化                                       │\n");
    printf("   │                                                           │\n");
    printf("   │ 异构计算                                                  │\n");
    printf("   │   - CPU-GPU 协同优化                                      │\n");
    printf("   │   - 多种加速器统一编程                                    │\n");
    printf("   │   - 自动任务调度                                          │\n");
    printf("   │                                                           │\n");
    printf("   │ 云原生支持                                                │\n");
    printf("   │   - Kubernetes GPU 调度                                   │\n");
    printf("   │   - MIG (Multi-Instance GPU)                              │\n");
    printf("   │   - 虚拟化增强                                            │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("3. 新兴应用领域:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 生成式 AI                                                 │\n");
    printf("   │   - 大语言模型 (LLM) 服务                                 │\n");
    printf("   │   - 图像/视频生成 (Diffusion)                             │\n");
    printf("   │   - 多模态模型                                            │\n");
    printf("   │                                                           │\n");
    printf("   │ 数字孪生                                                  │\n");
    printf("   │   - 物理模拟实时化                                        │\n");
    printf("   │   - Omniverse 平台                                        │\n");
    printf("   │   - 工业元宇宙                                            │\n");
    printf("   │                                                           │\n");
    printf("   │ 边缘 AI                                                   │\n");
    printf("   │   - 自动驾驶                                              │\n");
    printf("   │   - 机器人                                                │\n");
    printf("   │   - 智能物联网                                            │\n");
    printf("   │                                                           │\n");
    printf("   │ 科学发现                                                  │\n");
    printf("   │   - AI for Science                                        │\n");
    printf("   │   - 蛋白质结构预测                                        │\n");
    printf("   │   - 材料设计                                              │\n");
    printf("   │   - 药物发现                                              │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");

    printf("4. 学习建议:\n");
    printf("   ┌───────────────────────────────────────────────────────────┐\n");
    printf("   │ 基础                                                      │\n");
    printf("   │   - 掌握 CUDA C/C++ 编程                                  │\n");
    printf("   │   - 理解 GPU 架构和内存层次                               │\n");
    printf("   │   - 熟练使用性能分析工具                                  │\n");
    printf("   │                                                           │\n");
    printf("   │ 进阶                                                      │\n");
    printf("   │   - 学习 Tensor Core 编程                                 │\n");
    printf("   │   - 掌握多 GPU 编程和 NCCL                                │\n");
    printf("   │   - 了解 TensorRT/Triton 部署                             │\n");
    printf("   │                                                           │\n");
    printf("   │ 专精方向                                                  │\n");
    printf("   │   - HPC: MPI + CUDA, 大规模模拟                           │\n");
    printf("   │   - AI: 模型优化, 推理加速                                │\n");
    printf("   │   - 图形: CUDA + OpenGL/Vulkan                            │\n");
    printf("   │   - 边缘: Jetson 开发                                     │\n");
    printf("   └───────────────────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   CUDA 教程 35: 高性能计算案例与未来展望                         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    demoHPCApplications();
    demoCUDANewFeatures();
    demoMultiGPUCluster();
    demoAIAcceleration();
    demoAdvancedOptimization();
    demoFutureOutlook();

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                     CUDA 教程系列总结                           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("恭喜完成全部 35 节 CUDA 教程！\n\n");

    printf("教程回顾:\n");
    printf("  基础篇 (01-04): CUDA 编程基础\n");
    printf("    - 内核函数、线程组织、内存管理\n\n");

    printf("  进阶篇 (05-07): 性能优化入门\n");
    printf("    - 共享内存、同步、Streams\n\n");

    printf("  内存优化篇 (08-10): 内存系统深入\n");
    printf("    - 统一内存、纹理、常量内存\n\n");

    printf("  实战篇 (11-15): 实用算法实现\n");
    printf("    - 矩阵乘法、Profiling、动态并行\n\n");

    printf("  库应用篇 (16-20): CUDA 生态系统\n");
    printf("    - cuDNN, cuFFT, cuSPARSE, CUDA Graphs\n\n");

    printf("  高级篇 (21-25): 高级主题\n");
    printf("    - 图形互操作、内存池、深度优化\n\n");

    printf("  专题篇 (26-35): 前沿应用\n");
    printf("    - PTX、Tensor Core、调试、图像处理\n");
    printf("    - 神经网络、视频处理、科学计算\n");
    printf("    - Jetson 嵌入式、HPC 应用\n\n");

    printf("下一步建议:\n");
    printf("  1. 选择感兴趣的领域深入学习\n");
    printf("  2. 参与开源项目积累经验\n");
    printf("  3. 阅读 NVIDIA 官方文档和博客\n");
    printf("  4. 参加 GTC 大会了解最新技术\n");
    printf("  5. 实践是最好的老师！\n\n");

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                 感谢学习，祝你 CUDA 之旅愉快！                    ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
