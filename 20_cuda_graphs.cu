/**
 * =============================================================================
 * CUDA 教程 20: CUDA Graphs
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 CUDA Graph 的概念和优势
 * 2. 学会使用流捕获创建图
 * 3. 掌握显式图构建 API
 * 4. 了解图更新和优化技巧
 *
 * 关键概念：
 * - CUDA Graph: 预定义的 GPU 操作序列
 * - 减少启动开销，适合重复执行的工作负载
 * - 流捕获 vs 显式构建
 *
 * 编译命令：
 *   nvcc 20_cuda_graphs.cu -o 20_cuda_graphs
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 简单的内核函数
// ============================================================================

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void vectorScale(float *data, float scale, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] *= scale;
    }
}

__global__ void vectorSqrt(float *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = sqrtf(data[tid]);
    }
}

// ============================================================================
// 第一部分：CUDA Graph 基础
// ============================================================================

void demoGraphBasics() {
    printf("=== 第一部分：CUDA Graph 基础 ===\n\n");

    printf("CUDA Graph 概念:\n");
    printf("  - 将一系列 CUDA 操作定义为一个图结构\n");
    printf("  - 图可以被实例化并多次执行\n");
    printf("  - 减少 CPU-GPU 交互开销\n\n");

    printf("主要优势:\n");
    printf("  1. 减少内核启动开销\n");
    printf("  2. 优化执行调度\n");
    printf("  3. 支持复杂依赖关系\n");
    printf("  4. 适合重复执行的工作负载\n\n");

    printf("创建方式:\n");
    printf("  1. 流捕获 (Stream Capture): 自动记录操作\n");
    printf("  2. 显式 API: 手动构建图节点\n\n");

    printf("工作流程:\n");
    printf("  cudaGraph_t graph              - 图定义\n");
    printf("  cudaGraphExec_t graphExec      - 可执行图实例\n");
    printf("  \n");
    printf("  创建图 -> 实例化 -> 执行 -> (可选)更新 -> 执行\n\n");
}

// ============================================================================
// 第二部分：流捕获方式创建图
// ============================================================================

void demoStreamCapture() {
    printf("=== 第二部分：流捕获创建图 ===\n\n");

    const int N = 1 << 20;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 分配内存
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 创建流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 创建图
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // 开始流捕获
    printf("1. 开始流捕获...\n");
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 记录操作序列
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream));

    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);
    vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, 2.0f, N);
    vectorSqrt<<<gridSize, blockSize, 0, stream>>>(d_c, N);

    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream));

    // 结束捕获
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    printf("   图捕获完成\n");

    // 获取图信息
    size_t numNodes;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    printf("   图节点数: %zu\n\n", numNodes);

    // 实例化图
    printf("2. 实例化图...\n");
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // 性能测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测试图执行
    printf("\n3. 图执行性能测试:\n");
    {
        // 预热
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < 1000; i++) {
            CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float graphTime;
        CHECK_CUDA(cudaEventElapsedTime(&graphTime, start, stop));
        printf("   图执行 (1000次): %.3f ms, 平均: %.4f ms\n", graphTime, graphTime / 1000);
    }

    // 对比常规执行
    printf("\n4. 常规执行对比:\n");
    {
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < 1000; i++) {
            CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream));
            vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);
            vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, 2.0f, N);
            vectorSqrt<<<gridSize, blockSize, 0, stream>>>(d_c, N);
            CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream));
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float normalTime;
        CHECK_CUDA(cudaEventElapsedTime(&normalTime, start, stop));
        printf("   常规执行 (1000次): %.3f ms, 平均: %.4f ms\n", normalTime, normalTime / 1000);
    }

    // 验证结果
    printf("\n5. 结果验证:\n");
    printf("   c[0] = %.4f (期望: sqrt((1+2)*2) = %.4f)\n", h_c[0], sqrtf(6.0f));

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    printf("\n");
}

// ============================================================================
// 第三部分：显式 API 创建图
// ============================================================================

void demoExplicitAPI() {
    printf("=== 第三部分：显式 API 创建图 ===\n\n");

    const int N = 1 << 18;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 分配内存
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    // 创建空图
    cudaGraph_t graph;
    CHECK_CUDA(cudaGraphCreate(&graph, 0));

    printf("1. 创建图节点:\n");

    // 创建内存复制节点
    cudaGraphNode_t memcpyNode_a, memcpyNode_b, memcpyNode_result;

    cudaMemcpy3DParms memcpyParams_a = {0};
    memcpyParams_a.srcPtr = make_cudaPitchedPtr(h_data, size, N, 1);
    memcpyParams_a.dstPtr = make_cudaPitchedPtr(d_a, size, N, 1);
    memcpyParams_a.extent = make_cudaExtent(size, 1, 1);
    memcpyParams_a.kind = cudaMemcpyHostToDevice;

    CHECK_CUDA(cudaGraphAddMemcpyNode(&memcpyNode_a, graph, NULL, 0, &memcpyParams_a));
    printf("   - 添加 memcpy 节点 (H2D for a)\n");

    cudaMemcpy3DParms memcpyParams_b = memcpyParams_a;
    memcpyParams_b.dstPtr = make_cudaPitchedPtr(d_b, size, N, 1);
    CHECK_CUDA(cudaGraphAddMemcpyNode(&memcpyNode_b, graph, NULL, 0, &memcpyParams_b));
    printf("   - 添加 memcpy 节点 (H2D for b)\n");

    // 创建内核节点
    cudaGraphNode_t kernelNode_add, kernelNode_scale;

    // vectorAdd 内核节点
    cudaKernelNodeParams kernelParams_add = {0};
    void *args_add[] = {&d_a, &d_b, &d_c, (void*)&N};
    kernelParams_add.func = (void*)vectorAdd;
    kernelParams_add.gridDim = dim3(gridSize);
    kernelParams_add.blockDim = dim3(blockSize);
    kernelParams_add.sharedMemBytes = 0;
    kernelParams_add.kernelParams = args_add;
    kernelParams_add.extra = NULL;

    // 依赖于两个 memcpy 节点
    cudaGraphNode_t dependencies_add[] = {memcpyNode_a, memcpyNode_b};
    CHECK_CUDA(cudaGraphAddKernelNode(&kernelNode_add, graph, dependencies_add, 2, &kernelParams_add));
    printf("   - 添加 kernel 节点 (vectorAdd)\n");

    // vectorScale 内核节点
    float scale = 3.0f;
    cudaKernelNodeParams kernelParams_scale = {0};
    void *args_scale[] = {&d_c, &scale, (void*)&N};
    kernelParams_scale.func = (void*)vectorScale;
    kernelParams_scale.gridDim = dim3(gridSize);
    kernelParams_scale.blockDim = dim3(blockSize);
    kernelParams_scale.sharedMemBytes = 0;
    kernelParams_scale.kernelParams = args_scale;
    kernelParams_scale.extra = NULL;

    cudaGraphNode_t dependencies_scale[] = {kernelNode_add};
    CHECK_CUDA(cudaGraphAddKernelNode(&kernelNode_scale, graph, dependencies_scale, 1, &kernelParams_scale));
    printf("   - 添加 kernel 节点 (vectorScale)\n");

    // 结果复制节点
    cudaMemcpy3DParms memcpyParams_result = {0};
    memcpyParams_result.srcPtr = make_cudaPitchedPtr(d_c, size, N, 1);
    memcpyParams_result.dstPtr = make_cudaPitchedPtr(h_data, size, N, 1);
    memcpyParams_result.extent = make_cudaExtent(size, 1, 1);
    memcpyParams_result.kind = cudaMemcpyDeviceToHost;

    cudaGraphNode_t dependencies_result[] = {kernelNode_scale};
    CHECK_CUDA(cudaGraphAddMemcpyNode(&memcpyNode_result, graph, dependencies_result, 1, &memcpyParams_result));
    printf("   - 添加 memcpy 节点 (D2H)\n");

    // 获取图信息
    size_t numNodes, numEdges;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    CHECK_CUDA(cudaGraphGetEdges(graph, NULL, NULL, &numEdges));
    printf("\n2. 图结构:\n");
    printf("   节点数: %zu\n", numNodes);
    printf("   边数: %zu\n", numEdges);

    // 打印图结构
    printf("\n   图结构:\n");
    printf("   memcpy_a ─┐\n");
    printf("             ├─> vectorAdd ─> vectorScale ─> memcpy_result\n");
    printf("   memcpy_b ─┘\n\n");

    // 实例化并执行
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("3. 结果验证:\n");
    printf("   c[0] = %.1f (期望: (1+1)*3 = 6)\n\n", h_data[0]);

    // 清理
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_data);
}

// ============================================================================
// 第四部分：图更新
// ============================================================================

void demoGraphUpdate() {
    printf("=== 第四部分：图更新 ===\n\n");

    const int N = 1 << 18;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = 4.0f;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 创建初始图
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    float scale1 = 2.0f;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
    vectorScale<<<gridSize, blockSize, 0, stream>>>(d_data, scale1, N);
    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    printf("1. 初始图执行 (scale = %.1f):\n", scale1);
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("   结果: data[0] = %.1f (期望: 4 * 2 = 8)\n\n", h_data[0]);

    // 获取内核节点
    size_t numNodes;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));

    std::vector<cudaGraphNode_t> nodes(numNodes);
    CHECK_CUDA(cudaGraphGetNodes(graph, nodes.data(), &numNodes));

    // 查找内核节点
    cudaGraphNode_t kernelNode = NULL;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType type;
        CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &type));
        if (type == cudaGraphNodeTypeKernel) {
            kernelNode = nodes[i];
            break;
        }
    }

    // 更新内核参数
    printf("2. 更新图参数 (scale = 3.0):\n");
    if (kernelNode) {
        cudaKernelNodeParams params;
        CHECK_CUDA(cudaGraphKernelNodeGetParams(kernelNode, &params));

        // 创建新参数
        float newScale = 3.0f;
        void *newArgs[] = {&d_data, &newScale, (void*)&N};
        params.kernelParams = newArgs;

        // 更新节点
        CHECK_CUDA(cudaGraphKernelNodeSetParams(kernelNode, &params));

        // 更新可执行图
        cudaGraphExecUpdateResult updateResult;
        CHECK_CUDA(cudaGraphExecUpdate(graphExec, graph, NULL, &updateResult));

        if (updateResult == cudaGraphExecUpdateSuccess) {
            printf("   图更新成功\n");
        } else {
            printf("   需要重新实例化\n");
            CHECK_CUDA(cudaGraphExecDestroy(graphExec));
            CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        }
    }

    // 重新执行
    for (int i = 0; i < N; i++) h_data[i] = 4.0f;
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("   结果: data[0] = %.1f (期望: 4 * 3 = 12)\n\n", h_data[0]);

    printf("图更新场景:\n");
    printf("  - 更新内核参数 (不改变结构)\n");
    printf("  - 更新内存地址\n");
    printf("  - 修改执行配置\n");
    printf("  注意: 结构性变化需要重新实例化\n\n");

    // 清理
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

// ============================================================================
// 第五部分：子图和条件节点
// ============================================================================

void demoChildGraph() {
    printf("=== 第五部分：子图 ===\n\n");

    const int N = 1 << 16;
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // 创建子图
    cudaGraph_t childGraph;
    CHECK_CUDA(cudaGraphCreate(&childGraph, 0));

    // 在子图中添加两个内核
    cudaGraphNode_t scaleNode, sqrtNode;

    float scale = 2.0f;
    cudaKernelNodeParams scaleParams = {0};
    void *scaleArgs[] = {&d_data, &scale, (void*)&N};
    scaleParams.func = (void*)vectorScale;
    scaleParams.gridDim = dim3(gridSize);
    scaleParams.blockDim = dim3(blockSize);
    scaleParams.kernelParams = scaleArgs;

    CHECK_CUDA(cudaGraphAddKernelNode(&scaleNode, childGraph, NULL, 0, &scaleParams));

    cudaKernelNodeParams sqrtParams = {0};
    void *sqrtArgs[] = {&d_data, (void*)&N};
    sqrtParams.func = (void*)vectorSqrt;
    sqrtParams.gridDim = dim3(gridSize);
    sqrtParams.blockDim = dim3(blockSize);
    sqrtParams.kernelParams = sqrtArgs;

    cudaGraphNode_t deps[] = {scaleNode};
    CHECK_CUDA(cudaGraphAddKernelNode(&sqrtNode, childGraph, deps, 1, &sqrtParams));

    printf("1. 创建子图 (scale -> sqrt)\n");

    // 创建父图
    cudaGraph_t parentGraph;
    CHECK_CUDA(cudaGraphCreate(&parentGraph, 0));

    // 添加子图节点
    cudaGraphNode_t childNode;
    CHECK_CUDA(cudaGraphAddChildGraphNode(&childNode, parentGraph, NULL, 0, childGraph));
    printf("2. 将子图添加到父图\n");

    // 获取信息
    size_t parentNodes, childNodes;
    CHECK_CUDA(cudaGraphGetNodes(parentGraph, NULL, &parentNodes));
    CHECK_CUDA(cudaGraphGetNodes(childGraph, NULL, &childNodes));

    printf("\n图结构:\n");
    printf("  父图节点: %zu\n", parentNodes);
    printf("  子图节点: %zu\n", childNodes);
    printf("\n  父图\n");
    printf("    └── 子图节点\n");
    printf("          ├── vectorScale\n");
    printf("          └── vectorSqrt\n\n");

    // 实例化并执行
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, parentGraph, NULL, NULL, 0));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 初始化数据
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = 8.0f;
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    printf("3. 执行结果:\n");
    printf("   data[0] = %.4f (期望: sqrt(8*2) = %.4f)\n\n", h_data[0], sqrtf(16.0f));

    // 清理
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(childGraph));
    CHECK_CUDA(cudaGraphDestroy(parentGraph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

// ============================================================================
// 第六部分：性能最佳实践
// ============================================================================

void demoBestPractices() {
    printf("=== 第六部分：最佳实践 ===\n\n");

    printf("1. 何时使用 CUDA Graph:\n");
    printf("   ✓ 重复执行相同操作序列\n");
    printf("   ✓ 小内核，启动开销显著\n");
    printf("   ✓ 复杂的依赖关系\n");
    printf("   ✓ 需要确定性执行\n\n");

    printf("2. 何时不使用:\n");
    printf("   ✗ 操作序列每次都不同\n");
    printf("   ✗ 大内核，启动开销可忽略\n");
    printf("   ✗ 需要动态分支\n\n");

    printf("3. 流捕获 vs 显式 API:\n");
    printf("   流捕获:\n");
    printf("     + 简单易用\n");
    printf("     + 自动处理依赖\n");
    printf("     - 灵活性较低\n");
    printf("   显式 API:\n");
    printf("     + 完全控制图结构\n");
    printf("     + 可以构建复杂图\n");
    printf("     - 代码更复杂\n\n");

    printf("4. 性能优化技巧:\n");
    printf("   - 预先实例化图\n");
    printf("   - 使用图更新而非重新创建\n");
    printf("   - 合并小操作到单个图\n");
    printf("   - 使用子图组织复杂逻辑\n\n");

    printf("5. 注意事项:\n");
    printf("   - 图中不能使用 cudaMalloc/cudaFree\n");
    printf("   - 主机回调有限制\n");
    printf("   - 同步操作会打断捕获\n");
    printf("   - 图实例化有一定开销\n\n");

    // 性能对比演示
    const int N = 1 << 14;  // 小数据，突出启动开销
    const int size = N * sizeof(float);
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 创建图
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);
    vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, 2.0f, N);
    vectorSqrt<<<gridSize, blockSize, 0, stream>>>(d_c, N);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // 预热
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("6. 小数据性能对比 (N=%d):\n\n", N);

    // 图执行
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < 10000; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float graphTime;
    CHECK_CUDA(cudaEventElapsedTime(&graphTime, start, stop));

    // 常规执行
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < 10000; i++) {
        vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);
        vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, 2.0f, N);
        vectorSqrt<<<gridSize, blockSize, 0, stream>>>(d_c, N);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float normalTime;
    CHECK_CUDA(cudaEventElapsedTime(&normalTime, start, stop));

    printf("   图执行 (10000次): %.2f ms\n", graphTime);
    printf("   常规执行 (10000次): %.2f ms\n", normalTime);
    printf("   加速比: %.2fx\n\n", normalTime / graphTime);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 20: CUDA Graphs                               ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n\n", prop.major, prop.minor);

    demoGraphBasics();
    demoStreamCapture();
    demoExplicitAPI();
    demoGraphUpdate();
    demoChildGraph();
    demoBestPractices();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. CUDA Graph 工作流:\n");
    printf("   创建图 -> 实例化 -> 启动 -> (更新) -> 启动\n\n");

    printf("2. 主要 API:\n");
    printf("   ┌──────────────────────────────┬────────────────────────────┐\n");
    printf("   │ 函数                         │ 功能                       │\n");
    printf("   ├──────────────────────────────┼────────────────────────────┤\n");
    printf("   │ cudaStreamBeginCapture       │ 开始流捕获                 │\n");
    printf("   │ cudaStreamEndCapture         │ 结束捕获，获取图           │\n");
    printf("   │ cudaGraphCreate              │ 创建空图                   │\n");
    printf("   │ cudaGraphAdd*Node            │ 添加节点                   │\n");
    printf("   │ cudaGraphInstantiate         │ 实例化可执行图             │\n");
    printf("   │ cudaGraphLaunch              │ 启动图                     │\n");
    printf("   │ cudaGraphExecUpdate          │ 更新可执行图               │\n");
    printf("   └──────────────────────────────┴────────────────────────────┘\n\n");

    printf("3. 节点类型:\n");
    printf("   - Kernel: GPU 内核\n");
    printf("   - Memcpy: 内存复制\n");
    printf("   - Memset: 内存设置\n");
    printf("   - Host: 主机回调\n");
    printf("   - Child Graph: 子图\n");
    printf("   - Event: 事件记录/等待\n\n");

    printf("4. 适用场景:\n");
    printf("   - 推理管道\n");
    printf("   - 迭代算法\n");
    printf("   - 小内核密集工作负载\n");
    printf("   - 需要低延迟的应用\n\n");

    return 0;
}
