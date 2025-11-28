/**
 * =============================================================================
 * CUDA 教程 18: cuSPARSE 稀疏矩阵运算
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解稀疏矩阵的存储格式
 * 2. 学会使用 cuSPARSE 进行稀疏矩阵运算
 * 3. 掌握 SpMV (稀疏矩阵-向量乘法) 操作
 * 4. 了解稀疏矩阵在实际应用中的优势
 *
 * 关键概念：
 * - 稀疏矩阵：大部分元素为零的矩阵
 * - CSR (Compressed Sparse Row): 压缩稀疏行格式
 * - COO (Coordinate): 坐标格式
 * - cuSPARSE: NVIDIA 稀疏线性代数库
 *
 * 编译命令：
 *   nvcc -lcusparse 18_cusparse.cu -o 18_cusparse
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE 错误 %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：稀疏矩阵基础
// ============================================================================

void demoSparseBasics() {
    printf("=== 第一部分：稀疏矩阵基础 ===\n\n");

    printf("稀疏矩阵定义:\n");
    printf("  - 大部分元素为零的矩阵\n");
    printf("  - 稀疏度 = 非零元素数 / 总元素数\n");
    printf("  - 通常稀疏度 < 10%% 时使用稀疏格式\n\n");

    printf("常见存储格式:\n\n");

    printf("1. COO (Coordinate) 格式:\n");
    printf("   - 存储: (行索引, 列索引, 值) 三元组\n");
    printf("   - 优点: 构建简单，易于添加元素\n");
    printf("   - 缺点: 随机访问慢\n\n");

    printf("2. CSR (Compressed Sparse Row) 格式:\n");
    printf("   - rowPtr[]: 每行第一个非零元素的索引\n");
    printf("   - colInd[]: 非零元素的列索引\n");
    printf("   - values[]: 非零元素的值\n");
    printf("   - 优点: 行访问快，存储紧凑\n");
    printf("   - 缺点: 列访问慢，修改困难\n\n");

    printf("3. CSC (Compressed Sparse Column) 格式:\n");
    printf("   - 类似 CSR，但按列压缩\n");
    printf("   - 优点: 列访问快\n\n");

    // 示例矩阵
    printf("示例矩阵 (4×4):\n");
    printf("  ┌                   ┐\n");
    printf("  │  1  0  2  0       │\n");
    printf("  │  0  3  0  4       │\n");
    printf("  │  5  0  6  0       │\n");
    printf("  │  0  0  0  7       │\n");
    printf("  └                   ┘\n\n");

    printf("CSR 表示:\n");
    printf("  rowPtr  = [0, 2, 4, 6, 7]\n");
    printf("  colInd  = [0, 2, 1, 3, 0, 2, 3]\n");
    printf("  values  = [1, 2, 3, 4, 5, 6, 7]\n\n");

    printf("COO 表示:\n");
    printf("  rows    = [0, 0, 1, 1, 2, 2, 3]\n");
    printf("  cols    = [0, 2, 1, 3, 0, 2, 3]\n");
    printf("  values  = [1, 2, 3, 4, 5, 6, 7]\n\n");
}

// ============================================================================
// 第二部分：创建稀疏矩阵
// ============================================================================

void demoCreateSparse() {
    printf("=== 第二部分：创建稀疏矩阵 ===\n\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // 定义一个 5×5 稀疏矩阵
    const int num_rows = 5;
    const int num_cols = 5;
    const int nnz = 9;  // 非零元素数量

    // CSR 格式数据
    int h_rowPtr[] = {0, 2, 4, 6, 8, 9};  // 6 elements (num_rows + 1)
    int h_colInd[] = {0, 1, 1, 2, 2, 3, 3, 4, 4};  // nnz elements
    float h_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    printf("创建 %d×%d 稀疏矩阵，%d 个非零元素\n", num_rows, num_cols, nnz);
    printf("稀疏度: %.1f%%\n\n", 100.0f * nnz / (num_rows * num_cols));

    // 打印矩阵
    printf("矩阵内容:\n");
    int idx = 0;
    for (int i = 0; i < num_rows; i++) {
        printf("  ");
        for (int j = 0; j < num_cols; j++) {
            float val = 0.0f;
            if (idx < h_rowPtr[i + 1] && h_colInd[idx] == j) {
                val = h_values[idx];
                idx++;
            }
            printf("%5.1f ", val);
        }
        printf("\n");
    }
    printf("\n");

    // 分配设备内存
    int *d_rowPtr, *d_colInd;
    float *d_values;

    CHECK_CUDA(cudaMalloc(&d_rowPtr, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // 创建稀疏矩阵描述符
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
        num_rows, num_cols, nnz,
        d_rowPtr, d_colInd, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    printf("cuSPARSE 矩阵描述符已创建\n\n");

    // 内存使用对比
    size_t dense_size = num_rows * num_cols * sizeof(float);
    size_t sparse_size = (num_rows + 1) * sizeof(int) + nnz * sizeof(int) + nnz * sizeof(float);

    printf("内存使用对比:\n");
    printf("  稠密格式: %zu 字节\n", dense_size);
    printf("  稀疏格式: %zu 字节\n", sparse_size);
    printf("  节省: %.1f%%\n\n", 100.0f * (1.0f - (float)sparse_size / dense_size));

    // 清理
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

// ============================================================================
// 第三部分：SpMV (稀疏矩阵-向量乘法)
// ============================================================================

void demoSpMV() {
    printf("=== 第三部分：SpMV 稀疏矩阵-向量乘法 ===\n\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // 创建一个较大的稀疏矩阵（三对角矩阵）
    const int N = 10000;
    const int nnz = 3 * N - 2;  // 三对角矩阵的非零元素数

    printf("三对角矩阵: %d × %d, %d 非零元素\n", N, N, nnz);
    printf("稀疏度: %.4f%%\n\n", 100.0f * nnz / ((long long)N * N));

    // 分配主机内存
    int *h_rowPtr = (int*)malloc((N + 1) * sizeof(int));
    int *h_colInd = (int*)malloc(nnz * sizeof(int));
    float *h_values = (float*)malloc(nnz * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));

    // 构建三对角矩阵
    // [2 -1  0  0 ...]
    // [-1 2 -1  0 ...]
    // [0 -1  2 -1 ...]
    // ...
    int idx = 0;
    for (int i = 0; i < N; i++) {
        h_rowPtr[i] = idx;

        if (i > 0) {
            h_colInd[idx] = i - 1;
            h_values[idx] = -1.0f;
            idx++;
        }

        h_colInd[idx] = i;
        h_values[idx] = 2.0f;
        idx++;

        if (i < N - 1) {
            h_colInd[idx] = i + 1;
            h_values[idx] = -1.0f;
            idx++;
        }
    }
    h_rowPtr[N] = idx;

    // 初始化向量 x
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
    }

    // 分配设备内存
    int *d_rowPtr, *d_colInd;
    float *d_values, *d_x, *d_y;

    CHECK_CUDA(cudaMalloc(&d_rowPtr, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // 创建描述符
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
        N, N, nnz,
        d_rowPtr, d_colInd, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F));

    // 分配工作空间
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize;
    void *dBuffer;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    printf("工作空间: %zu 字节\n\n", bufferSize);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    CHECK_CUSPARSE(cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // 性能测试
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 1000; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算性能指标
    // SpMV: 2 * nnz FLOPS (一次乘法，一次加法)
    double flops = 2.0 * nnz;
    double gflops = flops / (ms / 1000 * 1e6);

    // 带宽: 读取 values + colInd + rowPtr + x + 写入 y
    double bytes = (nnz * sizeof(float) + nnz * sizeof(int) +
                    (N + 1) * sizeof(int) + N * sizeof(float) + N * sizeof(float));
    double bandwidth = bytes / (ms / 1000 * 1e6);

    printf("SpMV 性能:\n");
    printf("  时间: %.4f ms\n", ms / 1000);
    printf("  吞吐量: %.2f GFLOPS\n", gflops);
    printf("  带宽: %.2f GB/s\n\n", bandwidth);

    // 验证结果
    CHECK_CUDA(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 对于三对角矩阵和全1向量，内部元素结果应为0
    printf("结果验证:\n");
    printf("  y[0] = %.1f (期望: 1)\n", h_y[0]);
    printf("  y[1] = %.1f (期望: 0)\n", h_y[1]);
    printf("  y[%d] = %.1f (期望: 0)\n", N/2, h_y[N/2]);
    printf("  y[%d] = %.1f (期望: 1)\n\n", N-1, h_y[N-1]);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    free(h_rowPtr);
    free(h_colInd);
    free(h_values);
    free(h_x);
    free(h_y);
}

// ============================================================================
// 第四部分：SpMM (稀疏矩阵-稠密矩阵乘法)
// ============================================================================

void demoSpMM() {
    printf("=== 第四部分：SpMM 稀疏-稠密矩阵乘法 ===\n\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // 矩阵维度
    const int M = 1000;  // 稀疏矩阵行数
    const int K = 1000;  // 稀疏矩阵列数，稠密矩阵行数
    const int N = 64;    // 稠密矩阵列数

    // 创建随机稀疏矩阵（约 1% 稀疏度）
    const float sparsity = 0.01f;
    int nnz = (int)(M * K * sparsity);

    printf("SpMM: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, K, K, N, M, N);
    printf("A 的非零元素: %d (%.2f%%)\n\n", nnz, sparsity * 100);

    // 分配主机内存
    int *h_rowPtr = (int*)malloc((M + 1) * sizeof(int));
    int *h_colInd = (int*)malloc(nnz * sizeof(int));
    float *h_valuesA = (float*)malloc(nnz * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // 生成随机稀疏矩阵
    srand(42);
    int nnz_per_row = nnz / M;
    int idx = 0;
    for (int i = 0; i < M; i++) {
        h_rowPtr[i] = idx;
        for (int j = 0; j < nnz_per_row && idx < nnz; j++) {
            h_colInd[idx] = rand() % K;
            h_valuesA[idx] = (float)(rand() % 100) / 100.0f;
            idx++;
        }
    }
    h_rowPtr[M] = idx;
    nnz = idx;  // 实际非零元素数

    // 初始化稠密矩阵 B
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // 分配设备内存
    int *d_rowPtr, *d_colInd;
    float *d_valuesA, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_rowPtr, (M + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_valuesA, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_valuesA, h_valuesA, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 创建描述符
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
        M, K, nnz,
        d_rowPtr, d_colInd, d_valuesA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, K, N, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, M, N, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // 分配工作空间
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize;
    void *dBuffer;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    CHECK_CUSPARSE(cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // 性能测试
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CHECK_CUSPARSE(cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * nnz * N;
    double gflops = flops / (ms / 100 * 1e6);

    printf("SpMM 性能:\n");
    printf("  时间: %.3f ms\n", ms / 100);
    printf("  吞吐量: %.2f GFLOPS\n\n", gflops);

    // 清理
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_valuesA));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    free(h_rowPtr);
    free(h_colInd);
    free(h_valuesA);
    free(h_B);
    free(h_C);
}

// ============================================================================
// 第五部分：格式转换
// ============================================================================

void demoFormatConversion() {
    printf("=== 第五部分：格式转换 ===\n\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    const int num_rows = 4;
    const int num_cols = 4;
    const int nnz = 7;

    // COO 格式数据
    int h_cooRows[] = {0, 0, 1, 1, 2, 2, 3};
    int h_cooCols[] = {0, 2, 1, 3, 0, 2, 3};
    float h_cooValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    printf("原始 COO 格式:\n");
    printf("  rows   = ");
    for (int i = 0; i < nnz; i++) printf("%d ", h_cooRows[i]);
    printf("\n  cols   = ");
    for (int i = 0; i < nnz; i++) printf("%d ", h_cooCols[i]);
    printf("\n  values = ");
    for (int i = 0; i < nnz; i++) printf("%.0f ", h_cooValues[i]);
    printf("\n\n");

    // 分配设备内存
    int *d_cooRows, *d_cooCols, *d_csrRowPtr;
    float *d_values;

    CHECK_CUDA(cudaMalloc(&d_cooRows, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cooCols, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_csrRowPtr, (num_rows + 1) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_cooRows, h_cooRows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cooCols, h_cooCols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_cooValues, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // COO -> CSR 转换
    CHECK_CUSPARSE(cusparseXcoo2csr(handle,
        d_cooRows, nnz, num_rows,
        d_csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));

    // 获取结果
    int h_csrRowPtr[5];
    CHECK_CUDA(cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    printf("转换后 CSR 格式:\n");
    printf("  rowPtr = ");
    for (int i = 0; i <= num_rows; i++) printf("%d ", h_csrRowPtr[i]);
    printf("\n  colInd = ");
    for (int i = 0; i < nnz; i++) printf("%d ", h_cooCols[i]);  // 列索引不变
    printf("\n  values = ");
    for (int i = 0; i < nnz; i++) printf("%.0f ", h_cooValues[i]);  // 值不变
    printf("\n\n");

    printf("格式选择建议:\n");
    printf("  COO: 构建阶段、添加元素、格式转换\n");
    printf("  CSR: 行遍历、SpMV、大多数计算\n");
    printf("  CSC: 列遍历、某些求解器\n");
    printf("  BSR: 块稀疏、结构化稀疏\n\n");

    // 清理
    CHECK_CUDA(cudaFree(d_cooRows));
    CHECK_CUDA(cudaFree(d_cooCols));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

// ============================================================================
// 第六部分：实际应用 - 图的 PageRank
// ============================================================================

void demoPageRank() {
    printf("=== 第六部分：应用示例 - 简化 PageRank ===\n\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // 创建一个小型图的邻接矩阵
    // 5个节点的简单图
    const int N = 5;
    const int nnz = 8;

    // 转置的邻接矩阵（列归一化）
    int h_rowPtr[] = {0, 2, 4, 5, 7, 8};
    int h_colInd[] = {1, 2, 0, 2, 3, 0, 4, 3};
    float h_values[] = {0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 0.5f, 0.5f, 1.0f};

    printf("图结构 (5个节点):\n");
    printf("  0 <-> 1, 0 <-> 2\n");
    printf("  2 -> 3, 3 <-> 4\n\n");

    // 分配内存
    int *d_rowPtr, *d_colInd;
    float *d_values, *d_rank, *d_newRank;

    CHECK_CUDA(cudaMalloc(&d_rowPtr, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rank, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_newRank, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化 rank（均匀分布）
    float h_rank[N];
    for (int i = 0; i < N; i++) h_rank[i] = 1.0f / N;
    CHECK_CUDA(cudaMemcpy(d_rank, h_rank, N * sizeof(float), cudaMemcpyHostToDevice));

    // 创建描述符
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecRank, vecNewRank;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
        N, N, nnz,
        d_rowPtr, d_colInd, d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecRank, N, d_rank, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecNewRank, N, d_newRank, CUDA_R_32F));

    // 分配工作空间
    float alpha = 0.85f, beta = 0.0f;  // damping factor
    size_t bufferSize;
    void *dBuffer;

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecRank, &beta, vecNewRank,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // PageRank 迭代
    printf("PageRank 迭代:\n");
    for (int iter = 0; iter < 10; iter++) {
        // newRank = alpha * A * rank
        CHECK_CUSPARSE(cusparseSpMV(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecRank, &beta, vecNewRank,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        // 添加 (1-alpha)/N（简化版，实际需要处理悬挂节点）
        float teleport = (1.0f - 0.85f) / N;
        float *h_newRank = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_newRank, d_newRank, N * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) h_newRank[i] += teleport;
        CHECK_CUDA(cudaMemcpy(d_newRank, h_newRank, N * sizeof(float), cudaMemcpyHostToDevice));

        // 交换
        float *temp = d_rank;
        d_rank = d_newRank;
        d_newRank = temp;

        cusparseDnVecDescr_t tempVec = vecRank;
        vecRank = vecNewRank;
        vecNewRank = tempVec;

        CHECK_CUSPARSE(cusparseDnVecSetValues(vecRank, d_rank));
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecNewRank, d_newRank));

        if (iter % 3 == 0) {
            CHECK_CUDA(cudaMemcpy(h_rank, d_rank, N * sizeof(float), cudaMemcpyDeviceToHost));
            printf("  迭代 %d: [", iter);
            for (int i = 0; i < N; i++) printf("%.3f ", h_rank[i]);
            printf("]\n");
        }

        free(h_newRank);
    }

    CHECK_CUDA(cudaMemcpy(h_rank, d_rank, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n最终 PageRank:\n");
    for (int i = 0; i < N; i++) {
        printf("  节点 %d: %.4f\n", i, h_rank[i]);
    }
    printf("\n");

    // 清理
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecRank));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecNewRank));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_rank));
    CHECK_CUDA(cudaFree(d_newRank));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 18: cuSPARSE 稀疏矩阵运算                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n\n", prop.name);

    demoSparseBasics();
    demoCreateSparse();
    demoSpMV();
    demoSpMM();
    demoFormatConversion();
    demoPageRank();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. 稀疏矩阵格式:\n");
    printf("   ┌──────┬──────────────────────┬────────────────────┐\n");
    printf("   │ 格式 │ 存储                 │ 适用场景           │\n");
    printf("   ├──────┼──────────────────────┼────────────────────┤\n");
    printf("   │ COO  │ (row, col, val) 三元组│ 构建、格式转换     │\n");
    printf("   │ CSR  │ rowPtr + colInd + val│ 行访问、SpMV       │\n");
    printf("   │ CSC  │ colPtr + rowInd + val│ 列访问             │\n");
    printf("   │ BSR  │ 块稀疏               │ 结构化稀疏         │\n");
    printf("   └──────┴──────────────────────┴────────────────────┘\n\n");

    printf("2. 主要操作:\n");
    printf("   - cusparseSpMV(): 稀疏-向量乘法\n");
    printf("   - cusparseSpMM(): 稀疏-稠密矩阵乘法\n");
    printf("   - cusparseSpGEMM(): 稀疏-稀疏矩阵乘法\n");
    printf("   - cusparseScsrsv2_*(): 稀疏三角求解\n\n");

    printf("3. cuSPARSE 工作流:\n");
    printf("   1. cusparseCreate() 创建句柄\n");
    printf("   2. cusparseCreateCsr/Coo 创建矩阵描述符\n");
    printf("   3. cusparse*_bufferSize 获取工作空间\n");
    printf("   4. cusparse* 执行操作\n");
    printf("   5. cusparseDestroy* 清理\n\n");

    printf("4. 性能考虑:\n");
    printf("   - 稀疏度 < 10%% 时稀疏格式有优势\n");
    printf("   - SpMV 通常是内存带宽受限\n");
    printf("   - 选择合适的格式很重要\n");
    printf("   - 预分析可以提高性能\n\n");

    printf("5. 应用场景:\n");
    printf("   - 图算法 (PageRank, BFS)\n");
    printf("   - 科学计算 (有限元, 偏微分方程)\n");
    printf("   - 推荐系统\n");
    printf("   - 自然语言处理\n\n");

    printf("编译命令:\n");
    printf("  nvcc -lcusparse 18_cusparse.cu -o 18_cusparse\n\n");

    return 0;
}
