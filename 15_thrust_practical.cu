/**
 * =============================================================================
 * CUDA 教程 15: Thrust 库与实战案例
 * =============================================================================
 *
 * 学习目标：
 * 1. 了解 Thrust 库的基本概念和使用方法
 * 2. 掌握 Thrust 向量、算法和迭代器
 * 3. 学习实际应用案例：直方图、排序、稀疏矩阵
 * 4. 了解 Thrust 与原生 CUDA 的结合
 *
 * 关键概念：
 * - Thrust 是 CUDA 的高级 C++ 模板库
 * - 提供类似 STL 的容器和算法
 * - 自动管理内存和执行策略
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// ============================================================================
// 第一部分：Thrust 基础
// ============================================================================

void demoThrustBasics() {
    printf("=== 第一部分：Thrust 基础 ===\n\n");

    // 1. 向量创建和初始化
    printf("1. 向量操作:\n");

    // 主机向量
    thrust::host_vector<int> h_vec(10);
    thrust::sequence(h_vec.begin(), h_vec.end());  // 0, 1, 2, ..., 9

    printf("   主机向量: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_vec[i]);
    printf("\n");

    // 设备向量（自动复制）
    thrust::device_vector<int> d_vec = h_vec;

    // 填充
    thrust::fill(d_vec.begin(), d_vec.begin() + 5, 100);

    // 复制回主机
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    printf("   填充后: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_vec[i]);
    printf("\n\n");

    // 2. 变换操作
    printf("2. 变换操作:\n");

    thrust::device_vector<float> d_x(5), d_y(5), d_z(5);
    thrust::sequence(d_x.begin(), d_x.end(), 1.0f);  // 1, 2, 3, 4, 5
    thrust::fill(d_y.begin(), d_y.end(), 2.0f);

    // z = x + y
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(),
                      d_z.begin(), thrust::plus<float>());

    thrust::host_vector<float> h_z = d_z;
    printf("   x + y = ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_z[i]);
    printf("\n");

    // 使用 lambda（C++11）或函数对象
    struct square {
        __host__ __device__
        float operator()(float x) { return x * x; }
    };

    thrust::transform(d_x.begin(), d_x.end(), d_z.begin(), square());
    h_z = d_z;
    printf("   x^2 = ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_z[i]);
    printf("\n\n");

    // 3. 归约操作
    printf("3. 归约操作:\n");

    thrust::device_vector<int> d_data(1000);
    thrust::fill(d_data.begin(), d_data.end(), 1);

    int sum = thrust::reduce(d_data.begin(), d_data.end(), 0, thrust::plus<int>());
    printf("   sum(1000个1) = %d\n", sum);

    thrust::sequence(d_data.begin(), d_data.end(), 1);
    int maxVal = thrust::reduce(d_data.begin(), d_data.end(), 0, thrust::maximum<int>());
    printf("   max(1..1000) = %d\n\n", maxVal);
}

// ============================================================================
// 第二部分：高级迭代器
// ============================================================================

void demoAdvancedIterators() {
    printf("=== 第二部分：高级迭代器 ===\n\n");

    // 1. counting_iterator - 虚拟计数序列
    printf("1. counting_iterator:\n");
    {
        // 计算 0 + 1 + 2 + ... + 99
        int sum = thrust::reduce(thrust::counting_iterator<int>(0),
                                  thrust::counting_iterator<int>(100));
        printf("   sum(0..99) = %d (期望: %d)\n\n", sum, 99 * 100 / 2);
    }

    // 2. transform_iterator - 即时变换
    printf("2. transform_iterator:\n");
    {
        struct square {
            __host__ __device__
            int operator()(int x) { return x * x; }
        };

        // 计算 1^2 + 2^2 + ... + 10^2 而不分配额外内存
        auto begin = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(1), square());
        auto end = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(11), square());

        int sumSquares = thrust::reduce(begin, end);
        printf("   sum(1^2..10^2) = %d (期望: 385)\n\n", sumSquares);
    }

    // 3. zip_iterator - 组合多个序列
    printf("3. zip_iterator:\n");
    {
        thrust::device_vector<int> keys(5);
        thrust::device_vector<float> values(5);

        thrust::sequence(keys.begin(), keys.end());
        thrust::fill(values.begin(), values.end(), 1.5f);

        // 创建 zip 迭代器
        auto zip_begin = thrust::make_zip_iterator(
            thrust::make_tuple(keys.begin(), values.begin()));

        // 使用 tuple 访问
        thrust::tuple<int, float> first = *zip_begin;
        printf("   first tuple: (%d, %.1f)\n\n",
               thrust::get<0>(first), thrust::get<1>(first));
    }

    // 4. permutation_iterator - 重排访问
    printf("4. permutation_iterator:\n");
    {
        thrust::device_vector<float> data(4);
        data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;

        thrust::device_vector<int> indices(4);
        indices[0] = 3; indices[1] = 0; indices[2] = 2; indices[3] = 1;

        // 按索引顺序访问
        auto perm_begin = thrust::make_permutation_iterator(
            data.begin(), indices.begin());

        thrust::host_vector<float> result(4);
        thrust::copy(perm_begin, perm_begin + 4, result.begin());

        printf("   原始: 10 20 30 40\n");
        printf("   索引: 3 0 2 1\n");
        printf("   结果: ");
        for (int i = 0; i < 4; i++) printf("%.0f ", result[i]);
        printf("\n\n");
    }
}

// ============================================================================
// 第三部分：排序和搜索
// ============================================================================

void demoSortingSearching() {
    printf("=== 第三部分：排序和搜索 ===\n\n");

    const int N = 10;

    // 1. 基本排序
    printf("1. 基本排序:\n");
    {
        thrust::device_vector<int> d_vec(N);
        thrust::host_vector<int> h_vec(N);

        // 随机初始化
        for (int i = 0; i < N; i++) h_vec[i] = rand() % 100;
        d_vec = h_vec;

        printf("   排序前: ");
        for (int i = 0; i < N; i++) printf("%d ", h_vec[i]);
        printf("\n");

        thrust::sort(d_vec.begin(), d_vec.end());
        h_vec = d_vec;

        printf("   排序后: ");
        for (int i = 0; i < N; i++) printf("%d ", h_vec[i]);
        printf("\n\n");
    }

    // 2. 键值对排序
    printf("2. 键值对排序:\n");
    {
        thrust::device_vector<int> keys(N);
        thrust::device_vector<char> values(N);

        thrust::host_vector<int> h_keys(N);
        thrust::host_vector<char> h_values(N);

        for (int i = 0; i < N; i++) {
            h_keys[i] = rand() % 100;
            h_values[i] = 'A' + (i % 26);
        }
        keys = h_keys;
        values = h_values;

        printf("   排序前 keys: ");
        for (int i = 0; i < N; i++) printf("%d ", h_keys[i]);
        printf("\n   排序前 vals: ");
        for (int i = 0; i < N; i++) printf("%c ", h_values[i]);
        printf("\n");

        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
        h_keys = keys;
        h_values = values;

        printf("   排序后 keys: ");
        for (int i = 0; i < N; i++) printf("%d ", h_keys[i]);
        printf("\n   排序后 vals: ");
        for (int i = 0; i < N; i++) printf("%c ", h_values[i]);
        printf("\n\n");
    }

    // 3. 去重
    printf("3. 去重操作:\n");
    {
        thrust::device_vector<int> d_vec(10);
        int data[] = {1, 1, 2, 2, 2, 3, 3, 4, 4, 4};
        thrust::copy(data, data + 10, d_vec.begin());

        // 需要先排序
        auto new_end = thrust::unique(d_vec.begin(), d_vec.end());
        int new_size = new_end - d_vec.begin();

        thrust::host_vector<int> h_vec(d_vec.begin(), new_end);
        printf("   原始: 1 1 2 2 2 3 3 4 4 4\n");
        printf("   去重后: ");
        for (int i = 0; i < new_size; i++) printf("%d ", h_vec[i]);
        printf("\n\n");
    }
}

// ============================================================================
// 第四部分：扫描操作
// ============================================================================

void demoScanOperations() {
    printf("=== 第四部分：扫描操作 ===\n\n");

    const int N = 8;
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8};

    thrust::device_vector<int> d_input(data, data + N);
    thrust::device_vector<int> d_output(N);

    // 1. inclusive_scan (前缀和)
    printf("1. inclusive_scan:\n");
    {
        thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
        thrust::host_vector<int> h_output = d_output;

        printf("   输入: ");
        for (int i = 0; i < N; i++) printf("%d ", data[i]);
        printf("\n   输出: ");
        for (int i = 0; i < N; i++) printf("%d ", h_output[i]);
        printf("\n   (每个位置是之前所有元素的和，包括自己)\n\n");
    }

    // 2. exclusive_scan
    printf("2. exclusive_scan:\n");
    {
        thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
        thrust::host_vector<int> h_output = d_output;

        printf("   输入: ");
        for (int i = 0; i < N; i++) printf("%d ", data[i]);
        printf("\n   输出: ");
        for (int i = 0; i < N; i++) printf("%d ", h_output[i]);
        printf("\n   (每个位置是之前所有元素的和，不包括自己)\n\n");
    }

    // 3. 分段扫描（按键）
    printf("3. 分段扫描:\n");
    {
        int keys_data[] = {0, 0, 0, 1, 1, 2, 2, 2};
        thrust::device_vector<int> d_keys(keys_data, keys_data + N);
        thrust::device_vector<int> d_vals(data, data + N);

        thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(),
                                       d_vals.begin(), d_output.begin());
        thrust::host_vector<int> h_output = d_output;

        printf("   键:   ");
        for (int i = 0; i < N; i++) printf("%d ", keys_data[i]);
        printf("\n   值:   ");
        for (int i = 0; i < N; i++) printf("%d ", data[i]);
        printf("\n   输出: ");
        for (int i = 0; i < N; i++) printf("%d ", h_output[i]);
        printf("\n   (每个键段内独立求和)\n\n");
    }
}

// ============================================================================
// 第五部分：实战案例 - 直方图计算
// ============================================================================

void demoHistogram() {
    printf("=== 第五部分：直方图计算 ===\n\n");

    const int N = 100000;
    const int NUM_BINS = 10;

    // 生成随机数据 [0, NUM_BINS)
    thrust::device_vector<int> d_data(N);
    thrust::host_vector<int> h_data(N);

    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % NUM_BINS;
    }
    d_data = h_data;

    // 方法1：排序 + 归约按键
    printf("方法1: 排序 + reduce_by_key\n");
    {
        thrust::device_vector<int> d_sorted = d_data;
        thrust::sort(d_sorted.begin(), d_sorted.end());

        thrust::device_vector<int> d_bins(NUM_BINS);
        thrust::device_vector<int> d_counts(NUM_BINS);
        thrust::device_vector<int> d_ones(N, 1);

        auto new_end = thrust::reduce_by_key(
            d_sorted.begin(), d_sorted.end(),
            d_ones.begin(),
            d_bins.begin(),
            d_counts.begin()
        );

        int num_unique = new_end.first - d_bins.begin();

        thrust::host_vector<int> h_bins(d_bins.begin(), d_bins.begin() + num_unique);
        thrust::host_vector<int> h_counts(d_counts.begin(), d_counts.begin() + num_unique);

        printf("   ");
        for (int i = 0; i < num_unique; i++) {
            printf("bin[%d]=%d ", h_bins[i], h_counts[i]);
        }
        printf("\n\n");
    }

    // 方法2：使用 counting
    printf("方法2: 直接计数\n");
    {
        thrust::host_vector<int> h_histogram(NUM_BINS);
        for (int i = 0; i < NUM_BINS; i++) {
            h_histogram[i] = thrust::count(d_data.begin(), d_data.end(), i);
        }

        printf("   ");
        for (int i = 0; i < NUM_BINS; i++) {
            printf("bin[%d]=%d ", i, h_histogram[i]);
        }
        printf("\n\n");
    }
}

// ============================================================================
// 第六部分：实战案例 - 向量范数计算
// ============================================================================

void demoVectorNorm() {
    printf("=== 第六部分：向量范数计算 ===\n\n");

    const int N = 1000000;

    thrust::device_vector<float> d_vec(N);
    thrust::host_vector<float> h_vec(N);

    // 初始化
    for (int i = 0; i < N; i++) {
        h_vec[i] = (float)(i + 1) / N;
    }
    d_vec = h_vec;

    // L1 范数: sum(|x|)
    printf("1. L1 范数 (sum of absolute values):\n");
    {
        struct abs_value {
            __host__ __device__
            float operator()(float x) { return fabsf(x); }
        };

        float l1_norm = thrust::transform_reduce(
            d_vec.begin(), d_vec.end(),
            abs_value(),
            0.0f,
            thrust::plus<float>()
        );
        printf("   L1 = %.6f\n\n", l1_norm);
    }

    // L2 范数: sqrt(sum(x^2))
    printf("2. L2 范数 (Euclidean norm):\n");
    {
        struct square {
            __host__ __device__
            float operator()(float x) { return x * x; }
        };

        float sum_squares = thrust::transform_reduce(
            d_vec.begin(), d_vec.end(),
            square(),
            0.0f,
            thrust::plus<float>()
        );
        float l2_norm = sqrtf(sum_squares);
        printf("   L2 = %.6f\n\n", l2_norm);
    }

    // L_inf 范数: max(|x|)
    printf("3. L_inf 范数 (maximum absolute value):\n");
    {
        struct abs_value {
            __host__ __device__
            float operator()(float x) { return fabsf(x); }
        };

        float linf_norm = thrust::transform_reduce(
            d_vec.begin(), d_vec.end(),
            abs_value(),
            0.0f,
            thrust::maximum<float>()
        );
        printf("   L_inf = %.6f\n\n", linf_norm);
    }
}

// ============================================================================
// 第七部分：Thrust 与原生 CUDA 结合
// ============================================================================

// 原生 CUDA 内核
__global__ void customKernel(float *data, int n, float scale) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * scale + sinf(data[tid]);
    }
}

void demoThrustWithCUDA() {
    printf("=== 第七部分：Thrust 与原生 CUDA ===\n\n");

    const int N = 10000;

    thrust::device_vector<float> d_vec(N);
    thrust::sequence(d_vec.begin(), d_vec.end(), 0.0f, 0.01f);

    printf("1. 获取原始指针:\n");
    {
        // 获取 device_vector 的原始指针
        float *raw_ptr = thrust::raw_pointer_cast(d_vec.data());

        // 使用原生 CUDA 内核
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        customKernel<<<gridSize, blockSize>>>(raw_ptr, N, 2.0f);
        CHECK_CUDA(cudaDeviceSynchronize());

        printf("   thrust::raw_pointer_cast() 获取原始指针\n");
        printf("   可传递给原生 CUDA 内核\n\n");
    }

    printf("2. 执行策略:\n");
    {
        thrust::host_vector<int> h_vec(1000);
        thrust::sequence(h_vec.begin(), h_vec.end());

        // 强制在 CPU 执行
        int cpu_sum = thrust::reduce(thrust::host, h_vec.begin(), h_vec.end());

        thrust::device_vector<int> d_vec2 = h_vec;

        // 强制在 GPU 执行
        int gpu_sum = thrust::reduce(thrust::device, d_vec2.begin(), d_vec2.end());

        printf("   thrust::host - 强制 CPU 执行\n");
        printf("   thrust::device - 强制 GPU 执行\n");
        printf("   CPU sum = %d, GPU sum = %d\n\n", cpu_sum, gpu_sum);
    }

    printf("3. 使用 CUDA 流:\n");
    {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

        thrust::device_vector<int> d_vec3(10000, 1);

        // 在指定流上执行
        int sum = thrust::reduce(thrust::cuda::par.on(stream),
                                  d_vec3.begin(), d_vec3.end());

        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("   thrust::cuda::par.on(stream) 指定流\n");
        printf("   结果: %d\n\n", sum);

        CHECK_CUDA(cudaStreamDestroy(stream));
    }
}

// ============================================================================
// 第八部分：性能比较
// ============================================================================

void demoPerformance() {
    printf("=== 第八部分：性能比较 ===\n\n");

    const int N = 1 << 24;  // 16M

    thrust::device_vector<float> d_vec(N);
    thrust::sequence(d_vec.begin(), d_vec.end());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    thrust::reduce(d_vec.begin(), d_vec.end());

    // 测试归约
    printf("1. 归约性能 (%d 元素):\n", N);
    {
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            thrust::reduce(d_vec.begin(), d_vec.end());
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float bandwidth = N * sizeof(float) / (ms / 100 * 1e6);
        printf("   Thrust reduce: %.3f ms, %.1f GB/s\n", ms / 100, bandwidth);
    }

    // 测试排序
    printf("\n2. 排序性能 (%d 元素):\n", N);
    {
        thrust::device_vector<float> d_sort = d_vec;

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 10; i++) {
            thrust::copy(d_vec.begin(), d_vec.end(), d_sort.begin());
            thrust::sort(d_sort.begin(), d_sort.end());
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("   Thrust sort: %.3f ms\n", ms / 10);
    }

    // 测试变换
    printf("\n3. 变换性能 (%d 元素):\n", N);
    {
        thrust::device_vector<float> d_out(N);

        struct scale_add {
            __host__ __device__
            float operator()(float x) { return x * 2.0f + 1.0f; }
        };

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < 100; i++) {
            thrust::transform(d_vec.begin(), d_vec.end(), d_out.begin(), scale_add());
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float bandwidth = 2 * N * sizeof(float) / (ms / 100 * 1e6);  // 读+写
        printf("   Thrust transform: %.3f ms, %.1f GB/s\n\n", ms / 100, bandwidth);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     CUDA 教程 15: Thrust 库与实战案例                       ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s\n\n", prop.name);

    demoThrustBasics();
    demoAdvancedIterators();
    demoSortingSearching();
    demoScanOperations();
    demoHistogram();
    demoVectorNorm();
    demoThrustWithCUDA();
    demoPerformance();

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                    学习要点总结                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("1. Thrust 容器:\n");
    printf("   - host_vector<T>: CPU 内存\n");
    printf("   - device_vector<T>: GPU 内存\n");
    printf("   - 自动内存管理和传输\n\n");

    printf("2. 常用算法:\n");
    printf("   ┌────────────────┬────────────────────────────────┐\n");
    printf("   │ 算法           │ 功能                           │\n");
    printf("   ├────────────────┼────────────────────────────────┤\n");
    printf("   │ transform      │ 元素级变换                     │\n");
    printf("   │ reduce         │ 归约（求和、最大等）           │\n");
    printf("   │ sort           │ 排序                           │\n");
    printf("   │ scan           │ 前缀和/扫描                    │\n");
    printf("   │ copy           │ 数据复制                       │\n");
    printf("   │ fill           │ 填充固定值                     │\n");
    printf("   │ count          │ 计数                           │\n");
    printf("   │ unique         │ 去重                           │\n");
    printf("   └────────────────┴────────────────────────────────┘\n\n");

    printf("3. 高级迭代器:\n");
    printf("   - counting_iterator: 虚拟计数序列\n");
    printf("   - transform_iterator: 即时变换\n");
    printf("   - zip_iterator: 组合多个序列\n");
    printf("   - permutation_iterator: 重排访问\n\n");

    printf("4. 与原生 CUDA 结合:\n");
    printf("   - raw_pointer_cast(): 获取原始指针\n");
    printf("   - thrust::cuda::par.on(stream): 指定流\n");
    printf("   - thrust::host/device: 指定执行位置\n\n");

    printf("5. 最佳实践:\n");
    printf("   - 简单操作使用 Thrust\n");
    printf("   - 复杂内核使用原生 CUDA\n");
    printf("   - 避免频繁的 host/device 数据传输\n");
    printf("   - 使用 fancy iterators 减少内存分配\n\n");

    printf("6. Thrust 优势:\n");
    printf("   - 生产力高，代码简洁\n");
    printf("   - 自动选择最优实现\n");
    printf("   - 跨平台（CUDA, OpenMP, TBB）\n");
    printf("   - 与 STL 接口一致\n\n");

    return 0;
}
