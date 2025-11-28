cmake_minimum_required(VERSION 3.18)
project(CUDA_Tutorial LANGUAGES CXX CUDA)

# ============================================================================
# CUDA 教程构建系统
# ============================================================================
# 使用方法:
#   mkdir build && cd build
#   cmake ..
#   make                    # 编译全部
#   make hello_cuda         # 编译单个教程
#   make basics             # 编译基础篇 (01-04)
#   make advanced           # 编译进阶篇 (05-10)
# ============================================================================

# 包含 CUDA 配置
include(cmake/CUDASetup.cmake)

# ============================================================================
# 基础篇 (01-04) - 无额外依赖
# ============================================================================
set(BASICS
    hello_cuda
    kernel_basics
    memory_management
    vector_add
)

foreach(target ${BASICS})
    add_executable(${target} ${target}.cu)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endforeach()

# 创建基础篇分组目标
add_custom_target(basics DEPENDS ${BASICS})

# ============================================================================
# 进阶篇 (05-07) - 无额外依赖
# ============================================================================
set(INTERMEDIATE
    05_shared_memory
    06_sync_atomic
    07_cuda_streams
)

foreach(target ${INTERMEDIATE})
    add_executable(${target} ${target}.cu)
endforeach()

# ============================================================================
# 内存优化篇 (08-10) - 无额外依赖
# ============================================================================
set(MEMORY_OPT
    08_unified_memory
    09_texture_memory
    10_constant_reduction
)

foreach(target ${MEMORY_OPT})
    add_executable(${target} ${target}.cu)
endforeach()

# 创建进阶篇分组目标 (05-10)
add_custom_target(advanced DEPENDS ${INTERMEDIATE} ${MEMORY_OPT})

# ============================================================================
# 实战篇 (11-15) - 部分需要额外库
# ============================================================================

# 11: 矩阵乘法 - 需要 cuBLAS
add_executable(11_matrix_multiply 11_matrix_multiply.cu)
target_link_libraries(11_matrix_multiply PRIVATE CUDA::cublas)

# 12: 性能分析 - 无额外依赖
add_executable(12_profiling_debug 12_profiling_debug.cu)

# 13: 动态并行 - 需要分离编译
if(CUDADEVRT_LIBRARY)
    add_executable(13_dynamic_parallelism 13_dynamic_parallelism.cu)
    set_target_properties(13_dynamic_parallelism PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(13_dynamic_parallelism PRIVATE ${CUDADEVRT_LIBRARY})
else()
    message(STATUS "Skipping 13_dynamic_parallelism (cudadevrt not found)")
endif()

# 14: 多 GPU - 无额外依赖
add_executable(14_multi_gpu 14_multi_gpu.cu)

# 15: Thrust - 无额外依赖 (Thrust 是 header-only)
add_executable(15_thrust_practical 15_thrust_practical.cu)

# 实战篇分组目标
set(PRACTICAL_TARGETS
    11_matrix_multiply
    12_profiling_debug
    14_multi_gpu
    15_thrust_practical
)
if(CUDADEVRT_LIBRARY)
    list(APPEND PRACTICAL_TARGETS 13_dynamic_parallelism)
endif()
add_custom_target(practical DEPENDS ${PRACTICAL_TARGETS})

# ============================================================================
# 库应用篇 (16-20) - 需要各种 CUDA 库
# ============================================================================

# 16: cuDNN (可选，不是所有系统都有)
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(CUDNN_LIBRARY)
    add_executable(16_cudnn_deeplearning 16_cudnn_deeplearning.cu)
    target_link_libraries(16_cudnn_deeplearning PRIVATE ${CUDNN_LIBRARY})
    message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
else()
    message(STATUS "cuDNN not found, skipping 16_cudnn_deeplearning")
endif()

# 17: cuFFT
add_executable(17_cufft 17_cufft.cu)
target_link_libraries(17_cufft PRIVATE CUDA::cufft)

# 18: cuSPARSE
add_executable(18_cusparse 18_cusparse.cu)
target_link_libraries(18_cusparse PRIVATE CUDA::cusparse)

# 19: cuRAND
add_executable(19_curand 19_curand.cu)
target_link_libraries(19_curand PRIVATE CUDA::curand)

# 20: CUDA Graphs - 无额外依赖
add_executable(20_cuda_graphs 20_cuda_graphs.cu)

add_custom_target(libraries DEPENDS
    17_cufft
    18_cusparse
    19_curand
    20_cuda_graphs
)

# ============================================================================
# 高级篇 (21-25)
# ============================================================================

# 21: 图形互操作 (可选，需要 OpenGL)
find_package(OpenGL QUIET)
find_package(GLUT QUIET)
if(OpenGL_FOUND AND GLUT_FOUND)
    add_executable(21_interop_graphics 21_interop_graphics.cu)
    target_link_libraries(21_interop_graphics PRIVATE
        OpenGL::GL
        OpenGL::GLU
        GLUT::GLUT
    )
    message(STATUS "OpenGL/GLUT found, building 21_interop_graphics")
else()
    message(STATUS "OpenGL/GLUT not found, skipping 21_interop_graphics")
endif()

# 22: 内存池 - 无额外依赖
add_executable(22_memory_pools 22_memory_pools.cu)

# 23: 协作组高级 - 需要分离编译
if(CUDADEVRT_LIBRARY)
    add_executable(23_cooperative_groups_advanced 23_cooperative_groups_advanced.cu)
    set_target_properties(23_cooperative_groups_advanced PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(23_cooperative_groups_advanced PRIVATE ${CUDADEVRT_LIBRARY})
else()
    message(STATUS "Skipping 23_cooperative_groups_advanced (cudadevrt not found)")
endif()

# 24: 优化实战 - 无额外依赖
add_executable(24_optimization_workshop 24_optimization_workshop.cu)

# 25: 深度学习集成 - 无额外依赖
add_executable(25_deep_learning_integration 25_deep_learning_integration.cu)

# 高级篇分组目标
set(HIGH_LEVEL_TARGETS
    22_memory_pools
    24_optimization_workshop
    25_deep_learning_integration
)
if(CUDADEVRT_LIBRARY)
    list(APPEND HIGH_LEVEL_TARGETS 23_cooperative_groups_advanced)
endif()
add_custom_target(high_level DEPENDS ${HIGH_LEVEL_TARGETS})

# ============================================================================
# 专题篇 (26-30)
# ============================================================================

# 26: PTX 内联汇编 - 无额外依赖
add_executable(26_ptx_inline_assembly 26_ptx_inline_assembly.cu)

# 27: Tensor Core - 无额外依赖
add_executable(27_warp_matrix_tensor_cores 27_warp_matrix_tensor_cores.cu)

# 28: 异步拷贝 - 无额外依赖
add_executable(28_async_copy_pipeline 28_async_copy_pipeline.cu)

# 29: 调��技巧 - 无额外依赖
add_executable(29_debugging_best_practices 29_debugging_best_practices.cu)

# 30: 图像处理 - 无额外依赖
add_executable(30_image_processing_project 30_image_processing_project.cu)

add_custom_target(special_topics DEPENDS
    26_ptx_inline_assembly
    27_warp_matrix_tensor_cores
    28_async_copy_pipeline
    29_debugging_best_practices
    30_image_processing_project
)

# ============================================================================
# 前沿应用篇 (31-35)
# ============================================================================

# 31: 神经网络推理 - 无额外依赖
add_executable(31_neural_network_inference 31_neural_network_inference.cu)

# 32: 实时视频处理 - 需要 cuFFT
add_executable(32_realtime_video_processing 32_realtime_video_processing.cu)
target_link_libraries(32_realtime_video_processing PRIVATE CUDA::cufft)

# 33: 科学计算 - 需要 cuRAND
add_executable(33_scientific_computing 33_scientific_computing.cu)
target_link_libraries(33_scientific_computing PRIVATE CUDA::curand)

# 34: Jetson 嵌入式 - 无额外依赖
add_executable(34_jetson_embedded 34_jetson_embedded.cu)

# 35: HPC 前沿 - 无额外依赖
add_executable(35_hpc_future 35_hpc_future.cu)

add_custom_target(frontier DEPENDS
    31_neural_network_inference
    32_realtime_video_processing
    33_scientific_computing
    34_jetson_embedded
    35_hpc_future
)

# ============================================================================
# 全部编译目标
# ============================================================================
add_custom_target(all_tutorials DEPENDS
    basics
    advanced
    practical
    libraries
    high_level
    special_topics
    frontier
)

# ============================================================================
# 打印配置信息
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "CUDA Tutorial Build Configuration")
message(STATUS "========================================")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  make basics        - 基础篇 (01-04)")
message(STATUS "  make advanced      - 进阶篇 (05-10)")
message(STATUS "  make practical     - 实战篇 (11-15)")
message(STATUS "  make libraries     - 库应用篇 (16-20)")
message(STATUS "  make high_level    - 高级篇 (21-25)")
message(STATUS "  make special_topics - 专题篇 (26-30)")
message(STATUS "  make frontier      - 前沿应用篇 (31-35)")
message(STATUS "  make all_tutorials - 全部教程")
message(STATUS "========================================")
message(STATUS "")
