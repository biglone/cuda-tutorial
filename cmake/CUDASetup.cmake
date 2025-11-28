# ============================================================================
# CUDA 配置文件
# ============================================================================
# 自动检测 GPU 架构，设置编译选项
# ============================================================================

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 查找 cudadevrt 库（用于动态并行）
find_library(CUDADEVRT_LIBRARY cudadevrt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)
if(CUDADEVRT_LIBRARY)
    message(STATUS "cudadevrt found: ${CUDADEVRT_LIBRARY}")
else()
    message(STATUS "cudadevrt not found, dynamic parallelism examples may not build")
endif()

# ============================================================================
# GPU 架构设置
# ============================================================================
# 如果未指定架构，尝试自动检测
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 尝试运行 nvidia-smi 获取 GPU 信息
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE NVIDIA_SMI_RESULT
    )

    if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAP)
        # 解析计算能力 (例如 "8.7" -> "87")
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _ ${GPU_COMPUTE_CAP})
        set(DETECTED_ARCH "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCH})
        message(STATUS "Auto-detected GPU architecture: sm_${DETECTED_ARCH}")
    else()
        # 默认支持常见架构
        # 60: Pascal (GTX 1080, Tesla P100)
        # 70: Volta (Tesla V100)
        # 75: Turing (RTX 2080, T4)
        # 80: Ampere (RTX 3080, A100)
        # 86: Ampere (RTX 3060, 3070, 3090)
        # 87: Jetson Orin
        # 89: Ada Lovelace (RTX 4090)
        # 90: Hopper (H100)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
        message(STATUS "Using default GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# ============================================================================
# 编译选项
# ============================================================================

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Debug 模式选项
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

# Release 模式选项
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

# 启用行信息（用于性能分析）
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# ============================================================================
# 通用设置
# ============================================================================

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 导出编译命令（用于 IDE 和代码分析工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# 辅助函数
# ============================================================================

# 添加 CUDA 教程目标的辅助函数
function(add_cuda_tutorial TARGET_NAME)
    cmake_parse_arguments(TUTORIAL
        "SEPARABLE"                    # 布尔选项
        ""                             # 单值参数
        "SOURCES;LIBRARIES"            # 多值参数
        ${ARGN}
    )

    # 如果没有指定源文件，使用目标名.cu
    if(NOT TUTORIAL_SOURCES)
        set(TUTORIAL_SOURCES ${TARGET_NAME}.cu)
    endif()

    # 添加可执行文件
    add_executable(${TARGET_NAME} ${TUTORIAL_SOURCES})

    # 设置分离编译
    if(TUTORIAL_SEPARABLE)
        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudadevrt)
    endif()

    # 链接库
    if(TUTORIAL_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${TUTORIAL_LIBRARIES})
    endif()
endfunction()
