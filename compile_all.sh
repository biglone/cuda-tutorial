#!/bin/bash

# =============================================================================
# CUDA 教程全量编译测试脚本
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 计数器
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# 创建输出目录
BUILD_DIR="build_test"
mkdir -p "$BUILD_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           CUDA 教程全量编译测试                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# 定义编译规则（文件名 -> 编译选项）
declare -A COMPILE_RULES=(
    # 基础篇
    ["hello_cuda.cu"]="nvcc"
    ["kernel_basics.cu"]="nvcc"
    ["memory_management.cu"]="nvcc"
    ["vector_add.cu"]="nvcc"

    # 进阶篇
    ["05_shared_memory.cu"]="nvcc"
    ["06_sync_atomic.cu"]="nvcc"
    ["07_cuda_streams.cu"]="nvcc"

    # 内存优化篇
    ["08_unified_memory.cu"]="nvcc"
    ["09_texture_memory.cu"]="nvcc"
    ["10_constant_reduction.cu"]="nvcc"

    # 实战篇
    ["11_matrix_multiply.cu"]="nvcc -lcublas"
    ["12_profiling_debug.cu"]="nvcc"
    ["13_dynamic_parallelism.cu"]="nvcc -rdc=true -lcudadevrt"
    ["14_multi_gpu.cu"]="nvcc"
    ["15_thrust_practical.cu"]="nvcc"

    # 库应用篇
    ["16_cudnn_deeplearning.cu"]="nvcc -lcudnn"
    ["17_cufft.cu"]="nvcc -lcufft"
    ["18_cusparse.cu"]="nvcc -lcusparse"
    ["19_curand.cu"]="nvcc -lcurand"
    ["20_cuda_graphs.cu"]="nvcc"

    # 高级篇
    ["21_interop_graphics.cu"]="nvcc -lGL -lGLU -lglut"
    ["22_memory_pools.cu"]="nvcc -lcuda"
    ["23_cooperative_groups_advanced.cu"]="nvcc -rdc=true -lcudadevrt"
    ["24_optimization_workshop.cu"]="nvcc"
    ["25_deep_learning_integration.cu"]="nvcc"

    # 专题篇
    ["26_ptx_inline_assembly.cu"]="nvcc"
    ["27_warp_matrix_tensor_cores.cu"]="nvcc"
    ["28_async_copy_pipeline.cu"]="nvcc -arch=sm_80"
    ["29_debugging_best_practices.cu"]="nvcc"
    ["30_image_processing_project.cu"]="nvcc"

    # 前沿应用篇
    ["31_neural_network_inference.cu"]="nvcc"
    ["32_realtime_video_processing.cu"]="nvcc -lcufft"
    ["33_scientific_computing.cu"]="nvcc -lcurand"
    ["34_jetson_embedded.cu"]="nvcc"
    ["35_hpc_future.cu"]="nvcc"
)

# 编译函数
compile_file() {
    local file=$1
    local compile_cmd=$2
    local output="$BUILD_DIR/${file%.cu}"

    TOTAL=$((TOTAL + 1))

    echo -e "${YELLOW}[${TOTAL}/${#COMPILE_RULES[@]}]${NC} 编译: ${BLUE}$file${NC}"

    if [[ ! -f "$file" ]]; then
        echo -e "   ${RED}✗ 文件不存在${NC}"
        SKIPPED=$((SKIPPED + 1))
        echo
        return 1
    fi

    # 执行编译
    local full_cmd="$compile_cmd $file -o $output -I. 2>&1"
    local compile_output

    if compile_output=$(eval $full_cmd); then
        echo -e "   ${GREEN}✓ 编译成功${NC}"
        SUCCESS=$((SUCCESS + 1))

        # 检查生成的二进制文件
        if [[ -f "$output" ]]; then
            local size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null)
            echo -e "   输出: $output ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size} bytes"))"
        fi
        echo
        return 0
    else
        echo -e "   ${RED}✗ 编译失败${NC}"
        FAILED=$((FAILED + 1))

        # 保存错误信息
        echo "$compile_output" > "$BUILD_DIR/${file%.cu}.error.log"
        echo -e "   错误日志: $BUILD_DIR/${file%.cu}.error.log"

        # 显示前几行错误
        echo -e "${RED}   错误信息:${NC}"
        echo "$compile_output" | head -5 | sed 's/^/     /'
        if [[ $(echo "$compile_output" | wc -l) -gt 5 ]]; then
            echo "     ..."
        fi
        echo
        return 1
    fi
}

# 开始编译
echo -e "${BLUE}开始编译测试...${NC}"
echo -e "${BLUE}输出目录: $BUILD_DIR${NC}"
echo
echo "================================================================================"
echo

# 按顺序编译所有文件
for file in "${!COMPILE_RULES[@]}"; do
    compile_file "$file" "${COMPILE_RULES[$file]}" || true
done

# 统计信息
echo "================================================================================"
echo
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    编译测试完成                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "总文件数:     ${BLUE}$TOTAL${NC}"
echo -e "编译成功:     ${GREEN}$SUCCESS${NC}"
echo -e "编译失败:     ${RED}$FAILED${NC}"
echo -e "跳过/不存在:  ${YELLOW}$SKIPPED${NC}"
echo

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}✓ 所有文件编译成功！${NC}"
    echo
    exit 0
else
    echo -e "${RED}✗ 有 $FAILED 个文件编译失败${NC}"
    echo
    echo "失败的文件错误日志位于: $BUILD_DIR/*.error.log"
    echo
    exit 1
fi
