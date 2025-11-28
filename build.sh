#!/bin/bash
# =============================================================================
# CUDA 教程构建脚本
# =============================================================================
# 用法:
#   ./build.sh              # 编译全部教程
#   ./build.sh basics       # 编译基础篇
#   ./build.sh clean        # 清理构建目录
#   ./build.sh help         # 显示帮助信息
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# 打印带颜色的消息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 显示帮助信息
show_help() {
    cat << EOF
CUDA 教程构建脚本

用法: ./build.sh [命令] [选项]

命令:
  (无参数)        编译全部教程
  basics          编译基础篇 (01-04)
  advanced        编译进阶篇 (05-10)
  practical       编译实战篇 (11-15)
  libraries       编译库应用篇 (16-20)
  high_level      编译高级篇 (21-25)
  special_topics  编译专题篇 (26-30)
  frontier        编译前沿应用篇 (31-35)
  <target_name>   编译单个教程 (如: hello_cuda, 08_unified_memory)
  clean           清理构建目录
  rebuild         清理后重新编译
  list            列出所有可用目标
  run <target>    编译并运行指定程序
  help            显示此帮助信息

选项:
  -j <N>          并行编译任务数 (默认: 自动检测)
  -v              详细输出模式

示例:
  ./build.sh                    # 编译全部
  ./build.sh basics             # 编译基础篇
  ./build.sh hello_cuda         # 编译单个程序
  ./build.sh run hello_cuda     # 编译并运行
  ./build.sh clean              # 清理构建

EOF
}

# 列出所有目标
list_targets() {
    cat << EOF
可用的编译目标:

章节目标:
  basics          基础篇 (hello_cuda, kernel_basics, memory_management, vector_add)
  advanced        进阶篇 (05_shared_memory ~ 10_constant_reduction)
  practical       实战篇 (11_matrix_multiply ~ 15_thrust_practical)
  libraries       库应用篇 (16_cudnn ~ 20_cuda_graphs)
  high_level      高级篇 (21_interop ~ 25_deep_learning_integration)
  special_topics  专题篇 (26_ptx ~ 30_image_processing)
  frontier        前沿应用篇 (31_neural_network ~ 35_hpc_future)
  all_tutorials   全部教程

单个教程:
  hello_cuda                    第1课: Hello CUDA
  kernel_basics                 第2课: 核函数基础
  memory_management             第3课: 内存管理
  vector_add                    第4课: 向量加法
  05_shared_memory              第5课: 共享内存
  06_sync_atomic                第6课: 同步与原子操作
  07_cuda_streams               第7课: CUDA Streams
  08_unified_memory             第8课: 统一内存
  09_texture_memory             第9课: 纹理内存
  10_constant_reduction         第10课: 常量内存与归约
  ... (更多请查看 CMakeLists.txt)

EOF
}

# 检查环境
check_environment() {
    # 检查 CUDA
    if ! command -v nvcc &> /dev/null; then
        error "未找到 nvcc，请确保已安装 CUDA Toolkit"
    fi

    # 检查 CMake
    if ! command -v cmake &> /dev/null; then
        error "未找到 cmake，请安装 CMake 3.18+"
    fi

    # 检查 CMake 版本
    CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+')
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)

    if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 18 ]); then
        error "CMake 版本过低，需要 3.18+，当前版本: $CMAKE_VERSION"
    fi

    info "环境检查通过 (CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ','), CMake: $CMAKE_VERSION)"
}

# 配置项目
configure() {
    if [ ! -d "$BUILD_DIR" ]; then
        info "创建构建目录..."
        mkdir -p "$BUILD_DIR"
    fi

    cd "$BUILD_DIR"

    if [ ! -f "Makefile" ]; then
        info "配置 CMake..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi
}

# 编译
build() {
    local target="$1"
    local jobs="${2:-$(nproc)}"

    configure

    cd "$BUILD_DIR"

    if [ -z "$target" ]; then
        info "编译全部教程..."
        make -j"$jobs" all_tutorials
    else
        info "编译目标: $target"
        make -j"$jobs" "$target"
    fi

    success "编译完成！可执行文件位于: $BUILD_DIR/bin/"
}

# 清理
clean() {
    if [ -d "$BUILD_DIR" ]; then
        info "清理构建目录..."
        rm -rf "$BUILD_DIR"
        success "清理完成"
    else
        warn "构建目录不存在"
    fi
}

# 运行程序
run_target() {
    local target="$1"

    if [ -z "$target" ]; then
        error "请指定要运行的程序名称"
    fi

    # 先编译
    build "$target"

    # 查找可执行文件
    local exe="$BUILD_DIR/bin/$target"
    if [ ! -f "$exe" ]; then
        exe="$BUILD_DIR/$target"
    fi

    if [ -f "$exe" ]; then
        info "运行 $target..."
        echo "----------------------------------------"
        "$exe"
        echo "----------------------------------------"
        success "运行完成"
    else
        error "找不到可执行文件: $target"
    fi
}

# 主函数
main() {
    local cmd="$1"
    shift || true

    case "$cmd" in
        help|-h|--help)
            show_help
            ;;
        list)
            list_targets
            ;;
        clean)
            clean
            ;;
        rebuild)
            clean
            check_environment
            build "" "$@"
            ;;
        run)
            check_environment
            run_target "$@"
            ;;
        "")
            check_environment
            build "all_tutorials"
            ;;
        *)
            check_environment
            build "$cmd" "$@"
            ;;
    esac
}

main "$@"
