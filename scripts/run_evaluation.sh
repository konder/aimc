#!/bin/bash
# STEVE-1 评估框架启动脚本
# 通过参数配置评估任务，无需修改 Python 代码

set -e

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
TASK=""
TASK_LIST=""
TEST_SET=""
N_TRIALS=3
MAX_STEPS=2000
RENDER=""
REPORT_NAME="evaluation_report"
USE_X86=false

# 显示帮助信息
show_help() {
    echo -e "${BLUE}=========================================="
    echo "STEVE-1 评估框架启动脚本"
    echo -e "==========================================${NC}"
    echo ""
    echo "用法："
    echo "  $0 [选项]"
    echo ""
    echo "评估模式（三选一）："
    echo "  --task TASK_ID              评估单个任务"
    echo "  --task-list \"T1 T2 T3\"      评估任务列表（用空格分隔）"
    echo "  --test-set SET_NAME         评估测试集（quick_test, baseline_test）"
    echo ""
    echo "参数配置："
    echo "  --n-trials N                试验次数（默认: 3）"
    echo "  --max-steps N               最大步数（默认: 2000）"
    echo "  --render                    启用渲染"
    echo "  --report-name NAME          报告名称（默认: evaluation_report）"
    echo ""
    echo "环境配置："
    echo "  --x86                       使用 x86 架构（M1/M2 Mac）"
    echo ""
    echo "其他："
    echo "  -h, --help                  显示此帮助信息"
    echo ""
    echo "示例："
    echo "  # 评估单个中文任务"
    echo "  $0 --task harvest_wood_zh --n-trials 3 --render"
    echo ""
    echo "  # 批量评估中英文任务"
    echo "  $0 --task-list \"harvest_wood_en harvest_wood_zh\" --n-trials 3"
    echo ""
    echo "  # 评估测试集"
    echo "  $0 --test-set quick_test --n-trials 5"
    echo ""
    echo "  # 使用 x86 架构（M1/M2 Mac）"
    echo "  $0 --task harvest_wood_zh --x86"
    echo ""
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --task-list)
            TASK_LIST="$2"
            shift 2
            ;;
        --test-set)
            TEST_SET="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --render)
            RENDER="--render"
            shift
            ;;
        --report-name)
            REPORT_NAME="$2"
            shift 2
            ;;
        --x86)
            USE_X86=true
            shift
            ;;
        *)
            echo -e "${RED}错误: 未知参数 '$1'${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证参数
MODE_COUNT=0
[[ -n "$TASK" ]] && ((MODE_COUNT++))
[[ -n "$TASK_LIST" ]] && ((MODE_COUNT++))
[[ -n "$TEST_SET" ]] && ((MODE_COUNT++))

if [[ $MODE_COUNT -eq 0 ]]; then
    echo -e "${RED}错误: 必须指定评估模式（--task, --task-list 或 --test-set）${NC}"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

if [[ $MODE_COUNT -gt 1 ]]; then
    echo -e "${RED}错误: 只能指定一种评估模式${NC}"
    exit 1
fi

# 构建 Python 命令
PYTHON_CMD="python src/evaluation/eval_framework.py"

if [[ -n "$TASK" ]]; then
    PYTHON_CMD="$PYTHON_CMD --task $TASK"
elif [[ -n "$TASK_LIST" ]]; then
    PYTHON_CMD="$PYTHON_CMD --task-list $TASK_LIST"
elif [[ -n "$TEST_SET" ]]; then
    PYTHON_CMD="$PYTHON_CMD --test-set $TEST_SET"
fi

PYTHON_CMD="$PYTHON_CMD --n-trials $N_TRIALS --max-steps $MAX_STEPS"

if [[ -n "$RENDER" ]]; then
    PYTHON_CMD="$PYTHON_CMD $RENDER"
fi

PYTHON_CMD="$PYTHON_CMD --report-name $REPORT_NAME"

# 显示配置信息
echo -e "${BLUE}=========================================="
echo "STEVE-1 评估框架"
echo -e "==========================================${NC}"
echo -e "${GREEN}项目根目录:${NC} $PROJECT_ROOT"

if [[ -n "$TASK" ]]; then
    echo -e "${GREEN}评估模式:${NC} 单任务"
    echo -e "${GREEN}任务ID:${NC} $TASK"
elif [[ -n "$TASK_LIST" ]]; then
    echo -e "${GREEN}评估模式:${NC} 批量任务"
    echo -e "${GREEN}任务列表:${NC} $TASK_LIST"
elif [[ -n "$TEST_SET" ]]; then
    echo -e "${GREEN}评估模式:${NC} 测试集"
    echo -e "${GREEN}测试集:${NC} $TEST_SET"
fi

echo -e "${GREEN}试验次数:${NC} $N_TRIALS"
echo -e "${GREEN}最大步数:${NC} $MAX_STEPS"
echo -e "${GREEN}启用渲染:${NC} $([ -n "$RENDER" ] && echo "是" || echo "否")"
echo -e "${GREEN}报告名称:${NC} $REPORT_NAME"
echo -e "${GREEN}架构模式:${NC} $([ "$USE_X86" = true ] && echo "x86_64" || echo "原生")"
echo -e "${BLUE}==========================================${NC}"
echo ""

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查 Python 环境（提示但不强制）
if ! python -c "import gym" &> /dev/null; then
    echo -e "${YELLOW}警告: 未检测到 gym 模块${NC}"
    echo -e "${YELLOW}提示: 请确保已激活正确的 conda 环境${NC}"
    echo -e "${YELLOW}      conda activate minedojo-x86  (或 minedojo)${NC}"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 1
    fi
fi

# 执行评估
if [[ "$USE_X86" = true ]]; then
    echo -e "${YELLOW}使用 x86 架构启动...${NC}"
    echo ""
    exec "$SCRIPT_DIR/run_minedojo_x86.sh" $PYTHON_CMD
else
    echo -e "${YELLOW}执行命令:${NC} $PYTHON_CMD"
    echo -e "${YELLOW}当前目录:${NC} $(pwd)"
    echo ""
    # 使用 bash -c 确保在当前目录执行
    bash -c "$PYTHON_CMD"
fi
