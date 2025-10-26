#!/bin/bash
#
# 模型评估脚本
# 
# 功能: 评估指定模型的性能
# 
# 使用方法:
#   bash scripts/run_evaluation.sh --task harvest_1_log --model data/tasks/harvest_1_log/baseline_model/bc_baseline.zip
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数（默认值）
# ============================================================================

# 任务配置
TASK_ID="harvest_1_log"
MAX_STEPS=1000

# 评估配置
EVAL_EPISODES=20
MODEL_PATH=""

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

function print_header() {
    echo -e "\n${BLUE}============================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# ============================================================================
# 参数解析
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_ID="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "评估指定模型的性能"
            echo ""
            echo "Options:"
            echo "  --task TASK_ID              任务ID (默认: harvest_1_log)"
            echo "  --model MODEL_PATH          模型路径 (必需)"
            echo "  --episodes N                评估episode数 (默认: 20)"
            echo "  --max-steps N               最大步数 (默认: 1000)"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  # 评估BC基线"
            echo "  bash $0 \\"
            echo "      --task harvest_1_log \\"
            echo "      --model data/tasks/harvest_1_log/baseline_model/bc_baseline.zip \\"
            echo "      --episodes 20"
            echo ""
            echo "  # 评估DAgger迭代模型"
            echo "  bash $0 \\"
            echo "      --task harvest_1_log \\"
            echo "      --model data/tasks/harvest_1_log/dagger_model/dagger_iter_2.zip \\"
            echo "      --episodes 50"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 参数验证
# ============================================================================

if [ -z "$MODEL_PATH" ]; then
    print_error "必须指定模型路径 (--model)"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    print_error "模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 尝试从配置文件读取参数
TASK_ROOT="data/tasks/${TASK_ID}"
CONFIG_FILE="$TASK_ROOT/config.yaml"

if [ -f "$CONFIG_FILE" ]; then
    print_info "发现配置文件: $CONFIG_FILE"
    
    # 尝试从配置文件读取 eval_episodes（如果未通过命令行指定）
    if [ "$EVAL_EPISODES" -eq 20 ]; then
        CONFIG_EVAL_EPISODES=$(grep "^eval_episodes:" "$CONFIG_FILE" | awk '{print $2}')
        if [ -n "$CONFIG_EVAL_EPISODES" ]; then
            EVAL_EPISODES=$CONFIG_EVAL_EPISODES
            print_info "从配置文件读取 eval_episodes: $EVAL_EPISODES"
        fi
    fi
    
    # 尝试从配置文件读取 max_steps
    if [ "$MAX_STEPS" -eq 1000 ]; then
        CONFIG_MAX_STEPS=$(grep "^max_steps:" "$CONFIG_FILE" | awk '{print $2}')
        if [ -n "$CONFIG_MAX_STEPS" ]; then
            MAX_STEPS=$CONFIG_MAX_STEPS
            print_info "从配置文件读取 max_steps: $MAX_STEPS"
        fi
    fi
fi

# ============================================================================
# 主流程
# ============================================================================

print_header "模型评估"
echo "任务ID: $TASK_ID"
echo "模型路径: $MODEL_PATH"
echo "评估Episodes: $EVAL_EPISODES"
echo "最大步数: $MAX_STEPS"
echo ""

print_info "开始评估..."
echo ""

python src/training/dagger/evaluate_policy.py \
    --model "$MODEL_PATH" \
    --episodes "$EVAL_EPISODES" \
    --task-id "$TASK_ID" \
    --max-steps "$MAX_STEPS"

EVAL_EXIT_CODE=$?

echo ""

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    print_success "评估完成！"
else
    print_error "评估失败或出错（退出码: $EVAL_EXIT_CODE）"
    exit $EVAL_EXIT_CODE
fi

echo ""

