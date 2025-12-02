#!/bin/bash
# ==============================================================================
# Prior 模型评估脚本
# ==============================================================================
#
# 用法:
#   bash scripts/run_prior_eval.sh [选项]
#
# 选项:
#   --config FILE         配置文件路径（默认: config/eval_tasks_prior.yaml）
#   --task-set SETS       指定任务集（逗号分隔，如 harvest,combat）
#   --task-ids IDS        指定任务ID列表（空格分隔）
#   --output-dir DIR      输出目录（可选）
#
# 示例:
#   # 评估所有启用的任务集
#   bash scripts/run_prior_eval.sh
#
#   # 评估harvest任务集
#   bash scripts/run_prior_eval.sh --task-set harvest
#
#   # 评估多个任务集
#   bash scripts/run_prior_eval.sh --task-set harvest,combat,techtree
#
#   # 评估指定任务
#   bash scripts/run_prior_eval.sh --task-ids harvest_1_log harvest_1_dirt
#
# ==============================================================================

set -e

# 默认参数
CONFIG="config/eval_tasks_prior.yaml"
TASK_SET=""
TASK_IDS=""
OUTPUT_DIR=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --task-set)
            TASK_SET="$2"
            shift 2
            ;;
        --task-ids)
            shift
            TASK_IDS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TASK_IDS="$TASK_IDS $1"
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "      Prior 模型评估"
echo "========================================"
echo ""
echo -e "${BLUE}配置文件:${NC} $CONFIG"
if [ -n "$TASK_SET" ]; then
    echo -e "${BLUE}任务集:${NC} $TASK_SET"
fi
if [ -n "$TASK_IDS" ]; then
    echo -e "${BLUE}任务ID:${NC}$TASK_IDS"
fi
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${BLUE}输出目录:${NC} $OUTPUT_DIR"
fi
echo "========================================"
echo ""

echo ""
echo -e "${YELLOW}开始评估...${NC}"
echo ""

# 构建命令
CMD="python src/evaluation/prior_eval_framework.py --config $CONFIG"

if [ -n "$TASK_SET" ]; then
    CMD="$CMD --task-set $TASK_SET"
fi

if [ -n "$TASK_IDS" ]; then
    CMD="$CMD --task-ids$TASK_IDS"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

# 运行评估
$CMD

echo ""
echo "========================================"
echo -e "${GREEN}✓ 评估完成${NC}"
echo "========================================"

