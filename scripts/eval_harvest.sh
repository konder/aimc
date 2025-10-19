#!/bin/bash
# harvest_1_paper 模型评估脚本

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MineDojo 模型评估${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export JAVA_OPTS="-Djava.awt.headless=true"

cd "$PROJECT_ROOT"

# 检查模型文件
MODEL_PATH="${1:-checkpoints/harvest_paper/best_model.zip}"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}✗ 模型文件未找到: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请先训练模型或指定正确的模型路径${NC}"
    echo -e "${YELLOW}用法: ./eval_harvest.sh <模型路径>${NC}\n"
    exit 1
fi

echo -e "${GREEN}✓ 找到模型: $MODEL_PATH${NC}\n"

# 运行评估
python "$PROJECT_ROOT/src/training/train_harvest_paper.py" \
    --mode eval \
    --model-path "$MODEL_PATH" \
    --task-id harvest_milk \
    --n-eval-episodes 10 \
    --max-episode-steps 2000

echo -e "\n${GREEN}✓ 评估完成${NC}\n"

