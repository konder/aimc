#!/bin/bash
# harvest_1_paper 训练启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MineDojo harvest_1_paper 训练${NC}"
echo -e "${BLUE}========================================${NC}\n"

# ============================================================================
# 1. 环境检查
# ============================================================================
echo -e "${YELLOW}[1/4] 检查环境...${NC}"

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python未找到，请先安装Python 3.9+${NC}"
    exit 1
fi

# 检查必要的包
echo -e "  检查依赖包..."
python -c "import minedojo" 2>/dev/null || {
    echo -e "${RED}✗ MineDojo未安装，请运行: pip install minedojo${NC}"
    exit 1
}

python -c "import stable_baselines3" 2>/dev/null || {
    echo -e "${RED}✗ Stable-Baselines3未安装，请运行: pip install stable-baselines3[extra]${NC}"
    exit 1
}

echo -e "${GREEN}  ✓ 环境检查通过${NC}\n"

# ============================================================================
# 2. 创建必要的目录
# ============================================================================
echo -e "${YELLOW}[2/4] 创建目录结构...${NC}"

cd "$PROJECT_ROOT"

mkdir -p logs/training
mkdir -p logs/tensorboard
mkdir -p checkpoints/harvest_paper

echo -e "${GREEN}  ✓ 目录创建完成${NC}\n"

# ============================================================================
# 3. 设置环境变量
# ============================================================================
echo -e "${YELLOW}[3/4] 设置环境变量...${NC}"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export JAVA_OPTS="-Djava.awt.headless=true"

echo -e "  PYTHONPATH=$PYTHONPATH"
echo -e "  JAVA_OPTS=$JAVA_OPTS"
echo -e "${GREEN}  ✓ 环境变量设置完成${NC}\n"

# ============================================================================
# 4. 启动训练
# ============================================================================
echo -e "${YELLOW}[4/4] 启动训练...${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 解析命令行参数
MODE="standard"  # 默认: standard
if [ "$1" == "test" ]; then
    MODE="test"
    echo -e "${YELLOW}使用快速测试模式 (10K steps)${NC}\n"
elif [ "$1" == "long" ]; then
    MODE="long"
    echo -e "${YELLOW}使用长时间训练模式 (2M steps)${NC}\n"
else
    echo -e "${YELLOW}使用标准训练模式 (500K steps)${NC}"
    echo -e "${YELLOW}提示: 使用 './train_harvest.sh test' 进行快速测试${NC}\n"
fi

# 根据模式设置参数
if [ "$MODE" == "test" ]; then
    TIMESTEPS=10000
    SAVE_FREQ=5000
    EVAL_FREQ=5000
elif [ "$MODE" == "long" ]; then
    TIMESTEPS=2000000
    SAVE_FREQ=20000
    EVAL_FREQ=20000
else
    TIMESTEPS=500000
    SAVE_FREQ=10000
    EVAL_FREQ=10000
fi

# 运行训练
python "$PROJECT_ROOT/src/training/train_harvest_paper.py" \
    --mode train \
    --task-id harvest_milk \
    --total-timesteps $TIMESTEPS \
    --save-freq $SAVE_FREQ \
    --eval-freq $EVAL_FREQ \
    --n-envs 1 \
    --device auto \
    --log-dir logs/training \
    --checkpoint-dir checkpoints/harvest_paper \
    --tensorboard-dir logs/tensorboard

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}训练脚本执行完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n查看训练日志: tail -f logs/training/training_*.log"
echo -e "查看TensorBoard: tensorboard --logdir logs/tensorboard"
echo -e "检查点位置: checkpoints/harvest_paper/\n"

