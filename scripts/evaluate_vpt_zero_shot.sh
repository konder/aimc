#!/bin/bash
#
# 快速启动VPT零样本评估
#
# 使用方法:
#   bash scripts/evaluate_vpt_zero_shot.sh                    # 评估VPTAgent，10轮
#   bash scripts/evaluate_vpt_zero_shot.sh 20                 # 评估VPTAgent，20轮
#   bash scripts/evaluate_vpt_zero_shot.sh 10 cpu            # 指定设备
#

set -e

# 默认参数
EPISODES=${1:-10}
DEVICE=${2:-auto}
MAX_STEPS=${3:-1200}
DEBUG_ACTIONS=${4:-false}

# VPT权重路径
WEIGHTS="data/pretrained/vpt/rl-from-early-game-2x.weights"

# 检查权重文件
if [ ! -f "$WEIGHTS" ]; then
    echo "❌ VPT权重文件不存在: $WEIGHTS"
    echo ""
    echo "请先下载VPT权重："
    echo "  mkdir -p data/pretrained/vpt"
    echo "  cd data/pretrained/vpt"
    echo "  wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.weights -O rl-from-early-game-2x.weights"
    exit 1
fi

echo "========================================"
echo "🎯 VPT零样本评估"
echo "========================================"
echo "任务:      harvest_1_log"
echo "Agent:     VPTAgent (完整版)"
echo "评估轮数:  $EPISODES"
echo "设备:      $DEVICE"
echo "最大步数:  $MAX_STEPS"
echo "权重:      $WEIGHTS"
echo "========================================"
echo ""

# 运行评估（在minedojo-x86环境中）
if [ "$DEBUG_ACTIONS" = "true" ]; then
    bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_vpt_zero_shot.py \
        --episodes $EPISODES \
        --max_steps $MAX_STEPS \
        --device $DEVICE \
        --weights $WEIGHTS \
        --debug-actions
else
    bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_vpt_zero_shot.py \
        --episodes $EPISODES \
        --max_steps $MAX_STEPS \
        --device $DEVICE \
        --weights $WEIGHTS
fi

echo ""
echo "✅ 评估完成！"

