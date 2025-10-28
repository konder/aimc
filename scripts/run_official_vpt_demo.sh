#!/bin/bash
# 运行官方VPT演示
# 使用官方run_agent.py脚本在MineRL环境中运行VPT Agent

set -e

MODEL_FILE="${1:-data/pretrained/vpt/rl-from-early-game-2x.model}"
WEIGHTS_FILE="${2:-data/pretrained/vpt/rl-from-early-game-2x.weights}"

echo "========================================"
echo "🎮 运行官方VPT演示"
echo "========================================"
echo "模型文件:   $MODEL_FILE"
echo "权重文件:   $WEIGHTS_FILE"
echo "环境:       MineRL (官方)"
echo "========================================"
echo ""

# 检查文件存在
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 模型文件不存在: $MODEL_FILE"
    exit 1
fi

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "❌ 权重文件不存在: $WEIGHTS_FILE"
    exit 1
fi

# 切换到VPT目录
cd src/models/Video-Pre-Training

echo "🚀 启动官方VPT Agent (使用MineRL环境)..."
echo "   提示: 这将打开Minecraft窗口，你可以观察VPT的行为"
echo "   按 Ctrl+C 停止"
echo ""

# 使用run_minedojo_x86.sh确保在正确环境中运行
exec ../../../scripts/run_minedojo_x86.sh python run_agent.py \
    --model "../../../$MODEL_FILE" \
    --weights "../../../$WEIGHTS_FILE"
