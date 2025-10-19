#!/bin/bash
# 快速演示训练脚本 - 使用小的n_steps快速看到TensorBoard数据

cd "$(dirname "$0")/.."

echo "🚀 启动快速演示训练..."
echo "   - n_steps=32 (快速更新)"
echo "   - total_timesteps=1000"
echo ""

python src/training/train_harvest_paper.py \
    --task-id harvest_milk \
    --total-timesteps 1000 \
    --n-steps 32 \
    --batch-size 32 \
    --device mps \
    --save-freq 500 \
    2>&1 | tee logs/quick_demo.log

echo ""
echo "✓ 演示训练完成！"
echo "📊 现在刷新 TensorBoard: http://localhost:6006"

