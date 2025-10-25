#!/bin/bash
# 快速运行v2微调

cd /Users/nanzhang/aimc

echo "======================================================================"
echo "MineCLIP v2 微调 (任务感知版本)"
echo "======================================================================"
echo ""

python tmp/finetune_mineclip.py \
    --expert-dir data/tasks/harvest_1_log/expert_demos \
    --base-model data/mineclip/attn.pth \
    --output data/mineclip/attn_finetuned_harvest_v2_quick.pth \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-6 \
    --freeze-ratio 0.0 \
    --checkpoint-dir tmp/mineclip_checkpoints_v2

echo ""
echo "======================================================================"
echo "训练完成！"
echo "======================================================================"
echo ""
echo "下一步:"
echo "  1. 测试v2模型:"
echo "     python tmp/test_mineclip_rewards.py \\"
echo "         --mineclip-model data/mineclip/attn_finetuned_harvest_v2_quick.pth \\"
echo "         --output tmp/finetuned_v2 \\"
echo "         --num-episodes 20"
echo ""
echo "  2. 对比v1和v2:"
echo "     python tmp/analyze_finetuned_simple.py"
echo ""

