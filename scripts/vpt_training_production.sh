#!/bin/bash
#
# VPT生产环境训练脚本示例
# 
# 功能: 演示如何在磁盘空间有限的生产环境中进行VPT训练
# 
# 特点:
#   - Checkpoint保存到独立大容量磁盘
#   - 只保留最新3个checkpoint
#   - 重要模型保存在项目目录
# 
# 使用前：
#   1. 修改LARGE_DISK_PATH为实际的大容量磁盘路径
#   2. 确保该路径有足够空间（至少5GB）
# 
# 使用方法:
#   bash scripts/vpt_training_production.sh
#

set -e

# ============================================================================
# 配置区域 - 根据实际环境修改
# ============================================================================

# 任务配置
TASK_ID="harvest_1_log"
EXPERT_DIR="data/tasks/${TASK_ID}/expert_demos"
OUTPUT_DIR="data/tasks/${TASK_ID}/vpt_bc_model"
VPT_WEIGHTS="data/pretrained/vpt/rl-from-early-game-2x.weights"

# ⚠️ 重要：修改为实际的大容量磁盘路径
# 示例：
#   - Linux: LARGE_DISK_PATH="/mnt/large_disk/vpt_checkpoints"
#   - macOS: LARGE_DISK_PATH="/Volumes/ExternalDrive/vpt_checkpoints"
#   - 服务器: LARGE_DISK_PATH="/data/checkpoints/vpt"
LARGE_DISK_PATH="/root/autodl-tmp/vpt_checkpoints"  # 临时目录示例，请修改！

# Checkpoint配置
CHECKPOINT_DIR="$LARGE_DISK_PATH/$TASK_ID"
KEEP_CHECKPOINTS=3        # 只保留最新3个checkpoint
SAVE_FREQ=5               # 每5轮保存一次checkpoint

# 训练参数
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
DEVICE="cuda"              # 生产环境通常使用cpu

# 日志
LOG_DIR="/root/autodl-tmp/logs/vpt_training"

# ============================================================================
# 环境检查
# ============================================================================

echo "============================================================================"
echo "VPT生产环境训练配置"
echo "============================================================================"
echo ""

# 检查conda环境
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "minedojo"* ]]; then
    echo "✗ 请先激活minedojo环境: conda activate minedojo"
    exit 1
fi
echo "✓ Conda环境: $CONDA_DEFAULT_ENV"

# 检查VPT权重
if [ ! -f "$VPT_WEIGHTS" ]; then
    echo "✗ VPT权重文件不存在: $VPT_WEIGHTS"
    exit 1
fi
echo "✓ VPT权重: $(du -h "$VPT_WEIGHTS" | awk '{print $1}')"

# 检查专家数据
if [ ! -d "$EXPERT_DIR" ]; then
    echo "✗ 专家数据目录不存在: $EXPERT_DIR"
    exit 1
fi
EPISODE_COUNT=$(ls -d "$EXPERT_DIR"/episode_* 2>/dev/null | wc -l | tr -d ' ')
echo "✓ 专家数据: $EPISODE_COUNT episodes"

# ============================================================================
# 磁盘空间检查
# ============================================================================

echo ""
echo "磁盘空间检查:"
echo "----------------------------------------------------------------------------"

# 检查项目目录磁盘
PROJECT_DISK=$(df -h "$(pwd)" | awk 'NR==2 {print $4}')
echo "项目目录可用空间: $PROJECT_DISK"
echo "  需要空间: ~2.5GB (best_model.pth + final_model.pth)"

# 创建checkpoint目录（如果不存在）
mkdir -p "$CHECKPOINT_DIR"

# 检查checkpoint目录磁盘
CHECKPOINT_DISK=$(df -h "$CHECKPOINT_DIR" | awk 'NR==2 {print $4}')
echo "Checkpoint目录可用空间: $CHECKPOINT_DISK"
echo "  路径: $CHECKPOINT_DIR"
echo "  需要空间: ~3.5GB ($KEEP_CHECKPOINTS checkpoints)"

# 警告：如果使用/tmp目录
if [[ "$LARGE_DISK_PATH" == /tmp* ]]; then
    echo ""
    echo "⚠️  警告: 当前使用/tmp目录，可能在重启后丢失"
    echo "   建议修改LARGE_DISK_PATH为持久化存储路径"
fi

# ============================================================================
# 训练配置确认
# ============================================================================

echo ""
echo "训练配置:"
echo "----------------------------------------------------------------------------"
echo "任务: $TASK_ID"
echo "专家数据: $EPISODE_COUNT episodes"
echo ""
echo "模型保存:"
echo "  输出目录: $OUTPUT_DIR"
echo "    - best_model.pth (验证loss最低)"
echo "    - final_model.pth (最后一个epoch)"
echo "    - train_config.yaml (训练配置)"
echo "    - train_history.yaml (训练历史)"
echo ""
echo "  Checkpoint目录: $CHECKPOINT_DIR"
echo "    - checkpoint_epoch_N.pth (每$SAVE_FREQ轮保存)"
echo "    - 只保留最新$KEEP_CHECKPOINTS个"
echo "    - 自动清理旧checkpoint"
echo ""
echo "训练参数:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo "  VPT参数冻结: 是（只训练action heads）"
echo ""
echo "预计磁盘占用:"
echo "  项目目录: ~2.2GB (best + final)"
echo "  Checkpoint目录: ~3.3GB ($KEEP_CHECKPOINTS checkpoints)"
echo "  总计: ~5.5GB"
echo ""

read -p "确认开始训练？(y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

# ============================================================================
# 开始训练
# ============================================================================

echo ""
echo "============================================================================"
echo "开始训练"
echo "============================================================================"
echo ""

# 创建日志目录
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/production_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"
echo "监控训练: tail -f $LOG_FILE"
echo ""

START_TIME=$(date +%s)

bash scripts/run_minedojo_x86.sh python src/training/vpt/train_bc_vpt.py \
  --expert-dir "$EXPERT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --vpt-weights "$VPT_WEIGHTS" \
  --freeze-vpt \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --device $DEVICE \
  --save-freq $SAVE_FREQ \
  --keep-checkpoints $KEEP_CHECKPOINTS \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ 训练失败（退出码: $TRAIN_EXIT_CODE）"
    echo "查看日志: $LOG_FILE"
    exit 1
fi

echo ""
echo "✓ 训练完成！（用时: ${ELAPSED_MIN}分钟）"

# ============================================================================
# 训练后检查
# ============================================================================

echo ""
echo "============================================================================"
echo "训练结果"
echo "============================================================================"
echo ""

echo "保存的模型:"
echo "----------------------------------------------------------------------------"
if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
    echo "✓ $OUTPUT_DIR/best_model.pth ($(du -h "$OUTPUT_DIR/best_model.pth" | awk '{print $1}'))"
else
    echo "✗ best_model.pth 未找到"
fi

if [ -f "$OUTPUT_DIR/final_model.pth" ]; then
    echo "✓ $OUTPUT_DIR/final_model.pth ($(du -h "$OUTPUT_DIR/final_model.pth" | awk '{print $1}'))"
else
    echo "✗ final_model.pth 未找到"
fi

echo ""
echo "Checkpoints:"
echo "----------------------------------------------------------------------------"
CKPT_COUNT=$(ls "$CHECKPOINT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | wc -l | tr -d ' ')
if [ "$CKPT_COUNT" -gt 0 ]; then
    echo "保留的checkpoint: $CKPT_COUNT 个"
    ls -lh "$CHECKPOINT_DIR"/checkpoint_epoch_*.pth
else
    echo "无checkpoint（可能save_freq=0）"
fi

echo ""
echo "磁盘使用:"
echo "----------------------------------------------------------------------------"
echo "项目目录: $(du -sh "$OUTPUT_DIR" | awk '{print $1}')"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint目录: $(du -sh "$CHECKPOINT_DIR" | awk '{print $1}')"
fi

# ============================================================================
# 下一步
# ============================================================================

echo ""
echo "============================================================================"
echo "下一步: 评估模型"
echo "============================================================================"
echo ""

echo "评估best_model.pth:"
echo ""
echo "bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_bc_vpt.py \\"
echo "  --model $OUTPUT_DIR/best_model.pth \\"
echo "  --task-id $TASK_ID \\"
echo "  --episodes 20 \\"
echo "  --max-steps 500 \\"
echo "  --device $DEVICE"
echo ""

echo "查看模型详情:"
echo ""
echo "python tools/verify_vpt_model.py \\"
echo "  --model $OUTPUT_DIR/best_model.pth"
echo ""

echo "============================================================================"

