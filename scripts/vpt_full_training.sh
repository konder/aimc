#!/bin/bash
#
# VPT完整训练脚本
# 
# 功能: 完整的VPT BC训练（20 epochs）
# 
# 使用方法:
#   bash scripts/vpt_full_training.sh
#

set -e  # 遇到错误立即退出

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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
# 配置
# ============================================================================

TASK_ID="harvest_1_log"
EXPERT_DIR="data/tasks/${TASK_ID}/expert_demos"
OUTPUT_DIR="data/tasks/${TASK_ID}/vpt_bc_model"
VPT_WEIGHTS="data/pretrained/vpt/rl-from-early-game-2x.weights"
LOG_DIR="logs/vpt_training"

# Checkpoint保存配置（可选：指向大容量磁盘）
# 如果不设置，checkpoint会保存到OUTPUT_DIR
# 示例：CHECKPOINT_DIR="/mnt/large_disk/vpt_checkpoints"
CHECKPOINT_DIR=""  # 留空则使用OUTPUT_DIR
KEEP_CHECKPOINTS=3  # 保留最新的N个checkpoint

# 完整训练参数
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4
DEVICE="mps"  # 或 cuda, cpu
SAVE_FREQ=5  # 设为0则不保存checkpoint，只保存best/final

# ============================================================================
# 环境检查
# ============================================================================

print_header "VPT完整训练 - 环境检查"

# 检查conda环境
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "minedojo"* ]]; then
    print_error "请先激活minedojo环境: conda activate minedojo"
    exit 1
fi
print_success "Conda环境: $CONDA_DEFAULT_ENV"

# 检查VPT权重
if [ ! -f "$VPT_WEIGHTS" ]; then
    print_error "VPT权重文件不存在: $VPT_WEIGHTS"
    exit 1
fi
print_success "VPT权重: $(du -h $VPT_WEIGHTS | awk '{print $1}')"

# 检查专家数据
EPISODE_COUNT=$(find "$EXPERT_DIR" -type d -name "episode_*" 2>/dev/null | wc -l | tr -d ' ')
if [ "$EPISODE_COUNT" -eq 0 ]; then
    print_error "未找到专家演示数据: $EXPERT_DIR"
    exit 1
fi
print_success "专家数据: $EPISODE_COUNT 个episodes"

# 数据量建议
if [ "$EPISODE_COUNT" -lt 50 ]; then
    print_warning "专家数据较少（$EPISODE_COUNT个），建议至少50个episodes"
    print_info "可以使用 python src/training/dagger/record_manual_chopping.py 录制更多数据"
    read -p "继续训练？(y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
print_success "输出目录: $OUTPUT_DIR"

# ============================================================================
# 训练配置确认
# ============================================================================

print_header "训练配置"

echo "任务: $TASK_ID"
echo "专家数据: $EXPERT_DIR ($EPISODE_COUNT episodes)"
echo "输出目录: $OUTPUT_DIR"
if [ -n "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint目录: $CHECKPOINT_DIR (独立大容量磁盘)"
else
  echo "Checkpoint目录: $OUTPUT_DIR (默认)"
fi
echo ""
echo "训练参数:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
if [ "$SAVE_FREQ" -eq 0 ]; then
  echo "  Checkpoint: 禁用（只保存best/final）"
else
  echo "  Checkpoint频率: 每 $SAVE_FREQ epochs"
  echo "  保留Checkpoint数: $KEEP_CHECKPOINTS 个"
fi
echo ""

# 估算训练时间
case "$DEVICE" in
    mps)
        TIME_PER_EPOCH="2-3"
        TOTAL_TIME="40-60"
        ;;
    cuda)
        TIME_PER_EPOCH="1-2"
        TOTAL_TIME="20-40"
        ;;
    cpu)
        TIME_PER_EPOCH="10-15"
        TOTAL_TIME="200-300"
        ;;
    *)
        TIME_PER_EPOCH="?"
        TOTAL_TIME="?"
        ;;
esac

print_info "预计时间:"
echo "  每个epoch: ~${TIME_PER_EPOCH}分钟"
echo "  总计: ~${TOTAL_TIME}分钟"
echo ""

read -p "确认开始完整训练？(y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    print_info "训练已取消"
    exit 0
fi

# ============================================================================
# 开始训练
# ============================================================================

print_header "开始VPT完整训练"

LOG_FILE="$LOG_DIR/full_training_$(date +%Y%m%d_%H%M%S).log"
print_info "日志文件: $LOG_FILE"
print_info "监控训练: tail -f $LOG_FILE"
echo ""

START_TIME=$(date +%s)

# 构建训练命令
TRAIN_CMD="bash scripts/run_minedojo_x86.sh python src/training/vpt/train_bc_vpt.py \
  --expert-dir \"$EXPERT_DIR\" \
  --output-dir \"$OUTPUT_DIR\" \
  --vpt-weights \"$VPT_WEIGHTS\" \
  --freeze-vpt \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --device $DEVICE \
  --save-freq $SAVE_FREQ \
  --keep-checkpoints $KEEP_CHECKPOINTS"

# 如果指定了checkpoint目录，添加参数
if [ -n "$CHECKPOINT_DIR" ]; then
  TRAIN_CMD="$TRAIN_CMD --checkpoint-dir \"$CHECKPOINT_DIR\""
  print_info "Checkpoint保存目录: $CHECKPOINT_DIR"
fi

# 执行训练
eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    print_error "训练失败（退出码: $TRAIN_EXIT_CODE）"
    print_info "查看日志: $LOG_FILE"
    exit 1
fi

print_success "训练完成！（用时: ${ELAPSED_MIN}分钟）"

# ============================================================================
# 训练结果分析
# ============================================================================

print_header "训练结果分析"

# Loss趋势
print_info "Loss趋势:"
grep "平均Loss" "$LOG_FILE" | awk '{printf "  Epoch %s: %s\n", $2, $6}'

# 提取首末loss
FIRST_LOSS=$(grep "平均Loss" "$LOG_FILE" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
LAST_LOSS=$(grep "平均Loss" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)

if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
    IMPROVEMENT=$(python -c "print(f'{(($FIRST_LOSS - $LAST_LOSS) / $FIRST_LOSS * 100):.1f}%')")
    echo ""
    echo "Loss改善: $FIRST_LOSS -> $LAST_LOSS (↓ $IMPROVEMENT)"
fi

# 已保存的模型
echo ""
print_info "已保存的模型:"
ls -lht "$OUTPUT_DIR"/*.pth | head -10

# ============================================================================
# 评估提示
# ============================================================================

print_header "下一步：评估模型"

# 优先使用best_model，其次是final_model
if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
    EVAL_MODEL="$OUTPUT_DIR/best_model.pth"
elif [ -f "$OUTPUT_DIR/final_model.pth" ]; then
    EVAL_MODEL="$OUTPUT_DIR/final_model.pth"
else
    EVAL_MODEL="$OUTPUT_DIR/latest.pth"
fi

echo "使用以下命令评估训练的模型："
echo ""
echo "# 快速评估（10 episodes）"
echo "bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_bc_vpt.py \\"
echo "  --model $EVAL_MODEL \\"
echo "  --task-id $TASK_ID \\"
echo "  --episodes 10 \\"
echo "  --max-steps 500 \\"
echo "  --device $DEVICE"
echo ""
echo "# 完整评估（50 episodes）"
echo "bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_bc_vpt.py \\"
echo "  --model $EVAL_MODEL \\"
echo "  --task-id $TASK_ID \\"
echo "  --episodes 50 \\"
echo "  --max-steps 500 \\"
echo "  --device $DEVICE \\"
echo "  --render"
echo ""

read -p "现在进行快速评估（10 episodes）？(y/N): " eval_now
if [[ "$eval_now" =~ ^[Yy]$ ]]; then
    print_header "快速评估（10 episodes）"
    
    EVAL_LOG="$LOG_DIR/eval_final_$(date +%Y%m%d_%H%M%S).log"
    
    bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_bc_vpt.py \
      --model "$EVAL_MODEL" \
      --task-id "$TASK_ID" \
      --episodes 10 \
      --max-steps 500 \
      --device $DEVICE \
      2>&1 | tee "$EVAL_LOG"
    
    print_success "评估完成！"
    print_info "评估日志: $EVAL_LOG"
    
    # 显示结果
    echo ""
    print_info "评估结果:"
    grep "成功:" "$EVAL_LOG"
    grep "平均步数:" "$EVAL_LOG"
    grep "平均奖励:" "$EVAL_LOG"
fi

# ============================================================================
# 总结
# ============================================================================

print_header "训练完成总结"

echo "✓ 训练: $EPOCHS epochs (${ELAPSED_MIN}分钟)"
echo "✓ 模型: $EVAL_MODEL"
echo "✓ 日志: $LOG_FILE"
echo ""
print_success "VPT完整训练已完成！"

