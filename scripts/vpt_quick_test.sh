#!/bin/bash
#
# VPT快速测试训练脚本
# 
# 功能: 快速测试VPT训练流程（2 epochs）
# 
# 使用方法:
#   bash scripts/vpt_quick_test.sh
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

# 测试训练参数
EPOCHS=2
BATCH_SIZE=16
LEARNING_RATE=1e-4
DEVICE="mps"  # 或 cuda, cpu

# ============================================================================
# 步骤1: 环境检查
# ============================================================================

print_header "步骤1: 环境检查"

# 检查conda环境
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "minedojo"* ]]; then
    print_error "请先激活minedojo环境: conda activate minedojo"
    exit 1
fi
print_success "Conda环境: $CONDA_DEFAULT_ENV"

# 检查VPT权重
if [ ! -f "$VPT_WEIGHTS" ]; then
    print_error "VPT权重文件不存在: $VPT_WEIGHTS"
    print_info "请确保VPT权重已下载"
    exit 1
fi
print_success "VPT权重文件存在: $(du -h $VPT_WEIGHTS | awk '{print $1}')"

# 检查专家数据
if [ ! -d "$EXPERT_DIR" ]; then
    print_error "专家数据目录不存在: $EXPERT_DIR"
    print_info "请先录制专家数据"
    exit 1
fi

EPISODE_COUNT=$(find "$EXPERT_DIR" -type d -name "episode_*" 2>/dev/null | wc -l | tr -d ' ')
if [ "$EPISODE_COUNT" -eq 0 ]; then
    print_error "未找到专家演示数据"
    exit 1
fi
print_success "专家数据: $EPISODE_COUNT 个episodes"

# 创建输出目录
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
print_success "输出目录已准备: $OUTPUT_DIR"

# ============================================================================
# 步骤2: VPT环境验证
# ============================================================================

print_header "步骤2: VPT环境验证"

print_info "验证VPT模型加载..."

python -c "
import sys
sys.path.insert(0, '.')
from src.models.vpt import load_vpt_policy

try:
    # 加载VPT policy和权重（一步完成）
    policy, result = load_vpt_policy('$VPT_WEIGHTS', device='cpu', verbose=False)
    
    missing = len(result.missing_keys)
    unexpected = len(result.unexpected_keys)
    
    if missing > 0 or unexpected > 0:
        print(f'✗ 权重加载有问题: Missing={missing}, Unexpected={unexpected}')
        sys.exit(1)
    
    print('✓ Policy创建成功')
    print(f'✓ 权重加载成功: Missing=0, Unexpected=0')
    
    # 统计参数
    total_params = sum(p.numel() for p in policy.parameters())
    print(f'✓ 总参数: {total_params:,}')
    
except Exception as e:
    print(f'✗ VPT环境验证失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "VPT环境验证失败"
    exit 1
fi

print_success "VPT环境ready！"

# ============================================================================
# 步骤3: 快速测试训练（2 epochs）
# ============================================================================

print_header "步骤3: 快速测试训练（2 epochs）"

print_info "训练配置:"
echo "  专家数据: $EXPERT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  VPT权重: $VPT_WEIGHTS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Device: $DEVICE"
echo ""
print_info "预计时间: 5-10分钟（取决于设备）"
echo ""

read -p "按Enter开始测试训练，或按Ctrl+C取消..." 

# 开始训练
print_info "开始训练..."
LOG_FILE="$LOG_DIR/test_run_$(date +%Y%m%d_%H%M%S).log"

bash scripts/run_minedojo_x86.sh python src/training/vpt/train_bc_vpt.py \
  --expert-dir "$EXPERT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --vpt-weights "$VPT_WEIGHTS" \
  --freeze-vpt \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --device $DEVICE \
  --save-freq 1 \
  --keep-checkpoints 2 \
  2>&1 | tee "$LOG_FILE"

if [ $? -ne 0 ]; then
    print_error "训练失败，请查看日志: $LOG_FILE"
    exit 1
fi

print_success "训练完成！"
print_info "日志保存: $LOG_FILE"

# ============================================================================
# 步骤4: 检查训练结果
# ============================================================================

print_header "步骤4: 检查训练结果"

# 检查loss趋势
print_info "Loss趋势:"
grep "平均Loss" "$LOG_FILE" | tail -$EPOCHS

# 检查模型文件
print_info "已保存的模型:"
ls -lh "$OUTPUT_DIR"/*.pth 2>/dev/null || print_warning "未找到模型文件"

# 检查训练时间
print_info "训练时间:"
grep "用时" "$LOG_FILE" | tail -$EPOCHS

# 验证loss下降
FIRST_LOSS=$(grep "平均Loss" "$LOG_FILE" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
LAST_LOSS=$(grep "平均Loss" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)

if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
    LOSS_DECREASED=$(python -c "print('yes' if $LAST_LOSS < $FIRST_LOSS else 'no')")
    
    if [ "$LOSS_DECREASED" = "yes" ]; then
        print_success "Loss下降: $FIRST_LOSS -> $LAST_LOSS ✓"
    else
        print_warning "Loss未下降: $FIRST_LOSS -> $LAST_LOSS"
    fi
fi

# ============================================================================
# 步骤5: 快速评估
# ============================================================================

print_header "步骤5: 快速评估（5 episodes）"

print_info "使用训练的模型进行快速评估..."
read -p "按Enter开始评估，或按Ctrl+C跳过..." 

EVAL_LOG="$LOG_DIR/test_eval_$(date +%Y%m%d_%H%M%S).log"

# 优先使用best_model，其次是final_model
if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
    EVAL_MODEL="$OUTPUT_DIR/best_model.pth"
    print_info "使用最佳模型: best_model.pth"
elif [ -f "$OUTPUT_DIR/final_model.pth" ]; then
    EVAL_MODEL="$OUTPUT_DIR/final_model.pth"
    print_info "使用最终模型: final_model.pth"
else
    print_error "未找到模型文件 (best_model.pth 或 final_model.pth)"
    exit 1
fi

print_info "评估模型: $EVAL_MODEL"

bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_bc_vpt.py \
  --model "$EVAL_MODEL" \
  --task-id "$TASK_ID" \
  --episodes 5 \
  --max-steps 500 \
  --device $DEVICE \
  2>&1 | tee "$EVAL_LOG"

if [ $? -ne 0 ]; then
    print_error "评估失败，请查看日志: $EVAL_LOG"
    exit 1
fi

print_success "评估完成！"
print_info "日志保存: $EVAL_LOG"

# 提取评估结果
print_info "评估结果:"
grep "成功:" "$EVAL_LOG" || print_warning "未找到成功率统计"
grep "平均步数:" "$EVAL_LOG" || print_warning "未找到平均步数"
grep "平均奖励:" "$EVAL_LOG" || print_warning "未找到平均奖励"

# ============================================================================
# 总结
# ============================================================================

print_header "测试完成！"

SUCCESS_COUNT=$(grep -c "Success: True" "$EVAL_LOG" 2>/dev/null || echo "0")
TOTAL_EPISODES=5

echo "快速测试结果:"
echo "  训练: $EPOCHS epochs"
echo "  评估: $TOTAL_EPISODES episodes"
echo "  成功: $SUCCESS_COUNT episodes"
echo ""

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    print_success "✓ 训练有效！模型能够完成任务"
    echo ""
    print_info "下一步建议:"
    echo "  1. 进行完整训练（20 epochs）:"
    echo "     bash scripts/vpt_full_training.sh"
    echo ""
    echo "  2. 查看详细训练指南:"
    echo "     cat VPT_TRAINING_GUIDE.md"
else
    print_warning "测试训练完成，但成功率为0"
    echo ""
    print_info "这可能是正常的（只训练了2个epoch）"
    print_info "建议："
    echo "  1. 检查模型是否有明显行动（不是完全静止）"
    echo "  2. 继续完整训练（20 epochs）看效果"
    echo "  3. 如果完整训练后仍无效，查看VPT_TRAINING_GUIDE.md的问题排查部分"
fi

echo ""
print_success "所有日志保存在: $LOG_DIR"

