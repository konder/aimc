#!/bin/bash
#
# DAgger完整工作流脚本
# 
# 功能: 自动化执行BC训练 + DAgger迭代优化
# 
# 使用方法:
#   bash scripts/run_dagger_workflow.sh
#
# 或者指定参数:
#   bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 任务配置
TASK_ID="harvest_1_log"
MAX_STEPS=1000

# BC训练配置
BC_EPOCHS=50
BC_LEARNING_RATE=0.0003
BC_BATCH_SIZE=64

# DAgger配置
DAGGER_ITERATIONS=3
COLLECT_EPISODES=20
DAGGER_EPOCHS=30

# 评估配置
EVAL_EPISODES=20

# 数据路径
BASE_DIR="data"
EXPERT_DIR="${BASE_DIR}/expert_demos"
POLICY_STATES_DIR="${BASE_DIR}/policy_states"
EXPERT_LABELS_DIR="${BASE_DIR}/expert_labels"
DAGGER_DATA_DIR="${BASE_DIR}/dagger"
CHECKPOINTS_DIR="checkpoints"

# 标注配置
SMART_SAMPLING=true
FAILURE_WINDOW=10

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
# 参数解析
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_ID="$2"
            shift 2
            ;;
        --iterations)
            DAGGER_ITERATIONS="$2"
            shift 2
            ;;
        --bc-epochs)
            BC_EPOCHS="$2"
            shift 2
            ;;
        --collect-episodes)
            COLLECT_EPISODES="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --skip-recording)
            SKIP_RECORDING=true
            shift
            ;;
        --skip-bc)
            SKIP_BC=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TASK_ID              任务ID (默认: harvest_1_log)"
            echo "  --iterations N              DAgger迭代次数 (默认: 3)"
            echo "  --bc-epochs N               BC训练轮数 (默认: 50)"
            echo "  --collect-episodes N        每轮收集episode数 (默认: 20)"
            echo "  --eval-episodes N           评估episode数 (默认: 20)"
            echo "  --skip-recording            跳过手动录制 (假设已有数据)"
            echo "  --skip-bc                   跳过BC训练 (假设已有BC模型)"
            echo "  -h, --help                  显示帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# 环境检查
# ============================================================================

print_header "环境检查"

# 检查conda环境
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "minedojo"* ]]; then
    print_error "请先激活minedojo环境: conda activate minedojo"
    exit 1
fi
print_success "Conda环境: $CONDA_DEFAULT_ENV"

# 创建必要的目录
mkdir -p "$EXPERT_DIR" "$POLICY_STATES_DIR" "$EXPERT_LABELS_DIR" "$DAGGER_DATA_DIR" "$CHECKPOINTS_DIR"
print_success "目录结构已准备"

# ============================================================================
# 阶段0: 手动录制专家演示 (可选)
# ============================================================================

if [[ -z "$SKIP_RECORDING" ]]; then
    print_header "阶段0: 录制专家演示"
    
    print_info "准备录制专家演示数据..."
    print_info "请在游戏中演示如何完成任务 (${TASK_ID})"
    print_info "建议录制 10-15 个成功的episode"
    echo ""
    print_info "控制说明:"
    echo "  WASD     - 移动"
    echo "  IJKL     - 视角"
    echo "  F        - 攻击"
    echo "  Space    - 跳跃"
    echo "  Q        - 重录当前回合"
    echo "  ESC      - 退出录制"
    echo ""
    
    read -p "按Enter开始录制，或按Ctrl+C取消..." 
    
    python tools/record_manual_chopping.py \
        --base-dir "$EXPERT_DIR" \
        --max-episodes 15 \
        --max-frames 1000 \
        --task-id "$TASK_ID"
    
    if [ $? -eq 0 ]; then
        print_success "专家演示录制完成"
    else
        print_error "录制失败"
        exit 1
    fi
else
    print_info "跳过录制，使用已有数据: $EXPERT_DIR"
fi

# 检查是否有数据
EPISODE_COUNT=$(find "$EXPERT_DIR" -type d -name "episode_*" | wc -l)
if [ "$EPISODE_COUNT" -eq 0 ]; then
    print_error "未找到专家演示数据！请先录制数据。"
    exit 1
fi
print_success "找到 $EPISODE_COUNT 个episode"

# ============================================================================
# 阶段1: BC基线训练
# ============================================================================

BC_MODEL="${CHECKPOINTS_DIR}/bc_baseline.zip"

if [[ -z "$SKIP_BC" ]]; then
    print_header "阶段1: BC基线训练"
    
    print_info "训练参数:"
    echo "  数据目录: $EXPERT_DIR"
    echo "  训练轮数: $BC_EPOCHS"
    echo "  学习率: $BC_LEARNING_RATE"
    echo "  批次大小: $BC_BATCH_SIZE"
    echo ""
    
    python src/training/train_bc.py \
        --data "$EXPERT_DIR" \
        --output "$BC_MODEL" \
        --epochs "$BC_EPOCHS" \
        --learning-rate "$BC_LEARNING_RATE" \
        --batch-size "$BC_BATCH_SIZE"
    
    if [ $? -eq 0 ]; then
        print_success "BC训练完成: $BC_MODEL"
    else
        print_error "BC训练失败"
        exit 1
    fi
else
    print_info "跳过BC训练，使用已有模型: $BC_MODEL"
    if [ ! -f "$BC_MODEL" ]; then
        print_error "BC模型不存在: $BC_MODEL"
        exit 1
    fi
fi

# ============================================================================
# 阶段2: 评估BC基线
# ============================================================================

print_header "阶段2: 评估BC基线"

print_info "评估BC策略 (${EVAL_EPISODES} episodes)..."

python tools/evaluate_policy.py \
    --model "$BC_MODEL" \
    --episodes "$EVAL_EPISODES" \
    --task-id "$TASK_ID" \
    --max-steps "$MAX_STEPS" > /tmp/bc_eval.txt

BC_SUCCESS_RATE=$(grep "成功率:" /tmp/bc_eval.txt | awk '{print $2}')
print_success "BC基线成功率: $BC_SUCCESS_RATE"

# ============================================================================
# 阶段3: DAgger迭代优化
# ============================================================================

CURRENT_MODEL="$BC_MODEL"

for iter in $(seq 1 $DAGGER_ITERATIONS); do
    print_header "阶段3: DAgger迭代 $iter/$DAGGER_ITERATIONS"
    
    # 3.1 收集失败状态
    print_info "[$iter] 步骤1: 收集策略失败状态..."
    
    STATES_DIR="${POLICY_STATES_DIR}/iter_${iter}"
    
    python tools/run_policy_collect_states.py \
        --model "$CURRENT_MODEL" \
        --episodes "$COLLECT_EPISODES" \
        --output "$STATES_DIR" \
        --task-id "$TASK_ID" \
        --max-steps "$MAX_STEPS" \
        --save-failures-only
    
    if [ $? -ne 0 ]; then
        print_error "状态收集失败"
        exit 1
    fi
    print_success "状态收集完成: $STATES_DIR"
    
    # 3.2 交互式标注
    print_info "[$iter] 步骤2: 智能标注失败场景..."
    print_warning "即将打开标注界面，请手动标注失败场景"
    echo ""
    print_info "标注控制:"
    echo "  WASD/IJKL/F  - 标注动作"
    echo "  N            - 跳过当前状态"
    echo "  Z            - 撤销上一个标注"
    echo "  X/ESC        - 完成标注"
    echo ""
    
    read -p "按Enter开始标注..." 
    
    LABELS_FILE="${EXPERT_LABELS_DIR}/iter_${iter}.pkl"
    
    LABEL_ARGS="--states $STATES_DIR --output $LABELS_FILE"
    if [ "$SMART_SAMPLING" = true ]; then
        LABEL_ARGS="$LABEL_ARGS --smart-sampling --failure-window $FAILURE_WINDOW"
    fi
    
    python tools/label_states.py $LABEL_ARGS
    
    if [ $? -ne 0 ]; then
        print_error "标注失败"
        exit 1
    fi
    print_success "标注完成: $LABELS_FILE"
    
    # 3.3 聚合数据并训练
    print_info "[$iter] 步骤3: 聚合数据并训练DAgger模型..."
    
    DAGGER_MODEL="${CHECKPOINTS_DIR}/dagger_iter_${iter}.zip"
    
    # 确定基础数据
    if [ $iter -eq 1 ]; then
        BASE_DATA="$EXPERT_DIR"
    else
        BASE_DATA="${DAGGER_DATA_DIR}/combined_iter_$((iter-1)).pkl"
    fi
    
    python src/training/train_dagger.py \
        --iteration "$iter" \
        --base-data "$BASE_DATA" \
        --new-data "$LABELS_FILE" \
        --output "$DAGGER_MODEL" \
        --epochs "$DAGGER_EPOCHS"
    
    if [ $? -ne 0 ]; then
        print_error "DAgger训练失败"
        exit 1
    fi
    print_success "DAgger训练完成: $DAGGER_MODEL"
    
    # 3.4 评估新策略
    print_info "[$iter] 步骤4: 评估迭代 $iter 策略..."
    
    python tools/evaluate_policy.py \
        --model "$DAGGER_MODEL" \
        --episodes "$EVAL_EPISODES" \
        --task-id "$TASK_ID" \
        --max-steps "$MAX_STEPS" > "/tmp/dagger_iter_${iter}_eval.txt"
    
    ITER_SUCCESS_RATE=$(grep "成功率:" "/tmp/dagger_iter_${iter}_eval.txt" | awk '{print $2}')
    print_success "迭代 $iter 成功率: $ITER_SUCCESS_RATE"
    
    # 更新当前模型
    CURRENT_MODEL="$DAGGER_MODEL"
    
    echo ""
done

# ============================================================================
# 最终总结
# ============================================================================

print_header "训练完成！"

echo "训练历史:"
echo "  BC基线:       $BC_SUCCESS_RATE"

for iter in $(seq 1 $DAGGER_ITERATIONS); do
    if [ -f "/tmp/dagger_iter_${iter}_eval.txt" ]; then
        RATE=$(grep "成功率:" "/tmp/dagger_iter_${iter}_eval.txt" | awk '{print $2}')
        echo "  DAgger迭代$iter:  $RATE"
    fi
done

echo ""
echo "最终模型: $CURRENT_MODEL"
echo ""

print_info "下一步建议:"
echo "  1. 在更多episode上测试最终模型:"
echo "     python tools/evaluate_policy.py --model $CURRENT_MODEL --episodes 50"
echo ""
echo "  2. (可选) 继续DAgger迭代:"
echo "     bash scripts/run_dagger_workflow.sh --skip-recording --skip-bc --iterations 2"
echo ""
echo "  3. (可选) PPO精调:"
echo "     python src/training/train_get_wood.py --resume --checkpoint $CURRENT_MODEL"
echo ""

print_success "DAgger工作流执行完成！"

