#!/bin/bash
#
# 录制专家演示 + BC基线训练脚本
# 
# 功能: 录制专家演示 → 训练BC基线 → 评估
# 
# 使用方法:
#   bash scripts/run_recording_and_baseline.sh --task harvest_1_log
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数（默认值）
# ============================================================================

# 任务配置
TASK_ID="harvest_1_log"
MAX_STEPS=1000

# BC训练配置
BC_EPOCHS=50
BC_LEARNING_RATE=0.0003
BC_BATCH_SIZE=64
DEVICE="mps"

# 评估配置
EVAL_EPISODES=20

# 录制配置
NUM_EXPERT_EPISODES=10
MOUSE_SENSITIVITY=0.15
MAX_FRAMES=6000
SKIP_IDLE_FRAMES=true
FULLSCREEN=false
APPEND_RECORDING=false

# 流程控制
SKIP_RECORDING=false
SKIP_BC=false
SKIP_EVAL=false

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
        --num-episodes)
            NUM_EXPERT_EPISODES="$2"
            shift 2
            ;;
        --mouse-sensitivity)
            MOUSE_SENSITIVITY="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --no-skip-idle)
            SKIP_IDLE_FRAMES=false
            shift
            ;;
        --fullscreen)
            FULLSCREEN=true
            shift
            ;;
        --append-recording)
            APPEND_RECORDING=true
            shift
            ;;
        --bc-epochs)
            BC_EPOCHS="$2"
            shift 2
            ;;
        --bc-learning-rate)
            BC_LEARNING_RATE="$2"
            shift 2
            ;;
        --bc-batch-size)
            BC_BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
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
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "录制专家演示并训练BC基线模型"
            echo ""
            echo "Options:"
            echo "  --task TASK_ID              任务ID (默认: harvest_1_log)"
            echo "  --num-episodes N            录制专家演示数量 (默认: 10)"
            echo "  --mouse-sensitivity N       鼠标灵敏度 (默认: 0.15)"
            echo "  --max-frames N              每个episode最大帧数 (默认: 6000)"
            echo "  --no-skip-idle              保存所有帧（包括IDLE帧）"
            echo "  --fullscreen                全屏显示（推荐）"
            echo "  --append-recording          追加录制（继续已有数据）"
            echo "  --bc-epochs N               BC训练轮数 (默认: 50)"
            echo "  --bc-learning-rate N        BC学习率 (默认: 0.0003)"
            echo "  --bc-batch-size N           BC批次大小 (默认: 64)"
            echo "  --device DEVICE             训练设备 (默认: mps)"
            echo "  --eval-episodes N           评估episode数 (默认: 20)"
            echo "  --max-steps N               最大步数 (默认: 1000)"
            echo "  --skip-recording            跳过录制（假设已有数据）"
            echo "  --skip-bc                   跳过BC训练（假设已有模型）"
            echo "  --skip-eval                 跳过评估"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  bash $0 --task harvest_1_log --num-episodes 10"
            echo "  bash $0 --task harvest_1_log --skip-recording --skip-bc"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 目录设置
# ============================================================================

TASK_ROOT="data/tasks/${TASK_ID}"
EXPERT_DIR="${TASK_ROOT}/expert_demos"
BASELINE_MODEL_DIR="${TASK_ROOT}/baseline_model"

# 创建必要的目录
mkdir -p "$EXPERT_DIR"
mkdir -p "$BASELINE_MODEL_DIR"

# ============================================================================
# 主流程
# ============================================================================

print_header "录制专家演示和BC基线训练"
echo "任务ID: $TASK_ID"
echo "专家演示目录: $EXPERT_DIR"
echo "基线模型目录: $BASELINE_MODEL_DIR"
echo ""

# ============================================================================
# 阶段1: 录制专家演示
# ============================================================================

if [ "$SKIP_RECORDING" = false ]; then
    print_header "阶段 1/3: 录制专家演示"
    
    print_info "录制参数:"
    echo "  Episodes数量: $NUM_EXPERT_EPISODES"
    echo "  鼠标灵敏度: $MOUSE_SENSITIVITY"
    echo "  最大帧数: $MAX_FRAMES"
    echo "  跳过静止帧: $SKIP_IDLE_FRAMES"
    echo "  全屏模式: $FULLSCREEN"
    echo "  追加录制: $APPEND_RECORDING"
    echo ""
    
    # 构建录制命令
    RECORD_CMD="bash scripts/run_minedojo_x86.sh python src/training/dagger/record_manual_chopping.py"
    RECORD_CMD="$RECORD_CMD --base-dir $EXPERT_DIR"
    RECORD_CMD="$RECORD_CMD --max-frames $MAX_FRAMES"
    RECORD_CMD="$RECORD_CMD --mouse-sensitivity $MOUSE_SENSITIVITY"
    RECORD_CMD="$RECORD_CMD --fps 20"
    
    if [ "$SKIP_IDLE_FRAMES" = true ]; then
        RECORD_CMD="$RECORD_CMD --skip-idle"
    fi
    
    if [ "$FULLSCREEN" = true ]; then
        RECORD_CMD="$RECORD_CMD --fullscreen"
    fi
    
    if [ "$APPEND_RECORDING" = true ]; then
        RECORD_CMD="$RECORD_CMD --append"
    fi
    
    print_info "执行录制..."
    eval $RECORD_CMD
    
    # 检查录制结果
    RECORDED_COUNT=$(find "$EXPERT_DIR" -maxdepth 1 -type d -name "episode_*" | wc -l)
    if [ "$RECORDED_COUNT" -eq 0 ]; then
        print_error "未找到录制的演示数据"
        exit 1
    fi
    
    print_success "录制完成！共 $RECORDED_COUNT 个episodes"
else
    print_warning "跳过录制阶段"
    
    # 检查是否有现有数据
    RECORDED_COUNT=$(find "$EXPERT_DIR" -maxdepth 1 -type d -name "episode_*" 2>/dev/null | wc -l)
    if [ "$RECORDED_COUNT" -eq 0 ]; then
        print_error "未找到专家演示数据，请先录制或移除 --skip-recording"
        exit 1
    fi
    print_info "使用已有的 $RECORDED_COUNT 个episodes"
fi

echo ""

# ============================================================================
# 阶段2: 训练BC基线
# ============================================================================

if [ "$SKIP_BC" = false ]; then
    print_header "阶段 2/3: 训练BC基线"
    
    BC_MODEL="$BASELINE_MODEL_DIR/bc_baseline.zip"
    
    print_info "训练参数:"
    echo "  Epochs: $BC_EPOCHS"
    echo "  Learning Rate: $BC_LEARNING_RATE"
    echo "  Batch Size: $BC_BATCH_SIZE"
    echo "  Device: $DEVICE"
    echo "  输出模型: $BC_MODEL"
    echo ""
    
    python src/training/dagger/train_bc.py \
        --data "$EXPERT_DIR" \
        --output "$BC_MODEL" \
        --epochs "$BC_EPOCHS" \
        --learning-rate "$BC_LEARNING_RATE" \
        --batch-size "$BC_BATCH_SIZE" \
        --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        print_error "BC训练失败"
        exit 1
    fi
    
    print_success "BC训练完成: $BC_MODEL"
else
    print_warning "跳过BC训练阶段"
    
    BC_MODEL="$BASELINE_MODEL_DIR/bc_baseline.zip"
    if [ ! -f "$BC_MODEL" ]; then
        print_error "未找到BC模型，请先训练或移除 --skip-bc"
        exit 1
    fi
    print_info "使用已有模型: $BC_MODEL"
fi

echo ""

# ============================================================================
# 阶段3: 评估BC基线
# ============================================================================

if [ "$SKIP_EVAL" = false ]; then
    print_header "阶段 3/3: 评估BC基线"
    
    print_info "评估参数:"
    echo "  Episodes: $EVAL_EPISODES"
    echo "  Max Steps: $MAX_STEPS"
    echo ""
    
    python src/training/dagger/evaluate_policy.py \
        --model "$BC_MODEL" \
        --episodes "$EVAL_EPISODES" \
        --task-id "$TASK_ID" \
        --max-steps "$MAX_STEPS"
    
    if [ $? -ne 0 ]; then
        print_warning "评估失败或出错"
    else
        print_success "评估完成"
    fi
else
    print_warning "跳过评估阶段"
fi

echo ""

# ============================================================================
# 完成
# ============================================================================

print_header "✅ 录制和BC基线训练完成！"

echo "下一步建议:"
echo "  1. 查看评估结果，确认BC基线性能"
echo "  2. 如果需要改进，运行DAgger迭代:"
echo "     bash scripts/run_dagger_iteration.sh --task $TASK_ID"
echo ""

