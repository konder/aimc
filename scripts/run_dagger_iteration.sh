#!/bin/bash
#
# DAgger迭代训练脚本
# 
# 功能: 收集失败状态 → 标注 → 聚合数据 → 训练 → 评估（循环）
# 
# 使用方法:
#   bash scripts/run_dagger_iteration.sh --task harvest_1_log
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数（默认值）
# ============================================================================

# 任务配置
TASK_ID="harvest_1_log"
MAX_STEPS=1000

# DAgger配置
DAGGER_ITERATIONS=1
COLLECT_EPISODES=20
DAGGER_EPOCHS=30
DEVICE="mps"

# 标注配置
SMART_SAMPLING=true
FAILURE_WINDOW=10
RANDOM_SAMPLE_RATE=0.1

# 评估配置
EVAL_EPISODES=20

# 流程控制
SKIP_EVAL=false
SKIP_COLLECT=false  # 跳过收集状态，直接进入标注
CONTINUE_FROM=""
START_ITERATION=1
CLEAN_RESTART=false  # 清理历史数据，从头开始

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
        --iterations)
            DAGGER_ITERATIONS="$2"
            shift 2
            ;;
        --collect-episodes)
            COLLECT_EPISODES="$2"
            shift 2
            ;;
        --dagger-epochs)
            DAGGER_EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --failure-window)
            FAILURE_WINDOW="$2"
            shift 2
            ;;
        --random-sample-rate)
            RANDOM_SAMPLE_RATE="$2"
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
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-collect)
            SKIP_COLLECT=true
            shift
            ;;
        --continue-from)
            CONTINUE_FROM="$2"
            shift 2
            ;;
        --start-iteration)
            START_ITERATION="$2"
            shift 2
            ;;
        --clean-restart)
            CLEAN_RESTART=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "DAgger迭代训练"
            echo ""
            echo "Options:"
            echo "  --task TASK_ID              任务ID (默认: harvest_1_log)"
            echo "  --iterations N              DAgger迭代次数 (默认: 1)"
            echo "  --collect-episodes N        每轮收集episode数 (默认: 20)"
            echo "  --dagger-epochs N           DAgger训练轮数 (默认: 30)"
            echo "  --device DEVICE             训练设备 (默认: mps)"
            echo "  --failure-window N          失败前N步需要标注 (默认: 10)"
            echo "  --random-sample-rate N      成功采样率 (默认: 0.1)"
            echo "  --eval-episodes N           评估episode数 (默认: 20)"
            echo "  --max-steps N               最大步数 (默认: 1000)"
            echo "  --skip-eval                 跳过迭代后的评估"
            echo "  --skip-collect              跳过收集状态，直接进入标注流程"
            echo "  --continue-from MODEL       从指定模型继续"
            echo "  --start-iteration N         起始迭代编号 (默认: 1)"
            echo "  --clean-restart             清理历史DAgger数据，从BC基线重新开始"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  # 执行1轮DAgger迭代"
            echo "  bash $0 --task harvest_1_log --iterations 1"
            echo ""
            echo "  # 从指定模型继续迭代"
            echo "  bash $0 --task harvest_1_log --continue-from data/tasks/harvest_1_log/dagger_model/dagger_iter_2.zip --start-iteration 3"
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
POLICY_STATES_DIR="${TASK_ROOT}/policy_states"
EXPERT_LABELS_DIR="${TASK_ROOT}/expert_labels"
DAGGER_DATA_DIR="${TASK_ROOT}/dagger"
BASELINE_MODEL_DIR="${TASK_ROOT}/baseline_model"
DAGGER_MODEL_DIR="${TASK_ROOT}/dagger_model"

# 创建必要的目录
mkdir -p "$POLICY_STATES_DIR"
mkdir -p "$EXPERT_LABELS_DIR"
mkdir -p "$DAGGER_DATA_DIR"
mkdir -p "$DAGGER_MODEL_DIR"

# ============================================================================
# 清理历史数据（如果指定）
# ============================================================================

if [ "$CLEAN_RESTART" = true ]; then
    print_warning "清理模式: 删除历史DAgger数据，从BC基线重新开始"
    
    # 清理策略收集的状态
    if [ -d "$POLICY_STATES_DIR" ] && [ "$(ls -A $POLICY_STATES_DIR 2>/dev/null)" ]; then
        print_info "清理: $POLICY_STATES_DIR"
        rm -rf "${POLICY_STATES_DIR:?}/"*
        print_success "已清理策略状态"
    fi
    
    # 清理专家标注
    if [ -d "$EXPERT_LABELS_DIR" ] && [ "$(ls -A $EXPERT_LABELS_DIR 2>/dev/null)" ]; then
        print_info "清理: $EXPERT_LABELS_DIR"
        rm -rf "${EXPERT_LABELS_DIR:?}/"*
        print_success "已清理专家标注"
    fi
    
    # 清理聚合数据
    if [ -d "$DAGGER_DATA_DIR" ] && [ "$(ls -A $DAGGER_DATA_DIR 2>/dev/null)" ]; then
        print_info "清理: $DAGGER_DATA_DIR"
        rm -rf "${DAGGER_DATA_DIR:?}/"*
        print_success "已清理聚合数据"
    fi
    
    # 清理DAgger迭代模型
    if [ -d "$DAGGER_MODEL_DIR" ] && [ "$(ls -A $DAGGER_MODEL_DIR 2>/dev/null)" ]; then
        print_info "清理: $DAGGER_MODEL_DIR"
        rm -rf "${DAGGER_MODEL_DIR:?}/"*
        print_success "已清理DAgger模型"
    fi
    
    # 强制从BC基线开始
    CONTINUE_FROM=""
    START_ITERATION=1
    
    echo ""
    print_success "✓ 历史数据已清理，将从BC基线重新开始第1轮迭代"
    echo ""
fi

# ============================================================================
# 确定起始模型（用于收集策略失败状态）
# ============================================================================
# 说明：
# - 继续迭代时，使用最新的DAgger模型来收集失败状态
# - 如果没有DAgger模型，使用BC基线
# - 这样可以确保每次迭代都在改进现有策略

if [ -n "$CONTINUE_FROM" ]; then
    CURRENT_MODEL="$CONTINUE_FROM"
    print_info "从指定模型继续: $CURRENT_MODEL"
else
    # 查找最新的DAgger模型
    LATEST_DAGGER=$(find "$DAGGER_MODEL_DIR" -name "dagger_iter_*.zip" 2>/dev/null | sort -V | tail -1)
    
    if [ -n "$LATEST_DAGGER" ]; then
        CURRENT_MODEL="$LATEST_DAGGER"
        # 自动推断起始迭代
        ITER_NUM=$(basename "$LATEST_DAGGER" | sed 's/dagger_iter_//' | sed 's/.zip//')
        START_ITERATION=$((ITER_NUM + 1))
        print_info "发现已有DAgger模型，从迭代 $START_ITERATION 开始"
        print_info "将使用模型: $(basename $LATEST_DAGGER)"
    else
        # 使用BC基线
        CURRENT_MODEL="$BASELINE_MODEL_DIR/bc_baseline.zip"
        START_ITERATION=1
        print_info "从BC基线开始第一轮DAgger迭代"
    fi
fi

# 检查模型是否存在
if [ ! -f "$CURRENT_MODEL" ]; then
    print_error "模型不存在: $CURRENT_MODEL"
    echo "请先训练BC基线: bash scripts/run_recording_and_baseline.sh --task $TASK_ID"
    exit 1
fi

# ============================================================================
# 主流程
# ============================================================================

print_header "DAgger迭代训练"
echo "任务ID: $TASK_ID"
echo "起始模型: $CURRENT_MODEL"
echo "起始迭代: $START_ITERATION"
echo "迭代次数: $DAGGER_ITERATIONS"
echo ""

# 计算结束迭代
END_ITERATION=$((START_ITERATION + DAGGER_ITERATIONS - 1))

# DAgger迭代循环
for iter in $(seq $START_ITERATION $END_ITERATION); do
    print_header "DAgger 迭代 $iter / $END_ITERATION"
    
    # ========================================================================
    # 步骤1: 收集失败状态
    # ========================================================================
    # 注意：这里使用当前迭代的模型（DAgger或BC基线）来运行并收集失败状态
    
    STATES_DIR="$POLICY_STATES_DIR/iter_${iter}"
    
    if [ "$SKIP_COLLECT" = true ]; then
        print_warning "[$iter] 步骤 1/4: 跳过收集状态 (--skip-collect)"
        
        # 检查状态目录是否存在
        if [ ! -d "$STATES_DIR" ] || [ -z "$(ls -A $STATES_DIR 2>/dev/null)" ]; then
            print_error "状态目录不存在或为空: $STATES_DIR"
            echo "提示: 使用 --skip-collect 时，必须已有收集好的状态数据"
            exit 1
        fi
        
        print_info "  使用已有状态: $STATES_DIR"
        print_success "[$iter] 跳过收集，使用现有状态"
        echo ""
    else
        print_info "[$iter] 步骤 1/4: 收集策略失败状态"
        echo "  使用模型: $(basename $CURRENT_MODEL)"
        echo "  Episodes: $COLLECT_EPISODES"
        echo ""
        
        mkdir -p "$STATES_DIR"
        
        python src/training/dagger/run_policy_collect_states.py \
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
        
        print_success "[$iter] 状态收集完成"
        echo ""
    fi
    
    # ========================================================================
    # 步骤2: 标注失败状态
    # ========================================================================
    
    print_info "[$iter] 步骤 2/4: 标注失败状态"
    
    if [ "$SMART_SAMPLING" = true ]; then
        echo "  智能采样已启用:"
        echo "    - 失败前 $FAILURE_WINDOW 步: 100%标注"
        echo "    - 成功episode: ${RANDOM_SAMPLE_RATE}%随机采样"
    fi
    echo ""
    
    LABELS_FILE="$EXPERT_LABELS_DIR/iter_${iter}.pkl"
    
    LABEL_ARGS="--states $STATES_DIR --output $LABELS_FILE"
    if [ "$SMART_SAMPLING" = true ]; then
        LABEL_ARGS="$LABEL_ARGS --smart-sampling --failure-window $FAILURE_WINDOW --random-sample-rate $RANDOM_SAMPLE_RATE"
    fi
    
    python src/training/dagger/label_states.py $LABEL_ARGS
    
    if [ $? -ne 0 ]; then
        print_error "标注失败或被取消"
        exit 1
    fi
    
    if [ ! -f "$LABELS_FILE" ]; then
        print_error "标注文件未生成: $LABELS_FILE"
        exit 1
    fi
    
    print_success "[$iter] 标注完成"
    echo ""
    
    # ========================================================================
    # 步骤3: 聚合数据并训练
    # ========================================================================
    
    print_info "[$iter] 步骤 3/4: 聚合数据并训练"
    
    DAGGER_MODEL="$DAGGER_MODEL_DIR/dagger_iter_${iter}.zip"
    COMBINED_FILE="$DAGGER_DATA_DIR/combined_iter_${iter}.pkl"
    
    # 确定基础数据
    if [ "$iter" -eq 1 ]; then
        BASE_DATA="$EXPERT_DIR"
        print_info "  基础数据: 专家演示"
    else
        BASE_DATA="$DAGGER_DATA_DIR/combined_iter_$((iter-1)).pkl"
        print_info "  基础数据: 上一轮聚合数据"
    fi
    
    echo "  新数据: $LABELS_FILE"
    echo "  输出模型: $DAGGER_MODEL"
    echo ""
    
    python src/training/dagger/train_dagger.py \
        --mode manual \
        --iteration "$iter" \
        --base-data "$BASE_DATA" \
        --new-data "$LABELS_FILE" \
        --output "$DAGGER_MODEL" \
        --combined-output "$COMBINED_FILE" \
        --epochs "$DAGGER_EPOCHS" \
        --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        print_error "DAgger训练失败"
        exit 1
    fi
    
    print_success "[$iter] DAgger训练完成: $DAGGER_MODEL"
    echo ""
    
    # ========================================================================
    # 步骤4: 评估新策略（可选）
    # ========================================================================
    
    if [ "$SKIP_EVAL" = false ]; then
        print_info "[$iter] 步骤 4/4: 评估迭代 $iter 策略"
        echo ""
        
        python src/training/dagger/evaluate_policy.py \
            --model "$DAGGER_MODEL" \
            --episodes "$EVAL_EPISODES" \
            --task-id "$TASK_ID" \
            --max-steps "$MAX_STEPS"
        
        if [ $? -eq 0 ]; then
            print_success "[$iter] 迭代 $iter 评估完成"
        else
            print_warning "[$iter] 评估失败或出错"
        fi
    else
        print_warning "[$iter] 跳过迭代后的自动评估（--skip-eval）"
    fi
    
    # 更新当前模型为新训练的模型
    CURRENT_MODEL="$DAGGER_MODEL"
    
    echo ""
done

# ============================================================================
# 完成
# ============================================================================

print_header "✅ DAgger迭代训练完成！"

echo "完成的迭代: $START_ITERATION 到 $END_ITERATION"
echo "最终模型: $CURRENT_MODEL"
echo ""
echo "下一步建议:"
echo "  1. 评估最终模型:"
echo "     python src/training/dagger/evaluate_policy.py --model $CURRENT_MODEL --episodes 50"
echo ""
echo "  2. 继续DAgger迭代:"
echo "     bash $0 --task $TASK_ID --iterations 2"
echo ""

