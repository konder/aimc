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
DEVICE="mps"  # 训练设备: auto/cpu/cuda/mps (CPU更稳定)

# DAgger配置
DAGGER_ITERATIONS=3
COLLECT_EPISODES=20
DAGGER_EPOCHS=30

# 评估配置
EVAL_EPISODES=20

# 录制配置
NUM_EXPERT_EPISODES=10
MOUSE_SENSITIVITY=0.15  # 鼠标灵敏度（已优化）
MAX_FRAMES=6000
SKIP_IDLE_FRAMES=true  # 跳过静止帧（不保存IDLE帧）
APPEND_RECORDING=false  # 是否追加录制（继续已有数据）
FULLSCREEN=false  # 是否全屏显示（推荐！防止鼠标移出窗口）

# 数据路径（新结构：data/tasks/task_id/）
TASK_ROOT="data/tasks/${TASK_ID}"
EXPERT_DIR="${TASK_ROOT}/expert_demos"
POLICY_STATES_DIR="${TASK_ROOT}/policy_states"
EXPERT_LABELS_DIR="${TASK_ROOT}/expert_labels"
DAGGER_DATA_DIR="${TASK_ROOT}/dagger"

# 模型路径（新结构：分离BC基线和DAgger迭代）
BASELINE_MODEL_DIR="${TASK_ROOT}/baseline_model"   # BC基线模型
DAGGER_MODEL_DIR="${TASK_ROOT}/dagger_model"       # DAgger迭代模型

# 标注配置
SMART_SAMPLING=true
FAILURE_WINDOW=10
RANDOM_SAMPLE_RATE=0.1  # 成功episode的随机采样率（10%）

# 流程控制（通过参数设置）
SKIP_RECORDING=false      # 跳过录制
SKIP_BC=false             # 跳过BC训练
SKIP_BC_EVAL=false        # 跳过BC基线评估
SKIP_ITER_EVAL=false      # 跳过每次迭代后的自动评估
CONTINUE_FROM=""          # 从指定模型继续
START_ITERATION=1         # 起始迭代次数
CLEAN_RESTART=false       # 清理所有训练数据，完全重新开始
CLEAN_DAGGER_ONLY=false   # 仅清理DAgger数据，保留BC基线

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
        --skip-recording)
            SKIP_RECORDING=true
            shift
            ;;
        --skip-bc)
            SKIP_BC=true
            shift
            ;;
        --skip-bc-eval)
            SKIP_BC_EVAL=true
            shift
            ;;
        --skip-iter-eval)
            SKIP_ITER_EVAL=true
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
        --clean-dagger-only)
            CLEAN_DAGGER_ONLY=true
            shift
            ;;
        --method)
            TRAINING_METHOD="$2"
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TASK_ID              任务ID (默认: harvest_1_log)"
            echo "  --iterations N              DAgger迭代次数 (默认: 3)"
            echo "  --bc-epochs N               BC训练轮数 (默认: 50)"
            echo "  --collect-episodes N        每轮收集episode数 (默认: 20)"
            echo "  --eval-episodes N           评估episode数 (默认: 20)"
            echo "  --num-episodes N            录制专家演示数量 (默认: 10)"
            echo "  --mouse-sensitivity N       鼠标灵敏度 (默认: 0.15)"
            echo "  --max-frames N              每个episode最大帧数 (默认: 6000)"
            echo "  --no-skip-idle              保存所有帧（包括IDLE帧，默认跳过）"
            echo "  --fullscreen                全屏显示（推荐！防止鼠标移出窗口）"
            echo "  --append-recording          追加录制（继续已有数据）"
            echo "  --skip-recording            跳过手动录制 (假设已有数据)"
            echo "  --skip-bc                   跳过BC训练 (假设已有BC模型)"
            echo "  --skip-bc-eval              跳过BC基线评估（直接进入DAgger）"
            echo "  --skip-iter-eval            跳过每次DAgger迭代后的自动评估"
            echo "  --continue-from MODEL       从指定模型继续DAgger训练"
            echo "  --start-iteration N         从第N轮DAgger开始（与--continue-from配合）"
            echo "  --clean-restart             清理DAgger数据，从BC基线重新开始（保留专家演示和BC基线）"
            echo "  --clean-dagger-only         同 --clean-restart（兼容旧参数）"
            echo "  --method METHOD             训练方法 (默认: dagger, 可选: ppo, hybrid)"
            echo "  --device DEVICE             训练设备 (默认: mps, 可选: auto, cpu, cuda, mps)"
            echo "  --failure-window N          失败前N步需要标注 (默认: 10)"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "目录结构:"
            echo "  checkpoints/dagger/TASK_ID/     DAgger训练模型"
            echo "  checkpoints/ppo/TASK_ID/        PPO训练模型"
            echo "  checkpoints/hybrid/TASK_ID/     混合训练模型"
            echo ""
            echo "标注优化（默认已启用）:"
            echo "  智能采样: 只标注失败前${FAILURE_WINDOW}步 + 成功episode的${RANDOM_SAMPLE_RATE}%"
            echo "  组合键: Q(前进+跳跃), E(前进+攻击), R(前进+跳跃+攻击)"
            echo "  快捷操作: X(保持策略), C(跳过), Z(撤销), ESC(完成)"
            echo ""
            echo "示例:"
            echo "  # 跳过录制和BC训练（使用已有模型）"
            echo "  bash $0 --skip-recording --skip-bc --iterations 3"
            echo ""
            echo "  # 跳过所有前置步骤，直接进入DAgger迭代"
            echo "  bash $0 --skip-recording --skip-bc --skip-bc-eval --iterations 3"
            echo ""
            echo "  # 继续已有的DAgger训练"
            echo "  bash $0 --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_1.zip --start-iteration 2 --iterations 5"
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

# 更新模型路径（基于解析后的参数）
CHECKPOINTS_DIR="checkpoints/${TRAINING_METHOD}/${TASK_ID}"

# 验证训练方法
case "$TRAINING_METHOD" in
    dagger|ppo|hybrid)
        print_success "训练方法: $TRAINING_METHOD"
        ;;
    *)
        print_error "不支持的训练方法: $TRAINING_METHOD"
        print_error "支持的方法: dagger, ppo, hybrid"
        exit 1
        ;;
esac

# 显示配置信息
print_info "配置信息:"
echo "  任务ID: $TASK_ID"
echo "  训练方法: $TRAINING_METHOD"
echo "  训练设备: $DEVICE"
echo "  数据目录: $EXPERT_DIR"
echo "  BC基线目录: $BASELINE_MODEL_DIR"
echo "  DAgger模型目录: $DAGGER_MODEL_DIR"
echo ""

# 创建必要的目录
mkdir -p "$EXPERT_DIR" "$POLICY_STATES_DIR" "$EXPERT_LABELS_DIR" "$DAGGER_DATA_DIR" "$BASELINE_MODEL_DIR" "$DAGGER_MODEL_DIR"
print_success "目录结构已准备"

# ============================================================================
# 数据清理（如果指定）
# ============================================================================

if [ "$CLEAN_RESTART" = true ] || [ "$CLEAN_DAGGER_ONLY" = true ]; then
    print_header "数据清理: 清理DAgger数据"
    print_warning "将删除DAgger相关数据，从BC基线重新开始"
    print_info "专家演示和BC基线将被保留（数据珍贵）"
    
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
    
    # 跳过录制和BC训练，从BC基线开始DAgger
    SKIP_RECORDING=true
    SKIP_BC=true
    CONTINUE_FROM=""
    START_ITERATION=1
    
    print_success "✓ DAgger数据已清理，将从BC基线重新开始第1轮迭代"
    print_info "注意: 专家演示和BC基线已保留"
    echo ""
fi

# ============================================================================
# 阶段0: 手动录制专家演示 (可选)
# ============================================================================

if [[ -z "$SKIP_RECORDING" ]]; then
    print_header "阶段0: 录制专家演示 (任务: ${TASK_ID})"
    
    # 检查已有数据
    EXISTING_EPISODES=$(find "$EXPERT_DIR" -type d -name "episode_*" 2>/dev/null | wc -l | tr -d ' ')
    
    if [ "$EXISTING_EPISODES" -gt 0 ]; then
        print_info "已有数据: $EXISTING_EPISODES 个episode"
        if [ "$APPEND_RECORDING" = true ]; then
            print_info "追加模式: 继续录制更多episodes"
            REMAINING=$((NUM_EXPERT_EPISODES - EXISTING_EPISODES))
            if [ $REMAINING -le 0 ]; then
                print_warning "已有 $EXISTING_EPISODES 个episodes，达到目标 $NUM_EXPERT_EPISODES"
                print_info "如需继续录制，请使用 --num-episodes 指定更大的数量"
            else
                print_info "将录制 $REMAINING 个额外的episodes (目标: ${NUM_EXPERT_EPISODES})"
            fi
        else
            print_warning "发现已有数据！"
            print_warning "使用 --append-recording 继续录制，或 --skip-recording 跳过"
            read -p "是否覆盖已有数据？(y/N): " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                print_info "取消录制，使用已有数据"
                SKIP_RECORDING=true
            else
                print_warning "将覆盖已有数据..."
                rm -rf "${EXPERT_DIR}/episode_"*
                EXISTING_EPISODES=0
            fi
        fi
    else
        print_info "未找到已有数据，将录制 $NUM_EXPERT_EPISODES 个episodes"
    fi
    
    if [[ -z "$SKIP_RECORDING" ]]; then
        echo ""
        print_info "录制配置:"
        echo "  任务ID: $TASK_ID"
        echo "  目标episodes: $NUM_EXPERT_EPISODES"
        echo "  已有episodes: $EXISTING_EPISODES"
        echo "  每episode最大帧数: $MAX_FRAMES"
        echo "  鼠标灵敏度: $MOUSE_SENSITIVITY"
        echo "  跳过静止帧: $SKIP_IDLE_FRAMES"
        echo "  全屏显示: $FULLSCREEN"
        echo "  数据保存路径: $EXPERT_DIR"
        echo ""
        print_info "控制说明 (Pygame + 鼠标):"
        echo "  🖱️  鼠标移动   - 转动视角（上下左右）"
        echo "  🖱️  鼠标左键   - 攻击/挖掘"
        echo "  ⌨️  WASD      - 移动"
        echo "  ⌨️  Space     - 跳跃"
        echo "  ⌨️  方向键 ↑↓←→ - 精确调整视角（1°增量）"
        echo "  ⌨️  F11       - 切换全屏/窗口"
        echo "  ⌨️  Q         - 重录当前回合（不保存）"
        echo "  ⌨️  ESC       - 退出录制"
        echo ""
        print_info "提示: 全屏模式可防止鼠标移出窗口（推荐）"
        echo ""
        
        read -p "按Enter开始录制，或按Ctrl+C取消..." 
        
        # 确保输出目录存在
        mkdir -p "$EXPERT_DIR"
        
        # 构建录制命令
        RECORD_CMD="bash scripts/run_minedojo_x86.sh python src/training/dagger/record_manual_chopping.py \
            --base-dir \"$EXPERT_DIR\" \
            --max-frames $MAX_FRAMES \
            --mouse-sensitivity $MOUSE_SENSITIVITY \
            --fps 20"
        
        # 根据SKIP_IDLE_FRAMES添加参数
        if [ "$SKIP_IDLE_FRAMES" = false ]; then
            RECORD_CMD="$RECORD_CMD --no-skip-idle-frames"
        fi
        
        # 根据FULLSCREEN添加参数
        if [ "$FULLSCREEN" = true ]; then
            RECORD_CMD="$RECORD_CMD --fullscreen"
        fi
        
        # 执行录制
        eval $RECORD_CMD
        
        if [ $? -eq 0 ]; then
            print_success "专家演示录制完成"
        else
            print_error "录制失败或被用户中断"
            # 不退出，允许用户使用已录制的数据继续
        fi
    fi
else
    print_info "跳过录制，使用已有数据: $EXPERT_DIR"
fi

# 检查是否有数据
EPISODE_COUNT=$(find "$EXPERT_DIR" -type d -name "episode_*" 2>/dev/null | wc -l | tr -d ' ')
if [ "$EPISODE_COUNT" -eq 0 ]; then
    print_error "未找到专家演示数据！"
    print_error "数据路径: $EXPERT_DIR"
    print_info "请先录制数据，或使用正确的 --task 参数"
    exit 1
fi
print_success "数据路径: $EXPERT_DIR"
print_success "找到 $EPISODE_COUNT 个episode"

# 警告数据量不足
if [ "$EPISODE_COUNT" -lt 5 ]; then
    print_warning "警告: 只有 $EPISODE_COUNT 个episodes，建议至少 5 个"
    print_warning "BC训练效果可能较差，建议使用 --append-recording 继续录制"
fi

# ============================================================================
# 阶段1: BC基线训练
# ============================================================================

BC_MODEL="${BASELINE_MODEL_DIR}/bc_baseline.zip"

if [[ -z "$SKIP_BC" ]]; then
    print_header "阶段1: BC基线训练"
    
    print_info "训练参数:"
    echo "  数据目录: $EXPERT_DIR"
    echo "  训练轮数: $BC_EPOCHS"
    echo "  学习率: $BC_LEARNING_RATE"
    echo "  批次大小: $BC_BATCH_SIZE"
    echo "  训练设备: $DEVICE"
    echo ""
    
    python src/training/dagger/train_bc.py \
        --data "$EXPERT_DIR" \
        --output "$BC_MODEL" \
        --epochs "$BC_EPOCHS" \
        --learning-rate "$BC_LEARNING_RATE" \
        --batch-size "$BC_BATCH_SIZE" \
        --device "$DEVICE"
    
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

if [[ -z "$CONTINUE_FROM" ]] && [[ -z "$SKIP_BC_EVAL" ]]; then
    # 仅在从头开始且未跳过评估时评估BC基线
    print_header "阶段2: 评估BC基线"
    
    print_info "评估BC策略 $BC_MODEL (${EVAL_EPISODES} episodes)..."
    
    python src/training/dagger/evaluate_policy.py \
        --model "$BC_MODEL" \
        --episodes "$EVAL_EPISODES" \
        --task-id "$TASK_ID" \
        --max-steps "$MAX_STEPS" | tee /tmp/bc_eval.txt
    
    BC_SUCCESS_RATE=$(grep "成功率:" /tmp/bc_eval.txt | awk '{print $2}')
    print_success "BC基线成功率: $BC_SUCCESS_RATE"
elif [[ -n "$SKIP_BC_EVAL" ]]; then
    print_info "跳过BC基线评估，直接进入DAgger迭代"
    BC_SUCCESS_RATE="(未评估)"
else
    print_info "继续训练模式: 跳过BC基线评估"
    BC_SUCCESS_RATE="(未评估)"
fi

# ============================================================================
# 阶段3: DAgger迭代优化
# ============================================================================

# 确定起始模型和迭代编号
if [[ -n "$CONTINUE_FROM" ]]; then
    # 继续训练模式
    print_info "继续训练模式: 从 $CONTINUE_FROM 开始"
    CURRENT_MODEL="$CONTINUE_FROM"
    
    if [ ! -f "$CURRENT_MODEL" ]; then
        print_error "指定的模型不存在: $CURRENT_MODEL"
        exit 1
    fi
    
    # 确定起始迭代编号
    if [[ -n "$START_ITERATION" ]]; then
        START_ITER=$START_ITERATION
    else
        # 从模型文件名自动推断
        if [[ "$CURRENT_MODEL" =~ dagger_iter_([0-9]+) ]]; then
            LAST_ITER=${BASH_REMATCH[1]}
            START_ITER=$((LAST_ITER + 1))
            print_info "自动检测: 上一轮为 iter_${LAST_ITER}，从 iter_${START_ITER} 开始"
        else
            print_error "无法从模型文件名推断迭代编号，请使用 --start-iteration 指定"
            exit 1
        fi
    fi
    
    # 自动跳过录制和BC训练
    SKIP_RECORDING=true
    SKIP_BC=true
    
    print_success "将执行 DAgger 迭代 $START_ITER 到 $DAGGER_ITERATIONS"
else
    # 从头开始训练
    CURRENT_MODEL="$BC_MODEL"
    START_ITER=1
fi

for iter in $(seq $START_ITER $DAGGER_ITERATIONS); do
    print_header "阶段3: DAgger迭代 $iter/$DAGGER_ITERATIONS"
    
    # 3.1 收集失败状态
    print_info "[$iter] 步骤1: 收集策略失败状态..."
    
    STATES_DIR="${POLICY_STATES_DIR}/iter_${iter}"
    
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
    print_success "状态收集完成: $STATES_DIR"
    
    # 3.2 交互式标注
    print_info "[$iter] 步骤2: 智能标注失败场景..."
    print_warning "即将打开标注界面，请手动标注失败场景"
    echo ""
    print_info "智能采样已启用:"
    echo "  - 失败前 $FAILURE_WINDOW 步: 100%标注（关键决策）"
    echo "  - 成功episode: ${RANDOM_SAMPLE_RATE}%随机采样"
    echo "  - 预计节省 80%+ 标注时间"
    echo ""
    print_info "标注控制:"
    echo "  基础动作: WASD (移动), 方向键↑↓←→ (视角), F (攻击), Space (跳跃)"
    echo "  组合动作: Q (前进+跳跃), E (前进+攻击), R (前进+跳跃+攻击)"
    echo "  快捷操作: X (保持策略), C (跳过), Z (撤销), ESC (完成)"
    echo ""
    print_info "标注策略:"
    echo "  - 专注失败前的关键步骤"
    echo "  - 标注'应该做什么'而非'不应该做什么'"
    echo "  - 策略正确时按 X 保持，不确定时按 C 跳过"
    echo ""
    
    read -p "按Enter开始标注..." 
    
    LABELS_FILE="${EXPERT_LABELS_DIR}/iter_${iter}.pkl"
    
    LABEL_ARGS="--states $STATES_DIR --output $LABELS_FILE"
    if [ "$SMART_SAMPLING" = true ]; then
        LABEL_ARGS="$LABEL_ARGS --smart-sampling --failure-window $FAILURE_WINDOW --random-sample-rate $RANDOM_SAMPLE_RATE"
    fi
    
    python src/training/dagger/label_states.py $LABEL_ARGS
    
    if [ $? -ne 0 ]; then
        print_error "标注失败"
        exit 1
    fi
    print_success "标注完成: $LABELS_FILE"
    
    # 3.3 聚合数据并训练
    print_info "[$iter] 步骤3: 聚合数据并训练DAgger模型..."
    
    DAGGER_MODEL="${DAGGER_MODEL_DIR}/dagger_iter_${iter}.zip"
    COMBINED_FILE="${DAGGER_DATA_DIR}/combined_iter_${iter}.pkl"
    
    # 确定基础数据
    if [ $iter -eq 1 ]; then
        BASE_DATA="$EXPERT_DIR"
    else
        BASE_DATA="${DAGGER_DATA_DIR}/combined_iter_$((iter-1)).pkl"
    fi
    
    # 训练前先聚合数据
    print_info "  聚合数据: $BASE_DATA + $LABELS_FILE -> $COMBINED_FILE"
    
    python src/training/dagger/train_dagger.py \
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
    print_success "DAgger训练完成: $DAGGER_MODEL"
    
    # 3.4 评估新策略（可选）
    if [ "$SKIP_ITER_EVAL" = false ]; then
        print_info "[$iter] 步骤4: 评估迭代 $iter 策略..."
        
        python src/training/dagger/evaluate_policy.py \
            --model "$DAGGER_MODEL" \
            --episodes "$EVAL_EPISODES" \
            --task-id "$TASK_ID" \
            --max-steps "$MAX_STEPS" | tee "/tmp/dagger_iter_${iter}_eval.txt"
        
        ITER_SUCCESS_RATE=$(grep "成功率:" "/tmp/dagger_iter_${iter}_eval.txt" | awk '{print $2}')
        print_success "迭代 $iter 成功率: $ITER_SUCCESS_RATE"
    else
        print_warning "[$iter] 跳过迭代后的自动评估（--skip-iter-eval）"
    fi
    
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
echo "     python src/training/dagger/evaluate_policy.py --model $CURRENT_MODEL --episodes 50"
echo ""
echo "  2. (可选) 继续DAgger迭代:"
echo "     bash scripts/run_dagger_workflow.sh --skip-recording --skip-bc --iterations 2"
echo ""

print_success "DAgger工作流执行完成！"

