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
DEVICE="cpu"  # 训练设备: auto/cpu/cuda/mps (CPU更稳定)

# DAgger配置
DAGGER_ITERATIONS=3
COLLECT_EPISODES=20
DAGGER_EPOCHS=30

# 评估配置
EVAL_EPISODES=20

# 录制配置
NUM_EXPERT_EPISODES=10
CAMERA_DELTA=1
MAX_FRAMES=500
APPEND_RECORDING=false  # 是否追加录制（继续已有数据）

# 数据路径（基础路径，会根据 TASK_ID 自动创建子目录）
BASE_DIR="data"
EXPERT_DIR="${BASE_DIR}/expert_demos/${TASK_ID}"
POLICY_STATES_DIR="${BASE_DIR}/policy_states/${TASK_ID}"
EXPERT_LABELS_DIR="${BASE_DIR}/expert_labels/${TASK_ID}"
DAGGER_DATA_DIR="${BASE_DIR}/dagger/${TASK_ID}"

# 模型路径（按训练方法和任务分类）
TRAINING_METHOD="dagger"  # dagger, ppo, hybrid
CHECKPOINTS_DIR="checkpoints/${TRAINING_METHOD}/${TASK_ID}"

# 标注配置
SMART_SAMPLING=true
FAILURE_WINDOW=10
RANDOM_SAMPLE_RATE=0.1  # 成功episode的随机采样率（10%）

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
        --camera-delta)
            CAMERA_DELTA="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
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
        --continue-from)
            CONTINUE_FROM="$2"
            shift 2
            ;;
        --start-iteration)
            START_ITERATION="$2"
            shift 2
            ;;
        --method)
            TRAINING_METHOD="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
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
            echo "  --camera-delta N            相机灵敏度 (默认: 1)"
            echo "  --max-frames N              每个episode最大帧数 (默认: 500)"
            echo "  --append-recording          追加录制（继续已有数据）"
            echo "  --skip-recording            跳过手动录制 (假设已有数据)"
            echo "  --skip-bc                   跳过BC训练 (假设已有BC模型)"
            echo "  --continue-from MODEL       从指定模型继续DAgger训练"
            echo "  --start-iteration N         从第N轮DAgger开始（与--continue-from配合）"
            echo "  --method METHOD             训练方法 (默认: dagger, 可选: ppo, hybrid)"
            echo "  --device DEVICE             训练设备 (默认: mps, 可选: auto, cpu, cuda, mps)"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "目录结构:"
            echo "  checkpoints/dagger/TASK_ID/     DAgger训练模型"
            echo "  checkpoints/ppo/TASK_ID/        PPO训练模型"
            echo "  checkpoints/hybrid/TASK_ID/     混合训练模型"
            echo ""
            echo "标注优化（默认已启用）:"
            echo "  智能采样: 只标注失败前${FAILURE_WINDOW}步 + 成功episode的${RANDOM_SAMPLE_RATE}%"
            echo "  组合键: Q(前进+攻击), R(前进+跳跃), G(前进+跳跃+攻击)"
            echo "  快捷操作: N(跳过), Z(撤销), X/ESC(完成)"
            echo ""
            echo "继续训练示例:"
            echo "  bash $0 --task harvest_1_log --method dagger --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_1.zip --start-iteration 2 --iterations 5"
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
echo "  模型目录: $CHECKPOINTS_DIR"
echo ""

# 创建必要的目录
mkdir -p "$EXPERT_DIR" "$POLICY_STATES_DIR" "$EXPERT_LABELS_DIR" "$DAGGER_DATA_DIR" "$CHECKPOINTS_DIR"
print_success "目录结构已准备"

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
        echo "  相机灵敏度: $CAMERA_DELTA"
        echo "  数据保存路径: $EXPERT_DIR"
        echo ""
        print_info "控制说明:"
        echo "  WASD     - 移动"
        echo "  IJKL     - 视角"
        echo "  F        - 攻击"
        echo "  Space    - 跳跃"
        echo "  Q        - 重录当前回合（不保存）"
        echo "  ESC      - 退出录制"
        echo ""
        print_info "提示: 每次成功获得木头后，会自动保存为新的episode"
        echo ""
        
        read -p "按Enter开始录制，或按Ctrl+C取消..." 
        
        # 确保输出目录存在
        mkdir -p "$EXPERT_DIR"
        
        # 直接录制到正确的任务目录
        python tools/record_manual_chopping.py \
            --base-dir "$EXPERT_DIR" \
            --max-frames "$MAX_FRAMES" \
            --camera-delta "$CAMERA_DELTA"
        
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

BC_MODEL="${CHECKPOINTS_DIR}/bc_baseline.zip"

if [[ -z "$SKIP_BC" ]]; then
    print_header "阶段1: BC基线训练"
    
    print_info "训练参数:"
    echo "  数据目录: $EXPERT_DIR"
    echo "  训练轮数: $BC_EPOCHS"
    echo "  学习率: $BC_LEARNING_RATE"
    echo "  批次大小: $BC_BATCH_SIZE"
    echo "  训练设备: $DEVICE"
    echo ""
    
    python src/training/train_bc.py \
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

if [[ -z "$CONTINUE_FROM" ]]; then
    # 仅在从头开始时评估BC基线
    print_header "阶段2: 评估BC基线"
    
    print_info "评估BC策略 $BC_MODEL (${EVAL_EPISODES} episodes)..."
    
    python tools/evaluate_policy.py \
        --model "$BC_MODEL" \
        --episodes "$EVAL_EPISODES" \
        --task-id "$TASK_ID" \
        --max-steps "$MAX_STEPS" > /tmp/bc_eval.txt
    
    BC_SUCCESS_RATE=$(grep "成功率:" /tmp/bc_eval.txt | awk '{print $2}')
    print_success "BC基线成功率: $BC_SUCCESS_RATE"
else
    print_info "继续训练模式: 跳过BC基线评估"
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
    print_info "智能采样已启用:"
    echo "  - 失败前 $FAILURE_WINDOW 步: 100%标注（关键决策）"
    echo "  - 成功episode: ${RANDOM_SAMPLE_RATE}%随机采样"
    echo "  - 预计节省 80%+ 标注时间"
    echo ""
    print_info "标注控制:"
    echo "  基础动作: W/A/S/D (移动), I/J/K/L (视角), F (攻击), Space (跳跃)"
    echo "  组合动作: Q (前进+攻击), R (前进+跳跃), G (前进+跳跃+攻击)"
    echo "  快捷操作: N (跳过), Z (撤销), X/ESC (完成)"
    echo ""
    print_info "标注策略:"
    echo "  - 专注失败前的关键步骤"
    echo "  - 标注'应该做什么'而非'不应该做什么'"
    echo "  - 不确定的状态直接按 N 跳过"
    echo ""
    
    read -p "按Enter开始标注..." 
    
    LABELS_FILE="${EXPERT_LABELS_DIR}/iter_${iter}.pkl"
    
    LABEL_ARGS="--states $STATES_DIR --output $LABELS_FILE"
    if [ "$SMART_SAMPLING" = true ]; then
        LABEL_ARGS="$LABEL_ARGS --smart-sampling --failure-window $FAILURE_WINDOW --random-sample-rate $RANDOM_SAMPLE_RATE"
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
        --epochs "$DAGGER_EPOCHS" \
        --device "$DEVICE"
    
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

