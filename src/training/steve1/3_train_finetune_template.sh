#!/bin/bash
#
# STEVE-1 微调训练脚本模板
# 用途: 在预训练STEVE-1模型基础上进行任务特定微调
# 
# 使用方法:
#   1. 准备你的自定义数据集 (参考 docs/technical/STEVE1_TRAINING_ANALYSIS.md)
#   2. 修改下方的参数配置
#   3. 运行: bash 3_train_finetune_template.sh
#

set -e  # 遇到错误立即停止

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# ============================================================
# 用户配置区域 - 根据你的需求修改这里
# ============================================================

# 1. 模型权重路径
IN_MODEL="$PROJECT_ROOT/data/weights/vpt/2x.model"
IN_WEIGHTS="$PROJECT_ROOT/data/weights/steve1/steve1.weights"  # 预训练STEVE-1
OUT_WEIGHTS="$PROJECT_ROOT/data/weights/steve1/steve1_finetuned.weights"  # 输出权重名称

# 2. 数据采样配置 (需要先运行 2_create_sampling.sh 生成)
SAMPLING_NAME="neurips"  # 修改为你的采样配置名称，例如 "finetune_task"
SAMPLING_DIR="$PROJECT_ROOT/data/samplings/"

# 3. 训练超参数 (微调推荐值)
BATCH_SIZE=8                      # 批量大小 (显存24GB推荐8, 16GB用4)
GRADIENT_ACCUM_STEPS=2            # 梯度累积 (有效批量 = BATCH_SIZE × GRADIENT_ACCUM_STEPS)
LEARNING_RATE=1e-5                # 学习率 (微调推荐1e-5, 比从头训练小10倍)
WARMUP_FRAMES=1000000             # 学习率预热帧数 (100万帧)
TOTAL_FRAMES=10000000             # 总训练帧数 (1000万帧，约数小时)

# 4. 序列长度 (显存不足时减小)
T=320                             # 序列总长度 (默认640, 减半可节省显存)
TRUNC_T=64                        # 截断长度 (梯度回传步数)

# 5. 条件训练参数
P_UNCOND=0.1                      # 无条件训练概率 (用于Classifier-Free Guidance)
MIN_BTWN_GOALS=15                 # 目标间隔最小帧数
MAX_BTWN_GOALS=100                # 目标间隔最大帧数 (微调时可减小)

# 6. 验证和保存频率
VAL_FREQ=200                      # 验证频率 (每200步验证一次)
VAL_FREQ_BEGIN=50                 # 初期验证频率
VAL_FREQ_SWITCH=500               # 切换到常规验证的步数
SAVE_FREQ=500                     # 保存检查点频率
SNAPSHOT_EVERY_N_FRAMES=5000000   # 每500万帧保存快照

# 7. 其他参数
NUM_WORKERS=4                     # 数据加载线程数 (CPU核心多可增加)
WEIGHT_DECAY=0.039428             # 权重衰减
MAX_GRAD_NORM=5.0                 # 梯度裁剪阈值

# 8. 检查点路径
CHECKPOINT_DIR="$PROJECT_ROOT/data/finetuning_checkpoint"

# ============================================================
# 预检查
# ============================================================
echo "=========================================="
echo "STEVE-1 微调训练配置"
echo "=========================================="
echo "输入模型: $IN_MODEL"
echo "输入权重: $IN_WEIGHTS"
echo "输出权重: $OUT_WEIGHTS"
echo "数据采样: $SAMPLING_NAME"
echo "批量大小: $BATCH_SIZE (有效批量: $((BATCH_SIZE * GRADIENT_ACCUM_STEPS)))"
echo "学习率: $LEARNING_RATE"
echo "总训练帧数: $TOTAL_FRAMES"
echo "检查点目录: $CHECKPOINT_DIR"
echo "=========================================="

# 检查必需文件是否存在
if [ ! -f "$IN_MODEL" ]; then
    echo "❌ 错误: 模型文件不存在: $IN_MODEL"
    exit 1
fi

if [ ! -f "$IN_WEIGHTS" ]; then
    echo "❌ 错误: 权重文件不存在: $IN_WEIGHTS"
    echo "请先下载预训练STEVE-1权重"
    exit 1
fi

if [ ! -f "$SAMPLING_DIR/${SAMPLING_NAME}_train.txt" ]; then
    echo "❌ 错误: 采样配置不存在: $SAMPLING_DIR/${SAMPLING_NAME}_train.txt"
    echo "请先运行 2_create_sampling.sh 生成采样配置"
    exit 1
fi

echo "✅ 预检查通过，准备开始训练..."
echo ""

# ============================================================
# 训练命令
# ============================================================

# 启动训练 (使用Hugging Face Accelerate)
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    "$SCRIPT_DIR/training/train.py" \
    --in_model "$IN_MODEL" \
    --in_weights "$IN_WEIGHTS" \
    --out_weights "$OUT_WEIGHTS" \
    --trunc_t $TRUNC_T \
    --T $T \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
    --num_workers $NUM_WORKERS \
    --weight_decay $WEIGHT_DECAY \
    --n_frames $TOTAL_FRAMES \
    --learning_rate $LEARNING_RATE \
    --warmup_frames $WARMUP_FRAMES \
    --p_uncond $P_UNCOND \
    --min_btwn_goals $MIN_BTWN_GOALS \
    --max_btwn_goals $MAX_BTWN_GOALS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --val_freq $VAL_FREQ \
    --val_freq_begin $VAL_FREQ_BEGIN \
    --val_freq_switch_steps $VAL_FREQ_SWITCH \
    --save_freq $SAVE_FREQ \
    --save_each_val False \
    --sampling "$SAMPLING_NAME" \
    --sampling_dir "$SAMPLING_DIR" \
    --snapshot_every_n_frames $SNAPSHOT_EVERY_N_FRAMES \
    --max_grad_norm $MAX_GRAD_NORM \
    --val_every_nth 1

EXIT_STATUS=$?

# ============================================================
# 训练后处理
# ============================================================
if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练成功完成！"
    echo "=========================================="
    echo "输出权重: $OUT_WEIGHTS"
    echo "最佳权重: ${OUT_WEIGHTS/.weights/_best.weights}"
    echo "最新权重: ${OUT_WEIGHTS/.weights/_latest.weights}"
    echo ""
    echo "下一步操作:"
    echo "1. 查看TensorBoard: tensorboard --logdir $CHECKPOINT_DIR"
    echo "2. 测试模型: 修改 2_gen_vid_for_text_prompt.sh 使用新权重"
    echo "3. 评估性能: 参考 docs/guides/STEVE1_EVALUATION_GUIDE.md"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败 (退出码: $EXIT_STATUS)"
    echo "=========================================="
    echo "请检查:"
    echo "1. 日志输出中的错误信息"
    echo "2. 显存是否足够 (建议24GB+)"
    echo "3. 数据集路径是否正确"
    echo "4. 参考文档: docs/technical/STEVE1_TRAINING_ANALYSIS.md"
    echo "=========================================="
    exit $EXIT_STATUS
fi

