#!/bin/bash
#
# Steve1 策略评估运行脚本
#
# 策略评估 vs 结果评估：
# - 策略评估（本脚本）：分析Prior和Policy模型的策略质量
# - 结果评估（run_evaluation.sh）：评估任务成功率
#
# 使用方法:
#   bash scripts/run_policy_evaluation.sh
#

# macOS M系列兼容性设置
export JAVA_OPTS="-Djava.awt.headless=true"
export KMP_DUPLICATE_LIB_OK=TRUE

# 默认参数
MODE="policy"  # prior=仅Prior评估, policy=完整评估
CONFIG="config/eval_tasks_comprehensive.yaml"
TASK_SET="harvest_tasks"
MAX_TASKS=3
N_TRIALS=3
OUTPUT_DIR="results/policy_evaluation"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --task-set)
            TASK_SET="$2"
            shift 2
            ;;
        --max-tasks)
            MAX_TASKS="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 验证mode参数
if [[ "$MODE" != "prior" && "$MODE" != "policy" ]]; then
    echo "错误: --mode 必须是 'prior' 或 'policy'"
    echo "  prior:  仅评估Prior模型（快速）"
    echo "  policy: 完整评估Prior+Policy+端到端（默认）"
    exit 1
fi

echo "========================================="
echo "Steve1 策略评估"
echo "========================================="
echo "评估模式: $MODE"
if [ "$MODE" = "prior" ]; then
    echo "  └─ 仅Prior评估（快速）"
else
    echo "  └─ 完整评估（Prior + Policy + 端到端）"
fi
echo "配置文件: $CONFIG"
echo "任务集: $TASK_SET"
echo "最大任务数: $MAX_TASKS"
echo "试验次数: $N_TRIALS"
echo "输出目录: $OUTPUT_DIR"
echo "========================================="

# 运行策略评估
python src/evaluation/policy_eval_framework.py \
    --mode "$MODE" \
    --config "$CONFIG" \
    --task-set "$TASK_SET" \
    --max-tasks "$MAX_TASKS" \
    --n-trials "$N_TRIALS" \
    --output-dir "$OUTPUT_DIR"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 策略评估完成！"
    echo ""
    echo "📊 查看结果:"
    echo "   - HTML报告: open $OUTPUT_DIR/policy_evaluation_report.html"
    echo "   - JSON数据: $OUTPUT_DIR/summary_report.json"
    echo ""
    echo "📚 详细说明:"
    echo "   - 指标解释: docs/guides/DEEP_EVALUATION_METRICS_EXPLAINED.md"
    echo "   - 使用指南: docs/guides/STEVE1_DEEP_EVALUATION_GUIDE.md"
else
    echo ""
    echo "❌ 策略评估失败，请检查日志"
    exit 1
fi

