#!/bin/bash
# 训练样本录制脚本
# Training Sample Recording Script

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
CONFIG="config/eval_tasks.yaml"
TASK=""
LIST_TASKS=false
BASE_DIR="data/train_samples"
MAX_FRAMES=1000
FPS=20
CAMERA_DELTA=4
MOUSE_SENSITIVITY=0.5
FULLSCREEN=false

# 显示帮助信息
show_help() {
    echo -e "${BLUE}=========================================="
    echo "训练样本录制工具"
    echo -e "==========================================${NC}"
    echo ""
    echo "用法："
    echo "  $0 [选项]"
    echo ""
    echo "基本选项："
    echo "  --task TASK_ID          录制指定任务（如 harvest_1_log）"
    echo "  --list-tasks            列出所有可用任务"
    echo "  --config FILE           配置文件路径（默认: config/eval_tasks.yaml）"
    echo ""
    echo "录制参数："
    echo "  --base-dir DIR          输出目录（默认: data/train_samples）"
    echo "  --max-frames N          最大帧数（默认: 1000）"
    echo "  --fps N                 录制帧率（默认: 20）"
    echo "  --fullscreen            全屏显示（推荐，防止鼠标移出窗口）"
    echo ""
    echo "控制参数："
    echo "  --camera-delta N        相机灵敏度-键盘（默认: 4）"
    echo "  --mouse-sensitivity N   鼠标灵敏度（默认: 0.5，范围: 0.1-2.0）"
    echo ""
    echo "其他："
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "输出结构："
    echo "  data/train_samples/{task_id}/"
    echo "    trial1/"
    echo "      frames/        # 帧图像"
    echo "      actions.json   # 动作序列"
    echo "      trial_info.json"
    echo "      visual_embeds.pkl  # 自动生成的视觉嵌入"
    echo "    trial2/"
    echo "    ..."
    echo ""
    echo "示例："
    echo "  # 列出所有可用任务"
    echo "  $0 --list-tasks"
    echo ""
    echo "  # 录制 harvest_1_log 任务（全屏模式）"
    echo "  $0 --task harvest_1_log --fullscreen"
    echo ""
    echo "  # 录制 combat_chicken 任务"
    echo "  $0 --task combat_chicken"
    echo ""
    echo "控制说明："
    echo "  移动: W/A/S/D | 跳跃: Space | 攻击: F/左键"
    echo "  相机: 鼠标移动（快速）| 方向键（精确）"
    echo "  重录: Q | 退出: ESC | 全屏: F11"
    echo ""
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --list-tasks)
            LIST_TASKS=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --camera-delta)
            CAMERA_DELTA="$2"
            shift 2
            ;;
        --mouse-sensitivity)
            MOUSE_SENSITIVITY="$2"
            shift 2
            ;;
        --fullscreen)
            FULLSCREEN=true
            shift
            ;;
        *)
            echo -e "${RED}错误: 未知参数 '$1'${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 构建Python命令
PYTHON_CMD="python src/evaluation/record_samples.py"
PYTHON_CMD="$PYTHON_CMD --config $CONFIG"

if [ "$LIST_TASKS" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --list-tasks"
elif [ -n "$TASK" ]; then
    PYTHON_CMD="$PYTHON_CMD --task $TASK"
    PYTHON_CMD="$PYTHON_CMD --base-dir $BASE_DIR"
    PYTHON_CMD="$PYTHON_CMD --max-frames $MAX_FRAMES"
    PYTHON_CMD="$PYTHON_CMD --fps $FPS"
    PYTHON_CMD="$PYTHON_CMD --camera-delta $CAMERA_DELTA"
    PYTHON_CMD="$PYTHON_CMD --mouse-sensitivity $MOUSE_SENSITIVITY"
    
    if [ "$FULLSCREEN" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --fullscreen"
    fi
else
    echo -e "${RED}错误: 请指定 --task 或 --list-tasks${NC}"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 显示配置信息
if [ "$LIST_TASKS" != true ]; then
    echo -e "${BLUE}=========================================="
    echo "训练样本录制"
    echo -e "==========================================${NC}"
    echo -e "${GREEN}项目根目录:${NC} $PROJECT_ROOT"
    echo -e "${GREEN}配置文件:${NC} $CONFIG"
    echo -e "${GREEN}任务ID:${NC} $TASK"
    echo -e "${GREEN}输出目录:${NC} $BASE_DIR/$TASK/"
    echo -e "${GREEN}最大帧数:${NC} $MAX_FRAMES"
    echo -e "${GREEN}录制帧率:${NC} $FPS FPS"
    echo -e "${GREEN}显示模式:${NC} $([ "$FULLSCREEN" = true ] && echo "全屏（推荐）" || echo "窗口")"
    echo -e "${GREEN}鼠标灵敏度:${NC} $MOUSE_SENSITIVITY"
    echo -e "${GREEN}自动生成:${NC} visual_embeds.pkl（录制完成后）"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
fi

# 检查Python环境
if ! python -c "import minedojo" &> /dev/null; then
    echo -e "${YELLOW}警告: 未检测到 minedojo 模块${NC}"
    echo -e "${YELLOW}提示: 请确保已激活 minedojo 环境${NC}"
    echo -e "${YELLOW}      conda activate minedojo${NC}"
    echo ""
fi

# 执行录制
echo -e "${YELLOW}执行命令:${NC} $PYTHON_CMD"
echo ""
exec $PYTHON_CMD
