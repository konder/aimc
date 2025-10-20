#!/bin/bash
# 获得木头训练脚本
# 使用MineDojo内置任务 harvest_1_log

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 激活 conda 环境
if command -v conda &> /dev/null; then
    # 检测 conda 是否已初始化
    if declare -f conda &> /dev/null; then
        # conda 函数存在，可以直接激活
        if [[ "$CONDA_DEFAULT_ENV" != "minedojo" ]]; then
            echo "激活 minedojo 环境..."
            conda activate minedojo
        else
            echo "已在 minedojo 环境中"
        fi
    else
        # conda 未初始化，尝试初始化
        echo "正在初始化 conda..."
        
        # 尝试找到 conda.sh
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [[ -n "$CONDA_BASE" ]] && [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
            source "/opt/anaconda3/etc/profile.d/conda.sh"
        elif [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "/opt/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "/usr/local/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]]; then
            source "/usr/local/Caskroom/miniforge/base/etc/profile.d/conda.sh"
        else
            echo -e "${YELLOW}⚠️  无法找到 conda.sh，请手动激活环境：${NC}"
            echo -e "${YELLOW}   source ~/miniconda3/etc/profile.d/conda.sh${NC}"
            echo -e "${YELLOW}   conda activate minedojo${NC}"
            echo -e "${YELLOW}   然后重新运行此脚本${NC}"
            echo ""
        fi
        
        # 再次尝试激活
        if declare -f conda &> /dev/null; then
            conda activate minedojo
            echo "✓ minedojo 环境已激活"
        fi
    fi
else
    echo -e "${RED}✗ conda 未安装${NC}"
    exit 1
fi

# 设置Java无头模式（性能优化）
export JAVA_OPTS="-Djava.awt.headless=true"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 默认参数
MODE="standard"
TIMESTEPS=200000
USE_MINECLIP=""
DEVICE="auto"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        test)
            MODE="test"
            TIMESTEPS=10000
            shift
            ;;
        quick)
            MODE="quick"
            TIMESTEPS=50000
            shift
            ;;
        long)
            MODE="long"
            TIMESTEPS=500000
            shift
            ;;
        --mineclip)
            USE_MINECLIP="--use-mineclip"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [模式] [选项]"
            echo ""
            echo "模式:"
            echo "  test        快速测试 (10K步, 5-10分钟)"
            echo "  quick       快速训练 (50K步, 30-60分钟)"
            echo "  standard    标准训练 (200K步, 2-4小时) [默认]"
            echo "  long        长时间训练 (500K步, 5-10小时)"
            echo ""
            echo "选项:"
            echo "  --mineclip          使用MineCLIP加速（推荐，3-5倍加速）"
            echo "  --device DEVICE     设备: auto/cpu/cuda/mps (默认: auto)"
            echo "  --timesteps N       自定义总步数"
            echo "  -h, --help          显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                  # 标准训练"
            echo "  $0 --mineclip       # 使用MineCLIP加速"
            echo "  $0 test             # 快速测试"
            echo "  $0 long --mineclip  # 长时间训练+MineCLIP"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MineDojo 获得木头训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo "任务:       harvest_1_log (获得1个原木)"
echo "模式:       $MODE"
echo "总步数:     $TIMESTEPS"
echo "MineCLIP:   ${USE_MINECLIP:-禁用}"
echo "设备:       $DEVICE"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# 启动 TensorBoard (如果未运行)
# ============================================================================
echo -e "${YELLOW}[1/2] 检查 TensorBoard...${NC}"

TB_PORT=6006
TB_LOGDIR="$PROJECT_ROOT/logs/tensorboard"

# 检查 TensorBoard 是否已在运行
if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${GREEN}  ✓ TensorBoard 已在运行 (端口 $TB_PORT)${NC}"
else
    echo -e "  启动 TensorBoard..."
    
    # 在后台启动 TensorBoard
    nohup tensorboard --logdir "$TB_LOGDIR" --port $TB_PORT \
        > logs/tensorboard.log 2>&1 &
    
    TB_PID=$!
    sleep 2
    
    # 检查是否成功启动
    if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}  ✓ TensorBoard 已启动 (PID: $TB_PID, 端口: $TB_PORT)${NC}"
        echo -e "${GREEN}  ✓ 访问地址: http://localhost:$TB_PORT${NC}"
    else
        echo -e "${RED}  ✗ TensorBoard 启动失败，查看日志: logs/tensorboard.log${NC}"
    fi
fi
echo ""

# ============================================================================
# 运行训练
# ============================================================================
echo -e "${YELLOW}[2/2] 开始训练...${NC}"
echo ""
python src/training/train_get_wood.py \
    --total-timesteps "$TIMESTEPS" \
    --device "$DEVICE" \
    $USE_MINECLIP \
    --save-freq 10000 \
    --checkpoint-dir "checkpoints/get_wood" \
    --tensorboard-dir "logs/tensorboard" \
    --log-dir "logs/training"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 训练完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}查看训练结果:${NC}"
echo "  TensorBoard: http://localhost:$TB_PORT"
echo ""
echo -e "${YELLOW}模型保存在:${NC}"
echo "  checkpoints/get_wood/"
echo ""
echo -e "${YELLOW}评估模型:${NC}"
echo "  python scripts/evaluate_get_wood.py"
echo ""

