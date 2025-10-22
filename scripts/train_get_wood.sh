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
# 支持 minedojo 和 minedojo-x86 环境
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    # 已经在 conda 环境中
    if [[ "$CONDA_DEFAULT_ENV" == "minedojo"* ]]; then
        echo -e "${GREEN}✓ 使用环境: $CONDA_DEFAULT_ENV${NC}"
    else
        echo -e "${YELLOW}⚠️  当前环境: $CONDA_DEFAULT_ENV${NC}"
        echo -e "${YELLOW}   推荐使用: minedojo 或 minedojo-x86${NC}"
        echo -e "${YELLOW}   继续使用当前环境...${NC}"
    fi
else
    # 未在 conda 环境中，尝试激活
    if command -v conda &> /dev/null; then
        # 检测 conda 是否已初始化
        if declare -f conda &> /dev/null; then
            echo "激活 minedojo 环境..."
            conda activate minedojo 2>/dev/null || conda activate minedojo-x86 2>/dev/null || {
                echo -e "${RED}✗ 无法激活 minedojo 或 minedojo-x86 环境${NC}"
                echo -e "${YELLOW}请先运行: conda activate minedojo${NC}"
                exit 1
            }
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
                exit 1
            fi
            
            # 再次尝试激活
            if declare -f conda &> /dev/null; then
                conda activate minedojo 2>/dev/null || conda activate minedojo-x86 2>/dev/null || {
                    echo -e "${RED}✗ 无法激活环境${NC}"
                    exit 1
                }
                echo "✓ minedojo 环境已激活"
            fi
        fi
    else
        echo -e "${RED}✗ conda 未安装${NC}"
        exit 1
    fi
fi

# 默认参数
CONFIG_FILE="config/get_wood_config.yaml"  # 默认配置文件
PRESET=""  # 预设配置（test/quick/standard/long）
HEADLESS="true"  # 默认启用无头模式
OVERRIDES=()  # 命令行覆盖参数数组

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        # 预设配置快捷方式
        test)
            PRESET="test"
            shift
            ;;
        quick)
            PRESET="quick"
            shift
            ;;
        standard)
            PRESET="standard"
            shift
            ;;
        long)
            PRESET="long"
            shift
            ;;
        # 配置文件
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        # 预设参数
        --preset)
            PRESET="$2"
            shift 2
            ;;
        # 无头模式（在Shell层控制）
        --headless)
            HEADLESS="true"
            shift
            ;;
        --no-headless)
            HEADLESS="false"
            shift
            ;;
        # 覆盖参数（传递给Python脚本）
        --override)
            OVERRIDES+=("--override" "$2")
            shift 2
            ;;
        # 常用覆盖的快捷方式
        --mineclip)
            OVERRIDES+=("--override" "mineclip.use_mineclip=true")
            shift
            ;;
        --task-id)
            OVERRIDES+=("--override" "task.task_id=$2")
            shift 2
            ;;
        # 帮助信息
        -h|--help)
            echo "用法: $0 [预设] [选项]"
            echo ""
            echo "预设配置（推荐使用）:"
            echo "  test        快速测试 (10K步, 5-10分钟)"
            echo "  quick       快速训练 (50K步, 30-60分钟)"
            echo "  standard    标准训练 (200K步, 2-4小时)"
            echo "  long        长时间训练 (500K步, 5-10小时)"
            echo ""
            echo "基本选项:"
            echo "  --config FILE           指定配置文件 (默认: config/get_wood_config.yaml)"
            echo "  --preset PRESET         使用预设配置 (test/quick/standard/long)"
            echo "  --headless              启用无头模式，不显示游戏窗口 (默认)"
            echo "  --no-headless           禁用无头模式，显示游戏窗口（调试用）"
            echo ""
            echo "覆盖参数（高级用法）:"
            echo "  --override KEY=VALUE    覆盖配置文件中的参数（可多次使用）"
            echo "                          格式: section.key=value"
            echo "                          示例: --override mineclip.use_mineclip=true"
            echo ""
            echo "常用快捷方式:"
            echo "  --mineclip              启用MineCLIP (等同于 --override mineclip.use_mineclip=true)"
            echo "  --task-id TASK          设置任务ID (等同于 --override task.task_id=TASK)"
            echo "                          可选: harvest_1_log, harvest_1_log_forest, harvest_1_log_plains"
            echo ""
            echo "其他:"
            echo "  -h, --help              显示帮助"
            echo ""
            echo "示例:"
            echo "  $0                                    # 使用默认配置"
            echo "  $0 test                               # 快速测试模式"
            echo "  $0 standard --mineclip                # 标准训练+MineCLIP"
            echo "  $0 --config my_config.yaml            # 使用自定义配置"
            echo "  $0 quick --task-id harvest_1_log_forest --mineclip  # 森林快速训练"
            echo "  $0 --preset long --override training.learning_rate=0.001  # 长训练+自定义学习率"
            echo ""
            echo "配置文件:"
            echo "  所有训练参数现在通过YAML配置文件管理"
            echo "  默认配置: config/get_wood_config.yaml"
            echo "  可以通过 --override 覆盖任何配置项"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 设置Java无头模式
if [[ "$HEADLESS" == "true" ]]; then
    export JAVA_OPTS="-Djava.awt.headless=true"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MineDojo 获得木头训练${NC}"
echo -e "${BLUE}========================================${NC}"
echo "配置文件:   $CONFIG_FILE"
if [[ -n "$PRESET" ]]; then
    echo "预设模式:   $PRESET"
fi
# 显示无头模式状态
if [[ "$HEADLESS" == "true" ]]; then
    echo -e "无头模式:   ${GREEN}启用${NC}"
else
    echo -e "无头模式:   ${YELLOW}禁用（显示游戏窗口）${NC}"
fi
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    echo "参数覆盖:   ${#OVERRIDES[@]} 个"
fi
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

# 构建Python命令
PYTHON_CMD="python src/training/train_get_wood.py $CONFIG_FILE"

# 添加预设配置
if [[ -n "$PRESET" ]]; then
    PYTHON_CMD="$PYTHON_CMD --preset $PRESET"
fi

# 添加覆盖参数
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    PYTHON_CMD="$PYTHON_CMD ${OVERRIDES[@]}"
fi

# 执行训练
eval $PYTHON_CMD

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
echo -e "${YELLOW}下次训练:${NC}"
echo "  $0 $([[ -n "$PRESET" ]] && echo "$PRESET" || echo "")"
echo ""

