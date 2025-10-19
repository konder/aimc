#!/bin/bash
# 训练监控脚本
# 在一个终端显示日志，提示打开 TensorBoard

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MineDojo 训练监控${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查日志文件是否存在
if ! ls logs/training/training_*.log 1> /dev/null 2>&1; then
    echo -e "${YELLOW}警告: 未找到训练日志文件${NC}"
    echo -e "${YELLOW}请先启动训练: ./scripts/train_harvest.sh${NC}\n"
fi

echo -e "${GREEN}[1] 训练日志监控${NC}"
echo -e "    实时显示训练日志...\n"

# 显示最新日志文件
LATEST_LOG=$(ls -t logs/training/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo -e "${BLUE}监控文件: $LATEST_LOG${NC}\n"
    echo -e "${YELLOW}----------------------------------------${NC}"
fi

echo -e "\n${GREEN}[2] TensorBoard 可视化${NC}"
echo -e "    在另一个终端运行:"
echo -e "    ${YELLOW}tensorboard --logdir logs/tensorboard${NC}"
echo -e "    然后在浏览器打开: ${BLUE}http://localhost:6006${NC}\n"

echo -e "${GREEN}[3] 关键指标位置${NC}"
echo -e "    TensorBoard SCALARS 标签页:"
echo -e "    • ${YELLOW}rollout/ep_rew_mean${NC}  - 平均奖励 (最重要!)"
echo -e "    • ${YELLOW}train/policy_loss${NC}    - 策略损失"
echo -e "    • ${YELLOW}train/value_loss${NC}     - 价值损失"
echo -e "    • ${YELLOW}train/entropy_loss${NC}   - 熵损失"
echo -e "    • ${YELLOW}eval/mean_reward${NC}     - 评估奖励\n"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}开始监控日志 (Ctrl+C 退出)${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 监控日志
if [ -n "$LATEST_LOG" ]; then
    tail -f "$LATEST_LOG"
else
    echo -e "${YELLOW}等待训练日志文件创建...${NC}"
    # 等待日志文件出现
    while ! ls logs/training/training_*.log 1> /dev/null 2>&1; do
        sleep 2
    done
    LATEST_LOG=$(ls -t logs/training/training_*.log | head -1)
    echo -e "${GREEN}检测到日志文件: $LATEST_LOG${NC}\n"
    tail -f "$LATEST_LOG"
fi

