#!/bin/bash
# MineDojo 便捷别名设置脚本
# 执行此脚本将自动向你的shell配置文件添加便捷别名

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MineDojo 便捷别名设置${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检测shell类型
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.profile"
fi

echo -e "${YELLOW}检测到的shell配置文件: ${SHELL_RC}${NC}\n"

# 要添加的别名配置
ALIAS_CONFIG="
# ========================================
# MineDojo 快捷命令 (自动生成)
# ========================================
alias minedojo-shell='${SCRIPT_DIR}/run_minedojo_x86.sh'
alias minedojo-run='${SCRIPT_DIR}/minedojo_quick.sh'
alias minedojo-hello='${SCRIPT_DIR}/run_minedojo_x86.sh python ${PROJECT_ROOT}/src/hello_minedojo.py'

# MineDojo项目快速导航
alias cdaimc='cd ${PROJECT_ROOT}'
alias aimc-logs='tail -f ${PROJECT_ROOT}/logs/*.log'
"

# 检查是否已经添加过
if grep -q "# MineDojo 快捷命令" "$SHELL_RC" 2>/dev/null; then
    echo -e "${YELLOW}警告: 配置文件中已存在MineDojo别名${NC}"
    echo -e "${YELLOW}是否要更新? (y/n)${NC}"
    read -r response
    if [ "$response" != "y" ]; then
        echo -e "${BLUE}取消设置${NC}"
        exit 0
    fi
    # 删除旧的配置
    sed -i.bak '/# MineDojo 快捷命令/,/^$/d' "$SHELL_RC"
fi

# 添加新配置
echo "$ALIAS_CONFIG" >> "$SHELL_RC"

echo -e "${GREEN}✓ 别名已成功添加到 ${SHELL_RC}${NC}\n"
echo -e "${BLUE}可用的快捷命令:${NC}"
echo -e "  ${GREEN}minedojo-shell${NC}     - 启动x86交互式shell环境"
echo -e "  ${GREEN}minedojo-run${NC}       - 运行Python脚本: minedojo-run <script.py>"
echo -e "  ${GREEN}minedojo-hello${NC}     - 快速运行hello_minedojo.py测试"
echo -e "  ${GREEN}cdaimc${NC}             - 快速切换到项目目录"
echo -e "  ${GREEN}aimc-logs${NC}          - 实时查看项目日志\n"

echo -e "${YELLOW}请执行以下命令使别名生效:${NC}"
echo -e "${BLUE}source ${SHELL_RC}${NC}\n"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}设置完成！${NC}"
echo -e "${BLUE}========================================${NC}"

