#!/bin/bash
#
# 停止 DAgger Web 控制台
#
# 使用方法:
#   bash web/stop_web.sh
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}停止 DAgger Web 控制台${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# 查找运行中的Web服务器进程
WEB_PIDS=$(pgrep -f "python.*src.web.app" 2>/dev/null || true)

if [ -z "$WEB_PIDS" ]; then
    echo -e "${YELLOW}⚠️  没有运行中的 Web 控制台${NC}"
    echo ""
    exit 0
fi

# 显示找到的进程
echo -e "${GREEN}找到运行中的进程:${NC}"
for pid in $WEB_PIDS; do
    echo -e "  PID: ${YELLOW}$pid${NC}"
    ps -p $pid -o command= | head -1 | sed 's/^/    /'
done
echo ""

# 停止进程
echo -e "${GREEN}正在停止 Web 服务器...${NC}"

for pid in $WEB_PIDS; do
    # 先尝试优雅停止 (SIGTERM)
    kill $pid 2>/dev/null || true
    
    # 等待进程结束
    sleep 1
    
    # 检查进程是否还在运行
    if ps -p $pid > /dev/null 2>&1; then
        echo -e "${YELLOW}进程 $pid 未响应，强制停止...${NC}"
        kill -9 $pid 2>/dev/null || true
    fi
done

# 等待一下确保进程完全停止
sleep 1

# 验证是否停止成功
REMAINING_PIDS=$(pgrep -f "python.*web/app.py" 2>/dev/null || true)

if [ -z "$REMAINING_PIDS" ]; then
    echo ""
    echo -e "${GREEN}✅ Web 控制台已成功停止${NC}"
else
    echo ""
    echo -e "${RED}❌ 部分进程仍在运行:${NC}"
    echo "$REMAINING_PIDS"
    exit 1
fi

# 检查端口是否释放
echo ""
echo -e "${BLUE}检查端口状态...${NC}"
PORT_CHECK=$(lsof -ti:5000 2>/dev/null || true)

if [ -z "$PORT_CHECK" ]; then
    echo -e "${GREEN}✅ 端口 5000 已释放${NC}"
else
    echo -e "${YELLOW}⚠️  端口 5000 仍被占用 (PID: $PORT_CHECK)${NC}"
    echo -e "${YELLOW}   如果不是 Web 控制台进程，可能是其他程序使用了该端口${NC}"
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo ""

