#!/bin/bash
# DAgger Web 控制台管理脚本
#
# 使用方法:
#   bash scripts/run_web.sh [start|stop|status]
#
# 参数:
#   start  - 启动 Web 控制台
#   stop   - 停止 Web 控制台
#   status - 显示运行状态 (默认)

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取命令参数，默认为 status
COMMAND=${1:-status}

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 打印标题
print_header() {
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
}

# 检查Web服务器状态
check_status() {
    local WEB_PIDS=$(pgrep -f "python.*src.web.app" 2>/dev/null || true)
    
    if [ -z "$WEB_PIDS" ]; then
        return 1  # 未运行
    else
        echo "$WEB_PIDS"
        return 0  # 运行中
    fi
}

# 显示状态
show_status() {
    print_header "DAgger Web 控制台 - 状态"
    
    if WEB_PIDS=$(check_status); then
        echo -e "${GREEN}✅ Web 控制台正在运行${NC}"
        echo ""
        echo -e "${GREEN}运行中的进程:${NC}"
        for pid in $WEB_PIDS; do
            echo -e "  PID: ${YELLOW}$pid${NC}"
            ps -p $pid -o command= | head -1 | sed 's/^/    /'
        done
        echo ""
        
        # 检查端口
        PORT_CHECK=$(lsof -ti:5000 2>/dev/null || true)
        if [ -n "$PORT_CHECK" ]; then
            echo -e "${GREEN}端口: ${YELLOW}5000${NC}"
            echo -e "${GREEN}访问地址: ${YELLOW}http://localhost:5000${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Web 控制台未运行${NC}"
        echo ""
        echo -e "使用 ${BLUE}bash scripts/run_web.sh start${NC} 启动服务"
    fi
    
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
}

# 启动Web服务器
start_web() {
    print_header "DAgger Web 控制台 - 启动"
    
    # 检查是否已经在运行
    if check_status > /dev/null; then
        echo -e "${YELLOW}⚠️  Web 控制台已经在运行${NC}"
        echo ""
        show_status
        exit 0
    fi
    
    # 检查conda环境
    if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "minedojo"* ]]; then
        echo -e "${YELLOW}⚠️  警告: 未检测到 minedojo 环境${NC}"
        echo "请先激活环境: conda activate minedojo-x86"
        echo ""
        read -p "是否继续? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # 检查Web依赖
    echo "检查 Web 依赖..."
    if ! python -c "import flask" 2>/dev/null; then
        echo ""
        echo -e "${RED}❌ 缺少 Flask 依赖！${NC}"
        echo ""
        echo "请先安装依赖："
        echo "  pip install -r requirements.txt"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}✅ 依赖检查通过${NC}"
    
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}启动服务器...${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
    echo -e "${GREEN}访问地址: ${YELLOW}http://localhost:5000${NC}"
    echo -e "${YELLOW}按 Ctrl+C 停止服务器${NC}"
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
    
    # 启动服务器
    python -m src.web.app
}

# 停止Web服务器
stop_web() {
    print_header "DAgger Web 控制台 - 停止"
    
    # 查找运行中的Web服务器进程
    if ! WEB_PIDS=$(check_status); then
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
    if check_status > /dev/null; then
        echo ""
        echo -e "${RED}❌ 部分进程仍在运行${NC}"
        exit 1
    else
        echo ""
        echo -e "${GREEN}✅ Web 控制台已成功停止${NC}"
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
}

# 显示使用帮助
show_help() {
    echo "DAgger Web 控制台管理脚本"
    echo ""
    echo "使用方法:"
    echo "  bash scripts/run_web.sh [COMMAND]"
    echo ""
    echo "命令:"
    echo "  start   - 启动 Web 控制台"
    echo "  stop    - 停止 Web 控制台"
    echo "  status  - 显示运行状态 (默认)"
    echo "  help    - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash scripts/run_web.sh start"
    echo "  bash scripts/run_web.sh stop"
    echo "  bash scripts/run_web.sh"
    echo ""
}

# 主逻辑
case "$COMMAND" in
    start)
        start_web
        ;;
    stop)
        stop_web
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ 未知命令: $COMMAND${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

