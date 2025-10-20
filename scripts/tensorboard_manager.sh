#!/bin/bash
# TensorBoard 管理工具

TB_PORT=6006
TB_LOGDIR="logs/tensorboard"

case "$1" in
    start)
        if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            echo "❌ TensorBoard 已在端口 $TB_PORT 运行"
            echo "   访问地址: http://localhost:$TB_PORT"
        else
            echo "🚀 启动 TensorBoard..."
            nohup ./scripts/run_minedojo_x86.sh tensorboard \
                --logdir="$TB_LOGDIR" \
                --port=$TB_PORT \
                --reload_interval=5 \
                --reload_multifile=true \
                > logs/tensorboard.log 2>&1 &
            sleep 2
            if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
                echo "✅ TensorBoard 已启动"
                echo "   访问地址: http://localhost:$TB_PORT"
            else
                echo "❌ 启动失败，查看日志: tail -f logs/tensorboard.log"
            fi
        fi
        ;;
    
    stop)
        if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            echo "🛑 停止 TensorBoard..."
            pkill -f "tensorboard.*$TB_PORT"
            sleep 1
            if ! lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
                echo "✅ TensorBoard 已停止"
            else
                echo "❌ 停止失败，请手动检查"
            fi
        else
            echo "ℹ️  TensorBoard 未在运行"
        fi
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
            PID=$(lsof -t -i:$TB_PORT)
            echo "✅ TensorBoard 正在运行"
            echo "   PID: $PID"
            echo "   端口: $TB_PORT"
            echo "   访问地址: http://localhost:$TB_PORT"
            echo "   日志: logs/tensorboard.log"
        else
            echo "❌ TensorBoard 未运行"
        fi
        ;;
    
    logs)
        if [ -f "logs/tensorboard.log" ]; then
            tail -f logs/tensorboard.log
        else
            echo "❌ 日志文件不存在"
        fi
        ;;
    
    open)
        if command -v open >/dev/null 2>&1; then
            if lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
                echo "🌐 在浏览器中打开 TensorBoard..."
                open "http://localhost:$TB_PORT"
            else
                echo "❌ TensorBoard 未运行，请先启动: $0 start"
            fi
        else
            echo "❌ 不支持的操作系统"
        fi
        ;;
    
    *)
        echo "TensorBoard 管理工具"
        echo ""
        echo "用法: $0 {start|stop|restart|status|logs|open}"
        echo ""
        echo "命令:"
        echo "  start    - 启动 TensorBoard"
        echo "  stop     - 停止 TensorBoard"
        echo "  restart  - 重启 TensorBoard"
        echo "  status   - 查看运行状态"
        echo "  logs     - 查看实时日志"
        echo "  open     - 在浏览器中打开"
        echo ""
        echo "示例:"
        echo "  $0 start     # 启动"
        echo "  $0 status    # 检查状态"
        echo "  $0 open      # 在浏览器中打开"
        exit 1
        ;;
esac

