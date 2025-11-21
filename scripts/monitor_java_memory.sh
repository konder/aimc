#!/bin/bash
# Java内存监控和自动清理脚本
# 监控Java进程内存，超过阈值时自动清理MineDojo saves

set -e

LOG_FILE="logs/java_memory_monitor.log"
ALERT_THRESHOLD_GB=8
CLEAN_THRESHOLD_GB=7
CHECK_INTERVAL=600  # 10分钟

# 创建日志目录
mkdir -p logs

echo "════════════════════════════════════════════════════════════"
echo "  Java内存监控启动"
echo "  警告阈值: ${ALERT_THRESHOLD_GB}GB"
echo "  清理阈值: ${CLEAN_THRESHOLD_GB}GB"
echo "  检查间隔: ${CHECK_INTERVAL}秒"
echo "  日志文件: ${LOG_FILE}"
echo "════════════════════════════════════════════════════════════"
echo ""

while true; do
    # 获取Java进程内存（KB）
    java_pids=$(pgrep -f "java.*Minecraft" || true)
    
    if [ -z "$java_pids" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 未检测到Java进程" | tee -a $LOG_FILE
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 计算所有Java进程的总内存
    total_mem_kb=0
    for pid in $java_pids; do
        mem_kb=$(ps -p $pid -o rss= 2>/dev/null || echo "0")
        total_mem_kb=$((total_mem_kb + mem_kb))
    done
    
    # 转换为GB
    java_mem_gb=$(echo "scale=2; $total_mem_kb / 1024 / 1024" | bc)
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 记录当前状态
    echo "$timestamp - Java内存: ${java_mem_gb}GB" | tee -a $LOG_FILE
    
    # 检查是否需要清理
    need_clean=$(echo "$java_mem_gb > $CLEAN_THRESHOLD_GB" | bc)
    if [ "$need_clean" -eq 1 ]; then
        echo "$timestamp - 内存达到${java_mem_gb}GB，开始清理..." | tee -a $LOG_FILE
        
        # 清理MineDojo saves
        saves_path="$HOME/.minedojo/saves"
        if [ -d "$saves_path" ]; then
            saves_size=$(du -sh "$saves_path" 2>/dev/null | cut -f1)
            echo "$timestamp - Saves目录大小: ${saves_size}" | tee -a $LOG_FILE
            
            rm -rf "$saves_path"/*
            echo "$timestamp - ✓ 已清理MineDojo saves目录" | tee -a $LOG_FILE
        fi
        
        # 触发Java GC（尝试）
        for pid in $java_pids; do
            if command -v jcmd &> /dev/null; then
                jcmd $pid GC.run 2>/dev/null || true
                echo "$timestamp - ✓ 已触发Java GC (PID: $pid)" | tee -a $LOG_FILE
            fi
        done
    fi
    
    # 检查是否超过警告阈值
    need_alert=$(echo "$java_mem_gb > $ALERT_THRESHOLD_GB" | bc)
    if [ "$need_alert" -eq 1 ]; then
        echo "$timestamp - ⚠️  警告: Java内存超过${ALERT_THRESHOLD_GB}GB!" | tee -a $LOG_FILE
        echo "$timestamp - 建议考虑停止评估并分批运行" | tee -a $LOG_FILE
    fi
    
    # 显示系统内存
    if command -v free &> /dev/null; then
        free_mem=$(free -h | grep Mem | awk '{print $7}')
        echo "$timestamp - 系统可用内存: ${free_mem}" | tee -a $LOG_FILE
    fi
    
    echo "---" >> $LOG_FILE
    
    sleep $CHECK_INTERVAL
done

