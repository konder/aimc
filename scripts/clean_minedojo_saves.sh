#!/bin/bash
# 快速清理MineDojo saves目录
# 释放Java内存占用

set -e

SAVES_PATH="$HOME/.minedojo/saves"

echo "════════════════════════════════════════════════════════════"
echo "  MineDojo Saves 清理工具"
echo "════════════════════════════════════════════════════════════"
echo ""

# 检查目录是否存在
if [ ! -d "$SAVES_PATH" ]; then
    echo "✓ Saves目录不存在，无需清理"
    exit 0
fi

# 显示当前大小
echo "📊 当前saves目录:"
du -sh "$SAVES_PATH"
echo ""

# 显示文件数量
file_count=$(find "$SAVES_PATH" -type f 2>/dev/null | wc -l)
echo "📁 文件数量: ${file_count}"
echo ""

# 确认清理
read -p "❓ 是否清理saves目录? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消清理"
    exit 0
fi

# 执行清理
echo "🗑️  清理中..."
rm -rf "$SAVES_PATH"/*

# 验证
remaining=$(find "$SAVES_PATH" -type f 2>/dev/null | wc -l)
echo ""
echo "✅ 清理完成！"
echo "剩余文件: ${remaining}"

# 显示Java内存
echo ""
echo "📊 当前Java内存使用:"
ps aux | grep -E "java.*Minecraft" | grep -v grep | awk '{printf "  进程 %s: %.2fGB\n", $2, $6/1024/1024}' || echo "  未检测到Java进程"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "提示: 建议定期运行此脚本，或使用自动监控脚本"
echo "      bash scripts/monitor_java_memory.sh &"
echo "════════════════════════════════════════════════════════════"

