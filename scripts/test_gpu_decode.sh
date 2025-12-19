#!/bin/bash
# 测试 GPU 解码配置

if [ -z "$1" ]; then
    echo "用法: $0 <测试视频路径>"
    exit 1
fi

VIDEO="$1"

echo "========================================="
echo "测试 GPU 解码配置"
echo "========================================="
echo "测试视频: $VIDEO"
echo

# 方案 1: hwaccel cuda + hwdownload + CPU scale
echo "方案 1: hwaccel cuda + hwdownload + scale (推荐)"
ffmpeg -hwaccel cuda -hwaccel_device 0 \
       -hwaccel_output_format cuda \
       -ss 0 -i "$VIDEO" -t 5 \
       -vf 'hwdownload,format=nv12,scale=256:160' \
       -frames:v 10 \
       -f rawvideo -pix_fmt rgb24 \
       -loglevel error \
       /tmp/test_output.rgb 2>&1

if [ $? -eq 0 ] && [ -f /tmp/test_output.rgb ]; then
    SIZE=$(stat -c%s /tmp/test_output.rgb 2>/dev/null || stat -f%z /tmp/test_output.rgb 2>/dev/null)
    EXPECTED=$((256*160*3*10))
    echo "  ✅ 成功! 输出大小: $SIZE 字节 (期望: $EXPECTED)"
    rm -f /tmp/test_output.rgb
else
    echo "  ❌ 失败"
fi
echo

# 方案 2: 纯 hwaccel cuda + CPU scale
echo "方案 2: hwaccel cuda + scale (简化版)"
ffmpeg -hwaccel cuda -hwaccel_device 0 \
       -ss 0 -i "$VIDEO" -t 5 \
       -vf 'scale=256:160' \
       -frames:v 10 \
       -f rawvideo -pix_fmt rgb24 \
       -loglevel error \
       /tmp/test_output.rgb 2>&1

if [ $? -eq 0 ] && [ -f /tmp/test_output.rgb ]; then
    SIZE=$(stat -c%s /tmp/test_output.rgb 2>/dev/null || stat -f%z /tmp/test_output.rgb 2>/dev/null)
    EXPECTED=$((256*160*3*10))
    echo "  ✅ 成功! 输出大小: $SIZE 字节 (期望: $EXPECTED)"
    rm -f /tmp/test_output.rgb
else
    echo "  ❌ 失败"
fi
echo

# 方案 3: 纯 CPU（对比）
echo "方案 3: 纯 CPU 模式（对比）"
ffmpeg -ss 0 -i "$VIDEO" -t 5 \
       -vf 'scale=256:160' \
       -frames:v 10 \
       -f rawvideo -pix_fmt rgb24 \
       -loglevel error \
       /tmp/test_output.rgb 2>&1

if [ $? -eq 0 ] && [ -f /tmp/test_output.rgb ]; then
    SIZE=$(stat -c%s /tmp/test_output.rgb 2>/dev/null || stat -f%z /tmp/test_output.rgb 2>/dev/null)
    EXPECTED=$((256*160*3*10))
    echo "  ✅ 成功! 输出大小: $SIZE 字节 (期望: $EXPECTED)"
    rm -f /tmp/test_output.rgb
else
    echo "  ❌ 失败"
fi
echo

echo "========================================="
echo "建议"
echo "========================================="
echo "如果方案 1 或 2 成功，可以使用 GPU 模式"
echo "否则使用 CPU 模式（性能也很好）"
echo

