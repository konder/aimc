#!/bin/bash
# GPU 解码详细诊断脚本

VIDEO=${1:-"/root/autodl-tmp/processed/clips/000MGe3tydc_564_567.mp4"}

echo "============================================================"
echo "GPU 解码详细诊断"
echo "============================================================"
echo "测试视频: $VIDEO"
echo ""

# 1. 检查视频编码格式
echo "1. 视频编码格式"
echo "------------------------------------------------------------"
ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,codec_long_name,pix_fmt,width,height \
    -of default=noprint_wrappers=1 \
    "$VIDEO"
echo ""

# 2. 测试 h264_cuvid
echo "2. 测试 h264_cuvid 解码器"
echo "------------------------------------------------------------"
ffmpeg -y \
    -hwaccel cuda \
    -hwaccel_device 0 \
    -c:v h264_cuvid \
    -resize 256x160 \
    -i "$VIDEO" \
    -f rawvideo \
    -pix_fmt rgb24 \
    -frames:v 1 \
    /tmp/test_h264_cuvid.raw 2>&1 | tail -20
echo ""

# 3. 测试 scale_cuda
echo "3. 测试 scale_cuda 滤镜"
echo "------------------------------------------------------------"
ffmpeg -y \
    -hwaccel cuda \
    -hwaccel_device 0 \
    -i "$VIDEO" \
    -vf scale_cuda=256:160 \
    -f rawvideo \
    -pix_fmt rgb24 \
    -frames:v 1 \
    /tmp/test_scale_cuda.raw 2>&1 | tail -20
echo ""

# 4. 测试通用 hwaccel
echo "4. 测试通用 hwaccel（不指定解码器）"
echo "------------------------------------------------------------"
ffmpeg -y \
    -hwaccel cuda \
    -hwaccel_device 0 \
    -i "$VIDEO" \
    -vf scale=256:160 \
    -f rawvideo \
    -pix_fmt rgb24 \
    -frames:v 1 \
    /tmp/test_hwaccel.raw 2>&1 | tail -20
echo ""

# 5. 检查可用的解码器
echo "5. 可用的 CUDA 解码器"
echo "------------------------------------------------------------"
ffmpeg -decoders 2>/dev/null | grep cuvid
echo ""

# 6. 检查 CUDA 库
echo "6. 检查 CUDA 库"
echo "------------------------------------------------------------"
ldconfig -p | grep cuda | head -10
echo ""

# 7. ffmpeg 编译配置
echo "7. ffmpeg 编译配置（CUDA 相关）"
echo "------------------------------------------------------------"
ffmpeg -version 2>&1 | grep -i "configuration" | grep -o "cuda[^ ]*\|nvenc[^ ]*\|nvdec[^ ]*\|cuvid[^ ]*"
echo ""

echo "============================================================"
echo "诊断完成"
echo "============================================================"

