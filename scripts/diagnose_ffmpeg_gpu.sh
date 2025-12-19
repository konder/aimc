#!/bin/bash
# FFmpeg GPU 加速诊断脚本

echo "========================================"
echo "FFmpeg GPU 加速诊断"
echo "========================================"
echo

# 1. 检查 ffmpeg 版本
echo "1. FFmpeg 版本:"
ffmpeg -version | head -1
echo

# 2. 检查硬件加速支持
echo "2. 硬件加速支持:"
ffmpeg -hwaccels
echo

# 3. 检查 CUDA 解码器
echo "3. CUDA 视频解码器 (cuvid):"
if ffmpeg -decoders 2>/dev/null | grep -q cuvid; then
    ffmpeg -decoders 2>/dev/null | grep cuvid
else
    echo "  ❌ 未找到 cuvid 解码器"
    echo "  提示: FFmpeg 需要编译时启用 --enable-cuda-nvcc --enable-cuvid"
fi
echo

# 4. 检查 CUDA 滤镜
echo "4. CUDA 视频滤镜:"
if ffmpeg -filters 2>/dev/null | grep -q cuda; then
    ffmpeg -filters 2>/dev/null | grep cuda
else
    echo "  ❌ 未找到 cuda 滤镜"
    echo "  提示: FFmpeg 需要编译时启用 --enable-cuda-nvcc --enable-libnpp"
fi
echo

# 5. 检查 NVIDIA GPU
echo "5. NVIDIA GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
else
    echo "  ❌ nvidia-smi 不可用"
fi
echo

# 6. 测试视频（如果提供）
if [ -n "$1" ]; then
    echo "6. 测试 GPU 解码: $1"
    echo
    
    echo "  测试 1: hwaccel cuda + scale_cuda"
    if ffmpeg -hwaccel cuda -hwaccel_device 0 -i "$1" -vf scale_cuda=256:160 -frames:v 1 -f null - 2>&1 | grep -q "error"; then
        echo "  ❌ 失败"
    else
        echo "  ✅ 成功"
    fi
    echo
    
    echo "  测试 2: hwaccel cuda + scale (CPU)"
    if ffmpeg -hwaccel cuda -hwaccel_device 0 -i "$1" -vf scale=256:160 -frames:v 1 -f null - 2>&1 | grep -q "error"; then
        echo "  ❌ 失败"
    else
        echo "  ✅ 成功"
    fi
    echo
    
    echo "  测试 3: h264_cuvid 解码器"
    if ffmpeg -c:v h264_cuvid -i "$1" -vf scale=256:160 -frames:v 1 -f null - 2>&1 | grep -q "error"; then
        echo "  ❌ 失败"
    else
        echo "  ✅ 成功"
    fi
    echo
fi

echo "========================================"
echo "诊断建议"
echo "========================================"
echo

# 分析并给出建议
has_hwaccel=$(ffmpeg -hwaccels 2>/dev/null | grep -c cuda)
has_cuvid=$(ffmpeg -decoders 2>/dev/null | grep -c cuvid)
has_cuda_filter=$(ffmpeg -filters 2>/dev/null | grep -c cuda)

if [ "$has_hwaccel" -eq 0 ]; then
    echo "❌ FFmpeg 不支持 CUDA 硬件加速"
    echo "   解决: 需要重新编译 FFmpeg 或安装支持 CUDA 的版本"
    echo
elif [ "$has_cuvid" -eq 0 ] && [ "$has_cuda_filter" -eq 0 ]; then
    echo "⚠️  FFmpeg 支持 hwaccel cuda，但缺少 cuvid 解码器和 cuda 滤镜"
    echo "   说明: 这种配置下 GPU 加速效果有限"
    echo "   建议: 使用 CPU 模式（性能已经很好）"
    echo
else
    echo "✅ FFmpeg GPU 配置看起来正常"
    if [ -n "$1" ]; then
        echo "   请查看上面的测试结果"
    else
        echo "   建议: 使用 --use-gpu 参数测试实际视频"
    fi
    echo
fi

echo "========================================"
echo "使用建议"
echo "========================================"
echo
echo "CPU 模式（推荐）:"
echo "  python src/utils/generate_clip4mc_training_datas.py \\"
echo "      --metadata data/training/metadata.json \\"
echo "      --output-dir data/training/clip4mc \\"
echo "      --num-workers 8"
echo
echo "GPU 模式（如果测试成功）:"
echo "  python src/utils/generate_clip4mc_training_datas.py \\"
echo "      --metadata data/training/metadata.json \\"
echo "      --output-dir data/training/clip4mc \\"
echo "      --num-workers 4 \\"
echo "      --use-gpu"
echo
echo "========================================"

