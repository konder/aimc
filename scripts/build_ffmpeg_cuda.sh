#!/bin/bash
# 从源码编译带 CUDA 支持的 ffmpeg

set -e

echo "============================================================"
echo "编译带 CUDA 支持的 ffmpeg"
echo "============================================================"

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc 未找到，请先安装 CUDA toolkit"
    exit 1
fi

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PREFIX=${CONDA_PREFIX:-/root/miniconda3/envs/minedojo-x86}

echo "CUDA_HOME: $CUDA_HOME"
echo "安装路径: $PREFIX"
echo ""

# 安装依赖
echo "1. 安装编译依赖..."
conda install -y nasm yasm pkg-config

# 安装 nv-codec-headers
echo "2. 安装 nv-codec-headers..."
cd /tmp
if [ -d "nv-codec-headers" ]; then
    rm -rf nv-codec-headers
fi
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make install PREFIX=$PREFIX

# 下载 ffmpeg
echo "3. 下载 ffmpeg 6.1.1..."
cd /tmp
if [ -d "ffmpeg" ]; then
    rm -rf ffmpeg
fi
wget -q https://ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz
tar xf ffmpeg-6.1.1.tar.xz
cd ffmpeg-6.1.1

# 配置编译选项
echo "4. 配置..."
./configure \
    --prefix=$PREFIX \
    --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc \
    --enable-libnpp \
    --enable-gpl \
    --enable-nonfree \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libopus \
    --enable-libmp3lame \
    --extra-cflags=-I$CUDA_HOME/include \
    --extra-ldflags=-L$CUDA_HOME/lib64 \
    --nvccflags="-gencode arch=compute_89,code=sm_89" \
    --disable-static \
    --enable-shared

# 编译
echo "5. 编译（需要 10-20 分钟）..."
make -j$(nproc)

# 安装
echo "6. 安装..."
make install

# 验证
echo ""
echo "============================================================"
echo "✓ 编译完成！"
echo "============================================================"
ffmpeg -version | head -1
ffmpeg -decoders | grep cuvid
echo ""

