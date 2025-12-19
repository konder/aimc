#!/bin/bash
# FFmpeg with CUDA 支持编译安装脚本
# 适用于 Ubuntu 20.04+ with NVIDIA GPU

set -e  # 遇到错误立即退出

echo "========================================="
echo "FFmpeg CUDA 编译安装脚本"
echo "========================================="
echo

# 配置
FFMPEG_VERSION="6.0"
INSTALL_PREFIX="/usr/local"
BUILD_DIR="/tmp/ffmpeg_build_$$"
NPROC=$(nproc)

echo "配置:"
echo "  FFmpeg 版本: $FFMPEG_VERSION"
echo "  安装目录: $INSTALL_PREFIX"
echo "  构建目录: $BUILD_DIR"
echo "  并行任务数: $NPROC"
echo

# =========================================
# 步骤 1: 卸载现有 FFmpeg
# =========================================
echo "步骤 1: 卸载现有 FFmpeg..."
echo

# 卸载 conda 的 FFmpeg
if command -v conda &> /dev/null; then
    echo "  卸载 conda FFmpeg..."
    conda remove ffmpeg -y 2>/dev/null || true
fi

# 卸载 apt 的 FFmpeg
if command -v apt-get &> /dev/null; then
    echo "  卸载 apt FFmpeg..."
    sudo apt-get remove -y ffmpeg 2>/dev/null || true
    sudo apt-get autoremove -y 2>/dev/null || true
fi

echo "  ✅ 现有 FFmpeg 已卸载"
echo

# =========================================
# 步骤 2: 安装编译依赖
# =========================================
echo "步骤 2: 安装编译依赖..."
echo

sudo apt-get update

# 基础编译工具
echo "  安装基础工具..."
sudo apt-get install -y \
    build-essential \
    pkg-config \
    yasm \
    cmake \
    libtool \
    libc6 \
    libc6-dev \
    unzip \
    wget \
    git \
    autoconf \
    automake

# 核心库
echo "  安装核心库..."
sudo apt-get install -y \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libvorbis-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libnuma-dev

# NVIDIA CUDA 开发工具（如果没有）
echo "  检查 CUDA..."
if [ ! -d "/usr/local/cuda" ]; then
    echo "  ⚠️  警告: 未找到 /usr/local/cuda"
    echo "  请确保已安装 CUDA Toolkit"
    echo "  继续执行，但可能会失败..."
fi

echo "  ✅ 依赖安装完成"
echo

# =========================================
# 步骤 3: 安装 NVIDIA Video Codec SDK
# =========================================
echo "步骤 3: 安装 NVIDIA Video Codec SDK..."
echo

cd /tmp
if [ ! -d "nv-codec-headers" ]; then
    echo "  下载 nv-codec-headers..."
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
fi

cd nv-codec-headers
git checkout n12.1.14.0 2>/dev/null || git checkout sdk/12.1 2>/dev/null || true

echo "  编译并安装..."
make
sudo make install

echo "  ✅ NVIDIA Video Codec SDK 安装完成"
echo

# =========================================
# 步骤 4: 下载 FFmpeg 源码
# =========================================
echo "步骤 4: 下载 FFmpeg 源码..."
echo

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "  下载 FFmpeg $FFMPEG_VERSION..."
wget -q --show-progress "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2"

echo "  解压..."
tar -xf "ffmpeg-${FFMPEG_VERSION}.tar.bz2"
cd "ffmpeg-${FFMPEG_VERSION}"

echo "  ✅ FFmpeg 源码准备完成"
echo

# =========================================
# 步骤 5: 配置编译选项
# =========================================
echo "步骤 5: 配置编译选项..."
echo

# 设置 CUDA 路径
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo "  配置 FFmpeg..."
./configure \
    --prefix="$INSTALL_PREFIX" \
    --enable-gpl \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc \
    --enable-libnpp \
    --extra-cflags="-I/usr/local/cuda/include" \
    --extra-ldflags="-L/usr/local/cuda/lib64" \
    --enable-libass \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-shared \
    --disable-static \
    --disable-doc

if [ $? -ne 0 ]; then
    echo "  ❌ 配置失败！"
    echo "  请检查错误信息"
    exit 1
fi

echo "  ✅ 配置完成"
echo

# =========================================
# 步骤 6: 编译
# =========================================
echo "步骤 6: 编译 FFmpeg（约需 10-30 分钟）..."
echo

make -j"$NPROC"

if [ $? -ne 0 ]; then
    echo "  ❌ 编译失败！"
    exit 1
fi

echo "  ✅ 编译完成"
echo

# =========================================
# 步骤 7: 安装
# =========================================
echo "步骤 7: 安装 FFmpeg..."
echo

sudo make install

# 更新库缓存
sudo ldconfig

echo "  ✅ 安装完成"
echo

# =========================================
# 步骤 8: 验证安装
# =========================================
echo "步骤 8: 验证安装..."
echo

echo "  FFmpeg 版本:"
ffmpeg -version | head -1

echo
echo "  硬件加速支持:"
ffmpeg -hwaccels

echo
echo "  CUDA 解码器:"
ffmpeg -decoders 2>/dev/null | grep cuvid | head -5

echo
echo "  CUDA 编码器:"
ffmpeg -encoders 2>/dev/null | grep nvenc | head -3

echo
echo "  CUDA 滤镜:"
ffmpeg -filters 2>/dev/null | grep cuda

echo

# =========================================
# 步骤 9: 清理
# =========================================
echo "步骤 9: 清理构建文件..."
echo

cd /
rm -rf "$BUILD_DIR"

echo "  ✅ 清理完成"
echo

# =========================================
# 完成
# =========================================
echo "========================================="
echo "✅ FFmpeg with CUDA 安装完成！"
echo "========================================="
echo
echo "测试命令:"
echo "  cd /root/autodl-tmp/aimc"
echo "  bash scripts/test_gpu_decode.sh /root/autodl-tmp/TEST2000/视频.mp4"
echo
echo "如果测试成功，运行:"
echo "  python src/utils/generate_clip4mc_training_datas.py \\"
echo "      --metadata data/training/metadata.json \\"
echo "      --output-dir data/training/clip4mc \\"
echo "      --num-workers 4 \\"
echo "      --use-gpu"
echo
echo "========================================="

