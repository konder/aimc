#!/bin/bash
# FFmpeg with CUDA 精简版编译脚本
# 只包含 CUDA 硬件加速，无多余依赖

set -e  # 遇到错误立即退出

echo "========================================="
echo "FFmpeg CUDA 精简版编译安装"
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
echo "  特点: 精简版，只包含 CUDA + 基础编解码器"
echo

# =========================================
# 步骤 1: 卸载现有 FFmpeg
# =========================================
echo "步骤 1: 卸载现有 FFmpeg..."

# 卸载 conda 的 FFmpeg
if command -v conda &> /dev/null; then
    echo "  卸载 conda FFmpeg..."
    conda remove ffmpeg -y 2>/dev/null || true
fi

# 卸载 apt 的 FFmpeg
echo "  卸载 apt FFmpeg..."
sudo apt-get remove -y ffmpeg libavcodec* libavformat* libavutil* 2>/dev/null || true
sudo apt-get autoremove -y 2>/dev/null || true

echo "  ✅ 卸载完成"
echo

# =========================================
# 步骤 2: 安装最小依赖
# =========================================
echo "步骤 2: 安装最小编译依赖..."

sudo apt-get update

# 只安装绝对必需的工具
echo "  安装基础工具..."
sudo apt-get install -y \
    build-essential \
    pkg-config \
    yasm \
    cmake \
    git \
    wget \
    libnuma-dev

echo "  ✅ 依赖安装完成"
echo

# =========================================
# 步骤 3: 检查 CUDA
# =========================================
echo "步骤 3: 检查 CUDA 环境..."

if [ ! -d "/usr/local/cuda" ]; then
    echo "  ❌ 错误: 未找到 /usr/local/cuda"
    echo
    echo "  请先安装 CUDA Toolkit:"
    echo "    conda install -c nvidia cuda-toolkit -y"
    echo "  或"
    echo "    从 NVIDIA 官网下载安装"
    exit 1
fi

echo "  ✅ CUDA 路径: /usr/local/cuda"

# 设置 CUDA 环境变量
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# 检查 nvcc
if command -v nvcc &> /dev/null; then
    echo "  ✅ nvcc 版本: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
else
    echo "  ⚠️  警告: nvcc 未找到，可能会影响编译"
fi

echo

# =========================================
# 步骤 4: 安装 NVIDIA Video Codec SDK
# =========================================
echo "步骤 4: 安装 NVIDIA Video Codec SDK..."

cd /tmp
if [ -d "nv-codec-headers" ]; then
    rm -rf nv-codec-headers
fi

echo "  下载 nv-codec-headers..."
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers

# 使用较新版本
git checkout n12.1.14.0 2>/dev/null || git checkout n11.1.5.2 2>/dev/null || true

echo "  安装..."
make
sudo make install

echo "  ✅ NVIDIA Video Codec SDK 安装完成"
echo

# =========================================
# 步骤 5: 下载 FFmpeg 源码
# =========================================
echo "步骤 5: 下载 FFmpeg 源码..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "  下载 FFmpeg $FFMPEG_VERSION..."
wget -q --show-progress "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2"

echo "  解压..."
tar -xf "ffmpeg-${FFMPEG_VERSION}.tar.bz2"
cd "ffmpeg-${FFMPEG_VERSION}"

echo "  ✅ 源码准备完成"
echo

# =========================================
# 步骤 6: 配置（精简版）
# =========================================
echo "步骤 6: 配置 FFmpeg（精简版）..."
echo "  特点: 只启用 CUDA + 基础编解码器，无字幕、滤镜等"
echo

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
    --enable-shared \
    --disable-static \
    --disable-doc \
    --disable-htmlpages \
    --disable-manpages \
    --disable-podpages \
    --disable-txtpages

if [ $? -ne 0 ]; then
    echo
    echo "  ❌ 配置失败！"
    echo
    echo "  常见问题:"
    echo "    1. CUDA 未正确安装"
    echo "    2. nvcc 不在 PATH 中"
    echo "    3. CUDA 版本太旧（建议 11.0+）"
    echo
    echo "  查看详细日志:"
    echo "    cat $BUILD_DIR/ffmpeg-${FFMPEG_VERSION}/ffbuild/config.log"
    echo
    exit 1
fi

echo "  ✅ 配置完成"
echo

# =========================================
# 步骤 7: 编译
# =========================================
echo "步骤 7: 编译 FFmpeg..."
echo "  预计时间: 10-30 分钟"
echo "  进度可能不会实时显示，请耐心等待..."
echo

START_TIME=$(date +%s)

make -j"$NPROC"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $? -ne 0 ]; then
    echo
    echo "  ❌ 编译失败！"
    echo
    echo "  如果是 CUDA 相关错误，尝试:"
    echo "    1. 检查 CUDA 版本: nvcc --version"
    echo "    2. 检查 GPU 驱动: nvidia-smi"
    echo "    3. 降低 CUDA 版本或升级 FFmpeg 版本"
    echo
    exit 1
fi

echo
echo "  ✅ 编译完成（耗时: ${ELAPSED}秒）"
echo

# =========================================
# 步骤 8: 安装
# =========================================
echo "步骤 8: 安装 FFmpeg..."

sudo make install
sudo ldconfig

echo "  ✅ 安装完成"
echo

# =========================================
# 步骤 9: 验证
# =========================================
echo "步骤 9: 验证安装..."
echo

echo "  FFmpeg 版本:"
ffmpeg -version | head -1
echo

echo "  硬件加速支持:"
ffmpeg -hwaccels 2>/dev/null | grep -E "cuda|cuvid" || echo "    ⚠️  未检测到 CUDA"
echo

echo "  CUDA 解码器:"
if ffmpeg -decoders 2>/dev/null | grep -q cuvid; then
    ffmpeg -decoders 2>/dev/null | grep cuvid | head -5
    echo "    ✅ CUDA 解码器可用"
else
    echo "    ⚠️  CUDA 解码器不可用"
fi
echo

echo "  CUDA 编码器:"
if ffmpeg -encoders 2>/dev/null | grep -q nvenc; then
    ffmpeg -encoders 2>/dev/null | grep nvenc | head -3
    echo "    ✅ CUDA 编码器可用"
else
    echo "    ⚠️  CUDA 编码器不可用"
fi
echo

# =========================================
# 步骤 10: 清理
# =========================================
echo "步骤 10: 清理构建文件..."

cd /
rm -rf "$BUILD_DIR"
rm -rf /tmp/nv-codec-headers

echo "  ✅ 清理完成"
echo

# =========================================
# 完成
# =========================================
echo "========================================="
echo "✅ FFmpeg CUDA 精简版安装完成！"
echo "========================================="
echo
echo "下一步: 测试 GPU 解码"
echo
echo "  cd /root/autodl-tmp/aimc"
echo
echo "  # 测试单个视频"
echo "  bash scripts/test_gpu_decode.sh \\"
echo "      '/root/autodl-tmp/TEST2000/视频.mp4'"
echo
echo "  # 如果测试成功，运行完整处理"
echo "  python src/utils/generate_clip4mc_training_datas.py \\"
echo "      --metadata data/training/metadata.json \\"
echo "      --output-dir data/training/clip4mc \\"
echo "      --num-workers 4 \\"
echo "      --use-gpu"
echo
echo "========================================="
echo

# 保存安装信息
cat > /tmp/ffmpeg_cuda_install_info.txt << EOFINFO
FFmpeg CUDA 安装信息
==================

安装时间: $(date)
FFmpeg 版本: $FFMPEG_VERSION
安装路径: $INSTALL_PREFIX
编译时间: ${ELAPSED}秒

CUDA 信息:
$(nvcc --version 2>/dev/null || echo "nvcc 不可用")

GPU 信息:
$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用")

测试命令:
  bash scripts/test_gpu_decode.sh '/path/to/video.mp4'

EOFINFO

echo "安装信息已保存到: /tmp/ffmpeg_cuda_install_info.txt"
echo

