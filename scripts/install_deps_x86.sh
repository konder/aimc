#!/bin/bash
# 在 minedojo-x86 环境中安装训练所需的依赖
# 使用方法：在 x86 shell 中运行此脚本

set -e

echo "========================================"
echo "安装训练依赖包 (minedojo-x86)"
echo "========================================"
echo ""

# 检查是否在正确的环境中
if [[ "$CONDA_DEFAULT_ENV" != "minedojo-x86" ]]; then
    echo "⚠️  请先激活 minedojo-x86 环境："
    echo "   scripts/run_minedojo_x86.sh"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo "Python版本: $(python --version)"
echo ""

# 安装 PyTorch（CPU版本，适配x86）
echo "[1/4] 安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 Stable-Baselines3
echo ""
echo "[2/4] 安装 Stable-Baselines3..."
pip install stable-baselines3[extra]

# 安装 TensorBoard
echo ""
echo "[3/4] 安装 TensorBoard..."
pip install tensorboard

# 安装其他依赖
echo ""
echo "[4/5] 安装其他依赖..."
pip install tqdm pyyaml opencv-python

# 修复 gym 版本兼容性问题
echo ""
echo "[5/5] 修复 gym 版本兼容性..."
# MineDojo 使用 gym 0.21，但需要兼容的 shimmy 版本
pip install "gym==0.21.0" "shimmy<1.0.0"

echo ""
echo "========================================"
echo "✓ 依赖安装完成！"
echo "========================================"
echo ""
echo "验证安装："
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import stable_baselines3; print(f'✓ Stable-Baselines3: {stable_baselines3.__version__}')"
python -c "import tensorboard; print('✓ TensorBoard')"
python -c "import minedojo; print('✓ MineDojo')"

echo ""
echo "========================================"
echo "💡 运行训练示例："
echo "========================================"
echo ""
echo "快速测试（10K步，不使用MineCLIP）："
echo "  scripts/train_get_wood.sh test"
echo ""
echo "快速测试（10K步，使用MineCLIP，推荐）："
echo "  scripts/train_get_wood.sh test --mineclip"
echo ""
echo "标准训练（200K步，使用MineCLIP）："
echo "  scripts/train_get_wood.sh standard --mineclip"
echo ""
echo "⚠️  注意：一定要加 --mineclip 参数才能启用MineCLIP加速！"
echo ""

