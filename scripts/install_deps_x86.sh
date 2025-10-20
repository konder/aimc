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
echo "[4/4] 安装其他依赖..."
pip install tqdm pyyaml opencv-python

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
echo "现在可以运行训练了："
echo "  scripts/train_get_wood.sh test --mineclip"
echo ""

