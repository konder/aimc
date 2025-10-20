#!/bin/bash
# 修复 gym 版本兼容性问题
# 错误：AttributeError: module 'gym.spaces' has no attribute 'Sequence'

set -e

echo "========================================"
echo "修复 gym 版本兼容性问题"
echo "========================================"
echo ""

# 检查是否在正确的环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  请先激活 conda 环境"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 卸载并重新安装兼容版本
echo "[1/3] 卸载当前的 gym 和 shimmy..."
pip uninstall -y gym shimmy || true

echo ""
echo "[2/3] 安装兼容版本的 gym..."
pip install "gym==0.21.0"

echo ""
echo "[3/3] 安装兼容版本的 shimmy..."
pip install "shimmy<1.0.0"

echo ""
echo "========================================"
echo "✓ 修复完成！"
echo "========================================"
echo ""

# 验证
python -c "import gym; print(f'✓ gym: {gym.__version__}')" || echo "✗ gym 安装失败"
python -c "import shimmy; print(f'✓ shimmy: {shimmy.__version__}')" || echo "✗ shimmy 安装失败"

echo ""
echo "现在可以重新运行训练了！"
echo ""

