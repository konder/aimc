#!/bin/bash
# 一键修复环境并运行训练
# 修复 gym 版本兼容性问题并启动训练

set -e

echo "========================================"
echo "修复环境并启动训练"
echo "========================================"
echo ""

# 检查当前环境
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ 请先激活 conda 环境"
    echo "   运行: scripts/run_minedojo_x86.sh"
    exit 1
fi

echo "✓ 当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 修复 gym 版本兼容性
echo "[1/2] 修复 gym 版本兼容性..."
echo ""

# 卸载可能冲突的版本
echo "  卸载旧版本..."
pip uninstall -y gym shimmy 2>/dev/null || true

# 安装兼容版本
echo "  安装兼容版本..."
pip install -q "gym==0.21.0"
pip install -q "shimmy==0.2.1"  # 指定明确的兼容版本

echo "  ✓ gym 版本修复完成"
echo ""

# 验证
python -c "import gym; print(f'  ✓ gym: {gym.__version__}')" || {
    echo "  ❌ gym 安装失败"
    exit 1
}
python -c "import shimmy; print(f'  ✓ shimmy: {shimmy.__version__}')" || {
    echo "  ❌ shimmy 安装失败"
    exit 1
}

echo ""
echo "[2/2] 启动训练..."
echo ""

# 运行训练
scripts/train_get_wood.sh "$@"

