#!/bin/bash
# 应用 MineDojo Inventory Action 补丁
# 使用方法: bash apply_minedojo_patch.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║          应用 MineDojo Inventory Action 补丁                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检测 MineDojo 安装路径
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ 错误: 未检测到 Conda 环境"
    echo "   请先激活 minedojo 环境: conda activate minedojo-x86"
    exit 1
fi

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MINEDOJO_PATH="$CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/minedojo"

if [ ! -d "$MINEDOJO_PATH" ]; then
    echo "❌ 错误: 未找到 MineDojo 安装目录"
    echo "   路径: $MINEDOJO_PATH"
    exit 1
fi

echo "✓ MineDojo 安装路径: $MINEDOJO_PATH"
echo "✓ Python 版本: $PYTHON_VERSION"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/minedojo_inventory.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "❌ 错误: 未找到 patch 文件"
    echo "   路径: $PATCH_FILE"
    exit 1
fi

echo "应用补丁..."
cd "$MINEDOJO_PATH"

# 检查是否已应用
if grep -q "\"inventory\"" sim/sim.py 2>/dev/null && \
   grep -q "elif fn_action == 8:" sim/wrappers/ar_nn/nn_action_space_wrapper.py 2>/dev/null; then
    echo "✓ 补丁已应用，跳过"
else
    # 备份文件
    BACKUP_SUFFIX=".backup_$(date +%Y%m%d_%H%M%S)"
    echo "  备份文件..."
    cp sim/sim.py "sim/sim.py${BACKUP_SUFFIX}" 2>/dev/null || true
    cp sim/handlers/agent/action.py "sim/handlers/agent/action.py${BACKUP_SUFFIX}" 2>/dev/null || true
    cp sim/wrappers/ar_nn/nn_action_space_wrapper.py "sim/wrappers/ar_nn/nn_action_space_wrapper.py${BACKUP_SUFFIX}" 2>/dev/null || true
    
    # 应用补丁
    patch -p0 < "$PATCH_FILE"
    
    echo "✓ 补丁已应用"
fi

# 清理缓存
echo ""
echo "清理 Python 缓存..."
find "$MINEDOJO_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$MINEDOJO_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ Python 缓存已清理"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                        ✅ 补丁应用完成！                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "修改内容:"
echo "  1. ✓ sim/sim.py: 添加 inventory 到 common_actions"
echo "  2. ✓ handlers/agent/action.py: 移除 0 值过滤"
echo "  3. ✓ wrappers/ar_nn/nn_action_space_wrapper.py:"
echo "       • Functional actions: 8 → 9"
echo "       • 添加 inventory 处理逻辑"
echo ""
echo "Action Space 更新:"
echo "  MultiDiscrete([3, 3, 4, 36001, 36001, 9, 244, 36])"
echo "                                        ↑"
echo "  Index 5 (Functional): 0-7 + 8 (inventory)"
echo ""

