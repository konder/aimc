#!/bin/bash
# DAgger Web 控制台启动脚本

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo "======================================================================"
echo "DAgger Web 控制台"
echo "======================================================================"
echo ""

# 检查conda环境
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "minedojo"* ]]; then
    echo "⚠️  警告: 未检测到 minedojo 环境"
    echo "请先激活环境: conda activate minedojo-x86"
    echo ""
    read -p "是否继续? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查Web依赖
echo "检查 Web 依赖..."
if ! python -c "import flask" 2>/dev/null; then
    echo ""
    echo "❌ 缺少 Flask 依赖！"
    echo ""
    echo "请先安装依赖："
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi
echo "✅ 依赖检查通过"

echo ""
echo "======================================================================"
echo "启动服务器..."
echo "======================================================================"
echo ""
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo ""
echo "======================================================================"
echo ""

# 启动服务器
python -m src.web.app