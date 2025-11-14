#!/bin/bash
# MineDojo x86 环境启动脚本
# 自动处理JAVA_HOME设置、x86架构切换和conda环境激活

set -e

# 首先保存脚本的绝对路径（在arch切换之前）
if [ -z "$SCRIPT_REAL_PATH" ]; then
    export SCRIPT_REAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
fi

# 检查是否在x86模式下运行
CURRENT_ARCH=$(uname -m)
if [ "$CURRENT_ARCH" = "arm64" ]; then
    echo "切换到x86_64架构..."
    echo ""
    # 重新在x86模式下执行此脚本（使用保存的绝对路径）
    exec arch -x86_64 /bin/zsh "$SCRIPT_REAL_PATH" "$@"
fi

# 现在我们在x86模式下，设置JAVA_HOME
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# 设置 locale 环境变量，解决 Python subprocess 的编码问题
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export HF_HUB_OFFLINE=1
unset DISPLAY

echo "=========================================="
echo "MineDojo x86 环境"
echo "=========================================="
echo "✓ Architecture: $(uname -m)"
echo "✓ JAVA_HOME: ${JAVA_HOME}"
echo "✓ Java Version: $(java -version 2>&1 | head -n 1)"
echo "✓ Conda Env: minedojo-x86"
echo "=========================================="
echo ""

# 初始化conda
CONDA_BASE="/usr/local/Caskroom/miniforge/base"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# 激活minedojo-x86环境
conda activate minedojo-x86 2>/dev/null || {
    echo "警告: 无法激活minedojo-x86环境"
}

# 确保conda激活后JAVA_HOME和PATH依然正确
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# 再次确保 locale 设置正确
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

export MINE_RL_MC_PATH=/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/
export MINE_DOJO_MC_PATH=/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/

# 如果有参数，执行传入的命令
if [ $# -eq 0 ]; then
    # 没有参数，启动交互式shell
    echo "启动交互式x86 shell..."
    echo "提示: 现在可以直接运行MineDojo脚本了！"
    echo "退出请使用 'exit' 命令"
    echo ""
    cd "$PROJECT_ROOT"
    exec /bin/bash
else
    # 有参数，执行命令（保持当前目录）
    echo "执行: $@"
    echo ""
    # 不改变当前目录，让命令在调用者的目录执行
    exec "$@"
fi

