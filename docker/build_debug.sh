#!/bin/bash
# AIMC 调试镜像构建脚本
# 只构建到 Python 依赖安装，用于手动调试 MC 编译

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "======================================"
echo "🐛 AIMC 调试镜像构建脚本"
echo "======================================"
echo ""
echo "此镜像只包含基础环境和 Python 依赖"
echo "不包含 MineCLIP 和 MC 编译步骤"
echo "用于进入容器手动调试"
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装，请先安装 Docker"
    exit 1
fi

echo "✅ Docker 版本: $(docker --version)"
echo ""

# 切换到 docker 目录
cd "${DOCKER_DIR}"

echo "📦 开始构建调试镜像..."
echo "   平台: linux/amd64"
echo "   镜像名称: aimc-minedojo:debug"
echo "   Dockerfile: Dockerfile.debug"
echo ""

# 构建镜像
if docker build --platform linux/amd64 -t aimc-minedojo:debug -f Dockerfile.debug "${PROJECT_ROOT}"; then
    echo ""
    echo "======================================"
    echo "✅ 调试镜像构建成功！"
    echo "======================================"
    echo ""
    echo "镜像信息:"
    docker images aimc-minedojo:debug
    echo ""
    echo "======================================"
    echo "使用方法"
    echo "======================================"
    echo ""
    echo "1️⃣  启动容器（交互式）:"
    echo "   docker run -it --platform linux/amd64 \\"
    echo "     --name aimc-debug \\"
    echo "     -v ${PROJECT_ROOT}:/workspace \\"
    echo "     aimc-minedojo:debug"
    echo ""
    echo "2️⃣  在容器内手动执行调试步骤:"
    echo ""
    echo "   # 设置环境变量（方便后续使用）"
    echo "   export MC_PATH=\"/opt/conda/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft\""
    echo ""
    echo "   # 步骤 1: 克隆 MixinGradle"
    echo "   mkdir -p /opt/hotfix"
    echo "   cd /opt/hotfix"
    echo "   git clone https://github.com/verityw/MixinGradle-dcfaf61.git"
    echo "   ls -la MixinGradle-dcfaf61/"
    echo ""
    echo "   # 步骤 2: 修改 gradle-wrapper.properties"
    echo "   sed -i 's|https\\\\://services.gradle.org/distributions/|https://mirrors.aliyun.com/gradle/|g' \\"
    echo "     \${MC_PATH}/gradle/wrapper/gradle-wrapper.properties"
    echo ""
    echo "   # 步骤 3: 备份并修改 build.gradle"
    echo "   cp \${MC_PATH}/build.gradle \${MC_PATH}/build.gradle.bak"
    echo "   python /root/patch_buildgradle.py \${MC_PATH}/build.gradle"
    echo ""
    echo "   # 步骤 4: 查看修改结果"
    echo "   grep -A 5 \"repositories {\" \${MC_PATH}/build.gradle | head -20"
    echo ""
    echo "   # 步骤 5: 执行编译"
    echo "   cd \${MC_PATH}"
    echo "   ./gradlew shadowJar --no-daemon --stacktrace"
    echo ""
    echo "   # 步骤 6: 验证结果"
    echo "   bash /root/verify_mc_build.sh"
    echo ""
    echo "3️⃣  退出容器:"
    echo "   exit"
    echo ""
    echo "4️⃣  重新进入容器:"
    echo "   docker start aimc-debug"
    echo "   docker exec -it aimc-debug /bin/bash"
    echo ""
    echo "5️⃣  删除调试容器:"
    echo "   docker rm -f aimc-debug"
    echo ""
else
    echo ""
    echo "======================================"
    echo "❌ 调试镜像构建失败"
    echo "======================================"
    echo ""
    echo "请检查错误信息并重试"
    exit 1
fi

