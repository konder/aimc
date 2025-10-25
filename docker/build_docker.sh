#!/bin/bash
# AIMC MineDojo Docker 镜像构建脚本
# 自动处理 MC 编译步骤

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "======================================"
echo "AIMC MineDojo Docker 镜像构建脚本"
echo "======================================"
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

# 检查 requirements.txt 是否存在
if [ ! -f "${PROJECT_ROOT}/requirements.txt" ]; then
    echo "❌ 错误: ${PROJECT_ROOT}/requirements.txt 不存在"
    exit 1
fi

echo "📦 开始构建 Docker 镜像..."
echo "   平台: linux/amd64"
echo "   镜像名称: aimc-minedojo:latest"
echo ""

# 构建镜像
if docker build --platform linux/amd64 -t aimc-minedojo:latest -f Dockerfile "${PROJECT_ROOT}"; then
    echo ""
    echo "======================================"
    echo "✅ 镜像构建成功！"
    echo "======================================"
    echo ""
    echo "镜像信息:"
    docker images aimc-minedojo:latest
    echo ""
    echo "使用方法:"
    echo "1. 运行容器（挂载项目目录）:"
    echo "   docker run -it --platform linux/amd64 -v ${PROJECT_ROOT}:/workspace aimc-minedojo:latest"
    echo ""
    echo "2. 或使用 docker-compose（如果已配置）:"
    echo "   cd ${DOCKER_DIR} && docker-compose up"
    echo ""
else
    echo ""
    echo "======================================"
    echo "❌ 镜像构建失败"
    echo "======================================"
    echo ""
    echo "常见问题排查:"
    echo "1. 网络问题: 检查是否可以访问 GitHub 和阿里云镜像源"
    echo "2. 权限问题: 确保 Docker daemon 运行正常"
    echo "3. 磁盘空间: 确保有足够的磁盘空间（建议 >10GB）"
    echo "4. 编译问题: 查看上方日志中 gradlew shadowJar 的输出"
    echo ""
    echo "查看详细日志:"
    echo "   docker build --platform linux/amd64 --progress=plain -t aimc-minedojo:latest -f Dockerfile ${PROJECT_ROOT}"
    exit 1
fi

