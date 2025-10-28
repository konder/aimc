#!/bin/bash
# 安装打过补丁的 MineRL（支持多版本）
# 用法: 
#   ./scripts/install_minerl_patched.sh 0.4.4
#   ./scripts/install_minerl_patched.sh 1.0.0

set -e  # 遇到错误立即退出

# 获取脚本所在目录的父目录（项目根目录）- 必须在 cd 之前获取
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# 获取版本号（默认 1.0.0）
MINERL_VERSION="${1:-0.4.4}"
WORK_DIR="/tmp/minerl_install_$$"

# 清理函数
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ 安装失败！"
        echo "临时文件保留在: ${WORK_DIR}"
        echo "可以手动检查和清理"
    else
        # 成功时清理临时文件
        if [ -d "${WORK_DIR}" ]; then
            echo "[5/5] 清理临时文件..."
            rm -rf "${WORK_DIR}"
        fi
    fi
}

# 设置退出时执行清理
trap cleanup EXIT

echo "======================================"
echo "安装 MineRL ${MINERL_VERSION} (已打补丁)"
echo "======================================"
echo "项目根目录: ${PROJECT_ROOT}"
echo ""

# 1. 创建工作目录
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# 2. 下载 MineRL 源码
echo "[1/5] 下载 MineRL ${MINERL_VERSION} 源码..."
pip download --no-deps --no-binary :all: minerl==${MINERL_VERSION}

# 3. 解压
echo "[2/5] 解压源码..."
tar -xzf minerl-${MINERL_VERSION}.tar.gz
cd minerl-${MINERL_VERSION}

# 4. 应用补丁
echo "[3/5] 应用 Minecraft 构建补丁..."

# 检查是否存在 Malmo/Minecraft
MINECRAFT_DIR=""
if [ -d "minerl/Malmo/Minecraft" ]; then
    MINECRAFT_DIR="minerl/Malmo/Minecraft"
    echo "  ✓ 找到 Minecraft 目录: ${MINECRAFT_DIR}"
elif [ -d "minerl/env/Malmo/Minecraft" ]; then
    MINECRAFT_DIR="minerl/env/Malmo/Minecraft"
    echo "  ✓ 找到 Minecraft 目录: ${MINECRAFT_DIR}"
else
    echo "  ⚠️  未找到 Minecraft 目录，跳过补丁"
    MINECRAFT_DIR=""
fi

if [ -n "${MINECRAFT_DIR}" ]; then
    # 4.1 修改 gradle-wrapper.properties
    GRADLE_WRAPPER="${MINECRAFT_DIR}/gradle/wrapper/gradle-wrapper.properties"
    if [ -f "${GRADLE_WRAPPER}" ]; then
        echo "  - 修改 Gradle 下载地址为阿里云镜像..."
        sed -i.bak 's|https://services.gradle.org/distributions/|https://mirrors.aliyun.com/gradle/|g' "${GRADLE_WRAPPER}"
        sed -i.bak 's|https\\://services.gradle.org/distributions/|https\\://mirrors.aliyun.com/gradle/|g' "${GRADLE_WRAPPER}"
        echo "    ✓ Gradle wrapper 已修改"
    else
        echo "    ⚠️  未找到 gradle-wrapper.properties"
    fi

    # 4.2 修改 build.gradle
    BUILD_GRADLE="${MINECRAFT_DIR}/build.gradle"
    if [ -f "${BUILD_GRADLE}" ]; then
        echo "  - 修改 build.gradle 仓库配置..."
        
        # 检查是否已经有国内镜像
        if grep -q "maven.aliyun.com" "${BUILD_GRADLE}"; then
            echo "    ✓ 已包含国内镜像，跳过"
        else
            # 在 repositories { 后添加国内镜像
            awk '
            /^buildscript {/ { in_buildscript=1 }
            in_buildscript && /^    repositories {/ { 
                print
                print "        maven { url \"https://maven.aliyun.com/repository/public\" }"
                print "        maven { url \"https://maven.aliyun.com/repository/central\" }"
                print "        maven { url \"https://libraries.minecraft.net/\" }"
                print "        maven { url \"https://jitpack.io\" }"
                next
            }
            { print }
            ' "${BUILD_GRADLE}" > "${BUILD_GRADLE}.tmp" && mv "${BUILD_GRADLE}.tmp" "${BUILD_GRADLE}"
            echo "    ✓ build.gradle 已修改"
        fi
        
        # 修复 schemas.index 相对路径问题（如果存在）
        if grep -q "new File('src/main/resources/schemas.index')" "${BUILD_GRADLE}"; then
            sed -i.bak "s|new File('src/main/resources/schemas.index')|new File(projectDir, 'src/main/resources/schemas.index')|g" "${BUILD_GRADLE}"
            echo "    ✓ schemas.index 路径已修复"
        fi
    else
        echo "    ⚠️  未找到 build.gradle"
    fi
else
    echo "  ℹ️  MineRL ${MINERL_VERSION} 可能不需要 Minecraft 构建"
fi

# 5. 安装
echo "[4/5] 安装 MineRL ${MINERL_VERSION}..."
pip install .

echo ""
echo "✅ MineRL ${MINERL_VERSION} 安装成功！"
echo ""
echo "验证安装:"
echo "  python -c \"import minerl; print('MineRL 已安装')\""
echo ""
echo "注意事项："
if [ -n "${MINECRAFT_DIR}" ]; then
    echo "1. 首次运行时 Minecraft 仍需要构建，但现在使用的是国内镜像"
    echo "2. 如果构建失败，请检查网络连接"
    echo "3. 构建过程可能需要 10-30 分钟，请耐心等待"
else
    echo "1. MineRL ${MINERL_VERSION} 可能使用预编译的 Minecraft"
    echo "2. 首次运行可能需要下载额外文件"
fi
echo ""

