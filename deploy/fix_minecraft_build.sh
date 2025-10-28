#!/bin/bash
#
# Minecraft 编译修复脚本
# 
# 用途: 修复 MineDojo 和 MineRL 在国内环境下的 Minecraft 编译问题
# 
# 使用方法:
#   # MineDojo 修复
#   ./scripts/fix_minecraft_build.sh minedojo
#   
#   # MineRL 修复
#   ./scripts/fix_minecraft_build.sh minerl
#
# 适用场景:
#   - pip install minedojo 之后，首次启动前
#   - pip install minerl 之后，首次启动前
#   - Minecraft 编译失败后
#

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
Minecraft 编译修复脚本

用法: $0 [minedojo|minerl]

参数:
  minedojo    修复 MineDojo 的 Minecraft 编译配置
  minerl      修复 MineRL 的 Minecraft 编译配置

示例:
  # 修复 MineDojo
  $0 minedojo

  # 修复 MineRL
  $0 minerl

修复内容:
  1. Gradle 下载源（使用阿里云镜像）
  2. MixinGradle 依赖（本地仓库）
  3. ForgeGradle 依赖（MineDojo）
  4. schemas.index 路径
  5. build.gradle 镜像配置

注意:
  - 需要在激活的 conda 环境中运行
  - 首次运行会克隆 MixinGradle 到 /opt/hotfix
  - 建议在 pip install 之后、首次启动之前运行

EOF
}

# 检查参数
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

TARGET="$1"

if [ "$TARGET" != "minedojo" ] && [ "$TARGET" != "minerl" ]; then
    print_error "无效的参数: $TARGET"
    echo "使用 $0 --help 查看帮助"
    exit 1
fi

print_header "Minecraft 编译修复工具 - $TARGET"

# ============================================================================
# 步骤 1: 检查环境
# ============================================================================
print_header "1. 检查环境"

# 检查 Python
if ! command -v python &> /dev/null; then
    print_error "Python 未找到"
    exit 1
fi
print_success "Python: $(python --version 2>&1 | head -1)"

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "未检测到 conda 环境，确保你在正确的环境中"
else
    print_success "Conda 环境: $CONDA_DEFAULT_ENV"
fi

# 获取 site-packages 路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
print_info "site-packages: $SITE_PACKAGES"

# ============================================================================
# 步骤 2: 检测目标包路径
# ============================================================================
print_header "2. 检测 $TARGET 安装路径"

if [ "$TARGET" = "minedojo" ]; then
    PACKAGE_NAME="minedojo"
    MC_DIR="$SITE_PACKAGES/minedojo/sim/Malmo/Minecraft"
elif [ "$TARGET" = "minerl" ]; then
    PACKAGE_NAME="minerl"
    MC_DIR="$SITE_PACKAGES/minerl/MCP-Reborn"
fi

# 检查包是否安装
if ! python -c "import $PACKAGE_NAME" 2>/dev/null; then
    print_error "$PACKAGE_NAME 未安装"
    echo "请先运行: pip install $PACKAGE_NAME"
    exit 1
fi
print_success "$PACKAGE_NAME 已安装"

# 检查 Minecraft 目录
if [ ! -d "$MC_DIR" ]; then
    print_error "Minecraft 目录不存在: $MC_DIR"
    exit 1
fi
print_success "Minecraft 目录: $MC_DIR"

# ============================================================================
# 步骤 3: 准备 MixinGradle hotfix
# ============================================================================
print_header "3. 准备 MixinGradle hotfix"

HOTFIX_DIR="/opt/hotfix"
MIXIN_REPO="$HOTFIX_DIR/MixinGradle-dcfaf61"

if [ ! -d "$MIXIN_REPO" ]; then
    print_info "克隆 MixinGradle 到 $HOTFIX_DIR"
    
    # 创建 hotfix 目录
    if [ ! -d "$HOTFIX_DIR" ]; then
        sudo mkdir -p "$HOTFIX_DIR"
        sudo chown $USER "$HOTFIX_DIR"
    fi
    
    # 克隆仓库
    cd "$HOTFIX_DIR"
    if ! git clone https://github.com/verityw/MixinGradle-dcfaf61.git; then
        print_error "克隆 MixinGradle 失败"
        exit 1
    fi
    
    print_success "MixinGradle 克隆成功"
else
    print_success "MixinGradle 已存在: $MIXIN_REPO"
fi

# ============================================================================
# 步骤 4: 修复 gradle-wrapper.properties
# ============================================================================
print_header "4. 修复 Gradle 下载源"

GRADLE_WRAPPER="$MC_DIR/gradle/wrapper/gradle-wrapper.properties"

if [ ! -f "$GRADLE_WRAPPER" ]; then
    print_error "gradle-wrapper.properties 不存在: $GRADLE_WRAPPER"
    exit 1
fi

# 备份
if [ ! -f "$GRADLE_WRAPPER.bak" ]; then
    cp "$GRADLE_WRAPPER" "$GRADLE_WRAPPER.bak"
    print_info "已备份: $GRADLE_WRAPPER.bak"
fi

# 替换 Gradle 下载地址
if grep -q "mirrors.aliyun.com" "$GRADLE_WRAPPER"; then
    print_success "Gradle 镜像已配置"
else
    print_info "配置 Gradle 阿里云镜像..."
    sed -i.tmp 's|https://services.gradle.org/distributions/|https://mirrors.aliyun.com/gradle/|g' "$GRADLE_WRAPPER"
    rm -f "$GRADLE_WRAPPER.tmp"
    print_success "Gradle 镜像配置完成"
fi

# ============================================================================
# 步骤 5: 修复 build.gradle
# ============================================================================
print_header "5. 修复 build.gradle"

BUILD_GRADLE="$MC_DIR/build.gradle"

if [ ! -f "$BUILD_GRADLE" ]; then
    print_error "build.gradle 不存在: $BUILD_GRADLE"
    exit 1
fi

# 备份
if [ ! -f "$BUILD_GRADLE.bak" ]; then
    cp "$BUILD_GRADLE" "$BUILD_GRADLE.bak"
    print_info "已备份: $BUILD_GRADLE.bak"
fi

# 5.1 添加 maven 本地仓库
print_info "添加 maven 本地仓库..."
if grep -q "file:///opt/hotfix" "$BUILD_GRADLE"; then
    print_success "maven 本地仓库已配置"
else
    # 在 buildscript -> repositories 中添加
    sed -i.tmp '/buildscript {/,/repositories {/{ 
        /repositories {/a\
        maven { url "file:///opt/hotfix" }
    }' "$BUILD_GRADLE"
    rm -f "$BUILD_GRADLE.tmp"
    print_success "maven 本地仓库配置完成"
fi

# 5.2 替换 MixinGradle 依赖
print_info "替换 MixinGradle 依赖..."
if grep -q "MixinGradle-dcfaf61:MixinGradle:dcfaf61" "$BUILD_GRADLE"; then
    print_success "MixinGradle 依赖已替换"
else
    sed -i.tmp "s|classpath('com.github.SpongePowered:MixinGradle:dcfaf61')|classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61')|g" "$BUILD_GRADLE"
    rm -f "$BUILD_GRADLE.tmp"
    print_success "MixinGradle 依赖替换完成"
fi

# 5.3 修复 schemas.index 路径
print_info "修复 schemas.index 路径..."
if grep -q "new File(projectDir, 'src/main/resources/schemas.index')" "$BUILD_GRADLE"; then
    print_success "schemas.index 路径已修复"
else
    sed -i.tmp "s|new File('src/main/resources/schemas.index')|new File(projectDir, 'src/main/resources/schemas.index')|g" "$BUILD_GRADLE"
    rm -f "$BUILD_GRADLE.tmp"
    print_success "schemas.index 路径修复完成"
fi

# 5.4 ForgeGradle 修复（仅 MineDojo）
if [ "$TARGET" = "minedojo" ]; then
    print_info "替换 ForgeGradle 依赖（MineDojo）..."
    
    # 替换 buildscript 中的 ForgeGradle
    if grep -q "com.github.MineDojo:ForgeGradle:FG_2.2_patched-SNAPSHOT" "$BUILD_GRADLE"; then
        print_success "ForgeGradle classpath 已替换"
    else
        sed -i.tmp "s|classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'|classpath 'com.github.MineDojo:ForgeGradle:FG_2.2_patched-SNAPSHOT'|g" "$BUILD_GRADLE"
        rm -f "$BUILD_GRADLE.tmp"
        print_success "ForgeGradle classpath 替换完成"
    fi
    
    # 替换 dependencies 中的 forgegradle
    if grep -q "com.github.MineDojo:Forgegradle:FG_2.2_patched-SNAPSHOT" "$BUILD_GRADLE"; then
        print_success "Forgegradle implementation 已替换"
    else
        sed -i.tmp "s|implementation 'com.github.brandonhoughton:forgegradle:FG_2.2_patched-SNAPSHOT'|implementation 'com.github.MineDojo:Forgegradle:FG_2.2_patched-SNAPSHOT'|g" "$BUILD_GRADLE"
        rm -f "$BUILD_GRADLE.tmp"
        print_success "Forgegradle implementation 替换完成"
    fi
fi

# 5.5 添加国内镜像仓库
print_info "添加国内镜像仓库..."
if grep -q "maven.aliyun.com" "$BUILD_GRADLE"; then
    print_success "国内镜像仓库已配置"
else
    # 在 buildscript -> repositories 开头添加阿里云镜像
    sed -i.tmp '/buildscript {/,/repositories {/{ 
        /repositories {/a\
        maven { url "https://maven.aliyun.com/repository/public" }\
        maven { url "https://maven.aliyun.com/repository/central" }
    }' "$BUILD_GRADLE"
    rm -f "$BUILD_GRADLE.tmp"
    print_success "国内镜像仓库配置完成"
fi

# ============================================================================
# 步骤 6: 验证修复
# ============================================================================
print_header "6. 验证修复"

echo ""
echo "检查关键配置:"
echo ""

# 检查 gradle-wrapper
echo -n "  Gradle 镜像: "
if grep -q "mirrors.aliyun.com" "$GRADLE_WRAPPER"; then
    print_success "已配置"
else
    print_error "未配置"
fi

# 检查 build.gradle
echo -n "  maven 本地仓库: "
if grep -q "file:///opt/hotfix" "$BUILD_GRADLE"; then
    print_success "已配置"
else
    print_error "未配置"
fi

echo -n "  MixinGradle: "
if grep -q "MixinGradle-dcfaf61:MixinGradle:dcfaf61" "$BUILD_GRADLE"; then
    print_success "已替换"
else
    print_error "未替换"
fi

echo -n "  schemas.index: "
if grep -q "new File(projectDir," "$BUILD_GRADLE"; then
    print_success "已修复"
else
    print_error "未修复"
fi

if [ "$TARGET" = "minedojo" ]; then
    echo -n "  ForgeGradle: "
    if grep -q "com.github.MineDojo:ForgeGradle" "$BUILD_GRADLE"; then
        print_success "已替换"
    else
        print_error "未替换"
    fi
fi

# ============================================================================
# 完成
# ============================================================================
print_header "修复完成"

echo ""
print_success "所有修复已应用到 $TARGET"
echo ""
print_info "备份文件:"
echo "  - $GRADLE_WRAPPER.bak"
echo "  - $BUILD_GRADLE.bak"
echo ""
print_info "下一步:"
echo "  1. 测试 Minecraft 编译:"
echo "     python -c 'import $PACKAGE_NAME; env = $PACKAGE_NAME.make(\"harvest_1_log\" if \"$TARGET\" == \"minedojo\" else \"MineRLBasaltFindCave-v0\"); env.reset(); env.close()'"
echo ""
echo "  2. 如果遇到问题，查看日志:"
echo "     tail -f logs/mc_*.log"
echo ""
echo "  3. 恢复备份（如需要）:"
echo "     mv $GRADLE_WRAPPER.bak $GRADLE_WRAPPER"
echo "     mv $BUILD_GRADLE.bak $BUILD_GRADLE"
echo ""

print_success "✅ 修复脚本执行完毕"

