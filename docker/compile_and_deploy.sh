#!/bin/bash
# BiomeGenerator补丁编译和部署脚本
# 用法: ./compile_and_deploy.sh [MinerL安装路径]

set -e

MINERL_PATH="${1:-}"

echo "=========================================="
echo "MinerL BiomeGenerator编译和部署工具"
echo "=========================================="
echo ""

# 自动检测MinerL路径
if [ -z "$MINERL_PATH" ]; then
    echo "未指定路径，尝试自动检测..."
    MINERL_PATH=$(python -c "import minerl; import os; print(os.path.dirname(minerl.__file__))" 2>/dev/null || echo "")
    
    if [ -n "$MINERL_PATH" ]; then
        echo "✓ 检测到MinerL路径: $MINERL_PATH"
    fi
fi

if [ -z "$MINERL_PATH" ]; then
    echo "❌ 错误: 无法找到MinerL安装路径"
    echo "请手动指定: ./compile_and_deploy.sh /path/to/site-packages/minerl"
    exit 1
fi

MCP_PATH="$MINERL_PATH/MCP-Reborn"

if [ ! -d "$MCP_PATH" ]; then
    echo "❌ 错误: MCP-Reborn目录不存在: $MCP_PATH"
    exit 1
fi

JAR_FILE="$MCP_PATH/build/libs/mcprec-6.13.jar"

if [ ! -f "$JAR_FILE" ]; then
    echo "❌ 错误: jar文件不存在: $JAR_FILE"
    exit 1
fi

echo ""
echo "目标信息:"
echo "  MCP路径: $MCP_PATH"
echo "  JAR文件: $JAR_FILE"
echo ""

cd "$MCP_PATH"

# 检查补丁是否已应用
if ! grep -q "BiomeGenerator detected" src/main/java/com/minerl/multiagent/env/EnvServer.java; then
    echo "⚠️  警告: 补丁似乎未应用"
    echo "请先运行: ./apply_patch.sh"
    exit 1
fi

echo "✓ 检测到补丁已应用"
echo ""

# 备份jar文件
BACKUP_JAR="${JAR_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 备份jar文件..."
cp "$JAR_FILE" "$BACKUP_JAR"
echo "✓ 备份保存至: $BACKUP_JAR"

# 创建编译目录
echo ""
echo "📁 准备编译环境..."
mkdir -p build/custom_compile
echo "✓ 编译目录已创建"

# 构建classpath
echo ""
echo "🔍 构建classpath..."
GRADLE_JARS=$(find ~/.gradle/caches -name '*.jar' 2>/dev/null | tr '\n' ':' || echo "")
CLASSPATH="$JAR_FILE:$GRADLE_JARS"
echo "✓ Classpath已构建"

# 编译Java文件
echo ""
echo "🔨 编译EnvServer.java..."
if javac -cp "$CLASSPATH" \
         -d build/custom_compile \
         src/main/java/com/minerl/multiagent/env/EnvServer.java 2>&1; then
    echo "✓ 编译成功"
else
    echo "❌ 编译失败"
    exit 1
fi

# 检查编译产物
echo ""
echo "✅ 检查编译产物..."
if [ -f "build/custom_compile/com/minerl/multiagent/env/EnvServer.class" ]; then
    echo "✓ EnvServer.class 已生成"
else
    echo "❌ EnvServer.class 未找到"
    exit 1
fi

if [ -f "build/custom_compile/com/minerl/multiagent/env/EnvServer\$1.class" ]; then
    echo "✓ EnvServer\$1.class 已生成"
else
    echo "⚠️  警告: EnvServer\$1.class 未找到（可能不需要）"
fi

# 更新jar文件
echo ""
echo "📦 更新jar文件..."
if [ -f "build/custom_compile/com/minerl/multiagent/env/EnvServer\$1.class" ]; then
    # 如果有内部类
    jar uf "$JAR_FILE" \
        -C build/custom_compile com/minerl/multiagent/env/EnvServer.class \
        -C build/custom_compile com/minerl/multiagent/env/EnvServer\$1.class
else
    # 只更新主类
    jar uf "$JAR_FILE" \
        -C build/custom_compile com/minerl/multiagent/env/EnvServer.class
fi
echo "✓ jar文件已更新"

# 验证jar更新
echo ""
echo "✅ 验证jar更新..."
if jar tf "$JAR_FILE" | grep -q "com/minerl/multiagent/env/EnvServer.class"; then
    TIMESTAMP=$(unzip -l "$JAR_FILE" | grep "EnvServer.class" | awk '{print $1, $2}')
    echo "✓ jar文件包含更新的EnvServer.class"
    echo "  时间戳: $TIMESTAMP"
else
    echo "❌ jar文件未正确更新"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 编译和部署完成！"
echo "=========================================="
echo ""
echo "后续步骤:"
echo ""
echo "1. 运行测试验证:"
echo "   ./scripts/test_biome.sh --biome desert --save-images --steps 30"
echo ""
echo "2. 检查日志确认功能:"
echo "   grep 'BiomeGenerator detected' logs/mc_*.log | tail -1"
echo ""
echo "3. 查看生成的图片:"
echo "   open logs/biome_verification/*/comparison_grid.png"
echo ""
echo "备份文件:"
echo "  jar: $BACKUP_JAR"
echo ""
echo "如需回滚:"
echo "   cp $BACKUP_JAR $JAR_FILE"
echo ""

