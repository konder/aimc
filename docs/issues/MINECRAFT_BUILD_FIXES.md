# Minecraft 编译问题修复指南

## 📋 问题概述

MineDojo 和 MineRL 在国内环境安装时，会遇到 Minecraft JAR 包编译失败的问题。这是因为：

1. **MineDojo**: 直接编译 Minecraft 包
2. **MineRL**: 通过 MCP-Reborn 编译 Minecraft 包

两者都使用 Gradle 构建系统，在国内环境下会遇到以下问题。

**最后更新**: 2025-10-28  
**适用版本**: MineDojo 0.1, MineRL 1.0.0

---

## 🐛 问题清单

### 问题 1: pip/setuptools/wheel/numpy 版本依赖冲突

**症状**:
```
ERROR: Could not build wheels for minedojo/minerl
```

**原因**: 新版本的 pip (24.1+)、setuptools (58+) 和 wheel (0.38+) 与 MineDojo/MineRL 的构建脚本不兼容

**解决方案**:
```bash
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"
```

---

### 问题 2: Gradle 下载超时

**症状**:
```
Downloading https://services.gradle.org/distributions/gradle-4.10.2-all.zip
... (timeout or very slow)
```

**原因**: 国内访问 services.gradle.org 非常慢

**文件位置**:
```bash
# MineDojo
${CONDA_PREFIX}/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/gradle/wrapper/gradle-wrapper.properties

# MineRL
${CONDA_PREFIX}/lib/python3.9/site-packages/minerl/MCP-Reborn/gradle/wrapper/gradle-wrapper.properties
```

**解决方案**:
```bash
# 将
distributionUrl=https://services.gradle.org/distributions/gradle-4.10.2-all.zip

# 改为
distributionUrl=https://mirrors.aliyun.com/gradle/gradle-4.10.2-all.zip
```

---

### 问题 3: MixinGradle 找不到

**症状**:
```
Could not resolve com.github.SpongePowered:MixinGradle:dcfaf61
```

**原因**: JitPack 上的 MixinGradle 仓库无法访问或已失效

**解决方案**:

#### 步骤 1: 克隆本地 MixinGradle
```bash
sudo mkdir -p /opt/hotfix/
cd /opt/hotfix/
git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```

#### 步骤 2: 修改 build.gradle

**文件位置**:
```bash
# MineDojo
${CONDA_PREFIX}/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build.gradle

# MineRL
${CONDA_PREFIX}/lib/python3.9/site-packages/minerl/MCP-Reborn/build.gradle
```

**修改内容**:

1. 在 `buildscript` -> `repositories` 中添加：
```gradle
buildscript {
    repositories {
        maven { url "file:///opt/hotfix" }  // ← 添加这一行
        maven { url 'https://jitpack.io' }
        // ...
    }
}
```

2. 替换 MixinGradle 依赖：
```gradle
// 将
classpath('com.github.SpongePowered:MixinGradle:dcfaf61')

// 改为
classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61')
```

---

### 问题 4: ForgeGradle 找不到（MineDojo 特有）

**症状**:
```
Could not resolve com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT
```

**原因**: JitPack 上 brandonhoughton 的仓库已失效，MineDojo 已迁移到自己的仓库

**解决方案**:

**文件**: `minedojo/sim/Malmo/Minecraft/build.gradle`

1. 替换 buildscript classpath:
```gradle
// 将
classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'

// 改为
classpath 'com.github.MineDojo:ForgeGradle:FG_2.2_patched-SNAPSHOT'
```

2. 替换 dependencies implementation:
```gradle
// 将
implementation 'com.github.brandonhoughton:forgegradle:FG_2.2_patched-SNAPSHOT'

// 改为
implementation 'com.github.MineDojo:Forgegradle:FG_2.2_patched-SNAPSHOT'
```

---

### 问题 5: schemas.index 文件找不到

**症状**:
```
File 'src/main/resources/schemas.index' not found
```

**原因**: 相对路径在某些环境下解析错误

**解决方案**:

**文件**: `build.gradle`

```gradle
// 将
def schemaIndexFile = new File('src/main/resources/schemas.index')

// 改为
def schemaIndexFile = new File(projectDir, 'src/main/resources/schemas.index')
```

---

## 🚀 自动化修复

### 使用修复脚本（推荐）

项目提供了自动化修复脚本：

```bash
# 修复 MineDojo
./scripts/fix_minecraft_build.sh minedojo

# 修复 MineRL
./scripts/fix_minecraft_build.sh minerl
```

**脚本功能**:
- ✅ 自动检测安装路径
- ✅ 克隆 MixinGradle 到 /opt/hotfix
- ✅ 修改 gradle-wrapper.properties
- ✅ 修改 build.gradle（所有问题）
- ✅ 备份原始文件
- ✅ 验证修复结果

---

## 📝 完整安装流程

### MineDojo 安装（国内环境）

```bash
# 1. 激活环境
conda activate minedojo-x86

# 2. 降级构建工具
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"

# 3. 安装 numpy
pip install "numpy==1.24.3"

# 4. 安装 MineDojo（不会编译 Minecraft）
pip install minedojo

# 5. 应用修复
./scripts/fix_minecraft_build.sh minedojo

# 6. 测试（首次会编译 Minecraft）
python -c "
import minedojo
env = minedojo.make('harvest_1_log')
obs = env.reset()
env.close()
print('✓ MineDojo 安装成功')
"
```

### MineRL 安装（国内环境）

```bash
# 1. 激活环境
conda activate minedojo-x86

# 2. 降级构建工具
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"

# 3. 安装 numpy
pip install "numpy==1.24.3"

# 4. 从 GitHub 克隆 MineRL
cd /tmp
git clone https://github.com/minerllabs/minerl.git
cd minerl
git checkout v1.0.0
git submodule update --init --recursive

# 5. 修改 MCP-Reborn/launchClient.sh
# 添加 -XstartOnFirstThread（macOS 必需）

# 6. 安装 MineRL（不会编译）
pip install -e .

# 7. 应用修复
cd /path/to/aimc
./scripts/fix_minecraft_build.sh minerl

# 8. 安装正确的 OpenCV
pip install opencv-python==4.8.1.78 --force-reinstall

# 9. 测试（首次会编译 Minecraft）
python -c "
import gym
import minerl
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()
env.close()
print('✓ MineRL 安装成功')
"
```

---

## 🔍 验证修复

### 检查清单

运行以下命令检查修复是否正确应用：

```bash
# 设置变量
PACKAGE="minedojo"  # 或 "minerl"
CONDA_ENV="minedojo-x86"

# MineDojo 路径
if [ "$PACKAGE" = "minedojo" ]; then
    MC_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft"
else
    MC_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/minerl/MCP-Reborn"
fi

# 1. 检查 Gradle 镜像
echo "1. Gradle 镜像:"
grep "distributionUrl" "$MC_DIR/gradle/wrapper/gradle-wrapper.properties"
# 应该显示: distributionUrl=https://mirrors.aliyun.com/gradle/...

# 2. 检查 maven 本地仓库
echo "2. maven 本地仓库:"
grep "file:///opt/hotfix" "$MC_DIR/build.gradle"
# 应该显示: maven { url "file:///opt/hotfix" }

# 3. 检查 MixinGradle
echo "3. MixinGradle:"
grep "MixinGradle-dcfaf61" "$MC_DIR/build.gradle"
# 应该显示: classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61')

# 4. 检查 schemas.index
echo "4. schemas.index:"
grep "projectDir" "$MC_DIR/build.gradle" | grep schemas
# 应该显示: new File(projectDir, 'src/main/resources/schemas.index')

# 5. 检查 ForgeGradle（MineDojo）
if [ "$PACKAGE" = "minedojo" ]; then
    echo "5. ForgeGradle:"
    grep "com.github.MineDojo:ForgeGradle" "$MC_DIR/build.gradle"
    # 应该显示包含 MineDojo 的行
fi
```

---

## 🐛 故障排查

### Minecraft 编译仍然失败

**检查日志**:
```bash
tail -100 logs/mc_*.log
```

**常见错误**:

1. **权限问题**:
```bash
sudo chown -R $USER /opt/hotfix
```

2. **Gradle 缓存问题**:
```bash
rm -rf ~/.gradle/caches
```

3. **JitPack 仍然超时**:
```bash
# 添加更多国内镜像到 build.gradle
maven { url "https://maven.aliyun.com/repository/public" }
maven { url "https://maven.aliyun.com/repository/central" }
```

### 修复后仍然使用旧配置

**原因**: Minecraft 已经编译过，使用的是缓存的 JAR 包

**解决**:
```bash
# 删除编译缓存
rm -rf $MC_DIR/build/
rm -rf $MC_DIR/.gradle/

# 重新运行环境
python -c "import minedojo; env = minedojo.make('harvest_1_log'); env.reset(); env.close()"
```

---

## 📚 相关文档

- **自动化脚本**: [scripts/fix_minecraft_build.sh](../../scripts/fix_minecraft_build.sh)
- **补丁文件**: [docker/mc_config.patch](../../docker/mc_config.patch)
- **MineRL 安装**: [guides/MINERL_GUIDE.md](../guides/MINERL_GUIDE.md)
- **部署指南**: [DEPLOYMENT.md](../../DEPLOYMENT.md)

---

## 🙏 致谢

感谢以下项目和贡献者：

- [MineDojo](https://github.com/MineDojo/MineDojo) - Minecraft RL 环境
- [MineRL](https://github.com/minerllabs/minerl) - BASALT 竞赛平台
- [MixinGradle-dcfaf61](https://github.com/verityw/MixinGradle-dcfaf61) - 修复版 MixinGradle
- 阿里云 - Gradle 和 Maven 镜像服务

---

**文档创建**: 2025-10-28  
**维护状态**: 活跃  
**问题反馈**: [GitHub Issues](https://github.com/your-repo/aimc/issues)

