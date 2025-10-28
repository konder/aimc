# AIMC 完整部署文档

## 📋 文档说明

本文档提供 AIMC 项目的**完整、可复现的部署流程**，确保在任何环境下都能恢复到当前的工作状态。

**最后更新**: 2025-10-28  
**环境快照**: 已测试并验证的完整配置  
**适用场景**: 新机器部署、环境迁移、灾难恢复

---

## 🚨 重要提醒（国内环境）

**MineDojo 和 MineRL 在国内环境安装时**，必须在首次启动前修复 Minecraft 编译配置！

问题原因：
1. Gradle 下载源访问慢（services.gradle.org）
2. JitPack 依赖失效（MixinGradle、ForgeGradle）
3. pip/setuptools 版本不兼容

**解决方案**：使用自动修复脚本
```bash
./scripts/fix_minecraft_build.sh minedojo  # 或 minerl
```

详见：[第6章 - Minecraft 编译问题修复](#6-minecraft-编译问题修复国内必读)

---

## 📦 系统要求

### 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| **CPU** | 4核 | 8核+ |
| **内存** | 8GB | 16GB+ |
| **GPU** | 无 | NVIDIA / Apple MPS |
| **存储** | 20GB | 50GB+ |

### 操作系统

| 系统 | 版本 | 架构 | 支持 |
|------|------|------|------|
| **macOS** | 10.15+ | x86_64 | ✅ |
| **macOS** | 13+ | ARM64 (M系列) | ✅ (Rosetta 2) |
| **Ubuntu/Debian** | 18.04+ | x86_64 | ✅ |
| **Windows** | - | - | ❌ (建议 WSL2) |

---

## 🚀 快速开始

### 方案 A: 标准部署（Linux / Intel Mac）

```bash
# 1. 安装 Java 8
# macOS: brew install --cask temurin@8
# Ubuntu: sudo apt install openjdk-8-jdk

# 2. 创建 Python 环境
conda create -n minedojo python=3.9 -y
conda activate minedojo

# 3. 降级构建工具（重要！）
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"

# 4. 克隆项目
git clone <仓库地址> aimc
cd aimc

# 5. 安装 MineDojo
pip install minedojo

# 6. 修复 Minecraft 编译配置（国内必需）
./scripts/fix_minecraft_build.sh minedojo

# 7. 安装项目依赖
pip install -r requirements.txt

# 8. 验证
python tools/validate_environment.py
```

### 方案 B: Apple M 芯片部署

```bash
# 1. 安装 Rosetta 2
softwareupdate --install-rosetta --agree-to-license

# 2. 安装 x86 Java
arch -x86_64 brew install --cask temurin@8

# 3. x86 模式创建环境
arch -x86_64 /bin/zsh
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86

# 4. 降级构建工具（重要！）
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"

# 5. 安装 MineDojo
pip install minedojo

# 6. 修复 Minecraft 编译配置（国内必需）
cd /path/to/aimc
./scripts/fix_minecraft_build.sh minedojo

# 7. 安装项目依赖
pip install -r requirements.txt

# 8. 验证（使用启动脚本）
./scripts/run_minedojo_x86.sh python tools/validate_environment.py
```

**注意**: M 芯片用户每次运行都需要使用 `./scripts/run_minedojo_x86.sh` 或手动切换到 x86 模式

---

## 📋 依赖版本说明

### 核心依赖（必需精确版本）

| 包名 | 版本 | 原因 |
|------|------|------|
| **Python** | 3.9 | MineDojo 兼容性 |
| **numpy** | 1.24.3 | MineDojo 不支持 2.0+ |
| **gym** | 0.21.0 | MineDojo/MineCLIP 要求 |
| **opencv-python** | 4.8.1.78 | MineRL 窗口显示关键版本 |
| **pip** | <24.1 | Minecraft 编译兼容性 |
| **setuptools** | <58 | Minecraft 编译兼容性 |
| **wheel** | <0.38.0 | Minecraft 编译兼容性 |

### 依赖文件

```
requirements.txt         ← 核心依赖（推荐）
requirements-freeze.txt  ← 完整冻结版本（精确复现）
```

**使用建议**:
- **新部署**: 使用 `requirements.txt`（更灵活）
- **精确复现**: 使用 `requirements-freeze.txt`（完全一致）

---

## 📝 详细部署步骤

### 1. 安装系统依赖

#### macOS (Intel)
```bash
brew install --cask temurin@8
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
echo 'export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home' >> ~/.zshrc
```

#### macOS (M系列)
```bash
# 安装 x86 版本
arch -x86_64 brew install --cask temurin@8
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
echo 'export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home' >> ~/.zshrc
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk build-essential git
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
```

### 2. 创建 Python 环境

```bash
# 标准环境
conda create -n minedojo python=3.9 -y
conda activate minedojo

# M 芯片（需要 x86 模式）
arch -x86_64 /bin/zsh
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86
```

### 3. 降级构建工具（关键步骤）

**为什么需要降级**:
- pip 24.1+ 与 MineDojo/MineRL 构建脚本不兼容
- setuptools 58+ 会导致编译失败
- wheel 0.38+ 有兼容性问题

```bash
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"
```

### 4. 安装 MineDojo

```bash
pip install minedojo
```

**注意**: 这一步**不会**编译 Minecraft，只是安装 Python 包。

### 5. 克隆项目

```bash
git clone <仓库地址> aimc
cd aimc
```

### 6. 修复 Minecraft 编译配置（国内必需）⭐

**关键步骤**: 在首次启动 MineDojo/MineRL 之前运行修复脚本！

```bash
# MineDojo
./scripts/fix_minecraft_build.sh minedojo

# MineRL（如需要）
./scripts/fix_minecraft_build.sh minerl
```

**修复内容**:
- ✅ Gradle 下载源 → 阿里云镜像
- ✅ MixinGradle → 本地仓库 (/opt/hotfix)
- ✅ ForgeGradle → MineDojo 仓库
- ✅ Maven → 阿里云镜像
- ✅ schemas.index → 路径修复

详见：[第6章 - Minecraft 编译问题修复](#6-minecraft-编译问题修复国内必读)

### 7. 安装项目依赖

```bash
# 核心依赖（推荐）
pip install -r requirements.txt

# 或精确版本（完全复现）
pip install -r requirements-freeze.txt
```

### 8. 验证安装

```bash
# 标准环境
python tools/validate_environment.py

# M 芯片
./scripts/run_minedojo_x86.sh python tools/validate_environment.py
```

应该看到：
```
✅ 所有检查通过！环境配置正确
```

---

## 🔧 Minecraft 编译问题修复（国内必读）

### 问题概述

MineDojo 和 MineRL 在国内环境安装时，会遇到 5 个关键编译问题：

1. ❌ **Gradle 下载超时** - services.gradle.org 访问慢
2. ❌ **MixinGradle 找不到** - JitPack 失效
3. ❌ **ForgeGradle 找不到** - MineDojo 特有
4. ❌ **schemas.index 路径错误** - 相对路径问题
5. ❌ **构建工具版本冲突** - pip/setuptools 不兼容

### 自动修复（推荐）⭐

```bash
# MineDojo
./scripts/fix_minecraft_build.sh minedojo

# MineRL
./scripts/fix_minecraft_build.sh minerl
```

**脚本功能**:
- ✅ 自动检测安装路径
- ✅ 克隆 MixinGradle 到 /opt/hotfix
- ✅ 修改 gradle-wrapper.properties
- ✅ 修改 build.gradle（所有问题）
- ✅ 备份原始文件
- ✅ 验证修复结果

### 手动修复步骤

如果自动脚本失败，可以手动修复：

#### 问题 1: Gradle 下载源

**文件**: `gradle/wrapper/gradle-wrapper.properties`

```properties
# 将
distributionUrl=https://services.gradle.org/distributions/gradle-4.10.2-all.zip

# 改为
distributionUrl=https://mirrors.aliyun.com/gradle/gradle-4.10.2-all.zip
```

#### 问题 2: MixinGradle 依赖

**步骤 1**: 克隆本地仓库
```bash
sudo mkdir -p /opt/hotfix/
cd /opt/hotfix/
git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```

**步骤 2**: 修改 build.gradle
```gradle
buildscript {
    repositories {
        maven { url "file:///opt/hotfix" }  // ← 添加这一行
        maven { url "https://maven.aliyun.com/repository/public" }
        maven { url "https://maven.aliyun.com/repository/central" }
        maven { url 'https://jitpack.io' }
        // ...
    }
    dependencies {
        // 将
        classpath('com.github.SpongePowered:MixinGradle:dcfaf61')
        
        // 改为
        classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61')
    }
}
```

#### 问题 3: ForgeGradle（MineDojo 特有）

**文件**: `build.gradle`

```gradle
// buildscript 中，将
classpath 'com.github.brandonhoughton:ForgeGradle:FG_2.2_patched-SNAPSHOT'

// 改为
classpath 'com.github.MineDojo:ForgeGradle:FG_2.2_patched-SNAPSHOT'

// dependencies 中，将
implementation 'com.github.brandonhoughton:forgegradle:FG_2.2_patched-SNAPSHOT'

// 改为
implementation 'com.github.MineDojo:Forgegradle:FG_2.2_patched-SNAPSHOT'
```

#### 问题 4: schemas.index 路径

**文件**: `build.gradle`

```gradle
// 将
def schemaIndexFile = new File('src/main/resources/schemas.index')

// 改为
def schemaIndexFile = new File(projectDir, 'src/main/resources/schemas.index')
```

### 验证修复

```bash
# 获取包路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# MineDojo
MC_DIR="$SITE_PACKAGES/minedojo/sim/Malmo/Minecraft"

# MineRL
MC_DIR="$SITE_PACKAGES/minerl/MCP-Reborn"

# 检查 Gradle 镜像
grep "aliyun" "$MC_DIR/gradle/wrapper/gradle-wrapper.properties"

# 检查 MixinGradle
grep "MixinGradle-dcfaf61" "$MC_DIR/build.gradle"

# 检查本地仓库
ls -la /opt/hotfix/MixinGradle-dcfaf61/
```

---

## 🎯 可选组件安装

### MineRL 1.0.0（BASALT 任务）

```bash
# 1. 从 GitHub 克隆
cd /tmp
git clone https://github.com/minerllabs/minerl.git
cd minerl && git checkout v1.0.0
git submodule update --init --recursive

# 2. 修改 launchClient.sh（macOS 必需）
# 编辑 MCP-Reborn/launchClient.sh
# 在 java 命令行添加 -XstartOnFirstThread

# 3. 安装
pip install -e .

# 4. 修复编译配置
cd /path/to/aimc
./scripts/fix_minecraft_build.sh minerl

# 5. 安装正确的 OpenCV
pip install opencv-python==4.8.1.78 --force-reinstall

# 6. 测试
python -c "import gym, minerl; env = gym.make('MineRLBasaltFindCave-v0'); env.reset(); env.close(); print('✓ OK')"
```

详见: [docs/guides/MINERL_GUIDE.md](docs/guides/MINERL_GUIDE.md)

### VPT 预训练模型

```bash
mkdir -p data/pretrained/vpt
cd data/pretrained/vpt
# 下载模型（根据需要）
```

详见: [docs/guides/VPT_ZERO_SHOT_QUICKSTART.md](docs/guides/VPT_ZERO_SHOT_QUICKSTART.md)

---

## 🔍 环境验证

### 验证工具

```bash
# 完整验证（推荐）
python tools/validate_environment.py

# 输出示例
✓ Python 版本: 3.9.x
✓ 系统架构: x86_64
✓ Java 环境: 1.8.0_xxx
✓ 核心包版本正确
✓ MineDojo 环境创建成功
✅ 所有检查通过！
```

### 手动验证清单

- [ ] Python 3.9
- [ ] Java 8
- [ ] numpy < 2.0
- [ ] gym == 0.21.0
- [ ] opencv-python == 4.8.1.78（如安装 MineRL）
- [ ] MineDojo 环境可创建
- [ ] 架构 x86_64（M 芯片用户需在 x86 模式）

### MineDojo 测试

```bash
python -c "
import minedojo
env = minedojo.make('harvest_1_log', image_size=(160, 256))
obs = env.reset()
for _ in range(5):
    obs, reward, done, info = env.step(env.action_space.no_op())
env.close()
print('✓ MineDojo 测试通过')
"
```

---

## 🐛 常见问题排查

### Q1: Minecraft 编译失败

**症状**: `Minecraft process failed to start` 或 Gradle 错误

**解决**:
```bash
# 1. 确认已运行修复脚本
./scripts/fix_minecraft_build.sh minedojo

# 2. 删除 Gradle 缓存
rm -rf ~/.gradle/caches

# 3. 重新测试
python -c "import minedojo; env = minedojo.make('harvest_1_log'); env.reset(); env.close()"
```

### Q2: numpy 版本冲突

**症状**: `numpy.core._multiarray_umath failed to import`

**解决**:
```bash
pip uninstall numpy -y
pip install numpy==1.24.3
```

### Q3: M 芯片架构错误

**症状**: `Bad CPU type in executable`

**解决**:
```bash
# 确保在 x86 模式
arch -x86_64 /bin/zsh
uname -m  # 应显示 x86_64

# 或使用启动脚本
./scripts/run_minedojo_x86.sh <命令>
```

### Q4: Java 版本错误

**症状**: `JAVA_HOME not set` 或 `UnsupportedClassVersionError`

**解决**:
```bash
# 设置 JAVA_HOME
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home

# 验证
java -version  # 应显示 1.8.0
echo $JAVA_HOME
```

### Q5: OpenCV 窗口不显示（MineRL）

**症状**: `cv2.imshow` 失败

**解决**:
```bash
pip install opencv-python==4.8.1.78 --force-reinstall
```

---

## 🔄 环境迁移

### 导出当前环境

```bash
# 1. 导出 conda 环境
conda env export > environment.yml

# 2. 导出 pip 依赖
pip list --format=freeze > requirements-current.txt

# 3. 记录系统信息
uname -a > system_info.txt
python --version >> system_info.txt
java -version 2>> system_info.txt
```

### 在新机器恢复

```bash
# 1. 从 environment.yml 创建
conda env create -f environment.yml

# 或使用 pip
conda create -n minedojo python=3.9 -y
conda activate minedojo
pip install -r requirements-freeze.txt

# 2. 复制项目和数据
scp -r old-machine:/path/to/aimc ./

# 3. 运行修复脚本（国内环境）
cd aimc
./scripts/fix_minecraft_build.sh minedojo

# 4. 验证
python tools/validate_environment.py
```

---

## 📂 目录结构

部署完成后的标准目录结构：

```
aimc/
├── src/                    # 源代码
├── scripts/                # 脚本
│   ├── fix_minecraft_build.sh
│   ├── run_minedojo_x86.sh
│   └── ...
├── docs/                   # 文档
├── tools/                  # 工具
│   └── validate_environment.py
├── data/                   # 数据
│   ├── tasks/
│   ├── pretrained/
│   └── mineclip/
├── logs/                   # 日志
├── requirements.txt        # 核心依赖
├── requirements-freeze.txt # 冻结依赖
└── DEPLOYMENT.md          # 本文档
```

---

## 📚 相关文档

- **快速开始**: [README.md](README.md)
- **DAgger 训练**: [docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md](docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md)
- **VPT 使用**: [docs/guides/VPT_ZERO_SHOT_QUICKSTART.md](docs/guides/VPT_ZERO_SHOT_QUICKSTART.md)
- **MineRL 安装**: [docs/guides/MINERL_GUIDE.md](docs/guides/MINERL_GUIDE.md)
- **常见问题**: [FAQ.md](FAQ.md)

---

## 🎉 部署完成检查清单

部署完成后，确认以下所有项：

- [ ] Python 3.9 环境已创建并激活
- [ ] Java 8 已安装并配置 JAVA_HOME
- [ ] pip/setuptools/wheel 已降级
- [ ] numpy==1.24.3 已安装
- [ ] MineDojo 已安装
- [ ] 修复脚本已运行（国内环境）
- [ ] 项目代码已克隆
- [ ] 项目依赖已安装
- [ ] `tools/validate_environment.py` 通过
- [ ] MineDojo 环境可以创建和运行
- [ ] （M芯片）x86 启动脚本可用
- [ ] （可选）MineRL 已安装并测试
- [ ] （可选）VPT 模型已下载

---

## 💡 关键要点总结

### 必须执行的步骤

1. ⭐ **降级构建工具** - `pip<24.1`, `setuptools<58`, `wheel<0.38.0`
2. ⭐ **固定 numpy 版本** - `numpy==1.24.3`
3. ⭐ **运行修复脚本**（国内） - `./scripts/fix_minecraft_build.sh`

### 执行顺序

```
降级工具 → 安装 numpy → 安装 MineDojo → 运行修复 → 首次启动
```

### M 芯片特别注意

- 所有命令需在 x86 模式执行
- 使用 `./scripts/run_minedojo_x86.sh` 运行
- 环境名称: `minedojo-x86`

### 国内环境必做

- Gradle 阿里云镜像
- Maven 阿里云镜像
- MixinGradle 本地仓库
- ForgeGradle MineDojo 仓库

---

**文档创建**: 2025-10-28  
**验证环境**: macOS 14.4.1 (ARM64 + Rosetta 2 x86)  
**Python**: 3.9  
**MineDojo**: 0.1  
**状态**: ✅ 已测试并验证

