# Minecraft 编译修复方案总结

## 📋 问题背景

MineDojo 和 MineRL 在国内环境安装时，Minecraft JAR 包编译会失败。这是因为：

1. **国外依赖源访问慢**: Gradle、Maven 仓库
2. **依赖包失效**: JitPack 上的 MixinGradle、ForgeGradle
3. **构建工具版本问题**: pip 24.1+、setuptools 58+ 不兼容
4. **路径配置错误**: 相对路径在某些环境下解析失败

---

## ✅ 完整解决方案

### 1. 自动化修复脚本 ⭐

**文件**: `scripts/fix_minecraft_build.sh`

**功能**:
- ✅ 自动检测 MineDojo/MineRL 安装路径
- ✅ 克隆 MixinGradle 到 /opt/hotfix
- ✅ 修改 gradle-wrapper.properties（阿里云镜像）
- ✅ 修改 build.gradle（所有问题修复）
- ✅ 备份原始文件
- ✅ 验证修复结果

**使用方法**:
```bash
# MineDojo
./scripts/fix_minecraft_build.sh minedojo

# MineRL
./scripts/fix_minecraft_build.sh minerl
```

---

### 2. 详细问题文档

**文件**: `docs/issues/MINECRAFT_BUILD_FIXES.md`

**内容**:
- 📖 5 个关键问题的详细说明
- 🔧 手动修复步骤
- 📝 完整安装流程
- 🐛 故障排查指南
- ✅ 验证清单

---

### 3. 快速安装指南

**文件**: `INSTALL_GUIDE.md`

**内容**:
- 🚀 国内环境快速安装流程
- 📦 MineDojo 完整安装步骤
- 📦 MineRL 完整安装步骤（包含 OpenCV 修复）
- 🔍 验证和常见问题
- 💡 关键要点提示

---

## 🔧 修复的具体问题

### 问题 1: pip/setuptools/wheel 版本冲突

**修复**:
```bash
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"
```

---

### 问题 2: Gradle 下载源

**修复**: 
```
distributionUrl=https://mirrors.aliyun.com/gradle/gradle-4.10.2-all.zip
```

**位置**:
- MineDojo: `minedojo/sim/Malmo/Minecraft/gradle/wrapper/gradle-wrapper.properties`
- MineRL: `minerl/MCP-Reborn/gradle/wrapper/gradle-wrapper.properties`

---

### 问题 3: MixinGradle 依赖

**修复**:
1. 克隆本地仓库:
```bash
git clone https://github.com/verityw/MixinGradle-dcfaf61.git /opt/hotfix/MixinGradle-dcfaf61
```

2. 添加 maven 本地仓库:
```gradle
maven { url "file:///opt/hotfix" }
```

3. 替换依赖:
```gradle
classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61')
```

---

### 问题 4: ForgeGradle 依赖（MineDojo）

**修复**:
```gradle
// buildscript
classpath 'com.github.MineDojo:ForgeGradle:FG_2.2_patched-SNAPSHOT'

// dependencies
implementation 'com.github.MineDojo:Forgegradle:FG_2.2_patched-SNAPSHOT'
```

---

### 问题 5: schemas.index 路径

**修复**:
```gradle
def schemaIndexFile = new File(projectDir, 'src/main/resources/schemas.index')
```

---

## 📊 修复效果

### 修复前
```
❌ Gradle 下载超时
❌ MixinGradle 找不到
❌ ForgeGradle 找不到
❌ schemas.index 找不到
❌ Minecraft 编译失败
```

### 修复后
```
✅ Gradle 从阿里云下载（快速）
✅ MixinGradle 使用本地仓库
✅ ForgeGradle 使用 MineDojo 仓库
✅ schemas.index 路径正确
✅ Minecraft 编译成功
```

---

## 🎯 使用流程

### 标准流程

```bash
# 1. 安装 MineDojo
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"
pip install minedojo

# 2. 运行修复
./scripts/fix_minecraft_build.sh minedojo

# 3. 首次启动（会触发编译）
python -c "import minedojo; env = minedojo.make('harvest_1_log'); env.reset(); env.close()"
```

### M 芯片流程

```bash
# 1. x86 模式
arch -x86_64 /bin/zsh
conda activate minedojo-x86

# 2. 安装和修复
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy==1.24.3"
pip install minedojo
./scripts/fix_minecraft_build.sh minedojo

# 3. 测试
./scripts/run_minedojo_x86.sh python -c "import minedojo; env = minedojo.make('harvest_1_log'); env.reset(); env.close()"
```

---

## 📝 文档更新

### README.md

**添加内容**:
- ⚠️ 国内环境重要提示
- 📖 INSTALL_GUIDE.md 链接
- 🔧 修复脚本步骤

### 新增文档

1. **INSTALL_GUIDE.md** - 快速安装指南
2. **docs/issues/MINECRAFT_BUILD_FIXES.md** - 详细问题文档
3. **scripts/fix_minecraft_build.sh** - 自动修复脚本
4. **本文档** - 修复方案总结

---

## 💡 关键要点

### 必须执行

1. ✅ 降级 pip/setuptools/wheel
2. ✅ 安装 numpy==1.24.3
3. ✅ 运行 fix_minecraft_build.sh

### 顺序重要

```
pip install 工具降级
  ↓
pip install numpy
  ↓
pip install minedojo/minerl
  ↓
运行修复脚本  ← 关键！
  ↓
首次启动（触发编译）
```

### 国内专属

- Gradle 阿里云镜像
- Maven 阿里云镜像  
- MixinGradle 本地仓库
- ForgeGradle MineDojo 仓库

---

## 🐛 常见问题

### Q1: 运行修复脚本后仍然失败？

**A**: 删除 Gradle 缓存
```bash
rm -rf ~/.gradle/caches
```

### Q2: /opt/hotfix 权限问题？

**A**: 修改权限
```bash
sudo chown -R $USER /opt/hotfix
```

### Q3: 修复后发现配置没变？

**A**: Minecraft 可能已编译，删除缓存重试
```bash
rm -rf $MC_DIR/build/
rm -rf $MC_DIR/.gradle/
```

---

## 🎉 总结

### 成果

- ✅ 创建自动化修复脚本
- ✅ 编写详细问题文档
- ✅ 更新快速安装指南
- ✅ 完善 README 说明

### 影响

- 🚀 国内用户可以顺利安装
- ⏱️ 安装时间大幅减少
- 🐛 避免常见编译错误
- 📖 文档完整清晰

### 适用范围

- MineDojo 0.1
- MineRL 0.4.4, 1.0.0
- macOS (Intel/M系列)
- Linux x86_64
- 国内网络环境

---

**文档创建**: 2025-10-28  
**测试环境**: macOS 14.4.1 ARM64 + x86  
**维护状态**: 活跃

