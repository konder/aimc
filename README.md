# AIMC - MineDojo AI Minecraft 训练工程

AI agent training project for Minecraft using MineDojo.

---

## 快速开始

### 1. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 2. 测试环境

```bash
# 运行Hello World示例
python src/hello_minedojo.py

# 运行任务演示
python src/demo_harvest_task.py
```

### 3. 开始训练（获得木头 MVP）⭐

```bash
# 快速测试（10K步，5-10分钟）
./scripts/train_get_wood.sh test --mineclip

# 标准训练（200K步，2-4小时，推荐）
./scripts/train_get_wood.sh --mineclip

# 长时间训练（500K步，5-10小时）
./scripts/train_get_wood.sh long --mineclip
```

**使用MineCLIP可获得3-5倍训练加速！**

### 4. 评估模型

```bash
# 评估训练好的模型
python scripts/evaluate_get_wood.py

# 评估特定模型
python scripts/evaluate_get_wood.py --model checkpoints/get_wood/get_wood_10000_steps.zip --episodes 20
```

### 5. 查看训练数据和 Loss

```bash
# TensorBoard 可视化（查看 loss 曲线）
tensorboard --logdir logs/tensorboard
# 浏览器打开: http://localhost:6006
# 在 SCALARS 标签页查看所有指标

# 实时监控日志
./scripts/monitor_training.sh
```

**关键指标位置**：
- 📈 `rollout/ep_rew_mean` - 平均奖励
- 📉 `train/policy_loss` - 策略损失
- 📉 `train/value_loss` - 价值损失

---

## 🚀 加速训练方法（新！）

**问题**：从零开始训练Minecraft技能太慢（可能需要数天到数周）  
**解决方案**：使用加速训练方法，获得**3-10倍**的训练速度提升

### 方法1：MineCLIP密集奖励（推荐首选）⭐

使用MineDojo内置的MineCLIP预训练模型提供密集奖励信号：

```bash
# 快速训练砍树技能（3-5倍加速）
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000

# 训练采矿技能
./scripts/train_with_mineclip.sh --task mine_stone --timesteps 200000

# 训练其他技能
./scripts/train_with_mineclip.sh --task harvest_wool --timesteps 150000
```

**效果**：
- ✅ 训练时间从4-8天 → 1-2天
- ✅ 一行命令即可使用
- ✅ 适用于所有MineDojo任务

### 方法2：课程学习（系统化训练）

从简单到困难逐步训练，更稳定、性能更好：

```bash
# 砍树技能（4个难度级别）
./scripts/train_curriculum.sh --skill chop_tree

# 采矿技能
./scripts/train_curriculum.sh --skill mine_stone

# 狩猎技能
./scripts/train_curriculum.sh --skill hunt_animal
```

**课程结构**：
- Level 1: 近距离 + 有工具（50K步）
- Level 2: 中距离 + 有工具（100K步）
- Level 3: 远距离 + 有工具（100K步）
- Level 4: 完整任务（250K步）

### 方法3：技能库管理（组合技能）

训练多个技能并组合使用：

```bash
# 添加技能到库
./scripts/manage_skill_library.sh add chop_tree checkpoints/curriculum/chop_tree/chop_tree_final.zip

# 查看技能库
./scripts/manage_skill_library.sh list

# 查看技能详情
./scripts/manage_skill_library.sh info chop_tree
```

### 完整指南

- 📖 **[快速开始加速训练](docs/guides/QUICK_START_ACCELERATED_TRAINING.md)** - 1小时内上手
- 📚 **[加速训练完整指南](docs/guides/TRAINING_ACCELERATION_GUIDE.md)** - 所有方法详解
- 📊 **[方法对比](docs/guides/TRAINING_METHODS_COMPARISON.md)** - 选择最适合的方案

### 推荐路线（2-3周完成）

```
第1周：MineCLIP训练5-10个基础技能
  └── 每个技能 150K-200K 步（1-2天/技能）

第2周：课程学习优化核心技能
  └── 关键技能 500K 步（2-3天/技能）

第3周：组合技能并评估
  └── 构建技能库，训练元策略
```

**预期效果**：
- ⚡ 训练时间缩短 **70-90%**
- 🎯 最终性能提升 **20-30%**
- 🔧 10-15个可组合的技能

---

## 项目结构

```
aimc/
├── src/                      # 源代码
│   ├── utils/               # 工具模块
│   │   └── env_wrappers.py  # 环境包装器
│   ├── training/            # 训练模块
│   │   └── train_harvest_paper.py  # 训练脚本
│   ├── examples/            # 示例代码
│   └── demo_harvest_task.py # 任务演示
├── scripts/                 # 脚本
│   ├── train_harvest.sh     # 训练启动脚本
│   └── eval_harvest.sh      # 评估脚本
├── config/                  # 配置文件
│   └── training_config.yaml # 训练配置
├── docs/                    # 文档
│   ├── QUICK_START_TRAINING.md       # 快速开始
│   ├── TRAINING_HARVEST_PAPER.md     # 训练指南
│   └── MINEDOJO_TASKS_GUIDE.md       # 任务系统指南
├── checkpoints/             # 模型检查点
├── logs/                    # 日志
│   ├── training/           # 训练日志
│   └── tensorboard/        # TensorBoard日志
└── requirements.txt         # 依赖
```

---

## 文档

### MVP训练指南（新用户从这里开始）⭐
- **[获得木头训练指南](docs/guides/GET_WOOD_TRAINING_GUIDE.md)**: 🎯 **MVP任务，2-4小时完成**

### 加速训练指南
- **[快速开始加速训练](docs/guides/QUICK_START_ACCELERATED_TRAINING.md)**: 🚀 1小时上手
- **[MineCLIP详解](docs/guides/MINECLIP_EXPLAINED.md)**: 🧠 MineCLIP工作原理与应用
- **[加速训练完整指南](docs/guides/TRAINING_ACCELERATION_GUIDE.md)**: 所有加速方法详解
- **[MineRL数据集指南](docs/guides/MINERL_DATASET_GUIDE.md)**: 📦 离线RL数据集使用
- **[高级训练问题解答](docs/guides/ADVANCED_TRAINING_SOLUTIONS.md)**: 💡 数据不足、离线训练等
- **[常见问题FAQ](docs/FAQ.md)**: ❓ 快速问答（15个常见问题）

### 训练指南
- **[性能优化指南](docs/guides/QUICK_PERFORMANCE_GUIDE.md)**: 无头模式性能优化
- **[训练监控](docs/guides/MONITORING_TRAINING.md)**: 监控训练进度和Loss
- **[TensorBoard使用](docs/guides/TENSORBOARD_GUIDE.md)**: TensorBoard可视化
- **[任务快速开始](docs/guides/TASKS_QUICK_START.md)**: MineDojo任务系统

### 技术文档
- **[性能分析](docs/technical/HEADLESS_VS_WINDOW_PERFORMANCE.md)**: 无头模式 vs 窗口模式详细分析
- **[MineDojo任务参考](docs/technical/MINEDOJO_TASKS_REFERENCE.md)**: 所有可用任务
- **[训练总结](docs/summary/TRAINING_HARVEST_PAPER.md)**: harvest_paper任务训练总结

---

## 🚀 性能优化（重要！）

**使用无头模式可以让训练速度提升 20-40%！**

```bash
# 方法1: 运行性能测试（推荐先测试）
bash scripts/run_benchmark.sh --skip-window

# 方法2: 启用无头模式训练（一行配置）
export JAVA_OPTS="-Djava.awt.headless=true"
bash scripts/train_harvest.sh

# 方法3: 项目训练脚本已默认启用
bash scripts/train_harvest.sh  # 已自动设置无头模式
```

**实际效果：**
- ⚡ 训练速度提升 **20-40%**
- 🔥 CPU使用降低 **5-15%**
- 💾 内存节省 **~200MB**
- ✅ 更稳定（适合长时间训练）

**典型案例：**
- M1 MacBook Pro 训练 500K 步：从 12.4小时 → 9.4小时（节省3小时）
- RTX 3090 训练 2M 步：从 9.6小时 → 6.6小时（节省3小时）

📖 **详细文档：** [性能优化快速指南](docs/guides/QUICK_PERFORMANCE_GUIDE.md) | [完整技术分析](docs/technical/HEADLESS_VS_WINDOW_PERFORMANCE.md)

---

## 核心特性

✅ **完整的训练流程**: 环境包装 → 模型训练 → 评估监控  
✅ **成熟的RL框架**: 使用 Stable-Baselines3 + PPO算法  
✅ **性能优化**: 无头模式训练，速度提升20-40%  
✅ **丰富的监控**: TensorBoard + 详细日志  
✅ **灵活配置**: YAML配置文件 + 命令行参数  
✅ **详细文档**: 从入门到优化的完整指南  

---

## 重要说明

⚠️ **MineDojo内置任务不提供预训练模型，所有训练从头开始！**

- 默认任务: `harvest_milk`（更稳定）
- 目标任务: `harvest_1_paper`（可配置）
- 训练时间: 2-16小时（取决于配置）

---

## 系统要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| GPU | 无 | GTX 1060+ |
| 存储 | 10GB | 20GB+ |

---

## ARM64 部署指南

### 如何在ARM64上通过Rosetta 2部署minedojo

- 安装x86的jdk
```bash
arch -x86_64 brew install temurin@8
```
- 设置JAVA_HOME，用arch开启一个bash
```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/
source ~/.bash_profile
arch -x86_64 /bin/bash
```
- 创建minedojo-x86的python虚拟环境
```bash
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86
```
- 安装minedojo前的国内代理（可选）
```bash
mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf && \
    echo "[install]" >> ~/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf
```
- 安装minedojo
```bash
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy>=1.21.0,<2.0"
pip install minedojo
```
- 解决编译Minecraft的MixinGradle问题
```bash
mkdir /opt/MixinGradle
cd /opt/MixinGradle && git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```
- 修复Malmo的编译Minecraft一系列问题
```bash
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
sed -i '/repositories {/a\        maven { url "file:///opt/hotfix" }' build.gradle
sed -i '4i\     maven { url "https://maven.aliyun.com/repository/public" }' build.gradle
sed -i '5i\     maven { url "https://maven.aliyun.com/repository/central" }' build.gradle
sed -i '6i\     maven { url "https://libraries.minecraft.net/" }' build.gradle
sed -i "s|com.github.SpongePowered:MixinGradle:dcfaf61|MixinGradle-dcfaf61:MixinGradle:dcfaf61|g" build.gradle
sed -i "s|brandonhoughton:ForgeGradle|MineDojo:ForgeGradle|g" build.gradle
sed -i "s|brandonhoughton:forgegradle|MineDojo:ForgeGradle|g" build.gradle
sed -i "s|new File('src/main/resources/schemas.index')|new File(projectDir, 'src/main/resources/schemas.index')|g" build.gradle
```
- 编译Minecraft前的代理（可选）
```bash
mkdir -p /root/.gradle
echo 'allprojects {\n\
    repositories {\n\
    maven { url "https://maven.aliyun.com/repository/public" }\n\
    maven { url "https://maven.aliyun.com/repository/central" }\n\
    maven { url "https://maven.aliyun.com/repository/gradle-plugin" }\n\
    maven { url "https://maven.aliyun.com/repository/spring" }\n\
    maven { url "https://maven.aliyun.com/repository/spring-plugin" }\n\
    maven { url "https://libraries.minecraft.net/" }\n\
    mavenCentral()\n\
    gradlePluginPortal()\n\
    mavenLocal()\n\
    }\n\
    }' > ~/.gradle/init.gradle
```
- 编译Mminecraft
```bash
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/gradlew shadowJar
mkdir /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle && cp -r ~/.gradle/caches /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle
```
- 如果有lwjgl问题，手动下载LWJGL-2.93库和修改launchClient.sh启用
    - [下载地址](https://sf-west-interserver-1.dl.sourceforge.net/project/java-game-lib/Official%20Releases/LWJGL%202.9.3/lwjgl-2.9.3.zip?viasf=1)
    - 修改/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/launchClient.sh
    - 将启动命令改为java -Djava.library.path=/Users/nanzhang/lwjgl-2.9.3/native/macosx -Dorg.lwjgl.librarypath=/Users/nanzhang/lwjgl-2.9.3/native/macosx -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin -Xmx2G -Dfile.encoding=UTF-8 -Duser.country=US -Duser.language=en -Duser.variant -jar ../build/libs/MalmoMod-0.37.0-fat.jar