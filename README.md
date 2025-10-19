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

### 3. 开始训练

```bash
# 检查设备支持（查看是否有 GPU 加速）
python scripts/check_device.py

# 快速测试（10K步，5-10分钟）
./scripts/train_harvest.sh test

# 完整训练（500K步，2-4小时）
./scripts/train_harvest.sh

# 监控训练
./scripts/monitor_training.sh
```

### 4. 评估模型

```bash
./scripts/eval_harvest.sh
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

- **[快速开始](docs/QUICK_START_TRAINING.md)**: 30秒开始训练
- **[训练指南](docs/TRAINING_HARVEST_PAPER.md)**: 完整训练文档
- **[任务系统指南](docs/MINEDOJO_TASKS_GUIDE.md)**: MineDojo任务机制详解

---

## 核心特性

✅ **完整的训练流程**: 环境包装 → 模型训练 → 评估监控  
✅ **成熟的RL框架**: 使用 Stable-Baselines3 + PPO算法  
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
```
arch -x86_64 brew install temurin@8
```
- 设置JAVA_HOME，用arch开启一个bash
```
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/
source ~/.bash_profile
arch -x86_64 /bin/bash
```
- 创建minedojo-x86的python虚拟环境
```
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86
```
- 安装minedojo前的国内代理（可选）
```
mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf && \
    echo "[install]" >> ~/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf
```
- 安装minedojo
```
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy>=1.21.0,<2.0"
pip install minedojo
```
- 解决编译Minecraft的MixinGradle问题
```
mkdir /opt/MixinGradle
cd /opt/MixinGradle && git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```
- 修复Malmo的编译Minecraft一系列问题
```
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
```
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
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/gradlew shadowJar
mkdir /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle && cp -r ~/.gradle/caches /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle
```
- 如果有lwjgl问题，手动下载LWJGL-2.93库和修改launchClient.sh启用
    - 下载https://sf-west-interserver-1.dl.sourceforge.net/project/java-game-lib/Official%20Releases/LWJGL%202.9.3/lwjgl-2.9.3.zip?viasf=1
    - 修改/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/launchClient.sh
    - 将启动命令改为java -Djava.library.path=/Users/nanzhang/lwjgl-2.9.3/native/macosx -Dorg.lwjgl.librarypath=/Users/nanzhang/lwjgl-2.9.3/native/macosx -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin -Xmx2G -Dfile.encoding=UTF-8 -Duser.country=US -Duser.language=en -Duser.variant -jar ../build/libs/MalmoMod-0.37.0-fat.jar