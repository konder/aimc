# AIMC - MineDojo AI Minecraft 训练工程

基于 MineDojo 的 Minecraft AI 智能体训练项目，使用强化学习训练智能体完成各种 Minecraft 任务。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MineDojo](https://img.shields.io/badge/MineDojo-Latest-green.svg)](https://github.com/MineDojo/MineDojo)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目介绍

AIMC 是一个完整的 Minecraft AI 训练工程，专注于使用强化学习（PPO算法）训练智能体在 MineDojo 环境中完成各种任务。

### 核心特性

✅ **完整的训练流程**: 环境包装 → 模型训练 → 评估监控  
✅ **成熟的RL框架**: 使用 Stable-Baselines3 + PPO 算法  
✅ **加速训练方案**: MineCLIP 密集奖励，3-5倍训练加速  
✅ **性能优化**: 无头模式训练，速度提升 20-40%  
✅ **丰富的监控**: TensorBoard + 实时日志  
✅ **灵活配置**: YAML 配置文件 + 命令行参数  
✅ **详细文档**: 从入门到优化的完整指南  

### 技术栈

- **环境**: MineDojo (Minecraft 仿真环境)
- **算法**: PPO (Proximal Policy Optimization)
- **框架**: Stable-Baselines3
- **加速**: MineCLIP (视觉-语言多模态模型)
- **可视化**: TensorBoard

### 支持的任务类型

- 🪵 **采集任务**: 获得木头、石头、煤炭等
- 🐄 **收集任务**: 收集牛奶、羊毛、苹果等
- 🌾 **农业任务**: 种植和收获小麦等作物
- ⚔️ **战斗任务**: 狩猎动物、击败怪物
- 🏗️ **建造任务**: 制作工具、建造结构

---

## 📁 项目结构

```
aimc/
├── src/                          # 源代码
│   ├── training/                 # 训练模块
│   │   ├── __init__.py
│   │   └── train_get_wood.py     # 获得木头训练脚本（MVP）
│   └── utils/                    # 工具模块
│       ├── __init__.py
│       ├── env_wrappers.py       # 环境包装器
│       └── realtime_logger.py    # 实时日志工具
│
├── scripts/                      # 脚本
│   ├── train_get_wood.sh         # 获得木头训练启动脚本
│   ├── tensorboard_manager.sh    # TensorBoard 管理脚本
│   ├── run_minedojo_x86.sh       # x86/Rosetta2 运行脚本
│   └── validate_install.py       # 安装验证脚本
│
├── config/                       # 配置文件
│   └── training_config.yaml      # 训练配置
│
├── docs/                         # 文档
│   ├── guides/                   # 指南文档
│   ├── summaries/                # 总结文档
│   ├── technical/                # 技术文档
│
├── checkpoints/                  # 模型检查点
│   └── harvest_paper/            # harvest_paper 任务检查点
│
├── logs/                         # 日志
│   ├── training/                 # 训练日志
│   ├── tensorboard/              # TensorBoard 日志
│   └── watchdog/                 # 监控日志
│
├── README.md                     # 项目说明（本文件）
└── requirements.txt              # Python 依赖
```

---

## 🚀 部署指南

### 系统要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| GPU | 无 | GTX 1060+ 或 Apple M 系列 |
| 存储 | 10GB | 20GB+ |
| 系统 | macOS 10.15+ / Ubuntu 18.04+ | macOS 13+ / Ubuntu 22.04+ |

### 标准部署（Linux / Intel Mac）

#### 1. 安装 Java

MineDojo 需要 Java 8 或更高版本：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-8-jdk

# macOS (Intel)
brew install openjdk@8

# 验证安装
java -version
```

#### 2. 创建 Python 环境

```bash
# 创建虚拟环境
conda create -n minedojo python=3.9 -y
conda activate minedojo

# 或使用 venv
python3.9 -m venv minedojo-env
source minedojo-env/bin/activate
```

#### 3. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-repo/aimc.git
cd aimc

# 安装依赖
pip install -r requirements.txt
```

#### 4. 验证安装

```bash
# 运行验证脚本
python scripts/validate_install.py

# 应该看到：
# ✓ Python 版本正确
# ✓ MineDojo 已安装
# ✓ Java 可用
# ✓ 环境创建成功
```

---

### Apple M 芯片部署（ARM64）⭐

Apple M 系列芯片需要通过 Rosetta 2 运行 MineDojo（因为 Minecraft 服务端需要 x86 架构）。

#### 1. 安装 Rosetta 2

```bash
# 安装 Rosetta 2（如果尚未安装）
softwareupdate --install-rosetta --agree-to-license
```

#### 2. 安装 x86 版本的 Java

```bash
# 使用 Rosetta 2 安装 x86 版本的 JDK
arch -x86_64 brew install temurin@8

# 验证安装
/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/bin/java -version
```

#### 3. 设置环境变量

```bash
# 设置 JAVA_HOME
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/

# 添加到 ~/.zshrc 或 ~/.bash_profile
echo 'export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/' >> ~/.zshrc
source ~/.zshrc
```

#### 4. 在 x86 模式下启动 Shell

```bash
# 启动 x86 模式的 bash
arch -x86_64 /bin/bash
```

#### 5. 创建 x86 Python 环境

```bash
# 在 x86 模式下创建 conda 环境
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86
```

#### 6. 配置国内镜像（可选，加速下载）

```bash
# 配置 pip 镜像
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

#### 7. 安装 MineDojo

```bash
# 安装旧版本的构建工具（MineDojo 依赖）
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"

# 安装 NumPy（必须 < 2.0）
pip install "numpy>=1.21.0,<2.0"

# 安装 MineDojo
pip install minedojo
```

#### 8. 解决 MixinGradle 编译问题

```bash
# 创建 MixinGradle 目录
sudo mkdir -p /opt/MixinGradle
cd /opt/MixinGradle

# 克隆修复版本
sudo git clone https://github.com/verityw/MixinGradle-dcfaf61.git
```

#### 9. 修复 Malmo 编译配置

```bash
# 进入 Minecraft 目录
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft

# 修改 build.gradle（添加镜像和修复依赖）
sed -i '' '/repositories {/a\
        maven { url "file:///opt/hotfix" }
' build.gradle

sed -i '' '4i\
     maven { url "https://maven.aliyun.com/repository/public" }
' build.gradle

sed -i '' '5i\
     maven { url "https://maven.aliyun.com/repository/central" }
' build.gradle

sed -i '' '6i\
     maven { url "https://libraries.minecraft.net/" }
' build.gradle

sed -i '' "s|com.github.SpongePowered:MixinGradle:dcfaf61|MixinGradle-dcfaf61:MixinGradle:dcfaf61|g" build.gradle
sed -i '' "s|brandonhoughton:ForgeGradle|MineDojo:ForgeGradle|g" build.gradle
sed -i '' "s|brandonhoughton:forgegradle|MineDojo:ForgeGradle|g" build.gradle
sed -i '' "s|new File('src/main/resources/schemas.index')|new File(projectDir, 'src/main/resources/schemas.index')|g" build.gradle
```

#### 10. 配置 Gradle 镜像（可选）

```bash
# 配置 Gradle 使用国内镜像
mkdir -p ~/.gradle
cat > ~/.gradle/init.gradle << EOF
allprojects {
    repositories {
        maven { url "https://maven.aliyun.com/repository/public" }
        maven { url "https://maven.aliyun.com/repository/central" }
        maven { url "https://maven.aliyun.com/repository/gradle-plugin" }
        maven { url "https://libraries.minecraft.net/" }
        mavenCentral()
        gradlePluginPortal()
        mavenLocal()
    }
}
EOF
```

#### 11. 编译 Minecraft

```bash
# 编译 Minecraft（可能需要 10-30 分钟）
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
./gradlew shadowJar

# 备份 Gradle 缓存
sudo mkdir -p /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle
sudo cp -r ~/.gradle/caches /opt/MineDojo/minedojo/sim/Malmo/Minecraft/run/gradle
```

#### 12. 修复 LWJGL 问题（如果遇到）

如果遇到 LWJGL 相关错误，需要手动下载并配置：

```bash
# 下载 LWJGL 2.9.3
# 从 https://sourceforge.net/projects/java-game-lib/files/Official%20Releases/LWJGL%202.9.3/
# 下载 lwjgl-2.9.3.zip 并解压到 ~/lwjgl-2.9.3

# 修改 launchClient.sh
# 编辑文件：
# /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/launchClient.sh

# 将启动命令修改为：
java -Djava.library.path=$HOME/lwjgl-2.9.3/native/macosx \
     -Dorg.lwjgl.librarypath=$HOME/lwjgl-2.9.3/native/macosx \
     -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin \
     -Xmx2G -Dfile.encoding=UTF-8 \
     -Duser.country=US -Duser.language=en -Duser.variant \
     -jar ../build/libs/MalmoMod-0.37.0-fat.jar
```

#### 13. 安装项目依赖

```bash
# 返回项目目录
cd /Users/nanzhang/aimc

# 安装其他依赖
pip install -r requirements.txt
```

#### 14. 验证安装

```bash
# 运行验证脚本
python scripts/validate_install.py

# 或使用项目提供的脚本
./scripts/run_minedojo_x86.sh
```

#### Apple M 芯片注意事项

⚠️ **重要提示**：
- 每次运行训练前，都需要在 x86 模式下启动：`arch -x86_64 /bin/bash`
- 使用 `minedojo-x86` 虚拟环境：`conda activate minedojo-x86`
- GPU 加速：M 系列芯片使用 MPS (Metal Performance Shaders)，训练时指定 `--device mps`
- 性能：M1/M2/M3 芯片性能接近或超过中端 GPU

#### 快捷启动脚本

为方便使用，项目提供了 `scripts/run_minedojo_x86.sh` 脚本：

```bash
# 使用脚本启动（自动处理 x86 架构）
./scripts/run_minedojo_x86.sh python scripts/validate_install.py
./scripts/run_minedojo_x86.sh python src/training/train_get_wood.py --use-mineclip
```

---

### GPU 支持配置

#### NVIDIA GPU (CUDA)

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 可用
python -c "import torch; print('GPU可用:', torch.cuda.is_available())"

# 训练时使用 GPU
./scripts/train_get_wood.sh --mineclip --device cuda
```

#### Apple M 系列 (MPS)

```bash
# MPS 已内置在 PyTorch 中
# 验证 MPS 可用
python -c "import torch; print('MPS可用:', torch.backends.mps.is_available())"

# 训练时使用 MPS
./scripts/train_get_wood.sh --mineclip --device mps
```

---

## ⚡ 快速开始

### 1️⃣ 环境准备

```bash
# 激活 Python 环境
conda activate minedojo  # 或 minedojo-x86 (M芯片)

# 验证安装
python -c "import minedojo; print('✓ MineDojo 可用')"

# 设置无头模式（可选，提升 20-40% 性能）
export JAVA_OPTS="-Djava.awt.headless=true"
```

### 2️⃣ 快速测试（5-10 分钟）

```bash
# 运行快速测试，验证环境
./scripts/train_get_wood.sh test --mineclip
```

**预期输出**：
```
========================================
MineDojo 获得木头训练
========================================
任务:       harvest_1_log (获得1个原木)
模式:       test
总步数:     10000
MineCLIP:   --use-mineclip
设备:       mps
========================================

创建环境: harvest_1_log (获得1个原木)
  图像尺寸: (160, 256)
  MineCLIP: 启用
  ✓ 环境创建成功

[100步] ep_rew_mean: 0.05
[1000步] ep_rew_mean: 0.12
...
```

### 3️⃣ 标准训练（2-4 小时）⭐

```bash
# 使用 MineCLIP 训练获得木头任务（推荐）
./scripts/train_get_wood.sh --mineclip

# 训练过程会：
# - 自动保存检查点到 checkpoints/get_wood/
# - 记录日志到 logs/training/
# - 生成 TensorBoard 日志到 logs/tensorboard/
```

### 4️⃣ 监控训练

在另一个终端启动 TensorBoard：

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 浏览器访问: http://localhost:6006
```

**关键指标**：
- 📈 `rollout/ep_rew_mean` - 平均奖励（应该上升）
- 📉 `train/policy_loss` - 策略损失
- 📉 `train/value_loss` - 价值损失
- 📏 `rollout/ep_len_mean` - Episode 长度

### 5️⃣ 评估模型

```bash
# 评估训练好的模型
python scripts/evaluate_get_wood.py

# 评估特定检查点
python scripts/evaluate_get_wood.py --model checkpoints/get_wood/get_wood_100000_steps.zip --episodes 20
```

**预期结果**（200K 步训练后）：
```
========================================
评估结果
========================================
总Episodes: 10
成功次数: 8
成功率: 80.0%

平均奖励: 0.800 ± 0.400
平均步数: 542.3 ± 612.1
成功时平均步数: 267.5 ± 143.2
========================================

性能评级: 良好 ⭐⭐⭐⭐
```

---

## 🔧 常用命令

### 训练相关

```bash
# 快速测试（10K 步，5-10 分钟）
./scripts/train_get_wood.sh test --mineclip

# 标准训练（200K 步，2-4 小时）
./scripts/train_get_wood.sh --mineclip

# 长时间训练（500K 步，5-10 小时）
./scripts/train_get_wood.sh long --mineclip

# 自定义步数
./scripts/train_get_wood.sh --timesteps 300000 --mineclip

# 使用 GPU
./scripts/train_get_wood.sh --mineclip --device cuda

# 使用 MPS (Apple M 芯片)
./scripts/train_get_wood.sh --mineclip --device mps
```

### 监控相关

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 使用 TensorBoard 管理脚本
./scripts/tensorboard_manager.sh start    # 启动
./scripts/tensorboard_manager.sh stop     # 停止
./scripts/tensorboard_manager.sh status   # 查看状态

# 实时查看训练日志
tail -f logs/training/training_*.log

# 查看检查点
ls -lh checkpoints/get_wood/
```

### 其他任务训练

```bash
# 训练采集牛奶任务
./scripts/train_harvest.sh

# 查看可用任务
python -c "import minedojo; print(minedojo.tasks.ALL_PROGRAMMATIC_TASK_IDS[:20])"
```

---

## 🚀 性能优化技巧

### 1. 使用 MineCLIP（最重要）

MineCLIP 提供密集奖励，加速 **3-5 倍**：

```bash
# 始终添加 --mineclip 参数
./scripts/train_get_wood.sh --mineclip
```

**效果对比**：
| 训练方式 | 首次成功 | 达到 80% 成功率 |
|---------|---------|---------------|
| 纯 RL（稀疏奖励） | 100K-200K 步 | 500K-1M 步 |
| MineCLIP（密集奖励） | 20K-50K 步 | 150K-200K 步 |

### 2. 启用无头模式

无头模式可提升 **20-40%** 性能：

```bash
# 方法 1: 环境变量
export JAVA_OPTS="-Djava.awt.headless=true"
./scripts/train_get_wood.sh --mineclip

# 方法 2: 项目脚本已自动启用
./scripts/train_get_wood.sh --mineclip  # 已默认启用无头模式
```

### 3. 使用 GPU 加速

```bash
# NVIDIA GPU
./scripts/train_get_wood.sh --mineclip --device cuda

# Apple M 芯片
./scripts/train_get_wood.sh --mineclip --device mps
```

### 4. 并行环境

```bash
# 使用 4 个并行环境（需要更多内存）
python src/training/train_get_wood.py --use-mineclip --n-envs 4
```

### 性能基准

**M1 MacBook Pro** (8核 CPU, 8GB RAM, MPS):
- 无头模式 + MineCLIP + MPS: ~500 步/分钟
- 200K 步训练: 约 2-3 小时

**RTX 3090** (24GB VRAM):
- 无头模式 + MineCLIP + CUDA: ~1200 步/分钟
- 200K 步训练: 约 1-1.5 小时

---

## 📚 文档导航

### 新手入门

- 🎯 **[GET_STARTED.md](GET_STARTED.md)**: 快速开始指南（最先阅读）
- 📖 **[获得木头训练指南](docs/guides/GET_WOOD_TRAINING_GUIDE.md)**: MVP 任务详细教程

### 加速训练

- 🚀 **[快速开始加速训练](docs/guides/QUICK_START_ACCELERATED_TRAINING.md)**: 1 小时上手
- 🧠 **[MineCLIP 详解](docs/guides/MINECLIP_EXPLAINED.md)**: MineCLIP 工作原理
- 📦 **[MineRL 数据集指南](docs/guides/MINERL_DATASET_GUIDE.md)**: 离线 RL 数据集
- 🎓 **[加速训练完整指南](docs/guides/TRAINING_ACCELERATION_GUIDE.md)**: 所有加速方法
- 💡 **[高级训练解决方案](docs/guides/ADVANCED_TRAINING_SOLUTIONS.md)**: 进阶技巧

### 任务和监控

- 📋 **[任务快速开始](docs/guides/TASKS_QUICK_START.md)**: MineDojo 任务系统
- 📊 **[TensorBoard 中文指南](docs/guides/TENSORBOARD_中文指南.md)**: 可视化训练

### 参考文档

- 📑 **[MineDojo 任务参考](docs/technical/MINEDOJO_TASKS_REFERENCE.md)**: 所有可用任务
- 📝 **[训练总结](docs/summaries/TRAINING_HARVEST_PAPER.md)**: harvest_paper 任务经验
- ❓ **[常见问题 FAQ](docs/FAQ.md)**: 15+ 个常见问题解答

---

## ❓ FAQ（常见问题）

### Q1: MineCLIP 是什么？

**A**: MineCLIP 是一个视觉-语言多模态模型，在 73 万 YouTube Minecraft 视频上训练，可以：
- 提供密集奖励信号（将稀疏奖励转换为密集奖励）
- 加速训练 3-5 倍
- 完全离线运行（首次使用会下载模型到本地）

详见：[MineCLIP 详解](docs/guides/MINECLIP_EXPLAINED.md)

### Q2: 为什么训练这么慢？

**A**: 优化建议：
1. ✅ 使用 MineCLIP：`--mineclip` 参数
2. ✅ 启用无头模式：`export JAVA_OPTS="-Djava.awt.headless=true"`
3. ✅ 使用 GPU：`--device cuda` 或 `--device mps`
4. ✅ 减少图像尺寸：`--image-size 120 160`

### Q3: Apple M 芯片如何部署？

**A**: 需要通过 Rosetta 2 运行 x86 版本的 MineDojo，详细步骤见上文"Apple M 芯片部署"章节。

关键步骤：
1. 安装 x86 版本的 Java：`arch -x86_64 brew install temurin@8`
2. 在 x86 模式下启动：`arch -x86_64 /bin/bash`
3. 创建 x86 Python 环境：`conda create -n minedojo-x86 python=3.9`
4. 编译 Minecraft（需要修复多个配置）

### Q4: MineCLIP 需要联网吗？

**A**: 仅首次使用时需要下载模型（~250-350MB），之后完全离线运行。模型保存在 `~/.minedojo/models/`。

### Q5: 如何查看训练进度？

**A**: 
```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 浏览器访问 http://localhost:6006
# 查看关键指标：
# - rollout/ep_rew_mean（平均奖励）
# - train/policy_loss（策略损失）
# - rollout/success_rate（成功率，如果有）
```

### Q6: 模型不学习怎么办？

**A**: 检查清单：
1. ✅ 确认 MineCLIP 已启用（日志中应显示 "MineCLIP: 启用"）
2. ✅ 检查 TensorBoard 中 `ep_rew_mean` 是否上升
3. ✅ 尝试增加探索：`--ent-coef 0.02`
4. ✅ 训练更长时间（至少 100K 步）

### Q7: 环境创建失败？

**A**: 
```bash
# 1. 检查 Java
java -version  # 需要 Java 8+

# 2. 设置 JAVA_HOME
export JAVA_HOME=/path/to/java

# 3. 设置无头模式
export JAVA_OPTS="-Djava.awt.headless=true"

# 4. 重新安装 MineDojo
pip install --upgrade minedojo
```

### Q8: 内存不足？

**A**: 
```bash
# 1. 减少并行环境
--n-envs 1

# 2. 减少批次大小
--batch-size 32

# 3. 使用更小的图像
--image-size 120 160
```

### Q9: 如何训练其他任务？

**A**: 
```bash
# 修改训练脚本中的 task_id
# 可用任务列表：
python -c "import minedojo; print(minedojo.tasks.ALL_PROGRAMMATIC_TASK_IDS[:20])"

# 常用任务：
# - harvest_1_log（获得木头）
# - harvest_1_milk（获得牛奶）
# - harvest_8_log（获得 8 个木头）
# - harvest_1_wheat（获得小麦）
```

### Q10: 在哪里获取更多帮助？

**A**: 
- 📖 完整文档：`docs/guides/`
- ❓ 详细 FAQ：`docs/FAQ.md`
- 🔧 诊断工具：`python scripts/validate_install.py`
- 📊 任务参考：`docs/technical/MINEDOJO_TASKS_REFERENCE.md`

---

## 📊 预期训练时间线

### 使用 MineCLIP（推荐）

| 步数 | 时间 | 里程碑 |
|------|------|--------|
| 10K | 5-10分钟 | 测试完成，验证环境 |
| 20-50K | 20-40分钟 | 首次成功获得木头 |
| 100K | 1-2小时 | 成功率约 50% |
| 200K | 2-4小时 | 成功率约 80%，可以使用 |
| 500K | 5-10小时 | 成功率约 90%，性能优秀 |

### 不使用 MineCLIP

| 步数 | 时间 | 里程碑 |
|------|------|--------|
| 100K | 1-3小时 | 可能还未成功 |
| 200K | 3-6小时 | 首次成功 |
| 500K | 8-16小时 | 成功率约 60% |
| 1M+ | 16+小时 | 成功率约 70-80% |

**结论**：MineCLIP 加速约 **3-5 倍**！

---

## 🎉 成功标志

当你看到以下情况，说明训练成功：

1. ✅ **评估成功率 ≥ 80%**
2. ✅ **平均成功步数 < 500 步**
3. ✅ **TensorBoard 中 ep_rew_mean 稳定上升**
4. ✅ **模型能在测试中多次成功获得木头**

---

## 🤝 贡献

欢迎贡献代码、文档或提出建议！

### 开发规范

- **Python 代码**: 遵循 PEP 8，小写下划线命名
- **类名**: 驼峰命名 `class MyAgent`
- **文档**: 大写下划线命名 `TECHNICAL_GUIDE.md`
- **Git 提交**: `[类型] 简短描述`（feat/fix/docs/refactor/test/chore）

### 提交流程

```bash
# 1. Fork 项目
# 2. 创建分支
git checkout -b feature/my-feature

# 3. 提交代码
git add .
git commit -m "[feat] 添加新功能"

# 4. 推送并创建 Pull Request
git push origin feature/my-feature
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [MineDojo](https://github.com/MineDojo/MineDojo) - 提供 Minecraft 强化学习环境
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习算法库
- [Project Malmo](https://github.com/microsoft/malmo) - Minecraft AI 平台

---

## 📞 联系方式

- 📧 Email: your-email@example.com
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/aimc/issues)

---

**立即开始**：
```bash
# 1. 激活环境
conda activate minedojo  # 或 minedojo-x86

# 2. 快速测试
./scripts/train_get_wood.sh test --mineclip

# 3. 开始训练
./scripts/train_get_wood.sh --mineclip
```

祝训练成功！🚀
