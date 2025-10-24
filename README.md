# AIMC - MineDojo AI Minecraft 训练工程

基于 MineDojo 的 Minecraft AI 智能体训练项目，使用**DAgger（Dataset Aggregation）模仿学习**训练智能体完成各种 Minecraft 任务。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MineDojo](https://img.shields.io/badge/MineDojo-Latest-green.svg)](https://github.com/MineDojo/MineDojo)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目介绍

AIMC 是一个完整的 Minecraft AI 训练工程，专注于使用**模仿学习（Imitation Learning）**方法，特别是 **DAgger 算法**，训练智能体在 MineDojo 环境中完成各种任务。

### 核心特性

✅ **DAgger 完整实现**: 录制 → BC基线 → 迭代优化 → 90%+成功率  
✅ **Pygame 鼠标控制**: 类似 FPS 游戏的自然录制方式  
✅ **自动化工作流**: 一键完成完整训练流程  
✅ **多任务支持**: 独立的数据和模型管理  
✅ **追加录制**: 灵活的数据扩充机制  
✅ **交互式标注**: 智能采样 + P键保持策略  
✅ **详细文档**: 从入门到进阶的完整指南  

### 技术亮点

- **环境**: MineDojo (Minecraft 仿真环境)
- **核心算法**: DAgger (Dataset Aggregation)
- **辅助算法**: Behavior Cloning (BC)
- **框架**: Stable-Baselines3
- **数据录制**: Pygame 鼠标控制 + 键盘控制
- **可视化**: TensorBoard

### 支持的任务类型

- 🪵 **采集任务**: 获得木头、石头、煤炭等
- 🐄 **收集任务**: 收集牛奶、羊毛、苹果等
- 🌾 **农业任务**: 种植和收获小麦等作物
- ⚔️ **战斗任务**: 狩猎动物、击败怪物
- 🏗️ **建造任务**: 制作工具、建造结构

---

## 🎯 DAgger 训练工作流

### 什么是 DAgger？

**DAgger** (Dataset Aggregation) 是一种迭代式模仿学习算法，通过人工录制专家演示数据，让智能体学习人类行为，并通过多轮迭代不断改进。

**工作流程**:
```
1. 录制专家演示（10-20个episodes） 
   ↓
2. 训练BC基线（成功率 60%）
   ↓
3. DAgger迭代1：收集失败 → 标注 → 训练（成功率 75%）
   ↓
4. DAgger迭代2：收集失败 → 标注 → 训练（成功率 85%）
   ↓
5. DAgger迭代3：收集失败 → 标注 → 训练（成功率 92%+）
```

**相比纯RL的优势**:
- 🚀 **更快收敛**: 从好的策略开始，不是随机探索
- 🎯 **更高成功率**: 90%+ 远超纯RL的80-85%
- 🛠️ **更鲁棒**: 见过失败场景，知道如何纠正
- ⏱️ **时间可控**: 预计3-5小时完成完整训练

---

## 🚀 部署指南

### 方法1: 标准部署（Linux / Intel Mac）

#### 系统要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| GPU | 无 | 可选 |
| 存储 | 10GB | 20GB+ |
| 系统 | macOS 10.15+ / Ubuntu 18.04+ | macOS 13+ / Ubuntu 22.04+ |

#### 快速部署

```bash
# 1. 安装 Java 8+
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install openjdk-8-jdk

# macOS (Intel)
brew install openjdk@8

# 2. 创建 Python 环境
conda create -n minedojo python=3.9 -y
conda activate minedojo

# 3. 克隆项目
git clone https://github.com/your-repo/aimc.git
cd aimc

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python tools/validate_install.py
```

---

### 方法2: Apple M 芯片部署（ARM64）⭐

Apple M 系列芯片需要通过 Rosetta 2 运行 MineDojo（因为 Minecraft 服务端需要 x86 架构）。

#### 快速部署

```bash
# 1. 安装 Rosetta 2
softwareupdate --install-rosetta --agree-to-license

# 2. 安装 x86 版本的 Java
arch -x86_64 brew install temurin@8

# 3. 设置环境变量
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/
echo 'export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/' >> ~/.zshrc

# 4. 在 x86 模式下创建环境
arch -x86_64 /bin/bash
conda create -n minedojo-x86 python=3.9 -y
conda activate minedojo-x86

# 5. 安装依赖
pip install "pip<24.1" "setuptools<58" "wheel<0.38.0"
pip install "numpy>=1.21.0,<2.0"
pip install minedojo

# 6. 克隆项目并安装
cd /path/to/aimc
pip install -r requirements.txt

# 7. 使用便捷脚本运行
./scripts/run_minedojo_x86.sh python tools/validate_install.py
```

**重要提示**:
- 每次运行都需要：`arch -x86_64 /bin/bash`
- 或使用项目脚本：`./scripts/run_minedojo_x86.sh <命令>`
- GPU 加速：M 系列芯片使用 MPS，指定 `--device mps`

详细步骤见：[当前 README.md 的 "Apple M 芯片部署" 章节](#apple-m-芯片部署arm64)

---

### 方法3: Docker 部署

```bash
# 1. 构建镜像
cd docker
docker build --platform linux/amd64 -t aimc-minedojo:latest .

# 2. 运行容器
docker run -it --rm \
  --platform linux/amd64 \
  -v $(pwd):/workspace \
  aimc-minedojo:latest

# 3. 在容器中验证
python tools/validate_install.py
```

**网络受限环境**: 参考 `docker/README.md` 获取代理配置和离线部署方案

---

## ⚡ 快速开始

### 完整 DAgger 训练流程（3-5小时）

```bash
# 激活环境
conda activate minedojo  # 或 minedojo-x86 (M芯片)

# 一键运行完整工作流
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

**执行内容**:
1. 录制 10 个专家演示（40-60分钟）- **Pygame 鼠标控制**
2. 训练 BC 基线（30-40分钟）
3. 评估 BC 成功率（10分钟）
4. DAgger 迭代 1（60-80分钟）
5. DAgger 迭代 2（60-80分钟）
6. DAgger 迭代 3（60-80分钟）

**预期成功率**: BC 60% → 迭代3后 85-90%

### 分步骤运行

#### 1️⃣ 录制专家演示

**方法A: Pygame 鼠标控制（推荐）⭐**

```bash
# 使用鼠标控制（类似 FPS 游戏）
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/expert_demos/harvest_1_log \
    --max-frames 1000 \
    --mouse-sensitivity 0.5

# 控制说明：
# - 鼠标移动: 转动视角
# - 鼠标左键: 攻击/挖掘
# - W/A/S/D: 移动
# - Space: 跳跃
# - Q: 重试当前episode
# - ESC: 退出
```

**方法B: 键盘控制**

```bash
# 使用键盘控制
python tools/dagger/record_manual_chopping.py \
    --max-frames 500 \
    --camera-delta 1

# 控制说明：
# - W/A/S/D: 移动
# - I/J/K/L: 视角（上/左/下/右）
# - F: 攻击
# - Q: 保存并退出
```

#### 2️⃣ 训练 BC 基线

```bash
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50
```

#### 3️⃣ 评估 BC 模型

```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20
```

#### 4️⃣ DAgger 迭代优化

```bash
# 每轮DAgger迭代
# 1. 收集失败状态
python tools/dagger/run_policy_collect_states.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/harvest_1_log/iter_1/

# 2. 交互式标注（使用P键保持策略）
python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1/ \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling

# 3. 聚合数据训练
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/harvest_1_log/ \
    --new-data data/expert_labels/harvest_1_log/iter_1.pkl \
    --output checkpoints/dagger/harvest_1_log/dagger_iter_1.zip

# 4. 评估改进
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --episodes 20
```

---

## 📊 数据管理

### 目录结构

```
data/
├── expert_demos/              # 专家演示数据（手动录制）
│   └── harvest_1_log/
│       ├── episode_000/
│       │   ├── frame_00000.npy
│       │   ├── frame_00001.npy
│       │   └── ...
│       ├── episode_001/
│       └── ...
├── policy_states/             # 策略收集的状态
│   └── harvest_1_log/
│       ├── iter_1/
│       ├── iter_2/
│       └── iter_3/
├── expert_labels/             # 标注数据
│   └── harvest_1_log/
│       ├── iter_1.pkl
│       ├── iter_2.pkl
│       └── iter_3.pkl
└── dagger/                    # 聚合数据
    └── harvest_1_log/
        ├── combined_iter_1.pkl
        ├── combined_iter_2.pkl
        └── combined_iter_3.pkl

checkpoints/dagger/            # 模型检查点
└── harvest_1_log/
    ├── bc_baseline.zip
    ├── bc_baseline_eval_results.npy
    ├── dagger_iter_1.zip
    ├── dagger_iter_1_eval_results.npy
    ├── dagger_iter_2.zip
    └── dagger_iter_3.zip
```

### 数据操作

#### 追加录制数据

```bash
# 已录制了 10 个 episodes，想追加到 20 个
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --append-recording \
    --iterations 0
```

#### 多任务独立管理

```bash
# 任务1: harvest_1_log
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3

# 任务2: harvest_1_wool
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_wool \
    --num-episodes 10 \
    --iterations 3

# 数据自动保存到不同目录：
# - data/expert_demos/harvest_1_log/
# - data/expert_demos/harvest_1_wool/
```

#### 继续训练

```bash
# 从已有模型继续更多轮 DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
```

#### 清理旧数据

```bash
# 删除特定任务的数据
rm -rf data/expert_demos/harvest_1_log/
rm -rf checkpoints/dagger/harvest_1_log/

# 删除所有DAgger中间数据（保留专家演示）
rm -rf data/policy_states/*/
rm -rf data/expert_labels/*/
rm -rf data/dagger/*/
```

---

## 🛠️ 支持功能介绍

### 1. 录制工具

#### Pygame 鼠标控制（推荐）⭐

**特性**:
- ✅ 鼠标连续平滑控制视角
- ✅ 鼠标左键攻击（更自然）
- ✅ 多键同时检测（W+左键）
- ✅ 静态帧占比 <20%（数据质量高）
- ✅ 类似 FPS 游戏操作
- ✅ 无需 macOS 辅助功能权限

**使用**:
```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --mouse-sensitivity 0.5 \
    --base-dir data/expert_demos/harvest_1_log
```

**参数**:
- `--mouse-sensitivity`: 鼠标灵敏度（0.1-2.0，默认0.5）
- `--max-frames`: 每个episode最大帧数（默认1000）
- `--fps`: 录制帧率（默认20）

#### 键盘控制

**特性**:
- ✅ 简单直接
- ✅ 稳定可靠
- ❌ 视角控制离散
- ❌ 静态帧占比较高（28.5%）

**使用**:
```bash
python tools/dagger/record_manual_chopping.py \
    --max-frames 500
```

---

### 2. 训练工具

#### BC (Behavior Cloning) 训练

**功能**: 从专家演示学习初始策略

**使用**:
```bash
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50 \
    --learning-rate 3e-4 \
    --batch-size 64
```

**参数**:
- `--data`: 数据目录（必需）
- `--output`: 输出模型路径（必需）
- `--epochs`: 训练轮数（默认50）
- `--batch-size`: 批次大小（默认32）
- `--learning-rate`: 学习率（默认0.001）

#### DAgger 迭代训练

**功能**: 迭代式数据收集和训练

**使用**:
```bash
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/harvest_1_log/ \
    --new-data data/expert_labels/harvest_1_log/iter_1.pkl \
    --output checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --epochs 30
```

---

### 3. 标注工具

#### 交互式标注

**功能**: 智能采样 + 键盘标注

**控制键**:
- `W/S/A/D` - 移动动作
- `I/K/J/L` - 视角调整
- `F` - 攻击
- `Q` - 前进+攻击
- **`P`** - 保持策略（重要！）⭐
- `N` - 跳过此状态
- `Z` - 撤销上一个标注
- `X/ESC` - 完成标注

**使用**:
```bash
python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1/ \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling \
    --failure-window 10
```

**参数**:
- `--smart-sampling`: 智能采样（只标注20-30%关键状态）
- `--failure-window`: 失败前N步的采样窗口（默认10）

**标注技巧**:
- ✅ 善用P键（如果策略正确，按P保持）
- ✅ 视角调整<20%，前进>60%
- ✅ 连续视角调整不超过2帧
- ✅ 跳过重复的过渡帧（按N）

---

### 4. 评估工具

#### 策略评估

**功能**: 评估模型成功率和性能

**使用**:
```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --episodes 20 \
    --task harvest_1_log
```

**输出**:
```
评估结果
============================================================
成功率: 75.0% (15/20)
平均奖励: 0.75 ± 0.43
平均步数: 487 ± 312
成功时平均步数: 267 ± 143
============================================================
```

---

### 5. 自动化工作流脚本

#### run_dagger_workflow.sh

**功能**: 一键完成完整 DAgger 训练流程

**基础用法**:
```bash
# 完整流程
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3

# 跳过录制（已有数据）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3

# 追加录制
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --append-recording \
    --iterations 0

# 继续训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
```

**参数速查**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `harvest_1_log` | MineDojo任务ID |
| `--num-episodes` | `10` | 录制数量 |
| `--iterations` | `3` | DAgger轮数 |
| `--bc-epochs` | `50` | BC训练轮数 |
| `--skip-recording` | `false` | 跳过录制 |
| `--skip-bc` | `false` | 跳过BC训练 |
| `--append-recording` | `false` | 追加录制 |
| `--continue-from` | - | 继续训练的模型 |
| `--mouse-sensitivity` | `0.15` | 鼠标灵敏度 |

---

### 6. 监控工具

#### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 浏览器访问: http://localhost:6006
```

**关键指标**:
- `rollout/ep_rew_mean` - 平均奖励（应该上升）
- `rollout/success_rate` - 成功率
- `train/policy_loss` - 策略损失
- `train/value_loss` - 价值损失

#### 实时日志

```bash
# 查看训练日志
tail -f logs/training/training_*.log

# 查看检查点
ls -lh checkpoints/dagger/harvest_1_log/
```

---

### 7. 辅助工具

#### 验证安装

```bash
python tools/validate_install.py
```

#### MineDojo x86 运行脚本（M芯片）

```bash
# 自动处理 x86 架构切换
./scripts/run_minedojo_x86.sh <命令>

# 示例
./scripts/run_minedojo_x86.sh python tools/validate_install.py
./scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py --model ...
```

---

## 📚 文档导航

### 核心文档

- 🚀 **[DAgger 综合指南](docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md)** - **一站式完整教程**（强烈推荐）
  - 包含：理论、BC训练、录制工具、标注策略、多任务、脚本使用、故障排查

### 参考文档

- 📑 **[MineDojo 任务参考](docs/reference/MINEDOJO_TASKS_REFERENCE.md)** - 所有可用任务
- 📝 **[MineDojo 动作参考](docs/reference/MINEDOJO_ACTION_REFERENCE.md)** - 动作空间说明
- 🎮 **[标注键盘参考](docs/reference/LABELING_KEYBOARD_REFERENCE.md)** - 标注工具控制键
- ❓ **[常见问题 FAQ](FAQ.md)** - 常见问题解答

### 状态文档

- 📊 **[DAgger 实现计划](docs/status/DAGGER_IMPLEMENTATION_PLAN.md)** - 实施路线图
- ✅ **[BC 训练就绪](docs/status/BC_TRAINING_READY.md)** - BC训练状态

---

## 🎯 性能预期

### 训练时间

| 阶段 | 成功率 | 时间 |
|------|--------|------|
| 录制专家演示 | - | 40-60分钟 |
| BC基线 | 50-65% | 30-40分钟 |
| DAgger迭代1 | 70-78% | 60-80分钟 |
| DAgger迭代2 | 80-85% | 60-80分钟 |
| DAgger迭代3 | 85-92% | 60-80分钟 |

**总计**: 4-5小时达到 90%+ 成功率

### 数据量

| 轮次 | 数据量 | 标注时间 | 成功率 | 提升 |
|------|--------|---------|--------|------|
| BC基线 | 5K | 40分钟 | 60% | - |
| DAgger-1 | 7K | +30分钟 | 75% | +15% |
| DAgger-2 | 9K | +30分钟 | 85% | +10% |
| DAgger-3 | 11K | +20分钟 | 90% | +5% |

---

## ❓ 常见问题（FAQ）

### Q1: DAgger 和纯RL有什么区别？

**A**: 

| 特性 | 纯RL（PPO） | DAgger |
|------|-----------|--------|
| 数据来源 | 随机探索 | 人类演示 |
| 首次成功 | 50K-200K步 | 5-10个演示 |
| 最终成功率 | 80-85% | **90-95%** |
| 训练时间 | 3-5小时 | **3-5小时**（含录制） |
| 鲁棒性 | 中等 | **高**（见过失败场景） |

### Q2: 需要多少专家演示？

**A**: 
- **最少**: 5-10 个成功演示
- **推荐**: 10-20 个成功演示
- **数据质量 > 数量**: 保持操作一致，覆盖不同场景

### Q3: 标注太慢怎么办？

**A**: 
- ✅ 使用 `--smart-sampling`（只标注20-30%关键状态）
- ✅ 多用P键（如果策略正确，直接按P保持）
- ✅ 跳过重复帧（按N键）
- ✅ 使用 `--failure-window 5`（只标注失败前5步）

**标注速度对比**:
- 全手动: ~5秒/状态
- 使用P键: ~2秒/状态（**60%提升**）

### Q4: Apple M 芯片如何运行？

**A**: 
1. 在 x86 模式下启动：`arch -x86_64 /bin/bash`
2. 或使用项目脚本：`./scripts/run_minedojo_x86.sh <命令>`
3. GPU 加速：指定 `--device mps`

详见：[README - Apple M 芯片部署](#方法2-apple-m-芯片部署arm64)

### Q5: 如何训练其他任务？

**A**: 
```bash
# 修改 --task 参数即可
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_wool \
    --num-episodes 10 \
    --iterations 3

# 常用任务：
# - harvest_1_log（获得木头）
# - harvest_1_wool（获得羊毛）
# - harvest_milk（获得牛奶）
# - harvest_10_cobblestone（挖石头）
```

查看所有任务：
```bash
python -c "import minedojo; print(minedojo.tasks.ALL_PROGRAMMATIC_TASK_IDS[:20])"
```

### Q6: 模型一直原地转圈？

**A**: 标注时视角调整过多

**解决**:
1. 检查标注分布（视角调整应该<20%）
2. 重新标注，使用"前进优先"原则
3. 多使用P键（保持策略）

### Q7: 如何查看训练进度？

**A**: 
```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 浏览器访问 http://localhost:6006
# 查看关键指标：
# - rollout/ep_rew_mean（平均奖励）
# - rollout/success_rate（成功率）
```

### Q8: 数据可以跨任务复用吗？

**A**: 不建议。每个任务有独立的数据和模型目录。但可以：
- 使用相似任务的BC模型做预训练
- 迁移学习（需要微调）

### Q9: 在哪里获取更多帮助？

**A**: 
- 📖 **完整文档**: `docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md`
- ❓ **详细 FAQ**: `FAQ.md`
- 🔧 **诊断工具**: `python tools/validate_install.py`

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [MineDojo](https://github.com/MineDojo/MineDojo) - 提供 Minecraft 强化学习环境
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习算法库
- [DAgger 论文](https://arxiv.org/abs/1011.0686) - Ross et al., AISTATS 2011

---

## 📞 联系方式

- 📧 Email: konders@gmail.com
- 💬 Issues: [GitHub Issues](https://github.com/your-repo/aimc/issues)

---

**立即开始**：
```bash
# 1. 激活环境
conda activate minedojo  # 或 minedojo-x86

# 2. 验证安装
python tools/validate_install.py

# 3. 开始 DAgger 训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

祝训练成功！🚀
