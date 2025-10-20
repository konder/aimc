# 快速开始：加速训练指南

本指南将帮助你在**1小时内**开始使用加速训练方法，避免从零开始的漫长训练过程。

---

## 🎯 目标

训练一系列MineDojo技能（砍树、采矿、狩猎等），然后通过agent组合这些技能完成复杂任务。

**问题**：从零开始强化学习太慢（可能需要数天甚至数周）

**解决方案**：使用加速训练方法，预期可获得**3-10倍**的训练加速

---

## 📋 前置准备

### 1. 确认环境已安装

```bash
# 激活MineDojo环境
conda activate minedojo

# 验证安装
python scripts/validate_install.py
```

### 2. 添加脚本执行权限

```bash
chmod +x scripts/train_with_mineclip.sh
chmod +x scripts/train_curriculum.sh
chmod +x scripts/manage_skill_library.sh
```

---

## 🚀 方法一：MineCLIP加速训练（推荐首选）

### 什么是MineCLIP？

MineCLIP是MineDojo提供的预训练模型，可以将任务描述和游戏画面关联起来，提供**密集的语义奖励**。

**优点**：
- ✅ 最简单（一条命令即可）
- ✅ 最快速（3-5倍加速）
- ✅ 无需人工标注数据
- ✅ 适用于所有MineDojo任务

### 快速开始

#### Step 1: 训练第一个技能（砍树）

```bash
# 使用MineCLIP加速训练砍树技能
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000
```

**预期时间**：
- CPU: 2-4小时
- GPU: 1-2小时

#### Step 2: 查看训练进度

```bash
# 在另一个终端启动TensorBoard
tensorboard --logdir logs/tensorboard
```

然后在浏览器打开 http://localhost:6006

**关键指标**：
- `rollout/ep_rew_mean`: 平均奖励（应该逐渐增加）
- `rollout/ep_len_mean`: 平均episode长度
- `train/policy_loss`: 策略损失

#### Step 3: 训练更多技能

```bash
# 采矿技能
./scripts/train_with_mineclip.sh --task mine_stone --timesteps 200000

# 收集羊毛
./scripts/train_with_mineclip.sh --task harvest_wool --timesteps 150000

# 获取牛奶
./scripts/train_with_mineclip.sh --task harvest_milk --timesteps 150000
```

#### Step 4: 评估训练好的模型

```python
# evaluate_skill.py
from stable_baselines3 import PPO
import minedojo

# 加载模型
model = PPO.load("checkpoints/mineclip/harvest_log_mineclip_final.zip")

# 创建环境
env = minedojo.make("harvest_log", image_size=(160, 256))

# 运行测试
for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # 可选：渲染画面
        # env.render()
    
    print(f"Episode {episode + 1}: Reward = {total_reward}")

env.close()
```

---

## 🎓 方法二：课程学习（更系统化）

### 什么是课程学习？

从简单任务开始，逐步增加难度，就像人类学习一样。

**优点**：
- ✅ 更稳定的训练
- ✅ 更好的最终性能
- ✅ 适合复杂技能

**缺点**：
- ❌ 需要更多时间（但总时间仍比从零开始少）

### 快速开始

#### Step 1: 训练一个完整的课程

```bash
# 砍树技能（4个级别）
./scripts/train_curriculum.sh --skill chop_tree
```

**课程结构**：
1. **Level 1** (50K步): 近距离，有斧头
2. **Level 2** (100K步): 中距离，有斧头
3. **Level 3** (100K步): 远距离，有斧头
4. **Level 4** (250K步): 完整任务，无斧头

**总时间**：约4-8小时

#### Step 2: 训练其他技能

```bash
# 采矿技能
./scripts/train_curriculum.sh --skill mine_stone

# 狩猎技能
./scripts/train_curriculum.sh --skill hunt_animal
```

#### Step 3: 查看训练进度

课程学习会为每个级别创建检查点：

```
checkpoints/curriculum/chop_tree/
├── level1_final.zip
├── level2_final.zip
├── level3_final.zip
├── level4_final.zip
└── chop_tree_final.zip  ← 最终技能
```

---

## 📚 方法三：组合技能（分层强化学习）

训练好多个基础技能后，可以组合它们完成复杂任务。

### Step 1: 创建技能库

```bash
# 添加已训练的技能到库
./scripts/manage_skill_library.sh add chop_tree \
    checkpoints/curriculum/chop_tree/chop_tree_final.zip \
    "Chop down trees and collect wood"

./scripts/manage_skill_library.sh add mine_stone \
    checkpoints/curriculum/mine_stone/mine_stone_final.zip \
    "Mine stone blocks with pickaxe"

# 查看技能库
./scripts/manage_skill_library.sh list
```

### Step 2: 使用技能库

```python
# use_skill_library.py
from src.training.skill_library import SkillLibrary
import minedojo

# 加载技能库
library = SkillLibrary("checkpoints/skill_library.json")
library.info()

# 创建环境
env = minedojo.make("make_wooden_pickaxe", image_size=(160, 256))

# 使用技能完成任务
obs = env.reset()

# 1. 先使用"砍树"技能
chop_skill = library.get_skill("chop_tree")
chop_skill.load()

for _ in range(200):  # 执行200步砍树
    action, _ = chop_skill.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

# 2. 然后可以切换到其他技能...
# mine_skill = library.get_skill("mine_stone")
# ...

env.close()
```

### Step 3: 训练高级策略

使用技能库训练一个高级策略，学习何时使用哪个技能：

```python
from src.training.skill_library import SkillLibrary, HierarchicalAgent

# 加载技能库
library = SkillLibrary("checkpoints/skill_library.json")

# 创建分层智能体
agent = HierarchicalAgent(
    skill_library=library,
    skill_duration=100,  # 每个技能最多执行100步
    auto_switch=True
)

# 在复杂任务上训练
env = minedojo.make("make_wooden_pickaxe", image_size=(160, 256))

# 训练高级策略选择技能...
# (需要实现select_skill方法)
```

---

## 📊 方法对比

| 方法 | 训练时间 | 实施难度 | 最终性能 | 推荐场景 |
|------|----------|----------|----------|----------|
| **MineCLIP** | ⭐⭐⭐⭐⭐ 最快 | ⭐ 简单 | ⭐⭐⭐⭐ 好 | 快速原型，单个技能 |
| **课程学习** | ⭐⭐⭐⭐ 快 | ⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 很好 | 复杂技能，需要高质量 |
| **分层RL** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 难 | ⭐⭐⭐⭐⭐ 很好 | 复杂组合任务 |

---

## 💡 推荐路线（2-3周完成）

### 第1周：快速验证

使用MineCLIP训练3-5个基础技能：

```bash
# Day 1-2: 砍树
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000

# Day 3-4: 采矿
./scripts/train_with_mineclip.sh --task mine_stone --timesteps 200000

# Day 5-7: 其他技能
./scripts/train_with_mineclip.sh --task harvest_wool --timesteps 150000
./scripts/train_with_mineclip.sh --task combat_spider --timesteps 200000
```

**目标**：验证训练流程，获得可用的基础技能

### 第2周：优化提升

使用课程学习重新训练关键技能，获得更好的性能：

```bash
# 重点技能用课程学习
./scripts/train_curriculum.sh --skill chop_tree
./scripts/train_curriculum.sh --skill mine_stone
```

**目标**：获得高质量的核心技能

### 第3周：组合评估

1. 创建技能库
2. 实现技能组合逻辑
3. 在复杂任务上评估

**目标**：验证技能可以组合完成复杂任务

---

## 🐛 常见问题

### Q1: MineCLIP不可用怎么办？

**A**: MineCLIP需要MineDojo 0.1+版本。如果不可用：

```bash
# 升级MineDojo
pip install --upgrade minedojo

# 或者降级使用基础奖励塑形
python src/training/train_harvest_paper.py --task harvest_log --ent-coef 0.02
```

### Q2: 训练太慢怎么办？

**A**: 尝试以下优化：

```bash
# 1. 减少图像尺寸
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000
# 修改脚本添加: --image-size 120 160

# 2. 使用多个并行环境（需要更多内存）
./scripts/train_with_mineclip.sh --task harvest_log --n-envs 4

# 3. 使用GPU
./scripts/train_with_mineclip.sh --task harvest_log --device cuda
```

### Q3: 内存不足怎么办？

**A**: 减少资源使用：

```bash
# 使用单个环境
./scripts/train_with_mineclip.sh --task harvest_log --n-envs 1

# 减少批次大小（修改脚本）
# 添加: --batch-size 32
```

### Q4: 如何判断模型训练好了？

**A**: 查看以下指标：

1. **TensorBoard**:
   - `rollout/ep_rew_mean` > 0.5（持续获得奖励）
   - 曲线趋于平稳（不再提升）

2. **评估测试**:
   - 运行10个episode
   - 成功率 > 60%

3. **训练时间**:
   - MineCLIP: 至少15-20万步
   - 课程学习: 完成所有级别

### Q5: 多个技能如何组合？

**A**: 三种方式：

1. **简单顺序执行**（最简单）:
```python
# 先砍树，再采矿
chop_model.predict(obs)  # 执行200步
mine_model.predict(obs)  # 执行200步
```

2. **基于规则切换**（中等）:
```python
# 根据物品栏内容决定
if "log" not in inventory:
    use_skill("chop_tree")
elif "stone" not in inventory:
    use_skill("mine_stone")
```

3. **学习高级策略**（最复杂）:
```python
# 训练一个元策略选择技能
meta_policy.predict(obs) → select skill
selected_skill.predict(obs) → execute action
```

---

## 🎯 下一步

完成快速开始后，你可以：

1. **阅读完整指南**:
   - `docs/guides/TRAINING_ACCELERATION_GUIDE.md` - 所有加速方法详解

2. **自定义课程**:
   - 编辑 `src/training/curriculum_trainer.py`
   - 添加自己的技能课程

3. **实现高级策略**:
   - 训练元策略组合技能
   - 使用强化学习学习技能选择

4. **尝试其他方法**:
   - 行为克隆（收集人类演示）
   - 奖励塑形（自定义奖励函数）
   - VPT预训练模型

---

## 📞 需要帮助？

- 查看完整文档: `docs/guides/`
- 查看示例代码: `src/training/`
- 运行诊断工具: `python scripts/diagnose_minedojo.py`

---

## 🎉 总结

- ✅ **MineCLIP**: 最快最简单，立即开始
- ✅ **课程学习**: 更好的性能，值得等待
- ✅ **技能组合**: 完成复杂任务的关键
- ✅ **预期加速**: 3-10倍，从数周缩短到数天

**立即开始**:
```bash
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000
```

祝训练成功！🚀

