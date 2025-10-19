# MineDojo 任务系统指南

## 概述

MineDojo 的内置任务系统提供了预设的环境配置和奖励函数，但**不包含预训练模型**。本文档详细说明了内置任务的机制以及如何从头开始训练。

---

## 1. 内置任务提供什么？

### ✅ 提供的内容

| 组件 | 说明 | 示例 |
|------|------|------|
| **环境配置** | 初始世界状态、生成规则 | 平原、有甘蔗的地形 |
| **奖励函数** | 定义何时给予奖励 | 收集1个纸 → +1.0奖励 |
| **任务描述** | 自然语言的目标 | "Obtain paper" |
| **完成条件** | 判断任务是否完成 | 物品栏中有纸 |

### ❌ 不提供的内容

- **预训练模型/权重**: 没有可以直接加载的checkpoint
- **训练算法**: 需要自己实现PPO、DQN等
- **训练脚本**: 需要自己编写训练循环
- **超参数配置**: 需要自己调优

---

## 2. 任务类型

MineDojo 提供多种类型的内置任务:

### 2.1 Harvest 任务
收集特定物品:
```python
env = minedojo.make(task_id="harvest_milk")
env = minedojo.make(task_id="harvest_wool")
env = minedojo.make(task_id="harvest_bamboo")
```

### 2.2 Combat 任务
战斗相关任务:
```python
env = minedojo.make(
    task_id="combat_spider_plains_leather_armors_iron_sword_shield"
)
```

### 2.3 Tech Tree 任务
制作工具链任务:
```python
env = minedojo.make(task_id="techtree_Wooden_Pickaxe")
env = minedojo.make(task_id="techtree_Stone_Pickaxe")
env = minedojo.make(task_id="techtree_Iron_Pickaxe")
```

### 2.4 Creative 任务
开放式探索:
```python
env = minedojo.make(task_id="open-ended")
```

---

## 3. 从头训练流程

### 3.1 基本训练循环

```python
import minedojo

# 1. 创建环境
env = minedojo.make(task_id="harvest_milk", image_size=(160, 256))

# 2. 初始化策略（随机初始化，没有预训练权重）
policy = YourPolicyNetwork()  # 需要自己定义

# 3. 训练循环
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        # 策略推理
        action = policy.get_action(obs)
        
        # 环境交互
        obs, reward, done, info = env.step(action)
        
        # 更新策略（使用RL算法）
        policy.update(obs, action, reward, done)
    
    # 保存检查点
    if episode % save_interval == 0:
        policy.save_checkpoint(f"checkpoint_{episode}.pt")

env.close()
```

### 3.2 使用Stable-Baselines3训练

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import minedojo

# 创建环境
def make_env():
    env = minedojo.make(
        task_id="harvest_milk",
        image_size=(160, 256)
    )
    return env

env = DummyVecEnv([make_env])

# 创建PPO模型（从头训练）
model = PPO(
    "CnnPolicy",  # 使用CNN策略处理图像观察
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_harvest/"
)

# 训练
model.learn(total_timesteps=1000000)

# 保存模型
model.save("harvest_milk_ppo")

# 评估
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

---

## 4. harvest_1_paper 任务示例

### 4.1 任务分析

| 属性 | 说明 |
|------|------|
| **目标** | 收集1个纸 |
| **前置条件** | 需要找到甘蔗、制作纸 |
| **难度** | 中等（需要导航、采集、制作） |
| **平均步数** | 500-2000步（取决于策略） |

### 4.2 任务挑战

1. **导航**: 需要找到甘蔗（通常在水边）
2. **采集**: 需要收集3个甘蔗
3. **制作**: 需要打开合成界面制作纸

### 4.3 奖励设计

```python
# MineDojo内部实现（简化版本）
def compute_reward(prev_inventory, curr_inventory):
    prev_paper = prev_inventory.get("paper", 0)
    curr_paper = curr_inventory.get("paper", 0)
    
    if curr_paper > prev_paper:
        return 1.0  # 成功收集到纸
    return 0.0  # 未完成
```

---

## 5. 训练建议

### 5.1 算法选择

| 算法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **PPO** | 大多数任务 | 稳定、易调参 | 样本效率较低 |
| **DQN** | 简单任务 | 样本效率高 | 不适合连续动作 |
| **IMPALA** | 大规模训练 | 高吞吐量 | 需要分布式 |

### 5.2 网络架构建议

```python
import torch.nn as nn

class MinecraftPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 视觉编码器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        # 循环层（处理时序信息）
        self.lstm = nn.LSTM(1024, 512)
        # 动作头
        self.policy_head = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)
```

### 5.3 超参数参考

```yaml
# PPO推荐超参数
learning_rate: 0.0003
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
```

---

## 6. 评估与监控

### 6.1 关键指标

- **平均奖励**: 每个episode的累计奖励
- **成功率**: 完成任务的episode比例
- **平均步数**: 完成任务所需的平均步数
- **探索效率**: 发现新区域的速度

### 6.2 使用TensorBoard监控

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs/harvest_training')

for episode in range(num_episodes):
    # 训练...
    
    # 记录指标
    writer.add_scalar('reward/episode', total_reward, episode)
    writer.add_scalar('success/rate', success_rate, episode)
    writer.add_scalar('steps/episode', num_steps, episode)

writer.close()
```

---

## 7. 常见问题

### Q1: harvest_1_paper 任务ID无效怎么办？

A: 某些任务ID可能在特定MineDojo版本中不可用。尝试:
```python
# 方法1: 使用类似任务
env = minedojo.make(task_id="harvest_milk")

# 方法2: 自定义任务
env = minedojo.make(
    task_id="open-ended",
    task_prompt="Obtain paper",
    # 自定义奖励函数
)
```

### Q2: 训练需要多长时间？

A: 取决于:
- 任务复杂度: 简单任务(harvest_milk) 数小时，复杂任务(tech tree) 数天
- 硬件: GPU训练快10-100倍
- 算法: PPO通常需要1M-10M步

### Q3: 如何加速训练？

A:
1. **并行环境**: 使用多个环境同时收集数据
2. **分布式训练**: 使用Ray/IMPALA
3. **Curriculum Learning**: 从简单任务开始
4. **预训练**: 使用MineCLIP等预训练视觉模型

---

## 8. 参考资源

- [MineDojo官方文档](https://docs.minedojo.org/)
- [MineDojo论文](https://arxiv.org/abs/2206.08853)
- [MineDojo GitHub](https://github.com/MineDojo/MineDojo)
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)

---

## 总结

**关键要点**:
- ✅ MineDojo内置任务 = 环境 + 奖励函数
- ❌ 没有预训练模型，需要从头训练
- 🔄 训练流程: 创建环境 → 定义策略 → RL算法训练 → 评估
- 📊 建议使用PPO算法 + CNN+LSTM架构
- ⏱️ 预期训练时间: 数小时到数天

**下一步**:
1. 运行 `src/demo_harvest_task.py` 了解环境交互
2. 选择RL框架 (推荐Stable-Baselines3)
3. 编写训练脚本
4. 开始训练并监控指标

