# MineRL和MineDojo数据集详细指南

本文档详细介绍如何获取和使用MineRL与MineDojo的人类玩家游戏数据集。

---

## 🎯 问题1：MineRL和MineDojo数据集获取与结构

### MineRL数据集（推荐用于离线RL）

#### 1.1 数据集概述

**MineRL数据集**由卡内基梅隆大学创建，包含：
- **6000万+** 状态-动作对
- **人类玩家**真实游戏录像
- **多种任务**（导航、采矿、建造等）
- **自动标注**的轨迹数据

**官网**：http://minerl.io

#### 1.2 获取数据集

##### 方法1：使用MineRL Python包（推荐）

```bash
# 安装MineRL
pip install minerl

# 数据会在首次使用时自动下载
```

**Python代码示例**：

```python
import minerl

# 下载并加载数据集
# 第一次运行会自动下载数据（数据量较大，可能需要几小时）
data = minerl.data.make('MineRLTreechop-v0')

# 查看数据集信息
print(f"Dataset: MineRLTreechop-v0")
print(f"Available trajectories: {data.size}")

# 遍历数据
for state, action, reward, next_state, done in data.batch_iter(
    batch_size=1, 
    num_epochs=1,
    seq_len=1
):
    print("State keys:", state.keys())
    print("Action keys:", action.keys())
    print("Reward:", reward)
    break  # 只查看第一个样本
```

**输出示例**：
```
Dataset: MineRLTreechop-v0
Available trajectories: 1428
State keys: dict_keys(['pov', 'inventory', 'equipped_items'])
Action keys: dict_keys(['camera', 'forward', 'jump', 'attack'])
Reward: 0.0
```

##### 方法2：手动下载

```bash
# 数据集会下载到：
~/.minerl/datasets/

# 数据集大小：
# MineRLTreechop-v0: ~15GB
# MineRLNavigate-v0: ~20GB
# MineRLObtainDiamond-v0: ~45GB
```

#### 1.3 数据集结构

MineRL数据是**轨迹数据**（trajectories），不是简单的标记数据。

**数据格式**：

```python
# 每个样本包含：
{
    'state': {
        'pov': np.ndarray,           # 第一人称视角图像 (64, 64, 3)
        'inventory': dict,           # 物品栏状态
        'equipped_items': dict,      # 手持物品
        'compass': dict,             # 罗盘信息（导航任务）
    },
    'action': {
        'camera': np.ndarray,        # 摄像机移动 [pitch, yaw]
        'forward': int,              # 前进 (0或1)
        'back': int,                 # 后退
        'left': int,                 # 左移
        'right': int,                # 右移
        'jump': int,                 # 跳跃
        'sneak': int,                # 潜行
        'sprint': int,               # 冲刺
        'attack': int,               # 攻击
        'use': int,                  # 使用
        'craft': str,                # 合成物品
        'nearbyCraft': str,          # 工作台合成
        'nearbySmelt': str,          # 熔炉冶炼
        'equip': str,                # 装备物品
        'place': str,                # 放置方块
    },
    'reward': float,                 # 奖励值
    'done': bool,                    # episode是否结束
}
```

**完整示例代码**：

```python
import minerl
import numpy as np
from PIL import Image

def explore_minerl_dataset(dataset_name='MineRLTreechop-v0', num_samples=10):
    """
    探索MineRL数据集的结构
    
    Args:
        dataset_name: 数据集名称
        num_samples: 查看的样本数
    """
    print(f"{'='*70}")
    print(f"Exploring MineRL Dataset: {dataset_name}")
    print(f"{'='*70}\n")
    
    # 加载数据集
    print("[1/4] Loading dataset...")
    data = minerl.data.make(dataset_name)
    print(f"✓ Dataset loaded")
    print(f"  Total trajectories: {data.size}")
    print()
    
    # 查看轨迹统计
    print("[2/4] Trajectory statistics...")
    trajectory_names = data.get_trajectory_names()
    print(f"  Number of trajectories: {len(trajectory_names)}")
    print(f"  First 5 trajectories: {trajectory_names[:5]}")
    print()
    
    # 加载一条轨迹
    print("[3/4] Loading a trajectory...")
    first_trajectory = trajectory_names[0]
    trajectory = data.load_data(first_trajectory)
    
    trajectory_length = sum(1 for _ in trajectory)
    print(f"  Trajectory: {first_trajectory}")
    print(f"  Length: {trajectory_length} steps")
    print()
    
    # 详细查看前几个样本
    print(f"[4/4] Examining first {num_samples} samples...")
    print()
    
    trajectory = data.load_data(first_trajectory)
    for i, (state, action, reward, next_state, done) in enumerate(trajectory):
        if i >= num_samples:
            break
        
        print(f"--- Sample {i+1} ---")
        
        # State信息
        print(f"State:")
        print(f"  POV shape: {state['pov'].shape}")  # 视角图像
        print(f"  POV dtype: {state['pov'].dtype}")
        print(f"  POV range: [{state['pov'].min()}, {state['pov'].max()}]")
        
        if 'inventory' in state:
            print(f"  Inventory: {dict(list(state['inventory'].items())[:3])}...")
        
        # Action信息
        print(f"Action:")
        action_summary = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action_summary[key] = f"array{value.shape}"
            else:
                action_summary[key] = value
        print(f"  {action_summary}")
        
        # Reward和Done
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print()
        
        # 可选：保存第一帧图像
        if i == 0:
            pov_img = state['pov']
            img = Image.fromarray(pov_img)
            img.save(f"minerl_sample_{dataset_name}.png")
            print(f"✓ Saved first frame to: minerl_sample_{dataset_name}.png")
            print()
    
    print(f"{'='*70}")
    print("Dataset exploration completed!")
    print(f"{'='*70}")


def count_actions_in_dataset(dataset_name='MineRLTreechop-v0', max_trajectories=10):
    """
    统计数据集中的动作分布
    
    Args:
        dataset_name: 数据集名称
        max_trajectories: 最多统计多少条轨迹
    """
    data = minerl.data.make(dataset_name)
    
    action_counts = {}
    total_steps = 0
    
    print(f"Counting actions in {dataset_name}...")
    
    trajectory_names = data.get_trajectory_names()[:max_trajectories]
    
    for traj_name in trajectory_names:
        trajectory = data.load_data(traj_name)
        
        for state, action, reward, next_state, done in trajectory:
            total_steps += 1
            
            # 统计每个动作键
            for key, value in action.items():
                if key not in action_counts:
                    action_counts[key] = {'total': 0, 'nonzero': 0}
                
                action_counts[key]['total'] += 1
                
                # 检查是否为非零/非空动作
                if isinstance(value, np.ndarray):
                    if np.any(value != 0):
                        action_counts[key]['nonzero'] += 1
                elif isinstance(value, (int, float)):
                    if value != 0:
                        action_counts[key]['nonzero'] += 1
                elif isinstance(value, str):
                    if value != 'none' and value != '':
                        action_counts[key]['nonzero'] += 1
    
    print(f"\n{'='*70}")
    print(f"Action Statistics ({dataset_name})")
    print(f"{'='*70}")
    print(f"Total steps analyzed: {total_steps}")
    print()
    print(f"{'Action':<20} {'Total':<15} {'Active':<15} {'Active %':<10}")
    print(f"{'-'*70}")
    
    for action_name, counts in sorted(action_counts.items()):
        total = counts['total']
        nonzero = counts['nonzero']
        percentage = (nonzero / total * 100) if total > 0 else 0
        print(f"{action_name:<20} {total:<15} {nonzero:<15} {percentage:>6.2f}%")


if __name__ == "__main__":
    # 使用示例
    print("MineRL Dataset Explorer\n")
    
    # 可用的数据集
    available_datasets = [
        'MineRLTreechop-v0',           # 砍树
        'MineRLNavigate-v0',           # 导航
        'MineRLNavigateDense-v0',      # 密集奖励导航
        'MineRLNavigateExtreme-v0',    # 极限导航
        'MineRLObtainDiamond-v0',      # 获取钻石
        'MineRLObtainIronPickaxe-v0',  # 获取铁镐
    ]
    
    print("Available datasets:")
    for i, ds in enumerate(available_datasets, 1):
        print(f"  {i}. {ds}")
    print()
    
    # 探索数据集
    explore_minerl_dataset('MineRLTreechop-v0', num_samples=3)
    
    # 统计动作
    print("\n")
    count_actions_in_dataset('MineRLTreechop-v0', max_trajectories=5)
```

#### 1.4 可用的数据集列表

| 数据集名称 | 任务描述 | 数据量 | 难度 |
|-----------|---------|--------|------|
| `MineRLTreechop-v0` | 砍树获取木头 | ~15GB | 简单 |
| `MineRLNavigate-v0` | 导航到目标 | ~20GB | 简单 |
| `MineRLNavigateDense-v0` | 导航（密集奖励） | ~20GB | 简单 |
| `MineRLNavigateExtreme-v0` | 极限导航 | ~25GB | 中等 |
| `MineRLObtainIronPickaxe-v0` | 制作铁镐 | ~35GB | 困难 |
| `MineRLObtainDiamond-v0` | 获取钻石 | ~45GB | 非常困难 |

#### 1.5 数据是标记数据吗？

**不完全是**。MineRL数据集是：

✅ **自动标注的轨迹数据**：
- 包含人类玩家的完整游戏轨迹
- 状态（观察）和动作都被记录
- 奖励根据任务自动计算

❌ **不是监督学习意义上的标记数据**：
- 没有"最优动作"标签
- 人类玩家可能犯错
- 需要离线RL算法处理（不能直接用监督学习）

**适用算法**：
- 行为克隆（Behavior Cloning, BC）- 简单但受限于人类水平
- 离线强化学习（Offline RL）- CQL, IQL, BCQ等

---

### MineDojo数据集

#### 2.1 数据集概述

**MineDojo数据集**包含两类数据：

1. **YouTube视频数据库**（未公开原始数据）
   - 73万+ Minecraft游戏视频
   - 用于训练MineCLIP模型
   - 不直接提供下载（模型已训练好）

2. **Wiki知识库**（可下载）
   - 6,735页Minecraft Wiki
   - 文本、图像、表格、图示
   - 用于知识增强

#### 2.2 获取Wiki知识库

**下载地址**：https://zenodo.org/records/6693745

```bash
# 下载样本（10页）
wget https://zenodo.org/record/6693745/files/wiki_samples.zip

# 下载完整版（6,735页）
wget https://zenodo.org/record/6693745/files/wiki_full.zip

# 解压
unzip wiki_samples.zip
```

**数据结构**：

```
wiki_samples/
├── pages/
│   ├── page_0001.json      # 页面内容
│   ├── page_0002.json
│   └── ...
├── images/
│   ├── img_0001.png        # 提取的图片
│   ├── img_0002.png
│   └── ...
└── metadata.json           # 元数据
```

**JSON格式示例**：

```json
{
  "title": "Tree",
  "url": "https://minecraft.fandom.com/wiki/Tree",
  "text": "Trees are naturally generated structures...",
  "images": [
    {
      "url": "https://...",
      "caption": "Oak tree",
      "bbox": [100, 200, 300, 400]
    }
  ],
  "tables": [...],
  "diagrams": [...]
}
```

#### 2.3 MineDojo的YouTube数据

**注意**：YouTube视频**不直接提供下载**，而是用于训练MineCLIP模型。

**你可以**：
- ✅ 使用已训练好的MineCLIP模型（推荐）
- ✅ 阅读MineCLIP论文了解数据处理方法
- ❌ 直接下载73万视频（版权问题，数据量巨大）

---

## 🎯 实际应用：离线强化学习

### 使用MineRL数据集训练

```python
# offline_rl_example.py
import minerl
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import ReplayBuffer

def collect_offline_data(dataset_name='MineRLTreechop-v0', max_samples=10000):
    """
    从MineRL数据集收集离线数据
    
    Args:
        dataset_name: 数据集名称
        max_samples: 最多收集多少样本
        
    Returns:
        observations, actions, rewards, next_observations, dones
    """
    data = minerl.data.make(dataset_name)
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    
    print(f"Collecting data from {dataset_name}...")
    
    sample_count = 0
    for state, action, reward, next_state, done in data.batch_iter(
        batch_size=1, 
        num_epochs=1,
        seq_len=1
    ):
        # 提取POV（第一人称视角）
        obs = state['pov'][0]  # (64, 64, 3)
        next_obs = next_state['pov'][0]
        
        # 简化动作（只使用关键动作）
        # 这里需要根据你的动作空间定义转换
        simplified_action = simplify_action(action)
        
        observations.append(obs)
        actions.append(simplified_action)
        rewards.append(reward[0])
        next_observations.append(next_obs)
        dones.append(done[0])
        
        sample_count += 1
        if sample_count >= max_samples:
            break
        
        if sample_count % 1000 == 0:
            print(f"  Collected {sample_count} samples...")
    
    print(f"✓ Collected {sample_count} samples")
    
    return (
        np.array(observations),
        np.array(actions),
        np.array(rewards),
        np.array(next_observations),
        np.array(dones)
    )


def simplify_action(action):
    """
    将MineRL复杂动作转换为简化的动作表示
    
    这是一个示例，需要根据你的实际任务定义
    """
    # 示例：创建一个简单的动作向量
    simplified = []
    
    # 摄像机移动（连续）
    simplified.extend(action['camera'][0])  # [pitch, yaw]
    
    # 移动（离散）
    simplified.append(action['forward'][0])
    simplified.append(action['back'][0])
    simplified.append(action['left'][0])
    simplified.append(action['right'][0])
    
    # 其他动作（离散）
    simplified.append(action['jump'][0])
    simplified.append(action['attack'][0])
    
    return np.array(simplified, dtype=np.float32)


# 使用示例
if __name__ == "__main__":
    # 收集数据
    obs, actions, rewards, next_obs, dones = collect_offline_data(
        'MineRLTreechop-v0',
        max_samples=5000
    )
    
    print(f"\nDataset statistics:")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Total reward: {rewards.sum()}")
    print(f"  Episodes completed: {dones.sum()}")
    
    # 接下来可以：
    # 1. 使用行为克隆（BC）训练初始策略
    # 2. 使用离线RL算法（CQL/IQL）优化策略
    # 3. 在MineDojo环境中微调
```

---

## 📊 数据集对比

| 特性 | MineRL | MineDojo YouTube | MineDojo Wiki |
|------|--------|------------------|---------------|
| **数据类型** | 游戏轨迹 | 视频 | 文本+图像 |
| **可下载** | ✅ 是 | ❌ 否（模型已训练） | ✅ 是 |
| **数据量** | 60M+ 样本 | 730K 视频 | 6,735 页 |
| **用途** | 离线RL、BC | 预训练MineCLIP | 知识增强 |
| **是否标注** | 自动标注 | 无标注 | 结构化 |
| **推荐场景** | 想用人类数据 | 使用MineCLIP | 知识检索 |

---

## 🚀 快速开始

### 步骤1：安装和下载数据

```bash
# 安装MineRL
pip install minerl

# Python中自动下载数据
python -c "import minerl; data = minerl.data.make('MineRLTreechop-v0')"
```

### 步骤2：探索数据

```bash
# 运行探索脚本
python scripts/explore_minerl_dataset.py
```

### 步骤3：训练模型

```bash
# 行为克隆
python scripts/train_behavior_cloning.py --dataset MineRLTreechop-v0

# 离线RL（需要d3rlpy）
python scripts/train_offline_rl.py --dataset MineRLTreechop-v0 --algorithm CQL
```

---

## 📚 参考资料

### MineRL
- 官网：http://minerl.io
- 论文：https://arxiv.org/abs/1904.10079
- GitHub：https://github.com/minerllabs/minerl
- 文档：https://minerl.readthedocs.io/

### MineDojo
- 官网：https://minedojo.org
- 论文：https://arxiv.org/abs/2206.08853
- GitHub：https://github.com/MineDojo/MineDojo
- 数据集：https://zenodo.org/records/6693745

---

## 总结

**问题1回答**：

1. **MineRL数据集**：
   - 获取：`pip install minerl`，自动下载
   - 结构：轨迹数据（状态、动作、奖励）
   - 标注：自动标注，但不是"最优动作"标签
   - 用途：离线RL、行为克隆

2. **MineDojo数据集**：
   - YouTube视频：不直接提供（用于训练MineCLIP）
   - Wiki知识库：可下载（6,735页）
   - 用途：使用MineCLIP模型，而非原始数据

**关键点**：
- ✅ MineRL提供可下载的人类游戏轨迹
- ✅ 数据是自动标注的，但质量受人类玩家水平限制
- ✅ 适合离线RL和行为克隆
- ❌ MineDojo的YouTube数据不直接提供
- ✅ 使用已训练好的MineCLIP模型即可

