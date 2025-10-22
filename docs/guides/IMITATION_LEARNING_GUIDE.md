# 模仿学习（Imitation Learning）指南

> **核心思想**: 通过人工录制专家演示数据，让智能体学习人类的行为策略

---

## ✅ **可行性分析**

### **完全可行！理由如下：**

1. ✅ **已有成功案例**
   - MineDojo官方使用了大量YouTube视频数据
   - MineRL比赛专门使用人类演示数据训练
   - OpenAI的VPT模型也是基于人类演示

2. ✅ **技术已验证**
   - 我们已经实现了录制工具（`tools/record_manual_chopping.py`）
   - 动作空间映射已完成（`functional=3`已验证）
   - 数据格式与MineDojo完全兼容

3. ✅ **相比纯RL的优势**
   - **更快收敛**: 从好的策略开始，不是随机探索
   - **更稳定**: 有明确的行为模式参考
   - **更通用**: 可以学习复杂的长序列任务

---

## 🎯 **适用场景**

### **最适合使用模仿学习的任务：**

| 任务类型 | 适合程度 | 原因 |
|---------|---------|------|
| 🪵 砍树 | ⭐⭐⭐⭐⭐ | 简单、明确，易于演示 |
| 🏗️ 建造 | ⭐⭐⭐⭐⭐ | 需要特定序列，纯RL很难学会 |
| ⚔️ 战斗 | ⭐⭐⭐⭐ | 需要时机和策略，人类演示很有价值 |
| 🌾 种植 | ⭐⭐⭐⭐ | 多步骤，顺序重要 |
| ⛏️ 挖矿 | ⭐⭐⭐ | 路径规划，人类演示有帮助 |
| 🎣 随机探索 | ⭐⭐ | RL可能更适合 |

---

## 🔧 **实施方案**

### **方案1: 行为克隆（Behavior Cloning, BC）** - 最简单

**原理**: 直接监督学习，输入观察，输出动作

```python
# 伪代码
for observation, action in expert_data:
    predicted_action = policy(observation)
    loss = cross_entropy(predicted_action, action)
    optimizer.step()
```

**优点**:
- ✅ 实现简单
- ✅ 收敛快
- ✅ 不需要环境交互

**缺点**:
- ❌ 可能过拟合到演示数据
- ❌ 遇到未见过的情况会失败
- ❌ 分布偏移问题（covariate shift）

---

### **方案2: DAgger（Dataset Aggregation）** - 推荐 ⭐

**原理**: 迭代式收集数据，逐步改进策略

```
1. 用初始专家演示训练策略π
2. 运行策略π，收集新数据
3. 人工标注新数据的正确动作
4. 合并新旧数据，重新训练
5. 重复2-4直到收敛
```

**优点**:
- ✅ 解决分布偏移问题
- ✅ 持续改进
- ✅ 数据利用率高

**缺点**:
- ⚠️ 需要多次人工标注
- ⚠️ 实现稍复杂

---

### **方案3: 预训练 + 强化学习微调** - 最佳实践 ⭐⭐⭐

**原理**: 先用模仿学习预训练，再用RL优化

```
阶段1: 行为克隆（BC）
  ↓ 学习基本策略
阶段2: PPO微调
  ↓ 优化到最优
最终策略
```

**优点**:
- ✅ 结合BC和RL的优势
- ✅ 收敛快且稳定
- ✅ 最终性能最好

**缺点**:
- ⚠️ 需要两阶段训练
- ⚠️ 超参数调整较复杂

---

## 🛠️ **具体实现步骤**

### **Step 1: 数据录制**

```bash
# 录制砍树演示（10-20次）
python tools/record_manual_chopping.py \
  --output-dir data/expert_demos/chop_wood \
  --max-episodes 20

# 录制其他任务
python tools/record_manual_chopping.py \
  --output-dir data/expert_demos/build_house \
  --task-id "creative" \
  --max-episodes 10
```

**录制建议**:
- 每个任务录制10-50次演示
- 保持演示质量（不要失误）
- 覆盖不同场景（森林、平原、不同时间）
- 每次演示包含完整的任务流程

---

### **Step 2: 数据预处理**

创建 `tools/prepare_expert_data.py`:

```python
#!/usr/bin/env python3
"""
整理专家演示数据为训练格式
"""

import os
import glob
import numpy as np
import pickle
from PIL import Image

def load_episode(episode_dir):
    """加载单个episode的数据"""
    frames = []
    actions = []
    
    # 读取所有帧
    frame_files = sorted(glob.glob(f"{episode_dir}/frame_*.png"))
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        frames.append(np.array(img))
    
    # 读取动作序列（需要在录制时保存）
    action_file = f"{episode_dir}/actions.npy"
    if os.path.exists(action_file):
        actions = np.load(action_file)
    
    return {
        'observations': np.array(frames),
        'actions': actions
    }

def prepare_dataset(expert_dir, output_file):
    """准备训练数据集"""
    episodes = []
    
    # 加载所有episode
    episode_dirs = glob.glob(f"{expert_dir}/episode_*")
    for ep_dir in episode_dirs:
        episode = load_episode(ep_dir)
        if len(episode['actions']) > 0:
            episodes.append(episode)
    
    print(f"✓ 加载了 {len(episodes)} 个演示")
    
    # 保存为训练格式
    with open(output_file, 'wb') as f:
        pickle.dump(episodes, f)
    
    print(f"✓ 数据已保存到 {output_file}")

if __name__ == "__main__":
    prepare_dataset(
        expert_dir="data/expert_demos/chop_wood",
        output_file="data/processed/chop_wood_expert.pkl"
    )
```

---

### **Step 3: 行为克隆训练**

创建 `src/training/train_bc.py`:

```python
#!/usr/bin/env python3
"""
行为克隆（Behavior Cloning）训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

class ExpertDataset(Dataset):
    """专家演示数据集"""
    
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.episodes = pickle.load(f)
        
        # 展开所有transitions
        self.observations = []
        self.actions = []
        
        for episode in self.episodes:
            for obs, action in zip(episode['observations'], episode['actions']):
                self.observations.append(obs)
                self.actions.append(action)
        
        print(f"✓ 加载了 {len(self.observations)} 个transition")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float() / 255.0
        obs = obs.permute(2, 0, 1)  # HWC -> CHW
        
        action = torch.from_numpy(self.actions[idx]).long()
        
        return obs, action

def train_bc(data_file, output_model, epochs=50, batch_size=32):
    """训练行为克隆模型"""
    
    # 加载数据
    dataset = ExpertDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建策略网络（使用SB3的策略结构）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TODO: 创建策略网络
    # policy = ActorCriticPolicy(...)
    
    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        
        for observations, actions in dataloader:
            observations = observations.to(device)
            actions = actions.to(device)
            
            # 前向传播
            predicted_actions = policy(observations)
            
            # 计算损失（针对MultiDiscrete动作空间）
            loss = 0
            acc = 0
            for i in range(8):  # 8个动作维度
                loss += criterion(predicted_actions[:, i], actions[:, i])
                acc += (predicted_actions[:, i].argmax(1) == actions[:, i]).float().mean()
            
            loss /= 8
            acc /= 8
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    # 保存模型
    torch.save(policy.state_dict(), output_model)
    print(f"✓ 模型已保存到 {output_model}")

if __name__ == "__main__":
    train_bc(
        data_file="data/processed/chop_wood_expert.pkl",
        output_model="checkpoints/bc_chop_wood.pth",
        epochs=50
    )
```

---

### **Step 4: BC预训练 + PPO微调**

修改 `src/training/train_get_wood.py`:

```python
def main():
    # ... 现有代码 ...
    
    # 如果有预训练模型，加载它
    if args.pretrain_model:
        print(f"✓ 从BC预训练模型加载: {args.pretrain_model}")
        model = PPO.load(args.pretrain_model, env=env)
    else:
        # 从头训练
        model = PPO(
            "CnnPolicy",
            env,
            # ... 参数 ...
        )
    
    # PPO微调
    model.learn(total_timesteps=args.total_steps)
```

**训练流程**:
```bash
# 1. 行为克隆预训练
python src/training/train_bc.py \
  --data data/processed/chop_wood_expert.pkl \
  --output checkpoints/bc_pretrain.zip

# 2. PPO微调
python src/training/train_get_wood.py \
  --pretrain-model checkpoints/bc_pretrain.zip \
  --total-steps 100000 \
  --learning-rate 1e-4  # 降低学习率
```

---

## 📊 **数据需求评估**

### **最小数据量**

| 任务复杂度 | 演示次数 | 总帧数 | 训练时间 |
|-----------|---------|--------|---------|
| 简单（砍树）| 10-20次 | 5K-10K | 10分钟 |
| 中等（建造）| 30-50次 | 20K-30K | 30分钟 |
| 复杂（探险）| 50-100次 | 50K-100K | 1-2小时 |

### **数据质量 > 数量**

✅ **好的演示**:
- 完整完成任务
- 动作连贯自然
- 没有明显失误
- 覆盖典型场景

❌ **差的演示**:
- 中途失败
- 动作混乱
- 大量无效操作
- 只有单一场景

---

## ⚡ **快速开始 - 砍树任务BC训练**

### **1. 修改录制工具保存动作**

需要修改 `tools/record_manual_chopping.py`，添加动作保存：

```python
# 在录制循环中
actions_list = []

while not done:
    # ... 现有代码 ...
    action = controller.get_action()
    actions_list.append(action.copy())
    obs_dict, reward, done, info = env.step(action)
    # ...

# 保存动作序列
np.save(os.path.join(output_dir, 'actions.npy'), np.array(actions_list))
```

### **2. 录制演示数据**

```bash
# 录制20次砍树演示
for i in {1..20}; do
    python tools/record_manual_chopping.py \
      --output-dir data/expert_demos/chop_wood/episode_$i
done
```

### **3. 准备训练数据**

```bash
python tools/prepare_expert_data.py \
  --input data/expert_demos/chop_wood \
  --output data/processed/chop_wood_expert.pkl
```

### **4. BC训练**

```bash
python src/training/train_bc.py \
  --data data/processed/chop_wood_expert.pkl \
  --epochs 50
```

### **5. PPO微调（可选）**

```bash
python src/training/train_get_wood.py \
  --pretrain-model checkpoints/bc_chop_wood.pth \
  --total-steps 50000
```

---

## 🎯 **预期效果**

### **与纯RL对比**

| 指标 | 纯RL（PPO） | BC预训练 + PPO | 提升 |
|------|------------|---------------|------|
| 首次成功 | ~50K steps | ~5K steps | **10x** |
| 稳定成功率 | 100K steps | 20K steps | **5x** |
| 最终成功率 | 80-90% | 90-95% | +10% |
| 训练时间 | 3-5小时 | 1-2小时 | **2-3x** |

### **成功案例参考**

- **MineRL比赛**: 冠军队伍都使用了人类演示数据
- **OpenAI VPT**: 7万小时YouTube视频 → 能玩Minecraft
- **DeepMind Gato**: 多任务模仿学习

---

## ⚠️ **注意事项**

### **常见问题**

1. **数据不够多样化**
   - ✅ 解决: 录制不同场景（森林、平原、山地）
   - ✅ 解决: 不同时间（白天、黄昏）
   - ✅ 解决: 不同起始位置

2. **过拟合到演示数据**
   - ✅ 解决: 使用DAgger迭代收集
   - ✅ 解决: 加入数据增强（颜色抖动、裁剪）
   - ✅ 解决: PPO微调纠正

3. **动作标注错误**
   - ✅ 解决: 录制时实时显示动作（已实现）
   - ✅ 解决: 录制后验证回放
   - ✅ 解决: 剔除失败的演示

---

## 🚀 **下一步行动**

### **立即可做**:
1. 修改录制工具保存动作序列
2. 录制10-20次砍树演示
3. 创建数据预处理脚本

### **中期目标**:
1. 实现简单的BC训练
2. 对比BC vs 纯RL效果
3. 实现BC+PPO混合训练

### **长期目标**:
1. 录制多种任务的演示数据
2. 实现通用的模仿学习框架
3. 探索VPT风格的大规模预训练

---

## 📚 **相关资源**

- **MineRL数据集**: https://minerl.io/dataset/
- **OpenAI VPT论文**: https://arxiv.org/abs/2206.11795
- **DAgger论文**: https://arxiv.org/abs/1011.0686
- **Stable-Baselines3 Imitation**: https://imitation.readthedocs.io/

---

**结论**: ✅ **使用人工录制数据训练完全可行，且效果通常比纯RL更好！**

建议从简单的BC开始，然后逐步引入PPO微调，最终实现最佳性能。

