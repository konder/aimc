# Round vs Episode 概念说明

> **目的**: 澄清录制数据时的目录结构和BC训练的数据格式要求

---

## 📚 **背景**

在实现DAgger/BC训练时，出现了对`round`和`episode`概念的混淆。本文档解释这两个术语在不同上下文中的含义，以及我们的最终设计决策。

---

## 🔍 **术语定义**

### **在强化学习中的标准定义**

| 术语 | 英文 | 定义 | 示例 |
|------|------|------|------|
| **Episode** | Episode | 一次完整的任务执行（从reset到done） | 从出生到获得一块木头 |
| **Step** | Step | 一次动作执行 | 按一次W键前进 |
| **Round/Iteration** | Round/Iteration | 多次episode的集合（训练循环） | PPO的1000步rollout |

---

### **在本项目中的使用**

由于手动录制数据时，每次录制就是一个完整的任务（从reset到done），因此：

**统一术语**: 使用 **Episode** 表示一次完整的录制任务

```
data/expert_demos/
├── episode_000/    # 第1次录制（第1个完整任务）
│   ├── frame_00000.png
│   ├── frame_00000.npy
│   └── metadata.txt
├── episode_001/    # 第2次录制（第2个完整任务）
├── episode_002/    # 第3次录制（第3个完整任务）
└── ...
```

---

## 🎯 **为什么改为Episode？**

### **1. 与BC训练代码一致**

`src/training/train_bc.py` 期望的数据格式：

```python
# BC训练脚本会查找以下文件：
episode_files = sorted(data_path.glob("episode_*.npy"))  # ✅ 正确
# 或者
frame_files = sorted(data_path.glob("frame_*.npy"))      # ✅ 正确

# 不会查找：
round_files = sorted(data_path.glob("round_*/"))  # ❌ 错误
```

---

### **2. 符合强化学习标准术语**

在强化学习社区中：
- **Episode** = 一次完整任务（reset → done）✅
- **Round** = 多次episode的训练迭代 ⚠️

我们的每次录制就是一个**完整任务**，因此应该叫**Episode**。

---

### **3. 简化数据结构**

**之前的设计（错误）**:
```
data/expert_demos/
├── round_0/           # ❌ round是什么？
│   ├── episode_0/     # ❌ 为什么round下还有episode？
│   └── episode_1/
└── round_1/
```

**当前设计（正确）**:
```
data/expert_demos/
├── episode_000/       # ✅ 清晰：第1个完整任务
├── episode_001/       # ✅ 清晰：第2个完整任务
└── episode_002/       # ✅ 清晰：第3个完整任务
```

---

## 📂 **数据目录结构（最终版）**

```
data/expert_demos/
├── episode_000/                    # Episode 0 (第1次录制)
│   ├── frame_00000.png            # 可视化图片
│   ├── frame_00000.npy            # BC训练数据 {observation, action}
│   ├── frame_00001.png
│   ├── frame_00001.npy
│   ├── ...
│   └── metadata.txt               # 元数据
├── episode_001/                    # Episode 1 (第2次录制)
│   ├── frame_00000.png
│   ├── frame_00000.npy
│   └── ...
├── episode_002/                    # Episode 2 (第3次录制)
│   └── ...
└── summary.txt                     # 全局统计
```

---

## 🔧 **每个文件的作用**

### **`frame_XXXXX.png`**

- **格式**: PNG图片
- **用途**: 可视化验证（人工检查录制质量）
- **shape**: `(H, W, 3)` RGB uint8
- **不用于训练**（只是为了方便人眼检查）

---

### **`frame_XXXXX.npy`**

- **格式**: NumPy字典 `{'observation': obs, 'action': action}`
- **用途**: BC训练
- **observation shape**: `(H, W, 3)` RGB uint8
- **action shape**: `(8,)` int64（MineDojo MultiDiscrete）
- **BC训练脚本读取这个文件**

---

### **`metadata.txt`**

- **格式**: 文本文件
- **内容**: episode统计信息
  ```
  Episode: 0
  Frames: 234
  Actions: 234
  Total Reward: 1.0
  Task Completed: True
  Recording Time: 2025-10-21 15:30:00
  ```

---

### **`summary.txt`**

- **格式**: 文本文件
- **内容**: 全局统计
  ```
  Total Completed Episodes: 10
  Episode Range: episode_000 ~ episode_009
  Camera Delta: 1
  Max Frames per Episode: 1000
  
  Saved Episodes:
    episode_000: 234 frames
    episode_001: 189 frames
    ...
  ```

---

## 🎮 **录制工作流程**

### **首次录制**

```bash
python tools/record_manual_chopping.py --base-dir data/expert_demos --max-episodes 10

# 输出:
✓ 目录为空，从 episode_000 开始
Episode范围: episode_000 ~ episode_009
```

---

### **继续录制（自动检测）**

```bash
# 假设已有 episode_000 ~ episode_009（共10个）
python tools/record_manual_chopping.py --base-dir data/expert_demos --max-episodes 5

# 输出:
✓ 检测到已有 10 个episode，从 episode_010 开始
Episode范围: episode_010 ~ episode_014
```

**自动续录特性**:
- 自动检测已有的episode
- 从下一个编号开始
- 无需手动指定起始编号
- 不会覆盖已有数据

---

## 🚀 **BC训练使用**

### **训练命令**

```bash
# 使用单个episode训练
python src/training/train_bc.py \
    --data data/expert_demos/episode_000/ \
    --output checkpoints/bc_test.zip

# 使用整个目录训练（推荐）
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 50
```

---

### **BC脚本如何加载数据**

`src/training/train_bc.py` 的 `load_expert_demonstrations` 函数：

```python
def load_expert_demonstrations(data_path):
    if data_path.is_dir():
        # 查找frame_*.npy文件（单episode目录）
        frame_files = sorted(data_path.glob("frame_*.npy"))
        
        for file in frame_files:
            frame_data = np.load(file, allow_pickle=True).item()
            obs = frame_data['observation']  # (H, W, 3)
            action = frame_data['action']    # (8,)
            observations.append(obs)
            actions.append(action)
        
        # 如果data_path是父目录，自动递归查找所有episode
        episode_dirs = sorted(data_path.glob("episode_*/"))
        for ep_dir in episode_dirs:
            # 处理每个episode...
    
    return observations, actions
```

---

## 📊 **数据统计示例**

假设录制了10个episode：

```
data/expert_demos/
├── episode_000/  # 234 帧
├── episode_001/  # 189 帧
├── episode_002/  # 201 帧
├── episode_003/  # 178 帧
├── episode_004/  # 256 帧
├── episode_005/  # 198 帧
├── episode_006/  # 212 帧
├── episode_007/  # 187 帧
├── episode_008/  # 223 帧
└── episode_009/  # 195 帧

总帧数: 2073 帧
平均每episode: 207.3 帧
```

---

## 🎯 **总结**

| 项目 | 说明 |
|------|------|
| **目录名** | `episode_000`, `episode_001`, ... |
| **编号格式** | 3位数字（000-999） |
| **自动续录** | ✅ 自动检测最后一个episode，从下一个开始 |
| **数据格式** | PNG（可视化）+ NPY（BC训练） |
| **BC训练** | 直接读取整个目录，自动加载所有episode |
| **术语一致** | 与强化学习标准术语一致 ✅ |

---

## 🔗 **相关文档**

- [`docs/guides/DAGGER_QUICK_START.md`](../guides/DAGGER_QUICK_START.md) - DAgger快速开始
- [`docs/guides/DAGGER_DETAILED_GUIDE.md`](../guides/DAGGER_DETAILED_GUIDE.md) - DAgger详细指南
- [`src/training/train_bc.py`](../../src/training/train_bc.py) - BC训练脚本

---

**最后更新**: 2025-10-21  
**关键改动**: 统一使用 `episode_XXX` 目录结构，移除 `round` 概念

