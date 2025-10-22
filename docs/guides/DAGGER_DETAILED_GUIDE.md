# DAgger (Dataset Aggregation) 详细指南

> **DAgger**: 解决模仿学习中"分布偏移"问题的迭代式数据收集算法

---

## 🎯 **什么是DAgger？**

**DAgger** (Dataset Aggregation) 是由Ross等人在2011年提出的改进版行为克隆算法。

### **核心思想**

传统行为克隆(BC)的问题：
```
专家演示: s₀ → s₁ → s₂ → s₃ (专家轨迹)
学习策略: s₀ → s₁' → s₂'' → s₃''' (略有偏差)
         ↑    ↑     ↑      ↑
      相同  略偏  更偏   完全偏离！
```

**问题**: 一旦偏离专家演示，策略会越来越差（**分布偏移问题**）

**DAgger解决方案**: 在策略访问的新状态上收集专家标注！

---

## 🔄 **DAgger算法流程**

### **完整流程图**

```
┌─────────────────────────────────────────┐
│  阶段1: 初始训练                         │
│  用专家演示D₀训练初始策略π₁             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  阶段2: 策略执行 (第i轮)                 │
│  运行当前策略πᵢ，收集新轨迹              │
│  记录访问的状态 Sᵢ = {s₁, s₂, ...}      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  阶段3: 专家标注                         │
│  人工/专家对Sᵢ中的状态标注正确动作       │
│  Dᵢ = {(s, a*) | s ∈ Sᵢ}               │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  阶段4: 数据聚合                         │
│  D_all = D₀ ∪ D₁ ∪ ... ∪ Dᵢ           │
│  用所有数据重新训练策略πᵢ₊₁              │
└─────────────────────────────────────────┘
              ↓
         重复2-4，直到收敛
```

### **伪代码**

```python
def dagger(expert, initial_demonstrations, num_iterations):
    # 初始化
    D = initial_demonstrations  # 专家演示数据集
    π = train_policy(D)         # 初始策略
    
    for i in range(num_iterations):
        # 1. 用当前策略收集轨迹
        trajectories = []
        for episode in range(num_episodes):
            states = rollout_policy(π)  # 运行策略
            trajectories.extend(states)
        
        # 2. 专家标注
        labeled_data = []
        for state in trajectories:
            expert_action = expert.get_action(state)  # 人工标注
            labeled_data.append((state, expert_action))
        
        # 3. 聚合数据
        D = D + labeled_data
        
        # 4. 重新训练
        π = train_policy(D)
    
    return π
```

---

## 🔬 **为什么DAgger有效？**

### **问题：传统BC的分布偏移**

**训练分布** vs **测试分布**:

```python
# BC训练时
训练状态分布 = 专家访问的状态
P_train(s) = 只包含专家轨迹上的状态

# BC测试时
测试状态分布 = 学习策略访问的状态
P_test(s) = 包含学习策略偏离后的状态

# 问题: P_train ≠ P_test ⚠️
```

**结果**: 策略在训练时见过的状态表现好，但偏离后没见过新状态，表现崩溃！

### **DAgger的解决**

```python
# DAgger每轮迭代
P_train(s) 逐渐包含策略πᵢ访问的状态

# 经过多轮后
P_train(s) ≈ P_test(s)  ✅

# 结果: 策略在自己访问的状态上也有训练数据！
```

---

## 📊 **DAgger vs BC 对比**

### **训练过程对比**

| 方面 | 传统BC | DAgger |
|------|--------|--------|
| 数据收集 | 一次性（专家演示） | 迭代式（多轮收集）|
| 数据分布 | 只有专家状态 | 专家+策略状态 |
| 标注成本 | 低（一次） | 中（多次，但每次少量）|
| 鲁棒性 | 差（偏离后崩溃） | 好（见过偏离状态）|
| 最终性能 | 中 | 高 |

### **性能对比示例**

**任务**: Minecraft砍树，200步

| 算法 | 初始成功率 | 10轮后成功率 | 数据量 | 标注成本 |
|------|-----------|-------------|--------|---------|
| BC | 60% | 60% | 5K | 1小时（一次）|
| DAgger | 60% | **90%** | 15K | 3小时（分3次）|
| BC + PPO | 60% → 85% | 85% | 5K | 1小时 |

---

## 🛠️ **在Minecraft砍树中实现DAgger**

### **完整实现方案**

#### **第1轮: 初始BC训练**

```python
# 1. 收集初始专家演示（手动录制）
演示数量: 10次
数据量: ~4000帧
时间: 30分钟

# 2. 训练初始策略π₁
python src/training/train_bc.py \
  --data data/expert_demos/round_0.pkl \
  --output checkpoints/dagger_round_1.zip

# 3. 评估
成功率: ~60%
主要失败: 偏离树木、卡在地形
```

#### **第2轮: 第一次DAgger迭代**

```python
# 1. 运行策略π₁收集新状态
python tools/run_policy_collect_states.py \
  --model checkpoints/dagger_round_1.zip \
  --episodes 20 \
  --output data/policy_states/round_1/

# 输出: 
# - 20个episode的状态序列
# - 包含失败场景（偏离、卡住等）

# 2. 人工标注（关键步骤！）
python tools/label_states.py \
  --states data/policy_states/round_1/ \
  --output data/expert_labels/round_1.pkl

# 交互式标注界面:
# 显示状态 → 你给出正确动作 → 保存标注
# 重点标注: 失败/偏离的关键时刻

# 标注量: ~500-1000个关键状态
# 时间: 30-40分钟

# 3. 聚合数据
D₁ = D₀ ∪ 新标注数据

# 4. 重新训练π₂
python src/training/train_bc.py \
  --data data/dagger_combined/round_1.pkl \
  --output checkpoints/dagger_round_2.zip

# 5. 评估
成功率: ~75% (+15%提升！)
```

#### **第3轮: 第二次DAgger迭代**

```python
# 重复相同流程
运行π₂ → 标注新失败场景 → 聚合数据 → 训练π₃

成功率: ~85% (+10%提升)
```

#### **第4-5轮: 继续迭代直到收敛**

```python
成功率曲线:
60% → 75% → 85% → 90% → 92% (收敛)
```

---

## 💻 **代码实现**

### **1. 策略运行并收集状态**

```python
# tools/run_policy_collect_states.py

import gym
import minedojo
import numpy as np
from stable_baselines3 import PPO

def collect_policy_states(model_path, num_episodes, output_dir):
    """运行策略并收集访问的状态"""
    
    # 加载策略
    policy = PPO.load(model_path)
    
    # 创建环境
    env = minedojo.make(task_id="harvest_1_log")
    
    all_states = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_states = []
        
        while not done:
            # 策略选择动作
            action, _ = policy.predict(obs, deterministic=False)
            
            # 保存当前状态
            episode_states.append({
                'observation': obs.copy(),
                'step': len(episode_states),
                'episode': ep
            })
            
            # 执行动作
            obs, reward, done, info = env.step(action)
        
        # 保存episode
        episode_success = info.get('success', False)
        np.save(
            f"{output_dir}/episode_{ep}_success_{episode_success}.npy",
            episode_states
        )
        
        all_states.extend(episode_states)
        print(f"Episode {ep}: {len(episode_states)} states, success={episode_success}")
    
    env.close()
    return all_states
```

### **2. 交互式状态标注工具**

```python
# tools/label_states.py

import cv2
import numpy as np
from collections import deque

class StateLabeler:
    """交互式状态标注工具"""
    
    def __init__(self):
        self.action_mapping = {
            'w': [1, 0, 0, 12, 12, 0, 0, 0],  # 前进
            's': [2, 0, 0, 12, 12, 0, 0, 0],  # 后退
            'a': [0, 1, 0, 12, 12, 0, 0, 0],  # 左移
            'd': [0, 2, 0, 12, 12, 0, 0, 0],  # 右移
            'f': [0, 0, 0, 12, 12, 3, 0, 0],  # 攻击
            'i': [0, 0, 0, 8, 12, 0, 0, 0],   # 向上看
            'k': [0, 0, 0, 16, 12, 0, 0, 0],  # 向下看
            'j': [0, 0, 0, 12, 8, 0, 0, 0],   # 向左看
            'l': [0, 0, 0, 12, 16, 0, 0, 0],  # 向右看
            'n': None,  # 跳过此状态
        }
    
    def label_episode(self, states_file, output_file):
        """标注一个episode的关键状态"""
        
        # 加载状态
        states = np.load(states_file, allow_pickle=True)
        
        labeled_data = []
        current_idx = 0
        
        print("\n" + "="*60)
        print("状态标注工具")
        print("="*60)
        print("控制:")
        print("  WASD - 移动")
        print("  IJKL - 视角")
        print("  F - 攻击")
        print("  N - 跳过此状态")
        print("  Q - 完成标注")
        print("="*60)
        
        while current_idx < len(states):
            state = states[current_idx]
            obs = state['observation']
            
            # 显示当前状态
            display_img = cv2.cvtColor(
                obs.transpose(1, 2, 0), 
                cv2.COLOR_RGB2BGR
            )
            display_img = cv2.resize(display_img, (640, 480))
            
            # 添加信息
            info_text = f"Episode step: {state['step']} | Total: {current_idx}/{len(states)}"
            cv2.putText(display_img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('State Labeling', display_img)
            
            # 等待标注
            key = chr(cv2.waitKey(0) & 0xFF)
            
            if key == 'q':
                break
            elif key in self.action_mapping:
                action = self.action_mapping[key]
                if action is not None:
                    labeled_data.append({
                        'observation': obs,
                        'action': np.array(action),
                        'step': state['step'],
                        'episode': state['episode']
                    })
                    print(f"  ✓ 已标注: {key} → {action}")
                current_idx += 1
            else:
                print(f"  ⚠️ 未知按键: {key}")
        
        # 保存标注数据
        np.save(output_file, labeled_data)
        print(f"\n✓ 已保存 {len(labeled_data)} 个标注")
        
        cv2.destroyAllWindows()
        return labeled_data
```

### **3. DAgger主循环**

```python
# src/training/train_dagger.py

def dagger_training(
    initial_data_path,
    num_iterations=5,
    episodes_per_iteration=20,
    bc_epochs=30
):
    """DAgger训练主循环"""
    
    # 初始化
    all_data = load_data(initial_data_path)
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"DAgger Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # 1. 训练当前策略
        model_path = f"checkpoints/dagger_iter_{iteration}.zip"
        train_bc(
            data=all_data,
            output=model_path,
            epochs=bc_epochs
        )
        
        # 2. 评估当前策略
        success_rate = evaluate_policy(model_path, num_episodes=10)
        print(f"  当前成功率: {success_rate:.1%}")
        
        if success_rate > 0.90:
            print("  ✓ 达到目标成功率，提前结束")
            break
        
        # 3. 收集新状态
        states_dir = f"data/policy_states/iter_{iteration}"
        collect_policy_states(
            model_path=model_path,
            num_episodes=episodes_per_iteration,
            output_dir=states_dir
        )
        
        # 4. 人工标注（关键步骤）
        print(f"\n  请标注新收集的状态...")
        new_labels = label_states_interactive(states_dir)
        
        # 5. 聚合数据
        all_data = aggregate_data(all_data, new_labels)
        print(f"  数据集大小: {len(all_data)} samples")
    
    return model_path
```

---

## 🎯 **智能标注策略**

### **不需要标注所有状态！**

**关键原则**: 只标注**失败场景**和**边界情况**

#### **标注优先级**

| 优先级 | 场景 | 标注比例 | 原因 |
|--------|------|---------|------|
| 🔴 高 | 失败前5步 | 100% | 关键失败点 |
| 🟡 中 | 偏离轨迹 | 50% | 纠正偏差 |
| 🟢 低 | 正常执行 | 10% | 已有专家演示 |

#### **智能采样策略**

```python
def smart_sampling(states, policy, expert_demo):
    """智能选择需要标注的状态"""
    
    to_label = []
    
    for state in states:
        # 1. 失败episode的所有状态
        if state['episode_failed']:
            to_label.append(state)
        
        # 2. 策略不确定的状态（熵高）
        elif policy_entropy(state) > threshold:
            to_label.append(state)
        
        # 3. 偏离专家轨迹的状态
        elif distance_to_expert(state, expert_demo) > threshold:
            to_label.append(state)
        
        # 4. 随机采样10%
        elif random.random() < 0.1:
            to_label.append(state)
    
    return to_label
```

---

## 📈 **预期效果**

### **在Minecraft砍树任务上**

| 轮次 | 数据量 | 标注时间 | 成功率 | 提升 |
|------|--------|---------|--------|------|
| 初始BC | 5K | 40分钟 | 60% | - |
| DAgger-1 | 7K | +30分钟 | 75% | +15% |
| DAgger-2 | 9K | +30分钟 | 85% | +10% |
| DAgger-3 | 11K | +20分钟 | 90% | +5% |
| DAgger-4 | 12K | +20分钟 | 92% | +2% |

**总时间**: 2.5小时  
**总标注**: ~12K样本  
**最终成功率**: 92%

---

## ⚠️ **DAgger的局限性**

### **1. 需要多次人工标注**

**工作量**:
- BC: 一次性录制（1小时）
- DAgger: 多轮标注（2-3小时总计）

**缓解方法**:
- 使用智能采样（只标注20-30%）
- 专注失败场景
- 后期可以用策略自己玩代替人工

### **2. 专家需要一致**

如果不同轮次标注风格不同，会混淆策略

**解决**:
- 同一个人标注
- 制定明确的标注规范
- 回顾之前的标注保持一致

### **3. 标注延迟**

每轮需要等待标注完成

**解决**:
- 异步标注（晚上标注，白天训练）
- 批量标注（积累多个episode一起标注）

---

## 🚀 **快速开始DAgger（Minecraft砍树）**

### **完整流程（预计3小时）**

```bash
# ===== 第0轮: 初始BC =====
# 1. 录制专家演示（40分钟）
python tools/record_manual_chopping.py --episodes 10

# 2. 训练初始BC（10分钟）
python src/training/train_bc.py \
  --data data/expert_demos/initial.pkl \
  --output checkpoints/dagger_r0.zip

# ===== 第1轮: DAgger迭代1 =====
# 3. 运行策略收集状态（5分钟）
python tools/run_policy_collect_states.py \
  --model checkpoints/dagger_r0.zip \
  --episodes 20

# 4. 标注失败场景（30分钟）
python tools/label_states.py \
  --states data/policy_states/round_1/

# 5. 聚合并重新训练（10分钟）
python src/training/train_dagger.py --iteration 1

# ===== 第2轮: DAgger迭代2 =====
# 重复3-5（30分钟）

# ===== 第3轮: DAgger迭代3 =====
# 重复3-5（20分钟）

# 总计: ~2.5小时 → 成功率从60% → 90%
```

---

## 💡 **DAgger vs BC+PPO 选择建议**

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 有充足时间标注 | **DAgger** | 最终性能最好 |
| 快速验证 | BC+PPO | 更快看到结果 |
| 标注资源有限 | BC+PPO | 一次性标注 |
| 需要极致性能 | **DAgger** + PPO | 两者结合 |
| 复杂长序列任务 | **DAgger** | 更鲁棒 |

---

## 📚 **理论基础**

### **关键论文**

**DAgger原论文**:
- Ross, S., Gordon, G., & Bagnell, D. (2011). 
- "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
- AISTATS 2011

**核心定理**:

DAgger的性能界限:
```
ε(π_dagger) ≤ ε_expert + O(T·ε_BC)
```

其中:
- T: 轨迹长度
- ε_expert: 专家误差
- ε_BC: 行为克隆误差

**vs 传统BC**:
```
ε(π_BC) ≤ ε_expert + O(T²·ε_BC)  # 注意是T²！
```

**结论**: DAgger的误差增长是线性的，BC是二次的！

---

## 🔗 **相关资源**

- **论文**: https://arxiv.org/abs/1011.0686
- **代码库**: https://github.com/jj-zhu/dagger
- **imitation库**: https://imitation.readthedocs.io/
- **MineRL比赛**: https://minerl.io/

---

## 📝 **总结**

### **DAgger的核心优势**

1. ✅ 解决分布偏移问题
2. ✅ 性能优于纯BC
3. ✅ 理论保证（线性误差增长）
4. ✅ 适合Minecraft等复杂任务

### **适合你的项目因为**

1. ✅ 你已有录制工具
2. ✅ 砍树任务适中（不太长）
3. ✅ 有时间做迭代标注
4. ✅ 预期效果显著（60% → 90%）

### **下一步**

1. 实现状态收集工具
2. 实现交互式标注工具
3. 运行第一轮DAgger
4. 评估效果决定是否继续

---

**推荐阅读顺序**:
1. 本文档（DAgger详解）
2. [`IMITATION_LEARNING_GUIDE.md`](IMITATION_LEARNING_GUIDE.md)（模仿学习概览）
3. [`IMITATION_LEARNING_ROADMAP.md`](../status/IMITATION_LEARNING_ROADMAP.md)（实施计划）

