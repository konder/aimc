# DAgger 采样策略优化与任务适配性分析

> **作者**: AIMC Team  
> **日期**: 2025-10-25  
> **主题**: 智能采样策略改进 + 序列任务适配性分析

---

## 📋 问题背景

在实际使用 DAgger 训练过程中，发现了两个关键问题：

### 问题1: 当前采样策略的局限性

**观察**：在观察失败的 episodes 时，发现：
- 有些 episode 前面的决策是**正确的**，只有最后几步错误
- 有些 episode **从一开始就有问题**
- 当前的"取最后N步"策略可能不够智能

**当前逻辑**：
```python
# label_states.py:193-203
if not success:
    # 失败episode: 标注最后N步（默认N=10）
    start_idx = max(0, len(states) - self.failure_window)
    for state_idx in range(start_idx, len(states)):
        states_to_label.append(...)
```

### 问题2: 任务适配性疑问

**任务特性**：砍树任务有明确的**动作序列依赖**：
1. 找到树（视觉搜索）
2. 向树的方向前进（导航）
3. 连续攻击（执行）

**疑问**：这种有序列依赖的任务适合 BC + DAgger 流程吗？

---

## 🔬 问题1: 采样策略优化

### 当前策略的问题

| 场景 | Episode轨迹示意 | 当前采样结果 | 问题分析 |
|------|---------------|-------------|---------|
| **场景A**: 前期正确，后期错误 | `[✓✓✓✓✓✓✓✓✗✗]` (80步正确 + 20步错误) | 采样最后10步，包含5步正确 + 5步错误 | ❌ 标注了很多**已经正确**的状态，浪费时间 |
| **场景B**: 从头到尾都错 | `[✗✗✗✗✗✗✗✗✗✗]` (100步全错) | 采样最后10步 | ❌ 错过了**开始的关键错误**决策点 |
| **场景C**: 中间段错误 | `[✓✓✓✗✗✗✓✓✓✗]` | 采样最后10步 | ❌ **中间的错误**被遗漏 |
| **场景D**: 多次试错 | `[✗✗✓✓✗✗✓✗✗✗]` | 采样最后10步 | ❌ 只看到最后的错误，未看到**反复试错**过程 |

### 根本原因

**"最后N步"假设的问题**：
1. ❌ **假设**: 失败是因为最后的决策错误
2. ✅ **现实**: 失败可能源于**任意时刻**的错误累积

**MineDojo 砍树任务的失败模式**：
- **早期失败**: 找不到树 → 方向错误 → 走远了
- **中期失败**: 找到树但导航失败 → 撞墙/绕圈
- **后期失败**: 接近树但没有持续攻击 → 时间耗尽

### 🎯 优化方案：基于奖励的智能采样

#### 核心思想

利用 **奖励序列** 来识别错误决策发生的时刻：

```python
# 砍树任务的奖励特征
rewards = [0, 0, 0, ..., 0, 1.0]  # 最后获得木头时有大奖励
              ↑           ↑
           无进展      成功时刻
```

**关键观察**：
- 奖励**停滞不前** → 策略没有进展（可能错误）
- 奖励**突然下降** → 做了错误决策
- 奖励**持续增长** → 策略在正确执行

#### 算法设计

**步骤1: 分析奖励轨迹**

```python
def analyze_episode_trajectory(episode_data):
    rewards = episode_data['rewards']
    
    # 1. 计算累积奖励
    cumulative_rewards = np.cumsum(rewards)
    
    # 2. 计算奖励变化率（一阶导数）
    reward_velocity = np.gradient(cumulative_rewards)
    
    # 3. 识别"奖励停滞/下降"区间
    error_threshold = np.percentile(reward_velocity, 25)
    error_mask = reward_velocity < error_threshold
    
    # 4. 识别关键转折点
    peaks, _ = find_peaks(-reward_velocity, distance=5)
    
    return {
        'error_regions': [...],      # 错误区间
        'critical_points': [...],    # 关键转折点
        'trajectory_type': '...'     # 轨迹类型
    }
```

**步骤2: 智能采样**

```python
def smart_sample_states_improved(episodes):
    for episode in episodes:
        if episode['success']:
            # 成功episode: 随机采样（保持原策略）
            sample_randomly(episode)
        else:
            # 失败episode: 基于分析结果采样
            analysis = analyze_episode_trajectory(episode)
            
            # 策略1: 在错误区间内采样
            for start, end in analysis['error_regions']:
                sample_uniformly(start, end)
            
            # 策略2: 在关键转折点周围密集采样
            for critical_point in analysis['critical_points']:
                sample_around(critical_point, window=3)
            
            # 策略3: 根据轨迹类型调整优先级
            if analysis['trajectory_type'] == 'early_error':
                # 早期错误 → 前1/3高优先级
                mark_high_priority(0, len(states) // 3)
```

#### 优势对比

| 维度 | 原策略（最后N步） | 改进策略（基于奖励） |
|------|-----------------|-------------------|
| **适应性** | 固定窗口，不考虑失败原因 | 动态识别错误发生时刻 |
| **覆盖率** | 只覆盖最后N步 | 覆盖所有错误区间 |
| **效率** | 可能标注很多正确状态 | 聚焦于真正需要纠正的状态 |
| **优先级** | 所有状态平等 | 根据错误类型设置优先级 |

### 实现与使用

**新文件**: `src/training/dagger/label_states_improved.py`

```bash
# 使用改进的采样策略
python src/training/dagger/label_states_improved.py \
    --states data/tasks/harvest_1_log/policy_states/iter_1 \
    --output data/tasks/harvest_1_log/expert_labels/iter_1.pkl \
    --smart-sampling \
    --failure-window 10
```

**可视化分析**：

```python
# 生成采样策略可视化
from label_states_improved import visualize_sampling_strategy

episode_data = load_episode(...)
sampled_indices = [...]
visualize_sampling_strategy(episode_data, sampled_indices)
```

---

## 🎮 问题2: 序列任务的适配性分析

### 任务特性分析

**砍树任务 (harvest_1_log) 的序列结构**：

```
阶段1: 搜索树
├── 视觉搜索（旋转视角）
├── 识别树的位置
└── 确定前进方向
       ↓
阶段2: 导航到树
├── 调整朝向
├── 前进 + 跳跃（避障）
└── 接近到攻击范围
       ↓
阶段3: 执行砍伐
├── 对准树
├── 连续攻击
└── 获得木头
```

**关键特征**：
1. **状态依赖**: 下一步依赖前面的状态（找到树后才能前进）
2. **长期规划**: 需要记住"树在哪里"
3. **重复执行**: 最后需要连续攻击多次

### BC + DAgger 的适配性评估

#### ✅ 适合的方面

| 特性 | BC/DAgger 能力 | 砍树任务契合度 |
|------|---------------|--------------|
| **状态-动作映射** | 学习 s → a 的映射 | ✅ 每个状态都有对应的正确动作 |
| **感知能力** | CNN 可以学习视觉特征 | ✅ 可以识别树的视觉特征 |
| **迭代改进** | DAgger 纠正分布偏移 | ✅ 失败状态可以被专家纠正 |
| **端到端训练** | 无需手动特征工程 | ✅ 直接从像素到动作 |

#### ⚠️ 挑战的方面

| 挑战 | BC/DAgger 局限 | 对砍树任务的影响 |
|------|---------------|----------------|
| **时序记忆** | MLP/CNN 无记忆机制 | ⚠️ 难以记住"树的位置" |
| **长期依赖** | 只看当前帧 | ⚠️ 可能忘记之前看到的树 |
| **序列规划** | 无显式规划 | ⚠️ 可能在各阶段间切换混乱 |

### 🎯 适配性结论

**✅ 砍树任务总体适合 BC + DAgger，但需要优化**

#### 原因分析

1. **任务序列相对简单**
   - 只有3个明确阶段
   - 每个阶段的视觉特征差异大（容易区分）
   - 状态转换清晰（找到树 vs 未找到树）

2. **视觉线索充足**
   - 树的特征明显（从像素可以判断）
   - 距离信息可以从视觉推断
   - 不需要复杂的空间记忆

3. **DAgger 的纠错能力**
   - 专家可以在关键阶段纠正错误
   - 迭代训练可以覆盖各种失败场景

#### 对比更复杂的任务

| 任务类型 | 序列复杂度 | BC+DAgger 适配性 | 示例 |
|---------|----------|----------------|------|
| **简单感知-动作** | 低 | ✅✅✅ 非常适合 | 躲避障碍 |
| **短序列任务** | 中 | ✅✅ 适合（当前任务）| **砍树** |
| **长序列任务** | 高 | ⚠️ 需要改进 | 建造房屋 |
| **开放探索** | 极高 | ❌ 不太适合 | 自由探索地图 |

### 🚀 优化建议

#### 建议1: 增加帧堆叠（Frame Stacking）

**问题**: 单帧无法提供运动信息

**解决**: 堆叠最近的 4 帧作为输入

```python
# 在 env_wrappers.py 中
from stable_baselines3.common.vec_env import VecFrameStack

env = make_minedojo_env(...)
env = VecFrameStack(env, n_stack=4)  # 堆叠4帧
```

**效果**:
- ✅ 可以感知物体运动方向
- ✅ 可以判断"正在接近树"还是"正在远离树"
- ✅ 提供时序上下文

#### 建议2: 使用 LSTM 策略（可选）

**问题**: CNN 无记忆能力

**解决**: 在策略网络中加入 LSTM 层

```python
# 在 train_bc.py 中
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMPolicy(ActorCriticPolicy):
    def __init__(self, ...):
        super().__init__(
            ...,
            features_extractor_class=CNNLSTMExtractor,
            lstm_hidden_size=256
        )
```

**效果**:
- ✅ 可以记住"之前看到树在左边"
- ✅ 可以规划"先转向，再前进"
- ⚠️ 训练更慢，需要更多数据

#### 建议3: 分阶段训练（推荐）

**思路**: 将任务分解，分别训练

```python
# 阶段1: 训练"找树"子策略
train_bc(
    task="find_tree",  # 自定义任务：奖励=转向树的方向
    episodes=50
)

# 阶段2: 训练"导航"子策略
train_bc(
    task="navigate_to_tree",  # 自定义任务：奖励=接近树
    episodes=50
)

# 阶段3: 训练"砍伐"子策略
train_bc(
    task="chop_tree",  # 自定义任务：奖励=攻击树
    episodes=50
)

# 阶段4: 端到端微调
train_bc(
    task="harvest_1_log",  # 完整任务
    episodes=100,
    init_from="merged_subpolicies"  # 从子策略初始化
)
```

**效果**:
- ✅ 每个阶段更容易学习
- ✅ 可以分别诊断问题
- ✅ 专家标注更聚焦

#### 建议4: 改进奖励函数（强烈推荐）

**问题**: MineDojo 的原始奖励太稀疏

```python
# 原始奖励
reward = 1.0 if got_wood else 0.0  # 只有最后一步有奖励
```

**解决**: 添加中间奖励（Reward Shaping）

```python
# 在 env_wrappers.py 中
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance_to_tree = None
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 中间奖励1: 接近树
        distance_to_tree = self.compute_distance_to_tree(obs)
        if self.prev_distance_to_tree is not None:
            approach_reward = (self.prev_distance_to_tree - distance_to_tree) * 0.01
            reward += approach_reward
        self.prev_distance_to_tree = distance_to_tree
        
        # 中间奖励2: 面向树
        facing_tree = self.is_facing_tree(obs)
        if facing_tree:
            reward += 0.001
        
        # 中间奖励3: 攻击树
        if action[5] == 3 and facing_tree:  # 攻击动作
            reward += 0.01
        
        return obs, reward, done, info
```

**效果**:
- ✅ 奖励变密集，更容易学习
- ✅ 智能采样策略更有效（可以识别错误时刻）
- ✅ 策略收敛更快

---

## 📊 实验对比

### 实验设计

**对照组**:
- 采样策略: 最后10步
- 策略架构: CNN + MLP
- 奖励函数: 原始稀疏奖励

**实验组**:
- 采样策略: **基于奖励的智能采样**
- 策略架构: CNN + MLP + **帧堆叠(4帧)**
- 奖励函数: **改进的密集奖励**

### 预期结果

| 指标 | 对照组 | 实验组 | 改进幅度 |
|------|-------|-------|---------|
| **BC基线成功率** | 20% | 35% | +75% |
| **DAgger第3轮成功率** | 45% | 70% | +55% |
| **标注时间/轮** | 30分钟 | 20分钟 | -33% |
| **收敛轮数** | 5-6轮 | 3-4轮 | -40% |

---

## 🎓 总结与建议

### 问题1: 采样策略优化

**✅ 推荐方案**: 使用基于奖励的智能采样

**实施步骤**:
1. 使用 `label_states_improved.py` 替代原版
2. 分析几个失败 episode，验证采样是否合理
3. 可视化采样结果，调整参数

**预期收益**:
- 标注效率提升 30%
- 标注质量提升（聚焦真正错误）
- 训练收敛更快

### 问题2: 任务适配性

**✅ 结论**: 砍树任务**适合** BC + DAgger 流程

**理由**:
- 序列相对简单（3阶段）
- 视觉线索充足
- DAgger 可以有效纠错

**✅ 推荐改进**（按优先级）:

1. **必做**: 添加帧堆叠（简单且有效）
   ```bash
   # 修改 env_wrappers.py
   env = VecFrameStack(env, n_stack=4)
   ```

2. **推荐**: 改进奖励函数（显著提升）
   ```python
   # 添加 RewardShapingWrapper
   ```

3. **可选**: 使用 LSTM 策略（如果有时间）

4. **高级**: 分阶段训练（适合深入研究）

### 不适合 BC + DAgger 的任务

❌ 以下任务不太适合：
- **超长序列任务**: 建造复杂建筑（需要几百步规划）
- **强记忆依赖**: 需要记住地图布局
- **多目标切换**: 同时管理多个子任务
- **开放探索**: 没有明确目标的自由探索

对于这些任务，考虑使用：
- **强化学习** (PPO/SAC): 更好的长期规划
- **层次化强化学习** (Options/HRL): 分层决策
- **基于模型的方法** (World Model): 显式建模环境

---

## 📚 相关资源

### 代码文件
- `src/training/dagger/label_states.py` - 原始采样策略
- `src/training/dagger/label_states_improved.py` - 改进的采样策略（新）
- `src/training/dagger/run_policy_collect_states.py` - 状态收集工具
- `src/envs/env_wrappers.py` - 环境包装器

### 文档
- [DAgger 完整指南](../guides/DAGGER_COMPREHENSIVE_GUIDE.md)
- [Label States 快捷键指南](../guides/LABEL_STATES_SHORTCUTS_GUIDE.md)
- [任务包装器指南](../guides/TASK_WRAPPERS_GUIDE.md)

### 论文参考
- **DAgger**: Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (2011)
- **Frame Stacking**: Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- **Reward Shaping**: Ng et al., "Policy Invariance Under Reward Transformations" (1999)

---

**最后更新**: 2025-10-25  
**版本**: v1.0  
**维护**: AIMC Team

