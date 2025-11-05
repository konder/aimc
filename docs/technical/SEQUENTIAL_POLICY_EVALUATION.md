# 序贯策略评估方法详解

> **核心问题**: 如何评估Minecraft这样的序贯决策任务？
> 
> 当模型动作和专家不一致时，轨迹会分叉（diverge），如何对比？

---

## 1. 问题的本质

### 1.1 你发现的问题

```
图像分类（简单）✓:
  
  输入: 一张猫的图片
  专家标注: "cat"
  模型预测: "dog"
  评估: 直接对比 → 错误 ❌
  
  ✅ 评估简单，一次性对比

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Minecraft策略（复杂）❓:
  
  t=0:
    专家看到: 初始森林场景
    专家动作: forward (向前走)
    
    模型看到: 相同的初始森林场景
    模型动作: turn_left (向左转) ← 和专家不一致！
  
  t=1:
    专家看到: 前方的树 🌲
    专家动作: attack (砍树)
    
    模型看到: 左边的草地 🌿 ← 不同的场景！
    模型动作: ???
    
  问题: 
    ❌ 场景已经不同了，如何对比？
    ❌ 后续轨迹完全分叉，无法逐帧对比
```

### 1.2 这个问题的专业术语

```
Distribution Shift / Compounding Error:
  
  专家轨迹分布: p_expert(s_t | s_0)
  模型轨迹分布: p_model(s_t | s_0)
  
  即使在t=0做了不同的决策
  → t=1状态分布已经不同
  → t=2差异更大
  → t=T完全diverge
  
  这是序贯决策中的经典问题
```

---

## 2. 三种评估方式

### 2.1 Open-Loop评估（不可行）❌

**定义**: 强制模型看专家看到的场景，对比每一步的动作

```python
def open_loop_evaluation(model, expert_trajectory):
    """强制使用专家的状态序列"""
    
    errors = 0
    for t in range(len(expert_trajectory)):
        # 强制使用专家看到的状态
        expert_state = expert_trajectory.states[t]
        expert_action = expert_trajectory.actions[t]
        
        # 模型在专家状态下预测
        model_action = model.predict(expert_state)
        
        # 对比动作
        if model_action != expert_action:
            errors += 1
    
    accuracy = 1 - errors / len(expert_trajectory)
    return accuracy

问题:
  ❌ 模型从未在"犯错后的状态"上训练过
  ❌ 不反映真实运行时的性能
  ❌ 无法评估模型的恢复能力
  
  类比: 就像训练驾驶员，只让他在完美道路上开，
       从不让他处理偏离路线后的情况
```

### 2.2 Closed-Loop评估（标准方法）✅

**定义**: 让模型完整运行，评估最终结果

```python
def closed_loop_evaluation(model, expert_trajectory):
    """让模型完整运行，对比结果"""
    
    # 1. 从相同初始状态开始
    env.reset(seed=expert_trajectory.seed)
    initial_state = expert_trajectory.states[0]
    env.set_state(initial_state)
    
    # 2. 模型完整运行（允许diverge）
    model_trajectory = []
    state = initial_state
    
    for t in range(expert_trajectory.max_steps):
        # 模型自主决策
        action = model.predict(state)
        
        # 执行动作，获得新状态
        next_state, reward, done, info = env.step(action)
        
        model_trajectory.append({
            'state': state,
            'action': action,
            'next_state': next_state
        })
        
        state = next_state
        
        if done:
            break
    
    # 3. 评估最终结果（不是逐帧对比）
    metrics = {
        # 任务完成
        'task_success': check_task_completion(model_trajectory),
        
        # 最终状态对比
        'final_state_similarity': compare_final_states(
            model_trajectory[-1]['state'],
            expert_trajectory.states[-1]
        ),
        
        # 效率
        'efficiency': len(expert_trajectory) / len(model_trajectory),
        
        # 累积奖励
        'total_reward': sum([t['reward'] for t in model_trajectory])
    }
    
    return metrics

优点:
  ✅ 反映真实性能
  ✅ 评估模型处理diverge的能力
  ✅ 更有实际意义
```

### 2.3 STEVE-1实际使用的方法

**混合评估**: Closed-loop + 多个指标

```python
# STEVE-1的评估流程

def steve1_evaluation(model, text_video_pairs):
    """
    STEVE-1论文中的评估方法
    """
    
    results = []
    
    for (text, expert_video) in text_video_pairs:
        # 1. 准备
        task_instruction = text  # "chop tree"
        expert_traj = expert_video.trajectory
        initial_state = expert_traj.states[0]
        
        # 2. 模型运行（Closed-loop）
        env.reset()
        env.set_state(initial_state)
        
        text_embed = mineclip.encode_text(task_instruction)
        
        model_traj = []
        obs = initial_state
        
        for t in range(1000):  # 最大步数
            action = model(obs, text_embed)
            obs, reward, done, info = env.step(action)
            
            model_traj.append({
                'obs': obs,
                'action': action,
                'reward': reward
            })
            
            if done or check_task_done(obs, task_instruction):
                break
        
        # 3. 评估（重点：不是逐帧对比动作！）
        metrics = evaluate_trajectory(
            model_traj=model_traj,
            expert_traj=expert_traj,
            task=task_instruction
        )
        
        results.append(metrics)
    
    return aggregate_results(results)


def evaluate_trajectory(model_traj, expert_traj, task):
    """
    多维度评估（不逐帧对比）
    """
    
    # 指标1: 任务成功率
    task_success = check_task_completion(model_traj, task)
    # 例如: "chop tree" → 检查是否获得了木头
    
    # 指标2: 效率（用时对比）
    expert_steps = len(expert_traj)
    model_steps = len(model_traj)
    efficiency = expert_steps / model_steps if model_steps > 0 else 0
    
    # 指标3: 最终状态相似度
    # 只对比最后一帧，不逐帧对比
    if task_success:
        final_similarity = compare_images(
            model_traj[-1]['obs'],
            expert_traj[-1].obs
        )
    else:
        final_similarity = 0
    
    # 指标4: 累积奖励
    total_reward = sum([t['reward'] for t in model_traj])
    
    # 指标5: 中间里程碑
    # 例如: "chop tree"任务的里程碑
    # [找到树, 接近树, 开始砍, 获得木头]
    milestones = check_milestones(model_traj, task)
    
    return {
        'success': task_success,          # 最重要
        'efficiency': efficiency,
        'final_similarity': final_similarity,
        'total_reward': total_reward,
        'milestones_reached': milestones
    }
```

---

## 3. 为什么不逐帧对比动作？

### 3.1 逐帧对比的问题

```
任务: "chop tree"

专家轨迹:
  t=0: 在位置A，向前走
  t=1: 在位置B，向前走
  t=2: 在位置C，转向树
  t=3: 在位置D，开始砍
  ...
  t=50: 获得木头 ✓

模型轨迹（逐帧对比）:
  t=0: 在位置A，向前走 ✓（和专家一致）
  t=1: 在位置B，向右走 ❌（和专家不一致）
  
  此时已经diverge，后续无法对比！
  
  但是...
  
  t=2: 在位置E（不同位置），转向树
  t=3: 在位置F，开始砍
  ...
  t=48: 获得木头 ✓（更快完成！）

结果:
  逐帧对比准确率: 1/50 = 2% ❌
  但实际上任务成功完成，且更高效！ ✓
  
  → 逐帧对比误导性强
```

### 3.2 任务导向评估更合理

```
关键洞察:
  我们不关心模型是否完全模仿专家的每一步
  我们关心模型是否能完成任务！

评估重点:
  ❌ 模型是否走了和专家一样的路径？
  ✅ 模型是否砍到了树？
  
  ❌ 模型每一步动作是否和专家一致？
  ✅ 模型最终是否达到目标？
  
  ❌ 轨迹是否完全相同？
  ✅ 效率是否合理？
```

---

## 4. 具体评估指标

### 4.1 主要指标：任务成功率

```python
def check_task_completion(trajectory, task):
    """
    检查任务是否完成（最重要的指标）
    """
    
    if task == "chop tree":
        # 检查是否获得了木头
        final_inventory = trajectory[-1]['obs']['inventory']
        return final_inventory.get('log', 0) > 0
    
    elif task == "hunt cow":
        # 检查是否获得了牛肉/皮革
        final_inventory = trajectory[-1]['obs']['inventory']
        return (final_inventory.get('beef', 0) > 0 or
                final_inventory.get('leather', 0) > 0)
    
    elif task == "swim to shore":
        # 检查是否上岸（不在水中）
        final_state = trajectory[-1]['obs']['pov']
        return not is_in_water(final_state)
    
    # ...更多任务

# 评估
success_rate = sum([
    check_task_completion(traj, task) 
    for traj, task in test_cases
]) / len(test_cases)

print(f"任务成功率: {success_rate:.1%}")
# 例如: 任务成功率: 78.5%
```

### 4.2 次要指标：效率

```python
def compute_efficiency(model_traj, expert_traj):
    """
    对比完成任务的效率
    """
    
    expert_steps = len(expert_traj)
    model_steps = len(model_traj)
    
    # 相对效率
    efficiency = expert_steps / model_steps
    
    # 效率评级
    if efficiency >= 0.9:
        rating = "excellent"  # 和专家差不多
    elif efficiency >= 0.7:
        rating = "good"       # 稍慢但可接受
    elif efficiency >= 0.5:
        rating = "fair"       # 较慢
    else:
        rating = "poor"       # 太慢
    
    return {
        'efficiency': efficiency,
        'rating': rating,
        'expert_steps': expert_steps,
        'model_steps': model_steps
    }

# 示例
# 专家用50步完成砍树
# 模型用70步完成
# 效率 = 50/70 = 71.4% → "good"
```

### 4.3 辅助指标：里程碑检测

```python
def check_milestones(trajectory, task):
    """
    检查是否达到了关键里程碑
    （部分评估，不需要完全一致）
    """
    
    if task == "chop tree":
        milestones = {
            'found_tree': False,
            'approached_tree': False,
            'started_chopping': False,
            'got_wood': False
        }
        
        for t, frame in enumerate(trajectory):
            obs = frame['obs']
            action = frame['action']
            inventory = obs.get('inventory', {})
            
            # 检测树在视野中
            if detect_tree_in_view(obs['pov']):
                milestones['found_tree'] = True
            
            # 检测接近树
            if is_near_tree(obs):
                milestones['approached_tree'] = True
            
            # 检测攻击动作
            if action['attack'] == 1:
                milestones['started_chopping'] = True
            
            # 检测获得木头
            if inventory.get('log', 0) > 0:
                milestones['got_wood'] = True
        
        # 计算完成的里程碑比例
        completion = sum(milestones.values()) / len(milestones)
        
        return {
            'milestones': milestones,
            'completion_rate': completion
        }

# 即使最终失败，也能知道卡在哪一步
# 例如: 找到了树，但没能成功砍下来
```

### 4.4 视觉相似度（可选）

```python
def compare_visual_similarity(model_traj, expert_traj):
    """
    对比视觉上的相似度
    （不是逐帧，而是采样关键帧）
    """
    
    # 采样关键时间点（开始、中间、结束）
    sample_points = [0, len(expert_traj) // 2, len(expert_traj) - 1]
    
    similarities = []
    for t in sample_points:
        if t < len(model_traj):
            # 使用预训练的图像编码器对比
            expert_img = expert_traj[t].obs['pov']
            model_img = model_traj[t].obs['pov']
            
            # 计算特征相似度
            sim = compute_image_similarity(expert_img, model_img)
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    return avg_similarity

# 这个指标不是很重要，作为参考
```

---

## 5. STEVE-1论文中的实际评估

### 5.1 论文中报告的指标

```
STEVE-1论文 Table 1 的评估指标:

任务                    成功率    平均步数    专家步数
─────────────────────────────────────────────────────
Find Cave               78%       245        180
Hunt Cow                85%       156        120
Chop Tree               92%       68         50
Swim to Shore           88%       95         75
...

评估方式:
  1. 成功率: Closed-loop，检查任务是否完成
  2. 平均步数: 模型完成任务的平均步数
  3. 对比专家: 和专家演示的步数对比
  
注意:
  ✅ 没有逐帧动作准确率
  ✅ 关注最终结果
  ✅ 关注效率
```

### 5.2 完整评估代码示例

```python
class STEVE1Evaluator:
    def __init__(self, test_dataset):
        """
        test_dataset: Text-Video Pair数据集
        """
        self.test_cases = test_dataset
    
    def evaluate(self, model):
        results = []
        
        for test_case in self.test_cases:
            result = self.evaluate_single(model, test_case)
            results.append(result)
        
        return self.aggregate(results)
    
    def evaluate_single(self, model, test_case):
        """
        评估单个测试case
        """
        # 1. 准备
        text = test_case.text
        expert_video = test_case.expert_trajectory
        init_state = expert_video.initial_state
        
        # 2. 运行模型（Closed-loop）
        model_traj = self.run_model(
            model=model,
            text=text,
            init_state=init_state,
            max_steps=1000
        )
        
        # 3. 评估
        return {
            'text': text,
            'success': self.check_success(model_traj, text),
            'steps': len(model_traj),
            'expert_steps': len(expert_video),
            'efficiency': len(expert_video) / len(model_traj),
            'milestones': self.check_milestones(model_traj, text),
        }
    
    def run_model(self, model, text, init_state, max_steps):
        """
        Closed-loop运行
        """
        env = make_env()
        env.set_state(init_state)
        
        text_embed = mineclip.encode_text(text)
        
        trajectory = []
        obs = env.get_obs()
        
        for t in range(max_steps):
            # 模型预测
            action = model(obs, text_embed)
            
            # 执行
            next_obs, reward, done, info = env.step(action)
            
            trajectory.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'done': done
            })
            
            obs = next_obs
            
            # 提前终止条件
            if done or self.check_success(trajectory, text):
                break
        
        env.close()
        return trajectory
    
    def aggregate(self, results):
        """
        汇总所有测试case的结果
        """
        total = len(results)
        
        return {
            'total_cases': total,
            'success_rate': sum([r['success'] for r in results]) / total,
            'avg_steps': np.mean([r['steps'] for r in results]),
            'avg_efficiency': np.mean([r['efficiency'] for r in results]),
            'by_task': self.group_by_task(results)
        }
```

---

## 6. 总结

### 6.1 回答你的问题

```
你的疑问:
  "当模型第一步动作和专家不一致，
   后续状态就不同了，如何对比？"

答案:
  不对比逐帧动作！ ❌
  对比最终结果！ ✅

具体做法:
  1. Closed-loop运行
     让模型完整执行，允许轨迹diverge
  
  2. 评估任务完成
     检查最终是否达到目标
     例如: "chop tree" → 是否获得木头？
  
  3. 对比效率
     模型用了多少步？和专家对比
  
  4. 检查里程碑
     关键步骤是否达到？
  
  5. 不逐帧对比动作
     因为这没有意义（会diverge）
```

### 6.2 关键洞察

```
序贯决策评估的核心原则:

1. 目标导向 > 行为模仿
   ✅ 完成任务最重要
   ❌ 不必完全模仿专家路径

2. Closed-loop > Open-loop
   ✅ 让模型真实运行
   ❌ 不强制使用专家状态

3. 结果评估 > 过程评估
   ✅ 最终是否成功
   ❌ 不逐帧对比动作

4. 多维度指标
   - 成功率（主要）
   - 效率
   - 里程碑
   - 视觉相似度（次要）
```

### 6.3 实践建议

```
评估你自己的STEVE-1模型:

第一步: 定义任务成功标准
  - "chop tree" → 获得木头
  - "hunt cow" → 获得牛肉/皮革
  - "build house" → 放置了方块

第二步: Closed-loop运行
  - 从初始状态开始
  - 让模型完整执行
  - 记录轨迹

第三步: 检查成功率
  - 是否达到目标？
  - 计算成功率

第四步: 分析效率
  - 用了多少步？
  - 和专家对比

第五步: 错误分析
  - 失败case卡在哪里？
  - 哪些里程碑没达到？
```

---

## 7. 代码实现参考

```bash
# 项目中的评估实现

src/training/steve1/run_agent/
├── interactive_run_custom_text_prompt.py  # 交互式评估
└── programmatic_eval.py                   # 自动化评估

# 使用方法
cd src/training/steve1
bash run_agent/interactive_run_custom_text_prompt.sh

# 输入文本指令，观察模型执行
# 手动判断是否成功
```

---

**关键要点**:
- ✅ 不逐帧对比动作（会diverge）
- ✅ 使用Closed-loop评估
- ✅ 关注任务成功率和效率
- ✅ 允许模型用不同路径达到目标

---

## 8. 专家数据在评估中的具体作用

### 8.1 用户的精准理解 ✅

```
用户提问:
  "那在评估STEVE-1时，只依赖了专家录制的第一帧数据
   作为相同的初始状态吧？"

答案: 完全正确！ ✅
```

### 8.2 专家数据的三个作用

```
Text-Video Pair中的专家数据包含:
  ├─ 文本描述: "chop tree"
  ├─ 初始状态: t=0的环境状态（第一帧）
  └─ 完整轨迹: t=0到t=T的所有帧和动作

在评估时的使用:
┌────────────────────────────────────────────────────────┐
│ 1. 初始状态 (t=0) → 用于环境初始化  ⭐⭐⭐⭐⭐        │
│    - 模型和专家从相同位置开始                          │
│    - 确保公平对比                                      │
│                                                        │
│ 2. 文本描述 → 作为模型输入  ⭐⭐⭐⭐⭐                  │
│    - "chop tree"编码为MineCLIP嵌入                    │
│    - 告诉模型要做什么                                  │
│                                                        │
│ 3. 完整轨迹 (t=1到t=T) → 作为参考标准  ⭐⭐⭐         │
│    - 不用于逐帧对比                                    │
│    - 用于计算专家效率（步数）                          │
│    - 用于对比最终状态                                  │
└────────────────────────────────────────────────────────┘

关键: 只有初始状态(t=0)真正"强制"模型使用
     其余轨迹(t>0)只作为参考标准，不强制对齐
```

### 8.3 详细的评估流程

```python
# 完整的评估流程示例

def evaluate_with_expert_data(model, expert_data):
    """
    展示专家数据的具体使用方式
    """
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 从专家数据中提取
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    text = expert_data.text                    # "chop tree"
    initial_state = expert_data.states[0]      # 只用第一帧！⭐
    expert_trajectory = expert_data.states     # 完整轨迹（参考）
    expert_actions = expert_data.actions       # 完整动作（不用于对比）
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 初始化环境：使用专家的初始状态
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    env = MinecraftEnv()
    env.set_state(initial_state)  # ⭐ 唯一使用专家数据的地方
    
    # 验证：确保初始状态相同
    assert env.get_state() == initial_state
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 模型运行：完全自主，不再参考专家
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    text_embed = mineclip.encode_text(text)
    
    model_trajectory = []
    obs = env.get_obs()
    
    for t in range(1000):
        # 模型自己决策（不看expert_actions！）
        action = model(obs, text_embed)  # ⭐ 完全自主
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        model_trajectory.append({
            'obs': obs,
            'action': action,
            'reward': reward
        })
        
        # 检查任务完成
        if check_task_done(obs, text):
            break
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 评估：对比结果（不对比过程）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    return {
        # 任务成功
        'success': check_success(model_trajectory, text),
        
        # 效率对比（用专家步数作为参考）
        'model_steps': len(model_trajectory),
        'expert_steps': len(expert_trajectory),  # ⭐ 参考标准
        'efficiency': len(expert_trajectory) / len(model_trajectory),
        
        # 最终状态对比（可选）
        'final_state_similarity': compare_states(
            model_trajectory[-1]['obs'],
            expert_trajectory[-1]  # ⭐ 参考标准
        )
    }
```

### 8.4 可视化对比

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
专家轨迹（Text-Video Pair数据）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=0:  🌲🌲  [位置A]  ← 初始状态（⭐用于环境初始化）
      👤
      
t=10: 🌲🌲  [位置B]  ← 参考轨迹（不强制模型跟随）
        👤
      
t=20: 🌲🌲  [位置C]  ← 参考轨迹
          👤
      
t=50: 🌲🪓  [完成]   ← 参考标准（对比效率和最终状态）
      👤 (获得木头)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模型评估
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

t=0:  🌲🌲  [位置A]  ← 从专家相同初始状态开始 ⭐
      👤
      
t=10: 🌲🌲  [位置D]  ← 模型自主决策（可能不同路径）
    👤
      
t=30: 🌲🌲  [位置E]  ← 模型自主决策（继续diverge）
  👤
      
t=70: 🌲🪓  [完成]   ← 对比结果：成功 ✓，但用了70步
  👤 (获得木头)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
评估结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

初始状态:  相同 ✓ (都从位置A开始)
过程路径:  不同 (专家B→C, 模型D→E) → 不对比 ✓
最终结果:  都成功 ✓ (都获得木头)
效率对比:  专家50步，模型70步 → 效率71.4%
```

### 8.5 为什么只用第一帧？

```
原因1: 公平性
  ✅ 确保相同起点
  ✅ 消除随机性影响
  ✅ 可重现评估

原因2: 实用性
  ✅ 真实场景中，用户给指令时环境是未知的
  ✅ 模型需要从任意状态开始执行
  ✅ 只需要相同起点来消除随机变量

原因3: 技术原因
  ✅ 一旦diverge，强制对齐没有意义
  ✅ 训练时模型也是自主运行
  ✅ 评估应该反映真实使用场景
```

### 8.6 初始状态包含什么？

```python
# 初始状态的详细内容

initial_state = {
    # 玩家状态
    'position': (x, y, z),           # 玩家位置
    'yaw': 180,                       # 朝向角度
    'pitch': 0,                       # 俯仰角度
    'health': 20,                     # 生命值
    'food': 20,                       # 饱食度
    
    # 物品栏
    'inventory': {
        'log': 0,
        'planks': 0,
        # ...
    },
    
    # 环境状态
    'world_seed': 12345,              # 世界种子
    'time_of_day': 1000,              # 游戏时间
    'weather': 'clear',               # 天气
    
    # 视觉观测（第一帧图像）
    'pov': np.array([...]),           # 第一人称视角图像
}

# 设置环境到这个状态
env.set_state(initial_state)

# 之后模型看到的第一帧就和专家完全相同
```

### 8.7 常见误解

```
误解1 ❌:
  "需要专家的所有帧来引导模型"
  
  事实 ✅:
  只需要第一帧初始化环境
  之后模型完全自主

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误解2 ❌:
  "专家轨迹用于逐帧监督"
  
  事实 ✅:
  专家轨迹只用于计算参考指标
  - 专家用了多少步？
  - 专家最终状态如何？
  不用于引导模型

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误解3 ❌:
  "评估时模型需要看专家的中间帧"
  
  事实 ✅:
  模型只看自己执行后的观测
  不看专家的后续帧

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误解4 ❌:
  "Text-Video Pair的视频全部都要用"
  
  事实 ✅:
  视频的第1帧: 用于环境初始化 ⭐
  视频的第2-T帧: 只作为参考标准
```

### 8.8 代码验证

```python
# 验证初始状态的使用

def verify_initial_state_usage():
    """
    验证评估只使用第一帧
    """
    
    expert_data = load_text_video_pair()
    
    # 提取初始状态
    initial_state = expert_data.frames[0]
    
    # 初始化环境
    env = MinecraftEnv()
    env.set_state(initial_state)
    
    # 验证：模型看到的第一帧和专家相同
    model_first_obs = env.get_obs()['pov']
    expert_first_obs = expert_data.frames[0]['pov']
    
    assert np.allclose(model_first_obs, expert_first_obs)
    print("✓ 初始状态相同")
    
    # 执行一步
    action = model.predict(model_first_obs)
    env.step(action)
    
    # 第二帧：模型和专家可能已经不同
    model_second_obs = env.get_obs()['pov']
    expert_second_obs = expert_data.frames[1]['pov']
    
    # 很可能不同（允许diverge）
    if not np.allclose(model_second_obs, expert_second_obs):
        print("✓ 第二帧开始diverge（正常现象）")
    
    # 关键：不需要对齐第二帧及之后
    # 只要最终完成任务即可
```

### 8.9 实际评估示例

```
具体案例: "chop tree"任务

专家数据:
  - 文本: "chop tree"
  - 初始帧: 玩家在森林边缘，面向一棵橡树
  - 完整视频: 50帧，最后获得木头
  
评估过程:
  
  Step 1: 环境初始化
    env.set_state(expert_data.frames[0])  ⭐
    → 模型也在森林边缘，面向相同的树
  
  Step 2: 模型运行
    text_embed = encode("chop tree")
    for t in range(1000):
        action = model(obs, text_embed)
        obs = env.step(action)
        
    → 模型自己探索，可能走不同路径
  
  Step 3: 评估结果
    模型用了70步，获得了木头 ✓
    专家用了50步
    
    结果:
      - 成功率: 100% (任务完成)
      - 效率: 50/70 = 71.4%
      - 结论: 成功但稍慢

关键:
  只在Step 1使用了专家的第一帧
  Step 2和3完全不依赖专家后续帧
```

### 8.10 总结

```
用户的理解 ✅:
  "评估时只依赖专家录制的第一帧数据作为初始状态"

补充细节:
  
  专家数据用途:
    ├─ 第一帧 (t=0):
    │   └─ 用于环境初始化 ⭐⭐⭐⭐⭐
    │       确保模型和专家从相同起点开始
    │
    ├─ 完整轨迹 (t=0 to T):
    │   └─ 作为参考标准 ⭐⭐⭐
    │       - 计算专家步数
    │       - 对比最终状态
    │       - 不用于逐帧对比
    │
    └─ 文本描述:
        └─ 模型输入 ⭐⭐⭐⭐⭐
            告诉模型任务目标

  关键原则:
    ✅ 相同起点（公平）
    ✅ 自主运行（真实）
    ✅ 结果对比（有效）
```

---

**最后更新**: 2025-11-05

