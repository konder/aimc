# Prior 模型评估框架设计

**创建日期**: 2025-12-01  
**目的**: 为 STEVE-1 的 Prior 模型 p(z_goal|y) 设计准确的评估方法  
**目标**: 建立基线指标，用于未来优化和再训练

---

## 🎯 核心挑战

### Prior 模型的特殊性

Prior 是一个**中间模型**，不直接产生最终输出：

```
指令 y → Prior → z_goal → Policy → 动作序列 τ
         ^^^^^^
      我们要评估这个
```

**问题**：
- ❌ 无法直接用任务成功率评估（那是 Policy 的表现）
- ❌ 无法直接观察 z_goal 的"好坏"（它是 512 维向量）
- ❌ 缺少 ground truth（没有"正确的 z_goal"）

**目前的尝试**：
- 用成功画面的视觉嵌入作为 ground truth
- 问题：discriminability = 0.12（但系统工作正常）

---

## 📊 Prior 评估的核心问题

### 问题：什么是"好的" Prior？

一个好的 Prior 应该：

1. **语义准确性**: 能够捕捉指令的语义
2. **目标合理性**: 输出的 z_goal 应该代表"成功状态"
3. **可区分性**: 不同任务应该有不同的 z_goal
4. **语义鲁棒性**: 相似表述应该产生相似的 z_goal
5. **可控性**: z_goal 应该能有效指导 Policy

**关键洞察**：
- ✅ 指标 1,2,4 - 当前方法可以评估
- ⚠️ 指标 3 - discriminability 的解释需要重新思考
- ❌ 指标 5 - **当前方法缺失！这是最重要的！**

---

## 🔬 新的评估框架

### 核心思想：多维度评估

**不要只看 Prior 本身，而要看 Prior 在整个系统中的作用！**

```
维度 1: Prior 内在质量（Intrinsic Quality）
  → 测试 Prior 自身的特性
  
维度 2: Prior 输出质量（Output Quality）
  → 测试 z_goal 与真实目标的对齐
  
维度 3: Prior 可控性（Controllability）
  → 测试 z_goal 是否能有效指导 Policy
  
维度 4: 端到端质量（End-to-End Quality）
  → 测试 Prior + Policy 组合的表现
```

---

## 📐 维度 1: Prior 内在质量

### 指标 1.1: 输出稳定性（Consistency）

**已实现** ✅

```python
# 同一指令多次采样，输出应该接近（如果是确定性）
consistency = compute_consistency(instruction, n_samples=10)

# 期望值: > 0.95 (高稳定性)
```

**解释**：
- Prior 是 VAE，推理时通常用均值（确定性）
- 高一致性（0.999）说明 Prior 输出稳定

---

### 指标 1.2: 语义鲁棒性（Semantic Robustness）

**已实现** ✅

```python
# 同一任务的不同表述应该产生相似的 z_goal
variants = ["dig dirt", "get dirt", "collect dirt"]
semantic_robustness = compute_semantic_robustness(variants)

# 期望值: > 0.85 (高鲁棒性)
```

**你的结果**: 0.9685 ✅ 优秀！

---

### 指标 1.3: 输出多样性（Output Diversity）

**已实现** ✅

```python
# 不同任务应该有不同的输出
all_z_goals = [get_prior_embed(task) for task in all_tasks]
variance = np.var(all_z_goals, axis=0).mean()

# 期望值: > 0.0001 (有足够变异)
```

**你的结果**: 0.097 ✅ 足够多样！

---

### 指标 1.4: 任务区分度（Task Discriminability）

**已实现，但需要重新解释** ⚠️

```python
discriminability = 1 - mean_inter_task_similarity

# 你的结果: 0.12 (低)
```

**重新解释**：

这个指标**不应该单独用来判断 Prior 好坏**！

原因：
1. MineCLIP 文本嵌入本身区分度就低（1.3%）
2. Prior 不能创造不存在的差异
3. CFG 可以放大微小差异

**建议**：
- 对比 MineCLIP 文本嵌入的区分度
- 计算 Prior 的"区分度保持率"

```python
text_discriminability = compute_text_discriminability(texts)
prior_discriminability = compute_prior_discriminability(priors)

preservation_rate = prior_discriminability / text_discriminability

# 如果 > 0.8: Prior 保持了输入的区分度 ✅
# 如果 < 0.5: Prior 降低了区分度 ❌
```

---

## 📐 维度 2: Prior 输出质量 ⭐ 关键

### 指标 2.1: 目标对齐度（Goal Alignment）

**当前实现有问题** ⚠️

**问题**：
```python
# 当前：先平均后计算
z_visual_mean = np.mean(success_visual_embeds, axis=0)
goal_accuracy = 1 - cosine(z_goal, z_visual_mean)
```

**改进**：
```python
# 方案 A: 先计算后平均（推荐）
similarities = [1 - cosine(z_goal, z_v) for z_v in success_visual_embeds]
goal_accuracy_mean = np.mean(similarities)
goal_accuracy_std = np.std(similarities)

# 方案 B: 使用 MineCLIP 的 forward_reward_head
goal_accuracy = mineclip.forward_reward_head(z_goal, z_visuals).mean()
```

---

### 指标 2.2: 与文本嵌入的对比

**新指标** ⭐

**核心思想**：对比 Prior 和直接文本嵌入的效果

```python
# Prior 路径
z_prior = get_prior_embed(instruction, mineclip, prior, device)
sim_prior = compute_similarity(z_prior, z_visual_mean)

# 直接文本路径
z_text = mineclip.encode_text([instruction])[0]
sim_text = compute_similarity(z_text, z_visual_mean)

# Prior 增益
prior_gain = sim_prior - sim_text

# 期望:
# prior_gain > 0.05: Prior 有正向作用 ✅
# prior_gain < 0: Prior 在拖后腿 ❌
```

**这个指标可以直接回答**: Prior 是否比直接用文本嵌入更好？

---

### 指标 2.3: 跨模态一致性

**新指标** ⭐

```python
# 测试 Prior 输出是否真的在"视觉空间"

# 1. 收集真实视觉嵌入的分布
visual_embeds = all_success_visual_embeds  # (N, 512)
visual_mean = visual_embeds.mean(axis=0)
visual_std = visual_embeds.std(axis=0)

# 2. Prior 输出的分布
prior_embeds = [get_prior_embed(task) for task in tasks]  # (M, 512)
prior_mean = np.array(prior_embeds).mean(axis=0)
prior_std = np.array(prior_embeds).std(axis=0)

# 3. 分布相似度（KL 散度或 Wasserstein 距离）
from scipy.stats import wasserstein_distance

cross_modal_consistency = []
for dim in range(512):
    dist = wasserstein_distance(
        visual_embeds[:, dim],
        np.array(prior_embeds)[:, dim]
    )
    cross_modal_consistency.append(dist)

consistency_score = 1 / (1 + np.mean(cross_modal_consistency))

# 期望: > 0.5 (Prior 输出确实在视觉空间)
```

---

## 📐 维度 3: Prior 可控性 ⭐⭐⭐ 最关键！

### 当前评估的盲点

**我们一直在问**：
- Prior 输出和真实画面有多相似？
- 不同任务的 Prior 输出有多大差异？

**但我们应该问**：
- **Prior 输出能否有效指导 Policy 完成任务？**

这才是 Prior 的终极目标！

---

### 指标 3.1: Policy 可控性（Policy Controllability）⭐⭐⭐⭐⭐

**核心思想**：直接测试 Prior 输出对 Policy 的控制效果

```python
def evaluate_prior_controllability(
    task_id,
    instruction,
    policy,
    mineclip,
    prior,
    env,
    n_trials=10
):
    """
    测试 Prior 输出能否有效指导 Policy
    
    关键：不看 z_goal 的数值，而看 Policy 的表现
    """
    # 1. 获取 Prior 输出
    z_goal = get_prior_embed(instruction, mineclip, prior, device)
    
    # 2. 用这个 z_goal 运行 Policy
    success_rate = 0
    for trial in range(n_trials):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = policy.get_action(obs, z_goal)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if info.get('success', False):
                success_rate += 1
                break
    
    success_rate /= n_trials
    
    return success_rate
```

**这个指标直接回答**：
- Prior 输出的 z_goal 是否能让 Policy 完成任务？
- 这是最终的评判标准！

---

### 指标 3.2: CFG 敏感度分析

**测试 CFG 对 Prior 效果的影响**

```python
def evaluate_cfg_sensitivity(task, instruction):
    """测试不同 CFG scale 下的表现"""
    cfg_scales = [0, 1, 3, 6, 9, 12]
    results = []
    
    for scale in cfg_scales:
        agent.reset(cond_scale=scale)
        success_rate = evaluate_task(task, instruction)
        results.append({
            'cfg_scale': scale,
            'success_rate': success_rate
        })
    
    return results
```

**预期**：
- 如果 λ=0 性能很差，λ=6 性能好 → Prior 需要 CFG 补偿
- 如果各个 λ 性能接近 → Prior 本身质量高

---

### 指标 3.3: Prior vs 直接文本对比实验 ⭐⭐⭐

**核心实验**：

```python
def compare_prior_vs_text(task, instruction):
    """对比使用 Prior vs 直接使用文本嵌入"""
    
    # 方法 A: 使用 Prior（STEVE-1 标准方式）
    z_goal_prior = get_prior_embed(instruction, mineclip, prior, device)
    success_rate_prior = run_trials(task, z_goal_prior, n_trials=10)
    
    # 方法 B: 直接使用文本嵌入（跳过 Prior）
    z_goal_text = mineclip.encode_text([instruction])[0]
    success_rate_text = run_trials(task, z_goal_text, n_trials=10)
    
    # Prior 增益
    prior_gain = success_rate_prior - success_rate_text
    
    return {
        'success_rate_prior': success_rate_prior,
        'success_rate_text': success_rate_text,
        'prior_gain': prior_gain,
    }
```

**判断标准**：
- `prior_gain > 0.1`: ✅ Prior 显著改善（值得使用）
- `prior_gain 0-0.1`: ⚠️ Prior 略有改善
- `prior_gain < 0`: ❌ Prior 在拖后腿（考虑跳过）

**这是最直接的 Prior 质量指标！**

---

## 📐 维度 4: 端到端质量

### 指标 4.1: 任务成功率（Task Success Rate）

```python
# 标准的端到端评估
success_rate = evaluate_task(task_id, instruction, n_trials=10)
```

**这个已经在你的框架中** ✅

但需要明确：这是 **Prior + Policy** 的联合表现，不能单独归因于 Prior。

---

## 🎯 完整的 Prior 评估方案

### 方案设计

```
┌────────────────────────────────────────────────┐
│        Prior 模型评估框架（四维度）              │
├────────────────────────────────────────────────┤
│                                                │
│  [维度 1] 内在质量 (Intrinsic)                  │
│    • 输出稳定性 (consistency)                   │
│    • 语义鲁棒性 (semantic_robustness)          │
│    • 输出多样性 (variance)                      │
│    • 区分度保持率 (discriminability_preservation)│
│                                                │
│  [维度 2] 输出质量 (Output)                     │
│    • 目标对齐度 (goal_alignment)                │
│    • Prior 增益 vs 文本 (prior_gain)            │
│    • 跨模态一致性 (cross_modal_consistency)     │
│                                                │
│  [维度 3] 可控性 (Controllability) ⭐ 最重要     │
│    • Policy 可控性 (policy_controllability)     │
│    • CFG 敏感度 (cfg_sensitivity)               │
│    • Prior vs 文本对比 (comparative_control)    │
│                                                │
│  [维度 4] 端到端 (End-to-End)                   │
│    • 任务成功率 (task_success_rate)             │
│    • CFG 增益 (cfg_gain)                        │
│                                                │
└────────────────────────────────────────────────┘
```

---

## 🛠️ 实现方案

### 阶段 1: 扩展现有评估器

修改 `steve1_prior_evaluator.py`，添加新指标：

```python
class Steve1PriorEvaluator:
    """扩展版 Prior 评估器"""
    
    def __init__(self, ..., policy_model=None, env_creator=None):
        # 添加 policy 和环境（用于可控性测试）
        self._policy = policy_model
        self._env_creator = env_creator
    
    # ========== 新增方法 ==========
    
    def compute_discriminability_preservation(self, tasks):
        """计算区分度保持率"""
        # 1. MineCLIP 文本嵌入的区分度
        text_embeds = [self._mineclip.encode_text([t]) for t in tasks]
        text_disc = self._compute_discriminability(text_embeds)
        
        # 2. Prior 输出的区分度
        prior_embeds = [self._get_prior_embed(t) for t in tasks]
        prior_disc = self._compute_discriminability(prior_embeds)
        
        # 3. 保持率
        preservation_rate = prior_disc / (text_disc + 1e-6)
        
        return {
            'text_discriminability': text_disc,
            'prior_discriminability': prior_disc,
            'preservation_rate': preservation_rate,
        }
    
    def compute_prior_gain(self, task_id, instruction):
        """计算 Prior 相对于直接文本的增益"""
        # 获取成功画面嵌入
        success_visual_embeds = self.success_visuals[task_id]['success_visual_embeds']
        
        # Prior 路径
        z_prior = self._get_prior_embed(instruction)
        sims_prior = [1 - cosine(z_prior, z_v) for z_v in success_visual_embeds]
        goal_alignment_prior = np.mean(sims_prior)
        
        # 文本路径
        z_text = self._mineclip.encode_text([instruction])[0].cpu().numpy()
        sims_text = [1 - cosine(z_text, z_v) for z_v in success_visual_embeds]
        goal_alignment_text = np.mean(sims_text)
        
        # 增益
        prior_gain = goal_alignment_prior - goal_alignment_text
        
        return {
            'goal_alignment_prior': goal_alignment_prior,
            'goal_alignment_text': goal_alignment_text,
            'prior_gain': prior_gain,
        }
    
    def evaluate_policy_controllability(self, task_id, instruction, n_trials=3):
        """测试 Prior 输出能否有效控制 Policy"""
        if self._policy is None or self._env_creator is None:
            raise ValueError("需要提供 policy 和 env_creator")
        
        # 获取 Prior 输出
        z_goal = self._get_prior_embed(instruction)
        
        # 运行试验
        successes = 0
        for trial in range(n_trials):
            env = self._env_creator(task_id)
            obs = env.reset()
            
            done = False
            steps = 0
            max_steps = 1000
            
            while not done and steps < max_steps:
                # 使用 z_goal 控制 policy
                action = self._policy.get_action(obs, z_goal)
                obs, reward, done, info = env.step(action)
                steps += 1
                
                if info.get('success', False):
                    successes += 1
                    break
            
            env.close()
        
        controllability = successes / n_trials
        
        return controllability
```

---

### 阶段 2: 创建独立的可控性评估器

**新文件**: `src/evaluation/prior_controllability_evaluator.py`

```python
class PriorControllabilityEvaluator:
    """
    Prior 可控性评估器
    
    核心思想：
    - 不看 Prior 输出的数值
    - 看 Prior 输出能否让 Policy 完成任务
    """
    
    def __init__(self, policy, mineclip, prior, env_config):
        self.policy = policy
        self.mineclip = mineclip
        self.prior = prior
        self.env_config = env_config
    
    def evaluate_prior_vs_text(self, tasks):
        """
        关键实验：Prior vs 直接文本
        
        这是最重要的 Prior 评估指标！
        """
        results = []
        
        for task in tasks:
            # 方法 A: 使用 Prior
            z_prior = get_prior_embed(task['instruction'], ...)
            success_prior = self._run_trials(task, z_prior)
            
            # 方法 B: 直接文本
            z_text = self.mineclip.encode_text([task['instruction']])[0]
            success_text = self._run_trials(task, z_text)
            
            results.append({
                'task_id': task['task_id'],
                'success_prior': success_prior,
                'success_text': success_text,
                'prior_gain': success_prior - success_text,
            })
        
        return results
    
    def evaluate_cfg_effect(self, task, cfg_scales=[0, 1, 3, 6, 9]):
        """测试 CFG 对不同 Prior 输出的影响"""
        z_goal = get_prior_embed(task['instruction'], ...)
        
        results = []
        for scale in cfg_scales:
            self.policy.reset(cond_scale=scale)
            success_rate = self._run_trials(task, z_goal)
            results.append({
                'cfg_scale': scale,
                'success_rate': success_rate,
            })
        
        return results
```

---

## 📊 完整评估流程

### 流程设计

```python
# scripts/run_comprehensive_prior_evaluation.py

def main():
    # 1. 加载组件
    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(...)
    policy = make_agent(...)
    
    # 2. 维度 1: 内在质量
    print("[维度 1] Prior 内在质量")
    intrinsic_results = evaluate_intrinsic_quality(
        tasks=all_tasks,
        mineclip=mineclip,
        prior=prior,
    )
    # 输出: consistency, semantic_robustness, variance, discriminability
    
    # 3. 维度 2: 输出质量
    print("[维度 2] Prior 输出质量")
    output_results = evaluate_output_quality(
        tasks=all_tasks,
        mineclip=mineclip,
        prior=prior,
        success_visuals=success_visuals,
    )
    # 输出: goal_alignment, prior_gain, cross_modal_consistency
    
    # 4. 维度 3: 可控性 ⭐ 最重要
    print("[维度 3] Prior 可控性")
    controllability_results = evaluate_controllability(
        tasks=selected_tasks,  # 选择代表性任务
        mineclip=mineclip,
        prior=prior,
        policy=policy,
        env_config=env_config,
    )
    # 输出: policy_controllability, prior_vs_text, cfg_sensitivity
    
    # 5. 维度 4: 端到端
    print("[维度 4] 端到端质量")
    e2e_results = evaluate_end_to_end(
        tasks=all_tasks,
        policy=policy,
        mineclip=mineclip,
        prior=prior,
    )
    # 输出: task_success_rate, cfg_gain
    
    # 6. 生成报告
    generate_comprehensive_report(
        intrinsic=intrinsic_results,
        output=output_results,
        controllability=controllability_results,
        e2e=e2e_results,
    )
```

---

## 🎯 关键洞察

### 为什么 discriminability = 0.12 但系统工作？

**答案**: 因为我们之前只看了维度 1 和 2，忽略了维度 3！

**可能的情况**：

```
维度 1 (内在): 
  discriminability = 0.12  ❌ 低

维度 2 (输出):
  goal_alignment = 0.97  ✅ 高

维度 3 (可控性): 
  prior_vs_text_gain = +0.15  ✅ Prior 显著优于直接文本
  cfg_sensitivity = 高  ✅ CFG 有效放大差异

维度 4 (端到端):
  task_success_rate = 0.80  ✅ 高

结论: Prior 虽然区分度低，但通过 CFG 和 Policy 的配合仍然有效！
```

---

## 📝 推荐的评估指标优先级

### Tier 1: 必须评估（可控性）⭐⭐⭐⭐⭐

1. **Prior vs Text 对比实验**
   - 最直接的 Prior 价值证明
   - 实现成本：中等（需要运行实际任务）

2. **Policy 可控性测试**
   - 测试 Prior 输出能否指导 Policy
   - 实现成本：中等

3. **CFG 敏感度分析**
   - 理解 CFG 的作用
   - 实现成本：低

---

### Tier 2: 应该评估（输出质量）⭐⭐⭐⭐

1. **目标对齐度（改进版）**
   - 先计算后平均
   - 实现成本：低

2. **Prior 增益 vs 文本**
   - 对齐度对比
   - 实现成本：低

3. **跨模态一致性**
   - 验证 Prior 输出在视觉空间
   - 实现成本：中等

---

### Tier 3: 可选评估（内在质量）⭐⭐⭐

1. **区分度保持率**
   - 对比文本和 Prior 的区分度
   - 实现成本：低

2. **输出稳定性**（已实现）
3. **语义鲁棒性**（已实现）
4. **输出多样性**（已实现）

---

## 🔧 实现建议

### 短期（本周）：快速验证

**优先实现 Tier 1 指标**：

```bash
# 创建脚本 scripts/test_prior_controllability.py

# 测试 1: Prior vs Text
python scripts/test_prior_controllability.py \
  --mode prior_vs_text \
  --tasks harvest_1_log,harvest_1_dirt,combat_pig

# 测试 2: CFG 敏感度
python scripts/test_prior_controllability.py \
  --mode cfg_sensitivity \
  --tasks harvest_1_log \
  --cfg-scales 0,3,6,9
```

**预期结果**：
- 如果 Prior 比直接文本好 → Prior 有价值
- 如果 CFG 显著影响性能 → CFG 是关键

---

### 中期（下周）：完善评估

1. 实现改进的目标对齐度计算
2. 添加跨模态一致性指标
3. 完善可视化

---

### 长期（本月）：建立基线

1. 在所有 39 个任务上运行完整评估
2. 生成基线报告
3. 建立性能数据库

---

## 📊 基线指标模板

```json
{
  "prior_evaluation_baseline": {
    "version": "1.0",
    "date": "2025-12-01",
    "model_weights": "steve1_prior.pt",
    
    "intrinsic_quality": {
      "consistency": 0.999,
      "semantic_robustness": 0.968,
      "output_variance": 0.097,
      "discriminability": 0.12,
      "discriminability_preservation": 0.85  // 新增
    },
    
    "output_quality": {
      "avg_goal_alignment": 0.94,  // 改进版
      "avg_goal_alignment_std": 0.08,  // 新增
      "avg_prior_gain": 0.12,  // 新增 ⭐
      "cross_modal_consistency": 0.68  // 新增
    },
    
    "controllability": {  // 新增 ⭐⭐⭐
      "avg_policy_controllability": 0.75,
      "prior_vs_text_success": {
        "with_prior": 0.78,
        "without_prior": 0.63,
        "gain": 0.15
      },
      "cfg_sensitivity": {
        "lambda_0": 0.45,
        "lambda_6": 0.78,
        "gain": 0.33
      }
    },
    
    "end_to_end": {
      "avg_task_success_rate": 0.80,
      "task_breakdown": {...}
    }
  }
}
```

---

## 🎯 回答你的问题

### 你的问题

> "prior 没有好的方式进行准确评估，你有什么方案"

### 我的方案

**核心答案**: **不要孤立地评估 Prior，而要评估 Prior 在系统中的作用！**

**三个关键实验**（按优先级）：

1. **Prior vs Text 对比** ⭐⭐⭐⭐⭐
   - 直接回答：Prior 是否比直接文本好？
   - 实现：对比两种方式的任务成功率
   - 时间：2-3 小时

2. **CFG 敏感度测试** ⭐⭐⭐⭐
   - 理解 CFG 在系统中的作用
   - 验证 Prior 是否需要 CFG 补偿
   - 时间：1-2 小时

3. **区分度保持率** ⭐⭐⭐
   - 重新解释 discriminability
   - 对比输入和输出的区分度
   - 时间：30 分钟

**其他指标**：
- 改进目标对齐度计算（先计算后平均）
- 添加跨模态一致性
- 可视化增强

---

## 📁 需要创建的文件

### 1. 可控性评估器

```
src/evaluation/prior_controllability_evaluator.py
  - PriorControllabilityEvaluator 类
  - prior_vs_text() 方法
  - cfg_sensitivity() 方法
```

### 2. 对比实验脚本

```
scripts/test_prior_controllability.py
  - 运行 Prior vs Text 实验
  - 运行 CFG 敏感度实验
  - 生成对比报告
```

### 3. 综合评估脚本

```
scripts/run_comprehensive_prior_evaluation.sh
  - 运行所有四个维度的评估
  - 生成完整报告
```

---

## 🚀 立即可执行的步骤

### 今天（2-3 小时）

1. **创建 Prior vs Text 测试脚本**
2. **在 3-5 个代表性任务上测试**
3. **查看 Prior 是否真的有价值**

**如果 Prior gain > 0.1**: 
- ✅ Prior 有价值，继续使用
- 💡 discriminability 低不是问题

**如果 Prior gain < 0**:
- ❌ Prior 在拖后腿
- 🔧 考虑跳过 Prior 或重新训练

---

要我帮你实现这些评估工具吗？我可以从最重要的 **Prior vs Text 对比实验**开始！🚀

