# Prior评估指标对比：错误 vs 正确

## 核心问题总结

**当前实现的根本错误**：
```python
# ❌ 当前代码（错误）
text_embed = mineclip.encode_text("chop tree")      # 文本空间 [512]
prior_embed = prior_model(text_embed)               # 视觉空间 [512]
similarity = cosine(text_embed, prior_embed)        # 跨空间比较，无意义！
```

**为什么错误**：
- MineCLIP的文本编码器和视觉编码器输出到**不同的嵌入空间**
- Prior模型应该输出在**视觉空间**（因为它要给Policy提供目标画面的表示）
- 直接比较文本嵌入和视觉嵌入没有语义意义

**正确的做法**：
```python
# ✅ 正确版本
z_goal = prior_model(text)                          # 视觉空间
z_visual = mineclip.encode_visual(success_frame)    # 视觉空间
similarity = cosine(z_goal, z_visual)               # 同空间比较，有意义！
```

---

## 详细对比表格

| 维度 | 当前实现（❌错误） | 正确实现（✅推荐） | 为什么当前实现错误 |
|------|-------------------|-------------------|-------------------|
| **核心指标** | `text_to_prior_similarity`<br>文本嵌入 vs Prior输出 | `goal_accuracy`<br>Prior输出 vs 真实成功画面 | 跨空间比较无语义意义；<br>应该比较"想象的画面"vs"真实画面" |
| **计算方法** | `cosine(text_embed, prior_embed)` | `cosine(z_goal, z_visual_success)` | 文本空间≠视觉空间 |
| **数据需求** | 只需要文本 | 需要成功trial的画面 | 当前缺少ground truth |
| **参考值** | 0.3-0.5（无理论依据） | 0.4-0.6（基于实验） | 当前阈值是猜的 |
| **改进方向** | 不明确 | 明确（收集更多text-visual配对数据） | 当前无法指导优化 |

---

## 指标对比详表

### 维度1: 目标准确性

#### ❌ 当前实现
```python
# policy_metrics.py (line 79-90)
"text_to_prior_similarity": MetricDefinition(
    name="文本-Prior相似度",
    description="MineCLIP文本嵌入与Prior输出的余弦相似度",
    excellent_min=0.5,
    good_min=0.3,
)

# 计算方法
def compute_text_to_prior_similarity(
    text_embed: np.ndarray,  # 文本空间
    prior_embed: np.ndarray  # 视觉空间
) -> float:
    return float(1 - cosine(text_embed, prior_embed))  # ❌ 跨空间
```

**问题**：
1. 文本嵌入和视觉嵌入不在同一空间
2. 即使相似度高也不能说明Prior准确
3. 无法反映Prior是否"想象"出正确的画面

#### ✅ 正确实现
```python
# prior_metrics_correct.py
"goal_accuracy": PriorMetricDefinition(
    name="目标准确性",
    description="Prior输出的z_goal与真实成功画面视觉嵌入的余弦相似度",
    excellent_min=0.6,
    good_min=0.4,
)

# 计算方法
def compute_goal_accuracy(
    z_goal: np.ndarray,              # 视觉空间（Prior输出）
    success_visual_embeds: List[np.ndarray]  # 视觉空间（真实画面）
) -> float:
    z_visual_mean = np.mean(success_visual_embeds, axis=0)
    return float(1 - cosine(z_goal, z_visual_mean))  # ✅ 同空间
```

**优势**：
1. 同空间比较，有语义意义
2. 直接衡量Prior是否"想象"对了
3. 可以指导数据收集和模型优化

---

### 维度2: 端到端验证（新增）

#### ❌ 当前实现
- **不存在**：没有端到端的验证机制
- 无法知道Prior对最终成功率的影响

#### ✅ 正确实现
```python
"prior_quality_gap": PriorMetricDefinition(
    name="Prior质量差距",
    description="使用Prior vs 使用真实视觉时，Policy成功率的差异",
    excellent_max=0.1,  # gap < 10%说明Prior不是瓶颈
    good_max=0.3,       # gap < 30%说明Prior可接受
)

# 实验设计
# 实验A: 正常流程
z_goal = prior_model(text)
success_rate_A = run_policy(z_goal, n_trials=10)

# 实验B: 理想情况（绕过Prior）
z_goal_gt = mineclip.encode_visual(success_frame)
success_rate_B = run_policy(z_goal_gt, n_trials=10)

# Prior质量gap
prior_quality_gap = success_rate_B - success_rate_A
```

**优势**：
1. 最直接的指标：Prior好不好直接看成功率
2. 明确告诉你Prior是否是瓶颈
3. 可以量化优化空间

---

### 维度3: 一致性

#### ✅ 当前实现（这个是对的）
```python
"prior_variance": MetricDefinition(
    name="Prior方差",
    description="Prior输出的方差，衡量嵌入多样性",
)
```

**保留但改进**：
```python
"consistency": PriorMetricDefinition(
    name="一致性",
    description="同一文本多次输入Prior，输出z_goal的稳定性",
    excellent_min=0.95,
    good_min=0.85,
)

# 计算方法（更直观）
def compute_consistency(z_goals: List[np.ndarray]) -> float:
    # 同一文本多次采样
    similarities = []
    for i in range(len(z_goals)):
        for j in range(i+1, len(z_goals)):
            sim = 1 - cosine(z_goals[i], z_goals[j])
            similarities.append(sim)
    return np.mean(similarities)
```

---

### 维度4: 可区分性（新增）

#### ❌ 当前实现
- **不存在**：没有检查不同任务是否得到不同的z_goal
- 无法检测Prior退化（所有任务输出相同）

#### ✅ 正确实现
```python
"discriminability": PriorMetricDefinition(
    name="可区分性",
    description="不同任务的z_goal是否足够不同（1 - 类间相似度）",
    excellent_min=0.5,
    good_min=0.3,
)

def compute_discriminability(z_goals: List[np.ndarray]) -> float:
    # 不同任务的Prior输出
    inter_similarities = []
    for i in range(len(z_goals)):
        for j in range(i+1, len(z_goals)):
            sim = 1 - cosine(z_goals[i], z_goals[j])
            inter_similarities.append(sim)
    
    mean_inter_sim = np.mean(inter_similarities)
    discriminability = 1 - mean_inter_sim
    return discriminability

# 退化检测
if discriminability < 0.3 and variance < 0.0001:
    print("⚠️ 警告：Prior退化（所有任务输出相同）")
```

**优势**：
1. 检测Prior是否退化
2. 验证Prior的表达能力
3. 多任务泛化能力评估

---

## 实施对比

### 当前实施流程（❌问题多）

```python
# steve1_prior_evaluator.py (当前)
def analyze_prior_model(self, tasks):
    for task in tasks:
        # 1. 获取文本嵌入
        text_embed = self.mineclip_agent.get_text_embed(task.instruction)
        
        # 2. 获取Prior嵌入
        prior_embed = self.steve1_agent.get_prior_embed(task.instruction)
        
        # 3. ❌ 错误比较
        similarity = 1 - cosine(text_embed, prior_embed)
        
        # 4. ❌ 没有ground truth
        # 5. ❌ 没有端到端验证
```

**问题**：
1. 跨空间比较
2. 缺少ground truth（真实成功画面）
3. 缺少端到端验证（Policy成功率对比）
4. 无法检测退化

---

### 正确实施流程（✅推荐）

```python
# steve1_prior_evaluator_correct.py (建议)
def analyze_prior_model(self, tasks):
    # 阶段1: 收集ground truth
    success_visual_database = {}
    for task in tasks:
        # 运行一些成功的trial
        success_trials = self._collect_success_trials(task, n=5)
        
        # 提取成功画面
        success_frames = []
        for trial in success_trials:
            # 取最后10帧或奖励最高帧
            frames = trial.trajectory[-10:]
            success_frames.extend(frames)
        
        # 编码为视觉嵌入
        success_visuals = [
            self.mineclip_agent.encode_visual(frame)
            for frame in success_frames
        ]
        success_visual_database[task.task_id] = success_visuals
    
    # 阶段2: 评估Prior
    for task in tasks:
        # 1. Prior输出
        z_goal = self.steve1_agent.get_prior_embed(task.instruction)
        
        # 2. ✅ 目标准确性（同空间）
        goal_accuracy = PriorMetrics.compute_goal_accuracy(
            z_goal,
            success_visual_database[task.task_id]
        )
        
        # 3. ✅ 一致性（多次采样）
        z_goals_multi = [
            self.steve1_agent.get_prior_embed(task.instruction)
            for _ in range(10)
        ]
        consistency = PriorMetrics.compute_consistency(z_goals_multi)
        
        # 4. ✅ Ablation Study
        # 实验A: 使用Prior
        success_rate_A = self._run_trials(task, use_prior=True)
        
        # 实验B: 使用真实视觉
        z_visual_gt = np.mean(success_visual_database[task.task_id], axis=0)
        success_rate_B = self._run_trials(task, z_goal_override=z_visual_gt)
        
        prior_quality_gap = success_rate_B - success_rate_A
        
        # 保存结果
        result = PriorAnalysisResult(
            text_instruction=task.instruction,
            task_id=task.task_id,
            goal_accuracy=goal_accuracy,
            consistency=consistency,
            z_goal=z_goal,
            success_visual_embeds=success_visual_database[task.task_id]
        )
    
    # 阶段3: 跨任务分析
    all_z_goals = [r.z_goal for r in results]
    discriminability = PriorMetrics.compute_discriminability(all_z_goals)
    variance = PriorMetrics.compute_variance(all_z_goals)
    
    # 退化检测
    is_degraded, warning = PriorMetrics.check_degradation(
        discriminability, variance
    )
```

**优势**：
1. 有ground truth对比
2. 端到端验证（Policy成功率）
3. 多维度分析
4. 退化检测

---

## 数据需求对比

### 当前实施
- ✅ 文本指令
- ✅ Prior模型
- ✅ MineCLIP模型
- ❌ 没有成功画面ground truth
- ❌ 没有Ablation实验

### 正确实施
- ✅ 文本指令
- ✅ Prior模型
- ✅ MineCLIP模型
- ✅ **成功画面数据集**（需要收集）
  - 每个任务5-10个成功trial
  - 提取最后10帧或奖励最高帧
  - 用MineCLIP编码保存
  - 文件: `data/success_visual_embeds/{task_id}.pkl`
- ✅ **Ablation实验**（需要实现）
  - 运行Policy with Prior
  - 运行Policy with GT visual
  - 对比成功率

---

## 迁移计划

### 第1步：收集成功画面数据集（高优先级）

```bash
# 运行数据收集脚本
python scripts/collect_success_visuals.py \
    --tasks harvest_1_log,mine_1_diamond \
    --n-trials 10 \
    --output data/success_visual_embeds/
```

### 第2步：实现Ablation实验（高优先级）

```python
# 在steve1_prior_evaluator.py中添加
def run_ablation_study(self, task):
    # A: 使用Prior
    results_A = []
    for _ in range(10):
        z_goal = self.steve1_agent.get_prior_embed(task.instruction)
        success = self._run_single_trial(task, z_goal)
        results_A.append(success)
    
    # B: 使用真实视觉
    results_B = []
    z_visual_gt = self._load_success_visual(task.task_id)
    for _ in range(10):
        success = self._run_single_trial(task, z_visual_gt)
        results_B.append(success)
    
    return {
        'success_rate_with_prior': np.mean(results_A),
        'success_rate_with_gt': np.mean(results_B),
        'prior_quality_gap': np.mean(results_B) - np.mean(results_A)
    }
```

### 第3步：替换错误指标（中优先级）

```python
# 删除
- text_to_prior_similarity
- compute_text_to_prior_similarity

# 添加
+ goal_accuracy
+ compute_goal_accuracy
+ prior_quality_gap
+ discriminability
```

### 第4步：更新文档和可视化（低优先级）

- 更新 `DEEP_EVALUATION_METRICS_EXPLAINED.md`
- 更新HTML报告中的Prior部分说明
- 添加新的可视化图表

---

## 关键洞察总结

### ❌ 错误的思路
> "Prior应该和文本嵌入相似"

**为什么错**：Prior输出在视觉空间，不是文本空间

### ✅ 正确的思路
> "Prior应该能'想象'出完成任务时的画面"

**如何验证**：
1. 和真实成功画面比较（goal_accuracy）
2. 用它跑Policy看成功率（prior_quality_gap）
3. 检查稳定性和区分性（consistency, discriminability）

---

**作者**: AI Assistant  
**日期**: 2025-11-27  
**版本**: v1.0

