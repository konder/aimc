# Prior评估目标分析与重新设计总结

## 问题回顾

用户提出的核心问题：
> 回到Prior评估，需要认真分析下，我们评估的目标是什么（识别文字->目标画面的余弦期望距离？大白话就是输入文字是否能准确获得未来目标画面），基于这个目标应该从哪几个维度来做评估。

## 核心发现

### 🎯 Prior评估的本质目标

**Prior模型 p(z_goal|y) 的作用**：
```
输入: 文本指令 "chop tree"
输出: 目标嵌入 z_goal（应该代表"砍到树后看到的画面"）
本质: 将文字映射到目标视觉状态的表示
```

**用大白话说**：
> 给它一句话，它能否"想象"出完成任务时应该看到的画面？

### ⚠️ 当前实现的严重错误

**问题代码**：
```python
# ❌ policy_metrics.py line 79-90
text_embed = mineclip.encode_text("chop tree")      # 文本空间 [512]
prior_embed = prior_model(text_embed)               # 视觉空间 [512]
similarity = cosine(text_embed, prior_embed)        # 跨空间比较，无意义！
```

**为什么错**：
1. MineCLIP的文本编码器和视觉编码器输出到**不同的嵌入空间**
2. Prior的输出应该在**视觉空间**（要给Policy提供目标画面的表示）
3. 直接比较文本嵌入和视觉嵌入没有语义意义
4. 即使相似度高也不能说明Prior准确

**类比**：
```
这就像：
- 文本空间是"中文"
- 视觉空间是"英文"
- 你在比较一个中文句子和一个英文句子的相似度
- 即使字符相似也不代表意思相同！
```

**正确的做法**：
```python
# ✅ 应该这样
z_goal = prior_model(text)                          # 视觉空间（Prior输出）
z_visual = mineclip.encode_visual(success_frame)    # 视觉空间（真实成功画面）
similarity = cosine(z_goal, z_visual)               # 同空间比较，有意义！
```

## 正确的评估维度

基于"输入文字是否能准确获得未来目标画面"这一核心目标，应该从以下维度评估：

### 维度1: 目标准确性（Goal Accuracy）⭐⭐⭐⭐⭐
**最核心指标**

**定义**：Prior生成的z_goal与真实成功画面的视觉嵌入之间的相似度

**计算方法**：
```python
z_goal = prior_model("chop tree")                    # Prior"想象"的画面
z_visual = mineclip.encode_visual(success_frame)     # 真实成功画面
goal_accuracy = cosine_similarity(z_goal, z_visual)  # 想象的准不准？
```

**参考值**：
- 优秀: ≥ 0.6（Prior准确"想象"出目标画面）
- 良好: 0.4 - 0.6
- 需改进: < 0.4（Prior输出与实际目标画面差距大）

**数据需求**：
- 从成功的trial中提取最后N帧（或奖励最高时的帧）
- 用MineCLIP的visual_encoder编码
- 保存为 `data/success_visual_embeds/{task_id}.pkl`

---

### 维度2: Policy成功率对比（Ablation Study）⭐⭐⭐⭐⭐
**最直接的端到端指标**

**定义**：使用Prior生成的z_goal vs 使用真实视觉z_visual，Policy的成功率差异

**实验设计**：
```python
# 实验A: 使用Prior（正常流程）
z_goal_prior = prior_model(text)
success_rate_A = run_policy_trials(z_goal_prior, n_trials=10)

# 实验B: 使用真实视觉嵌入（理想情况，绕过Prior）
z_goal_gt = mineclip.encode_visual(success_frame)
success_rate_B = run_policy_trials(z_goal_gt, n_trials=10)

# Prior质量gap
prior_quality_gap = success_rate_B - success_rate_A
```

**参考值**：
- 优秀: gap < 10%（Prior几乎不损失性能）
- 良好: gap 10-30%
- 需改进: gap > 30%（Prior是严重瓶颈）

**含义**：
- gap小 → Prior很好地捕获了目标画面，不是瓶颈
- gap大 → Prior是主要瓶颈，需优先优化

**优势**：这是最直接的指标，直接告诉你Prior好不好

---

### 维度3: 一致性（Consistency）⭐⭐⭐⭐
**稳定性指标**

**定义**：同一文本多次输入Prior，输出的z_goal是否稳定

**计算方法**：
```python
z_goals = [prior_model("chop tree") for _ in range(10)]  # 多次采样
consistency = mean(pairwise_cosine_similarity(z_goals))
```

**参考值**：
- 优秀: ≥ 0.95（高度一致）
- 良好: 0.85 - 0.95
- 需改进: < 0.85（输出不稳定，可能影响Policy）

**意义**：
- 高一致性：Prior训练充分，输出可靠
- 低一致性：Prior可能欠拟合或采样噪声过大

---

### 维度4: 跨模态检索准确性（Cross-modal Retrieval）⭐⭐⭐
**间接验证指标**

**定义**：Prior输出的z_goal，在视觉数据库中检索时，是否能找到语义相关的画面

**计算方法**：
```python
# 在MineCLIP的联合语义空间中
z_goal = prior_model("chop tree")

# 在视觉数据库中找最近邻
nearest_visuals = find_top_k_nearest(z_goal, visual_database, k=5)

# 检查这些视觉是否确实是"砍树完成"的画面
retrieval_accuracy = count_correct(nearest_visuals) / k
```

**参考值**：
- 优秀: ≥ 0.8（top-5中大部分是正确类别）
- 良好: 0.5 - 0.8
- 需改进: < 0.5（Prior输出指向错误的视觉概念）

---

### 维度5: 可区分性（Inter-task Discriminability）⭐⭐⭐
**多任务评估**

**定义**：不同任务的z_goal是否足够不同（避免退化）

**计算方法**：
```python
tasks = ["chop tree", "mine diamond", "hunt pig"]
z_goals = [prior_model(task) for task in tasks]

# 类间距离应该大
inter_similarity = mean([cosine(z_i, z_j) for i≠j])
discriminability = 1 - inter_similarity  # 越大越好
```

**参考值**：
- 优秀: ≥ 0.5（类间相似度<0.5，差异明显）
- 良好: 0.3 - 0.5
- 需改进: < 0.3（所有任务输出相似，模型退化）

**退化检测**：
```python
if discriminability < 0.3 and variance < 0.0001:
    print("⚠️ 警告：Prior退化（所有任务输出相同）")
```

---

### 维度6: 语义鲁棒性（Semantic Robustness）⭐⭐⭐⭐
**表述变体一致性 - 测试语义理解能力**

**定义**：同一任务目标，用不同的语言表达（paraphrases），Prior应该生成相似的目标嵌入

**计算方法**：
```python
# 同一任务的不同表述
variants = [
    "chop tree",                           # 简洁版
    "chop tree and get wood",              # 明确结果
    "cut down a tree to obtain wood",      # 详细版
    "harvest wood from tree",              # 换词表达
    "chop tree then get one wood",         # 添加细节
]

z_goals = [prior_model(v) for v in variants]
semantic_robustness = mean(pairwise_similarity(z_goals))
```

**参考值**：
- 优秀: ≥ 0.9（Prior准确抓住核心任务，不被表述差异影响）
- 良好: 0.7 - 0.9（基本理解任务目标，但受表述影响）
- 需改进: < 0.7（过度关注表层词汇，语义理解不足）

**为什么重要**：
1. **实用价值**：真实使用时，用户可能用各种方式表达同一个意图
2. **测试语义理解**：区分"理解任务目标"vs"记忆表面词汇"
3. **模型泛化能力**：好的Prior应该关注任务本质，而不是具体措辞
4. **用户体验**：高鲁棒性意味着用户不需要精确匹配训练时的表述

**与其他指标的关系**：
- **vs 一致性**：一致性测试同一表述多次采样，鲁棒性测试不同表述
- **vs 可区分性**：可区分性测试不同任务应该不同，鲁棒性测试同一任务应该相同
- **理想状态**：高可区分性 + 高鲁棒性（不同任务不同，同一任务相同）

---

### 维度7: 方差分析（Variance）⭐⭐
**辅助指标**

当前的`prior_variance`有一定意义，但应该：
- 在多个任务上计算
- 配合可区分性使用
- 检查是否退化（所有任务输出相同z_goal）

**参考值**：
- 过低: < 0.0001（可能退化）
- 正常: 0.0001 - 0.01
- 过高: > 0.01（可能不稳定）

---

## 实施计划

### 阶段1: 数据收集（高优先级）

**任务**：收集成功画面数据集

**步骤**：
```bash
# 1. 创建数据收集脚本
python scripts/collect_success_visuals.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --n-trials-per-task 10 \
    --output data/success_visual_embeds/

# 2. 输出格式
data/success_visual_embeds/
  ├── harvest_1_log.pkl          # 包含10个成功trial的视觉嵌入
  ├── mine_1_diamond.pkl
  └── ...
```

**数据结构**：
```python
{
    'task_id': 'harvest_1_log',
    'instruction': 'chop tree',
    'success_visual_embeds': [
        np.ndarray([512]),  # trial 1的成功画面嵌入（最后10帧的平均）
        np.ndarray([512]),  # trial 2
        ...
    ],
    'success_frames': [
        [...],  # 原始图像（可选，用于可视化）
    ]
}
```

---

### 阶段2: 实现Ablation实验（高优先级）

**任务**：实现Policy成功率对比

**代码**：
```python
# 在steve1_prior_evaluator.py中添加
def run_ablation_study(self, task, n_trials=10):
    """
    对比使用Prior vs 使用真实视觉的Policy成功率
    """
    # 实验A: 使用Prior（正常流程）
    results_with_prior = []
    for _ in range(n_trials):
        z_goal = self.steve1_agent.get_prior_embed(task.instruction)
        success = self._run_single_trial(task, z_goal)
        results_with_prior.append(success)
    
    # 实验B: 使用真实视觉（绕过Prior）
    success_visuals = self._load_success_visuals(task.task_id)
    z_visual_gt = np.mean(success_visuals, axis=0)
    
    results_with_gt = []
    for _ in range(n_trials):
        success = self._run_single_trial(task, z_visual_gt)
        results_with_gt.append(success)
    
    return {
        'success_rate_with_prior': np.mean(results_with_prior),
        'success_rate_with_gt': np.mean(results_with_gt),
        'prior_quality_gap': np.mean(results_with_gt) - np.mean(results_with_prior)
    }
```

---

### 阶段3: 替换错误指标（中优先级）

**任务**：修改`policy_metrics.py`和`steve1_prior_evaluator.py`

**删除**：
```python
# policy_metrics.py
- "text_to_prior_similarity"  # 错误指标
- compute_text_to_prior_similarity()
```

**添加**：
```python
# policy_metrics.py
+ "goal_accuracy"  # Prior vs 真实成功画面
+ "prior_quality_gap"  # Policy成功率差异
+ "consistency"  # 多次采样稳定性
+ "discriminability"  # 不同任务可区分性

# 对应的计算函数
+ compute_goal_accuracy()
+ compute_prior_quality_gap()
+ compute_consistency()
+ compute_discriminability()
```

**参考实现**：
- 见 `src/evaluation/prior_metrics.py`

---

### 阶段4: 更新文档和可视化（低优先级）

**任务**：
1. 更新 `docs/guides/DEEP_EVALUATION_METRICS_EXPLAINED.md`
2. 更新HTML报告中的Prior部分说明
3. 添加新的可视化图表：
   - Prior vs 真实画面的相似度分布
   - Ablation实验的成功率对比柱状图
   - 多任务可区分性热力图

---

## 技术细节

### MineCLIP空间分析

```python
# MineCLIP有两个encoder
mineclip = MineCLIP()

# 文本编码器 → 文本空间
text_embed = mineclip.encode_text("chop tree")     # [512]

# 视觉编码器 → 视觉空间
visual_embed = mineclip.encode_visual(frame)       # [512]

# 虽然维度相同（都是512），但它们在不同的嵌入空间！
# 通过对比学习（contrastive learning）对齐，但不是同一空间

# Prior的输出应该在视觉空间
z_goal = prior_model(text_embed)  # 输出: 视觉空间 [512]

# ✅ 正确比较（同空间）
similarity = cosine(z_goal, visual_embed)

# ❌ 错误比较（跨空间）
similarity = cosine(z_goal, text_embed)  # 无意义！
```

### 成功画面定义

**选项1**：最后N帧的平均
```python
success_frames = trajectory[-10:]  # 最后10帧
z_visual = mean([mineclip.encode_visual(f) for f in success_frames])
```

**选项2**：奖励最高时刻
```python
max_reward_idx = np.argmax(rewards)
success_frame = trajectory[max_reward_idx]
z_visual = mineclip.encode_visual(success_frame)
```

**建议**：两者结合
```python
# 取最后5帧 + 奖励最高帧的平均
last_frames = trajectory[-5:]
max_reward_frame = trajectory[np.argmax(rewards)]
all_frames = last_frames + [max_reward_frame]
z_visual = mean([mineclip.encode_visual(f) for f in all_frames])
```

---

## 关键洞察

### ❌ 错误的思路
> "Prior应该和文本嵌入相似"

**为什么错**：
- Prior输出在视觉空间，不是文本空间
- 跨空间比较没有语义意义
- 无法反映Prior是否准确

### ✅ 正确的思路
> "Prior应该能'想象'出完成任务时的画面"

**如何验证**：
1. **目标准确性**：和真实成功画面比较（goal_accuracy）
2. **端到端验证**：用它跑Policy看成功率（prior_quality_gap）
3. **稳定性**：多次采样的一致性（consistency）
4. **泛化性**：不同任务的可区分性（discriminability）

---

## 类比说明

**Prior评估就像评估一个画家**：

❌ **错误评估**：
- "画家画的画和文字描述像不像"
- 问题：画是画，文字是文字，怎么比？

✅ **正确评估**：
- "画家画的画和真实照片像不像"（目标准确性）
- "用画家的画能不能指导别人完成任务"（Policy成功率对比）
- "同一个主题画多次，是否风格一致"（一致性）
- "不同主题的画是否有区别"（可区分性）

---

## 相关文档

1. **详细设计**：`docs/technical/PRIOR_EVALUATION_REDESIGN.md`
   - 每个维度的完整定义、计算方法、参考值

2. **对比分析**：`docs/technical/PRIOR_METRICS_COMPARISON.md`
   - 当前实现 vs 正确实现的详细对比表格

3. **代码实现**：`src/evaluation/prior_metrics.py`
   - 正确的Prior指标计算代码

---

## 下一步行动

### 立即执行（必要）
1. ✅ 理解Prior评估的本质目标
2. ✅ 认识到当前`text_to_prior_similarity`是错误的
3. ⏳ 开始收集成功画面数据集

### 短期执行（1-2天）
4. ⏳ 实现Ablation实验（Policy成功率对比）
5. ⏳ 实现`goal_accuracy`指标
6. ⏳ 准备指令变体数据集，实现`semantic_robustness`指标

### 中期执行（1周）
7. ⏳ 替换所有错误指标
8. ⏳ 实现可区分性和退化检测
9. ⏳ 更新HTML报告和文档

---

## 总结

**核心问题**：
- 当前的`text_to_prior_similarity`指标是**根本性错误**
- 跨空间比较（文本空间 vs 视觉空间）没有意义

**核心目标**：
- Prior评估的本质是：**输入文字是否能准确"想象"出未来目标画面**

**核心方法**：
1. **和真实成功画面比较**（目标准确性）
2. **用Policy成功率验证**（Ablation Study）
3. **检查稳定性和泛化性**（一致性、可区分性）

**立即行动**：
- 收集成功画面数据集
- 实现Policy成功率对比实验
- 替换错误的指标定义

---

**作者**: AI Assistant  
**日期**: 2025-11-27  
**版本**: v1.0  
**状态**: 已完成分析，待实施

