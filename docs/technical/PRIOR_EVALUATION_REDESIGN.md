# Prior模型评估维度重新设计

## 问题诊断

### 当前评估的根本问题
**`text_to_prior_similarity` 指标是错误的！**

```python
# ❌ 错误：跨空间比较
text_embed = mineclip.encode_text("chop tree")      # 文本空间
prior_embed = prior_model(text_embed)               # 视觉空间
similarity = cosine(text_embed, prior_embed)        # 无意义的比较
```

**问题根源**：
- MineCLIP的文本编码器和视觉编码器输出到不同的嵌入空间
- Prior模型输出应该在**视觉嵌入空间**
- 不应该直接和文本嵌入比较

## 核心目标重新定义

**Prior模型 p(z_goal|y) 的本质**：
> 给定文本指令y，生成一个目标嵌入z_goal，该嵌入应该代表"完成任务时应该看到的画面"在MineCLIP视觉空间中的表示

**用大白话说**：
> 输入"砍树"，Prior能否"想象"出砍到树后看到的画面（木头、手持斧头等）？

## 正确的评估维度

### 维度1: 目标准确性（Goal Accuracy）⭐⭐⭐⭐⭐
**最核心指标 - 直接衡量Prior质量**

#### 定义
Prior生成的z_goal与真实成功画面的视觉嵌入之间的相似度

#### 计算方法
```python
def compute_goal_accuracy(prior_model, mineclip, text, success_frames):
    """
    Args:
        text: "chop tree"
        success_frames: 成功完成任务时的视频帧 [T, H, W, 3]
    """
    # 1. Prior生成目标嵌入
    z_goal = prior_model(text)
    
    # 2. 提取成功画面的视觉嵌入
    z_visual_list = []
    for frame in success_frames[-16:]:  # 最后16帧（与MineCLIP设计一致）
        z_visual = mineclip.encode_visual(frame)
        z_visual_list.append(z_visual)
    z_visual_mean = np.mean(z_visual_list, axis=0)
    
    # 3. 计算相似度（同空间）
    goal_accuracy = cosine_similarity(z_goal, z_visual_mean)
    return goal_accuracy
```

#### 参考值
- **优秀**: ≥ 0.6（Prior准确"想象"出目标画面）
- **良好**: 0.4 - 0.6
- **需改进**: < 0.4（Prior输出与实际目标画面差距大）

#### 数据获取
- 从成功的trial中提取最后16帧（与MineCLIP设计一致）或奖励最高时刻的帧
- 使用MineCLIP的visual_encoder编码
- 与Prior输出计算相似度

---

### 维度2: Policy成功率对比（Ablation Study）⭐⭐⭐⭐⭐
**最直接的端到端指标**

#### 定义
使用Prior生成的z_goal vs 使用真实视觉z_visual，Policy的成功率差异

#### 计算方法
```python
# 实验A: 使用Prior（正常流程）
z_goal_prior = prior_model(text)
success_rate_A = run_policy_trials(z_goal_prior, n_trials=10)

# 实验B: 使用真实视觉嵌入（理想情况）
z_goal_gt = mineclip.encode_visual(success_frame)
success_rate_B = run_policy_trials(z_goal_gt, n_trials=10)

# Prior质量gap
prior_quality_gap = success_rate_B - success_rate_A
```

#### 参考值
- **优秀**: gap < 10%（Prior几乎不损失性能）
- **良好**: gap 10-30%
- **需改进**: gap > 30%（Prior是严重瓶颈）

#### 含义
- gap小 → Prior很好地捕获了目标画面
- gap大 → Prior是主要瓶颈，需优先优化

---

### 维度3: 一致性（Consistency）⭐⭐⭐⭐
**稳定性指标**

#### 定义
同一文本多次输入Prior，输出的z_goal是否稳定

#### 计算方法
```python
def compute_consistency(prior_model, text, n_samples=10):
    z_goals = [prior_model(text) for _ in range(n_samples)]
    
    # 计算两两相似度
    similarities = []
    for i in range(len(z_goals)):
        for j in range(i+1, len(z_goals)):
            sim = cosine_similarity(z_goals[i], z_goals[j])
            similarities.append(sim)
    
    consistency = np.mean(similarities)
    return consistency
```

#### 参考值
- **优秀**: ≥ 0.95（高度一致）
- **良好**: 0.85 - 0.95
- **需改进**: < 0.85（输出不稳定，可能影响Policy）

#### 意义
- 高一致性：Prior训练充分，输出可靠
- 低一致性：Prior可能欠拟合或采样噪声过大

---

### 维度4: 跨模态检索准确性（Cross-modal Retrieval）⭐⭐⭐
**间接验证指标**

#### 定义
Prior输出的z_goal，在视觉数据库中检索时，是否能找到语义相关的画面

#### 计算方法
```python
def compute_retrieval_accuracy(prior_model, text, visual_database):
    """
    Args:
        visual_database: {
            "chop_tree_success": [z_visual_1, z_visual_2, ...],
            "mine_diamond_success": [...],
            ...
        }
    """
    z_goal = prior_model(text)
    
    # 在数据库中找最近邻
    all_visuals = []
    all_labels = []
    for label, visuals in visual_database.items():
        all_visuals.extend(visuals)
        all_labels.extend([label] * len(visuals))
    
    # 找top-k最近邻
    similarities = [cosine_similarity(z_goal, v) for v in all_visuals]
    top_k_indices = np.argsort(similarities)[-5:]  # top-5
    top_k_labels = [all_labels[i] for i in top_k_indices]
    
    # 计算准确率（top-k中有多少是正确类别）
    correct_label = text.replace(" ", "_") + "_success"
    retrieval_accuracy = top_k_labels.count(correct_label) / len(top_k_labels)
    
    return retrieval_accuracy
```

#### 参考值
- **优秀**: ≥ 0.8（top-5中大部分是正确类别）
- **良好**: 0.5 - 0.8
- **需改进**: < 0.5（Prior输出指向错误的视觉概念）

---

### 维度5: 可区分性（Inter-task Discriminability）⭐⭐⭐
**多任务泛化指标**

#### 定义
不同任务的z_goal是否足够不同（避免退化）

#### 计算方法
```python
def compute_discriminability(prior_model, tasks):
    """
    Args:
        tasks: ["chop tree", "mine diamond", "hunt pig", ...]
    """
    z_goals = [prior_model(task) for task in tasks]
    
    # 计算类间相似度（应该低）
    inter_similarities = []
    for i in range(len(z_goals)):
        for j in range(i+1, len(z_goals)):
            sim = cosine_similarity(z_goals[i], z_goals[j])
            inter_similarities.append(sim)
    
    mean_inter_sim = np.mean(inter_similarities)
    
    # 可区分性 = 1 - 类间相似度
    discriminability = 1 - mean_inter_sim
    return discriminability
```

#### 参考值
- **优秀**: ≥ 0.5（类间相似度<0.5，差异明显）
- **良好**: 0.3 - 0.5
- **需改进**: < 0.3（所有任务输出相似，模型退化）

#### 退化检测
如果`mean_inter_sim > 0.7`，说明Prior可能退化为输出相同的嵌入

---

### 维度6: 语义鲁棒性（Semantic Robustness）⭐⭐⭐⭐
**表述变体一致性 - 测试语义理解能力**

#### 定义
同一任务目标，用不同的语言表达（paraphrases），Prior应该生成相似的目标嵌入

#### 计算方法
```python
def compute_semantic_robustness(prior_model, instruction_variants):
    """
    Args:
        instruction_variants: 同一任务的不同表述
            例如: [
                "chop tree",
                "chop tree and get wood",
                "cut down a tree to obtain wood",
                "harvest wood from tree"
            ]
    """
    # 为每个变体生成目标嵌入
    z_goals = [prior_model(variant) for variant in instruction_variants]
    
    # 计算两两相似度（类内相似度）
    intra_similarities = []
    for i in range(len(z_goals)):
        for j in range(i+1, len(z_goals)):
            sim = cosine_similarity(z_goals[i], z_goals[j])
            intra_similarities.append(sim)
    
    # 语义鲁棒性 = 平均类内相似度
    semantic_robustness = np.mean(intra_similarities)
    return semantic_robustness
```

#### 参考值
- **优秀**: ≥ 0.9（Prior准确抓住核心任务，不被表述差异影响）
- **良好**: 0.7 - 0.9（基本理解任务目标，但受表述影响）
- **需改进**: < 0.7（过度关注表层词汇，语义理解不足）

#### 意义
- **高鲁棒性**: Prior能够抽象出任务的核心目标（"what to achieve"）
- **低鲁棒性**: Prior过度依赖表层词汇，不同表述导致不同的"想象画面"

#### 测试样例
```python
# 任务1: 砍树
harvest_tree_variants = [
    "chop tree",
    "chop tree and get wood",
    "cut down a tree to obtain wood",
    "harvest wood from tree",
    "chop tree then get one wood",
]

# 任务2: 挖钻石
mine_diamond_variants = [
    "mine diamond",
    "mine diamond ore",
    "dig to find diamond",
    "obtain diamond by mining",
    "mine and collect diamond",
]

# 评估
for task_name, variants in [("harvest_tree", harvest_tree_variants), 
                             ("mine_diamond", mine_diamond_variants)]:
    robustness = compute_semantic_robustness(prior_model, variants)
    print(f"{task_name}: {robustness:.3f}")
```

#### 为什么重要
1. **实用价值**: 真实使用时，用户可能用各种方式表达同一个意图
2. **测试语义理解**: 区分"理解任务目标"vs"记忆表面词汇"
3. **模型泛化能力**: 好的Prior应该关注任务本质，而不是具体措辞
4. **用户体验**: 高鲁棒性意味着用户不需要精确匹配训练时的表述

#### 与其他指标的关系
- **vs 一致性**: 一致性测试同一表述多次采样，鲁棒性测试不同表述
- **vs 可区分性**: 可区分性测试不同任务应该不同，鲁棒性测试同一任务应该相同
- **互补性**: 
  - 高可区分性 + 高鲁棒性 = 理想（不同任务不同，同一任务相同）
  - 低可区分性 + 低鲁棒性 = 最差（既不能区分任务，也不能理解语义）

---

### 维度7: 方差分析（Variance）⭐⭐
**辅助指标 - 检测退化**

#### 定义
Prior输出的方差（在多个任务上）

#### 计算方法
```python
def compute_prior_variance(prior_model, tasks):
    z_goals = [prior_model(task) for task in tasks]
    z_goals = np.array(z_goals)  # [n_tasks, d]
    
    # 每个维度的方差
    variance_per_dim = np.var(z_goals, axis=0)
    mean_variance = np.mean(variance_per_dim)
    
    return mean_variance
```

#### 参考值
- **过低**: < 0.0001（可能退化，所有任务输出相同）
- **正常**: 0.0001 - 0.01
- **过高**: > 0.01（可能不稳定）

#### 配合可区分性使用
- 低方差 + 低可区分性 → **退化警告**
- 低方差 + 高可区分性 → 正常（输出紧凑但有区分）

---

## 实施建议

### 优先级
1. **维度2（Policy成功率对比）** - 最直接，立即实施
2. **维度1（目标准确性）** - 核心指标，需要收集成功画面数据
3. **维度6（语义鲁棒性）** - 重要且易实施，只需准备指令变体
4. **维度3（一致性）** - 简单易实施
5. **维度5（可区分性）** - 多任务评估时实施
6. **维度4（跨模态检索）** - 需要构建视觉数据库，后期实施
7. **维度7（方差分析）** - 配合可区分性使用

### 数据需求
1. **成功画面数据集**：
   - 每个任务收集10-20个成功的trial
   - 提取最后10帧或奖励最高时刻的帧
   - 用MineCLIP编码为视觉嵌入
   - 保存为 `data/success_visual_embeds/{task_id}.pkl`

2. **多任务文本列表**：
   - 至少20个不同的任务
   - 用于可区分性和方差分析

### 代码修改
1. **修改 `policy_metrics.py`**：
   - 删除 `text_to_prior_similarity`
   - 添加 `goal_accuracy`、`prior_quality_gap`、`consistency`等

2. **修改 `steve1_prior_evaluator.py`**：
   - 添加成功画面提取逻辑
   - 实现新的指标计算

3. **修改 `policy_html_generator.py`**：
   - 更新Prior评估部分的说明
   - 添加新指标的可视化

---

## 技术细节

### MineCLIP空间分析
```python
# MineCLIP有两个encoder
text_embed = mineclip.encode_text("chop tree")     # [512]
visual_embed = mineclip.encode_visual(frame)       # [512]

# 它们通过对比学习对齐，但不是同一空间
# Prior应该输出到visual空间
z_goal = prior_model(text_embed)  # 应该接近 visual_embed

# ✅ 正确比较
similarity = cosine(z_goal, visual_embed)

# ❌ 错误比较
similarity = cosine(z_goal, text_embed)  # 无意义
```

### 成功画面定义
**选项1**: 最后N帧的平均
```python
success_frames = trajectory[-10:]  # 最后10帧
z_visual = mean([mineclip.encode_visual(f) for f in success_frames])
```

**选项2**: 奖励最高时刻
```python
max_reward_idx = np.argmax(rewards)
success_frame = trajectory[max_reward_idx]
z_visual = mineclip.encode_visual(success_frame)
```

**建议**: 两者结合，取最后5帧 + 奖励最高帧的平均

---

## 总结

### 关键洞察
1. **Prior评估的核心**：是否能准确"想象"目标画面（在视觉空间中）
2. **错误的做法**：直接比较文本嵌入和Prior输出（跨空间无意义）
3. **正确的做法**：Prior输出 vs 真实成功画面的视觉嵌入
4. **最直接验证**：用Prior的z_goal跑Policy，看成功率差异

### 下一步行动
1. 收集成功画面数据集（优先）
2. 实现Policy成功率对比实验（最直接）
3. 重构Prior指标定义
4. 更新文档和可视化

---

**作者**: AI Assistant  
**日期**: 2025-11-27  
**版本**: v1.0

