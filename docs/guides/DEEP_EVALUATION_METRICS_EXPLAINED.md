# Steve1 深度评估指标详解

## 📖 **导读**

本文档详细解释 Steve1 深度评估系统的所有指标、如何理解它们，以及应该关注哪些关键指标。

---

## 🎯 **核心概念：两阶段模型**

Steve1 将复杂的"文本 → 动作序列"问题分解为两个阶段：

```
输入文本 y ("chop tree")
    ↓
【阶段1: Prior 模型】p(z^τ_goal | y)
    将文本嵌入转换为"视觉风格"的目标嵌入
    ↓
目标嵌入 z^τ_goal
    ↓
【阶段2: Policy 模型】p(τ | z^τ_goal)
    根据目标嵌入生成动作序列
    ↓
动作序列 τ = [a₁, a₂, ..., aₜ]
```

**为什么要两阶段？**
- Policy 模型在**视觉嵌入**上训练（MineCLIP 图像特征）
- 推理时只有**文本**（"chop tree"），没有目标图像
- Prior 模型**桥接这个域差异**：文本 → "假装的"视觉嵌入

---

## 📊 **指标体系总览**

### 1. **Prior 模型指标**
评估 Prior 将文本转换为目标嵌入的质量

### 2. **Policy 模型指标**
评估 Policy 根据目标嵌入执行任务的能力

### 3. **端到端指标**
评估完整系统的性能，识别瓶颈

---

## 🎨 **一、Prior 模型指标**

### 1.1 **文本-Prior 相似度（Text-to-Prior Similarity）**

#### **定义**
```python
similarity = cosine_similarity(
    MineCLIP_text_embedding,
    Prior_output_embedding
)
```

MineCLIP 文本嵌入与 Prior 输出嵌入的余弦相似度。

#### **计算方式**
1. 用 MineCLIP 编码文本 → 得到 512 维文本嵌入
2. 用 Prior 模型转换文本嵌入 → 得到 512 维"视觉风格"嵌入
3. 计算两者的余弦相似度：`cos(θ) = (a·b) / (||a|| ||b||)`

#### **数值范围**
- **-1 到 +1**（余弦相似度）
- 实际范围通常在 **0.3 - 0.6**

#### **如何理解**

| 相似度值 | 含义 | 评价 |
|---------|------|------|
| **0.5 - 0.6** | Prior 输出与文本嵌入高度对齐 | ✅ 优秀 |
| **0.4 - 0.5** | 中等对齐，有改进空间 | ⚠️ 良好 |
| **0.3 - 0.4** | 对齐较弱，Prior 可能是瓶颈 | ⚠️ 需改进 |
| **< 0.3** | 严重不对齐，Prior 失效 | ❌ 差 |

#### **为什么重要**

- **高相似度** → Prior 保留了文本的语义信息
- **低相似度** → Prior 可能"曲解"了文本意图
- 这是 Prior 质量的**最直接指标**

#### **示例**

```python
# 您的评估结果
avg_text_to_prior_similarity: 0.374

# 解读
→ 相似度 0.374 处于"良好-需改进"区间
→ Prior 保留了部分语义，但有较大改进空间
→ 建议：增加训练数据或调整 CVAE 架构
```

---

### 1.2 **相似度范围（Similarity Range）**

#### **定义**
```python
min_similarity = min(similarity for each instruction)
max_similarity = max(similarity for each instruction)
range = max_similarity - min_similarity
```

所有测试指令的相似度的最小值、最大值和跨度。

#### **如何理解**

| 指标 | 数值 | 含义 |
|------|------|------|
| **最小值** | 0.354 | 最差的指令对齐度 |
| **最大值** | 0.395 | 最好的指令对齐度 |
| **跨度** | 0.041 | 不同指令间的差异 |

**跨度分析**：
- **跨度小（< 0.05）** → Prior 对不同指令的处理**一致性高** ✅
- **跨度大（> 0.1）** → Prior 对某些指令特别好/差，**不稳定** ⚠️

#### **示例**

```python
min_similarity: 0.354
max_similarity: 0.395
→ 跨度 = 0.041 (小)
→ Prior 对不同指令的处理较为一致 ✅
```

---

### 1.3 **Prior 输出方差（Prior Output Variance）**

#### **定义**
```python
variance = np.var([prior_embed_1, prior_embed_2, ...])
```

Prior 对不同指令的输出嵌入的方差。

#### **如何理解**

| 方差 | 含义 | 评价 |
|------|------|------|
| **过高** | 不同指令的输出差异很大 | ⚠️ 可能过拟合 |
| **适中** | 不同指令有合理区分 | ✅ 理想 |
| **过低** | 所有指令输出几乎相同 | ❌ Prior 失效（mode collapse） |

**理想状态**：
- 相似任务（如 "dig dirt" vs "dig sand"）→ 输出相似 ✅
- 不同任务（如 "dig dirt" vs "kill cow"）→ 输出有区分 ✅

---

### 1.4 **嵌入空间可视化（Embedding Space Visualization）**

#### **t-SNE 降维图**

**作用**：将 512 维嵌入降维到 2D，保留局部相似性

**如何解读**：
- 🔵 **蓝色点**：MineCLIP 文本嵌入
- 🟢 **绿色点**：Prior 输出嵌入
- ➡️ **箭头**：文本 → Prior 的转换

**好的模式**：
```
✅ 箭头指向相似区域（如都在"采集"聚类中）
✅ 相似任务在空间中靠近
✅ 转换方向一致
```

**坏的模式**：
```
❌ 箭头方向混乱
❌ 相似任务分散
❌ 绿点聚成一团（mode collapse）
```

#### **PCA 降维图**

**作用**：保留最大方差方向

**如何解读**：
- 与 t-SNE 相比，PCA 更关注**全局结构**
- 观察主成分方向是否有语义意义

---

### 1.5 **相似度矩阵（Similarity Matrix）**

#### **作用**
显示不同指令的 Prior 输出之间的相似度

#### **如何解读**

```
           dig_dirt  dig_sand  kill_cow  ...
dig_dirt     1.00     0.85      0.32
dig_sand     0.85     1.00      0.30
kill_cow     0.32     0.30      1.00
```

**期望模式**：
- **对角线**：应该是 1.0（自己与自己）
- **相似任务**：高相似度（> 0.7）
  - "dig dirt" vs "dig sand" → 0.85 ✅
- **不同类任务**：低相似度（< 0.5）
  - "dig dirt" vs "kill cow" → 0.32 ✅

**问题信号**：
- 所有值都很高（> 0.9）→ Prior 输出过于相似 ❌
- 相似任务值很低 → Prior 没有学到语义 ❌

---

## 🎮 **二、Policy 模型指标**

### 2.1 **动作多样性（Action Diversity）**

#### **定义**
```python
diversity = entropy(action_distribution)
```

动作序列的熵值，衡量动作的多样性。

#### **如何理解**

| 多样性 | 含义 | 评价 |
|-------|------|------|
| **高** | 使用多种动作组合 | ✅ Policy 灵活 |
| **中** | 合理的动作组合 | ✅ 正常 |
| **低** | 重复使用少数几个动作 | ⚠️ 可能卡住 |

**示例**：
```python
# 高多样性（好）
actions = ['forward', 'left', 'attack', 'jump', 'right', ...]
diversity = 2.5 bits

# 低多样性（差）
actions = ['forward', 'forward', 'forward', ...]
diversity = 0.3 bits
```

---

### 2.2 **时序一致性（Temporal Consistency）**

#### **定义**
动作序列的平滑度，相邻动作的相似性。

#### **如何理解**

**高一致性**：
```
步骤: forward → forward → attack → attack
→ 动作连贯，有明确意图 ✅
```

**低一致性**：
```
步骤: forward → back → left → right → jump → ...
→ 动作混乱，可能是随机行为 ❌
```

---

### 2.3 **重复动作比例（Repeated Action Ratio）**

#### **定义**
```python
ratio = count(action[i] == action[i+1]) / total_transitions
```

连续相同动作的比例。

#### **如何理解**

| 比例 | 含义 | 评价 |
|------|------|------|
| **20-40%** | 合理的持续行为 | ✅ 正常 |
| **> 60%** | 卡在循环中 | ❌ 异常 |
| **< 10%** | 动作过于跳跃 | ⚠️ 可能混乱 |

**示例**：
```python
# 正常（30%）
[forward, forward, attack, attack, attack, left, ...]
→ 持续前进、持续攻击 ✅

# 异常（80%）
[forward]*50 + [attack]*30 + ...
→ 卡在墙上持续前进 ❌
```

---

## 🎯 **三、端到端指标**

### 3.1 **成功率（Success Rate）**

#### **定义**
```python
success_rate = successful_trials / total_trials
```

任务完成的百分比。

#### **如何理解**

| 成功率 | 评价 |
|-------|------|
| **90-100%** | ✅ 优秀 |
| **70-90%** | ✅ 良好 |
| **50-70%** | ⚠️ 需改进 |
| **< 50%** | ❌ 有严重问题 |

---

### 3.2 **Prior 贡献 / Policy 贡献**

#### **定义**

通过**消融实验**（Ablation Study）计算各阶段的贡献：

```python
# 实验1: 使用 Prior 嵌入（正常）
success_with_prior = run_task(use_prior=True)

# 实验2: 使用真实视觉嵌入（上限）
success_with_visual = run_task(use_visual=True)

# 计算贡献
if success_with_visual > success_with_prior:
    prior_contribution = success_with_prior / success_with_visual
    policy_contribution = 1 - prior_contribution
else:
    # 两者相同，平分贡献
    prior_contribution = 0.5
    policy_contribution = 0.5
```

#### **如何理解**

**场景1：Prior 贡献低**
```
success_with_prior = 50%
success_with_visual = 90%
→ Prior 贡献 = 50/90 = 56%
→ Policy 贡献 = 44%

解读：Prior 是瓶颈，限制了性能
建议：改进 Prior 模型 ⚠️
```

**场景2：Policy 贡献低**
```
success_with_prior = 80%
success_with_visual = 85%
→ Prior 贡献 = 94%
→ Policy 贡献 = 6%

解读：Prior 很好，但 Policy 即使有完美嵌入也不行
建议：改进 Policy 模型 ⚠️
```

**场景3：均衡**
```
success_with_prior = 90%
success_with_visual = 95%
→ 两者都很好 ✅
```

---

### 3.3 **瓶颈分析（Bottleneck Analysis）**

#### **定义**

识别哪个阶段限制了整体性能：

| 瓶颈类型 | 判断条件 | 含义 |
|---------|---------|------|
| **无瓶颈** | `success_with_prior ≈ success_with_visual` | 两阶段都好 ✅ |
| **Prior 瓶颈** | `success_with_prior << success_with_visual` | Prior 拖后腿 ⚠️ |
| **Policy 瓶颈** | `success_with_visual` 本身低 | Policy 不行 ⚠️ |

#### **可视化解读**

**饼图**：
```
🟢 无瓶颈: 80%     → 大部分任务表现良好
🔵 Prior瓶颈: 15%  → 少数任务受 Prior 限制
🟠 Policy瓶颈: 5%  → 极少数任务 Policy 有问题
```

**建议**：
- Prior 瓶颈多 → 优先改进 Prior
- Policy 瓶颈多 → 优先改进 Policy
- 无瓶颈多 → 继续扩展到更难的任务

---

## 🎓 **四、如何解读您的评估结果**

### 步骤1：查看总览

```json
{
  "avg_success_rate": 1.0,           // 100% 成功 ✅
  "avg_text_to_prior_similarity": 0.374,  // 相似度 0.374 ⚠️
  "prior_contribution": 0.5,          // 50% 贡献
  "policy_contribution": 0.5          // 50% 贡献
}
```

**快速判断**：
- ✅ 成功率高（100%）→ 模型整体工作良好
- ⚠️ Prior 相似度偏低（0.374）→ 有改进空间

---

### 步骤2：深入 Prior 分析

查看可视化：
1. **t-SNE/PCA 图**
   - 转换箭头是否合理？
   - 聚类是否有语义意义？

2. **相似度矩阵**
   - 相似任务是否高相似度？
   - 不同类任务是否低相似度？

3. **质量指标图**
   - 相似度分布是否集中？
   - 方差是否适中？

---

### 步骤3：查看端到端分析

1. **成功率**
   - 整体成功率如何？
   - 哪些任务失败了？

2. **瓶颈分析**
   - Prior 和 Policy 哪个是瓶颈？
   - 瓶颈分布如何？

3. **贡献度**
   - 两阶段贡献是否均衡？
   - 是否某个阶段明显拖后腿？

---

### 步骤4：根据建议改进

系统会自动生成建议，例如：

```
[HIGH] Prior模型的文本-嵌入相似度较低 (0.374)
→ 考虑重新训练Prior模型，或增加训练数据
```

**改进优先级**：
1. **HIGH** → 立即处理
2. **MEDIUM** → 重要但不紧急
3. **LOW** → 可以稍后优化

---

## 🎯 **五、关键指标优先级**

### **必看指标** ⭐⭐⭐

1. **成功率**
   - 最直接的性能指标
   - 低于 70% → 有严重问题

2. **文本-Prior 相似度**
   - Prior 质量的核心指标
   - 低于 0.4 → Prior 需改进

3. **瓶颈分布**
   - 识别改进方向
   - 集中优化瓶颈阶段

### **重要指标** ⭐⭐

4. **Prior/Policy 贡献度**
   - 评估两阶段平衡性
   - 不平衡 → 优先改进弱项

5. **相似度范围**
   - 评估 Prior 稳定性
   - 跨度大 → 不稳定

6. **动作多样性**
   - Policy 灵活性
   - 过低 → Policy 可能卡住

### **辅助指标** ⭐

7. **重复动作比例**
   - 检测异常行为
   - 过高 → 可能有 bug

8. **时序一致性**
   - 评估行为连贯性
   - 辅助理解策略

---

## 📚 **六、常见问题 FAQ**

### Q1: 成功率 100%，为什么还说 Prior 有问题？

**A**: 成功率只看"能否完成任务"，不看"过程是否优雅"。

即使 Prior 相似度低（0.374），Policy 可能通过**强大的适应能力**仍然完成任务。但这意味着：
- Policy 需要**额外努力**补偿 Prior 的不足
- 换到**更难的任务**，Prior 可能成为瓶颈
- **改进 Prior** 可以让系统更稳定

---

### Q2: 相似度 0.374 到底算高还是低？

**A**: 对于 CVAE 模型，**0.3-0.4 是常见范围**，但有改进空间。

**参考标准**：
- 原始 MineCLIP 文本-图像相似度：~0.5-0.6
- Prior 输出应该接近这个水平
- 0.374 表示 Prior **保留了部分语义**，但不够理想

---

### Q3: 如何提升 Prior 相似度？

**A**: 几个方向：

1. **增加训练数据**
   - 更多文本-视觉配对数据
   - 更多样化的指令

2. **调整 CVAE 架构**
   - 增加隐藏层维度
   - 调整 latent dimension
   - 修改 KL divergence 权重

3. **改进训练策略**
   - 增加训练 epochs
   - 调整学习率
   - 使用更好的优化器

---

### Q4: 瓶颈分析为什么重要？

**A**: **确定改进方向**，避免浪费精力。

```
场景1: Prior 瓶颈占 80%
→ 优化 Prior 收益最大
→ 优化 Policy 效果有限 ❌

场景2: Policy 瓶颈占 80%
→ 优化 Policy 收益最大
→ 优化 Prior 效果有限 ❌

场景3: 无瓶颈占 80%
→ 两阶段都好 ✅
→ 可以尝试更难的任务
```

---

## 🚀 **七、下一步行动建议**

基于您的评估结果（成功率 100%，Prior 相似度 0.374）：

### **短期（1-2周）**

1. ✅ **扩展评估范围**
   ```bash
   # 测试更多任务
   bash scripts/run_deep_evaluation.sh \
       --max-tasks 20 --n-trials 5
   ```

2. ⚠️ **收集 Prior 相似度低的案例**
   - 查看哪些指令相似度最低
   - 分析共同特征
   - 针对性改进

### **中期（1-2月）**

3. 🔧 **改进 Prior 模型**
   - 增加训练数据
   - 调整架构参数
   - 重新训练

4. 📊 **对比新旧 Prior**
   - 使用相同的评估任务
   - 对比相似度提升
   - 对比成功率变化

### **长期（3-6月）**

5. 🎯 **挑战更难任务**
   - 多步骤任务
   - 长时间任务
   - 复杂组合任务

6. 📈 **建立性能基线**
   - 定期运行评估
   - 跟踪指标变化
   - 持续优化

---

## 📖 **相关文档**

- [Steve1 深度评估指南](STEVE1_DEEP_EVALUATION_GUIDE.md)
- [Prior 模型技术解析](../technical/STEVE1_PRIOR_EXPLAINED.md)
- [快速开始](../../DEEP_EVALUATION_QUICKSTART.md)

---

**最后更新**: 2025-11-27  
**适用版本**: Steve1 Deep Evaluation v1.0

