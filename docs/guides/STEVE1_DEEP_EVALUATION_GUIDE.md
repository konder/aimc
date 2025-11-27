# Steve1 两阶段模型深度评估指南

> **目标**: 深入评估和可视化 Steve1 论文中的两阶段模型  
> **创建日期**: 2025-11-27  
> **适合**: 想要深入理解 Steve1 模型性能和瓶颈的研究者

---

## 📋 目录

1. [概述](#概述)
2. [理论背景](#理论背景)
3. [评估维度](#评估维度)
4. [快速开始](#快速开始)
5. [详细使用](#详细使用)
6. [结果解读](#结果解读)
7. [常见问题](#常见问题)

---

## 🎯 概述

### Steve1 的两阶段分解

根据 Steve1 论文，模型分解为两个独立的概率模型：

```
p(τ|y) = p(z_goal^τ |y) × p(τ |z_goal^τ )
   ↑         ↑                ↑
总策略    阶段1: Prior     阶段2: Policy
```

**阶段1: Prior 模型** `p(z_goal|y)`
- **输入**: 文本指令 y（"chop tree"）
- **输出**: 目标嵌入 z_goal（512维MineCLIP嵌入）
- **作用**: 将文本指令转换为"类视觉"目标表示
- **实现**: Conditional VAE（TranslatorVAE）

**阶段2: 策略模型** `p(τ|z_goal)`
- **输入**: 目标嵌入 z_goal + 当前观察 obs
- **输出**: 动作序列 τ
- **作用**: 根据目标嵌入生成达成目标的动作
- **实现**: VPT + Classifier-Free Guidance

### 为什么需要深度评估？

传统评估只关注**端到端成功率**，但无法回答：

❌ **传统评估的局限**:
- 失败是 Prior 的问题还是策略的问题？
- Prior 能否正确理解指令？
- 策略能否基于正确的目标嵌入完成任务？
- 哪个阶段是瓶颈？

✅ **深度评估的优势**:
- 分别评估两个阶段的能力
- 识别性能瓶颈
- 可视化嵌入空间和动作分布
- 提供针对性的改进建议

---

## 📚 理论背景

### Prior 模型详解

**训练方式**:
```python
# 训练数据: (文本嵌入, 视觉嵌入) 对
for text_embed, visual_embed in dataset:
    # 编码
    mu, logvar = prior.encode(text_embed, visual_embed)
    
    # 采样
    z = mu + eps * exp(0.5 * logvar)
    
    # 解码（重建视觉嵌入）
    visual_recon = prior.decode(z, text_embed)
    
    # 损失
    loss = MSE(visual_recon, visual_embed) + KL(N(mu, var), N(0, 1))
```

**推理时**:
```python
text_embed = mineclip.encode_text("chop tree")
z_goal = prior.sample(text_embed)  # 采样"类视觉"嵌入
```

**关键指标**:
1. **文本-Prior相似度**: 衡量 Prior 输出与原始文本嵌入的对齐度
2. **Prior方差**: 衡量 Prior 输出的稳定性/多样性
3. **重建质量**: VAE 的重建能力
4. **嵌入空间结构**: 语义相近的指令是否在嵌入空间中也相近

### 策略模型详解

**训练方式**:
```python
# 事后重标记 (Hindsight Relabeling)
for episode in dataset:
    for t in range(T):
        future_t = sample(t+15, t+200)
        goal_embed = mineclip.encode_image(frames[future_t])
        
        # 训练策略: 给定当前obs和future goal，预测action
        loss = -log P(actions[t] | obs[t], goal_embed)
```

**推理时**:
```python
goal_embed = prior.sample(text)  # 或直接用文本嵌入
action = policy(obs, goal_embed)
```

**关键指标**:
1. **动作多样性**: 是否能生成多样化的动作
2. **时序一致性**: 动作序列是否连贯
3. **成功率**: 能否完成任务
4. **动作分布**: 各类动作的使用频率

---

## 📊 评估维度

### 1. Prior 模型评估

#### 1.1 嵌入空间可视化

**目的**: 理解 Prior 如何将文本映射到嵌入空间

**方法**:
- t-SNE/UMAP 降维到 2D
- 可视化文本嵌入和 Prior 输出的关系
- 观察语义相近的指令是否聚类

**示例输出**:
```
embedding_space_tsne.png:
  - 左图: MineCLIP 文本嵌入
  - 中图: Prior 输出嵌入
  - 右图: 文本 → Prior 的转换（箭头图）
```

**解读**:
- ✅ **好**: Prior 输出围绕文本嵌入，箭头短而一致
- ❌ **差**: Prior 输出远离文本嵌入，箭头长且分散

#### 1.2 相似度矩阵

**目的**: 检查语义相近的指令在嵌入空间中的距离

**示例输出**:
```
similarity_matrix.png:
  - 热力图显示所有指令对之间的相似度
```

**解读**:
- ✅ **好**: "chop tree" 和 "get wood" 相似度高（暖色）
- ❌ **差**: 语义相近的指令相似度低（冷色）

#### 1.3 质量指标分布

**目的**: 统计 Prior 的整体质量

**指标**:
```python
{
  "text_to_prior_similarity": 0.85,  # ✅ >0.8 优秀, 0.6-0.8 良好, <0.6 需改进
  "prior_variance": 0.02,            # ✅ <0.05 稳定, >0.1 不稳定
  "reconstruction_quality": 0.75     # ✅ >0.7 良好, <0.5 需改进
}
```

### 2. 策略模型评估

#### 2.1 动作分布

**目的**: 理解策略的行为模式

**示例输出**:
```
action_distribution.png:
  - 条形图: 各类动作的频率
  - 饼图: 动作占比
```

**解读**:
- ✅ **好**: 动作分布符合任务需求（如砍树任务中攻击动作占比高）
- ❌ **差**: 动作过于单一（如只会前进）或过于混乱

#### 2.2 性能指标

**指标**:
```python
{
  "action_diversity": 2.3,           # ✅ >2.0 多样, 1.0-2.0 一般, <1.0 单调
  "temporal_consistency": 0.7,       # ✅ >0.6 连贯, <0.4 不连贯
  "repeated_action_ratio": 0.15,     # ✅ <0.3 正常, >0.5 卡住
  "success_rate": 0.6                # ✅ >0.5 良好, <0.3 需改进
}
```

### 3. 端到端分析

#### 3.1 错误归因

**目的**: 识别失败是哪个阶段导致的

**方法**:
```python
对比实验:
  实验1: 文本 → Prior → 策略 (正常流程)
  实验2: 文本 → MineCLIP直接 → 策略 (跳过Prior)
  
分析:
  if 实验1成功 and 实验2失败:
    → Prior 有正面贡献
  elif 实验1失败 and 实验2成功:
    → Prior 是瓶颈
```

**示例输出**:
```json
{
  "failure_attribution": {
    "prior": 0.7,   # 70% 的失败归因于 Prior
    "policy": 0.3   # 30% 的失败归因于策略
  }
}
```

#### 3.2 瓶颈分析

**示例输出**:
```
bottleneck_distribution:
  - Prior 瓶颈: 45%
  - 策略瓶颈: 30%
  - 无瓶颈: 25%
```

**解读**:
- **Prior 瓶颈占主导** → 优先改进 Prior（重新训练、增加数据、换编码器）
- **策略瓶颈占主导** → 优先改进策略（DAgger迭代、调超参）
- **均衡分布** → 两者都需改进

---

## 🚀 快速开始

### 安装依赖

```bash
# 安装可视化依赖
pip install umap-learn seaborn scikit-learn
```

### 快速测试（3个任务，3次试验）

```bash
# macOS M 系列芯片（推荐使用封装脚本）
bash scripts/run_deep_evaluation.sh \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --max-tasks 3 \
    --n-trials 3 \
    --output-dir results/deep_evaluation_test
```

**预计时间**: 15-20分钟

**输出**:
```
results/deep_evaluation_test/
├── prior_analysis/
│   ├── prior_analysis.json
│   ├── embedding_space_tsne.png
│   ├── embedding_space_pca.png
│   ├── similarity_matrix.png
│   └── quality_metrics.png
├── end_to_end/
│   ├── harvest_1_log/
│   │   └── end_to_end_analysis.png
│   ├── harvest_1_dirt/
│   │   └── end_to_end_analysis.png
│   ├── harvest_1_sand/
│   │   └── end_to_end_analysis.png
│   └── task_comparison.png
├── summary_report.json
└── deep_evaluation_*.log
```

### 查看结果

```bash
# 1. 查看 Prior 嵌入空间
open results/deep_evaluation_test/prior_analysis/embedding_space_tsne.png

# 2. 查看任务对比
open results/deep_evaluation_test/end_to_end/task_comparison.png

# 3. 查看总结报告
cat results/deep_evaluation_test/summary_report.json | jq
```

---

## 📖 详细使用

### 评估特定任务集

```bash
# 评估 harvest 任务（13个任务）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --n-trials 5

# 评估 combat 任务（10个任务）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set combat_tasks \
    --n-trials 5

# 评估 techtree 任务（14个任务）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set techtree_tasks \
    --n-trials 5

# 评估所有任务（39个任务，需要较长时间）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set all_tasks \
    --n-trials 5 \
    --max-tasks 10  # 限制为10个任务
```

### 只评估 Prior 模型

```bash
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --skip-e2e  # 跳过端到端评估
```

**用途**: 快速检查 Prior 的嵌入质量，不运行环境

### 只评估策略模型

```bash
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --skip-prior  # 跳过 Prior 分析
```

### 自定义参数

```bash
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --n-trials 10 \
    --max-steps 2000 \
    --output-dir results/my_evaluation \
    --prior-weights data/weights/steve1/my_prior.pt \
    --seed 123 \
    --verbose
```

---

## 🔍 结果解读

### Prior 分析示例

假设你得到了以下结果：

```json
{
  "prior_analysis": {
    "avg_text_to_prior_similarity": 0.65,
    "min_similarity": 0.45,
    "max_similarity": 0.85
  }
}
```

**解读**:
- 平均相似度 0.65 → **中等水平**，有改进空间
- 最低 0.45 → 某些指令的 Prior 输出质量较差
- 最高 0.85 → Prior 对某些指令理解较好

**改进建议**:
1. 检查相似度低的指令（查看 `quality_metrics.png` 中的 Bottom 5）
2. 考虑增加 Prior 训练数据
3. 尝试更强的文本编码器（如中文场景用 Chinese-CLIP）

### 端到端分析示例

假设你得到了以下结果：

```json
{
  "end_to_end_analysis": {
    "avg_success_rate": 0.35,
    "avg_stage1_contribution": 0.65,
    "avg_stage2_contribution": 0.35,
    "bottleneck_distribution": {
      "prior_bottleneck": 12,
      "policy_bottleneck": 5,
      "no_bottleneck": 3
    }
  }
}
```

**解读**:
- 成功率 35% → 整体性能需要提升
- Prior 贡献 65% > 策略贡献 35% → **Prior 是主要瓶颈**
- Prior 瓶颈出现 12 次 > 策略瓶颈 5 次 → 确认 Prior 是问题

**改进建议**:
1. **优先级1**: 改进 Prior 模型
   - 重新训练 Prior（增加数据量）
   - 使用多语言编码器（如果用中文）
   - 调整 VAE 架构（增加容量）

2. **优先级2**: 改进策略模型
   - 进行 DAgger 迭代
   - 调整 CFG scale
   - 微调 VPT 权重

### 可视化结果解读

#### 嵌入空间图（t-SNE）

**好的例子**:
```
[Text Embeddings]    [Prior Embeddings]    [Transformation]
    ●  ●                 ●  ●                  ●→●
  ●  ●  ●             ●  ●  ●                ●→●
    ●  ●                 ●  ●                  ●→●

语义相近的点聚在一起，箭头短而一致
```

**差的例子**:
```
[Text Embeddings]    [Prior Embeddings]    [Transformation]
●       ●                     ●              ●────────→●
  ●   ●                ●           ●         ●─→●
    ●                        ●               ●──────────→●

点分散，箭头长短不一，方向混乱
```

#### 动作分布图

**好的例子（砍树任务）**:
```
forward:  ████████ 40%
attack:   ██████████ 50%
jump:     ██ 10%

说明: Agent 主要执行前进和攻击，符合砍树逻辑
```

**差的例子（砍树任务）**:
```
forward:  ████ 20%
back:     ████ 20%
left:     ████ 20%
right:    ████ 20%
sneak:    ████ 20%

说明: 动作过于均匀，Agent 可能在随机游走
```

---

## ❓ 常见问题

### Q1: 评估需要多长时间？

**答**: 取决于任务数量和试验次数

| 配置 | 任务数 | 试验数 | 预计时间 |
|------|--------|--------|----------|
| 快速测试 | 3 | 3 | 15-20分钟 |
| 单个分类 | 10-14 | 5 | 1-2小时 |
| 所有任务 | 39 | 5 | 4-6小时 |

**建议**: 先用 `--max-tasks 3` 快速测试

### Q2: 如何理解 "Prior 贡献" vs "策略贡献"？

**答**: 
- **Prior 贡献高**: 使用 Prior 后成功率提升明显，说明 Prior 的作用大
- **策略贡献高**: 即使跳过 Prior（直接用文本嵌入），策略也能完成任务，说明策略很强

### Q3: Prior 相似度多少算好？

**答**:
- **>0.8**: 优秀
- **0.6-0.8**: 良好
- **0.4-0.6**: 一般，需改进
- **<0.4**: 较差，建议重新训练

### Q4: 如何改进 Prior 模型？

**答**:
1. **增加训练数据**: 收集更多 (文本, 视频) 对
2. **调整架构**: 增加 VAE 的隐藏维度
3. **换文本编码器**: 使用 Chinese-CLIP（中文场景）
4. **调整训练超参**: β-VAE 的 β 值

### Q5: 如何改进策略模型？

**答**:
1. **DAgger 迭代**: 收集 on-policy 数据并重新训练
2. **调整 CFG scale**: 尝试不同的 text_cond_scale 和 visual_cond_scale
3. **微调**: 在目标任务上 fine-tune VPT
4. **增强训练**: 使用数据增强（随机crop、翻转等）

### Q6: 能否只可视化 Prior，不运行环境？

**答**: 可以，使用 `--skip-e2e`:

```bash
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set all_tasks \
    --skip-e2e
```

这样只分析 Prior 的嵌入空间，速度很快（几分钟）

### Q7: 如何对比不同的 Prior 模型？

**答**: 分别评估并对比 `prior_analysis.json`:

```bash
# 评估 Prior A
python src/evaluation/run_steve1_deep_evaluation.py \
    --prior-weights data/weights/steve1/prior_a.pt \
    --output-dir results/prior_a \
    --skip-e2e

# 评估 Prior B
python src/evaluation/run_steve1_deep_evaluation.py \
    --prior-weights data/weights/steve1/prior_b.pt \
    --output-dir results/prior_b \
    --skip-e2e

# 对比
diff <(jq .prior_analysis results/prior_a/prior_analysis.json) \
     <(jq .prior_analysis results/prior_b/prior_analysis.json)
```

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| `STEVE1_PRIOR_EXPLAINED.md` | Prior 模型的详细原理 |
| `STEVE1_TRAINING_EXPLAINED.md` | Steve1 训练流程 |
| `STEVE1_QUICK_REFERENCE.md` | Steve1 快速参考 |
| `EVAL_TASKS_COMPREHENSIVE.yaml` | 评估任务配置 |

---

## 🎓 高级用法

### 自定义评估指标

编辑 `src/evaluation/steve1_deep_analyzer.py`，添加自定义指标：

```python
def _compute_custom_metric(self, actions: List[Dict]) -> float:
    """自定义指标"""
    # 例如: 计算移动效率
    total_movement = sum(
        action.get('forward', 0) + action.get('back', 0) 
        for action in actions
    )
    return total_movement / len(actions)
```

### 导出嵌入用于进一步分析

```python
import numpy as np

# 读取保存的嵌入
text_embeds = np.load('results/deep_evaluation/prior_analysis/text_embeds.npy')
prior_embeds = np.load('results/deep_evaluation/prior_analysis/prior_embeds.npy')

# 进一步分析（聚类、分类等）
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(prior_embeds)
```

---

## 📊 总结

深度评估工具帮助你：

✅ **理解** Prior 如何将文本映射到嵌入空间  
✅ **评估** 策略模型的动作质量和成功率  
✅ **识别** 性能瓶颈（Prior vs 策略）  
✅ **可视化** 嵌入空间和动作分布  
✅ **获得** 针对性的改进建议

**下一步**:
1. 运行快速测试熟悉工具
2. 分析你关心的任务集
3. 根据建议改进模型
4. 重新评估验证效果

