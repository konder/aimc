# Prior 模型评估快速入门指南

> STEVE-1 Prior Model Evaluation Quickstart Guide

本指南帮助你快速开始 Prior 模型评估，生成完整的评估报告。

---

## 📋 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [配置文件](#配置文件)
- [评估指标](#评估指标)
- [报告解读](#报告解读)
- [常见问题](#常见问题)

---

## 概述

### 什么是 Prior 评估？

Prior 是 STEVE-1 模型的第一部分，负责将文本指令转换为 MineCLIP 视觉嵌入（或"类视觉"嵌入）。评估 Prior 质量对于了解模型性能至关重要。

### 评估框架

我们的评估框架包含 **3个维度** 和 **7个核心指标**：

**维度1：内在质量 (Intrinsic Quality)**
- 1.1 输出稳定性 (Consistency)
- 1.2 语义鲁棒性 (Semantic Robustness)
- 1.3 输出多样性 (Output Diversity)
- 1.4 区分度保持率 (Discriminability Preservation)

**维度2：输出质量 (Output Quality)**
- 2.1 目标对齐度 (Goal Alignment)
- 2.2 Prior增益 (Prior Gain)
- 2.3 跨模态一致性 (Cross-Modal Consistency)

**维度3：可控性 (Controllability)**
- 注意：CFG 是 Policy 级别的概念，在 Prior 评估中禁用

---

## 快速开始

### 方法1：使用脚本（推荐）

```bash
# 激活环境
conda activate minedojo

# 切换到项目目录
cd /Users/nanzhang/aimc

# 运行评估（使用正式配置）
bash scripts/run_prior_eval.sh --config config/eval_tasks_prior.yaml

# 或者使用 harvest 任务集
bash scripts/run_prior_eval.sh --config config/eval_tasks_harvest.yaml
```

### 方法2：直接使用 Python

```bash
conda activate minedojo
cd /Users/nanzhang/aimc

python src/evaluation/prior_eval_framework.py \
    --config config/eval_tasks_prior.yaml
```

### 方法3：使用修复版脚本（如果环境激活有问题）

```bash
bash scripts/run_prior_eval_fixed.sh --config config/eval_tasks_prior.yaml
```

---

## 配置文件

### 可用配置

| 配置文件 | 任务数 | 用途 | 评估时间 |
|---------|-------|------|---------|
| `config/eval_tasks_prior.yaml` | 8个 | 正式评估（推荐） | ~5分钟 |
| `config/eval_tasks_harvest.yaml` | 12个 | Harvest任务集 | ~8分钟 |

### 配置文件结构

```yaml
# 全局配置
global:
  prior_weights: "data/weights/steve1/steve1_prior.pt"
  mineclip_config: "data/weights/mineclip"
  output_dir: "results/prior_evaluation/official"
  
  report:
    generate_html: true
    generate_json: true
    generate_plots: true

# 评估维度
evaluation_dimensions:
  intrinsic_quality:
    enabled: true
    consistency:
      enabled: true
      n_samples: 10  # 一致性评估采样次数
  
  output_quality:
    enabled: true
    goal_alignment:
      enabled: true
      use_reward_head: true  # 使用MineCLIP reward_head
  
  controllability:
    enabled: false  # Prior不支持CFG

# 任务配置
tasks:
  - task_id: harvest_1_log
    instruction: "chop tree and get a log"
    instruction_variants:
      - "chop tree"
      - "cut down a tree"
    success_visuals_path: "data/visual_prompt_embeds/wood.pkl"
```

### 自定义配置

复制现有配置并修改：

```bash
# 复制模板
cp config/eval_tasks_prior.yaml config/my_eval_config.yaml

# 编辑配置
vim config/my_eval_config.yaml

# 运行评估
bash scripts/run_prior_eval.sh --config config/my_eval_config.yaml
```

---

## 评估指标

### 维度1：内在质量

#### 1.1 输出稳定性 (Consistency)

**定义**：同一指令多次采样的相似度

**计算方法**：
```python
# 对同一指令采样10次
embeddings = [get_prior_embed(instruction) for _ in range(10)]
# 计算两两之间的余弦相似度
consistency = mean(cosine_similarity(embeddings))
```

**阈值**：
- 优秀：≥ 0.95
- 良好：≥ 0.85
- 需改进：< 0.85

**解读**：
- **高稳定性**（~0.99）：Prior输出非常可靠
- **中等稳定性**（0.85-0.95）：可接受，但有改进空间
- **低稳定性**（< 0.85）：Prior可能有问题

---

#### 1.2 语义鲁棒性 (Semantic Robustness)

**定义**：同一任务不同表述的一致性

**示例**：
- "chop tree" vs "cut down tree" vs "harvest wood"
- 应该产生相似的嵌入

**计算方法**：
```python
# 对同一任务的不同指令变体
variants = ["chop tree", "cut down tree", "harvest wood"]
embeddings = [get_prior_embed(v) for v in variants]
robustness = mean(cosine_similarity(embeddings))
```

**阈值**：
- 优秀：≥ 0.90
- 良好：≥ 0.70
- 需改进：< 0.70

**解读**：
- **高鲁棒性**（~0.95）：Prior理解语义，不依赖具体用词
- **中等鲁棒性**（0.70-0.90）：基本理解，但对措辞敏感
- **低鲁棒性**（< 0.70）：Prior可能在记忆文本而非理解语义

---

#### 1.3 输出多样性 (Output Diversity)

**定义**：不同任务输出的方差

**计算方法**：
```python
# 收集所有任务的Prior输出
all_embeddings = [get_prior_embed(task.instruction) for task in tasks]
# 计算每个维度的方差，然后平均
diversity = mean(variance(all_embeddings, axis=0))
```

**阈值**：
- 适中：> 0.0001
- 偏低：≤ 0.0001

**解读**：
- **适度多样性**：不同任务有明显区分
- **多样性太低**（~0.00001）：所有任务输出过于相似（潜在退化）
- **多样性太高**（> 0.001）：可能不稳定

⚠️ **注意**：单任务评估此指标无意义

---

#### 1.4 区分度保持率 (Discriminability Preservation)

**定义**：Prior 输出相对于文本输入的区分度变化

**计算方法**：
```python
# 区分度 = 1 - 平均相似度
text_discriminability = 1 - mean(similarity(text_embeddings))
prior_discriminability = 1 - mean(similarity(prior_embeddings))
preservation_rate = prior_discriminability / text_discriminability
```

**阈值**：
- 保持/放大：≥ 1.0
- 轻微压缩：≥ 0.5
- 严重压缩：< 0.5

**解读**：
- **保持率 > 1.0**：Prior放大了任务差异（好！）
- **保持率 = 1.0**：Prior保持了原有区分度
- **保持率 < 1.0**：Prior压缩了任务差异（可能有问题）

⚠️ **注意**：单任务评估此指标无意义

---

### 维度2：输出质量

#### 2.1 目标对齐度 (Goal Alignment)

**定义**：Prior 输出与真实成功画面的相似度

**计算方法**：
```python
# 使用 MineCLIP 的 reward_head
prior_embed = get_prior_embed(instruction)
success_visuals = load_pkl(success_visuals_path)  # 任务成功时的游戏画面
similarity = mineclip.forward_reward_head(prior_embed, success_visuals).mean()
```

**阈值**：
- 优秀：≥ 0.60
- 良好：≥ 0.40
- 需改进：< 0.40

**解读**：
- **高对齐度**（> 0.60）：Prior准确指向正确目标
- **中等对齐度**（0.40-0.60）：基本正确，可以优化
- **低对齐度**（< 0.40）：Prior可能指向错误目标

⚠️ **注意**：需要提供 `success_visuals_path`

---

#### 2.2 Prior 增益 (Prior Gain)

**定义**：Prior 相对于直接使用文本嵌入的改进

**计算方法**：
```python
# 方案A：Prior(文本) → 视觉嵌入
alignment_prior = similarity(prior_embed, success_visuals)

# 方案B：MineCLIP(文本) → 文本嵌入直接用作视觉嵌入
text_embed = mineclip.encode_text(instruction)
alignment_text = similarity(text_embed, success_visuals)

# Prior增益
prior_gain = alignment_prior - alignment_text
```

**阈值**：
- 显著提升：> 0.05
- 轻微提升：> 0
- 负增益：≤ 0

**解读**：
- **正增益**：Prior有价值，比直接文本更好
- **零增益**：Prior没有带来改进
- **负增益**：Prior反而降低了对齐度（需要调查）

---

#### 2.3 跨模态一致性 (Cross-Modal Consistency)

**定义**：Prior 输出是否在视觉空间

**计算方法**：
```python
# 使用 Wasserstein 距离比较分布
distance = wasserstein_distance(
    distribution(prior_embeddings),
    distribution(visual_embeddings)
)
consistency_score = 1.0 / (1.0 + distance)
```

**阈值**：
- 高度一致：≥ 0.70
- 基本一致：≥ 0.50
- 分布偏离：< 0.50

**解读**：
- **高一致性**：Prior输出接近真实视觉嵌入分布
- **低一致性**：Prior可能生成"伪视觉"嵌入

---

## 报告解读

### 报告结构

评估完成后会生成：

```
results/prior_evaluation/[output_dir]/
├── prior_evaluation_results.json          # JSON格式结果
├── prior_evaluation_report.html           # HTML可视化报告
└── prior_evaluation_visualization.png     # 可视化图表
```

### 打开HTML报告

```bash
# macOS
open results/prior_evaluation/official/prior_evaluation_report.html

# Linux
xdg-open results/prior_evaluation/official/prior_evaluation_report.html

# 或者直接用浏览器打开
```

### HTML报告内容

#### 1. 📊 评估总结

顶部卡片展示关键指标的平均值：
- 平均一致性
- 平均语义鲁棒性
- 输出多样性
- 平均目标对齐度
- 平均 Prior 增益

#### 2. 📖 指标说明与解读

详细解释每个指标的含义、计算方法和解读标准

#### 3. 📊 可视化图表

包含4张子图：
- **左上 - 相似度矩阵**：任务间的 Prior 输出相似度
- **右上 - t-SNE降维**：Prior 输出在2D空间的分布
- **左下 - PCA降维**：主成分分析
- **右下 - 方差分布**：每个维度的方差直方图

#### 4. 📐 维度详细结果

每个维度展开显示：
- 具体指标值
- 解释徽章（优秀/良好/需改进）
- 任务级详细数据表格

---

## 常见问题

### Q1: 如何添加新任务？

编辑配置文件，添加新任务：

```yaml
tasks:
  - task_id: my_new_task
    instruction: "my instruction"
    instruction_variants:
      - "variant 1"
      - "variant 2"
    success_visuals_path: "data/visual_prompt_embeds/my_task.pkl"
    metadata:
      category: "my_category"
      difficulty: "easy"
```

### Q2: 如何创建 success visuals？

使用 `success_visual_extractor.py`：

```bash
python src/utils/extract_success_visuals_by_task.py \
    --task-id my_new_task \
    --video-dir path/to/videos \
    --output data/visual_prompt_embeds/my_task.pkl
```

详见：`docs/guides/EXTRACT_SUCCESS_VISUALS_QUICKSTART.md`

### Q3: 评估时间太长怎么办？

**方法1：减少采样次数**

```yaml
evaluation_dimensions:
  intrinsic_quality:
    consistency:
      n_samples: 5  # 默认10，改为5
```

**方法2：减少任务数量**

只评估关键任务，或创建一个小型测试配置

**方法3：禁用某些维度**

```yaml
evaluation_dimensions:
  intrinsic_quality:
    enabled: true
  output_quality:
    enabled: false  # 禁用这个维度
  controllability:
    enabled: false
```

### Q4: success_visuals_path 文件不存在怎么办？

**临时解决**：使用通用的 `dirt.pkl`

```yaml
success_visuals_path: "data/visual_prompt_embeds/dirt.pkl"
```

**正式解决**：为每个任务创建专门的 success visuals（推荐）

### Q5: 输出多样性为0是否正常？

**单任务评估**：是的，正常。单任务无法计算多样性。

**多任务评估**：不正常，说明所有任务输出过于相似，可能是模型退化。

### Q6: 区分度保持率是什么意思？

**保持率 > 1.0**：Prior 放大了输入差异
- 例如：文本区分度0.1，Prior区分度0.15，保持率1.5x
- 解释：Prior 让不同任务更容易区分

**保持率 < 1.0**：Prior 压缩了输入差异
- 例如：文本区分度0.2，Prior区分度0.1，保持率0.5x
- 解释：Prior 让不同任务变得更相似（潜在问题）

### Q7: 如何解读 Prior 增益为负？

**负增益示例**：
- Prior 对齐度：0.50
- 直接文本对齐度：0.55
- Prior 增益：-0.05

**可能原因**：
1. Prior 训练不足或训练有问题
2. Prior 训练数据与测试任务不匹配
3. success visuals 不准确

**解决方案**：
1. 检查 Prior 训练日志
2. 重新训练 Prior
3. 验证 success visuals 是否正确

### Q8: CFG 在哪里配置？

**重要**：CFG（Classifier-Free Guidance）是 Policy 层面的概念，Prior 不支持。

Prior 评估中 `controllability` 维度始终禁用：

```yaml
evaluation_dimensions:
  controllability:
    enabled: false
```

CFG 应该在 Policy 评估中测试。

---

## 下一步

### 优化 Prior

基于评估结果优化 Prior：

1. **一致性低**：增加训练稳定性（学习率、正则化）
2. **鲁棒性低**：增加数据增强、指令变体
3. **对齐度低**：调整训练目标、增加成功样本
4. **负增益**：检查训练流程、数据质量

### 完整评估流程

```
1. Prior 评估 (当前)
   ├── 内在质量
   ├── 输出质量
   └── 生成基线指标

2. Policy 评估 (下一步)
   ├── 动作质量
   ├── CFG 敏感度
   └── 任务完成率

3. End-to-End 评估
   └── 真实环境测试
```

---

## 参考文档

- **评估框架设计**：`docs/design/STEVE1_COMPLETE_EVALUATION_FRAMEWORK.md`
- **指标详解**：`docs/technical/PRIOR_METRICS_COMPARISON.md`
- **成功画面提取**：`docs/guides/EXTRACT_SUCCESS_VISUALS_QUICKSTART.md`
- **指令变体生成**：`docs/guides/INSTRUCTION_VIDEO_PAIRS_GUIDE.md`

---

## 联系支持

如有问题，请参考：
- 项目 README：`/Users/nanzhang/aimc/README.md`
- FAQ：`/Users/nanzhang/aimc/FAQ.md`
- Issues 文档：`/Users/nanzhang/aimc/docs/issues/`

---

**版本**: v1.0  
**最后更新**: 2025-12-01  
**维护者**: AIMC Team

