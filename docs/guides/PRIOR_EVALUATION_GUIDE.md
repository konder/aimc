# Prior模型评估指南

## 概述

本指南介绍如何使用正确的方法评估Steve1的Prior模型 p(z_goal|y)。

**核心思想**：Prior模型的目标是将文本指令"翻译"成目标画面的视觉嵌入。因此，评估Prior时应该比较：
- Prior输出的z_goal（视觉空间）
- 真实成功画面的视觉嵌入（视觉空间）

**不应该**：直接比较文本嵌入和Prior输出（跨空间，无意义）

---

## 评估维度

### 1. 目标准确性（Goal Accuracy）⭐⭐⭐⭐⭐
**最核心指标**

**定义**：Prior生成的z_goal与真实成功画面的视觉嵌入之间的余弦相似度

**含义**：Prior能否准确"想象"出完成任务时的画面

**参考值**：
- 优秀: ≥ 0.6
- 良好: 0.4 - 0.6
- 需改进: < 0.4

---

### 2. 语义鲁棒性（Semantic Robustness）⭐⭐⭐⭐
**表述变体一致性**

**定义**：同一任务用不同表述时，Prior输出的z_goal相似度

**示例**：
```
"chop tree"
"chop tree and get wood"
"cut down a tree to obtain a log"
"harvest wood from tree"
```
这些表述的目标画面应该相同，Prior应该为它们生成相似的z_goal

**参考值**：
- 优秀: ≥ 0.9
- 良好: 0.7 - 0.9
- 需改进: < 0.7

---

### 3. 一致性（Consistency）⭐⭐⭐⭐
**采样稳定性**

**定义**：同一指令多次输入Prior，输出的z_goal相似度

**参考值**：
- 优秀: ≥ 0.95
- 良好: 0.85 - 0.95
- 需改进: < 0.85

---

### 4. 可区分性（Discriminability）⭐⭐⭐
**多任务评估**

**定义**：不同任务的z_goal是否足够不同（避免退化）

**参考值**：
- 优秀: ≥ 0.5
- 良好: 0.3 - 0.5
- 需改进: < 0.3（模型可能退化）

---

## 快速开始

### 前提条件

1. 已有评估结果（包含成功trial的视频帧）
2. 已安装Steve1模型和MineCLIP
3. 准备了指令变体配置文件（可选）

### 一键运行

```bash
# 从评估结果运行（自动提取成功画面）
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/all_tasks_20251121_214545

# 从已提取的成功画面运行
bash scripts/run_prior_evaluation.sh \
    --success-visuals data/success_visual_embeds/all_tasks.pkl \
    --output-dir results/prior_evaluation/custom

# 自定义所有参数
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/all_tasks_20251121_214545 \
    --success-visuals data/success_visual_embeds/all_tasks.pkl \
    --instruction-variants data/instruction_variants.json \
    --prior-weights data/weights/steve1/steve1_prior.pt \
    --n-samples 10 \
    --last-n-frames 16
```

这个脚本会自动执行：
1. 从评估结果中提取成功画面的视觉嵌入（或使用已提取的）
2. 运行Prior模型评估
3. 生成评估报告

---

## 分步骤运行

### 步骤1: 准备指令变体（可选）

创建或编辑 `data/instruction_variants.json`：

```json
{
  "harvest_1_log": {
    "canonical": "chop tree, get a log",
    "variants": [
      "chop tree",
      "chop tree and get wood",
      "cut down a tree to obtain a log",
      "harvest wood from tree"
    ]
  },
  "harvest_1_dirt": {
    "canonical": "dig dirt",
    "variants": [
      "dig dirt",
      "dig dirt block",
      "dig and collect dirt",
      "obtain dirt by digging"
    ]
  }
}
```

---

### 步骤2: 运行Prior评估

```bash
# 直接从评估结果运行（自动提取和缓存成功画面）
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/all_tasks_20251121_214545

# 或使用Python直接调用
python src/evaluation/prior_eval_framework.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/all_tasks_20251121_214545 \
    --n-samples 10 \
    --last-n-frames 16
```

**注意**：
- 成功画面嵌入会自动提取并缓存到输出目录
- 第二次运行时会自动使用缓存，无需重新提取
- 缓存文件名：`success_visuals_<eval_result_dir_name>.pkl`

---

## 输出文件

评估完成后，会在输出目录生成：

### `prior_evaluation_summary.json`

```json
{
  "task_results": [
    {
      "task_id": "harvest_1_log",
      "instruction": "chop tree, get a log",
      "goal_accuracy": 0.7234,
      "consistency": 0.9567,
      "semantic_robustness": 0.8912,
      "n_success_visuals": 3,
      "n_samples": 10,
      "n_variants": 4
    }
  ],
  "discriminability": 0.5432,
  "avg_goal_accuracy": 0.6789,
  "avg_consistency": 0.9234,
  "avg_semantic_robustness": 0.8567,
  "is_degraded": false,
  "degradation_warning": "✓ Prior未退化"
}
```

---

## 结果解读

### 指标等级

| 指标 | 优秀 | 良好 | 需改进 | 含义 |
|------|------|------|--------|------|
| 目标准确性 | ≥0.6 | 0.4-0.6 | <0.4 | Prior"想象"的画面与真实画面的相似度 |
| 一致性 | ≥0.95 | 0.85-0.95 | <0.85 | 同一指令多次采样的稳定性 |
| 语义鲁棒性 | ≥0.9 | 0.7-0.9 | <0.7 | 不同表述的一致性 |
| 可区分性 | ≥0.5 | 0.3-0.5 | <0.3 | 不同任务的差异程度 |

### 常见问题诊断

#### 1. 目标准确性低（<0.4）
**原因**：Prior无法准确"想象"目标画面

**改进方法**：
- 重新训练Prior，增加高质量text-visual配对数据
- 调整text_cond_scale参数
- 增加Prior模型容量

#### 2. 一致性低（<0.85）
**原因**：Prior输出不稳定

**改进方法**：
- 增加Prior训练轮数
- 减小VAE采样噪声
- 检查Prior是否欠拟合

#### 3. 语义鲁棒性低（<0.7）
**原因**：Prior过度依赖表层词汇

**改进方法**：
- 增加paraphrase数据训练
- 使用数据增强（同义词替换）
- 增强MineCLIP的文本编码器

#### 4. 可区分性低（<0.3）
**原因**：Prior可能退化（所有任务输出相似）

**改进方法**：
- 检查训练数据是否多样化
- 检查Prior权重是否加载正确
- 重新训练Prior

#### 5. 退化警告
**症状**：可区分性<0.3 且 方差<0.0001

**诊断**：Prior退化为输出常数（所有任务输出相同）

**紧急处理**：
- 检查Prior模型权重
- 检查训练数据
- 重新训练Prior

---

## 高级用法

### 只评估特定维度

```python
evaluator = Steve1PriorEvaluator(
    prior_weights='data/weights/steve1/steve1_prior.pt',
    success_visuals_path='data/success_visual_embeds/...',  # 只加载需要的
    instruction_variants_path=None,  # 不评估语义鲁棒性
)
```

### 自定义采样次数

```python
summary = evaluator.analyze_prior_model(
    task_ids=task_ids,
    n_samples=20,  # 增加采样次数以获得更稳定的一致性评估
)
```

### 批量评估多个Prior模型

```python
prior_models = [
    'data/weights/steve1/steve1_prior.pt',
    'data/weights/steve1/steve1_prior_v2.pt',
    'data/weights/steve1/steve1_prior_v3.pt',
]

for prior_path in prior_models:
    evaluator = Steve1PriorEvaluator(
        prior_weights=prior_path,
        success_visuals_path='...',
    )
    
    summary = evaluator.analyze_prior_model(
        task_ids=task_ids,
        output_dir=Path(f'results/prior_evaluation/{Path(prior_path).stem}')
    )
```

---

## 技术细节

### MineCLIP空间说明

```python
# MineCLIP有两个encoder
mineclip = MineCLIP()

# 文本编码器 → 文本空间
text_embed = mineclip.encode_text("chop tree")     # [512]

# 视觉编码器 → 视觉空间
visual_embed = mineclip.encode_image(frame)       # [512]

# Prior的输出在视觉空间
z_goal = prior_model(text_embed)  # [512]

# ✅ 正确：同空间比较
similarity = cosine(z_goal, visual_embed)

# ❌ 错误：跨空间比较
similarity = cosine(z_goal, text_embed)  # 无意义！
```

### 成功画面定义

**选择策略**：
1. 最后N帧（默认16帧，与STEVE-1论文中MineCLIP使用的16帧视频一致）
2. 最大奖励帧

**组合方式**：
- 提取所有选中帧
- 用MineCLIP编码为视觉嵌入
- 取平均作为该trial的成功画面嵌入

**原因**：
- 最后16帧：与MineCLIP训练时使用的视频帧数一致，捕获任务完成时的典型画面
- 最大奖励帧：捕获任务成功的关键时刻
- 平均多帧：提高鲁棒性，减少单帧噪声

---

## 常见问题（FAQ）

### Q1: 为什么不能直接比较文本嵌入和Prior输出？

**A**: 它们在不同的嵌入空间。这就像比较"中文句子"和"英文句子"的字符相似度，即使相似也不代表意思相同。Prior的输出应该在视觉空间，需要和真实画面比较。

### Q2: 如果没有成功的trial怎么办？

**A**: 目标准确性无法计算（会返回0.0）。但仍可以评估一致性、语义鲁棒性和可区分性。建议选择至少有一些成功trial的任务集。

### Q3: 指令变体如何准备？

**A**: 可以：
1. 手工编写（推荐，质量高）
2. 使用GPT等大模型生成paraphrase
3. 使用同义词替换工具

注意保持语义等价（目标画面应该相同）。

### Q4: 评估需要多长时间？

**A**: 取决于任务数量和采样次数：
- 收集成功画面：约1-2分钟/任务（取决于帧数）
- Prior评估：约5-10秒/任务（n_samples=10时）

示例：32个任务，约5-10分钟完成整个流程。

### Q5: 可以用于其他模型吗？

**A**: 评估框架具有通用性。只要你的模型：
1. 输入文本，输出目标嵌入
2. 有成功任务的画面数据
3. 有对应的视觉编码器

就可以使用类似的评估方法。

---

## 相关文档

- **详细设计**: `docs/technical/PRIOR_EVALUATION_REDESIGN.md`
- **指标对比**: `docs/technical/PRIOR_METRICS_COMPARISON.md`
- **总结文档**: `docs/summaries/PRIOR_EVALUATION_ANALYSIS_SUMMARY.md`
- **代码实现**: `src/evaluation/prior_metrics.py`

---

**作者**: AI Assistant  
**日期**: 2025-11-27  
**版本**: v1.0

