# Prior评估系统实施总结

## 概述

基于用户的核心需求——"评估Prior模型能否将文字准确转换为目标画面"，我们重新设计并实现了正确的Prior评估系统。

**时间**: 2025-11-27  
**状态**: ✅ 已完成实施，可直接使用

---

## 核心修正

### ❌ 之前的错误实现

```python
# policy_metrics.py (之前)
text_embed = mineclip.encode_text("chop tree")      # 文本空间
prior_embed = prior_model(text_embed)               # 视觉空间
similarity = cosine(text_embed, prior_embed)        # ❌ 跨空间比较，无意义！
```

**问题**：
- MineCLIP的文本编码器和视觉编码器输出到不同的嵌入空间
- 直接比较跨空间的嵌入没有语义意义
- 无法反映Prior是否准确"想象"出目标画面

### ✅ 正确的实现

```python
# steve1_prior_evaluator.py (现在)
z_goal = prior_model(text)                          # 视觉空间（Prior输出）
z_visual = mineclip.encode_visual(success_frame)    # 视觉空间（真实画面）
goal_accuracy = cosine(z_goal, z_visual)            # ✅ 同空间比较，有意义！
```

**优势**：
- 同空间比较，有明确的语义意义
- 直接衡量Prior是否"想象"对了目标画面
- 可以指导模型优化和数据收集

---

## 新增功能

### 1. 四个核心评估维度

| 维度 | 指标 | 参考值 | 数据需求 |
|------|------|--------|----------|
| 目标准确性 | `goal_accuracy` | >0.6优秀 | 成功画面嵌入 |
| 语义鲁棒性 | `semantic_robustness` | >0.9优秀 | 指令变体 |
| 一致性 | `consistency` | >0.95优秀 | 无（多次采样） |
| 可区分性 | `discriminability` | >0.5优秀 | 无（多任务） |

### 2. 数据收集工具

**脚本**: `scripts/collect_success_visuals.py`

**功能**:
- 从评估结果中自动提取成功trial的视频帧
- 用MineCLIP编码为视觉嵌入
- 保存为.pkl格式供Prior评估使用

**用法**:
```bash
python scripts/collect_success_visuals.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output data/success_visual_embeds/all_tasks_20251121_214545.pkl \
    --last-n-frames 16 \
    --max-reward-frame
```

### 3. 指令变体配置

**文件**: `data/instruction_variants.json`

**功能**: 定义同一任务的不同表述，用于评估语义鲁棒性

**示例**:
```json
{
  "harvest_1_log": {
    "canonical": "chop tree, get a log",
    "variants": [
      "chop tree",
      "chop tree and get wood",
      "cut down a tree to obtain a log",
      "harvest wood from tree",
      "chop tree then get one log"
    ]
  }
}
```

**用户建议**: 用户建议增加这个维度，非常合理！测试Prior是否理解任务本质，而不是记忆表面词汇。

### 4. 重写的评估器

**文件**: `src/evaluation/steve1_prior_evaluator.py`

**重大变更**:
- 删除了错误的`text_to_prior_similarity`指标
- 实现了4个正确的评估维度
- 支持从已有评估结果中加载成功画面
- 支持指令变体评估
- 增加了退化检测

**核心方法**:
```python
class Steve1PriorEvaluator:
    def compute_goal_accuracy(task_id, instruction)
    def compute_consistency(instruction, n_samples)
    def compute_semantic_robustness(task_id)
    def analyze_prior_model(task_ids, n_samples, output_dir)
```

### 5. 一键运行脚本

**文件**: `scripts/run_prior_evaluation.sh`

**功能**: 自动完成完整的评估流程
1. 收集成功画面嵌入
2. 运行Prior评估
3. 生成报告

**用法**:
```bash
bash scripts/run_prior_evaluation.sh
```

---

## 文件清单

### 新增文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/evaluation/steve1_prior_evaluator.py` | 代码 | 重写的Prior评估器（正确版本） |
| `src/evaluation/prior_metrics.py` | 代码 | 正确的Prior指标定义和计算 |
| `scripts/collect_success_visuals.py` | 脚本 | 收集成功画面嵌入 |
| `src/evaluation/prior_eval_framework.py` | 代码 | Prior评估主框架 |
| `scripts/run_prior_evaluation.sh` | 脚本 | 一键运行评估流程 |
| `data/instruction_variants.json` | 配置 | 指令变体定义 |
| `docs/guides/PRIOR_EVALUATION_GUIDE.md` | 文档 | 完整使用指南 |
| `docs/technical/PRIOR_EVALUATION_REDESIGN.md` | 文档 | 详细设计文档 |
| `docs/technical/PRIOR_METRICS_COMPARISON.md` | 文档 | 错误vs正确对比 |
| `docs/summaries/PRIOR_EVALUATION_ANALYSIS_SUMMARY.md` | 文档 | 分析总结 |

### 需要修改的现有文件（未来）

| 文件 | 需要修改 | 说明 |
|------|----------|------|
| `src/evaluation/policy_metrics.py` | 删除 | 删除错误的`text_to_prior_similarity` |
| `src/evaluation/policy_eval_framework.py` | 更新 | 使用新的Prior评估器 |
| `src/utils/policy_html_generator.py` | 更新 | 更新Prior部分的说明和可视化 |

---

## 使用流程

### 快速开始（推荐）

```bash
# 一键运行完整流程（从评估结果）
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/all_tasks_20251121_214545

# 从已提取的成功画面运行
bash scripts/run_prior_evaluation.sh \
    --success-visuals data/success_visual_embeds/all_tasks.pkl \
    --output-dir results/prior_evaluation/custom
```

### 分步骤运行

#### 步骤1: 收集成功画面
```bash
python scripts/collect_success_visuals.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output data/success_visual_embeds/all_tasks_20251121_214545.pkl
```

#### 步骤2: 运行Prior评估
```python
from src.evaluation.steve1_prior_evaluator import Steve1PriorEvaluator
from pathlib import Path
import pickle

# 加载任务列表
with open('data/success_visual_embeds/all_tasks_20251121_214545.pkl', 'rb') as f:
    success_visuals = pickle.load(f)
task_ids = list(success_visuals.keys())

# 初始化评估器
evaluator = Steve1PriorEvaluator(
    prior_weights='data/weights/steve1/steve1_prior.pt',
    success_visuals_path='data/success_visual_embeds/all_tasks_20251121_214545.pkl',
    instruction_variants_path='data/instruction_variants.json',
)

# 运行评估
summary = evaluator.analyze_prior_model(
    task_ids=task_ids,
    n_samples=10,
    output_dir=Path('results/prior_evaluation/all_tasks_20251121_214545')
)
```

#### 步骤3: 查看结果
```bash
# 查看JSON报告
cat results/prior_evaluation/all_tasks_20251121_214545/prior_evaluation_summary.json

# 或在Python中
import json
with open('results/prior_evaluation/.../prior_evaluation_summary.json') as f:
    summary = json.load(f)
print(f"平均目标准确性: {summary['avg_goal_accuracy']:.4f}")
```

---

## 输出示例

### 控制台输出

```
================================================================================
初始化 Steve1PriorEvaluator（正确版本）...
================================================================================
加载 MineCLIP...
✓ MineCLIP 已加载
加载 Prior VAE: data/weights/steve1/steve1_prior.pt
✓ Prior VAE 已加载
加载成功画面嵌入: data/success_visual_embeds/all_tasks_20251121_214545.pkl
✓ 已加载 32 个任务的成功画面
加载指令变体: data/instruction_variants.json
✓ 已加载 8 个任务的指令变体
================================================================================
✓ Steve1PriorEvaluator 初始化完成
================================================================================

开始分析 Prior 模型 (32 个任务)...
================================================================================
分析任务: harvest_1_log
  目标准确性: 0.7234 (基于 3 个成功画面)
  一致性: 0.9567 (10 次采样)
  语义鲁棒性: 0.8912 (6 个变体)

分析任务: harvest_1_dirt
  目标准确性: 0.6891 (基于 5 个成功画面)
  一致性: 0.9234 (10 次采样)
  语义鲁棒性: 0.9123 (5 个变体)

...

可区分性（跨任务）: 0.5432
退化检测: ✓ Prior未退化

================================================================================
✓ Prior 模型分析完成
================================================================================

================================================================================
Prior 评估总结
================================================================================
任务数: 32
平均目标准确性: 0.6789
平均一致性: 0.9234
平均语义鲁棒性: 0.8567
可区分性: 0.5432

指标解读:
  目标准确性 > 0.6: 优秀,  0.4-0.6: 良好,  < 0.4: 需改进
  一致性 > 0.95: 优秀,  0.85-0.95: 良好,  < 0.85: 需改进
  语义鲁棒性 > 0.9: 优秀,  0.7-0.9: 良好,  < 0.7: 需改进
  可区分性 > 0.5: 优秀,  0.3-0.5: 良好,  < 0.3: 需改进
================================================================================
```

### JSON报告（部分）

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
      "n_variants": 6,
      "prior_embed_dim": 512,
      "visual_embed_dim": 512
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

## 技术亮点

### 1. 正确的空间比较

```python
# ✅ 同空间比较（视觉空间）
z_goal = prior_model(text)                    # [512] 视觉空间
z_visual = mineclip.encode_visual(frame)      # [512] 视觉空间
similarity = cosine(z_goal, z_visual)         # 有意义！

# ❌ 跨空间比较
text_embed = mineclip.encode_text(text)       # [512] 文本空间
similarity = cosine(text_embed, z_goal)       # 无意义！
```

### 2. 成功画面提取策略

**组合策略**：
- 最后N帧（默认16帧，与STEVE-1论文中MineCLIP设计一致）：捕获任务完成时的典型画面
- 最大奖励帧：捕获任务成功的关键时刻
- 平均嵌入：提高鲁棒性，减少单帧噪声

### 3. 语义鲁棒性评估（用户建议）

**创新点**：
- 测试Prior是否理解任务本质，而不是记忆表面词汇
- 通过指令变体（paraphrases）评估
- 实用价值高：真实使用时用户表述多样

### 4. 退化检测

**检测条件**：
```python
if discriminability < 0.3 and variance < 0.0001:
    # Prior退化（所有任务输出相同）
```

**预警机制**：
- 自动检测Prior是否退化为常数输出
- 提供改进建议

---

## 对比总结

| 方面 | 之前（错误） | 现在（正确） |
|------|-------------|-------------|
| 核心指标 | text_to_prior_similarity | goal_accuracy |
| 比较对象 | 文本嵌入 vs Prior输出 | Prior输出 vs 真实画面 |
| 空间 | 跨空间（无意义） | 同空间（视觉空间） |
| 数据需求 | 只需文本 | 需要成功画面 |
| 语义鲁棒性 | 无 | 有（用户建议） |
| 退化检测 | 无 | 有 |
| 指导优化 | 不明确 | 明确 |

---

## 后续工作（建议）

### 短期（1周内）

1. ✅ **收集成功画面数据**（已支持）
   ```bash
   bash scripts/run_prior_evaluation.sh
   ```

2. ⏳ **扩展指令变体**
   - 当前只有8个任务
   - 建议扩展到所有任务
   - 可以使用GPT生成paraphrase

3. ⏳ **集成到主评估流程**
   - 修改`policy_eval_framework.py`
   - 自动运行Prior评估
   - 集成到HTML报告

### 中期（1个月内）

4. ⏳ **Policy成功率对比（Ablation Study）**
   - 实现维度2：Policy成功率对比
   - 实验A：使用Prior
   - 实验B：使用真实视觉嵌入
   - 直接量化Prior对性能的影响

5. ⏳ **跨模态检索评估**
   - 构建视觉数据库
   - 实现检索准确率评估

6. ⏳ **可视化增强**
   - t-SNE/UMAP可视化Prior嵌入空间
   - 指令变体的相似度热力图
   - 目标准确性分布图

### 长期（持续优化）

7. ⏳ **Prior模型优化**
   - 基于评估结果优化Prior训练
   - 增加高质量text-visual配对数据
   - 调整模型架构

8. ⏳ **自动化数据收集**
   - 自动运行任务收集成功画面
   - 自动生成指令变体
   - 持续更新评估数据

---

## 关键洞察

### 1. Prior评估的本质

> Prior模型的目标是：将文本"翻译"成目标画面的视觉表示

因此，评估时应该问：
- **Prior "想象"的画面** vs **真实成功画面** 有多像？

而不是：
- ~~文本嵌入 vs Prior输出~~ （跨空间，无意义）

### 2. 语义鲁棒性的重要性（用户洞察）

用户建议的"同一指令不同表述的一致性"非常关键：
- 测试Prior是否理解任务本质
- 而不是记忆表面词汇
- 实用价值高：用户表述多样

### 3. 评估维度的互补性

| 维度 | 测试什么 | 理想状态 |
|------|----------|----------|
| 一致性 | 同一表述多次采样 | 应该相似 |
| 语义鲁棒性 | 同一任务不同表述 | 应该相似 |
| 可区分性 | 不同任务 | 应该不同 |

**理想模型**：高一致性 + 高鲁棒性 + 高可区分性

---

## 致谢

本系统的设计受益于用户的深刻洞察：
1. **核心问题**：识别文字->目标画面的准确性
2. **关键建议**：增加语义鲁棒性维度（指令变体一致性）

这些建议直击Prior评估的本质，使我们能够设计出正确且实用的评估系统。

---

## 参考文档

- **使用指南**: `docs/guides/PRIOR_EVALUATION_GUIDE.md`
- **详细设计**: `docs/technical/PRIOR_EVALUATION_REDESIGN.md`
- **指标对比**: `docs/technical/PRIOR_METRICS_COMPARISON.md`
- **分析总结**: `docs/summaries/PRIOR_EVALUATION_ANALYSIS_SUMMARY.md`
- **代码实现**: `src/evaluation/prior_metrics.py`

---

**作者**: AI Assistant  
**日期**: 2025-11-27  
**状态**: ✅ 已完成实施，可直接使用

