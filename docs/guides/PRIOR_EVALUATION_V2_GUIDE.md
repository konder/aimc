# Prior 评估框架 V2 使用指南

**日期**: 2025-12-01  
**版本**: 2.0  
**目的**: 实现完整的第一层 Prior 评估

---

## 🎯 概述

Prior 评估框架 V2 实现了完整的第一层评估，包括三个维度：

1. **内在质量** (Intrinsic Quality): 评估 Prior 输出本身的特性
2. **输出质量** (Output Quality): 评估 Prior 输出与真实目标的对齐度
3. **可控性** (Controllability): 评估 CFG 对 Prior 的影响（实际测试需要端到端评估）

---

## 📁 文件结构

```
src/evaluation/
├── prior_eval_framework_v2.py          # 主评估框架
└── steve1_prior_evaluator.py           # Prior评估器（已改进）

src/utils/
└── prior_html_generator_v2.py          # HTML报表生成器

scripts/
└── run_prior_evaluation_v2.sh          # 运行脚本

config/
├── eval_tasks_comprehensive.yaml       # 完整任务配置
├── eval_tasks_prior_example.yaml       # 配置示例
└── eval_tasks_prior_test.yaml          # 测试配置

docs/guides/
└── PRIOR_EVALUATION_V2_GUIDE.md        # 本文档
```

---

## 🚀 快速开始

### 1. 准备配置文件

您需要一个包含 Prior 评估字段的 YAML 配置文件。最简单的方法是使用测试配置：

```bash
# 查看测试配置
cat config/eval_tasks_prior_test.yaml
```

### 2. 准备成功画面嵌入

成功画面嵌入文件应该放在 `data/visual_prompt_embeds/` 目录下：

```
data/visual_prompt_embeds/
├── wood.pkl          # harvest_1_log 的成功画面
├── dirt.pkl          # harvest_1_dirt 的成功画面
├── flower.pkl        # harvest_1_flower 的成功画面
└── ...
```

**嵌入文件格式**：
- 文件类型：`.pkl` (pickle)
- 内容：numpy 数组，shape 为 `(N, 512)`，其中 N 是成功画面数量
- 或者：字典格式 `{'embeds': np.array, ...}`

### 3. 运行评估

**基本使用（评估所有任务）**：

```bash
bash scripts/run_prior_evaluation_v2.sh
```

**使用测试配置（快速测试）**：

```bash
bash scripts/run_prior_evaluation_v2.sh \
    --config config/eval_tasks_prior_test.yaml \
    --output results/prior_eval_test
```

**评估特定任务**：

```bash
bash scripts/run_prior_evaluation_v2.sh \
    --tasks "harvest_1_log harvest_1_dirt" \
    --output results/prior_eval_selected
```

**自定义参数**：

```bash
bash scripts/run_prior_evaluation_v2.sh \
    --config config/eval_tasks_comprehensive.yaml \
    --output results/prior_eval_custom \
    --success-visuals-dir data/custom_embeds \
    --n-samples 20 \
    --prior-weights data/weights/steve1/custom_prior.pt
```

---

## 📊 输出结果

评估完成后，会在输出目录生成以下文件：

```
results/prior_evaluation/20251201_123456/
├── prior_evaluation_20251201_123456.json           # JSON 报告
└── prior_evaluation_report_20251201_123456.html    # HTML 报告
```

### JSON 报告结构

```json
{
  "timestamp": "20251201_123456",
  "n_tasks": 3,
  "avg_intrinsic_quality": {
    "consistency": 0.9987,
    "semantic_robustness": 0.9685,
    "output_diversity": 0.0968,
    "discriminability": 0.1227,
    "discriminability_preservation": 9.23,
    "text_discriminability": 0.0133
  },
  "avg_output_quality": {
    "goal_alignment_mean": 0.9421,
    "goal_alignment_std": 0.0847,
    "prior_gain": 0.1235
  },
  "task_results": [
    {
      "task_id": "harvest_1_log",
      "instruction": "chop tree, get a log",
      "intrinsic_quality": {...},
      "output_quality": {...},
      "controllability": {...}
    },
    ...
  ],
  "is_degraded": false,
  "degradation_warnings": []
}
```

### HTML 报告特点

- 📊 **可视化指标**: 使用卡片和进度条展示各项指标
- 🎨 **美观设计**: 渐变色背景、动画效果、响应式布局
- 📈 **详细分析**: 包含执行摘要、维度评估、任务详情
- 💡 **改进建议**: 根据评估结果自动生成改进建议
- ⚠️ **退化检测**: 高亮显示退化警告

**查看 HTML 报告**：

```bash
# macOS
open results/prior_evaluation/*/prior_evaluation_report_*.html

# Linux
xdg-open results/prior_evaluation/*/prior_evaluation_report_*.html

# Windows
start results/prior_evaluation/*/prior_evaluation_report_*.html
```

---

## 📝 YAML 配置说明

### 全局配置

```yaml
prior_evaluation:
  # 评估维度开关
  enable_intrinsic_quality: true      # 内在质量评估
  enable_output_quality: true         # 输出质量评估
  enable_controllability: true        # 可控性评估
  
  # 评估参数
  n_consistency_samples: 10           # 一致性评估的采样次数
  cfg_scales: [0, 1, 3, 6, 9, 12]    # CFG敏感度测试的scale值
  
  # 数据路径
  success_visuals_base_dir: "data/visual_prompt_embeds"  # 成功画面嵌入基础目录
```

### 任务配置

每个任务需要添加以下 Prior 评估字段：

```yaml
harvest_tasks:
  - task_id: harvest_1_log
    en_instruction: chop tree, get a log
    
    # ========== Prior 评估专用字段 ==========
    
    # 成功画面嵌入文件名（相对于 success_visuals_base_dir）
    success_visual_embed: "wood.pkl"
    
    # 指令变体（用于语义鲁棒性评估）
    instruction_variants:
      - "chop tree, get a log"
      - "cut down tree"
      - "get wood from tree"
      - "harvest log"
    
    # CFG 测试配置（可选，覆盖全局配置）
    cfg_test:
      enabled: true
      scales: [0, 3, 6, 9]
    
    # ========== 其他字段（保持不变） ==========
    env_name: MineRLHarvestDefaultEnv-v0
    env_config: {...}
    max_steps: 300
    n_trials: 5
```

**字段说明**：

1. **success_visual_embed**: 
   - 成功画面嵌入文件名
   - 完整路径 = `{success_visuals_base_dir}/{success_visual_embed}`
   - 如果没有，设置为 `null`（将跳过输出质量评估）

2. **instruction_variants**:
   - 同一任务的不同表述
   - 用于语义鲁棒性评估
   - 建议 3-5 个变体
   - 如果没有，设置为空列表 `[]`

3. **cfg_test**:
   - 可控性评估：测试不同 CFG scale
   - 可以在任务级别覆盖全局配置
   - `enabled=false` 可以禁用特定任务的 CFG 测试

---

## 🔬 评估维度详解

### 维度1: 内在质量 (Intrinsic Quality)

评估 Prior 输出本身的特性。

| 指标 | 定义 | 计算方法 | 期望值 |
|------|------|----------|--------|
| **输出稳定性** (Consistency) | 同一指令多次采样的一致性 | 多次采样 z_goal，计算两两相似度 | > 0.95 (优秀) |
| **语义鲁棒性** (Semantic Robustness) | 指令变体的一致性 | 不同表述的 z_goal 相似度 | > 0.85 (优秀) |
| **输出多样性** (Output Diversity) | 不同任务输出的方差 | 所有任务 z_goal 的方差 | > 0.0001 |
| **任务可区分性** (Discriminability) | 不同任务的 z_goal 差异 | 1 - 任务间平均相似度 | > 0.3 (良好) |
| **区分度保持率** | Prior vs 文本的区分度比率 | Prior 区分度 / 文本区分度 | > 0.8 |

**解释**：

- **一致性高**（0.999）说明 Prior 输出稳定（推理时使用均值，应该是确定性的）
- **鲁棒性高**（0.9685）说明 Prior 能理解不同表述的同一语义
- **多样性高**（0.097）说明不同任务有不同的 z_goal（未退化）
- **可区分性**（0.12）看起来低，但要对比文本区分度（0.013）
- **保持率**（9.2x）说明 Prior 实际上**放大了**文本的区分度

---

### 维度2: 输出质量 (Output Quality)

评估 Prior 输出与真实目标的对齐度。

| 指标 | 定义 | 计算方法 | 期望值 |
|------|------|----------|--------|
| **目标对齐度** (Goal Alignment) | Prior 输出 vs 成功画面相似度 | 使用 `forward_reward_head` 计算 | > 0.6 (优秀) |
| **Prior 增益** (Prior Gain) | Prior vs 直接文本的改进 | 对齐度_prior - 对齐度_text | > 0.05 |

**计算方法**（改进版）：

```python
# 使用 MineCLIP 的 forward_reward_head
z_goal_tensor = torch.from_numpy(z_goal).unsqueeze(0)  # [1, 512]
z_visuals_tensor = torch.from_numpy(success_visual_embeds)  # [N, 512]

logits_per_goal, _ = mineclip.forward_reward_head(
    z_goal_tensor,
    text_tokens=z_visuals_tensor
)

# 先计算每个成功画面的相似度，再平均
similarities = logits_per_goal.squeeze(0).cpu().numpy()
goal_alignment_mean = np.mean(similarities)
goal_alignment_std = np.std(similarities)
```

**解释**：

- **高对齐度**（> 0.6）说明 Prior 输出接近真实成功画面
- **正增益**（> 0.05）说明 Prior 比直接文本更好，值得使用
- **负增益**（< 0）说明 Prior 在拖后腿，需要重新训练

---

### 维度3: 可控性 (Controllability)

评估 CFG 对 Prior 的影响。

**注意**: 完整的可控性测试需要在端到端评估中进行（需要 Policy 和环境）。

当前实现只记录 CFG 配置，实际的任务成功率测试需要：

```python
# 伪代码
for cfg_scale in [0, 1, 3, 6, 9, 12]:
    agent.reset(cond_scale=cfg_scale)
    success_rate = run_trials(task, agent, env, n_trials=10)
    # 对比不同 CFG scale 的成功率
```

---

## 📈 指标解读

### 优秀 (Excellent) ✅

- 一致性 > 0.95
- 语义鲁棒性 > 0.9
- 可区分性 > 0.5
- 目标对齐度 > 0.6
- Prior 增益 > 0.1

**说明**: Prior 模型表现优秀，可以继续使用。

---

### 良好 (Good) 💡

- 一致性 0.85 - 0.95
- 语义鲁棒性 0.7 - 0.9
- 可区分性 0.3 - 0.5
- 目标对齐度 0.4 - 0.6
- Prior 增益 0.05 - 0.1

**说明**: Prior 模型表现良好，可以考虑优化。

---

### 需改进 (Warning) ⚠️

- 一致性 < 0.85
- 语义鲁棒性 < 0.7
- 可区分性 < 0.3
- 目标对齐度 < 0.4
- Prior 增益 < 0.05

**说明**: Prior 模型需要改进，查看改进建议。

---

### 退化 (Degraded) ❌

**退化条件**：

1. 可区分性 < 0.3 **且** 输出多样性 < 0.0001 → 可能退化（输出常数）
2. 可区分性 < 0.3 → 可区分性低（任务输出过于相似）
3. 输出多样性 < 0.0001 → 方差极低（输出相似）

**说明**: Prior 模型可能退化，需要重新训练。

---

## 🔧 常见问题

### Q1: 成功画面嵌入如何获取？

**A**: 成功画面嵌入需要从评估结果中提取。您可以：

1. 运行端到端评估（例如使用 `eval_framework.py`）
2. 从成功的试验中提取最后 N 帧画面
3. 使用 MineCLIP 编码这些画面为 512 维嵌入
4. 保存为 `.pkl` 文件

**示例工具**:

```bash
# 使用现有工具提取
python src/utils/extract_success_visuals_by_task.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir data/visual_prompt_embeds
```

### Q2: 如果某些任务没有成功画面嵌入怎么办？

**A**: 在 YAML 配置中设置 `success_visual_embed: null`：

```yaml
- task_id: combat_pig
  en_instruction: kill pig
  success_visual_embed: null  # 没有成功画面
  instruction_variants:
    - "kill pig"
    - "attack pig"
```

框架会自动跳过这些任务的输出质量评估。

### Q3: discriminability 为什么这么低（0.12）？

**A**: 这是正常的！关键是看**区分度保持率**：

```
文本区分度: 0.013 (1.3%)
Prior 区分度: 0.12 (12%)
保持率: 9.2x
```

Prior 实际上**放大了** 9 倍的区分度。低 discriminability 的原因：

1. MineCLIP 文本嵌入本身区分度就很低
2. Minecraft 任务在语义空间上确实比较接近
3. CFG 可以进一步放大微小差异

### Q4: Prior 增益为负怎么办？

**A**: 如果 Prior 增益 < 0，说明 Prior 在拖后腿：

**可能的原因**：

1. Prior 训练不足或过拟合
2. Prior 训练数据质量差
3. Prior 架构问题

**解决方案**：

1. 检查 Prior 权重文件
2. 重新训练 Prior
3. 增加训练数据
4. 调整训练超参数
5. 考虑跳过 Prior，直接使用文本嵌入

### Q5: 如何评估 CFG 的实际影响？

**A**: 完整的 CFG 敏感度测试需要在端到端评估中进行：

```bash
# 未来实现（端到端评估时）
python src/evaluation/eval_framework.py \
    --config config/eval_tasks_comprehensive.yaml \
    --test-cfg-sensitivity \
    --cfg-scales "0,3,6,9,12"
```

当前的 Prior 评估只记录 CFG 配置，不进行实际的任务测试。

---

## 🎯 下一步

### 短期

1. **验证成功画面嵌入**: 确保所有任务都有对应的成功画面嵌入文件
2. **运行测试评估**: 使用 `eval_tasks_prior_test.yaml` 进行快速测试
3. **查看 HTML 报告**: 分析评估结果，查看改进建议

### 中期

1. **扩展任务配置**: 为所有 39 个任务添加 Prior 评估字段
2. **运行完整评估**: 评估所有任务，建立基线指标
3. **优化 Prior**: 根据评估结果优化 Prior 训练

### 长期

1. **实现端到端 CFG 测试**: 在 Policy 评估中测试 CFG 影响
2. **实现 Prior vs Text 对比实验**: 直接对比任务成功率
3. **建立完整的四层评估体系**: Prior + Policy + 交互 + 端到端

---

## 📚 参考资料

- **设计文档**: `docs/design/STEVE1_COMPLETE_EVALUATION_FRAMEWORK.md`
- **Prior 评估理论**: `docs/technical/PRIOR_EVALUATION_FRAMEWORK_DESIGN.md`
- **配置示例**: `config/eval_tasks_prior_example.yaml`
- **测试配置**: `config/eval_tasks_prior_test.yaml`

---

## 🆘 获取帮助

如果遇到问题：

1. 查看日志输出（框架会详细记录每个步骤）
2. 检查配置文件格式（使用 YAML 验证器）
3. 验证文件路径（成功画面嵌入、Prior 权重）
4. 查看错误信息和堆栈跟踪

**常见错误**：

```
FileNotFoundError: 成功画面嵌入不存在
→ 检查 success_visual_embed 路径是否正确

ModuleNotFoundError: No module named 'steve1'
→ 确保在 minedojo 环境中运行

KeyError: 'prior_evaluation'
→ 配置文件缺少 prior_evaluation 部分
```

---

**祝评估顺利！** 🚀

