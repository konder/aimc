# Steve1 深度评估快速参考

> **快速查找**: 常用命令和参数

---

## 🚀 快速命令

### 安装依赖

```bash
pip install umap-learn seaborn scikit-learn
```

### 快速测试（3任务 × 3试验，15分钟）

```bash
# macOS M 系列芯片（推荐）
bash scripts/run_deep_evaluation.sh \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --max-tasks 3 \
    --n-trials 3

# 或者直接使用（Linux/其他平台）
bash scripts/run_minedojo_x86.sh python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --max-tasks 3 \
    --n-trials 3
```

### 评估完整任务集

```bash
# Harvest 任务（13个，1-2小时）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set harvest_tasks \
    --n-trials 5

# Combat 任务（10个，1-2小时）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set combat_tasks \
    --n-trials 5

# TechTree 任务（14个，2-3小时）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set techtree_tasks \
    --n-trials 5

# 所有任务（39个，4-6小时）
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set all_tasks \
    --max-tasks 10  # 限制为10个
```

### 只评估 Prior（不运行环境，快速）

```bash
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set all_tasks \
    --skip-e2e
```

### 对比不同 Prior 模型

```bash
# 原版 Prior
python src/evaluation/run_steve1_deep_evaluation.py \
    --prior-weights data/weights/steve1/steve1_prior.pt \
    --output-dir results/prior_original \
    --skip-e2e

# 新 Prior
python src/evaluation/run_steve1_deep_evaluation.py \
    --prior-weights data/weights/steve1/my_prior.pt \
    --output-dir results/prior_new \
    --skip-e2e

# 对比
diff <(jq results/prior_original/prior_analysis.json) \
     <(jq results/prior_new/prior_analysis.json)
```

### 运行示例

```bash
# 所有示例
python examples/steve1_deep_analysis_demo.py

# 单独运行
python examples/steve1_deep_analysis_demo.py prior   # Prior 分析
python examples/steve1_deep_analysis_demo.py policy  # 策略分析
python examples/steve1_deep_analysis_demo.py e2e     # 端到端
```

---

## 📋 常用参数

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | **必需** | 评估配置文件路径 |
| `--task-set` | `harvest_tasks` | 任务集名称 |
| `--output-dir` | `results/deep_evaluation` | 输出目录 |
| `--n-trials` | `5` | 每个任务的试验次数 |
| `--max-steps` | `1000` | 每个试验的最大步数 |
| `--max-tasks` | `None` | 最大评估任务数（用于快速测试） |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `data/weights/vpt/2x.model` | VPT 模型路径 |
| `--weights-path` | `data/weights/steve1/steve1.weights` | Steve1 权重路径 |
| `--prior-weights` | `data/weights/steve1/steve1_prior.pt` | Prior VAE 权重路径 |

### 分析选项

| 参数 | 说明 |
|------|------|
| `--skip-prior` | 跳过 Prior 分析（只分析策略） |
| `--skip-policy` | 跳过策略分析（只分析 Prior） |
| `--skip-e2e` | 跳过端到端分析（只分析 Prior） |

### 其他选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--verbose` | `False` | 详细日志 |
| `--seed` | `42` | 随机种子 |

---

## 📊 输出结构

```
results/deep_evaluation/
├── prior_analysis/
│   ├── prior_analysis.json           # Prior 分析结果
│   ├── text_embeds.npy               # 文本嵌入 [N, 512]
│   ├── prior_embeds.npy              # Prior 嵌入 [N, 512]
│   ├── instructions.json             # 指令列表
│   ├── embedding_space_tsne.png      # t-SNE 可视化
│   ├── embedding_space_pca.png       # PCA 可视化
│   ├── similarity_matrix.png         # 相似度矩阵
│   └── quality_metrics.png           # 质量指标
├── end_to_end/
│   ├── {task_id}/
│   │   ├── {task_id}_end_to_end.json
│   │   └── end_to_end_analysis.png
│   └── task_comparison.png           # 多任务对比
├── summary_report.json               # 总结报告
└── deep_evaluation_*.log             # 运行日志
```

---

## 🔍 关键指标

### Prior 模型

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| `text_to_prior_similarity` | >0.8 | 0.6-0.8 | <0.6 |
| `prior_variance` | <0.05 | 0.05-0.1 | >0.1 |
| `reconstruction_quality` | >0.7 | 0.5-0.7 | <0.5 |

### 策略模型

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| `action_diversity` | >2.0 | 1.0-2.0 | <1.0 |
| `temporal_consistency` | >0.6 | 0.4-0.6 | <0.4 |
| `repeated_action_ratio` | <0.3 | 0.3-0.5 | >0.5 |
| `success_rate` | >0.5 | 0.3-0.5 | <0.3 |

---

## 💡 常见问题

### Q: 评估需要多久？

| 配置 | 任务数 | 试验数 | 时间 |
|------|--------|--------|------|
| 快速测试 | 3 | 3 | 15-20分钟 |
| 单个分类 | 10-14 | 5 | 1-2小时 |
| 所有任务 | 39 | 5 | 4-6小时 |

### Q: 只想看 Prior 的嵌入空间？

```bash
python src/evaluation/run_steve1_deep_evaluation.py \
    --config config/eval_tasks_comprehensive.yaml \
    --task-set all_tasks \
    --skip-e2e  # 不运行环境，几分钟完成
```

### Q: Prior 相似度多少算好？

- **>0.8**: 优秀
- **0.6-0.8**: 良好
- **<0.6**: 需改进

### Q: 如何查看总结报告？

```bash
cat results/deep_evaluation/summary_report.json | jq
```

关键字段：
- `.prior_analysis.avg_text_to_prior_similarity`: Prior 平均相似度
- `.end_to_end_analysis.avg_success_rate`: 平均成功率
- `.end_to_end_analysis.bottleneck_distribution`: 瓶颈分布
- `.recommendations`: 改进建议

---

## 🎨 可视化解读

### Prior 嵌入空间 (t-SNE)

```
[好的例子]
● ● ●   语义相近的点聚在一起
  ● ●   箭头短而一致
● ● ●

[差的例子]
●     ● 点分散
  ●     箭头长短不一
    ●
```

### 动作分布

```
[好的例子 - 砍树任务]
forward: ████████ 40%
attack:  ██████████ 50%
jump:    ██ 10%

[差的例子 - 砍树任务]
forward: ████ 20%
back:    ████ 20%
...      ... (过于均匀，随机游走)
```

---

## 🛠️ 改进建议

### Prior 相似度低 (< 0.6)

**建议**:
1. 重新训练 Prior，增加数据
2. 调整 VAE 架构（增加容量）
3. 使用更强的文本编码器

### Prior 瓶颈占主导 (>50%)

**建议**:
1. **优先级1**: 改进 Prior 模型
2. 增加 Prior 训练数据
3. 考虑换编码器（如 Chinese-CLIP）

### 策略瓶颈占主导 (>50%)

**建议**:
1. **优先级1**: 改进策略模型
2. DAgger 迭代
3. 调整 CFG scale
4. Fine-tune VPT

---

## 📚 相关文档

- **详细指南**: `docs/guides/STEVE1_DEEP_EVALUATION_GUIDE.md`
- **工具总结**: `docs/summaries/STEVE1_DEEP_EVALUATION_SUMMARY.md`
- **Prior 原理**: `docs/technical/STEVE1_PRIOR_EXPLAINED.md`

---

## 🔗 快速链接

```bash
# 查看帮助
python src/evaluation/run_steve1_deep_evaluation.py --help

# 查看示例
python examples/steve1_deep_analysis_demo.py

# 查看文档
cat docs/guides/STEVE1_DEEP_EVALUATION_GUIDE.md
```

---

**最后更新**: 2025-11-27

