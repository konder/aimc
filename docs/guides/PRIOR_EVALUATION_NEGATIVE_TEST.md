# Prior评估负向测试指南

> **目的**: 通过随机打散数据验证评估指标的有效性  
> **方法**: 对比正常评估和打散评估的结果差异  
> **作者**: AI Assistant  
> **日期**: 2025-11-27

---

## 🎯 测试原理

### **什么是负向测试？**

在软件测试中，**负向测试**是故意输入错误或异常的数据，验证系统能否正确识别并处理。

在Prior评估中：
- ✅ **正常输入**: 任务ID对应正确的成功画面嵌入
- ❌ **负向输入**: 任务ID被随机打散，对应错误的成功画面嵌入

### **测试假设**

如果评估指标正确，应该能区分正常和异常数据：

| 指标 | 正常数据 | 打散数据 | 预期差异 |
|------|---------|---------|---------|
| 目标准确性 | 高 (0.8+) | 低 (0.3-0.5) | > 0.3 ✅ |
| 语义鲁棒性 | 高 (0.9+) | 低 (0.3-0.6) | > 0.3 ✅ |
| 一致性 | 高 (1.0) | 高 (1.0) | ≈ 0 ✅ |
| 可区分性 | 中 (0.5+) | 中 (0.5+) | ≈ 0 ✅ |

---

## 🚀 快速使用

### **步骤1: 正常评估（基线）**

```bash
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/test_normal
```

记录结果：
- 目标准确性: _______
- 语义鲁棒性: _______
- 一致性: _______

### **步骤2: 打散评估（负向测试）**

```bash
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/test_shuffled \
    --shuffle
```

记录结果：
- 目标准确性: _______
- 语义鲁棒性: _______
- 一致性: _______

### **步骤3: 对比分析**

```bash
# 使用jq对比（如果已安装）
echo "正常评估:"
jq '.avg_goal_accuracy, .avg_semantic_robustness' \
    results/prior_evaluation/test_normal/prior_evaluation_summary.json

echo "打散评估:"
jq '.avg_goal_accuracy, .avg_semantic_robustness' \
    results/prior_evaluation/test_shuffled/prior_evaluation_summary.json
```

或手动对比两个JSON文件。

---

## 📊 打散的内容

### **1. 指令 -> 指令变体**

**正常情况**:
```json
{
  "harvest_1_log": {
    "variants": ["chop tree", "cut down tree", "harvest wood"]
  },
  "combat_pig": {
    "variants": ["hunt pig", "kill pig", "attack pig"]
  }
}
```

**打散后**:
```json
{
  "harvest_1_log": {
    "variants": ["chop tree", "kill pig", "dig dirt"]  ← 混合
  },
  "combat_pig": {
    "variants": ["hunt pig", "cut down tree", "mine stone"]  ← 混合
  }
}
```

**影响**: 语义鲁棒性应该下降

### **2. 任务ID -> 成功画面嵌入（未实现）**

**正常情况**:
```
harvest_1_log -> [砍树的16帧视频嵌入]
combat_pig -> [打猪的16帧视频嵌入]
```

**打散后**:
```
harvest_1_log -> [打猪的16帧视频嵌入]  ← 错配
combat_pig -> [砍树的16帧视频嵌入]  ← 错配
```

**影响**: 目标准确性应该下降

**注**: 当前版本仅打散指令变体，不打散视觉嵌入（需要额外实现）

---

## 📈 结果判断

### **场景A: 指标有效（理想情况）**

```
                   正常    打散    差异
目标准确性          0.86    0.45    0.41 ✅
语义鲁棒性          0.98    0.52    0.46 ✅
一致性             1.00    1.00    0.00 ✅
```

**结论**: ✅ 指标计算正确，能够区分正常和异常数据

### **场景B: 指标部分有效**

```
                   正常    打散    差异
目标准确性          0.86    0.72    0.14 ⚠️
语义鲁棒性          0.98    0.78    0.20 ⚠️
一致性             1.00    1.00    0.00 ✅
```

**结论**: ⚠️ 指标有一定区分能力，但差异不够大
- 可能MineCLIP对跨任务指令也有一定相似度
- 或者任务本身有重叠（如harvest类任务）

### **场景C: 指标无效（有问题）**

```
                   正常    打散    差异
目标准确性          0.86    0.84    0.02 🔴
语义鲁棒性          0.98    0.95    0.03 🔴
一致性             1.00    1.00    0.00 ✅
```

**结论**: 🔴 指标可能有问题，无法区分正常和异常数据
- 检查评估实现逻辑
- 检查Prior是否退化

---

## 🔧 实现细节

### **`--shuffle` 参数的工作流程**

1. **备份原始文件**
   ```bash
   cp data/instruction_variants.json \
      data/instruction_variants_backup_TIMESTAMP.json
   ```

2. **生成打散版本**
   ```bash
   python scripts/shuffle_eval_data.py \
       --instruction-variants data/instruction_variants.json \
       --output-variants data/instruction_variants_shuffled.json
   ```

3. **使用打散版本运行评估**
   ```bash
   python src/evaluation/prior_eval_framework.py \
       --instruction-variants data/instruction_variants_shuffled.json \
       ...
   ```

4. **清理临时文件**
   ```bash
   rm data/instruction_variants_shuffled.json
   ```

5. **保留备份**（用于恢复或分析）

### **打散算法**

```python
# 伪代码
all_variants = collect_all_variants()
random.shuffle(all_variants)  # 固定种子42

for each task:
    task.variants = all_variants[start:end]
    start = end
```

**关键**: 使用固定随机种子（42），确保结果可复现

---

## 🎓 为什么一致性不变？

**一致性**衡量的是：
```
同一指令多次采样Prior的输出稳定性
```

它与**数据内容**无关，只与Prior的推理过程有关：
- Prior在推理时使用均值（确定性）
- 所以输出完全一致 → 一致性 = 1.0

无论输入是正常指令还是打散指令，Prior都会给出确定的输出。

---

## 📝 最佳实践

### **1. 每次修改评估代码后运行负向测试**

```bash
# 修改代码后
bash scripts/run_prior_evaluation.sh --eval-result-dir ... --shuffle

# 检查结果是否仍然合理
```

### **2. 记录测试结果**

创建测试日志：
```markdown
## 测试 2025-11-27

### 正常评估
- 目标准确性: 0.864
- 语义鲁棒性: 0.983

### 打散评估
- 目标准确性: 0.428 (-0.436) ✅
- 语义鲁棒性: 0.513 (-0.470) ✅

### 结论
指标有效，能够正确区分正常和异常数据
```

### **3. 使用相同的随机种子**

确保每次打散的方式相同，便于对比不同版本的评估结果。

### **4. 保存两个HTML报告**

```
results/prior_evaluation/
├── test_normal/
│   └── prior_evaluation_report.html
└── test_shuffled/
    └── prior_evaluation_report.html
```

并排查看，对比可视化结果。

---

## 🐛 故障排查

### **问题1: 打散后结果没有变化**

**检查**:
```bash
# 查看是否真的打散了
diff data/instruction_variants.json \
     data/instruction_variants_shuffled_*.json
```

**可能原因**:
- 打散脚本没有正确执行
- 指令变体太少（<3个）
- 随机种子导致顺序恰好不变

### **问题2: 所有指标都下降（包括一致性）**

**可能原因**:
- 一致性计算逻辑有误
- Prior在采样（而不是使用均值）

**检查**:
```python
# 查看get_prior_embed是否采样
def get_prior_embed(text, mineclip, prior, device):
    ...
    z_goal = prior(text_embed)  # 应该是确定性的
    return z_goal
```

### **问题3: 备份文件太多**

每次运行都会创建备份。定期清理：
```bash
# 删除30天前的备份
find data/ -name "instruction_variants_backup_*.json" -mtime +30 -delete
```

---

## 🎯 总结

### **核心价值**

负向测试是验证评估指标有效性的强大工具：
- ✅ 不需要人工标注
- ✅ 完全自动化
- ✅ 结果清晰明确
- ✅ 可重复执行

### **使用场景**

1. **开发新指标时**: 验证指标能否区分好坏数据
2. **修改评估代码后**: 确保没有引入bug
3. **怀疑结果时**: 通过负向测试验证合理性
4. **演示时**: 向他人证明指标的有效性

### **关键洞察**

**好的评估指标应该对错误的输入敏感！**

如果一个指标在正常和异常数据上都给出相同的高分，那它就没有区分能力，不是一个有效的指标。

---

## 🔗 相关文件

- **脚本**:
  - `scripts/run_prior_evaluation.sh` - 主评估脚本（支持--shuffle）
  - `scripts/shuffle_eval_data.py` - 数据打散工具

- **文档**:
  - 本文档 - 负向测试指南
  - `docs/technical/SEMANTIC_ROBUSTNESS_VALIDATION_TEST.md` - 语义鲁棒性专项测试

---

## 💡 下一步

运行负向测试，验证当前的评估指标！

```bash
# 正常评估
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/normal

# 打散评估
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/shuffled \
    --shuffle

# 对比结果
echo "查看两个HTML报告并对比关键指标！"
```

