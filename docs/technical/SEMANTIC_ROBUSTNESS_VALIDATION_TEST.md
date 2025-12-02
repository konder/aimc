# 语义鲁棒性验证测试

> **目的**: 验证语义鲁棒性指标是否正确计算  
> **方法**: 负向测试 - 随机打散指令变体  
> **作者**: AI Assistant  
> **日期**: 2025-11-27

---

## 🎯 测试原理

### **核心思想**

如果指标正确，那么：
- ✅ **同任务的不同表述** → 高相似度（0.9+）
  - 例如："chop tree" vs "cut down a tree"
  
- ❌ **不同任务的指令混在一起** → 低相似度（0.3-0.6）
  - 例如："chop tree" vs "hunt pig" vs "dig dirt"

### **测试方法**

1. **原始版本**: 每个任务包含同义表述
   ```json
   "harvest_1_log": {
     "variants": [
       "chop tree",
       "cut down a tree",
       "harvest wood"
     ]
   }
   ```

2. **打散版本**: 随机混合不同任务的指令
   ```json
   "harvest_1_log": {
     "variants": [
       "chop tree",      ← 来自原harvest_1_log
       "hunt pig",       ← 来自combat_pig
       "dig dirt"        ← 来自harvest_1_dirt
     ]
   }
   ```

3. **预期结果**:
   ```
   原始版本: 语义鲁棒性 = 0.90+ ✅
   打散版本: 语义鲁棒性 = 0.30-0.60 ❌
   差异: > 0.3
   ```

---

## 🚀 快速开始

### **方式1: 一键自动化测试（推荐）**

```bash
bash scripts/auto_test_semantic_robustness.sh \
    results/evaluation/all_tasks_20251121_214545
```

**它会自动**:
1. ✓ 备份原始文件
2. ✓ 生成打散版本
3. ✓ 测试原始版本
4. ✓ 测试打散版本
5. ✓ 对比结果
6. ✓ 恢复原始文件

### **方式2: 手动步骤**

#### **步骤1: 生成打散版本**
```bash
python scripts/test_semantic_robustness.py
```

输出示例:
```
✓ 打散版本已保存到: data/instruction_variants_shuffled.json

任务: harvest_1_log
原始版本: chop tree, cut down tree, harvest wood
打散版本: chop tree, hunt pig, dig dirt
```

#### **步骤2: 测试原始版本**
```bash
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/test_original
```

#### **步骤3: 替换为打散版本**
```bash
# 备份原始文件
cp data/instruction_variants.json data/instruction_variants_backup.json

# 使用打散版本
cp data/instruction_variants_shuffled.json data/instruction_variants.json
```

#### **步骤4: 测试打散版本**
```bash
bash scripts/run_prior_evaluation.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output-dir results/prior_evaluation/test_shuffled
```

#### **步骤5: 对比结果**
```bash
python scripts/compare_semantic_robustness.py \
    --original results/prior_evaluation/test_original/prior_evaluation_summary.json \
    --shuffled results/prior_evaluation/test_shuffled/prior_evaluation_summary.json
```

#### **步骤6: 恢复原始文件**
```bash
cp data/instruction_variants_backup.json data/instruction_variants.json
```

---

## 📊 结果解读

### **对比输出示例**

```
================================================================================
语义鲁棒性对比结果
================================================================================

📊 整体统计
--------------------------------------------------------------------------------
原始版本平均语义鲁棒性: 0.9421
打散版本平均语义鲁棒性: 0.4837
差异: 0.4584 (48.6%)

✅ 验证结果
--------------------------------------------------------------------------------
🎉 指标计算正确！
   原始版本（0.942）>> 打散版本（0.484）
   说明：指标能够区分同任务变体和跨任务混合

📋 任务级对比
--------------------------------------------------------------------------------
任务ID                    原始        打散        差异      
--------------------------------------------------------------------------------
✓ harvest_1_log          0.9654      0.4521      0.5133    
✓ harvest_1_dirt         0.9321      0.4112      0.5209    
✓ combat_pig             0.9876      0.5234      0.4642    
...

🔍 其他指标（应该基本不变）
--------------------------------------------------------------------------------
目标准确性     原始: 0.8542  打散: 0.8539  ✓ 稳定
一致性         原始: 1.0000  打散: 1.0000  ✓ 稳定
可区分性       原始: 0.5432  打散: 0.5428  ✓ 稳定
```

### **判断标准**

| 差异范围 | 结论 | 说明 |
|---------|------|------|
| > 0.3 | ✅ **指标正确** | 能够明显区分同任务变体和跨任务混合 |
| 0.15-0.3 | ⚠️ **部分有效** | 有一定区分能力，可能MineCLIP本身很鲁棒 |
| < 0.15 | 🔴 **指标有问题** | 无法区分，可能实现有误 |

### **其他指标应该稳定**

如果以下指标显著变化（>0.05），说明实现可能有问题：
- ❌ 目标准确性变化很大
- ❌ 一致性变化很大  
- ❌ 可区分性变化很大

**原因**: 这些指标不依赖指令变体配置，应该保持稳定

---

## 🔬 实验解读

### **场景A: 差异很大（>0.4）- 理想情况**

```
原始: 0.95
打散: 0.45
差异: 0.50
```

**结论**: ✅ **指标非常可靠**
- Prior能够区分同义表述和不相关指令
- MineCLIP文本编码器工作正常
- 实现完全正确

### **场景B: 差异中等（0.2-0.4）- 正常情况**

```
原始: 0.92
打散: 0.65
差异: 0.27
```

**结论**: ✅ **指标基本可靠**
- 可能MineCLIP本身对语义很敏感
- 不同任务的指令可能有一些共同词汇（如"get", "obtain"）
- 实现正确，但MineCLIP的特性导致差异不够大

### **场景C: 差异很小（<0.15）- 异常情况**

```
原始: 0.94
打散: 0.88
差异: 0.06
```

**结论**: 🔴 **指标可能有问题**

**可能原因**:
1. ❌ 实现错误：比较了错误的对象
2. ❌ Prior退化：所有输入都输出相似内容
3. ⚠️ MineCLIP过于平滑：所有文本嵌入都很相似

**检查方法**:
- 查看可区分性是否也很低（<0.3）
- 检查Prior方差是否极低（<0.0001）
- 运行 `diagnose_semantic_robustness.py` 检查MineCLIP

---

## 🧪 打散示例

### **原始版本**

```json
{
  "harvest_1_log": {
    "variants": [
      "chop tree",
      "chop tree and get wood",
      "cut down a tree",
      "harvest wood from tree"
    ]
  },
  "combat_pig": {
    "variants": [
      "hunt pig",
      "kill a pig",
      "attack pig",
      "defeat a pig"
    ]
  }
}
```

**语义特点**: 每组内部高度相似

### **打散版本**

```json
{
  "harvest_1_log": {
    "variants": [
      "chop tree",          ← harvest_1_log
      "kill a pig",         ← combat_pig
      "dig dirt",           ← harvest_1_dirt
      "shear sheep"         ← harvest_1_wool
    ]
  },
  "combat_pig": {
    "variants": [
      "hunt pig",           ← combat_pig
      "cut down a tree",    ← harvest_1_log
      "mine cobblestone",   ← harvest_1_cobblestone
      "dig sand"            ← harvest_1_sand
    ]
  }
}
```

**语义特点**: 每组内部差异很大

---

## 🎓 理论背景

### **为什么这个测试有效？**

语义鲁棒性计算的是：
```
同一任务的不同表述的Prior输出相似度
```

如果实现正确：
1. **同任务同义词** → MineCLIP文本嵌入相似 → Prior输出相似 → 高相似度
2. **不同任务混合** → MineCLIP文本嵌入不同 → Prior输出不同 → 低相似度

如果实现错误或模型退化：
- 两种情况下相似度都高 → 差异小

### **这是负向测试（Negative Test）**

在软件测试中，负向测试故意输入错误数据来验证系统行为。

本测试中：
- **正常输入**: 同义词组
- **异常输入**: 随机混合
- **预期**: 两者结果应该明显不同

如果结果相同，说明系统无法区分正常和异常输入 → 有问题！

---

## 📝 实验记录模板

```markdown
## 实验日期: YYYY-MM-DD

### 测试配置
- 评估结果目录: results/evaluation/xxx
- 原始指令变体数量: 8任务 x 5-6变体
- 打散种子: 42

### 结果
- 原始版本平均语义鲁棒性: X.XXXX
- 打散版本平均语义鲁棒性: X.XXXX
- 差异: X.XXXX (XX.X%)

### 结论
[ ] ✅ 指标正确（差异 > 0.3）
[ ] ⚠️ 部分有效（差异 0.15-0.3）
[ ] 🔴 指标有问题（差异 < 0.15）

### 备注
- 
```

---

## 🔗 相关文件

- **测试工具**:
  - `scripts/test_semantic_robustness.py` - 生成打散版本
  - `scripts/compare_semantic_robustness.py` - 对比结果
  - `scripts/auto_test_semantic_robustness.sh` - 一键测试

- **数据文件**:
  - `data/instruction_variants.json` - 原始版本
  - `data/instruction_variants_shuffled.json` - 打散版本（生成）
  - `data/instruction_variants_backup.json` - 备份（生成）

---

## 💡 最佳实践

1. ✅ **定期运行此测试**，确保指标持续有效
2. ✅ **在修改评估代码后**，重新运行验证
3. ✅ **记录每次测试结果**，追踪指标稳定性
4. ✅ **使用相同的随机种子**，确保结果可复现
5. ✅ **保存测试输出**，便于调试和分析

---

## 🎯 总结

这是一个简单但强大的验证方法：

**原理**: 如果指标正确，应该能区分"同义词"和"不相关词"

**方法**: 人为制造"不相关词"情况（打散），看指标是否下降

**优点**:
- ✅ 不需要真实数据
- ✅ 不需要人工标注
- ✅ 完全自动化
- ✅ 结果清晰明确

**关键洞察**: 好的指标应该对**错误的输入**敏感！

