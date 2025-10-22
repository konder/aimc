# DAgger 继续训练指南

> **功能**: 从已有的DAgger模型继续训练更多轮次

---

## 🎯 **核心概念**

### **DAgger 多轮训练机制**

DAgger的每一轮都会生成**新的模型**，它们之间是**迭代改进**的关系：

```
BC基线 (bc_baseline.zip) - 成功率: 60%
  ↓ 
  ↓ [收集失败状态] → [标注] → [重新训练]
  ↓
迭代1 (dagger_iter_1.zip) - 成功率: 75% ← 新模型
  ↓
  ↓ [收集失败状态] → [标注] → [重新训练]
  ↓
迭代2 (dagger_iter_2.zip) - 成功率: 85% ← 新模型
  ↓
  ↓ [收集失败状态] → [标注] → [重新训练]
  ↓
迭代3 (dagger_iter_3.zip) - 成功率: 92% ← 最终模型 ✅
```

### **关键原理**

#### **1. 每轮使用新模型收集数据**

```python
# 迭代1: 用BC基线收集
run_policy_collect_states(model="bc_baseline.zip")

# 迭代2: 用迭代1的模型收集
run_policy_collect_states(model="dagger_iter_1.zip")

# 迭代3: 用迭代2的模型收集
run_policy_collect_states(model="dagger_iter_2.zip")
```

**为什么？**
- BC基线会犯错误A、B、C
- 迭代1修正了A、B，但可能在新场景D犯错
- 迭代2修正了D，探索到新场景E...

#### **2. 数据是累积的**

每轮训练使用**所有之前的数据**：

```
BC训练:
  数据 = 专家演示（10个episodes）

迭代1训练:
  数据 = 专家演示 + iter_1标注

迭代2训练:
  数据 = 专家演示 + iter_1标注 + iter_2标注

迭代3训练:
  数据 = 专家演示 + iter_1标注 + iter_2标注 + iter_3标注
```

**不需要合并模型文件**，数据已经自动累积了！

#### **3. 只需要最终模型**

```bash
# ✅ 正确：使用最新的模型
python tools/evaluate_policy.py --model checkpoints/dagger_iter_3.zip

# ❌ 错误：不需要合并模型
# 模型不需要合并，只用最新的即可
```

---

## 📋 **使用场景**

### **场景1: 训练3轮后继续训练**

```bash
# 第一次：完成3轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --iterations 3

# 结果: dagger_iter_3.zip (成功率 85%)

# 继续训练2轮（迭代4-5）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --iterations 5

# 结果: dagger_iter_5.zip (成功率 92%)
```

---

### **场景2: BC效果太差，跳过BC直接继续DAgger**

```bash
# 第一次：只训练了BC
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 0  # 只做BC，不做DAgger

# BC评估: 成功率只有 40%，太低了！

# 追加录制5个episodes
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 15 \
    --append-recording \
    --skip-bc

# 重新训练BC
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/harvest_1_log/bc_baseline.zip \
    --epochs 50

# BC评估: 成功率 65%，好多了！

# 从BC开始DAgger（迭代1-3）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/bc_baseline.zip \
    --start-iteration 1 \
    --iterations 3
```

---

### **场景3: 分多天训练**

```bash
# 第1天：录制 + BC + 迭代1
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --iterations 1

# 第2天：继续迭代2
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_1.zip \
    --iterations 2

# 第3天：继续迭代3-5
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_2.zip \
    --iterations 5
```

---

## 🛠️ **命令行参数**

### **继续训练专用参数**

| 参数 | 说明 | 必需 |
|------|------|------|
| `--continue-from MODEL` | 从指定模型继续训练 | ✅ 是 |
| `--start-iteration N` | 从第N轮开始（可选，自动推断）| ❌ 否 |
| `--iterations N` | 总迭代轮数（包含已完成的）| ✅ 是 |

### **自动推断起始迭代**

如果不指定 `--start-iteration`，脚本会从模型文件名自动推断：

```bash
# 自动推断（推荐）✅
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
# 自动检测: 上一轮为 iter_3，从 iter_4 开始

# 手动指定 ✅
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --start-iteration 4 \
    --iterations 5
```

---

## 📊 **完整示例**

### **从零到精通：完整训练流程**

```bash
# ============================================================================
# 第1阶段: BC基线训练
# ============================================================================

# 录制10个专家演示
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 0  # 只做BC，不做DAgger

# 输出: 
# - data/expert_demos/harvest_1_log/episode_000 ~ 009
# - checkpoints/harvest_1_log/bc_baseline.zip
# - BC成功率: 60%

# ============================================================================
# 第2阶段: DAgger迭代1-3
# ============================================================================

# 从BC基线开始，执行3轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/bc_baseline.zip \
    --start-iteration 1 \
    --iterations 3

# 输出:
# - checkpoints/harvest_1_log/dagger_iter_1.zip (成功率: 75%)
# - checkpoints/harvest_1_log/dagger_iter_2.zip (成功率: 85%)
# - checkpoints/harvest_1_log/dagger_iter_3.zip (成功率: 90%)

# ============================================================================
# 第3阶段: 继续优化（迭代4-5）
# ============================================================================

# 从迭代3继续训练2轮
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --iterations 5

# 输出:
# - checkpoints/harvest_1_log/dagger_iter_4.zip (成功率: 92%)
# - checkpoints/harvest_1_log/dagger_iter_5.zip (成功率: 95%)

# ============================================================================
# 第4阶段: 最终评估
# ============================================================================

# 使用最终模型评估
python tools/evaluate_policy.py \
    --model checkpoints/harvest_1_log/dagger_iter_5.zip \
    --episodes 50

# 预期: 成功率 95%+ ✅
```

---

## 🔍 **内部工作流程**

### **继续训练模式执行的步骤**

使用 `--continue-from` 时，脚本会：

1. **跳过录制和BC训练**
   ```
   ℹ️ 继续训练模式: 跳过专家演示录制
   ℹ️ 继续训练模式: 跳过BC基线训练
   ℹ️ 继续训练模式: 跳过BC基线评估
   ```

2. **从指定模型开始DAgger循环**
   ```
   ✓ 继续训练模式: 从 checkpoints/harvest_1_log/dagger_iter_3.zip 开始
   ℹ️ 自动检测: 上一轮为 iter_3，从 iter_4 开始
   ✓ 将执行 DAgger 迭代 4 到 5
   ```

3. **执行每一轮的标准流程**
   ```
   迭代4:
     1. 收集失败状态（使用 dagger_iter_3.zip）
     2. 交互式标注
     3. 聚合数据训练（专家演示 + iter_1~4标注）
     4. 评估（dagger_iter_4.zip）
   
   迭代5:
     1. 收集失败状态（使用 dagger_iter_4.zip）
     2. 交互式标注
     3. 聚合数据训练（专家演示 + iter_1~5标注）
     4. 评估（dagger_iter_5.zip）
   ```

---

## ⚠️ **常见问题**

### **Q1: 训练完3轮，想继续2轮，总共5轮，怎么写命令？**

```bash
# ✅ 正确
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --iterations 5  # 总共5轮，会执行迭代4和5

# ❌ 错误
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --iterations 2  # 错误！这会导致总迭代数 < 起始迭代
```

**关键**: `--iterations` 是**总轮数**，不是**新增轮数**

---

### **Q2: 需要保留所有中间模型吗？**

**不需要合并，但建议保留**：

```bash
checkpoints/harvest_1_log/
├── bc_baseline.zip          # 保留（可能回退）
├── dagger_iter_1.zip        # 保留（便于对比）
├── dagger_iter_2.zip        # 保留（便于对比）
├── dagger_iter_3.zip        # 保留（便于对比）
├── dagger_iter_4.zip        # 保留（便于对比）
└── dagger_iter_5.zip        # ✅ 最终模型（实际使用）
```

**原因**:
- 万一迭代5表现变差，可以回退到迭代4
- 便于分析成功率提升曲线
- 便于对比不同轮次的行为

**如果磁盘空间紧张**:
```bash
# 只保留最新的和BC基线
rm checkpoints/harvest_1_log/dagger_iter_{1..4}.zip
```

---

### **Q3: 数据文件需要手动合并吗？**

**不需要！`train_dagger.py` 会自动累积数据**

```bash
# 迭代1训练时
python src/training/train_dagger.py \
    --base-data data/expert_demos/harvest_1_log/ \
    --new-data data/expert_labels/harvest_1_log/iter_1.pkl
# 输出聚合数据: data/dagger/harvest_1_log/combined_iter_1.pkl

# 迭代2训练时
python src/training/train_dagger.py \
    --base-data data/dagger/harvest_1_log/combined_iter_1.pkl \
    --new-data data/expert_labels/harvest_1_log/iter_2.pkl
# 输出聚合数据: data/dagger/harvest_1_log/combined_iter_2.pkl

# combined_iter_2.pkl 已经包含了:
# - 专家演示
# - iter_1 标注
# - iter_2 标注
```

---

### **Q4: 如何查看训练历史？**

脚本会自动记录并显示：

```
训练历史
============================================================
BC基线:     60.0%
迭代1:      75.0% (+15.0%)
迭代2:      85.0% (+10.0%)
迭代3:      90.0% (+5.0%)
迭代4:      92.0% (+2.0%)
迭代5:      95.0% (+3.0%)
============================================================
```

---

## 🎯 **最佳实践**

### **1. 渐进式训练**

```bash
# 不推荐: 一次性训练10轮 ❌
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 10
# 问题: 标注10轮非常累，而且可能浪费（早期就收敛了）

# 推荐: 分批训练 ✅
# 第1批: 3轮
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3
# 评估成功率: 90%，还有提升空间

# 第2批: 继续2轮
bash scripts/run_dagger_workflow.sh \
    --continue-from checkpoints/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
# 评估成功率: 95%，已经很好了，停止训练
```

### **2. 评估驱动**

每次继续训练前，先评估当前模型：

```bash
# 评估迭代3
python tools/evaluate_policy.py \
    --model checkpoints/harvest_1_log/dagger_iter_3.zip \
    --episodes 50

# 如果成功率 >= 95%: 停止训练，已经足够好了
# 如果成功率 < 95%: 继续训练
```

### **3. 保存训练日志**

```bash
# 记录每轮的成功率
echo "$(date): 迭代3 - 成功率 90%" >> training_log.txt
echo "$(date): 迭代4 - 成功率 92%" >> training_log.txt
echo "$(date): 迭代5 - 成功率 95%" >> training_log.txt
```

---

## 📚 **相关文档**

- [`DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger快速开始
- [`DAGGER_DETAILED_GUIDE.md`](DAGGER_DETAILED_GUIDE.md) - 详细算法说明
- [`DAGGER_WORKFLOW_MULTI_TASK.md`](DAGGER_WORKFLOW_MULTI_TASK.md) - 多任务工作流
- [`DAGGER_WORKFLOW_SCRIPT_GUIDE.md`](DAGGER_WORKFLOW_SCRIPT_GUIDE.md) - 脚本使用指南

---

**祝训练顺利！** 🚀

记住：DAgger是**迭代改进**，不是一次性训练。根据评估结果灵活调整训练轮数。

