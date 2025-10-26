# DAgger 快速指南

> **简明教程**: 使用 DAgger 算法训练 Minecraft AI 的核心步骤

---

## 📑 **目录**

1. [快速开始](#-快速开始)
2. [核心概念](#-核心概念)
3. [完整流程](#-完整流程)
4. [标注技巧](#-标注技巧)
5. [常见问题](#-常见问题)
6. [命令速查](#-命令速查)

---

## 🚀 **快速开始**

```bash
# 1. 激活环境
conda activate minedojo

# 2. 一键运行（3-5小时）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

**流程**: 录制演示 → BC训练 → DAgger迭代优化  
**预期**: BC 60% → 迭代3 后 85-90%

---

## 🎯 **核心概念**

### **什么是 DAgger？**

**问题**: BC只在专家轨迹训练，一旦偏离就越来越差  
**解决**: 在策略的失败状态上收集专家标注，覆盖真实场景

### **DAgger 流程**

```
1. BC基线训练 (60%)
   ↓
2. 运行策略，收集失败状态
   ↓
3. 标注失败状态的正确动作
   ↓
4. 聚合数据重新训练 (75%)
   ↓
5. 重复2-4步骤 (85% → 92%)
```

### **关键原则**

- **数据累积**: 每轮使用所有历史数据训练
- **只用最新模型**: 不需要合并模型文件
- **渐进改进**: 每轮针对当前策略的失败场景优化

---

## 🔧 **完整流程**

### **1. 录制演示（40-60分钟）**

```bash
# 推荐：使用鼠标控制（更自然）
bash scripts/run_minedojo_x86.sh python src/training/dagger/record_manual_chopping_pygame.py \
    --base-dir data/expert_demos/harvest_1_log \
    --mouse-sensitivity 0.5
```

**控制**: 鼠标转视角 | 左键攻击 | WASD移动 | Space跳跃

### **2. 训练 BC 基线（30-40分钟）**

```bash
python src/training/bc/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50
```

### **3. 评估 BC（10分钟）**

```bash
bash scripts/run_minedojo_x86.sh python src/training/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20
```

**预期**: 成功率 50-65%

### **4. DAgger 迭代（每轮 60-80分钟）**

每轮包含4步：

#### **步骤1: 收集失败状态**

```bash
python src/training/dagger/run_policy_collect_states.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/harvest_1_log/iter_1/ \
    --save-failures-only
```

#### **步骤2: 标注失败场景**

```bash
python src/training/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1/ \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling
```

#### **步骤3: 聚合训练**

```bash
python src/training/dagger/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/harvest_1_log/ \
    --new-data data/expert_labels/harvest_1_log/iter_1.pkl \
    --output checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --epochs 30
```

#### **步骤4: 评估改进**

```bash
bash scripts/run_minedojo_x86.sh python src/training/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --episodes 20
```

**预期**: 成功率 60% → 75%

重复步骤1-4进行多轮迭代，每轮使用最新模型。

---

## 🎨 **标注技巧**

### **标注控制**

**基础操作**:
- W/A/S/D - 前后左右移动
- I/K/J/L - 视角上/下/左/右
- F - 攻击
- Q - 前进+攻击
- **P - 保持策略动作（重要！）** ⭐

**特殊控制**:
- P - 策略动作正确时，保持不变
- N - 跳过不确定的状态
- Z - 撤销上一个标注
- X/ESC - 完成标注

- R - 前进+跳跃
- G - 前进+跳跃+攻击

### **标注原则** ⭐

1. **视角调整<20%，前进>60%** - 避免原地转圈
2. **善用P键** - 策略合理时保持不变，提速50%+
3. **关注HIGH优先级状态** - 失败前的关键决策
4. **主动探索优先** - 不确定时选择前进，不是转圈

### **健康的标注分布**

```
✅ 好的分布:
W (前进):        40%
Q (前进+攻击):   15%
P (保持策略):    20%
I/J/K/L (视角):  10%
其他:             15%

❌ 问题分布:
视角调整 > 30%  → 会导致原地转圈
前进 < 40%      → 探索不足
P键 < 10%       → 过度干预
```

---

## ⚠️ **常见问题**

### **录制问题**

| 问题 | 解决 |
|------|------|
| 键盘没反应 | 点击OpenCV窗口获得焦点 |
| 鼠标不灵敏 | 调整 `--mouse-sensitivity` 0.3-0.8 |
| 视角不动 | 使用IJKL，不是方向键 |

### **训练问题**

| 问题 | 解决 |
|------|------|
| BC成功率<50% | 增加专家演示到20+或提高epochs到100 |
| 未找到数据 | 检查 `data/expert_demos/harvest_1_log/` 结构 |
| Loss不下降 | 降低学习率到0.0001或增加数据 |
| 模型IDLE过多 | 补录数据到50+ episodes |

### **DAgger问题**

| 问题 | 解决 |
|------|------|
| 没有提升 | 检查标注质量，视角调整<20% |
| 模型转圈 | 重新标注，前进优先 |
| 标注太慢 | 使用`--smart-sampling`和P键 |
| 标注中断 | 重新运行相同命令会继续 |

---

## 📋 **命令速查**

### **快速参考**

```bash
# ✅ 完整流程（首次训练）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 10 --iterations 3

# ✅ 跳过录制（已有数据）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --iterations 3

# ✅ 追加录制数据
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 20 --append-recording --iterations 0

# ✅ 只录制，不训练
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 10 --iterations 0

# ✅ 只训练BC，不做DAgger
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --iterations 0

# ✅ 继续更多轮DAgger
bash scripts/run_dagger_workflow.sh --task harvest_1_log --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip --iterations 5

# ✅ 评估模型
bash scripts/run_minedojo_x86.sh python src/training/dagger/evaluate_policy.py --model checkpoints/dagger/harvest_1_log/bc_baseline.zip --episodes 20

# ✅ 重新训练BC（更多epochs）
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --bc-epochs 100 --iterations 0

# ✅ 使用鼠标录制
bash scripts/run_minedojo_x86.sh python src/training/dagger/record_manual_chopping_pygame.py --base-dir data/expert_demos/harvest_1_log --mouse-sensitivity 0.5
```

### **参数速查表**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务ID | `harvest_1_log` |
| `--num-episodes` | 录制数量 | `10` |
| `--iterations` | DAgger轮数 | `3` |
| `--bc-epochs` | BC训练轮数 | `50` |
| `--skip-recording` | 跳过录制 | `false` |
| `--skip-bc` | 跳过BC训练 | `false` |
| `--append-recording` | 追加录制 | `false` |
| `--continue-from` | 继续训练的模型 | - |
| `--start-iteration` | 起始迭代 | 自动推断 |
| `--mouse-sensitivity` | 鼠标灵敏度 | `0.15` |
| `--collect-episodes` | 每轮收集数 | `20` |
| `--eval-episodes` | 评估数量 | `20` |

### **故障速查**

| 问题 | 快速解决 |
|------|----------|
| 未找到数据 | 移除`--skip-recording`或手动录制 |
| BC模型不存在 | 移除`--skip-bc`或手动训练BC |
| IDLE > 70% | 补录到50+ episodes |
| 标注中断 | 重新运行相同命令会继续 |
| 模型原地转圈 | 检查标注分布，视角调整应<20% |
| 键盘没反应 | 点击OpenCV窗口获得焦点 |
| 鼠标不灵敏 | 调整`--mouse-sensitivity` |

### **继续训练**

```bash
# 从之前的模型继续训练
bash scripts/run_dagger_workflow.sh \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
```

---

## 📈 **性能预期**

| 阶段 | 成功率 | 时间 |
|------|--------|------|
| BC基线 | 50-65% | 40-70分钟 |
| 迭代1 | 70-78% | 60-80分钟 |
| 迭代2 | 80-85% | 60-80分钟 |
| 迭代3 | 85-92% | 60-80分钟 |

**总计**: 4-5小时达到90%+成功率

---

## 🎯 **核心要点**

1. **数据累积** - 每轮使用所有历史数据，不需要合并模型
2. **善用P键** - 策略合理时保持不变，标注提速50%+
3. **前进优先** - 视角调整<20%，避免原地转圈
4. **渐进训练** - 分批评估，不要一次训练太多轮
5. **鼠标录制** - 数据质量更高，操作更自然

---

**版本**: 4.0.0 (简化版)  
**最后更新**: 2025-10-24  

**祝训练顺利！** 🚀
