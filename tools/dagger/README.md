# DAgger Tools

DAgger（Dataset Aggregation）训练工具集。

## 📁 文件说明

### **1. record_manual_chopping.py**
手动录制专家演示工具（Pygame + 鼠标控制）

**特性**:
- 🖱️ 鼠标控制视角（连续平滑）
- 🖱️ 鼠标左键攻击
- ⌨️ WASD移动控制
- ✅ 自动跳过静止帧（默认）
- ✅ 无需macOS权限

**使用**:
```bash
python tools/dagger/record_manual_chopping.py \
    --base-dir data/expert_demos/harvest_1_log \
    --max-frames 1000 \
    --mouse-sensitivity 0.2
```

---

### **2. evaluate_policy.py**
评估已训练策略的性能

**功能**:
- 运行N个episodes评估策略
- 统计成功率和平均奖励
- 可视化游戏画面（可选）

**使用**:
```bash
python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --task-id harvest_1_log
```

---

### **3. run_policy_collect_states.py**
收集策略运行时的状态

**功能**:
- 运行策略收集失败状态
- 保存observation和metadata
- 用于后续标注

**使用**:
```bash
python tools/dagger/run_policy_collect_states.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/harvest_1_log/iter_1 \
    --task-id harvest_1_log
```

---

### **4. label_states.py**
交互式标注收集的状态

**特性**:
- 智能采样（只标注失败前N步）
- 组合键支持（Q=前进+攻击等）
- 撤销功能
- 进度保存

**使用**:
```bash
python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1 \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling \
    --failure-window 10
```

**组合键**:
- `Q`: 前进 + 攻击
- `R`: 前进 + 跳跃
- `G`: 前进 + 跳跃 + 攻击
- `Z`: 撤销上一步
- `N`: 跳过当前状态
- `X/ESC`: 完成标注

---

## 🔄 **DAgger工作流**

### **完整流程**

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

### **单步执行**

1. **录制专家演示**:
```bash
python tools/dagger/record_manual_chopping.py \
    --base-dir data/expert_demos/harvest_1_log \
    --max-frames 1000
```

2. **训练BC基线**:
```bash
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log \
    --output checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 50
```

3. **评估BC基线**:
```bash
python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20
```

4. **收集失败状态**:
```bash
python tools/dagger/run_policy_collect_states.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/harvest_1_log/iter_1
```

5. **标注状态**:
```bash
python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1 \
    --output data/expert_labels/harvest_1_log/iter_1.pkl
```

6. **DAgger训练**:
```bash
python src/training/train_dagger.py \
    --base-data data/expert_demos/harvest_1_log \
    --new-labels data/expert_labels/harvest_1_log/iter_1.pkl \
    --output checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
    --epochs 30
```

7. **重复步骤3-6**直到性能收敛

---

## 📚 **相关文档**

- [DAgger快速开始](../../docs/guides/DAGGER_QUICK_START.md)
- [Pygame鼠标控制指南](../../docs/guides/PYGAME_MOUSE_CONTROL.md)
- [BC训练指南](../../docs/guides/BC_TRAINING_QUICK_START.md)
- [DAgger工作流脚本指南](../../docs/guides/DAGGER_WORKFLOW_SCRIPT_GUIDE.md)

---

## 🎯 **目录结构**

```
tools/dagger/
├── __init__.py                      # 包初始化
├── README.md                        # 本文件
├── record_manual_chopping.py        # 录制工具 (Pygame+鼠标)
├── evaluate_policy.py               # 评估工具
├── run_policy_collect_states.py    # 状态收集工具
└── label_states.py                  # 标注工具
```

---

**版本**: 1.0.0  
**更新日期**: 2025-10-22

