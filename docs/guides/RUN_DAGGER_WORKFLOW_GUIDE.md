# 📘 run_dagger_workflow.sh 使用指南

## 🎯 目录

1. [快速开始](#快速开始)
2. [完整参数说明](#完整参数说明)
3. [常用场景](#常用场景)
4. [跳过特定步骤](#跳过特定步骤)
5. [继续训练](#继续训练)
6. [故障排查](#故障排查)

---

## 快速开始

### **场景1: 从零开始完整的DAgger训练**

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

这会执行：
1. 录制10个专家演示
2. 训练BC基线
3. 评估BC基线
4. 进行3轮DAgger迭代

---

## 完整参数说明

### **任务配置**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task TASK_ID` | MineDojo任务ID | `harvest_1_log` |
| `--method METHOD` | 训练方法 (dagger/ppo/hybrid) | `dagger` |
| `--device DEVICE` | 训练设备 (auto/cpu/cuda/mps) | `cpu` |

### **录制配置**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-episodes N` | 录制专家演示数量 | `10` |
| `--mouse-sensitivity N` | 鼠标灵敏度 (0.1-2.0) | `0.15` |
| `--max-frames N` | 每个episode最大帧数 | `6000` |
| `--no-skip-idle` | 保存所有帧（包括IDLE） | `false` |
| `--append-recording` | 追加录制（不覆盖已有数据） | `false` |

### **训练配置**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bc-epochs N` | BC训练轮数 | `50` |
| `--iterations N` | DAgger迭代次数 | `3` |
| `--collect-episodes N` | 每轮收集episode数 | `20` |
| `--eval-episodes N` | 评估episode数 | `20` |

### **跳过步骤**

| 参数 | 说明 |
|------|------|
| `--skip-recording` | 跳过手动录制（假设已有数据） |
| `--skip-bc` | 跳过BC训练（假设已有BC模型） |

### **继续训练**

| 参数 | 说明 |
|------|------|
| `--continue-from MODEL` | 从指定模型继续DAgger训练 |
| `--start-iteration N` | 从第N轮DAgger开始 |

---

## 常用场景

### **场景2: 已有数据，从BC训练开始**

如果你已经录制了数据（例如20个episodes），想跳过录制直接训练：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3
```

### **场景3: 补录更多数据**

如果BC效果不好，想补录更多数据（追加模式）：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 30 \
    --append-recording \
    --skip-bc \
    --iterations 0
```

说明：
- `--append-recording`: 不会删除已有的20个episodes，从episode_020继续
- `--skip-bc`: 只录制，不训练
- `--iterations 0`: 不执行DAgger迭代

录制完成后，重新训练BC：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3
```

### **场景4: 已有BC模型，直接开始DAgger**

如果你已经有一个训练好的BC模型：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --iterations 3
```

前提：
- 数据已存在于 `data/expert_demos/harvest_1_log/`
- BC模型已存在于 `checkpoints/dagger/harvest_1_log/bc_baseline.zip`

### **场景5: 继续更多轮DAgger迭代**

完成了3轮DAgger后，想再做2轮：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --start-iteration 4 \
    --iterations 2
```

说明：
- `--continue-from`: 使用第3轮训练的模型
- `--start-iteration 4`: 从第4轮开始
- `--iterations 2`: 再执行2轮（第4和第5轮）

最终会生成：
- `checkpoints/dagger/harvest_1_log/dagger_iter_4.zip`
- `checkpoints/dagger/harvest_1_log/dagger_iter_5.zip`

### **场景6: 调整训练参数重新开始**

如果想用更多epochs重新训练BC：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --bc-epochs 100 \
    --iterations 3
```

---

## 跳过特定步骤

### **只录制数据，不训练**

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 0
```

### **只训练BC基线，不做DAgger**

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 0
```

### **只评估现有模型**

```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20 \
    --task-id harvest_1_log
```

---

## 继续训练详解

### **自动推断起始迭代**

如果不指定 `--start-iteration`，脚本会自动从模型文件名推断：

```bash
# 这两个命令等价
bash scripts/run_dagger_workflow.sh \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 2

# 自动推断: start_iteration=4, 执行iter 4和5
```

### **完整的多轮训练示例**

```bash
# 第1阶段: 初始训练（BC + 3轮DAgger）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --iterations 3

# 检查性能，如果不满意...

# 第2阶段: 补录数据 + 重新训练BC
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 30 \
    --append-recording \
    --iterations 0

bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --bc-epochs 100 \
    --iterations 3

# 第3阶段: 继续更多轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --iterations 5
```

---

## 故障排查

### **问题1: "未找到专家演示数据"**

```
✗ 数据路径: data/expert_demos/harvest_1_log
```

**解决**：
- 确保已录制数据：`ls data/expert_demos/harvest_1_log/`
- 或者移除 `--skip-recording` 参数

### **问题2: "BC模型不存在"**

```
✗ 错误: 模型文件不存在: checkpoints/dagger/harvest_1_log/bc_baseline.zip
```

**解决**：
- 移除 `--skip-bc` 参数，让脚本训练BC模型
- 或者手动训练：
  ```bash
  bash scripts/run_minedojo_x86.sh python src/training/train_bc.py \
      --data data/expert_demos/harvest_1_log \
      --output checkpoints/dagger/harvest_1_log/bc_baseline.zip
  ```

### **问题3: BC基线效果很差（IDLE > 70%）**

**原因**: 数据量太少（< 20 episodes）

**解决**：
```bash
# 补录数据到50+ episodes
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 50 \
    --append-recording \
    --iterations 0

# 重新训练BC
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --bc-epochs 100 \
    --iterations 3
```

### **问题4: 标注时退出了，如何继续？**

DAgger的标注是可以中断的：

```bash
# 重新运行标注工具，会从上次中断处继续
bash scripts/run_minedojo_x86.sh python tools/dagger/label_states.py \
    --states data/policy_states/harvest_1_log/iter_1 \
    --output data/expert_labels/harvest_1_log/iter_1.pkl \
    --smart-sampling
```

### **问题5: 如何查看训练进度？**

```bash
# 查看当前有哪些模型
ls -lh checkpoints/dagger/harvest_1_log/

# 评估特定模型
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_2.zip \
    --episodes 20
```

---

## 高级用法

### **多任务训练**

```bash
# 任务1: harvest_1_log (砍树)
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --iterations 3

# 任务2: harvest_milk (挤奶)
bash scripts/run_dagger_workflow.sh \
    --task harvest_milk \
    --num-episodes 20 \
    --iterations 3
```

数据会自动保存到不同目录：
- `data/expert_demos/harvest_1_log/`
- `data/expert_demos/harvest_milk/`

### **混合训练方法**

```bash
# 先用DAgger训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --method dagger \
    --iterations 3

# 然后用PPO精调
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --method ppo \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip
```

---

## 📊 预期结果

### **正常的训练曲线**

| 阶段 | 成功率 | IDLE占比 |
|------|--------|----------|
| BC基线 | 30-50% | < 30% |
| DAgger Iter 1 | 50-70% | < 20% |
| DAgger Iter 2 | 70-85% | < 10% |
| DAgger Iter 3 | 85-95% | < 5% |

### **如果BC基线很差**

| 症状 | 原因 | 解决方案 |
|------|------|----------|
| IDLE > 70% | 数据量太少 | 补录到50+ episodes |
| 成功率 < 10% | 模型过拟合 | 增加epochs，降低learning rate |
| 动作分布偏离训练数据 | 数据质量差 | 重新录制高质量数据 |

---

## 🎯 推荐工作流

### **第一次使用DAgger**

```bash
# 步骤1: 录制20个高质量演示
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 20 \
    --iterations 0

# 步骤2: 训练BC + 3轮DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3

# 步骤3: 评估结果
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --episodes 50
```

### **如果BC基线效果差**

```bash
# 补录30个episodes（总共50个）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 30 \
    --append-recording \
    --iterations 0

# 重新训练BC（更多epochs）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --bc-epochs 100 \
    --iterations 3
```

### **如果想快速迭代**

即使BC基线差（IDLE 70%），也可以立即开始DAgger：

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --iterations 5  # 更多轮次补偿BC的不足
```

DAgger能通过收集失败状态并标注，快速改进策略！

---

## 📝 相关文档

- [DAgger快速开始](DAGGER_QUICK_START.md)
- [BC训练指南](BC_TRAINING_QUICK_START.md)
- [录制控制说明](RECORDING_CONTROLS.md)
- [Pygame鼠标控制](PYGAME_MOUSE_CONTROL.md)

---

**版本**: 1.0.0  
**更新日期**: 2025-10-22

