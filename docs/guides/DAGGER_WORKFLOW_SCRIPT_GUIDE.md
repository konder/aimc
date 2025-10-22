# DAgger工作流脚本使用指南

> **一键完成**: BC训练 + DAgger迭代优化全流程

---

## 🚀 **快速开始**

### **基础用法**

```bash
# 激活环境
conda activate minedojo

# 运行完整工作流（默认3次迭代）
bash scripts/run_dagger_workflow.sh
```

**执行流程**：
1. 手动录制专家演示（10-15个episode）
2. 训练BC基线模型
3. 评估BC成功率
4. DAgger迭代1：收集失败 → 标注 → 训练 → 评估
5. DAgger迭代2：收集失败 → 标注 → 训练 → 评估
6. DAgger迭代3：收集失败 → 标注 → 训练 → 评估
7. 显示完整训练历史

**预计时间**: 3-5小时（取决于迭代次数和标注速度）

---

## ⚙️ **配置参数**

### **基本参数**

```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \          # 任务ID
    --iterations 3 \                # DAgger迭代次数
    --bc-epochs 50 \                # BC训练轮数
    --collect-episodes 20 \         # 每轮收集episode数
    --eval-episodes 20              # 评估episode数
```

### **跳过选项**

```bash
# 跳过录制（使用已有数据）
bash scripts/run_dagger_workflow.sh --skip-recording

# 跳过BC训练（使用已有BC模型）
bash scripts/run_dagger_workflow.sh --skip-bc

# 两者都跳过（继续DAgger迭代）
bash scripts/run_dagger_workflow.sh --skip-recording --skip-bc --iterations 2
```

---

## 📋 **完整参数列表**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `harvest_1_log` | MineDojo任务ID |
| `--iterations` | `3` | DAgger迭代次数 |
| `--bc-epochs` | `50` | BC训练epoch数 |
| `--collect-episodes` | `20` | 每轮收集的episode数 |
| `--eval-episodes` | `20` | 评估用的episode数 |
| `--skip-recording` | false | 跳过手动录制 |
| `--skip-bc` | false | 跳过BC训练 |
| `-h, --help` | - | 显示帮助信息 |

---

## 🎯 **使用场景**

### **场景1: 从零开始训练**

```bash
# 完整流程（包含录制）
bash scripts/run_dagger_workflow.sh
```

**适用于**:
- 第一次训练此任务
- 没有任何专家演示数据
- 想要完整体验DAgger流程

---

### **场景2: 已有数据，重新训练**

```bash
# 跳过录制，使用已有数据
bash scripts/run_dagger_workflow.sh --skip-recording
```

**适用于**:
- 已录制过专家演示
- 想调整训练参数重新训练
- 之前训练中断，想重新开始

---

### **场景3: 继续DAgger迭代**

```bash
# 跳过录制和BC，继续迭代
bash scripts/run_dagger_workflow.sh \
    --skip-recording \
    --skip-bc \
    --iterations 2
```

**适用于**:
- 已有BC基线模型
- 想增加更多DAgger迭代
- 前几轮效果不错，想继续优化

---

### **场景4: 快速测试**

```bash
# 减少迭代和episode数，快速测试
bash scripts/run_dagger_workflow.sh \
    --iterations 1 \
    --bc-epochs 10 \
    --collect-episodes 10 \
    --eval-episodes 10
```

**适用于**:
- 测试脚本是否正常工作
- 快速验证想法
- 调试问题

---

## 📊 **输出说明**

### **控制台输出**

```
============================================================================
阶段1: BC基线训练
============================================================================

训练参数:
  数据目录: data/expert_demos
  训练轮数: 50
  学习率: 0.0003
  批次大小: 64

[训练过程...]

✓ BC训练完成: checkpoints/bc_baseline.zip

============================================================================
阶段2: 评估BC基线
============================================================================

ℹ️  评估BC策略 (20 episodes)...
✓ BC基线成功率: 65.0%

============================================================================
阶段3: DAgger迭代 1/3
============================================================================

ℹ️  [1] 步骤1: 收集策略失败状态...
✓ 状态收集完成: data/policy_states/iter_1

ℹ️  [1] 步骤2: 智能标注失败场景...
[标注界面...]
✓ 标注完成: data/expert_labels/iter_1.pkl

ℹ️  [1] 步骤3: 聚合数据并训练DAgger模型...
✓ DAgger训练完成: checkpoints/dagger_iter_1.zip

ℹ️  [1] 步骤4: 评估迭代 1 策略...
✓ 迭代 1 成功率: 78.0%

[迭代2, 3...]

============================================================================
训练完成！
============================================================================

训练历史:
  BC基线:       65.0%
  DAgger迭代1:  78.0%
  DAgger迭代2:  85.0%
  DAgger迭代3:  91.0%

最终模型: checkpoints/dagger_iter_3.zip
```

---

### **生成的文件**

```
data/
├── expert_demos/           # 专家演示数据
│   ├── episode_000/
│   ├── episode_001/
│   └── ...
├── policy_states/          # 策略收集的状态
│   ├── iter_1/
│   │   ├── episode_000.npy
│   │   └── ...
│   ├── iter_2/
│   └── iter_3/
├── expert_labels/          # 标注数据
│   ├── iter_1.pkl
│   ├── iter_2.pkl
│   └── iter_3.pkl
└── dagger/                 # 聚合数据
    ├── combined_iter_1.pkl
    ├── combined_iter_2.pkl
    └── combined_iter_3.pkl

checkpoints/
├── bc_baseline.zip         # BC基线模型
├── dagger_iter_1.zip       # DAgger迭代1
├── dagger_iter_2.zip       # DAgger迭代2
└── dagger_iter_3.zip       # DAgger迭代3 (最终)
```

---

## 🎮 **交互式操作**

### **录制专家演示**

脚本会提示：
```
准备录制专家演示数据...
请在游戏中演示如何完成任务 (harvest_1_log)
建议录制 10-15 个成功的episode

控制说明:
  WASD     - 移动
  IJKL     - 视角
  F        - 攻击
  Space    - 跳跃
  Q        - 重录当前回合
  ESC      - 退出录制

按Enter开始录制，或按Ctrl+C取消...
```

**操作**:
1. 按Enter启动Minecraft
2. 演示砍树过程
3. 获得木头后自动保存
4. 继续录制下一个episode
5. 完成10-15个后按ESC退出

---

### **标注失败场景**

脚本会提示：
```
[1] 步骤2: 智能标注失败场景...
⚠️  即将打开标注界面，请手动标注失败场景

标注控制:
  WASD/IJKL/F  - 标注动作
  N            - 跳过当前状态
  Z            - 撤销上一个标注
  X/ESC        - 完成标注

按Enter开始标注...
```

**标注界面显示**:
```
Progress: 1/200 | Priority: HIGH
Episode: 3 | Step: 445
Policy Action: [1 0 0 16 12 0 0 0]
>>> 前进 + 向下看

[显示当前画面]
```

**操作提示**:
- 看到AI做错了（如向下看而不是向上看树）
- 按正确的键（如'I'向上看）
- 智能采样会跳过大部分状态，只标注关键失败点
- 预计每轮标注30-40分钟

---

## ⚠️ **常见问题**

### **Q1: 脚本中断了，如何继续？**

```bash
# 如果在BC之前中断
bash scripts/run_dagger_workflow.sh

# 如果BC已完成，在DAgger迭代中中断
bash scripts/run_dagger_workflow.sh --skip-recording --skip-bc
```

---

### **Q2: 想调整某个阶段的参数？**

**修改脚本顶部的配置变量**：
```bash
# 编辑 scripts/run_dagger_workflow.sh
BC_EPOCHS=100          # 增加BC训练轮数
DAGGER_ITERATIONS=5    # 增加DAgger迭代次数
FAILURE_WINDOW=5       # 只标注失败前5步
```

---

### **Q3: BC成功率太低（<50%）怎么办？**

**方案1: 增加专家演示**
```bash
# 先单独录制更多数据
python tools/record_manual_chopping.py \
    --base-dir data/expert_demos \
    --max-episodes 20

# 然后重新训练
bash scripts/run_dagger_workflow.sh --skip-recording
```

**方案2: 调整BC参数**
```bash
bash scripts/run_dagger_workflow.sh \
    --bc-epochs 100 \
    --skip-recording
```

---

### **Q4: DAgger迭代没有提升？**

**检查标注质量**:
- 标注是否正确？
- 是否标注了足够的关键状态？
- 智能采样是否太激进？

**调整参数**:
```bash
bash scripts/run_dagger_workflow.sh \
    --collect-episodes 30 \      # 收集更多失败
    --skip-recording \
    --skip-bc
```

**手动标注更多**:
```bash
# 关闭智能采样，标注所有状态
# 修改脚本中的 SMART_SAMPLING=false
```

---

## 🔧 **高级用法**

### **自定义任务**

```bash
# 训练其他MineDojo任务
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_wool \
    --iterations 3
```

---

### **并行评估**

```bash
# 在后台运行，输出到日志
nohup bash scripts/run_dagger_workflow.sh > dagger_training.log 2>&1 &

# 查看进度
tail -f dagger_training.log
```

---

### **批量测试**

```bash
# 测试不同迭代次数的效果
for n in 1 2 3 4 5; do
    echo "Testing $n iterations..."
    bash scripts/run_dagger_workflow.sh \
        --iterations $n \
        --skip-recording \
        > "results_iter_${n}.log"
done
```

---

## 📈 **性能预期**

| 阶段 | 成功率 | 时间 |
|------|--------|------|
| BC基线 | 50-65% | 1小时 |
| DAgger迭代1 | 70-78% | +1小时 |
| DAgger迭代2 | 80-85% | +1小时 |
| DAgger迭代3 | 85-92% | +1小时 |

**总计**: 4-5小时达到90%+成功率

---

## 🎯 **最佳实践**

1. **录制专家演示时**:
   - 保持操作一致性
   - 确保每次都成功
   - 包含不同场景（近/远距离，不同树种）

2. **标注时**:
   - 专注失败前的关键步骤
   - 使用智能采样节省时间
   - 跳过不确定的状态（按'N'）

3. **迭代策略**:
   - BC基线尽量达到50%+
   - 每次迭代目标提升10-15%
   - 如果某轮没提升，检查标注质量

4. **数据管理**:
   - 定期备份checkpoints
   - 保存所有迭代的评估日志
   - 记录每轮的改进点

---

## 📚 **相关文档**

- [`DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger手动流程
- [`BC_TRAINING_QUICK_START.md`](BC_TRAINING_QUICK_START.md) - BC训练详解
- [`DAGGER_DETAILED_GUIDE.md`](DAGGER_DETAILED_GUIDE.md) - DAgger算法详解

---

## 🆘 **获取帮助**

```bash
# 查看脚本帮助
bash scripts/run_dagger_workflow.sh --help

# 查看详细日志
# 脚本会将评估结果保存在 /tmp/bc_eval.txt 和 /tmp/dagger_iter_*.txt
cat /tmp/bc_eval.txt
cat /tmp/dagger_iter_1_eval.txt
```

---

**祝训练顺利！** 🚀

如有问题，请查阅详细文档或提issue。

