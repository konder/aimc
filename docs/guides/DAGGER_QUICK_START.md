# DAgger 快速开始指南

> **从零开始**: 使用DAgger算法训练Minecraft砍树AI

---

## 📋 **前置准备**

确保已完成：
- ✅ MineDojo环境安装（`conda activate minedojo-x86`）
- ✅ 项目依赖安装（`pip install -r requirements.txt`）
- ✅ 熟悉手动录制工具（`tools/record_manual_chopping.py`）

---

## 🚀 **完整流程（3-5小时）**

### **阶段1: BC基线训练** (1-2小时)

#### **步骤1: 录制专家演示** ⏱️ 40-60分钟

```bash
# 激活环境
conda activate minedojo-x86

# 录制10次成功砍树演示
# 每次录制约200-500步，直到获得木头（env返回done=True时自动保存）
# 按Q退出当前episode不保存，ESC退出程序
python tools/record_manual_chopping.py \
    --max-frames 500 \
    --camera-delta 1
# 成功获得木头后，会自动保存为 episode_000, episode_001, ...
# 录制10次后按ESC退出
```

**控制说明**:
- `WASD` - 移动
- `IJKL` - 视角
- `F` - 攻击（砍树）
- `Q` - 退出录制

**录制技巧**:
1. 尽量录制完整流程：寻找树 → 靠近 → 砍树 → 获得木头
2. 包含不同场景：近距离、中距离、不同树种
3. 确保每次都成功获得木头（奖励>0）

**预期结果**:
- `data/expert_demos/episode_000/` ~ `episode_009/` (10个目录)
- 每个目录包含 `frame_*.npy` 文件（约200-500帧/episode）
- 总计约4000-5000帧

---

#### **步骤2: 训练BC基线** ⏱️ 30-40分钟

```bash
# 从专家演示训练初始BC策略
# data/expert_demos/ 包含所有 episode_XXX/ 子目录
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 30 \
    --learning-rate 3e-4 \
    --batch-size 64
```

**训练中会显示**:
```
加载了 10 个episode
总计:
  观察: (4523, 3, 160, 160)
  动作: (4523, 8)

Epoch 1/30 | Loss: 2.3456
Epoch 2/30 | Loss: 1.9834
...
Epoch 30/30 | Loss: 0.4521

✓ 模型已保存
```

---

#### **步骤3: 评估BC基线** ⏱️ 10分钟

```bash
# 评估BC策略的成功率
python tools/evaluate_policy.py \
    --model checkpoints/bc_baseline.zip \
    --episodes 20 \
    --task-id harvest_1_log
```

**预期输出**:
```
Episode 1/20 ✓ | 步数:234 | 奖励:  1.00
Episode 2/20 ✗ | 步数:1000 | 奖励:  0.00
Episode 3/20 ✓ | 步数:312 | 奖励:  1.00
...

评估结果
============================================================
成功率: 60.0% (12/20)
平均奖励: 0.60 ± 0.49
平均步数: 487 ± 312
============================================================
```

**决策点**:
- ✅ 成功率 ≥ 50% → 进入阶段2（DAgger优化）
- ❌ 成功率 < 50% → 增加专家演示或调整超参数

---

### **阶段2: DAgger迭代优化** (2-3小时)

#### **迭代1: 纠正偏离场景** ⏱️ 60-80分钟

**步骤1: 收集失败状态** ⏱️ 10分钟

```bash
# 运行BC策略，收集失败的episode
python tools/run_policy_collect_states.py \
    --model checkpoints/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/iter_1/ \
    --save-failures-only \
    --task-id harvest_1_log
```

**预期输出**:
```
Episode 1/20 ✗ 失败 | 步数:456 | 奖励:  0.00
Episode 2/20 ✓ 成功 | 步数:234 | 奖励:  1.00
Episode 3/20 ✗ 失败 | 步数:789 | 奖励:  0.00
...

收集完成！
============================================================
总episode数: 20
成功: 8 (40.0%)
失败: 12 (60.0%)
保存episode数: 12  # 只保存失败的
总状态数: 5432
============================================================
```

---

**步骤2: 智能标注失败场景** ⏱️ 30-40分钟

```bash
# 交互式标注失败状态
python tools/label_states.py \
    --states data/policy_states/iter_1/ \
    --output data/expert_labels/iter_1.pkl \
    --smart-sampling \
    --failure-window 10
```

**标注界面**:
```
开始标注 (543 个状态)  # 智能采样后
============================================================
控制:
  W/S/A/D    - 前进/后退/左/右
  I/K/J/L    - 视角 上/下/左/右
  F          - 攻击（砍树）
  Q          - 前进+攻击
  N          - 跳过此状态
  Z          - 撤销上一个标注
  X/ESC      - 完成标注
============================================================

[显示失败场景的画面]
Progress: 1/543  |  Priority: HIGH
Episode: 3  |  Step: 445
Policy Action: [0 0 0 16 12 0 0 0]  # 策略向下看

你的标注: 按 'I' (应该向上看寻找树)
  ✓ [1/543] 上看

[下一个状态...]
```

**标注策略**:
- 🔴 **失败前10步**: 100%标注（关键决策）
- 🟡 **中间步骤**: 跳过（按'N'）
- 只标注"专家会怎么做"

**预期标注量**: 500-800个关键状态（约30-40分钟）

---

**步骤3: 聚合数据并重新训练** ⏱️ 30-40分钟

```bash
# 聚合原始专家演示 + 新标注数据
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/ \
    --new-data data/expert_labels/iter_1.pkl \
    --output checkpoints/dagger_iter_1.zip \
    --epochs 30
```

**输出**:
```
数据聚合
============================================================
加载基础数据: data/expert_demos/
  找到 10 个episode目录
  [episode_000] 加载 455 个帧...
  ...
  基础数据: 4523 样本
加载新标注数据: data/expert_labels/iter_1.pkl
  新标注: 612 样本
  聚合后: 5135 样本
============================================================

[训练过程...]

✓ 模型已保存: checkpoints/dagger_iter_1.zip
✓ 聚合数据已保存: data/dagger/combined_iter_1.pkl
```

---

**步骤4: 评估改进** ⏱️ 10分钟

```bash
# 评估新策略
python tools/evaluate_policy.py \
    --model checkpoints/dagger_iter_1.zip \
    --episodes 20
```

**预期结果**:
```
成功率: 75.0% (15/20)  # 从60%提升到75% ✅
```

---

#### **迭代2-3: 继续优化** ⏱️ 60-90分钟

重复迭代1的步骤，但使用新模型：

```bash
# 迭代2
python tools/run_policy_collect_states.py \
    --model checkpoints/dagger_iter_1.zip \
    --episodes 20 \
    --output data/policy_states/iter_2/ \
    --save-failures-only

python tools/label_states.py \
    --states data/policy_states/iter_2/ \
    --output data/expert_labels/iter_2.pkl \
    --smart-sampling

python src/training/train_dagger.py \
    --iteration 2 \
    --base-data data/dagger/combined_iter_1.pkl \  # 使用上一轮聚合数据
    --new-data data/expert_labels/iter_2.pkl \
    --output checkpoints/dagger_iter_2.zip

python tools/evaluate_policy.py \
    --model checkpoints/dagger_iter_2.zip \
    --episodes 20

# 预期: 75% → 85% ✅
```

---

## 📈 **预期成功率曲线**

```
成功率
  100% ┤
       │
   90% ┤                             ●─── 迭代3
       │                         ╱
   80% ┤                     ●───     迭代2
       │                 ╱
   70% ┤             ●───             迭代1
       │         ╱
   60% ┤─────●                        BC基线
       │
   50% ┤
       └─────┬─────┬─────┬─────┬─────→ 时间
          BC   迭代1  迭代2  迭代3  迭代4
```

---

## 🎯 **成功指标**

### **BC基线（阶段1）**
- ✅ 成功率: 50-60%
- ✅ 平均获取木头步数: < 500步
- ✅ 能找到并靠近树木

### **DAgger迭代1**
- ✅ 成功率: 70-75%
- ✅ 偏离后能纠正
- ✅ 标注时间: < 40分钟

### **DAgger迭代2-3**
- ✅ 成功率: 85-90%
- ✅ 在不同地形都鲁棒
- ✅ 累计标注时间: < 2小时

---

## 🛠️ **工具总结**

| 工具 | 用途 | 时间 |
|------|------|------|
| `record_manual_chopping.py` | 录制专家演示 | 40-60分钟（一次性）|
| `train_bc.py` | BC训练 | 30分钟/次 |
| `evaluate_policy.py` | 评估成功率 | 10分钟/次 |
| `run_policy_collect_states.py` | 收集失败状态 | 10分钟/轮 |
| `label_states.py` | 交互式标注 | 30-40分钟/轮 |
| `train_dagger.py` | DAgger训练 | 30分钟/轮 |

---

## 💡 **最佳实践**

### **录制演示时**
1. ✅ 保持一致性（相同的操作习惯）
2. ✅ 包含多样性（不同场景、不同树种）
3. ✅ 确保成功（每次都获得木头）
4. ❌ 避免过度复杂（不要绕圈、跳跃等无关动作）

### **标注时**
1. ✅ 专注失败前的关键步骤
2. ✅ 使用智能采样（节省80%时间）
3. ✅ 标注"应该做什么"而非"不应该做什么"
4. ✅ 撤销功能（'Z'）纠正错误
5. ❌ 不需要标注所有状态

### **训练时**
1. ✅ BC基线: epochs=30, lr=3e-4
2. ✅ DAgger: 逐轮降低epochs（30→20→15）
3. ✅ 每轮评估2次（训练前/后）
4. ✅ 保存所有检查点（便于回滚）

---

## ⚠️ **常见问题**

### **Q1: BC成功率只有30%，太低了！**

**解决方案**:
- 增加专家演示到15-20次
- 检查演示质量（是否都成功了？）
- 延长训练轮数（30 → 50 epochs）
- 降低学习率（3e-4 → 1e-4）

---

### **Q2: 标注太慢了！**

**解决方案**:
- 使用 `--smart-sampling`（只标注20-30%）
- 使用 `--failure-window 5`（只标注失败前5步）
- 使用组合键（'Q'=前进+攻击）
- 跳过不确定的状态（按'N'）

---

### **Q3: DAgger迭代没有提升**

**解决方案**:
- 检查标注质量（是否正确？）
- 增加每轮收集的episode（20 → 30）
- 标注更多关键状态
- 检查是否存在系统性错误（如总是向下看）

---

## 📂 **数据组织**

训练完成后的目录结构：

```
data/
├── expert_demos/             # BC专家演示（手动录制）
│   ├── episode_000/
│   │   ├── frame_0000.npy
│   │   ├── frame_0001.npy
│   │   └── ...
│   ├── episode_001/
│   ├── ...
│   └── episode_009/
├── policy_states/
│   ├── iter_1/               # 迭代1失败状态
│   │   ├── episode_0.npy
│   │   ├── episode_1.npy
│   │   └── ...
│   ├── iter_2/
│   └── iter_3/
├── expert_labels/
│   ├── iter_1.pkl            # 迭代1标注
│   ├── iter_2.pkl
│   └── iter_3.pkl
└── dagger/
    ├── combined_iter_1.pkl   # 聚合数据
    ├── combined_iter_2.pkl
    └── combined_iter_3.pkl

checkpoints/
├── bc_baseline.zip          # BC基线（60%）
├── dagger_iter_1.zip        # 迭代1（75%）
├── dagger_iter_2.zip        # 迭代2（85%）
└── dagger_iter_3.zip        # 迭代3（90%）✅ 最终模型
```

---

## 🎉 **完成后**

恭喜！你已经完成DAgger训练，获得了一个90%+成功率的砍树AI！

### **下一步**

1. **测试鲁棒性**: 在不同生物群系测试
   ```bash
   python tools/evaluate_policy.py \
       --model checkpoints/dagger_iter_3.zip \
       --episodes 50
   ```

2. **（可选）PPO精调**: 进一步提升到95%+
   ```bash
   python src/training/train_get_wood.py \
       --resume \
       --checkpoint checkpoints/dagger_iter_3.zip \
       --total-timesteps 100000
   ```

3. **迁移到其他任务**: 用相同方法训练其他任务
   - `harvest_10_log` (砍10棵树)
   - `harvest_1_wool` (获取羊毛)
   - ...

---

## 📚 **相关文档**

- [`DAGGER_DETAILED_GUIDE.md`](DAGGER_DETAILED_GUIDE.md) - DAgger算法详解
- [`DAGGER_VS_BC_COMPARISON.md`](DAGGER_VS_BC_COMPARISON.md) - 可视化对比
- [`IMITATION_LEARNING_GUIDE.md`](IMITATION_LEARNING_GUIDE.md) - 模仿学习概览
- [`DAGGER_IMPLEMENTATION_PLAN.md`](../status/DAGGER_IMPLEMENTATION_PLAN.md) - 详细实施计划

---

**祝训练顺利！** 🚀

如有问题，请参考详细文档或提issue。

