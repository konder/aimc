# BC训练快速开始指南

> **目的**: 使用手动录制的数据训练行为克隆（Behavioral Cloning）模型

---

## 📋 **前提条件**

1. ✅ 已录制专家演示数据（使用`record_manual_chopping.py`）
2. ✅ 数据目录结构正确（`episode_000/`, `episode_001/`, ...）
3. ✅ 每个episode包含`frame_*.npy`文件

---

## 📂 **数据目录结构**

```
data/expert_demos/
├── episode_000/
│   ├── frame_00000.npy  # {'observation': obs, 'action': action}
│   ├── frame_00001.npy
│   └── ...
├── episode_001/
├── episode_002/
└── summary.txt
```

**验证数据**：
```bash
ls -l data/expert_demos/episode_000/ | head
# 应该看到 frame_00000.npy, frame_00001.npy, ...
```

---

## 🚀 **训练步骤**

### **Step 1: 激活Conda环境**

```bash
conda activate minedojo
```

---

### **Step 2: 训练BC模型**

```bash
# 基础训练（推荐用于测试）
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 30

# 完整训练（更多epochs）
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_final.zip \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001
```

---

### **Step 3: 查看训练输出**

训练脚本会输出：

```
从目录加载: data/expert_demos
  找到 10 个episode目录
  [episode_000] 加载 234 个帧...
    ✓ episode_000: 成功加载 234 帧
  [episode_001] 加载 189 个帧...
    ✓ episode_001: 成功加载 189 帧
  ...

总计:
  观察: (2073, 160, 256, 3)
  动作: (2073, 8)

开始训练...
Epoch 1/30: Loss=2.345, Accuracy=0.234
Epoch 2/30: Loss=1.987, Accuracy=0.345
...
Epoch 30/30: Loss=0.876, Accuracy=0.678

✓ 模型已保存: checkpoints/bc_baseline.zip
```

---

## 📊 **训练参数说明**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | - | 数据目录（必需） |
| `--output` | - | 输出模型路径（必需） |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--learning-rate` | 0.001 | 学习率 |
| `--test-split` | 0.2 | 测试集比例 |

---

## 🎯 **推荐配置**

### **快速测试（5分钟）**

```bash
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_test.zip \
    --epochs 10 \
    --batch-size 64
```

**用途**:
- 验证数据格式正确
- 快速检查训练流程
- 不期望好的性能

---

### **基线训练（30分钟）**

```bash
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 50 \
    --batch-size 32
```

**用途**:
- 建立性能基线
- 用于后续DAgger迭代
- 可直接评估效果

---

### **完整训练（2小时）**

```bash
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_final.zip \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.0005
```

**用途**:
- 追求最佳性能
- 用于最终部署
- 需要充足数据（>5000帧）

---

## 📈 **评估模型**

训练完成后，使用评估工具：

```bash
python tools/evaluate_policy.py \
    --policy checkpoints/bc_baseline.zip \
    --task harvest_1_log_forest \
    --episodes 10
```

**输出示例**：
```
Episode 1: Reward=1.0, Steps=234, Success=True
Episode 2: Reward=0.0, Steps=500, Success=False
...
Average Reward: 0.6
Success Rate: 60%
```

---

## 🔧 **常见问题**

### **Q1: 报错 "未加载到任何数据"**

**原因**: 数据目录结构不正确

**检查**:
```bash
ls data/expert_demos/
# 应该看到: episode_000/, episode_001/, ...

ls data/expert_demos/episode_000/
# 应该看到: frame_00000.npy, frame_00001.npy, ...
```

**解决**: 确保使用最新的`record_manual_chopping.py`录制数据

---

### **Q2: 训练Loss不下降**

**可能原因**:
1. 数据量太少（<1000帧）
2. 数据质量差（随机操作）
3. 学习率过高

**解决**:
```bash
# 降低学习率
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 50 \
    --learning-rate 0.0001

# 或增加数据量（录制更多episode）
python tools/record_manual_chopping.py --max-episodes 20
```

---

### **Q3: 训练Accuracy很低**

**原因**: BC训练的Accuracy是逐维度匹配的准确率

**解释**:
- MineDojo动作空间有8个维度
- 每个维度有多个可能值（如camera有25个值）
- 完全匹配所有8个维度很困难
- **Accuracy 0.3-0.5 是正常的**

**关键指标**: 在环境中评估实际表现（成功率）

---

### **Q4: 内存不足**

**原因**: 数据量大，全部加载到内存

**解决**:
```bash
# 减少batch size
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --batch-size 16

# 或者只使用部分数据
python src/training/train_bc.py \
    --data data/expert_demos/episode_000/ \
    --output checkpoints/bc_test.zip
```

---

## 📚 **下一步**

训练完BC模型后，可以：

1. **评估模型**:
   ```bash
   python tools/evaluate_policy.py \
       --policy checkpoints/bc_baseline.zip \
       --episodes 10
   ```

2. **进行DAgger迭代**:
   ```bash
   python src/training/train_dagger.py \
       --initial-policy checkpoints/bc_baseline.zip \
       --iterations 5
   ```

3. **可视化训练**:
   ```bash
   tensorboard --logdir logs/bc/
   ```

---

## 🎯 **预期效果**

根据数据质量和数量：

| 数据量 | Epochs | 预期成功率 | 训练时间 |
|--------|--------|-----------|---------|
| 1000帧 (2-3 episodes) | 30 | 20-40% | 5分钟 |
| 2000帧 (5-10 episodes) | 50 | 40-60% | 15分钟 |
| 5000帧 (20-30 episodes) | 100 | 60-80% | 1小时 |

**注意**:
- 成功率取决于任务难度
- `harvest_1_log_forest`相对简单
- 更复杂任务需要更多数据

---

## 🔗 **相关文档**

- [`docs/guides/DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger快速开始
- [`docs/reference/ROUND_VS_EPISODE_EXPLAINED.md`](../reference/ROUND_VS_EPISODE_EXPLAINED.md) - Episode概念说明
- [`tools/record_manual_chopping.py`](../../tools/record_manual_chopping.py) - 数据录制工具

---

**最后更新**: 2025-10-21  
**状态**: ✅ BC训练脚本已支持递归加载episode子目录

