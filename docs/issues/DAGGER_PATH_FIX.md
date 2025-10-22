# DAgger 数据路径修正

## ❌ **问题**

执行 `train_dagger.py` 时报错：

```bash
$ python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/round_0/ \
    --new-data data/expert_labels/iter_1.pkl \
    --output checkpoints/dagger_iter_1.zip \
    --epochs 30

ValueError: 不支持的数据格式: data/expert_demos/round_0
```

## 🔍 **原因**

1. **实际数据结构**：
   ```
   data/expert_demos/
   ├── episode_000/
   │   ├── frame_0000.npy
   │   ├── frame_0001.npy
   │   └── ...
   ├── episode_001/
   └── summary.txt
   ```

2. **文档中的错误路径**：
   - 文档示例使用了 `data/expert_demos/round_0/`
   - 但实际录制脚本 `tools/record_manual_chopping.py` 直接保存到 `data/expert_demos/episode_XXX/`

3. **正确的路径**：
   - 应该使用 `data/expert_demos/`（父目录）
   - `load_expert_demonstrations` 会自动递归查找所有 `episode_XXX/` 子目录

---

## ✅ **解决方案**

### **正确的命令**

```bash
# 训练BC基线
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 30 \
    --learning-rate 3e-4 \
    --batch-size 64

# DAgger迭代1
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/ \
    --new-data data/expert_labels/iter_1.pkl \
    --output checkpoints/dagger_iter_1.zip \
    --epochs 30

# DAgger迭代2
python src/training/train_dagger.py \
    --iteration 2 \
    --base-data data/dagger/combined_iter_1.pkl \
    --new-data data/expert_labels/iter_2.pkl \
    --output checkpoints/dagger_iter_2.zip \
    --epochs 30
```

---

## 📊 **数据加载逻辑**

`src/training/train_bc.py` 的 `load_expert_demonstrations` 函数支持3种格式：

### **格式1: 多个episode目录** ✅ 手动录制格式
```
data/expert_demos/
├── episode_000/
│   ├── frame_0000.npy
│   ├── frame_0001.npy
│   └── ...
└── episode_001/
    ├── frame_0000.npy
    └── ...
```

**加载方式**:
```python
python ... --data data/expert_demos/
```

**输出**:
```
从目录加载: data/expert_demos
  找到 2 个episode目录
  [episode_000] 加载 455 个帧...
    ✓ episode_000: 成功加载 455 帧
  [episode_001] 加载 312 个帧...
    ✓ episode_001: 成功加载 312 帧
  转置图像: (N, H, W, C) -> (N, C, H, W)
  归一化图像: [0, 255] -> [0, 1]
```

---

### **格式2: episode_*.npy文件** ✅ run_policy_collect_states格式
```
data/policy_states/iter_1/
├── episode_0.npy
├── episode_1.npy
└── ...
```

**加载方式**:
```python
python ... --data data/policy_states/iter_1/
```

---

### **格式3: .pkl文件** ✅ label_states标注格式
```
data/expert_labels/iter_1.pkl
```

**加载方式**:
```python
python ... --data data/expert_labels/iter_1.pkl
```

---

## 🎯 **完整DAgger流程**

### **阶段0: 录制专家演示**
```bash
# 录制10次成功砍树
python tools/record_manual_chopping.py \
    --max-frames 500 \
    --camera-delta 1
# 按ESC退出后，数据保存在:
# data/expert_demos/episode_000/ ~ episode_009/
```

### **阶段1: BC基线训练**
```bash
# 训练BC基线
python src/training/train_bc.py \
    --data data/expert_demos/ \
    --output checkpoints/bc_baseline.zip \
    --epochs 30

# 评估BC基线
python tools/evaluate_policy.py \
    --model checkpoints/bc_baseline.zip \
    --episodes 20
```

### **阶段2: DAgger迭代优化**

#### **迭代1**
```bash
# 1. 收集失败状态
python tools/run_policy_collect_states.py \
    --model checkpoints/bc_baseline.zip \
    --episodes 20 \
    --output data/policy_states/iter_1/ \
    --save-failures-only

# 2. 标注失败状态
python tools/label_states.py \
    --states data/policy_states/iter_1/ \
    --output data/expert_labels/iter_1.pkl \
    --smart-sampling

# 3. 聚合数据并训练
python src/training/train_dagger.py \
    --iteration 1 \
    --base-data data/expert_demos/ \
    --new-data data/expert_labels/iter_1.pkl \
    --output checkpoints/dagger_iter_1.zip \
    --epochs 30

# 4. 评估改进
python tools/evaluate_policy.py \
    --model checkpoints/dagger_iter_1.zip \
    --episodes 20
```

#### **迭代2**
```bash
python tools/run_policy_collect_states.py \
    --model checkpoints/dagger_iter_1.zip \
    --episodes 20 \
    --output data/policy_states/iter_2/ \
    --save-failures-only

python tools/label_states.py \
    --states data/policy_states/iter_2/ \
    --output data/expert_labels/iter_2.pkl \
    --smart-sampling

# 注意: 这里使用上一轮的聚合数据作为base-data
python src/training/train_dagger.py \
    --iteration 2 \
    --base-data data/dagger/combined_iter_1.pkl \
    --new-data data/expert_labels/iter_2.pkl \
    --output checkpoints/dagger_iter_2.zip \
    --epochs 30

python tools/evaluate_policy.py \
    --model checkpoints/dagger_iter_2.zip \
    --episodes 20
```

---

## 📝 **注意事项**

1. ✅ **第一次DAgger迭代**: 使用 `--base-data data/expert_demos/` (原始专家演示目录)
2. ✅ **后续DAgger迭代**: 使用 `--base-data data/dagger/combined_iter_N.pkl` (上一轮聚合数据)
3. ✅ **录制数据路径**: 直接是 `data/expert_demos/`，不需要 `round_0/` 子目录
4. ✅ **检查点命名**: 建议使用 `bc_baseline.zip` 而不是 `bc_round_0.zip`，更清晰

---

## 🎉 **已修正**

- ✅ `docs/guides/DAGGER_QUICK_START.md` - 所有路径已修正
- ✅ `docs/guides/BC_TRAINING_QUICK_START.md` - 需要检查（如果存在）
- ✅ `scripts/run_dagger_workflow.sh` - 需要检查（如果存在）

---

**当前你的数据情况**:
- ✅ `data/expert_demos/episode_000/` - 已录制1个episode（455帧）
- ⚠️ **建议**: 再录制9个episode（共10个），然后再开始BC训练
- ⚠️ **最少**: 至少录制5个episode才能有效训练BC基线

**下一步**:
```bash
# 继续录制（会自动保存为 episode_001, episode_002, ...）
python tools/record_manual_chopping.py \
    --max-frames 500 \
    --camera-delta 1
```

