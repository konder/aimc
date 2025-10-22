# DAgger 实施计划

> **目标**: 使用DAgger算法将砍树任务成功率从60%提升到90%+

---

## 📋 **实施路线图**

### **阶段0: 准备工作** ✅ 已完成
- ✅ 手动录制工具已就绪 (`tools/record_manual_chopping.py`)
- ✅ MineCLIP验证工具已完成 (`tools/verify_mineclip_16frames.py`)
- ✅ 环境包装器已完善 (`src/utils/env_wrappers.py`)
- ✅ DAgger理论文档已完成

---

### **阶段1: BC初始训练** (预计1-2天)

#### **任务1.1: 录制专家演示** 
```bash
时间: 1小时
目标: 10次成功砍树演示
工具: tools/record_manual_chopping.py

步骤:
1. python tools/record_manual_chopping.py --output data/expert_demos/round_0/
2. 每次录制 ~200-500 步（找树 → 靠近 → 砍树 → 获得木头）
3. 确保包含不同场景：
   - 近距离看到树
   - 中距离寻找树
   - 不同树木类型
```

**输出**: 
- `data/expert_demos/round_0/episode_*.npy` (10个文件)
- 总帧数: ~4000-5000帧

---

#### **任务1.2: 实现BC训练脚本**
```bash
时间: 2-3小时
文件: src/training/train_bc.py
```

**核心功能**:
1. 加载专家演示数据
2. 训练PPO/BC策略（行为克隆）
3. 保存模型检查点

**预期效果**:
- 初始成功率: ~50-60%
- 训练时间: ~30分钟

---

#### **任务1.3: 评估BC策略**
```bash
时间: 30分钟
工具: tools/evaluate_policy.py

python tools/evaluate_policy.py \
  --model checkpoints/bc_round_0.zip \
  --episodes 20 \
  --task-id harvest_1_log
```

**决策点**:
- ✅ 成功率 ≥ 50% → 进入阶段2
- ❌ 成功率 < 50% → 增加专家演示或调整超参数

---

### **阶段2: DAgger迭代优化** (预计3-5天)

#### **迭代1: 纠正偏离场景**

**步骤1: 收集策略状态**
```bash
时间: 10分钟
工具: tools/run_policy_collect_states.py

python tools/run_policy_collect_states.py \
  --model checkpoints/bc_round_0.zip \
  --episodes 20 \
  --output data/policy_states/iter_1/ \
  --save-failures-only  # 只保存失败的episode
```

**输出**: 
- 失败episode的所有状态
- 预计: ~10-15个失败episode × 200帧 = 2000-3000帧

---

**步骤2: 智能标注**
```bash
时间: 30-40分钟
工具: tools/label_states.py

python tools/label_states.py \
  --states data/policy_states/iter_1/ \
  --output data/expert_labels/iter_1.pkl \
  --smart-sampling  # 智能采样，只标注关键状态
```

**标注策略**:
- 🔴 **优先**: 失败前5-10步（100%标注）
- 🟡 **中等**: 偏离轨迹的状态（50%标注）
- 🟢 **低优先**: 正常执行（跳过，已有专家演示）

**预计标注量**: 500-800个关键状态

---

**步骤3: 聚合数据并重新训练**
```bash
时间: 30分钟
工具: src/training/train_dagger.py

python src/training/train_dagger.py \
  --iteration 1 \
  --base-data data/expert_demos/round_0/ \
  --new-data data/expert_labels/iter_1.pkl \
  --output checkpoints/dagger_iter_1.zip
```

**输出**: 
- 数据集大小: 5000 + 600 = 5600样本
- 新策略: `checkpoints/dagger_iter_1.zip`

---

**步骤4: 评估**
```bash
时间: 10分钟

python tools/evaluate_policy.py \
  --model checkpoints/dagger_iter_1.zip \
  --episodes 20
```

**预期结果**: 成功率 60% → **75%** (+15%)

---

#### **迭代2: 纠正复杂失败场景**

重复迭代1的步骤，但:
- 策略更好，失败更少
- 失败场景更复杂（边界情况）
- 标注量更少（~300-500个状态）

**预期结果**: 成功率 75% → **85%** (+10%)

---

#### **迭代3: 精细优化**

重复迭代1的步骤，但:
- 专注极端边界情况
- 标注量最少（~200-300个状态）

**预期结果**: 成功率 85% → **90%** (+5%)

---

#### **迭代4-5: 收敛（可选）**

如果还未达到90%，继续迭代

**预期结果**: 成功率 90% → **92-95%**

---

### **阶段3: （可选）PPO精调** (预计1-2天)

如果想进一步提升到95%+:

```bash
python src/training/train_get_wood.py \
  --resume \
  --checkpoint checkpoints/dagger_iter_3.zip \
  --total-timesteps 100000 \
  --sparse-weight 1.0 \
  --mineclip-weight 0.0  # 纯稀疏奖励
```

用DAgger策略初始化PPO，再训练10万步

**预期结果**: 成功率 90% → **95-98%**

---

## 🛠️ **需要实现的工具**

### **1. 状态收集工具** (优先级: 🔴 高)
```
文件: tools/run_policy_collect_states.py
功能: 运行策略并保存访问的状态
预计时间: 1-2小时
```

### **2. 交互式标注工具** (优先级: 🔴 高)
```
文件: tools/label_states.py
功能: 显示状态，接受键盘输入，保存标注
预计时间: 2-3小时
```

### **3. BC训练脚本** (优先级: 🔴 高)
```
文件: src/training/train_bc.py
功能: 从专家演示训练策略
预计时间: 1-2小时
```

### **4. DAgger主循环** (优先级: 🟡 中)
```
文件: src/training/train_dagger.py
功能: 自动化迭代流程
预计时间: 2小时
```

### **5. 策略评估工具** (优先级: 🟢 低)
```
文件: tools/evaluate_policy.py
功能: 评估策略成功率
预计时间: 1小时
```

---

## 📊 **预期时间表**

### **Week 1: BC基线**
- Day 1: 录制专家演示（1h）
- Day 2: 实现BC训练脚本（3h）
- Day 3: 训练BC模型，评估基线（2h）

**里程碑**: BC成功率达到50-60%

---

### **Week 2: DAgger迭代1-2**
- Day 1: 实现状态收集工具（2h）
- Day 2: 实现标注工具（3h）
- Day 3: 运行迭代1（标注40min + 训练30min）
- Day 4: 运行迭代2（标注30min + 训练30min）

**里程碑**: 成功率达到75-85%

---

### **Week 3: DAgger迭代3-4**
- Day 1: 运行迭代3（标注20min + 训练30min）
- Day 2: 运行迭代4（如需要）
- Day 3: 大规模评估和测试

**里程碑**: 成功率达到90%+

---

### **Week 4: （可选）PPO精调**
- Day 1-3: PPO训练
- Day 4-5: 最终评估和文档

**里程碑**: 成功率达到95%+

---

## 💾 **数据组织结构**

```
data/
├── expert_demos/
│   └── round_0/
│       ├── episode_0.npy
│       ├── episode_1.npy
│       └── ...              # 10次手动演示
│
├── policy_states/
│   ├── iter_1/
│   │   ├── episode_0_fail.npy
│   │   └── ...              # 失败episode的状态
│   ├── iter_2/
│   └── iter_3/
│
├── expert_labels/
│   ├── iter_1.pkl           # 第1轮标注
│   ├── iter_2.pkl           # 第2轮标注
│   └── iter_3.pkl
│
└── dagger_combined/
    ├── iter_1.pkl           # 聚合数据
    ├── iter_2.pkl
    └── iter_3.pkl

checkpoints/
├── bc_round_0.zip           # BC基线
├── dagger_iter_1.zip        # DAgger第1轮
├── dagger_iter_2.zip        # DAgger第2轮
├── dagger_iter_3.zip        # DAgger第3轮
└── dagger_final.zip         # 最终模型
```

---

## 🎯 **成功指标**

### **技术指标**
- ✅ BC基线成功率: ≥ 50%
- ✅ DAgger迭代3成功率: ≥ 90%
- ✅ 平均获取木头时间: < 300步
- ✅ 策略鲁棒性: 在不同地形/树种都能成功

### **效率指标**
- ✅ 总开发时间: < 3周
- ✅ 总标注时间: < 3小时
- ✅ 总训练时间: < 5小时

---

## ⚠️ **风险和缓解**

### **风险1: BC基线太差（<40%）**
**缓解**: 
- 增加专家演示到15-20次
- 确保演示质量（都是成功的）
- 调整BC超参数（学习率、epoch数）

---

### **风险2: 标注太耗时**
**缓解**:
- 使用智能采样（只标注20-30%）
- 专注失败前5-10步
- 使用快捷键加速标注

---

### **风险3: 迭代收敛慢**
**缓解**:
- 增加每轮收集的episode数（20→30）
- 标注更多关键状态
- 调整训练超参数

---

## 📝 **下一步行动**

### **立即开始**:
1. ✅ 实现 `tools/run_policy_collect_states.py`
2. ✅ 实现 `tools/label_states.py`
3. ✅ 实现 `src/training/train_bc.py`

### **本周完成**:
1. 录制10次专家演示
2. 训练BC基线
3. 评估并确认≥50%成功率

### **下周开始**:
1. 运行DAgger迭代1
2. 观察成功率提升

---

## 🔗 **相关文档**

- [`DAGGER_DETAILED_GUIDE.md`](../guides/DAGGER_DETAILED_GUIDE.md) - DAgger算法详解
- [`DAGGER_VS_BC_COMPARISON.md`](../guides/DAGGER_VS_BC_COMPARISON.md) - 可视化对比
- [`IMITATION_LEARNING_GUIDE.md`](../guides/IMITATION_LEARNING_GUIDE.md) - 模仿学习概览
- [`IMITATION_LEARNING_ROADMAP.md`](IMITATION_LEARNING_ROADMAP.md) - 总体路线图

---

**最后更新**: 2025-10-21  
**状态**: 📍 准备阶段 → 开始实现工具

