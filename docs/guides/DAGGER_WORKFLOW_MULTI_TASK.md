# DAgger多任务工作流指南

> **功能**: 支持多任务独立训练和追加录制

---

## 🎯 **核心特性**

### **1. 任务隔离**

每个任务有独立的数据和模型目录：

```
data/
├── expert_demos/
│   ├── harvest_1_log/          # 砍树任务
│   │   ├── episode_000/
│   │   ├── episode_001/
│   │   └── ...
│   └── harvest_1_wool/         # 获取羊毛任务
│       ├── episode_000/
│       └── ...
├── policy_states/
│   ├── harvest_1_log/
│   └── harvest_1_wool/
├── expert_labels/
│   ├── harvest_1_log/
│   └── harvest_1_wool/
└── dagger/
    ├── harvest_1_log/
    └── harvest_1_wool/

checkpoints/
├── harvest_1_log/
│   ├── bc_baseline.zip
│   ├── dagger_iter_1.zip
│   └── ...
└── harvest_1_wool/
    ├── bc_baseline.zip
    └── ...
```

### **2. 追加录制**

支持在已有数据基础上继续录制：

```bash
# 第一次录制 5 个 episodes
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 5

# 后续追加录制 5 个 episodes（共10个）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --append-recording \
    --skip-bc \
    --skip-dagger
```

---

## 📋 **使用场景**

### **场景1: 训练新任务**

```bash
# 完整流程：录制 → BC → DAgger
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3
```

**执行步骤**:
1. 录制 10 个 episodes
2. 训练 BC 基线
3. 执行 3 轮 DAgger 迭代

---

### **场景2: 追加录制数据**

```bash
# 第一次录制了 3 个 episodes，发现不够
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 3

# 追加录制 7 个（共 10 个）
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --append-recording \
    --skip-bc  # 跳过BC训练，稍后重新训练
```

**数据变化**:
```
录制前: data/expert_demos/harvest_1_log/
        ├── episode_000/
        ├── episode_001/
        └── episode_002/

录制后: data/expert_demos/harvest_1_log/
        ├── episode_000/
        ├── episode_001/
        ├── episode_002/
        ├── episode_003/  ← 新增
        ├── episode_004/  ← 新增
        ...
        └── episode_009/  ← 新增
```

---

### **场景3: 多任务并行训练**

```bash
# 任务1: 砍树
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10

# 任务2: 获取羊毛
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_wool \
    --num-episodes 10

# 任务3: 挖石头
bash scripts/run_dagger_workflow.sh \
    --task harvest_10_cobblestone \
    --num-episodes 10
```

**优势**:
- 各任务数据互不干扰
- 可以并行训练（分别执行）
- 便于对比不同任务的训练效果

---

### **场景4: BC训练效果差，增加数据**

```bash
# 第一次训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 5

# BC评估: 成功率只有 30%，太低了！

# 追加录制 5 个 episodes
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --append-recording

# 重新训练 BC（使用全部 10 个 episodes）
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/ \
    --output checkpoints/harvest_1_log/bc_baseline.zip \
    --epochs 50

# 重新评估: 成功率提升到 60%！✅
```

---

## 🛠️ **命令行参数**

### **基础参数**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task TASK_ID` | 任务ID | `harvest_1_log` |
| `--num-episodes N` | 目标专家演示数量 | `10` |
| `--camera-delta N` | 相机灵敏度 | `1` |
| `--max-frames N` | 每个episode最大帧数 | `500` |

### **工作流控制**

| 参数 | 说明 |
|------|------|
| `--append-recording` | 追加录制（继续已有数据） |
| `--skip-recording` | 跳过录制，使用已有数据 |
| `--skip-bc` | 跳过BC训练，使用已有BC模型 |

### **训练参数**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--iterations N` | DAgger迭代次数 | `3` |
| `--bc-epochs N` | BC训练轮数 | `50` |
| `--collect-episodes N` | 每轮收集episode数 | `20` |
| `--eval-episodes N` | 评估episode数 | `20` |

---

## 📊 **实战示例**

### **示例1: 从零开始训练砍树任务**

```bash
# 完整流程
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --iterations 3 \
    --bc-epochs 50

# 预期耗时: 3-4小时
# 预期成功率: BC 60% → 迭代3后 85-90%
```

---

### **示例2: 分阶段录制和训练**

```bash
# 阶段1: 先录制 3 个 episodes 测试流程
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 3 \
    --skip-bc  # 只录制，不训练

# 阶段2: 追加录制到 10 个
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --append-recording \
    --skip-bc

# 阶段3: 开始训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --iterations 3
```

---

### **示例3: 多任务对比实验**

```bash
# 训练3个不同任务，对比BC基线效果
for task in harvest_1_log harvest_1_wool harvest_10_cobblestone; do
    bash scripts/run_dagger_workflow.sh \
        --task "$task" \
        --num-episodes 10 \
        --iterations 0  # 只训练BC，不做DAgger
done

# 查看对比结果
echo "BC基线成功率对比:"
echo "harvest_1_log:         60%"
echo "harvest_1_wool:        45%"
echo "harvest_10_cobblestone: 35%"
```

---

## ⚠️ **注意事项**

### **1. 追加录制模式**

使用 `--append-recording` 时：
- ✅ 自动检测已有 episodes
- ✅ 从下一个编号开始录制
- ✅ 保留所有已有数据
- ❌ 不会覆盖已有数据

**不使用** `--append-recording` 时：
- ⚠️ 发现已有数据会提示
- ⚠️ 需要手动确认是否覆盖

### **2. 数据路径**

```bash
# ✅ 正确：使用任务级别的目录
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/

# ❌ 错误：不要指定具体的 episode 目录
python src/training/train_bc.py \
    --data data/expert_demos/harvest_1_log/episode_000/
```

### **3. 任务ID**

支持的任务ID参考 `docs/technical/MINEDOJO_TASKS_REFERENCE.md`

常用任务：
- `harvest_1_log` - 砍1棵树
- `harvest_10_log` - 砍10棵树
- `harvest_1_wool` - 获取1个羊毛
- `harvest_10_cobblestone` - 挖10个圆石

---

## 📈 **数据管理最佳实践**

### **1. 渐进式录制**

```bash
# 阶段1: 录制 3-5 个测试
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 5 \
    --skip-bc

# 检查数据质量（是否都成功？）
ls -la data/expert_demos/harvest_1_log/

# 阶段2: 追加到 10 个
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --append-recording
```

### **2. 版本管理**

```bash
# 为重要的训练数据创建备份
cp -r data/expert_demos/harvest_1_log \
      data/expert_demos/harvest_1_log_v1_20251022

# 或使用 git 管理数据（需要 git-lfs）
git lfs track "data/expert_demos/**/*.npy"
git add data/expert_demos/harvest_1_log/
git commit -m "添加harvest_1_log训练数据 v1"
```

### **3. 清理旧数据**

```bash
# 删除特定任务的数据
rm -rf data/expert_demos/harvest_1_log/
rm -rf checkpoints/harvest_1_log/

# 删除所有DAgger中间数据（保留专家演示）
rm -rf data/policy_states/*/
rm -rf data/expert_labels/*/
rm -rf data/dagger/*/
```

---

## 🎯 **快速参考**

### **首次录制**
```bash
bash scripts/run_dagger_workflow.sh --task TASK_ID --num-episodes 10
```

### **追加录制**
```bash
bash scripts/run_dagger_workflow.sh \
    --task TASK_ID \
    --num-episodes 15 \
    --append-recording \
    --skip-bc
```

### **只训练（不录制）**
```bash
bash scripts/run_dagger_workflow.sh \
    --task TASK_ID \
    --skip-recording \
    --iterations 3
```

### **只录制（不训练）**
```bash
bash scripts/run_dagger_workflow.sh \
    --task TASK_ID \
    --num-episodes 10 \
    --skip-bc
```

---

## 📚 **相关文档**

- [`DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger快速开始
- [`DAGGER_DETAILED_GUIDE.md`](DAGGER_DETAILED_GUIDE.md) - 详细算法说明
- [`DAGGER_WORKFLOW_SCRIPT_GUIDE.md`](DAGGER_WORKFLOW_SCRIPT_GUIDE.md) - 脚本使用指南
- [`MINEDOJO_TASKS_REFERENCE.md`](../technical/MINEDOJO_TASKS_REFERENCE.md) - 任务列表

---

**祝训练顺利！** 🚀

