# 🚀 DAgger Workflow 快速参考

## 常用命令速查

### 📝 **完整流程**
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 20 --iterations 3
```

### 🔄 **跳过录制，从BC开始**
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --iterations 3
```

### ➕ **补录更多数据（追加模式）**
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 30 --append-recording --iterations 0
```

### 🎯 **只录制，不训练**
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --num-episodes 10 --iterations 0
```

### 🔁 **继续更多轮DAgger**
```bash
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --skip-recording \
    --skip-bc \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --start-iteration 4 \
    --iterations 2
```

### 📊 **评估模型**
```bash
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 20
```

### 🛠️ **重新训练BC（更多epochs）**
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --skip-recording --bc-epochs 100 --iterations 3
```

---

## 参数速查表

| 短参数 | 完整参数 | 说明 | 默认值 |
|--------|----------|------|--------|
| - | `--task` | 任务ID | `harvest_1_log` |
| - | `--num-episodes` | 录制数量 | `10` |
| - | `--iterations` | DAgger轮数 | `3` |
| - | `--bc-epochs` | BC训练轮数 | `50` |
| - | `--skip-recording` | 跳过录制 | `false` |
| - | `--skip-bc` | 跳过BC训练 | `false` |
| - | `--append-recording` | 追加录制 | `false` |
| - | `--continue-from` | 继续训练的模型 | - |
| - | `--start-iteration` | 起始迭代 | 自动推断 |
| - | `--method` | 训练方法 | `dagger` |
| - | `--device` | 训练设备 | `cpu` |

---

## 故障速查

| 问题 | 快速解决 |
|------|----------|
| 未找到数据 | 移除`--skip-recording`或手动录制 |
| BC模型不存在 | 移除`--skip-bc`或手动训练BC |
| IDLE > 70% | 补录到50+ episodes |
| 标注中断 | 重新运行相同命令会继续 |

---

## 目录结构

```
data/expert_demos/
└── harvest_1_log/
    ├── episode_000/
    ├── episode_001/
    └── ...

checkpoints/dagger/harvest_1_log/
├── bc_baseline.zip
├── dagger_iter_1.zip
├── dagger_iter_2.zip
└── dagger_iter_3.zip

data/policy_states/harvest_1_log/
├── iter_1/
├── iter_2/
└── iter_3/

data/expert_labels/harvest_1_log/
├── iter_1.pkl
├── iter_2.pkl
└── iter_3.pkl
```

---

详细文档: `docs/guides/RUN_DAGGER_WORKFLOW_GUIDE.md`

