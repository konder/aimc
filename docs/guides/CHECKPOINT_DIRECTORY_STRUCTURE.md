# 检查点目录结构指南

> **新的分层目录结构**: 按训练方法和任务分类管理模型

---

## 🎯 **新目录结构**

### **完整目录树**

```
checkpoints/
├── dagger/                    # DAgger训练的模型
│   ├── harvest_1_log/         # 砍1棵树任务
│   │   ├── bc_baseline.zip    # BC基线模型
│   │   ├── dagger_iter_1.zip  # DAgger迭代1
│   │   ├── dagger_iter_2.zip  # DAgger迭代2
│   │   └── dagger_iter_3.zip  # DAgger迭代3（最终模型）
│   ├── harvest_1_wool/        # 获取羊毛任务
│   │   ├── bc_baseline.zip
│   │   ├── dagger_iter_1.zip
│   │   └── dagger_iter_2.zip
│   ├── harvest_10_log/        # 砍10棵树任务
│   │   └── bc_baseline.zip
│   └── harvest_10_cobblestone/ # 挖10个圆石任务
│       ├── bc_baseline.zip
│       └── dagger_iter_1.zip
├── ppo/                       # PPO训练的模型
│   ├── harvest_1_log/
│   │   ├── ppo_10000_steps.zip
│   │   ├── ppo_50000_steps.zip
│   │   ├── ppo_100000_steps.zip
│   │   └── ppo_final.zip      # 最终PPO模型
│   ├── harvest_1_wool/
│   │   ├── ppo_20000_steps.zip
│   │   └── ppo_final.zip
│   └── harvest_10_log/
│       └── ppo_final.zip
└── hybrid/                    # 混合训练（DAgger→PPO）
    ├── harvest_1_log/
    │   ├── dagger_to_ppo_init.zip  # DAgger初始化的PPO
    │   └── dagger_to_ppo_final.zip # 最终混合模型
    └── harvest_1_wool/
        └── dagger_to_ppo_final.zip
```

---

## 📋 **目录分类说明**

### **1. `checkpoints/dagger/`** - DAgger训练模型

**用途**: 存储所有DAgger算法训练的模型
- **BC基线**: `bc_baseline.zip`
- **DAgger迭代**: `dagger_iter_N.zip`

**训练方式**:
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --method dagger
```

**模型命名规则**:
- `bc_baseline.zip` - 行为克隆基线模型
- `dagger_iter_1.zip` - DAgger第1轮迭代
- `dagger_iter_2.zip` - DAgger第2轮迭代
- `dagger_iter_N.zip` - DAgger第N轮迭代

---

### **2. `checkpoints/ppo/`** - PPO训练模型

**用途**: 存储所有PPO算法训练的模型
- **定期保存**: 每10K步保存一次
- **最终模型**: `ppo_final.zip`

**训练方式**:
```bash
bash scripts/train_get_wood.sh quick  # 使用PPO训练
```

**模型命名规则**:
- `ppo_10000_steps.zip` - 训练10K步的模型
- `ppo_50000_steps.zip` - 训练50K步的模型
- `ppo_final.zip` - 训练完成的最终模型

---

### **3. `checkpoints/hybrid/`** - 混合训练模型

**用途**: 存储DAgger初始化后用PPO精调的模型
- **初始化**: 从DAgger模型开始
- **精调**: 使用PPO进一步优化

**训练方式**:
```bash
# 第1步: DAgger训练
bash scripts/run_dagger_workflow.sh --task harvest_1_log --method dagger

# 第2步: PPO精调（从DAgger模型开始）
python src/training/train_get_wood.py config/get_wood_config.yaml \
    --override checkpointing.checkpoint_dir="checkpoints/hybrid/harvest_1_log" \
    --override training.resume_from="checkpoints/dagger/harvest_1_log/dagger_iter_3.zip"
```

**模型命名规则**:
- `dagger_to_ppo_init.zip` - DAgger初始化的PPO模型
- `dagger_to_ppo_final.zip` - 混合训练的最终模型

---

## 🔄 **迁移现有模型**

### **自动迁移脚本**

如果你有旧格式的模型（`checkpoints/TASK_ID/`），使用迁移脚本：

```bash
# 运行迁移脚本
bash scripts/migrate_checkpoints.sh

# 脚本会自动：
# 1. 检测旧格式目录
# 2. 根据模型类型分类迁移
# 3. 显示迁移结果
# 4. 保留原目录（需手动删除）
```

### **手动迁移**

```bash
# 旧结构
checkpoints/harvest_1_log/
├── bc_baseline.zip
├── dagger_iter_1.zip
└── dagger_iter_2.zip

# 迁移到新结构
mkdir -p checkpoints/dagger/harvest_1_log
mv checkpoints/harvest_1_log/* checkpoints/dagger/harvest_1_log/
rmdir checkpoints/harvest_1_log
```

---

## 🛠️ **使用方法**

### **DAgger训练**

```bash
# 完整DAgger流程
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --method dagger \
    --iterations 3

# 继续DAgger训练
bash scripts/run_dagger_workflow.sh \
    --task harvest_1_log \
    --method dagger \
    --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_2.zip \
    --iterations 5

# 模型保存在: checkpoints/dagger/harvest_1_log/
```

### **PPO训练**

```bash
# PPO训练（从零开始）
bash scripts/train_get_wood.sh quick

# PPO训练（从DAgger模型开始）
python src/training/train_get_wood.py config/get_wood_config.yaml \
    --override training.resume_from="checkpoints/dagger/harvest_1_log/dagger_iter_3.zip" \
    --override checkpointing.checkpoint_dir="checkpoints/hybrid/harvest_1_log"

# 模型保存在: checkpoints/ppo/harvest_1_log/ 或 checkpoints/hybrid/harvest_1_log/
```

### **模型评估**

```bash
# 评估DAgger模型
python tools/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/dagger_iter_3.zip \
    --episodes 20

# 评估PPO模型
python tools/evaluate_policy.py \
    --model checkpoints/ppo/harvest_1_log/ppo_final.zip \
    --episodes 20

# 评估混合模型
python tools/evaluate_policy.py \
    --model checkpoints/hybrid/harvest_1_log/dagger_to_ppo_final.zip \
    --episodes 20
```

---

## 📊 **模型性能对比**

### **典型性能表现**

| 训练方法 | 训练时间 | 成功率 | 稳定性 | 适用场景 |
|---------|---------|--------|--------|---------|
| **BC基线** | 30分钟 | 50-60% | 中等 | 快速原型 |
| **DAgger** | 3-4小时 | 85-95% | 高 | 高质量模型 |
| **PPO** | 2-6小时 | 70-85% | 中高 | 传统RL |
| **混合** | 4-8小时 | 90-98% | 最高 | 最佳性能 |

### **选择建议**

#### **快速验证** → BC基线
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 0
# 只训练BC，不做DAgger迭代
```

#### **高质量模型** → DAgger
```bash
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3
# 完整DAgger流程，3-4小时
```

#### **最佳性能** → DAgger + PPO
```bash
# 第1步: DAgger训练
bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3

# 第2步: PPO精调
python src/training/train_get_wood.py config/get_wood_config.yaml \
    --override training.resume_from="checkpoints/dagger/harvest_1_log/dagger_iter_3.zip" \
    --override checkpointing.checkpoint_dir="checkpoints/hybrid/harvest_1_log"
```

---

## 🔍 **目录管理**

### **清理旧模型**

```bash
# 清理特定任务的所有模型
rm -rf checkpoints/*/harvest_1_log/

# 清理特定训练方法的模型
rm -rf checkpoints/dagger/

# 只保留最终模型
find checkpoints/ -name "*_steps.zip" -delete  # 删除中间步骤
find checkpoints/ -name "dagger_iter_[12].zip" -delete  # 只保留最新迭代
```

### **备份重要模型**

```bash
# 备份最佳模型
mkdir -p backups/$(date +%Y%m%d)
cp -r checkpoints/dagger/harvest_1_log/dagger_iter_3.zip backups/$(date +%Y%m%d)/
cp -r checkpoints/hybrid/harvest_1_log/dagger_to_ppo_final.zip backups/$(date +%Y%m%d)/
```

### **查看目录大小**

```bash
# 查看各目录占用空间
du -sh checkpoints/*/

# 查看特定任务占用空间
du -sh checkpoints/*/harvest_1_log/

# 查看总占用空间
du -sh checkpoints/
```

---

## ⚠️ **注意事项**

### **1. 路径兼容性**

**旧脚本可能需要更新路径**:
```bash
# 旧路径
--model checkpoints/harvest_1_log/dagger_iter_3.zip

# 新路径
--model checkpoints/dagger/harvest_1_log/dagger_iter_3.zip
```

### **2. 配置文件更新**

**YAML配置文件需要更新**:
```yaml
# config/get_wood_config.yaml
checkpointing:
  checkpoint_dir: "checkpoints/ppo/harvest_1_log"  # 新路径
```

### **3. 脚本参数**

**新的脚本参数**:
```bash
# DAgger工作流
bash scripts/run_dagger_workflow.sh --method dagger  # 指定训练方法

# 继续训练
--continue-from checkpoints/dagger/harvest_1_log/dagger_iter_2.zip  # 完整路径
```

---

## 📚 **相关文档**

- [`DAGGER_QUICK_START.md`](DAGGER_QUICK_START.md) - DAgger快速开始
- [`DAGGER_CONTINUE_TRAINING.md`](DAGGER_CONTINUE_TRAINING.md) - 继续训练指南
- [`DAGGER_WORKFLOW_MULTI_TASK.md`](DAGGER_WORKFLOW_MULTI_TASK.md) - 多任务工作流
- [`GET_WOOD_CONFIG_GUIDE.md`](GET_WOOD_CONFIG_GUIDE.md) - PPO配置指南

---

**总结**: 新的目录结构让不同训练方法和任务的模型更加清晰分类，便于管理和对比。使用迁移脚本可以轻松从旧结构升级到新结构。
