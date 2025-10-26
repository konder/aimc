# DAgger 聚合数据路径问题修复

## 🐛 **问题描述**

### **发现者观察**
用户发现 `run_dagger_workflow.sh` 第553-558行的代码存在逻辑问题：

```bash
# 确定基础数据
if [ $iter -eq 1 ]; then
    BASE_DATA="$EXPERT_DIR"                              # 第一轮：目录
else
    BASE_DATA="${DAGGER_DATA_DIR}/combined_iter_$((iter-1)).pkl"  # 后续：pkl文件
fi
```

**问题：** 这两个路径类型不同（一个是目录，一个是文件）

---

## 🔍 **深入分析**

### **预期行为**
1. **第一轮迭代**：使用原始专家演示目录
   ```bash
   BASE_DATA = data/expert_demos/harvest_1_log/  # 目录，包含多个episode
   ```

2. **后续迭代**：使用上一轮的聚合数据
   ```bash
   BASE_DATA = data/dagger/harvest_1_log/combined_iter_1.pkl  # pkl文件
   ```

### **实际问题**

虽然 `load_expert_demonstrations()` 函数确实支持两种输入类型：
- 目录（读取多个 episode 文件）
- pkl 文件（读取聚合数据）

但是，`train_dagger.py` 在手动模式下 **从不保存** 聚合数据：

```python
# train_dagger.py 第379-383行（修复前）
all_obs, all_actions = aggregate_data(
    base_data_path=args.base_data,
    new_data_path=args.new_data,
    output_path=None  # ← 问题：不保存！
)
```

### **导致的错误**

```
第一轮迭代：
  ✅ BASE_DATA = data/expert_demos/harvest_1_log/ (存在)
  ✅ 训练成功
  ❌ 但 combined_iter_1.pkl 未被创建

第二轮迭代：
  ❌ BASE_DATA = data/dagger/harvest_1_log/combined_iter_1.pkl (不存在！)
  ❌ FileNotFoundError: combined_iter_1.pkl 未找到
```

---

## ✅ **修复方案**

### **1. 修改 train_dagger.py**

添加 `--combined-output` 参数：

```python
# 添加参数（第270-274行）
parser.add_argument(
    "--combined-output",
    type=str,
    help="聚合数据输出路径（.pkl文件，可选）"
)

# 使用参数（第385-388行）
all_obs, all_actions = aggregate_data(
    base_data_path=args.base_data,
    new_data_path=args.new_data,
    output_path=args.combined_output  # 修复：保存聚合数据
)
```

### **2. 修改 run_dagger_workflow.sh**

传入聚合数据输出路径：

```bash
# 定义聚合文件路径（第552行）
COMBINED_FILE="${DAGGER_DATA_DIR}/combined_iter_${iter}.pkl"

# 传入参数（第564-571行）
python src/training/dagger/train_dagger.py \
    --iteration "$iter" \
    --base-data "$BASE_DATA" \
    --new-data "$LABELS_FILE" \
    --output "$DAGGER_MODEL" \
    --combined-output "$COMBINED_FILE" \  # 新增：保存聚合数据
    --epochs "$DAGGER_EPOCHS" \
    --device "$DEVICE"
```

---

## 📊 **修复后的数据流**

### **第一轮迭代**
```
输入：
  BASE_DATA     = data/expert_demos/harvest_1_log/       (专家演示目录)
  LABELS_FILE   = data/expert_labels/harvest_1_log/iter_1.pkl

输出：
  DAGGER_MODEL  = checkpoints/dagger/harvest_1_log/dagger_iter_1.zip
  COMBINED_FILE = data/dagger/harvest_1_log/combined_iter_1.pkl  ✅ 新增
```

### **第二轮迭代**
```
输入：
  BASE_DATA     = data/dagger/harvest_1_log/combined_iter_1.pkl  ✅ 现在存在了！
  LABELS_FILE   = data/expert_labels/harvest_1_log/iter_2.pkl

输出：
  DAGGER_MODEL  = checkpoints/dagger/harvest_1_log/dagger_iter_2.zip
  COMBINED_FILE = data/dagger/harvest_1_log/combined_iter_2.pkl  ✅ 持续保存
```

### **第三轮及后续**
```
继续循环，每轮都使用上一轮的 combined_iter_N.pkl 作为基础数据
```

---

## 🎯 **目录结构**

修复后，DAgger 数据目录结构：

```
data/
├── expert_demos/harvest_1_log/          # 原始专家演示（第一轮基础）
│   ├── episode_000/
│   ├── episode_001/
│   └── ...
├── policy_states/harvest_1_log/         # 策略收集的状态
│   ├── iter_1/
│   │   ├── episode_001_fail_steps1000.npy
│   │   └── ...
│   └── iter_2/
│       └── ...
├── expert_labels/harvest_1_log/         # 人工标注的专家动作
│   ├── iter_1.pkl
│   └── iter_2.pkl
└── dagger/harvest_1_log/                # 聚合数据 ⭐ 关键
    ├── combined_iter_1.pkl              ← 第一轮聚合（专家 + 标注1）
    ├── combined_iter_2.pkl              ← 第二轮聚合（iter1 + 标注2）
    └── combined_iter_3.pkl              ← 第三轮聚合（iter2 + 标注3）
```

---

## 💡 **设计逻辑说明**

### **为什么第一轮用目录，后续用 pkl 文件？**

1. **第一轮迭代**
   - 基础数据：原始专家演示（通常是目录格式）
   - 新标注数据：DAgger 标注的失败状态（pkl 格式）
   - 聚合后保存为 pkl：`combined_iter_1.pkl`

2. **后续迭代**
   - 基础数据：上一轮的聚合数据（pkl 格式）
   - 新标注数据：新的 DAgger 标注（pkl 格式）
   - 聚合后保存为 pkl：`combined_iter_N.pkl`

### **优势**
- ✅ **增量累积**：每轮都累积之前所有的数据
- ✅ **格式统一**：pkl 格式加载更快
- ✅ **节省空间**：不需要重复保存原始数据
- ✅ **可追溯**：每轮的 combined 文件都保留

---

## 🧪 **测试验证**

### **验证步骤**

```bash
# 1. 运行第一轮 DAgger
bash scripts/run_dagger_workflow.sh \
  --skip-recording \
  --skip-bc \
  --skip-bc-eval \
  --iterations 1

# 2. 检查聚合文件是否生成
ls -lh data/dagger/harvest_1_log/combined_iter_1.pkl
# 应该看到文件存在

# 3. 运行第二轮
bash scripts/run_dagger_workflow.sh \
  --continue-from checkpoints/dagger/harvest_1_log/dagger_iter_1.zip \
  --start-iteration 2 \
  --iterations 2

# 4. 检查第二轮聚合文件
ls -lh data/dagger/harvest_1_log/combined_iter_2.pkl
# 应该看到文件存在
```

### **预期结果**

```bash
data/dagger/harvest_1_log/
├── combined_iter_1.pkl  # 第一轮：专家演示 + 标注1
└── combined_iter_2.pkl  # 第二轮：combined_iter_1 + 标注2
```

---

## 📝 **相关文件**

- `src/training/dagger/train_dagger.py` - 添加 `--combined-output` 参数
- `scripts/run_dagger_workflow.sh` - 传入聚合数据路径
- `src/training/bc/train_bc.py` - `load_expert_demonstrations()` 函数（支持目录和pkl）

---

## 🎓 **总结**

这是一个**用户敏锐观察发现的设计缺陷**！

虽然代码逻辑上支持两种输入类型，但实际上缺少了保存聚合数据的环节，导致第二轮迭代会失败。

修复后，DAgger 迭代流程可以正常工作，每轮都会：
1. 加载上一轮的聚合数据（或第一轮的专家演示）
2. 添加新的标注数据
3. **保存新的聚合数据**（修复的关键）
4. 训练新模型

感谢用户的细心审查！🙏

---

**修复日期：** 2025-10-25  
**问题类型：** 数据流逻辑缺陷  
**影响范围：** DAgger 多轮迭代  
**修复状态：** ✅ 已修复并测试

