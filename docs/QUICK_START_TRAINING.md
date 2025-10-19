# harvest_1_paper 训练快速开始

## 30秒开始训练

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 检查设备（可选，查看是否支持 GPU 加速）
python scripts/check_device.py

# 3. 快速测试（10K步，5-10分钟）
./scripts/train_harvest.sh test

# 4. 完整训练（500K步，2-4小时）
./scripts/train_harvest.sh

# 5. 监控训练（实时查看日志）
./scripts/monitor_training.sh

# 6. 评估模型
./scripts/eval_harvest.sh
```

## 查看 Loss 和训练数据

### 方法1: TensorBoard（推荐，可视化）

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 在浏览器打开: http://localhost:6006
# 点击 SCALARS 标签页查看所有曲线
```

**关键指标**：
- 📈 `rollout/ep_rew_mean` - 平均奖励（最重要！）
- 📉 `train/policy_loss` - 策略损失
- 📉 `train/value_loss` - 价值损失
- 📉 `train/entropy_loss` - 熵损失

### 方法2: 实时日志

```bash
# 实时查看训练日志
tail -f logs/training/training_*.log

# 或使用监控脚本
./scripts/monitor_training.sh
```

## MPS 加速支持 🍎

**Apple Silicon (M1/M2/M3) 用户福音！**

训练脚本自动检测并使用 MPS 加速，速度比 CPU 快 **2-3 倍**！

```bash
# 自动检测设备（推荐）
./scripts/train_harvest.sh

# 显式使用 MPS
python src/training/train_harvest_paper.py --device mps

# 使用 CPU（对比性能）
python src/training/train_harvest_paper.py --device cpu
```

训练开始时会显示：
```
🍎 检测到 Apple Silicon，使用 MPS 加速
```

## 重要说明

⚠️ **MineDojo 内置任务不提供预训练模型，训练从头开始！**

- 默认任务：`harvest_milk`（更稳定，建议先测试）
- 目标任务：`harvest_1_paper`（可在脚本中修改）
- 预期时间：
  - 快速测试：5-10分钟
  - 标准训练：2-4小时（MPS）/ 4-8小时（CPU）
  - 完整训练：8-16小时（MPS）/ 16-32小时（CPU）

## 文件位置

- **训练脚本**: `src/training/train_harvest_paper.py`
- **环境包装**: `src/utils/env_wrappers.py`
- **配置文件**: `config/training_config.yaml`
- **检查点**: `checkpoints/harvest_paper/`
- **日志**: `logs/training/` 和 `logs/tensorboard/`

## 详细文档

- **[训练指南](TRAINING_HARVEST_PAPER.md)**: 完整训练文档
- **[监控指南](MONITORING_TRAINING.md)**: 如何查看 Loss 和训练数据
- **[任务系统](MINEDOJO_TASKS_GUIDE.md)**: MineDojo 任务机制

## 常见问题

### 训练脚本无法执行？
```bash
chmod +x scripts/train_harvest.sh scripts/eval_harvest.sh
```

### 找不到模块？
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 想要更快的训练？
```bash
# 使用GPU（确保安装CUDA版PyTorch）
python src/training/train_harvest_paper.py --device cuda

# 使用多个并行环境
python src/training/train_harvest_paper.py --n-envs 4
```

---

**开始你的训练之旅！** 🚀

