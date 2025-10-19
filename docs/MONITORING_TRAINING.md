# 训练监控指南

本文档详细说明如何监控和查看 MineDojo 训练过程中的各种数据。

---

## 目录

1. [查看 Loss 和训练指标](#1-查看-loss-和训练指标)
2. [TensorBoard 可视化](#2-tensorboard-可视化)
3. [查看日志文件](#3-查看日志文件)
4. [关键指标解读](#4-关键指标解读)
5. [MPS 设备支持](#5-mps-设备支持)

---

## 1. 查看 Loss 和训练指标

### 方法1: TensorBoard（推荐）

TensorBoard 提供最直观的可视化界面：

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 在浏览器打开
http://localhost:6006
```

**可以看到的指标**：

| 标签页 | 指标 | 说明 |
|--------|------|------|
| **SCALARS** | `train/policy_loss` | 策略损失（越低越好） |
| | `train/value_loss` | 价值函数损失 |
| | `train/entropy_loss` | 熵损失（探索程度） |
| | `train/approx_kl` | KL散度（策略变化） |
| | `train/clip_fraction` | 裁剪比例 |
| | `train/learning_rate` | 当前学习率 |
| | `rollout/ep_rew_mean` | 平均episode奖励 |
| | `rollout/ep_len_mean` | 平均episode长度 |
| | `eval/mean_reward` | 评估平均奖励 |
| **DISTRIBUTIONS** | 参数分布 | 网络权重分布 |
| **GRAPHS** | 计算图 | 模型结构可视化 |

### 方法2: 控制台实时输出

训练过程中会实时打印：

```
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 234      |
|    ep_rew_mean     | 0.15     |
| time/              |          |
|    fps             | 42       |
|    iterations      | 10       |
|    time_elapsed    | 485      |
|    total_timesteps | 20480    |
| train/             |          |
|    approx_kl       | 0.012    |
|    clip_fraction   | 0.089    |
|    clip_range      | 0.2      |
|    entropy_loss    | -2.45    |
|    learning_rate   | 0.0003   |
|    loss            | 1.23     |
|    policy_gradient_loss | -0.01 |
|    value_loss      | 0.45     |
---------------------------------
```

### 方法3: 查看日志文件

训练日志保存在 `logs/training/` 目录：

```bash
# 实时查看最新日志
tail -f logs/training/training_*.log

# 查看所有日志
cat logs/training/training_*.log

# 搜索特定指标
grep "ep_rew_mean" logs/training/training_*.log
```

---

## 2. TensorBoard 可视化

### 2.1 启动 TensorBoard

```bash
# 基本用法
tensorboard --logdir logs/tensorboard

# 指定端口
tensorboard --logdir logs/tensorboard --port 6007

# 绑定到所有网络接口（远程访问）
tensorboard --logdir logs/tensorboard --bind_all
```

### 2.2 TensorBoard 界面使用

#### 📊 Scalars（标量）

查看 loss 和各种指标的时间序列曲线。

**常用操作**：
- **缩放**：鼠标滚轮或拖拽
- **平滑**：左侧 "Smoothing" 滑块调整曲线平滑度
- **对比**：同时查看多个训练运行
- **下载**：点击曲线右上角下载数据

**关键曲线**：

1. **训练损失曲线**
   ```
   train/policy_loss    # 策略损失
   train/value_loss     # 价值损失  
   train/entropy_loss   # 熵损失
   ```

2. **性能曲线**
   ```
   rollout/ep_rew_mean  # 平均奖励（最重要）
   eval/mean_reward     # 评估奖励
   ```

3. **训练指标**
   ```
   train/approx_kl      # KL散度
   train/clip_fraction  # 裁剪比例
   train/explained_variance  # 解释方差
   ```

#### 📈 典型的健康训练曲线

**Policy Loss**:
```
高 |     ___________
   |    /           \____
   |   /                 \____
低 |__/________________________→ 时间
   开始  探索期   学习期   收敛期
```

**Episode Reward**:
```
高 |                   ________
   |                __/
   |          _____/
低 |_________/___________________→ 时间
   开始  探索期   提升期   稳定期
```

#### 🖼️ Images（图像）

如果记录了观察图像，可以在这里查看智能体看到的画面。

#### 📊 Distributions（分布）

查看网络权重和激活值的分布，用于诊断梯度消失/爆炸。

#### 🔍 Graphs（计算图）

可视化神经网络结构。

---

## 3. 查看日志文件

### 3.1 训练日志

主训练日志包含训练开始、进度、检查点保存等信息：

```bash
# 实时查看
tail -f logs/training/training_20241019_143022.log

# 查看完整日志
cat logs/training/training_*.log

# 只看错误
grep "ERROR\|✗" logs/training/training_*.log
```

**日志内容示例**：
```
[2024-10-19 14:30:22] ======================================================================
[2024-10-19 14:30:22] MineDojo harvest_1_paper 训练开始
[2024-10-19 14:30:22] ======================================================================
[2024-10-19 14:30:22] 训练配置:
[2024-10-19 14:30:22]   任务: harvest_milk
[2024-10-19 14:30:22]   总步数: 500,000
[2024-10-19 14:30:22]   设备: mps
[2024-10-19 14:30:25] [1/5] 创建训练环境...
[2024-10-19 14:30:28]   ✓ 训练环境创建成功
```

### 3.2 TensorBoard 事件文件

TensorBoard 读取 `logs/tensorboard/` 中的事件文件：

```bash
# 查看事件文件
ls -lh logs/tensorboard/ppo_harvest_paper_*/

# 输出示例
-rw-r--r--  events.out.tfevents.1234567890.hostname
```

### 3.3 导出 TensorBoard 数据

```python
# 使用 TensorBoard 数据导出工具
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/tensorboard/ppo_harvest_paper_1/')
ea.Reload()

# 获取所有标量
print(ea.Tags()['scalars'])

# 读取特定指标
policy_loss = ea.Scalars('train/policy_loss')
for item in policy_loss:
    print(f"Step: {item.step}, Value: {item.value}")
```

---

## 4. 关键指标解读

### 4.1 Loss 指标

#### Policy Loss（策略损失）
- **含义**: 策略网络的损失，衡量策略更新的幅度
- **正常范围**: 0.001 - 0.1
- **期望趋势**: 
  - 初期：较高且波动（0.05-0.2）
  - 中期：逐渐下降并稳定
  - 后期：小幅波动，维持低值
- **异常情况**:
  - 持续上升 → 学习率过高或探索不足
  - 持续为0 → 模型崩溃
  - 剧烈波动 → 学习率过高或批次太小

#### Value Loss（价值损失）
- **含义**: 价值函数的预测误差
- **正常范围**: 0.1 - 10
- **期望趋势**: 逐渐下降并稳定
- **异常情况**:
  - 持续上升 → 价值函数过拟合
  - 极大值 → 奖励缩放问题

#### Entropy Loss（熵损失）
- **含义**: 策略的随机性，负值
- **正常范围**: -2.0 到 -4.0
- **期望趋势**: 
  - 初期：高熵（接近-2），探索性强
  - 后期：低熵（接近-4），更确定性
- **异常情况**:
  - 快速降至-6以下 → 过早收敛，探索不足
  - 维持在-1附近 → 策略过于随机

### 4.2 性能指标

#### Episode Reward Mean（平均奖励）
- **含义**: 最近 episodes 的平均累计奖励
- **期望趋势**: **逐渐上升**（最重要的指标！）
- **harvest_milk 参考值**:
  - 初期 (0-50K步): 0.0 - 0.1
  - 中期 (50-200K步): 0.1 - 0.5
  - 后期 (200K+步): 0.5 - 1.0
- **异常情况**:
  - 长时间为0 → 任务太难或奖励稀疏
  - 剧烈波动 → 环境随机性大或策略不稳定

#### Episode Length Mean（平均步数）
- **含义**: 平均 episode 持续时间
- **期望趋势**: 
  - 简单任务：逐渐缩短（更快完成）
  - 复杂任务：可能保持稳定
- **harvest_milk**: 通常 200-800 步

### 4.3 训练健康度指标

#### Approx KL（近似KL散度）
- **含义**: 新旧策略的差异
- **正常范围**: 0.01 - 0.03
- **期望**: 保持稳定
- **异常**: > 0.05 表示策略变化过大

#### Clip Fraction（裁剪比例）
- **含义**: 被 PPO 裁剪的样本比例
- **正常范围**: 0.05 - 0.15
- **期望**: 适中，说明 PPO 起作用但不过度
- **异常**: 
  - 接近0 → 学习率可能过低
  - 接近1 → 学习率过高

#### Explained Variance（解释方差）
- **含义**: 价值函数预测的准确度
- **正常范围**: 0.5 - 1.0
- **期望**: 逐渐增加至 > 0.7
- **异常**: < 0 表示价值函数完全错误

---

## 5. MPS 设备支持

### 5.1 什么是 MPS？

**MPS (Metal Performance Shaders)** 是 Apple 为 Apple Silicon（M1/M2/M3 芯片）提供的 GPU 加速框架。

**优势**：
- 🚀 比 CPU 快 2-5 倍
- 🔋 能效更高
- 💻 适合 MacBook 训练

### 5.2 检查 MPS 支持

```python
import torch

# 检查 MPS 是否可用
print(f"PyTorch 版本: {torch.__version__}")
print(f"MPS 可用: {torch.backends.mps.is_available()}")
print(f"MPS 已构建: {torch.backends.mps.is_built()}")

# 如果可用，测试创建张量
if torch.backends.mps.is_available():
    x = torch.randn(3, 3).to('mps')
    print(f"MPS 测试成功: {x.device}")
```

### 5.3 使用 MPS 训练

#### 方法1: 自动检测（推荐）

```bash
# 脚本会自动检测并使用 MPS
./scripts/train_harvest.sh
```

#### 方法2: 显式指定

```bash
# 显式使用 MPS
python src/training/train_harvest_paper.py --device mps

# 或使用训练脚本
./scripts/train_harvest.sh
```

训练开始时会显示：
```
🍎 检测到 Apple Silicon，使用 MPS 加速
```

#### 方法3: 强制使用 CPU（对比性能）

```bash
python src/training/train_harvest_paper.py --device cpu
```

### 5.4 MPS 性能对比

在 M1 MacBook Pro 上的典型速度（harvest_milk，单环境）：

| 设备 | FPS | 10K步耗时 | 相对速度 |
|------|-----|-----------|----------|
| CPU | 15-25 | 8-10 min | 1x |
| MPS | 40-60 | 3-5 min | 2.5x |
| CUDA (参考) | 80-120 | 2-3 min | 4-5x |

### 5.5 MPS 故障排除

#### 问题1: MPS 不可用

```bash
# 检查 PyTorch 版本（需要 >= 1.12）
python -c "import torch; print(torch.__version__)"

# 升级 PyTorch
pip install --upgrade torch torchvision
```

#### 问题2: MPS 训练出错

如果遇到 MPS 相关错误，回退到 CPU：

```bash
python src/training/train_harvest_paper.py --device cpu
```

常见错误：
- `NotImplementedError: The operator ... is not currently implemented for the MPS device`
  → 某些操作 MPS 不支持，使用 CPU

#### 问题3: 内存不足

```bash
# 减少批次大小
python src/training/train_harvest_paper.py \
    --device mps \
    --batch-size 32

# 减少图像尺寸
python src/training/train_harvest_paper.py \
    --device mps \
    --image-size 120 160
```

---

## 6. 实时监控脚本

创建一个便捷的监控脚本：

```bash
#!/bin/bash
# scripts/monitor_training.sh

# 在一个终端显示日志
echo "训练日志:"
tail -f logs/training/training_*.log &
PID1=$!

# 提示打开 TensorBoard
echo ""
echo "========================================"
echo "在另一个终端运行:"
echo "  tensorboard --logdir logs/tensorboard"
echo "然后打开: http://localhost:6006"
echo "========================================"
echo ""

# 等待用户中断
trap "kill $PID1; exit" INT
wait
```

---

## 7. 训练数据分析示例

### 7.1 使用 Python 分析

```python
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# 加载 TensorBoard 数据
ea = event_accumulator.EventAccumulator(
    'logs/tensorboard/ppo_harvest_paper_1/'
)
ea.Reload()

# 提取关键指标
def extract_scalar(tag):
    events = ea.Scalars(tag)
    return pd.DataFrame([
        {'step': e.step, 'value': e.value} 
        for e in events
    ])

# 获取数据
reward_df = extract_scalar('rollout/ep_rew_mean')
policy_loss_df = extract_scalar('train/policy_loss')
value_loss_df = extract_scalar('train/value_loss')

# 绘图
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(reward_df['step'], reward_df['value'])
axes[0].set_title('Episode Reward')
axes[0].set_ylabel('Reward')

axes[1].plot(policy_loss_df['step'], policy_loss_df['value'])
axes[1].set_title('Policy Loss')
axes[1].set_ylabel('Loss')

axes[2].plot(value_loss_df['step'], value_loss_df['value'])
axes[2].set_title('Value Loss')
axes[2].set_ylabel('Loss')

plt.tight_layout()
plt.savefig('training_curves.png')
print("已保存训练曲线到 training_curves.png")
```

### 7.2 生成训练报告

```python
# 简单的训练总结
print("=" * 50)
print("训练总结")
print("=" * 50)
print(f"总步数: {reward_df['step'].max():,}")
print(f"最终平均奖励: {reward_df['value'].iloc[-10:].mean():.3f}")
print(f"最佳奖励: {reward_df['value'].max():.3f}")
print(f"最终策略损失: {policy_loss_df['value'].iloc[-10:].mean():.4f}")
print(f"最终价值损失: {value_loss_df['value'].iloc[-10:].mean():.4f}")
print("=" * 50)
```

---

## 8. 快速参考

### 启动监控
```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# 日志
tail -f logs/training/training_*.log
```

### 关键指标位置
- **TensorBoard**: `http://localhost:6006` → SCALARS 标签页
- **控制台**: 训练过程实时打印
- **日志文件**: `logs/training/training_*.log`

### 最重要的指标
1. 📈 `rollout/ep_rew_mean` - **必看！**奖励增长情况
2. 📉 `train/policy_loss` - 策略是否正常学习
3. 📉 `train/value_loss` - 价值函数质量

### 设备选择
```bash
--device auto   # 自动检测（推荐）
--device mps    # Apple Silicon GPU
--device cuda   # NVIDIA GPU
--device cpu    # CPU
```

---

## 总结

✅ **查看 Loss**: TensorBoard 的 SCALARS 标签页  
✅ **实时监控**: `tail -f logs/training/training_*.log`  
✅ **可视化**: `tensorboard --logdir logs/tensorboard`  
✅ **MPS 加速**: 自动检测 Apple Silicon，速度提升 2-3 倍  
✅ **关键指标**: `ep_rew_mean` 是最重要的性能指标  

**开始监控你的训练吧！** 📊🚀

