# MineCLIP 课程学习策略指南

## 概述

本文档介绍如何使用MineCLIP动态权重调整（课程学习）来提升训练效果。

## 什么是课程学习？

课程学习（Curriculum Learning）是一种从简单到复杂的训练策略，类似于人类学习过程：
- **初期**：高MineCLIP权重，强引导 → agent学会基本探索（如"找到树"）
- **中期**：权重逐渐降低 → agent开始依赖自身策略
- **后期**：低MineCLIP权重 → agent主要依赖稀疏奖励完成任务（如"砍树获得木头"）

## 为什么需要动态权重？

### 固定权重的问题

```python
# ❌ 固定低权重（0.1）- 早期探索困难
mineclip_weight = 0.1  # agent不知道去哪里找树

# ❌ 固定高权重（1.0）- 后期依赖过度
mineclip_weight = 1.0  # agent只会"看树"，不会完成任务
```

### 动态权重的优势

```python
# ✅ 动态调整
初始: mineclip_weight = 1.0   # 强引导，帮助探索
↓
中期: mineclip_weight = 0.5   # 逐渐独立
↓
最终: mineclip_weight = 0.01  # 主要靠稀疏奖励
```

## 权重衰减策略

当前实现使用**余弦衰减**：

```
权重
  ↑
1.0 |‾‾‾╲
    |     ╲
0.5 |      ╲___
    |           ╲___
0.01|________________╲_____
    └────────────────────→ 步数
    0     25k    50k    75k
```

数学公式：
```python
progress = min(step / decay_steps, 1.0)
decay_factor = 0.5 * (1.0 + cos(π * progress))
weight = min_weight + (initial_weight - min_weight) * decay_factor
```

## 使用方法

### 方法1: 训练脚本参数（推荐）

```bash
python src/training/train_get_wood.py \
    --use-mineclip \
    --sparse-weight 10.0 \           # 稀疏奖励权重（固定）
    --mineclip-weight 10.0 \         # MineCLIP初始权重（与稀疏权重相同）
    --use-dynamic-weight \           # 启用动态调整
    --weight-decay-steps 50000 \     # 衰减步数
    --min-weight 0.1                 # 最小权重
```

### 方法2: 使用Shell脚本

编辑 `scripts/train_get_wood.sh`，已经配置好默认参数：

```bash
./scripts/train_get_wood.sh mineclip
```

### 方法3: 禁用动态权重（使用固定权重）

```bash
python src/training/train_get_wood.py \
    --use-mineclip \
    --mineclip-weight 0.1 \
    --no-dynamic-weight              # 禁用，使用固定权重0.1
```

## 参数调优指南

### 1. 初始权重 `--mineclip-weight`

| 值 | 与sparse_weight比例 | 适用场景 | 说明 |
|----|-------------------|---------|------|
| 5.0 | 1:2 | 简单任务 | MineCLIP辅助引导 |
| **10.0** | **1:1（推荐）** | **中等任务** | **MineCLIP与稀疏奖励同等重要** |
| 20.0 | 2:1 | 困难任务 | MineCLIP主导，强引导 |

**原则**：初始阶段MineCLIP应该有足够的权重来引导agent探索

### 2. 衰减步数 `--weight-decay-steps`

| 值 | 总训练步数 | 说明 |
|----|-----------|------|
| 30000 | 100k | 快速降低权重 |
| **50000** | **200k（推荐）** | **平衡衰减** |
| 100000 | 500k | 缓慢衰减 |

**经验公式**：`decay_steps = total_steps * 0.25`

### 3. 最小权重 `--min-weight`

| 值 | 与初始权重比例 | 说明 |
|----|--------------|------|
| 0.01 | 0.1% | 几乎完全移除MineCLIP |
| **0.1（推荐）** | **1%** | **保持微弱引导信号** |
| 1.0 | 10% | 保持较强引导 |

**原则**：最小权重应该是初始权重的1%-10%，让agent在后期主要依赖稀疏奖励

### 4. 稀疏权重 `--sparse-weight`

| 值 | 说明 |
|----|------|
| 5.0 | MineCLIP主导 |
| **10.0（推荐）** | 平衡 |
| 20.0 | 稀疏奖励主导 |

## 实时监控

### 训练日志显示

新的实时日志会显示：

```
==================================================================================================================================
🚀 开始训练...
==================================================================================================================================
  回合数 |       步数 |     总时间 |      FPS |     总奖励 |   MineCLIP |  MC权重 |   权重比 |   相似度 |       损失
----------------------------------------------------------------------------------------------------------------------------------
       5 |        500 | 00:02:15 |    220.5 |     0.0234 |     0.0123 |  10.0000 |     1.00 |   0.5234 |     0.0456
      12 |      1,200 | 00:05:23 |    245.6 |     0.1234 |     0.0567 |   9.5000 |     1.05 |   0.6123 |     0.0389
      25 |      2,500 | 00:11:05 |    248.2 |     0.2567 |     0.0892 |   8.2000 |     1.22 |   0.6789 |     0.0312
     150 |     50,000 | 03:20:15 |    250.1 |     2.5678 |     0.3456 |   0.1000 |   100.00 |   0.8456 |     0.0156
```

关键指标：
- **MineCLIP**：未加权的MineCLIP奖励（原始相似度进步）
- **MC权重**：当前MineCLIP权重（动态变化，从10.0→0.1）
- **权重比**：sparse_weight / mineclip_weight（从1.0→100.0）
- **相似度**：与任务目标的相似度（0-1）

**权重比解读**：
- 比例 1:1（权重比=1.0）→ 初期，MineCLIP和稀疏奖励同等重要
- 比例 10:1（权重比=10.0）→ 中期，逐渐侧重稀疏奖励
- 比例 100:1（权重比=100.0）→ 后期，主要依赖稀疏奖励

### TensorBoard查看

TensorBoard会自动记录（如果MineCLIP包装器实现了记录）：
- `reward/mineclip_weight`: 权重变化曲线
- `reward/mineclip_raw`: 原始MineCLIP奖励
- `reward/similarity`: 相似度曲线

## 典型训练曲线

### 健康的训练过程

```
奖励
  ↑
  |           ／￣￣￣
  |         ／
  |      ／ 
  |   ／
  |／___________________→ 步数
  0    50k   100k  150k

相似度
  ↑
  |    ／￣￣￣￣
  |  ／
  |／___________________→ 步数
  0    50k   100k  150k

权重
  ↑
1.0|‾‾‾╲
  |     ╲___
0.01|_________╲_________→ 步数
  0    50k   100k  150k
```

**说明**：
1. 相似度快速上升（找到目标）
2. 权重逐渐下降
3. 奖励持续增长（完成任务）

### 需要调整的情况

#### 问题1: 相似度不增长

```
相似度
  ↑
  |＿＿＿＿＿＿＿＿＿
  |___________________→ 步数
```

**解决方案**：
- 增加初始权重：`--mineclip-weight 2.0`
- 延长衰减步数：`--weight-decay-steps 100000`

#### 问题2: 奖励在后期不增长

```
奖励
  ↑
  |  ／￣￣￣￣￣
  |／___________________→ 步数
```

**解决方案**：
- 降低最小权重：`--min-weight 0.0`
- 增加稀疏权重：`--sparse-weight 20.0`

## 无头模式控制

### 启用无头模式（默认，推荐用于训练）

```bash
python src/training/train_get_wood.py --headless
```

### 禁用无头模式（用于调试，可以看到游戏画面）

```bash
python src/training/train_get_wood.py --no-headless
```

**注意**：
- 无头模式可以提升训练速度（无渲染开销）
- 调试时使用有头模式可以观察agent行为
- macOS可能需要额外配置才能使用无头模式

## 完整示例

### 示例1: 标准训练（200k步）

```bash
python src/training/train_get_wood.py \
    --total-timesteps 200000 \
    --use-mineclip \
    --mineclip-model data/mineclip/attn.pth \
    --sparse-weight 10.0 \
    --mineclip-weight 10.0 \      # 初始1:1比例
    --use-dynamic-weight \
    --weight-decay-steps 50000 \
    --min-weight 0.1 \            # 最终100:1比例
    --headless \
    --device auto
```

### 示例2: 快速测试（10k步）

```bash
python src/training/train_get_wood.py \
    --total-timesteps 10000 \
    --use-mineclip \
    --mineclip-weight 10.0 \
    --use-dynamic-weight \
    --weight-decay-steps 5000 \
    --min-weight 0.1 \
    --headless
```

### 示例3: 困难任务（需要更强引导）

```bash
python src/training/train_get_wood.py \
    --total-timesteps 500000 \
    --use-mineclip \
    --sparse-weight 10.0 \
    --mineclip-weight 20.0 \       # 初始2:1比例，MineCLIP主导
    --weight-decay-steps 100000 \   # 更长衰减期
    --min-weight 0.2 \              # 最终50:1比例
    --headless
```

## 调试技巧

### 1. 观察权重变化

查看日志中的"权重"列，应该看到平滑下降。

### 2. 对比固定vs动态

```bash
# 运行1: 固定权重
python src/training/train_get_wood.py \
    --use-mineclip --mineclip-weight 0.1 --no-dynamic-weight \
    --tensorboard-dir logs/tensorboard/fixed

# 运行2: 动态权重
python src/training/train_get_wood.py \
    --use-mineclip --mineclip-weight 1.0 --use-dynamic-weight \
    --tensorboard-dir logs/tensorboard/dynamic
```

在TensorBoard中对比两条曲线。

### 3. 检查相似度

如果相似度始终很低（<0.3），可能：
- MineCLIP模型未正确加载
- 任务描述不匹配
- 图像预处理有问题

## 常见问题

**Q: 什么时候不需要动态权重？**
A: 
- 任务非常简单（几千步就能完成）
- 已经找到了最优的固定权重
- 纯探索任务（无明确稀疏奖励）

**Q: 权重降到最小值后会继续变化吗？**
A: 不会，达到`min_weight`后会保持不变。

**Q: 可以在训练中途改变权重策略吗？**
A: 可以，通过加载检查点并修改参数继续训练：
```bash
# 第一阶段：高权重探索
python train_get_wood.py --mineclip-weight 2.0 --total-timesteps 50000

# 第二阶段：从检查点继续，降低权重
python train_get_wood.py --load-checkpoint xxx.zip --mineclip-weight 0.1
```

**Q: 如何知道权重衰减是否太快/太慢？**
A: 观察奖励曲线：
- 太快：奖励在中期停止增长
- 太慢：后期仍然高度依赖MineCLIP
- 合适：奖励持续平稳增长

## 总结

✅ **推荐默认配置**：
```bash
--sparse-weight 10.0         # 稀疏奖励权重
--mineclip-weight 10.0       # 初始权重与稀疏权重相同（比例1:1）
--use-dynamic-weight         # 启用动态调整
--weight-decay-steps 50000   # 为总步数的25%
--min-weight 0.1             # 最终降到初始值的1%（10.0→0.1）
--headless                   # 无头模式
```

**权重设置原则**：
- **初期比例 1:1**（sparse:mineclip = 10:10）→ MineCLIP和稀疏奖励同等重要
- **最终比例 100:1**（sparse:mineclip = 10:0.1）→ 主要依赖稀疏奖励

这个配置适用于大多数MineDojo任务。根据具体任务表现微调参数。

## 相关文档

- [MineCLIP奖励详解](MINECLIP_REWARD_EXPLAINED.md)
- [MineCLIP设置指南](MINECLIP_SETUP_GUIDE.md)
- [TensorBoard中文指南](TENSORBOARD_中文指南.md)

