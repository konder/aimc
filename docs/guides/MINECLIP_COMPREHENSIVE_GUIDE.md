# MineCLIP 综合训练指南

本文档涵盖 MineCLIP 的完整使用流程：从安装配置到高级课程学习策略。

---

## 目录

1. [快速理解 MineCLIP](#快速理解-mineclip)
2. [安装和设置](#安装和设置)
3. [基础使用](#基础使用)
4. [课程学习策略](#课程学习策略)
5. [实时监控和调试](#实时监控和调试)
6. [常见问题](#常见问题)
7. [总结](#总结)

---

## 快速理解 MineCLIP

### 四个关键概念

| 概念 | 作用 | 你需要吗？ |
|------|------|-----------|
| **1. MineCLIP 模型** | 视觉-语言模型，计算画面与文本的相似度 | ✅ **需要** - 核心组件 |
| **2. 预训练权重** (attn.pth/avg.pth) | 模型参数，直接用于推理 | ✅ **需要** - 必须下载 |
| **3. MineAgent** | 策略网络示例（PPO等） | ❌ **不需要** - 你已经用PPO了 |
| **4. 640K视频数据** | 重新训练MineCLIP用的原始数据 | ❌ **不需要** - 研究用途 |

### MineCLIP 工作原理

#### 训练前（预训练阶段）

```
640K YouTube 视频 + 文本描述
         ↓
   MineCLIP 训练
         ↓
预训练权重 (attn.pth)  ← 你下载的就是这个
```

#### 你的训练中（推理阶段）

```
当前游戏画面 (RGB图像)
         ↓
   MineCLIP 编码
         ↓
   图像特征向量 (512维)
         ↓                    任务描述 "chop tree"
   计算相似度  ←──────────────   ↓
         ↓                 MineCLIP 编码
   相似度分数 0.75              ↓
         ↓                 文本特征向量 (512维)
   密集奖励 = 进步量
   (0.75 - 0.65 = +0.10)
```

**每一步都计算相似度 → 连续密集奖励！**

---

## 安装和设置

### 步骤1：安装 MineCLIP 包

```bash
pip install git+https://github.com/MineDojo/MineCLIP
```

### 步骤2：下载预训练权重

**两种变体选择**：

| 变体 | 特点 | 性能 | 推荐度 |
|------|------|------|--------|
| **attn** | 使用注意力机制，模型更大 | 更准确 | ⭐⭐⭐ **推荐** |
| **avg** | 简单平均，模型更小 | 稍差但更快 | ⭐⭐ 资源受限时用 |

**下载地址**（需要从 MineCLIP GitHub 获取）：
- attn.pth - 约500MB
- avg.pth - 约300MB

**存放位置建议**：
```
aimc/
  data/
    mineclip/
      attn.pth    ← 放这里
      avg.pth     ← 或这里
```

#### attn vs avg：应该用哪个？

**attn（推荐）⭐⭐⭐**

**优点**：
- ✅ 性能更好，相似度计算更准确
- ✅ 更好地理解时序信息
- ✅ 官方论文使用的主要变体

**缺点**：
- ⚠️ 模型更大（~500MB）
- ⚠️ 推理稍慢

**适合**：
- 你的主要训练（有足够GPU/MPS内存）
- 追求最佳性能

**avg（备选）⭐⭐**

**优点**：
- ✅ 模型更小（~300MB）
- ✅ 推理更快

**缺点**：
- ⚠️ 性能稍差
- ⚠️ 简单平均可能丢失时序信息

**适合**：
- 快速原型测试
- 资源受限（MPS内存不足）

**建议**：先用 **attn**，如果内存不够再降级到 avg

### 步骤3：修改训练代码

更新 `src/training/train_get_wood.py`：

```python
# 1. 导入官方 MineCLIP wrapper
from src.utils.mineclip_reward import MineCLIPRewardWrapper

# 2. 在创建环境时使用
def create_harvest_log_env(use_mineclip=False, image_size=(160, 256)):
    # 创建基础环境
    env = make_minedojo_env(
        task_id="harvest_1_log",
        image_size=image_size,
        use_frame_stack=False,
        use_discrete_actions=False
    )
    
    # 如果启用MineCLIP
    if use_mineclip:
        env = MineCLIPRewardWrapper(
            env,
            task_prompt="chop down a tree and collect one wood log",
            model_path="data/mineclip/attn.pth",  # ← 指定模型路径
            variant="attn",                        # ← 使用 attn 变体
            sparse_weight=10.0,
            mineclip_weight=10.0,                  # ← 初始权重
            use_dynamic_weight=True,               # ← 启用动态权重
            weight_decay_steps=50000,              # ← 衰减步数
            min_weight=0.1                         # ← 最小权重
        )
    
    return env
```

### 步骤4：运行训练

```bash
bash scripts/train_get_wood.sh test --mineclip
```

---

## 基础使用

### 完整代码示例

#### train_get_wood.py 修改

```python
#!/usr/bin/env python
import os
import sys
import argparse
from datetime import datetime
import gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

try:
    import minedojo
    MINEDOJO_AVAILABLE = True
except ImportError:
    MINEDOJO_AVAILABLE = False
    print("❌ MineDojo未安装")
    sys.exit(1)

from src.utils.realtime_logger import RealtimeLoggerCallback
from src.utils.env_wrappers import make_minedojo_env
from src.utils.mineclip_reward import MineCLIPRewardWrapper  # ← 新增


def create_harvest_log_env(use_mineclip=False, mineclip_model_path=None, image_size=(160, 256)):
    """
    创建采集木头任务环境
    
    Args:
        use_mineclip: 是否使用MineCLIP密集奖励
        mineclip_model_path: MineCLIP模型权重路径
        image_size: 图像尺寸
        
    Returns:
        MineDojo环境
    """
    print(f"创建环境: harvest_1_log (获得1个原木)")
    print(f"  图像尺寸: {image_size}")
    print(f"  MineCLIP: {'启用' if use_mineclip else '禁用'}")
    
    # 使用 env_wrappers 创建环境
    env = make_minedojo_env(
        task_id="harvest_1_log",
        image_size=image_size,
        use_frame_stack=False,
        use_discrete_actions=False
    )
    
    # 如果启用MineCLIP，使用官方包装器
    if use_mineclip:
        env = MineCLIPRewardWrapper(
            env,
            task_prompt="chop down a tree and collect one wood log",
            model_path=mineclip_model_path,  # ← 传入模型路径
            variant="attn",  # 或 "avg"
            sparse_weight=10.0,
            mineclip_weight=10.0,
            use_dynamic_weight=True,
            weight_decay_steps=50000,
            min_weight=0.1
        )
    
    return env

# ... 其余代码保持不变
```

### 运行参数

```python
# 在 main 函数中添加参数
parser.add_argument(
    '--mineclip-model',
    type=str,
    default='data/mineclip/attn.pth',
    help='MineCLIP 模型权重路径'
)

# 创建环境时传入
env_instance = create_harvest_log_env(
    use_mineclip=args.use_mineclip,
    mineclip_model_path=args.mineclip_model,  # ← 传入路径
    image_size=args.image_size
)
```

### 预期效果

#### 使用 MineCLIP attn

```
[1/4] 创建环境...
创建环境: harvest_1_log (获得1个原木)
  图像尺寸: (160, 256)
  MineCLIP: 启用
  MineCLIP 奖励包装器:
    任务描述: chop down a tree and collect one wood log
    模型变体: attn
    稀疏权重: 10.0
    MineCLIP权重: 10.0
    设备: mps
    正在加载 MineCLIP attn 模型...
    从 data/mineclip/attn.pth 加载权重...
    ✓ 权重加载成功
    状态: ✓ MineCLIP 模型已加载  ← 成功！
  ✓ 环境创建成功
```

#### TensorBoard 中会看到

```
info/mineclip_similarity  # 相似度曲线（0-1）
info/mineclip_reward      # MineCLIP 奖励（连续变化）
info/sparse_reward        # 稀疏奖励（0或1）
info/total_reward         # 总奖励
reward/mineclip_weight    # 权重变化曲线（动态权重）
```

---

## 课程学习策略

### 什么是课程学习？

课程学习（Curriculum Learning）是一种从简单到复杂的训练策略，类似于人类学习过程：
- **初期**：高MineCLIP权重，强引导 → agent学会基本探索（如"找到树"）
- **中期**：权重逐渐降低 → agent开始依赖自身策略
- **后期**：低MineCLIP权重 → agent主要依赖稀疏奖励完成任务（如"砍树获得木头"）

### 为什么需要动态权重？

#### 固定权重的问题

```python
# ❌ 固定低权重（0.1）- 早期探索困难
mineclip_weight = 0.1  # agent不知道去哪里找树

# ❌ 固定高权重（1.0）- 后期依赖过度
mineclip_weight = 1.0  # agent只会"看树"，不会完成任务
```

#### 动态权重的优势

```python
# ✅ 动态调整
初始: mineclip_weight = 10.0   # 强引导，帮助探索
↓
中期: mineclip_weight = 5.0    # 逐渐独立
↓
最终: mineclip_weight = 0.1    # 主要靠稀疏奖励
```

### 权重衰减策略

当前实现使用**余弦衰减**：

```
权重
  ↑
10.0|‾‾‾╲
    |     ╲
 5.0|      ╲___
    |           ╲___
 0.1|________________╲_____
    └────────────────────→ 步数
    0     25k    50k    75k
```

**数学公式**：
```python
progress = min(step / decay_steps, 1.0)
decay_factor = 0.5 * (1.0 + cos(π * progress))
weight = min_weight + (initial_weight - min_weight) * decay_factor
```

### 使用方法

#### 方法1: 训练脚本参数（推荐）

```bash
python src/training/train_get_wood.py \
    --use-mineclip \
    --sparse-weight 10.0 \           # 稀疏奖励权重（固定）
    --mineclip-weight 10.0 \         # MineCLIP初始权重（与稀疏权重相同）
    --use-dynamic-weight \           # 启用动态调整
    --weight-decay-steps 50000 \     # 衰减步数
    --min-weight 0.1                 # 最小权重
```

#### 方法2: 使用Shell脚本

编辑 `scripts/train_get_wood.sh`，已经配置好默认参数：

```bash
./scripts/train_get_wood.sh mineclip
```

#### 方法3: 禁用动态权重（使用固定权重）

```bash
python src/training/train_get_wood.py \
    --use-mineclip \
    --mineclip-weight 0.1 \
    --no-dynamic-weight              # 禁用，使用固定权重0.1
```

### 参数调优指南

#### 1. 初始权重 `--mineclip-weight`

| 值 | 与sparse_weight比例 | 适用场景 | 说明 |
|----|-------------------|---------|------|
| 5.0 | 1:2 | 简单任务 | MineCLIP辅助引导 |
| **10.0** | **1:1（推荐）** | **中等任务** | **MineCLIP与稀疏奖励同等重要** |
| 20.0 | 2:1 | 困难任务 | MineCLIP主导，强引导 |

**原则**：初始阶段MineCLIP应该有足够的权重来引导agent探索

#### 2. 衰减步数 `--weight-decay-steps`

| 值 | 总训练步数 | 说明 |
|----|-----------|------|
| 30000 | 100k | 快速降低权重 |
| **50000** | **200k（推荐）** | **平衡衰减** |
| 100000 | 500k | 缓慢衰减 |

**经验公式**：`decay_steps = total_steps * 0.25`

#### 3. 最小权重 `--min-weight`

| 值 | 与初始权重比例 | 说明 |
|----|--------------|------|
| 0.01 | 0.1% | 几乎完全移除MineCLIP |
| **0.1（推荐）** | **1%** | **保持微弱引导信号** |
| 1.0 | 10% | 保持较强引导 |

**原则**：最小权重应该是初始权重的1%-10%，让agent在后期主要依赖稀疏奖励

#### 4. 稀疏权重 `--sparse-weight`

| 值 | 说明 |
|----|------|
| 5.0 | MineCLIP主导 |
| **10.0（推荐）** | 平衡 |
| 20.0 | 稀疏奖励主导 |

### 完整示例

#### 示例1: 标准训练（200k步）

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

#### 示例2: 快速测试（10k步）

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

#### 示例3: 困难任务（需要更强引导）

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

---

## 实时监控和调试

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

**关键指标**：
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

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard
# 浏览器访问: http://localhost:6006
```

### 典型训练曲线

#### 健康的训练过程

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
10.0|‾‾‾╲
  |     ╲___
0.1|_________╲_________→ 步数
  0    50k   100k  150k
```

**说明**：
1. 相似度快速上升（找到目标）
2. 权重逐渐下降
3. 奖励持续增长（完成任务）

#### 需要调整的情况

**问题1: 相似度不增长**

```
相似度
  ↑
  |＿＿＿＿＿＿＿＿＿
  |___________________→ 步数
```

**解决方案**：
- 增加初始权重：`--mineclip-weight 20.0`
- 延长衰减步数：`--weight-decay-steps 100000`

**问题2: 奖励在后期不增长**

```
奖励
  ↑
  |  ／￣￣￣￣￣
  |／___________________→ 步数
```

**解决方案**：
- 降低最小权重：`--min-weight 0.01`
- 增加稀疏权重：`--sparse-weight 20.0`

### 调试技巧

#### 1. 观察权重变化

查看日志中的"MC权重"列，应该看到平滑下降。

#### 2. 对比固定vs动态

```bash
# 运行1: 固定权重
python src/training/train_get_wood.py \
    --use-mineclip --mineclip-weight 0.1 --no-dynamic-weight \
    --tensorboard-dir logs/tensorboard/fixed

# 运行2: 动态权重
python src/training/train_get_wood.py \
    --use-mineclip --mineclip-weight 10.0 --use-dynamic-weight \
    --tensorboard-dir logs/tensorboard/dynamic
```

在TensorBoard中对比两条曲线。

#### 3. 检查相似度

如果相似度始终很低（<0.3），可能：
- MineCLIP模型未正确加载
- 任务描述不匹配
- 图像预处理有问题

**调试代码**：
```python
# 在训练开始前测试一次
env = create_harvest_log_env(use_mineclip=True)
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
print(f"MineCLIP相似度: {info.get('mineclip_similarity', 0)}")
print(f"MineCLIP奖励: {info.get('mineclip_reward', 0)}")
```

### 无头模式控制

#### 启用无头模式（默认，推荐用于训练）

```bash
python src/training/train_get_wood.py --headless
```

#### 禁用无头模式（用于调试，可以看到游戏画面）

```bash
python src/training/train_get_wood.py --no-headless
```

**注意**：
- 无头模式可以提升训练速度（无渲染开销）
- 调试时使用有头模式可以观察agent行为
- macOS可能需要额外配置才能使用无头模式

---

## 常见问题

### Q1: 没有模型权重文件

**错误**：
```
⚠️ 未指定模型路径，使用随机初始化（性能会很差）
```

**解决**：
1. 从 MineCLIP GitHub 下载预训练权重
2. 放到 `data/mineclip/` 目录
3. 修改代码指定路径

### Q2: 内存不足（MPS OOM）

**错误**：
```
RuntimeError: MPS backend out of memory
```

**解决**：
```python
# 方案1：使用 avg 变体（更小）
variant="avg",
model_path="data/mineclip/avg.pth"

# 方案2：使用 CPU
device="cpu"

# 方案3：减小 batch size（在 PPO 配置中）
batch_size=32  # 从64降到32
```

### Q3: 相似度一直很低

**可能原因**：
- 图像预处理有问题
- 模型权重损坏
- 任务描述不够准确

**调试**：
```python
# 在训练开始前测试一次
env = create_harvest_log_env(use_mineclip=True)
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
print(f"MineCLIP相似度: {info.get('mineclip_similarity', 0)}")
print(f"MineCLIP奖励: {info.get('mineclip_reward', 0)}")
```

### Q4: 什么时候不需要动态权重？

**A**: 
- 任务非常简单（几千步就能完成）
- 已经找到了最优的固定权重
- 纯探索任务（无明确稀疏奖励）

### Q5: 权重降到最小值后会继续变化吗？

**A**: 不会，达到`min_weight`后会保持不变。

### Q6: 可以在训练中途改变权重策略吗？

**A**: 可以，通过加载检查点并修改参数继续训练：
```bash
# 第一阶段：高权重探索
python train_get_wood.py --mineclip-weight 20.0 --total-timesteps 50000

# 第二阶段：从检查点继续，降低权重
python train_get_wood.py --load-checkpoint xxx.zip --mineclip-weight 0.1
```

### Q7: 如何知道权重衰减是否太快/太慢？

**A**: 观察奖励曲线：
- 太快：奖励在中期停止增长
- 太慢：后期仍然高度依赖MineCLIP
- 合适：奖励持续平稳增长

---

## 高级用法：自定义训练 MineCLIP

**你可以自己收集视频训练吗？可以！但是...**

### 需要的资源

1. **数据**：
   - 大量 Minecraft 游戏视频（几千小时）
   - 每个视频的文本描述/字幕
   - 存储空间：~1TB

2. **计算**：
   - 多GPU训练（4-8个 A100）
   - 训练时间：数天到数周
   - 云GPU成本：数千美元

3. **技术**：
   - 视频处理（FFmpeg）
   - 分布式训练（PyTorch DDP）
   - 数据标注

### 适用场景

✅ **值得自己训练**：
- 你有特定领域的任务（如红石电路、建筑）
- 官方模型在你的任务上表现不好
- 你有充足的资源和时间

❌ **不建议自己训练**：
- 只是想训练一个"砍树"智能体
- 预训练模型已经够用
- 资源有限

**结论**：对于大多数任务，**直接用预训练权重就够了**！

---

## 总结

### 推荐默认配置

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

### 关键文件

```
aimc/
  data/
    mineclip/
      attn.pth              ← 下载这个（推荐）
      avg.pth               ← 或这个（备选）
  src/
    utils/
      mineclip_reward.py    ← 官方包装器
    training/
      train_get_wood.py     ← 使用MineCLIP
```

### 快速开始

```bash
# 1. 下载模型权重
# 从 MineCLIP GitHub 获取 attn.pth

# 2. 放置到项目目录
mv attn.pth data/mineclip/

# 3. 开始训练
python src/training/train_get_wood.py \
    --use-mineclip \
    --mineclip-model data/mineclip/attn.pth \
    --use-dynamic-weight \
    --total-timesteps 200000
```

### 相关文档

- [训练加速完整指南](TRAINING_ACCELERATION_GUIDE.md)
- [TensorBoard 中文指南](TENSORBOARD_中文指南.md)
- [任务快速开始](TASKS_QUICK_START.md)

---

希望这个综合指南能帮助你充分利用 MineCLIP 提升训练效果！🚀

