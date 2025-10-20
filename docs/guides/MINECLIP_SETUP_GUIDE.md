# MineCLIP 设置和使用指南

## 📚 快速理解 MineCLIP

### 四个关键概念

| 概念 | 作用 | 你需要吗？ |
|------|------|-----------|
| **1. MineCLIP 模型** | 视觉-语言模型，计算画面与文本的相似度 | ✅ **需要** - 核心组件 |
| **2. 预训练权重** (attn.pth/avg.pth) | 模型参数，直接用于推理 | ✅ **需要** - 必须下载 |
| **3. MineAgent** | 策略网络示例（PPO等） | ❌ **不需要** - 你已经用PPO了 |
| **4. 640K视频数据** | 重新训练MineCLIP用的原始数据 | ❌ **不需要** - 研究用途 |

---

## 🚀 完整设置步骤

### 步骤1：安装 MineCLIP 包 ✅

你已经完成了！

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
            mineclip_weight=0.1
        )
    
    return env
```

### 步骤4：运行训练

```bash
bash-3.2$ scripts/train_get_wood.sh test --mineclip
```

---

## 🎯 attn vs avg：应该用哪个？

### attn（推荐）⭐⭐⭐

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

### avg（备选）⭐⭐

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

---

## 💡 MineCLIP 工作原理

### 1. 训练前（预训练阶段）

```
640K YouTube 视频 + 文本描述
         ↓
   MineCLIP 训练
         ↓
预训练权重 (attn.pth)  ← 你下载的就是这个
```

### 2. 你的训练中（推理阶段）

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

## 🔧 完整代码示例

### train_get_wood.py 修改

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
            mineclip_weight=0.1
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

---

## 📊 预期效果

### 使用 MineCLIP attn

```
[1/4] 创建环境...
创建环境: harvest_1_log (获得1个原木)
  图像尺寸: (160, 256)
  MineCLIP: 启用
  MineCLIP 奖励包装器:
    任务描述: chop down a tree and collect one wood log
    模型变体: attn
    稀疏权重: 10.0
    MineCLIP权重: 0.1
    设备: mps
    正在加载 MineCLIP attn 模型...
    从 data/mineclip/attn.pth 加载权重...
    ✓ 权重加载成功
    状态: ✓ MineCLIP 模型已加载  ← 成功！
  ✓ 环境创建成功
```

### TensorBoard 中会看到

```
info/mineclip_similarity  # 相似度曲线（0-1）
info/mineclip_reward      # MineCLIP 奖励（连续变化）
info/sparse_reward        # 稀疏奖励（0或1）
info/total_reward         # 总奖励
```

---

## 🐛 常见问题

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

---

## 🎓 高级用法：自定义训练 MineCLIP

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

**结论**：对于你当前的任务（砍树），**直接用预训练权重就够了**！

---

## 📝 总结

### 你需要做的

1. ✅ 安装 MineCLIP - **已完成**
2. ⏳ 下载 attn.pth - **需要完成**
3. ⏳ 修改 train_get_wood.py - **需要完成**
4. ⏳ 运行训练 - **即将完成**

### 关键文件

```
aimc/
  data/
    mineclip/
      attn.pth              ← 下载这个（推荐）
      avg.pth               ← 或这个（备选）
  src/
    utils/
      mineclip_reward.py    ← 已创建（官方包装器）
    training/
      train_get_wood.py     ← 需要修改（使用新wrapper）
```

### 下一步

1. **获取模型权重**：
   - 检查 MineCLIP GitHub 的 releases
   - 或者联系 MineDojo 团队

2. **修改训练脚本**：
   - 使用 `src/utils/mineclip_reward.py`
   - 指定模型路径

3. **开始训练**：
   ```bash
   scripts/train_get_wood.sh test --mineclip
   ```

---

希望这个指南解答了你的所有疑问！🚀

如果还有问题，随时问我！

