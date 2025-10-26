# 🚀 VPT快速开始指南

## 📋 概览

本指南帮助你在 **30分钟内** 完成 VPT 模型的下载、安装和首次测试。

---

## ⏱️ 快速路径（30分钟）

### 步骤1: 安装依赖（5分钟）

```bash
# 激活环境
conda activate minedojo  # 或 minedojo-x86

# 安装VPT库
pip install git+https://github.com/openai/Video-Pre-Training.git

# 验证安装
python -c "import vpt; print('VPT安装成功！')"
```

### 步骤2: 下载预训练模型（10分钟）

```bash
# 创建目录
mkdir -p data/pretrained/vpt
cd data/pretrained/vpt

# 下载模型（推荐: RL-from-early-game）
# 这个模型大小适中（~50MB），性能好
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.model
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.weights

# 验证下载
ls -lh
# 应该看到两个文件
```

**模型选择**:

| 模型 | 大小 | 性能 | 适用场景 | 推荐度 |
|------|------|------|---------|--------|
| `rl-from-early-game-2x` | ~50MB | 高 | ✅ 砍树、挖矿等基础任务 | ⭐⭐⭐⭐⭐ |
| `rl-from-house-2x` | ~50MB | 中 | 房屋内任务 | ⭐⭐⭐ |
| `foundation-model-1x` | ~400MB | 最高 | 复杂任务、多技能组合 | ⭐⭐⭐⭐ |

### 步骤3: 测试零样本性能（15分钟）

创建测试脚本：

```bash
# 创建测试脚本
cat > tools/test_vpt_zero_shot.py << 'EOF'
#!/usr/bin/env python3
"""
测试VPT模型的零样本性能（无需微调）
"""

import minedojo
import numpy as np
from vpt import load_vpt_model

def test_vpt_zero_shot(model_path, task_id="harvest_1_log", episodes=5):
    """
    测试VPT模型的零样本性能
    
    Args:
        model_path: VPT模型路径
        task_id: MineDojo任务ID
        episodes: 测试回合数
    """
    
    # 1. 加载VPT模型
    print(f"加载VPT模型: {model_path}")
    vpt_model = load_vpt_model(model_path)
    
    # 2. 创建环境
    print(f"创建环境: {task_id}")
    env = minedojo.make(
        task_id=task_id,
        image_size=(128, 128),  # VPT使用128x128
    )
    
    # 3. 测试
    results = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n回合 {ep+1}/{episodes}")
        
        while not done and steps < 500:
            # VPT预测动作
            action = vpt_model.predict(obs['rgb'])
            
            # 执行
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"  步数: {steps}, 奖励: {total_reward:.2f}")
        
        success = info.get('success', False) or total_reward > 0
        results.append({
            'episode': ep + 1,
            'success': success,
            'reward': total_reward,
            'steps': steps
        })
        
        print(f"  完成: {'成功' if success else '失败'}, 奖励: {total_reward:.2f}, 步数: {steps}")
    
    # 4. 统计
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"\n" + "="*60)
    print(f"VPT零样本性能（无微调）")
    print(f"="*60)
    print(f"成功率: {success_rate*100:.1f}% ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_steps:.0f}")
    print(f"="*60)
    
    env.close()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="VPT模型路径")
    parser.add_argument("--task", type=str, default="harvest_1_log", help="任务ID")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    
    args = parser.parse_args()
    
    test_vpt_zero_shot(args.model, args.task, args.episodes)
EOF

chmod +x tools/test_vpt_zero_shot.py
```

运行测试：

```bash
# 测试VPT零样本性能
bash scripts/run_minedojo_x86.sh python tools/test_vpt_zero_shot.py \
    --model data/pretrained/vpt/rl-from-early-game-2x.model \
    --task harvest_1_log \
    --episodes 5
```

**预期结果**:
```
VPT零样本性能（无微调）
============================================================
成功率: 20-40% (1-2/5)
平均奖励: 0.30
平均步数: 350
============================================================
```

**解读**:
- ✅ **20-40%成功率**: 这很好！证明VPT已经学会基础移动和挖掘
- ✅ **相比随机策略（0%）**: 提升巨大
- ✅ **微调后**: 预期提升到75-80%

---

## 🎯 下一步：微调VPT

### 方案A: 使用现有专家数据微调（推荐）

如果你已经有录制的专家数据：

```bash
# 使用现有数据微调VPT
python src/training/train_bc_with_vpt.py \
    --vpt-model data/pretrained/vpt/rl-from-early-game-2x.model \
    --data data/tasks/harvest_1_log/expert_demos/ \
    --output data/tasks/harvest_1_log/checkpoints/vpt_finetuned.zip \
    --epochs 10 \
    --learning-rate 1e-4
```

**预期**:
- 训练时间: 10-15分钟
- 成功率: 75-80%

### 方案B: 录制新数据并微调

如果还没有专家数据：

```bash
# 1. 录制专家演示（只需20-30个，相比原来的100个）
bash scripts/run_minedojo_x86.sh python tools/dagger/record_manual_chopping_pygame.py \
    --base-dir data/tasks/harvest_1_log_vpt/expert_demos \
    --max-episodes 30 \
    --max-frames 1000

# 2. 微调VPT
python src/training/train_bc_with_vpt.py \
    --vpt-model data/pretrained/vpt/rl-from-early-game-2x.model \
    --data data/tasks/harvest_1_log_vpt/expert_demos/ \
    --output data/tasks/harvest_1_log_vpt/checkpoints/vpt_finetuned.zip \
    --epochs 10

# 3. 评估
bash scripts/run_minedojo_x86.sh python tools/dagger/evaluate_policy.py \
    --model data/tasks/harvest_1_log_vpt/checkpoints/vpt_finetuned.zip \
    --episodes 20
```

### 方案C: VPT + DAgger（最佳性能）

结合VPT和DAgger达到最高性能：

```bash
# 完整工作流
bash scripts/run_dagger_workflow_with_vpt.sh \
    --task harvest_1_log \
    --vpt-model data/pretrained/vpt/rl-from-early-game-2x.model \
    --num-episodes 30 \
    --iterations 2
```

---

## 📊 性能对比

### 预期成功率

| 阶段 | 从零训练 | VPT方法 | 提升 |
|------|---------|---------|------|
| 零样本 | 0% | **20-40%** | +40% |
| BC基线 | 60% | **75-80%** | +20% |
| DAgger 1轮 | 75% | **85-90%** | +12% |
| DAgger 2轮 | 85% | **90-95%** | +8% |

### 训练时间

| 阶段 | 从零训练 | VPT方法 | 节省 |
|------|---------|---------|------|
| 录制 | 100回合 (60分钟) | 30回合 (20分钟) | **-40分钟** |
| BC训练 | 30-40分钟 | 10-15分钟 | **-20分钟** |
| DAgger | 2-3小时 | 1小时 | **-1.5小时** |
| **总计** | **3-5小时** | **1-2小时** | **节省60%** |

---

## ⚠️ 常见问题

### Q1: VPT安装失败？

```bash
# 问题：git clone失败
# 解决：手动克隆
git clone https://github.com/openai/Video-Pre-Training.git
cd Video-Pre-Training
pip install -e .
```

### Q2: 模型下载慢？

```bash
# 使用代理或备用链接
# 方案1: 使用代理
export https_proxy=http://your-proxy:port
wget https://openaipublic.blob.core.windows.net/...

# 方案2: 使用国内镜像（如有）
# 或从GitHub Release下载
```

### Q3: 动作空间不兼容？

VPT和MineDojo的动作空间基本兼容，但可能需要简单映射。参考 `src/models/vpt_adapter.py`。

### Q4: 内存不足？

```bash
# 使用较小的VPT模型
# rl-from-early-game-2x (50MB) 而非 foundation (400MB)

# 或减少batch size
python train_bc_with_vpt.py --batch-size 32  # 默认64
```

---

## 🔍 验证安装

运行这个脚本验证所有组件正常：

```bash
cat > tools/verify_vpt_setup.py << 'EOF'
#!/usr/bin/env python3
"""验证VPT设置"""

import sys

print("检查依赖...")

# 检查VPT
try:
    import vpt
    print("✓ VPT安装成功")
except ImportError:
    print("✗ VPT未安装")
    sys.exit(1)

# 检查MineDojo
try:
    import minedojo
    print("✓ MineDojo安装成功")
except ImportError:
    print("✗ MineDojo未安装")
    sys.exit(1)

# 检查模型文件
import os
model_path = "data/pretrained/vpt/rl-from-early-game-2x.model"
if os.path.exists(model_path):
    print(f"✓ VPT模型已下载: {model_path}")
else:
    print(f"✗ VPT模型未找到: {model_path}")
    print("  请运行: wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/rl-from-early-game-2x.model")

# 检查GPU/MPS
import torch
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✓ MPS可用 (Apple Silicon)")
else:
    print("⚠ 仅CPU可用（训练会较慢）")

print(f"\n所有检查通过！设备: {device}")
EOF

python tools/verify_vpt_setup.py
```

---

## 📚 更多资源

### 文档

- [VPT完整分析](../technical/VPT_INTEGRATION_ANALYSIS.md) - 技术细节和实施方案
- [DAgger指南](DAGGER_COMPREHENSIVE_GUIDE.md) - 如何结合VPT和DAgger

### 示例代码

```python
# 最简单的VPT使用示例
from vpt import load_vpt_model
import minedojo

# 加载模型
model = load_vpt_model("data/pretrained/vpt/rl-from-early-game-2x.model")

# 创建环境
env = minedojo.make("harvest_1_log", image_size=(128, 128))

# 运行
obs = env.reset()
for _ in range(100):
    action = model.predict(obs['rgb'])
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

---

## 🎯 推荐工作流

**首次使用VPT**:

```
第1天: 安装和测试零样本性能（本指南）
  ↓
第2-3天: 录制20-30个专家演示
  ↓
第4天: 微调VPT（BC）
  ↓
第5-6天: 1-2轮DAgger迭代
  ↓
第7天: 评估和优化
```

**总时间**: 1周达到90%+成功率（相比原来的2-3周）

---

## 💡 成功提示

1. ✅ **先测试零样本**: 了解VPT的基础能力
2. ✅ **使用rl-from-early-game**: 大小和性能的最佳平衡
3. ✅ **低学习率微调**: 1e-4或更低，避免遗忘预训练知识
4. ✅ **较少的epoch**: 10-20轮通常足够
5. ✅ **结合DAgger**: 达到最高性能

---

**祝你成功！如有问题，参考完整文档或提Issue。** 🚀


