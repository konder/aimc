# VPT零样本评估快速开始

本指南介绍如何快速运行VPT零样本评估，测试预训练VPT模型在harvest_log任务上的表现。

## 前置条件

1. **VPT权重文件已下载**
   ```bash
   ls data/pretrained/vpt/rl-from-early-game-2x.weights
   ```
   
   如果文件不存在，请下载：
   ```bash
   mkdir -p data/pretrained/vpt
   cd data/pretrained/vpt
   wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.weights -O rl-from-early-game-2x.weights
   ```

2. **MineDojo环境已安装**
   ```bash
   conda activate minedojo
   python -c "import minedojo; print('✓ MineDojo OK')"
   ```

## 快速启动

### 方法1: 使用启动脚本（推荐）

```bash
# 默认设置：评估10轮，自动检测设备
bash scripts/evaluate_vpt_zero_shot.sh

# 自定义评估轮数
bash scripts/evaluate_vpt_zero_shot.sh 20

# 指定设备
bash scripts/evaluate_vpt_zero_shot.sh 10 cpu     # CPU
bash scripts/evaluate_vpt_zero_shot.sh 10 cuda    # CUDA
bash scripts/evaluate_vpt_zero_shot.sh 10 mps     # MPS (Mac M1/M2)

# 完整参数: [轮数] [设备] [最大步数]
bash scripts/evaluate_vpt_zero_shot.sh 20 mps 1500
```

### 方法2: 直接运行Python脚本

```bash
# 评估完整版VPT Agent
python src/training/vpt/evaluate_vpt_zero_shot.py \
    --agent complete \
    --episodes 10 \
    --max_steps 1200 \
    --device auto

# 查看帮助信息
python src/training/vpt/evaluate_vpt_zero_shot.py --help
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | `complete` | Agent类型（目前只有complete） |
| `--episodes` | `10` | 评估轮数 |
| `--max_steps` | `1200` | 每轮最大步数 |
| `--device` | `auto` | 设备：cpu/cuda/mps/auto |
| `--weights` | `data/pretrained/vpt/rl-from-early-game-2x.weights` | VPT权重路径 |

## 输出示例

```
======================================================================
🎯 零样本评估VPT Agent
======================================================================
任务: harvest_1_log
评估轮数: 10
最大步数: 1200
设备: mps
======================================================================

✓ 环境创建成功

======================================================================
📊 评估完整版VPT Agent (pi_head + hidden state)
======================================================================

📍 Episode 1/10
  Step 100/1200, Reward=0.00
  Step 200/1200, Reward=0.00
  ...
  ✅ 成功！第436步获得奖励: 1.0
  ✅ 成功 - 步数: 436, 累积奖励: 1.00

📍 Episode 2/10
  ...

----------------------------------------------------------------------
📈 完整版统计结果:
----------------------------------------------------------------------
成功率: 30.0% (3/10)
平均奖励: 0.300
平均步数: 978.5
----------------------------------------------------------------------

✅ 评估完成！
```

## VPT Agent特性

当前的`VPTAgent`（完整版）包含以下关键特性：

### ✅ 已实现

1. **Hidden State维护**
   - Transformer memory (256 steps)
   - Episode边界处理（first标志）
   - 跨时间步的连续性

2. **完整VPT Forward**
   - `policy.act()` - 官方forward路径
   - Pi head输出（智能决策）
   - Value head（可选）

3. **官方动作转换**
   - `CameraHierarchicalMapping` - 11-bin camera
   - `ActionTransformer` - mu-law quantization
   - MineRL → MineDojo动作映射

4. **设备自适应**
   - `device='auto'`: cuda > mps > cpu
   - 混合精度支持（FP16/BF16）
   - 批处理推理优化

### 🔧 可配置

- **冲突策略**: `priority` (默认) 或 `cancel`
- **详细输出**: `verbose=True` 查看详细日志
- **设备选择**: cpu/cuda/mps/auto

## 性能基准

在harvest_1_log任务上的预期性能：

| 指标 | 预期范围 | 说明 |
|------|----------|------|
| 成功率 | 10-30% | 零样本，无任务特定训练 |
| 平均步数 | 800-1000 | 成功episodes的平均步数 |
| 平均奖励 | 0.1-0.3 | 每episode平均（0或1） |

**注意**：这是零样本性能基线。通过BC fine-tuning可以显著提升。

## 常见问题

### 1. 显存不足（OOM）

```bash
# 使用CPU评估
bash scripts/evaluate_vpt_zero_shot.sh 10 cpu

# 或减少batch size（需修改代码）
```

### 2. 评估速度慢

```bash
# 减少评估轮数
bash scripts/evaluate_vpt_zero_shot.sh 5

# 减少最大步数
bash scripts/evaluate_vpt_zero_shot.sh 10 auto 600
```

### 3. 权重文件损坏

```bash
# 重新下载权重
rm data/pretrained/vpt/rl-from-early-game-2x.weights
cd data/pretrained/vpt
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/2x.weights -O rl-from-early-game-2x.weights
```

### 4. MineDojo环境问题

```bash
# 确保使用正确的环境
conda activate minedojo

# 在Mac M1/M2上需要x86模式
bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_vpt_zero_shot.py --episodes 5
```

## 下一步

评估完成后，可以进行：

1. **BC Fine-tuning**
   ```bash
   bash scripts/train_vpt_bc.sh
   ```

2. **对比fine-tune前后性能**
   ```bash
   # 评估fine-tuned模型
   python src/training/vpt/evaluate_bc_vpt.py \
       --checkpoint checkpoints/vpt_bc/best_model.pth \
       --episodes 20
   ```

3. **DAgger训练**
   ```bash
   bash scripts/run_dagger_workflow.sh
   ```

## 相关文档

- [VPT完整实现文档](../summaries/VPT_AGENT_COMPLETE_IMPLEMENTATION.md)
- [VPT BC训练指南](./VPT_BC_TRAINING_GUIDE.md)
- [DAgger综合指南](./DAGGER_COMPREHENSIVE_GUIDE.md)

## 技术支持

如有问题，请检查：
1. logs/目录下的日志文件
2. GitHub Issues
3. 项目README和FAQ

