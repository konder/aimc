# harvest_1_paper 训练指南

## 概述

这是一个 MVP（最小可行产品）级别的 MineDojo harvest_1_paper 任务训练系统。使用 PPO 算法从头开始训练智能体学习如何在 Minecraft 中收集纸。

**重要提醒**: MineDojo 内置任务**不提供预训练模型**，所有训练都是从随机初始化开始的。

---

## 快速开始

### 1. 安装依赖

```bash
# 激活 MineDojo 环境
conda activate minedojo

# 安装训练依赖
pip install -r requirements.txt
```

### 2. 快速测试（验证环境）

```bash
# 运行快速测试（10K步，约5-10分钟）
./scripts/train_harvest.sh test
```

### 3. 完整训练

```bash
# 标准训练（500K步，约2-4小时）
./scripts/train_harvest.sh

# 长时间训练（2M步，约8-16小时）
./scripts/train_harvest.sh long
```

### 4. 评估模型

```bash
# 评估最佳模型
./scripts/eval_harvest.sh

# 评估特定检查点
./scripts/eval_harvest.sh checkpoints/harvest_paper/harvest_paper_100000_steps.zip
```

---

## 文件结构

```
aimc/
├── src/
│   ├── utils/
│   │   └── env_wrappers.py          # 环境包装器
│   ├── training/
│   │   └── train_harvest_paper.py   # 训练脚本
│   └── demo_harvest_task.py         # 任务演示脚本
├── scripts/
│   ├── train_harvest.sh             # 训练启动脚本
│   └── eval_harvest.sh              # 评估脚本
├── config/
│   └── training_config.yaml         # 训练配置文件
├── checkpoints/
│   └── harvest_paper/               # 模型检查点保存位置
└── logs/
    ├── training/                    # 训练日志
    └── tensorboard/                 # TensorBoard日志
```

---

## 详细使用说明

### 环境包装器

`src/envs/env_wrappers.py` 提供了三个包装器：

#### 1. MinedojoWrapper
将 MineDojo 的复杂观察空间简化为 RGB 图像：
- 提取 RGB 图像
- 归一化到 [0, 1]
- 转换为 (C, H, W) 格式

#### 2. ActionWrapper
将复杂的动作空间映射到离散动作：
- 无操作
- 移动（前后左右）
- 视角控制（上下左右）
- 跳跃、攻击、使用、潜行
- 组合动作（前进+攻击、前进+跳跃）

#### 3. FrameStack
堆叠连续多帧，帮助模型学习时序信息（当前 MVP 版本未使用）。

### 训练脚本参数

`train_harvest_paper.py` 支持丰富的命令行参数：

```bash
python src/training/train_harvest_paper.py \
    --mode train \                    # 模式: train/eval
    --task-id harvest_milk \          # 任务ID
    --total-timesteps 500000 \        # 总训练步数
    --n-envs 1 \                      # 并行环境数
    --learning-rate 0.0003 \          # 学习率
    --device auto \                   # 设备: auto/cpu/cuda
    --save-freq 10000 \               # 保存频率
    --eval-freq 10000 \               # 评估频率
    --checkpoint-dir checkpoints/harvest_paper \
    --tensorboard-dir logs/tensorboard
```

完整参数列表：
```bash
python src/training/train_harvest_paper.py --help
```

---

## 训练配置

### 默认配置（标准训练）

```yaml
total_timesteps: 500000    # 总步数
n_envs: 1                  # 单个环境
learning_rate: 0.0003      # 学习率
n_steps: 2048              # 每次更新步数
batch_size: 64             # 批次大小
device: auto               # 自动选择设备
```

### 快速测试配置

```yaml
total_timesteps: 10000     # 快速验证
save_freq: 5000
eval_freq: 5000
```

### 高性能配置

```yaml
total_timesteps: 2000000   # 更长训练
n_envs: 4                  # 4个并行环境
batch_size: 128            # 更大批次
```

修改 `config/training_config.yaml` 可以调整所有参数。

---

## 监控训练

### 1. 查看训练日志

```bash
# 实时查看日志
tail -f logs/training/training_*.log

# 查看所有日志
cat logs/training/training_*.log
```

### 2. TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir logs/tensorboard

# 然后在浏览器打开: http://localhost:6006
```

TensorBoard 显示：
- 📈 Episode 奖励曲线
- 📊 策略损失、价值损失
- 🎯 评估指标
- 🔍 梯度和参数分布

### 3. 关键指标

| 指标 | 说明 | 期望值 |
|------|------|--------|
| `rollout/ep_rew_mean` | 平均 episode 奖励 | 逐渐增加 |
| `train/policy_loss` | 策略损失 | 稳定或缓慢下降 |
| `train/value_loss` | 价值损失 | 稳定或缓慢下降 |
| `train/entropy_loss` | 熵损失 | 逐渐减小（探索→利用） |
| `eval/mean_reward` | 评估平均奖励 | 逐渐增加 |

---

## 模型评估

### 评估命令

```bash
# 评估最佳模型（默认）
./scripts/eval_harvest.sh

# 评估特定检查点
./scripts/eval_harvest.sh checkpoints/harvest_paper/harvest_paper_50000_steps.zip

# Python命令评估
python src/training/train_harvest_paper.py \
    --mode eval \
    --model-path checkpoints/harvest_paper/best_model.zip \
    --n-eval-episodes 10
```

### 评估输出

```
Episode 1/10: reward=0.00, steps=842
Episode 2/10: reward=1.00, steps=654
Episode 3/10: reward=0.00, steps=1203
...
平均奖励: 0.40 ± 0.49
平均步数: 899.3 ± 210.5
成功率: 4/10 (40.0%)
```

---

## 常见问题

### Q1: harvest_1_paper 任务ID无效？

**A**: `harvest_1_paper` 在某些 MineDojo 版本中不可用。解决方案：

```bash
# 方案1: 使用 harvest_milk（默认，更稳定）
./scripts/train_harvest.sh

# 方案2: 尝试其他harvest任务
python src/training/train_harvest_paper.py --task-id harvest_wool
```

### Q2: 训练太慢怎么办？

**A**: 加速方法：

1. **使用 GPU**：
   ```bash
   # 确保安装CUDA版本的PyTorch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **增加并行环境**：
   ```bash
   python src/training/train_harvest_paper.py --n-envs 4
   ```

3. **减少图像尺寸**：
   ```bash
   python src/training/train_harvest_paper.py --image-size 120 160
   ```

### Q3: 模型不学习/奖励始终为0？

**A**: 常见原因：

1. **任务太难**：先用简单任务（如 harvest_milk）测试
2. **训练时间不足**：harvest 任务通常需要 100K-500K 步
3. **探索不足**：增加熵系数
   ```bash
   python src/training/train_harvest_paper.py --ent-coef 0.02
   ```
4. **学习率问题**：尝试调整学习率
   ```bash
   python src/training/train_harvest_paper.py --learning-rate 0.0001
   ```

### Q4: 内存不足？

**A**: 减少内存使用：

```bash
# 减少并行环境
python src/training/train_harvest_paper.py --n-envs 1

# 减少批次大小
python src/training/train_harvest_paper.py --batch-size 32

# 减少图像尺寸
python src/training/train_harvest_paper.py --image-size 120 160
```

### Q5: 训练中断后如何恢复？

**A**: 从检查点恢复：

```python
from stable_baselines3 import PPO

# 加载检查点
model = PPO.load("checkpoints/harvest_paper/harvest_paper_50000_steps.zip")

# 继续训练
model.learn(total_timesteps=500000)
```

---

## 超参数调优建议

### 学习率 (learning_rate)

| 值 | 适用场景 |
|----|----------|
| 0.0003 | 默认，适合大多数情况 |
| 0.0001 | 训练不稳定时 |
| 0.001 | 想要快速收敛时（风险更高） |

### 熵系数 (ent_coef)

| 值 | 效果 |
|----|------|
| 0.01 | 默认，平衡探索和利用 |
| 0.02-0.05 | 增加探索，适合稀疏奖励任务 |
| 0.001 | 减少探索，适合已有好策略时 |

### 并行环境数 (n_envs)

| 值 | 说明 |
|----|------|
| 1 | 内存有限，单机训练 |
| 4-8 | 推荐，显著加速训练 |
| 16+ | 分布式训练，需要大量资源 |

---

## 性能基准

### 训练时间估算（单环境）

| 步数 | CPU | GPU | 说明 |
|------|-----|-----|------|
| 10K | 5-10min | 2-5min | 快速测试 |
| 100K | 1-2h | 0.5-1h | 初步训练 |
| 500K | 4-8h | 2-4h | 标准训练 |
| 2M | 16-32h | 8-16h | 完整训练 |

### 硬件需求

| 配置 | CPU | 内存 | GPU |
|------|-----|------|-----|
| 最低 | 4核 | 8GB | 无 |
| 推荐 | 8核+ | 16GB | GTX 1060+ |
| 高性能 | 16核+ | 32GB+ | RTX 3070+ |

---

## 下一步优化

当前 MVP 实现后，可以考虑以下优化：

### 1. 使用帧堆叠
```python
env = make_minedojo_env(
    task_id="harvest_milk",
    use_frame_stack=True,
    frame_stack_n=4
)
```

### 2. 使用预训练视觉编码器
```python
# 使用 MineCLIP 等预训练模型
from mineclip import MineCLIP
encoder = MineCLIP.load_pretrained()
```

### 3. 课程学习
先训练简单任务，逐步增加难度：
1. harvest_milk (简单)
2. harvest_wool (中等)
3. harvest_1_paper (较难)

### 4. 奖励塑形
添加中间奖励，引导智能体：
- 发现甘蔗 → +0.1
- 采集甘蔗 → +0.3
- 制作纸 → +1.0

### 5. 使用更强的算法
- **IMPALA**: 分布式训练
- **PPG**: 改进的PPO变体
- **DreamerV3**: 基于世界模型

---

## 故障排除

### 训练脚本无法执行

```bash
# 添加执行权限
chmod +x scripts/train_harvest.sh scripts/eval_harvest.sh
```

### ModuleNotFoundError

```bash
# 确保在项目根目录
cd /Users/nanzhang/aimc

# 设置 PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# 或使用绝对路径运行
python /Users/nanzhang/aimc/src/training/train_harvest_paper.py
```

### Java 相关错误

```bash
# 设置无头模式
export JAVA_OPTS="-Djava.awt.headless=true"

# 检查Java版本
java -version  # 需要 Java 8+
```

### MineDojo 环境创建失败

```bash
# 重新构建 MineDojo
cd /opt/conda/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
./gradlew shadowJar

# 或尝试重新安装
pip uninstall minedojo
pip install minedojo
```

---

## 参考资源

- [MineDojo 官方文档](https://docs.minedojo.org/)
- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- [MineDojo 论文](https://arxiv.org/abs/2206.08853)

---

## 总结

**MVP 系统特点**：
- ✅ 完整的训练流程（环境包装、模型训练、评估）
- ✅ 使用成熟的 PPO 算法和 Stable-Baselines3
- ✅ 丰富的监控和日志系统
- ✅ 灵活的配置和参数调整
- ✅ 详细的文档和故障排除指南

**关键点**：
- 🔄 从头训练，无预训练权重
- 📊 预期训练时间：数小时到数天
- 🎯 建议先用 harvest_milk 测试
- 💡 可根据需求调整超参数

**开始训练**：
```bash
./scripts/train_harvest.sh test  # 快速验证
./scripts/train_harvest.sh       # 完整训练
```

祝训练顺利！🚀

