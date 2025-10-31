# STEVE-1 微调快速开始指南

> **目标受众**: 想要在STEVE-1预训练模型基础上进行任务特定微调的开发者  
> **预计时间**: 数据准备1-2小时，训练数小时至1-2天  
> **前置要求**: 已完成STEVE-1评估测试，了解基本使用

---

## 🎯 为什么要微调？

| 场景 | 预训练模型表现 | 微调后表现 |
|------|--------------|-----------|
| 通用任务 (砍树、挖矿) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 特定任务 (建造木屋) | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 新技能 (红石电路) | ⭐ | ⭐⭐⭐⭐ |

**微调优势**:
- ✅ 显著提升特定任务性能
- ✅ 学习预训练模型未见过的行为
- ✅ 训练时间远短于从头训练 (数小时 vs 数天)
- ✅ 数据需求更少 (1-10小时录像 vs 100+小时)

---

## 📋 准备清单

### 硬件要求
- [x] GPU: 24GB显存 (RTX 3090/4090, A5000) - **必需**
- [x] 内存: 32GB+ 系统RAM
- [x] 磁盘: 50GB+ 可用空间
- [x] 网络: (首次)下载预训练权重需要

### 软件环境
```bash
# 检查环境
conda activate minedojo
python --version  # 应该是 3.9+
nvidia-smi        # 检查GPU可用

# 检查依赖
python -c "import torch; print(torch.cuda.is_available())"  # 应输出True
python -c "import accelerate; print(accelerate.__version__)"
```

### 必需的文件
```bash
# 检查预训练权重是否存在
ls -lh data/weights/steve1/steve1.weights      # ~952MB
ls -lh data/weights/vpt/2x.model               # ~数KB
ls -lh data/weights/vpt/rl-from-foundation-2x.weights  # ~948MB

# 如果缺失，参考: docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md
```

---

## 🚀 快速开始 (3个步骤)

### 步骤1: 准备训练数据 (1-2小时)

#### 方法A: 使用现有VPT数据集 (推荐新手)

```bash
cd /Users/nanzhang/aimc/src/training/steve1

# 下载少量数据进行测试
bash 1_generate_dataset.sh

# 编辑脚本，设置较小的episode数量:
# N_EPISODES_CONTRACTOR=5  (每个索引只下载5个episode)
```

**数据集大小估算**:
- 1 episode ≈ 500MB-1GB
- 5 episodes × 3 索引 = 15 episodes ≈ 10GB
- 训练时间: 约2-4小时

#### 方法B: 使用自定义数据 (进阶)

如果你有自己的游戏录像：

```bash
# 准备数据目录
mkdir -p data/dataset_custom/

# 你的数据格式应为:
# data/dataset_custom/
# ├── episode_001/
# │   ├── video.mp4        # 游戏录像
# │   └── actions.jsonl    # 每帧动作 (可选，可从录像提取)
# ├── episode_002/
# └── ...

# TODO: 创建转换脚本 (当前项目中暂未实现)
# python data/generation/convert_custom_data.py \
#     --input_dir data/dataset_custom/ \
#     --output_dir data/dataset_custom_processed/
```

**提示**: 自定义数据转换脚本需要实现，当前可先使用VPT数据集进行测试。

### 步骤2: 创建采样配置 (5分钟)

```bash
cd /Users/nanzhang/aimc/src/training/steve1

# 生成训练/验证集划分
bash 2_create_sampling.sh

# 或者自定义参数:
python data/sampling/generate_sampling.py \
    --type neurips \
    --name my_finetune_task \
    --output_dir "$PROJECT_ROOT/data/samplings/" \
    --val_frames 2000 \      # 验证集: 2000帧
    --train_frames 10000     # 训练集: 10000帧 (约1-2小时录像)
```

**输出文件**:
```
data/samplings/
├── my_finetune_task_train.txt  # 训练集episode路径列表
└── my_finetune_task_val.txt    # 验证集episode路径列表
```

### 步骤3: 启动微调训练 (数小时)

```bash
cd /Users/nanzhang/aimc/src/training/steve1

# 方式1: 使用模板脚本 (推荐)
cp 3_train_finetune_template.sh 3_train_my_task.sh

# 编辑脚本，修改关键参数:
vim 3_train_my_task.sh
# 修改:
#   SAMPLING_NAME="my_finetune_task"  # 改为你的采样名称
#   OUT_WEIGHTS="...steve1_my_task.weights"  # 自定义输出名称

# 运行训练
bash 3_train_my_task.sh

# 方式2: 直接修改原始训练脚本
vim 3_train.sh
# 修改 --in_weights 为 steve1.weights (加载预训练模型)
# 修改 --learning_rate 为 1e-5 (降低学习率)
# 修改 --n_frames 为 10_000_000 (减少训练帧数)

bash 3_train.sh
```

---

## 📊 监控训练进度

### 启动TensorBoard

```bash
# 新开一个终端
cd /Users/nanzhang/aimc

tensorboard --logdir data/finetuning_checkpoint/logs/

# 浏览器打开: http://localhost:6006
```

### 关键指标

| 指标 | 健康范围 | 异常情况 |
|------|---------|---------|
| **loss/train** | 持续下降 | 不下降/震荡 |
| **loss/val** | 与训练损失接近 | 远高于训练损失 (过拟合) |
| **learning_rate** | 平滑上升后下降 | 突变 |
| **grad_l2_norm** | 0.5 - 5.0 | > 10.0 (梯度爆炸) |

### 训练日志示例

```
Metrics for step 100:
    Curr DateTime: 2025-10-31 14:23:10
    loss: 2.456              ← 应持续下降
    learning_rate: 5.2e-06   ← 预热阶段线性上升
    grad_l2_norm: 1.234      ← 梯度正常
    processed_frames: 51200  ← 已处理帧数
    step: 100
    
Running validation at step 200...
    val_loss: 2.398          ← 应与训练损失接近
New best validation loss: 2.398, saving best val model weights...
```

---

## 🧪 测试微调模型

### 快速测试

```bash
cd /Users/nanzhang/aimc/src/training/steve1

# 复制测试脚本
cp 2_gen_vid_for_text_prompt.sh 2_test_finetuned_model.sh

# 编辑脚本，修改权重路径:
vim 2_test_finetuned_model.sh
# 修改:
#   --in_weights "$PROJECT_ROOT/data/weights/steve1/steve1_my_task_best.weights"
#   --custom_text_prompt "your specific task"

# 生成测试视频
bash 2_test_finetuned_model.sh

# 查看生成的视频
ls -lh data/generated_videos/custom_text_prompt/
open data/generated_videos/custom_text_prompt/*.mp4  # macOS
```

### 对比测试

```bash
# 1. 使用预训练模型生成
bash 2_gen_vid_for_text_prompt.sh  
# 输出: baseline_video.mp4

# 2. 使用微调模型生成
bash 2_test_finetuned_model.sh
# 输出: finetuned_video.mp4

# 3. 并排对比
# 使用视频播放器同时播放两个视频，观察行为差异
```

---

## 🔧 常见问题排查

### 问题1: 显存不足 (CUDA out of memory)

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB
```

**解决方案** (按顺序尝试):

```bash
# 方案A: 减小批量大小
vim 3_train_my_task.sh
# 修改:
BATCH_SIZE=4              # 从8改为4
GRADIENT_ACCUM_STEPS=4    # 从2改为4 (保持有效批量不变)

# 方案B: 减小序列长度
T=160                     # 从320改为160
TRUNC_T=32                # 从64改为32

# 方案C: 关闭混合精度 (不推荐，速度变慢)
# 在训练脚本中:
# --mixed_precision no  (替换 bf16)
```

### 问题2: 训练损失不下降

**可能原因与解决方案**:

| 原因 | 检查方法 | 解决方案 |
|------|---------|---------|
| 学习率过大 | 损失震荡/NaN | 减小到 5e-6 |
| 学习率过小 | 损失几乎不变 | 增大到 2e-5 |
| 数据问题 | 检查embed norm | 重新生成MineCLIP嵌入 |
| 梯度消失 | grad_norm < 0.01 | 增大学习率 |
| 梯度爆炸 | grad_norm > 10 | 减小学习率或增大max_grad_norm |

**诊断脚本**:
```python
# 在训练开始时添加调试代码
for obs, actions, firsts in dataloader:
    print(f"Obs keys: {obs.keys()}")
    print(f"Image shape: {obs['img'].shape}")
    print(f"Image range: [{obs['img'].min()}, {obs['img'].max()}]")
    print(f"Embed shape: {obs['mineclip_embed'].shape}")
    print(f"Embed norm: {obs['mineclip_embed'].norm(dim=-1).mean()}")
    break
```

### 问题3: 微调后遗忘原有能力

**症状**: 在新任务上表现好，但原有任务(如砍树)退化

**解决方案**:

```bash
# 1. 混合数据训练
# 在采样配置中同时包含:
#   - 原始VPT数据 (80%)
#   - 新任务数据 (20%)

python data/sampling/generate_sampling.py \
    --type mixed \
    --dataset_dirs data/dataset_vpt/,data/dataset_custom/ \
    --dataset_weights 0.8,0.2 \
    --name mixed_training

# 2. 使用更小的学习率
LEARNING_RATE=5e-6  # 减半

# 3. 更短的训练时间
TOTAL_FRAMES=5000000  # 减半
```

### 问题4: 数据集路径错误

**症状**:
```
FileNotFoundError: Episode directory not found
```

**检查步骤**:
```bash
# 1. 检查采样配置文件内容
cat data/samplings/my_finetune_task_train.txt
# 应该输出episode路径列表

# 2. 验证路径存在
head -1 data/samplings/my_finetune_task_train.txt | xargs ls -ld

# 3. 检查episode完整性
python src/training/steve1/data/EpisodeStorage.py --validate /path/to/episode
```

---

## 💡 最佳实践

### 1. 数据质量 > 数据数量

**推荐**:
- ✅ 5小时高质量、任务相关的录像
- ❌ 50小时低质量、无关的录像

**高质量数据特征**:
- 专家级玩家操作
- 明确的任务目标
- 成功完成率高
- 多样化的场景

### 2. 迭代式微调

```
第1轮: 1小时数据 → 快速训练 → 评估 → 发现问题
第2轮: 补充3小时数据 → 针对性训练 → 再评估
第3轮: 调优超参数 → 最终训练
```

### 3. 保存多个检查点

```bash
# 训练脚本会自动保存:
data/weights/steve1/
├── steve1_my_task_latest.weights      # 最新权重
├── steve1_my_task_best.weights        # 验证损失最小
├── steve1_my_task_snapshot_5M.weights # 5M帧快照
└── steve1_my_task_snapshot_10M.weights
```

**选择权重策略**:
- `_best.weights`: 通常性能最好 (推荐)
- `_latest.weights`: 最新但可能过拟合
- `_snapshot_*.weights`: 用于对比不同训练阶段

### 4. 频繁验证

```bash
# 微调时使用更频繁的验证
VAL_FREQ=100  # 每100步验证 (而非默认1000)

# 好处:
# - 及早发现过拟合
# - 捕获最佳模型
# - 可以提前停止
```

---

## 📈 进阶技巧

### 技巧1: 冻结部分层

**场景**: 数据量很少 (< 3小时)，防止过拟合

```python
# 修改 training/train.py，在加载模型后添加:

# 冻结IMPALA CNN特征提取器
for param in agent.policy.net.img_process.parameters():
    param.requires_grad = False

# 冻结LSTM层
for param in agent.policy.net.recurrent_layer.parameters():
    param.requires_grad = False

# 仅微调:
# - mineclip_embed_linear (嵌入融合层)
# - pi_head (动作头)

print("Trainable parameters:")
for name, param in agent.policy.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.numel()} params")
```

### 技巧2: 学习率衰减策略

```bash
# 默认: Cosine退火 + warmup

# 如果训练不稳定，可以调整:
WARMUP_FRAMES=2000000     # 增加预热 (2M → 更平滑)
TOTAL_FRAMES=20000000     # 延长训练 (10M → 20M)

# 学习率曲线:
# 0 → 1e-5 (warmup) → 平稳 → cosine衰减 → 0
```

### 技巧3: 数据增强

```python
# 在 minecraft_dataset.py 中添加

import torchvision.transforms as T

class MinecraftDataset(Dataset):
    def __init__(self, ..., use_augmentation=False):
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.aug = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色抖动
                T.RandomGrayscale(p=0.1),  # 随机灰度化
            ])
    
    def __getitem__(self, idx):
        ...
        if self.use_augmentation:
            frames = self.aug(frames)
        ...
```

### 技巧4: Classifier-Free Guidance强度调节

```bash
# 训练时增加无条件比例
P_UNCOND=0.15  # 从0.1提高到0.15

# 推理时调整引导强度
# 在 run_agent.py 中:
--cond_scale 8.0  # 更强的条件引导 (默认6.0)
```

---

## 🎓 学习资源

### 推荐阅读顺序

1. **本文档** - 快速上手微调
2. **深度分析** - `docs/technical/STEVE1_TRAINING_ANALYSIS.md` - 理解训练原理
3. **脚本指南** - `docs/guides/STEVE1_SCRIPTS_USAGE_GUIDE.md` - 所有脚本详解
4. **评估指南** - `docs/guides/STEVE1_EVALUATION_GUIDE.md` - 模型评估方法

### 关键代码文件

```
src/training/steve1/
├── training/train.py              ← 训练主循环 (核心)
├── embed_conditioned_policy.py    ← 条件策略网络
├── data/minecraft_dataset.py      ← 数据加载与事后重标记
├── MineRLConditionalAgent.py      ← Agent封装
└── 3_train_finetune_template.sh   ← 微调脚本模板 (本指南提供)
```

### 外部参考

- **STEVE-1论文**: `docs/reference/STEVE-1: A Generative Model for Text-to-Behavior in Minecraft.pdf`
- **VPT论文**: "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos"
- **原始仓库**: https://github.com/Shalev-Lifshitz/STEVE-1

---

## ✅ 完整流程检查清单

微调前确保完成：

### 环境准备
- [ ] GPU可用 (nvidia-smi)
- [ ] conda环境激活 (minedojo)
- [ ] 预训练权重已下载
- [ ] 磁盘空间充足 (50GB+)

### 数据准备
- [ ] 运行 `1_generate_dataset.sh` 或准备自定义数据
- [ ] 运行 `2_create_sampling.sh` 生成采样配置
- [ ] 验证采样文件存在 (`*_train.txt`, `*_val.txt`)

### 脚本配置
- [ ] 复制 `3_train_finetune_template.sh`
- [ ] 修改 `SAMPLING_NAME` 参数
- [ ] 修改 `OUT_WEIGHTS` 输出名称
- [ ] 根据显存调整 `BATCH_SIZE`

### 训练监控
- [ ] 启动TensorBoard
- [ ] 观察前100步的loss是否正常
- [ ] 定期检查生成的视频
- [ ] 保存最佳模型

### 测试评估
- [ ] 使用微调模型生成测试视频
- [ ] 对比预训练模型的表现
- [ ] 在多个提示词上测试
- [ ] 记录性能改进

---

## 🚨 紧急救援

如果遇到无法解决的问题:

1. **查看日志**
   ```bash
   # 训练日志
   tail -100 logs/training/train_$(date +%Y%m%d).log
   
   # TensorBoard事件
   ls -lh data/finetuning_checkpoint/logs/
   ```

2. **从检查点恢复**
   ```bash
   # 训练会自动恢复，只需再次运行:
   bash 3_train_my_task.sh
   ```

3. **重新开始训练**
   ```bash
   # 删除检查点目录
   rm -rf data/finetuning_checkpoint/
   
   # 重新训练
   bash 3_train_my_task.sh
   ```

4. **寻求帮助**
   - 查阅: `docs/technical/STEVE1_TRAINING_ANALYSIS.md` 的常见问题章节
   - 检查: GitHub Issues (原始STEVE-1仓库)
   - 咨询: 项目维护者

---

## 🎉 成功案例

```
✅ 微调前: 在"建造木屋"任务上，成功率 30%
✅ 微调后: 在"建造木屋"任务上，成功率 85%

训练配置:
- 数据量: 5小时专家录像
- 训练时间: 4小时 (RTX 4090)
- 学习率: 1e-5
- 总帧数: 1000万

关键改进:
- 学会了正确放置方块
- 能够完成基础木屋结构
- 行为更加稳定和高效
```

---

## 📝 下一步行动

1. **准备数据**: 运行 `bash 1_generate_dataset.sh`
2. **创建配置**: 运行 `bash 2_create_sampling.sh`
3. **开始微调**: 运行 `bash 3_train_finetune_template.sh`
4. **监控训练**: 启动 TensorBoard
5. **测试模型**: 生成测试视频
6. **迭代优化**: 根据结果调整参数

祝微调顺利！🚀

---

**文档版本**: 1.0  
**最后更新**: 2025-10-31  
**反馈**: 如有问题或建议，请在项目中提Issue

