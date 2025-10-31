# STEVE-1 训练机制深度解析

> **文档目标**：深入分析 STEVE-1 的训练实现，为在现有模型上进行微调提供指导。  
> **创建日期**：2025-10-31  
> **适用场景**：理解训练原理、微调预训练模型、自定义训练策略

---

## 目录

1. [STEVE-1 架构概览](#1-steve-1-架构概览)
2. [训练流程详解](#2-训练流程详解)
3. [核心组件分析](#3-核心组件分析)
4. [训练数据流](#4-训练数据流)
5. [微调实战指南](#5-微调实战指南)
6. [调优策略](#6-调优策略)

---

## 1. STEVE-1 架构概览

### 1.1 整体架构

```
文本提示 "chop tree"
    ↓
[MineCLIP Text Encoder] → 512维文本嵌入
    ↓
[Prior VAE (可选)] → 512维条件嵌入
    ↓
┌─────────────────────────────────────┐
│   STEVE-1 Conditional Policy        │
│                                     │
│  [IMPALA CNN] ← 游戏画面 (128x128)  │
│       ↓                             │
│  [MineCLIP Embed融合层]             │
│       ↓                             │
│  [LSTM层 × N]                       │
│       ↓                             │
│  [Action Head] → Minecraft动作      │
└─────────────────────────────────────┘
```

### 1.2 关键特性

| 特性 | 说明 | 技术实现 |
|------|------|----------|
| **条件控制** | 通过MineCLIP嵌入控制行为 | 嵌入融合到视觉特征 |
| **无分类器引导** | Classifier-Free Guidance | 10%概率训练无条件策略 |
| **时序建模** | LSTM捕获动作序列 | 多层LSTM + 残差连接 |
| **事后重标记** | Hindsight Relabeling | 随机选择未来帧作为目标 |
| **分布式训练** | Accelerate支持多GPU | 混合精度bf16加速 |

---

## 2. 训练流程详解

### 2.1 完整训练管线

```bash
# 阶段1：准备数据 (1次性，数小时)
bash 1_generate_dataset.sh     # 下载VPT数据 → 生成MineCLIP嵌入
bash 2_create_sampling.sh       # 生成采样配置

# 阶段2：主模型训练 (数天)
bash 3_train.sh                 # 训练STEVE-1条件策略

# 阶段3：Prior训练 (可选，数小时)
bash 4_train_prior.sh           # 训练文本→嵌入的VAE
```

### 2.2 训练脚本核心参数解析

#### `3_train.sh` 参数详解

```bash
# ========== 模型路径 ==========
--in_model      # VPT架构定义文件 (.model)
--in_weights    # 初始权重 (可用VPT预训练权重)
--out_weights   # 输出权重路径

# ========== 扩散步数 ==========
--T 640                  # 完整扩散步数 (更高=更好质量，更慢训练)
--trunc_t 64             # 截断步数 (训练时每次处理的时间步)
                         # 梯度回传每64步截断一次，防止梯度消失

# ========== 批量大小 ==========
--batch_size 12                   # 每GPU每次处理12个序列
--gradient_accumulation_steps 4   # 梯度累积4次再更新
# 有效批量 = 12 × 4 × num_gpus = 48 (单GPU)

# ========== 学习率调度 ==========
--learning_rate 4e-5      # 峰值学习率
--warmup_frames 10M       # 预热1000万帧 (线性增长)
--n_frames 100M           # 总训练1亿帧
# 学习率: 0 → 4e-5 (预热) → cosine衰减 → 0

# ========== 条件训练 ==========
--p_uncond 0.1            # 10%概率零化条件 (用于Classifier-Free Guidance)
--min_btwn_goals 15       # 目标间隔最小15帧 (事后重标记)
--max_btwn_goals 200      # 目标间隔最大200帧

# ========== 数据采样 ==========
--sampling neurips        # 使用neurips采样配置
--sampling_dir data/samplings/

# ========== 验证与保存 ==========
--val_freq 1000               # 每1000步验证一次
--save_freq 1000              # 每1000步保存检查点
--snapshot_every_n_frames 50M # 每5000万帧保存快照
```

### 2.3 训练循环伪代码

```python
# 简化的训练循环 (training/train.py)
for epoch in range(num_epochs):
    for obs_batch, action_batch, firsts_batch in dataloader:
        # obs_batch: {
        #   'img': [B, T, 128, 128, 3],        # 游戏画面
        #   'mineclip_embed': [B, T, 512]      # 条件嵌入
        # }
        # action_batch: VPT格式动作
        # firsts_batch: [B, T] 序列开始标记
        
        # 1. 分块处理 (每次处理trunc_t=64步)
        for t in range(0, T, trunc_t):
            obs_chunk = obs_batch[:, t:t+trunc_t]
            action_chunk = action_batch[:, t:t+trunc_t]
            
            # 2. 前向传播
            policy_output = policy(obs_chunk, hidden_state, firsts_chunk)
            action_logits = policy_output['logits']
            
            # 3. 计算损失 (行为克隆 - Behavior Cloning)
            log_prob = compute_log_prob(action_logits, action_chunk)
            loss = -log_prob.mean()  # 负对数似然
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 梯度累积后更新
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 6. 定期验证
        if step % val_freq == 0:
            val_loss = validate(policy, val_dataloader)
            if val_loss < best_val_loss:
                save_model(policy, 'best_model.weights')
```

---

## 3. 核心组件分析

### 3.1 条件策略网络 (`MinecraftAgentPolicy`)

**文件位置**: `embed_conditioned_policy.py`

#### 架构分解

```python
class MinecraftAgentPolicy(nn.Module):
    def __init__(self, ...):
        # 1. 图像预处理
        self.img_preprocess = ImgPreprocessing()  # 归一化: x/255
        
        # 2. IMPALA CNN特征提取
        self.img_process = ImgObsProcess(
            cnn_outsize=256,     # CNN输出维度
            output_size=512      # 线性层映射到hidsize
        )
        
        # 3. MineCLIP嵌入融合 (STEVE-1核心创新)
        self.mineclip_embed_linear = nn.Linear(512, 512)
        
        # 4. 多层LSTM
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=512,
            n_block=n_recurrence_layers  # 默认多层
        )
        
        # 5. 输出层
        self.lastlayer = Linear(512, 512)
        self.final_ln = LayerNorm(512)
        
        # 6. 动作头
        self.pi_head = make_action_head(action_space, 512)
        self.value_head = ScaledMSEHead(512, 1)
    
    def forward(self, obs, state_in, first):
        # 处理图像
        x = self.img_preprocess(obs["img"])      # [B, T, 128, 128, 3]
        x = self.img_process(x)                   # [B*T, 512]
        
        # 融合MineCLIP嵌入 (关键步骤!)
        mineclip_embed = obs["mineclip_embed"]   # [B, T, 512]
        mineclip_embed = self.mineclip_embed_linear(mineclip_embed)
        x = x + mineclip_embed  # 残差连接，条件信息融入特征
        
        # LSTM处理时序
        x, state_out = self.recurrent_layer(x, first, state_in)
        
        # 输出动作分布
        x = self.lastlayer(x)
        pi_logits = self.pi_head(x)  # 预测动作概率
        vpred = self.value_head(x)   # 预测状态价值
        
        return pi_logits, vpred, state_out
```

#### 关键技术点

1. **嵌入融合方式**: 使用**加法残差连接**而非拼接
   - 优点: 保持特征维度，易于梯度流动
   - 实现: `x = visual_features + mineclip_embed`

2. **Classifier-Free Guidance 实现**
   ```python
   # 推理时使用 (run_agent.py)
   if cond_scale is not None:
       # 同时预测条件和无条件输出
       obs_cond = obs.copy()
       obs_uncond = obs.copy()
       obs_uncond["mineclip_embed"] = torch.zeros_like(obs["mineclip_embed"])
       
       logits_cond = policy(obs_cond)
       logits_uncond = policy(obs_uncond)
       
       # 加权组合 (cond_scale越大，条件影响越强)
       logits = (1 + cond_scale) * logits_cond - cond_scale * logits_uncond
   ```

3. **事后重标记机制**
   ```python
   # 数据加载时 (minecraft_dataset.py)
   def get_episode_chunk(...):
       # 随机选择未来帧作为"目标"
       goal_timesteps = []
       curr = 0
       while curr < total_timesteps:
           curr += random.randint(min_btwn_goals, max_btwn_goals)
           goal_timesteps.append(curr)
       
       # 每个时间步的条件 = 最近未来目标的MineCLIP嵌入
       for t in range(T):
           nearest_future_goal = find_next_goal(t, goal_timesteps)
           embed = mineclip_embeds[nearest_future_goal]
           # 训练时告诉模型: "向这个目标前进"
   ```

### 3.2 数据集类 (`MinecraftDataset`)

**文件位置**: `data/minecraft_dataset.py`

```python
class MinecraftDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 加载episode片段
        episode_path, start, end = self.episode_chunks[idx]
        episode = EpisodeStorage(episode_path)
        
        # 2. 加载MineCLIP嵌入 (预先计算)
        embeds = episode.load_embeds_attn()  # [T, 512]
        
        # 3. 事后重标记 - 选择目标
        goal_timesteps = self.sample_goal_timesteps()
        embeds_per_timestep = []
        for t in range(T):
            goal_idx = self.find_next_goal(t, goal_timesteps)
            embeds_per_timestep.append(embeds[goal_idx])
        
        # 4. 10%概率零化条件 (无分类器引导训练)
        if random.random() < self.p_uncond:
            embeds_per_timestep = [np.zeros_like(e) for e in embeds_per_timestep]
        
        # 5. 组装obs和actions
        obs = {
            'img': frames,                  # [T, 128, 128, 3]
            'mineclip_embed': embeds_per_timestep  # [T, 512]
        }
        actions = load_actions()            # [T, ...]
        firsts = [True] + [False] * (T-1)  # [T]
        
        return obs, actions, firsts
```

### 3.3 Prior VAE (可选组件)

**文件位置**: `data/text_alignment/vae.py`

#### 作用

- **训练时**: 学习 `P(visual_embed | text_embed)`
- **推理时**: 将文本嵌入转为视觉嵌入分布，增加多样性

```python
class TranslatorVAE(nn.Module):
    def forward(self, text_embed):
        # 编码器: text_embed → μ, σ (潜在分布)
        mu, logvar = self.encoder(text_embed)
        
        # 采样: z ~ N(μ, σ)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        
        # 解码器: z + text_embed → visual_embed
        visual_embed = self.decoder(torch.cat([z, text_embed]))
        
        return visual_embed
```

**使用场景**:
- 默认: 直接使用MineCLIP文本嵌入 (确定性)
- 加Prior: 采样多样化的视觉嵌入 (随机性)

---

## 4. 训练数据流

### 4.1 数据准备流程

```
[OpenAI VPT数据集下载]
    ↓
contractor_*.mp4 (人类玩家录像)
    ↓
[1_generate_dataset.sh] 
    ├─ 视频解码 → frames/*.png
    ├─ 提取动作 → actions.jsonl
    └─ MineCLIP编码 → embeds_attn.pkl
    ↓
EpisodeStorage/
├── frames/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── actions.jsonl       # 每行: {"camera": [x,y], "buttons": {...}}
└── embeds_attn.pkl     # [T, 512] MineCLIP特征
    ↓
[2_create_sampling.sh]
    ↓
samplings/neurips_train.txt  # episode路径列表
samplings/neurips_val.txt
```

### 4.2 数据采样配置

**文件内容示例** (`neurips_train.txt`):
```
/path/to/dataset/episode_0001
/path/to/dataset/episode_0002
...
```

**配置生成**:
```bash
# 2_create_sampling.sh
python data/sampling/generate_sampling.py \
    --type neurips \
    --name neurips \
    --output_dir data/samplings/ \
    --val_frames 10000    # 验证集1万帧
    --train_frames 30000  # 训练集3万帧
```

### 4.3 训练时数据流

```python
# 每个batch的数据形状
batch = dataloader[i]
obs, actions, firsts = batch

# obs字典:
obs = {
    'img': Tensor[B, T, 128, 128, 3],       # B=12, T=640
    'mineclip_embed': Tensor[B, T, 512]
}

# actions字典:
actions = {
    'buttons': Tensor[B, T, 8641],          # 离散按钮 (独热编码)
    'camera': Tensor[B, T, 121]             # 摄像机移动 (分层编码)
}

# firsts标记:
firsts = Tensor[B, T]  # True=序列开始, False=继续
```

---

## 5. 微调实战指南

### 5.1 微调场景分类

| 场景 | 数据需求 | 训练时长 | 难度 |
|------|---------|---------|------|
| **场景A**: 特定任务微调 | 1-5小时录像 | 数小时 | ⭐ |
| **场景B**: 新行为扩展 | 10-50小时录像 | 1-2天 | ⭐⭐ |
| **场景C**: 完整重训练 | 100+小时录像 | 3-7天 | ⭐⭐⭐ |

### 5.2 场景A：特定任务微调 (推荐)

#### 目标
在预训练STEVE-1基础上，微调用于特定任务（如"建造木屋"）。

#### 步骤1：准备自定义数据

```bash
# 1. 录制自己的游戏数据
# 使用MineDojo或MineRL环境，记录专家演示
cd src/training/steve1

# 2. 转换为STEVE-1格式
python data/generation/convert_custom_data.py \
    --input_dir /path/to/your/recordings/ \
    --output_dir data/dataset_custom/
    # 此脚本会:
    #   - 提取帧
    #   - 生成MineCLIP嵌入
    #   - 保存为EpisodeStorage格式
```

**自定义数据格式要求**:
```
your_recordings/
├── episode_001/
│   ├── video.mp4        # 游戏录像
│   └── actions.jsonl    # 每帧动作
├── episode_002/
└── ...
```

#### 步骤2：创建微调采样配置

```bash
# 创建自定义采样
python data/sampling/generate_sampling.py \
    --type custom \
    --name finetune_build_house \
    --dataset_dirs data/dataset_custom/ \
    --output_dir data/samplings/ \
    --val_frames 2000 \
    --train_frames 10000
```

#### 步骤3：修改训练脚本

创建 `3_train_finetune.sh`:

```bash
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

accelerate launch --num_processes 1 --mixed_precision bf16 "$SCRIPT_DIR/training/train.py" \
# ========== 关键：加载预训练权重 ==========
--in_model "$PROJECT_ROOT/data/weights/vpt/2x.model" \
--in_weights "$PROJECT_ROOT/data/weights/steve1/steve1.weights" \  # 使用预训练STEVE-1
--out_weights "$PROJECT_ROOT/data/weights/steve1/steve1_finetuned_house.weights" \

# ========== 微调专用参数 ==========
--batch_size 8 \              # 减小批量 (数据少)
--gradient_accumulation_steps 2 \
--trunc_t 64 \
--T 320 \                     # 减半序列长度 (加速训练)

# ========== 更激进的学习率 ==========
--learning_rate 1e-5 \        # 比从头训练小10倍 (避免遗忘)
--warmup_frames 1_000_000 \   # 减少预热
--n_frames 10_000_000 \       # 总帧数1000万 (约数小时)

# ========== 更频繁的验证 ==========
--val_freq 200 \              # 每200步验证
--save_freq 500 \

# ========== 使用自定义采样 ==========
--sampling finetune_build_house \
--sampling_dir "$PROJECT_ROOT/data/samplings/" \

# ========== 条件训练 ==========
--p_uncond 0.1 \
--min_btwn_goals 15 \
--max_btwn_goals 100 \

--checkpoint_dir "$PROJECT_ROOT/data/finetuning_checkpoint"
```

#### 步骤4：启动微调

```bash
cd src/training/steve1
bash 3_train_finetune.sh

# 监控训练
tensorboard --logdir ../../data/finetuning_checkpoint/logs/
```

#### 步骤5：测试微调模型

```bash
# 修改 2_gen_vid_for_text_prompt.sh
# 将 --in_weights 改为:
--in_weights "$PROJECT_ROOT/data/weights/steve1/steve1_finetuned_house.weights" \
--custom_text_prompt "build wooden house"

bash 2_gen_vid_for_text_prompt.sh
```

### 5.3 场景B：冻结部分层微调 (高级)

#### 策略
仅微调策略头和嵌入融合层，冻结IMPALA CNN和LSTM。

```python
# 修改 training/train.py 的 main() 函数

# 加载模型后，冻结特定层
agent = MineRLConditionalAgent(...)
agent.load_weights(args.in_weights)

# 冻结IMPALA CNN
for param in agent.policy.net.img_process.parameters():
    param.requires_grad = False

# 冻结LSTM层
for param in agent.policy.net.recurrent_layer.parameters():
    param.requires_grad = False

# 仅微调:
# - mineclip_embed_linear (嵌入融合层)
# - pi_head (动作头)
# - value_head (价值头)

policy = DDPPolicy(agent.policy)
optimizer = configure_optimizers(policy, args.weight_decay, args.learning_rate)
# 优化器会自动跳过 requires_grad=False 的参数
```

**优点**:
- 训练更快 (参数少)
- 防止过拟合 (保留通用特征)

**适用场景**:
- 数据量少 (< 10小时)
- 任务与VPT数据类似

### 5.4 场景C：调整条件嵌入维度

如果你有自定义的嵌入模型（如不同版本的CLIP），需要调整嵌入维度。

```python
# 修改 embed_conditioned_policy.py

class MinecraftPolicy(nn.Module):
    def __init__(self, ..., mineclip_embed_dim=512):  # 默认512
        ...
        # 修改这行以匹配你的嵌入维度
        self.mineclip_embed_linear = nn.Linear(mineclip_embed_dim, hidsize)

# 训练时传入参数
agent = MineRLConditionalAgent(
    env,
    policy_kwargs={
        ...
        'mineclip_embed_dim': 768  # 例如使用CLIP-ViT-L的768维
    }
)
```

---

## 6. 调优策略

### 6.1 超参数调优表

| 参数 | 默认值 | 调小 (少数据) | 调大 (多数据) | 影响 |
|------|-------|--------------|--------------|------|
| `learning_rate` | 4e-5 | 1e-5 | 8e-5 | 收敛速度 |
| `batch_size` | 12 | 4-8 | 16-32 | 训练稳定性 |
| `gradient_accumulation_steps` | 4 | 2 | 8 | 有效批量大小 |
| `T` | 640 | 320 | 1280 | 时序建模能力 |
| `trunc_t` | 64 | 32 | 128 | 梯度流长度 |
| `p_uncond` | 0.1 | 0.05 | 0.2 | CFG强度 |
| `min_btwn_goals` | 15 | 10 | 30 | 目标密度 |
| `max_btwn_goals` | 200 | 100 | 300 | 目标多样性 |
| `weight_decay` | 0.039 | 0.01 | 0.1 | 正则化强度 |

### 6.2 常见问题与解决

#### 问题1: 显存不足 (OOM)

**解决方案**:
```bash
# 方案A: 减小批量大小
--batch_size 4 \
--gradient_accumulation_steps 12  # 保持有效批量=48

# 方案B: 减小序列长度
--T 320 \
--trunc_t 32

# 方案C: 使用梯度检查点 (需修改代码)
# 在 embed_conditioned_policy.py 中添加:
import torch.utils.checkpoint as checkpoint
x = checkpoint.checkpoint(self.recurrent_layer, x, first, state_in)
```

#### 问题2: 训练损失不下降

**诊断步骤**:
```python
# 1. 检查学习率
print(f"Current LR: {optimizer.param_groups[0]['lr']}")
# 如果卡在warmup阶段 → 减少 warmup_frames

# 2. 检查梯度范数
grad_norm = compute_gradient_l2_norm(policy)
print(f"Gradient L2 Norm: {grad_norm}")
# 如果 > 10.0 → 梯度爆炸 → 减小学习率或增加 max_grad_norm

# 3. 检查数据分布
for obs, actions, firsts in dataloader:
    print(f"Embed norm: {obs['mineclip_embed'].norm()}")
    print(f"Image mean: {obs['img'].mean()}")
    break
# 如果异常 → 数据预处理问题
```

**常见原因**:
- 学习率过大/过小
- 批量大小过小 (噪声大)
- 数据质量问题 (嵌入未归一化)

#### 问题3: 微调后遗忘原有能力

**解决方案**:
```bash
# 1. 混合数据集训练
# 在采样配置中同时包含:
#   - 原始VPT数据 (80%)
#   - 新任务数据 (20%)

# 2. 使用更小的学习率
--learning_rate 5e-6  # 比默认小8倍

# 3. 早停策略
# 当验证损失不再下降时提前停止
# 在 train.py 中添加:
if val_loss > best_val_loss_in_last_5_validations:
    print("Early stopping!")
    break
```

#### 问题4: Classifier-Free Guidance 效果不明显

**增强CFG效果**:
```bash
# 1. 增加无条件训练比例
--p_uncond 0.2  # 从10%提高到20%

# 2. 推理时使用更大的引导强度
# 修改 run_agent.py:
--cond_scale 8.0  # 从6.0提高到8.0
# 注意: 过大会导致动作不自然

# 3. 确保零嵌入确实是零
# 检查数据加载:
if np.random.rand() < p_uncond:
    embeds = [np.zeros(512, dtype=np.float32) for _ in embeds]
```

### 6.3 性能优化技巧

#### 技巧1: 数据加载加速

```bash
# 增加DataLoader worker数量
--num_workers 8  # 从4提高到8 (如果CPU核心足够)

# 预先将数据集复制到本地SSD
rsync -av /remote/dataset/ /local/ssd/dataset/
# 修改采样配置中的路径
```

#### 技巧2: 多GPU训练

```bash
# 使用Accelerate自动分布式训练
accelerate config  # 首次配置

# 修改启动命令
accelerate launch \
    --num_processes 2 \           # 2块GPU
    --multi_gpu \
    --mixed_precision bf16 \
    training/train.py ...

# 相应调整批量大小
--batch_size 6  # 每GPU 6个 → 总共12个
```

#### 技巧3: 混合精度训练

```bash
# 已启用: --mixed_precision bf16

# 如果遇到数值不稳定:
# 1. 改用fp16 (更精细的缩放)
--mixed_precision fp16

# 2. 或关闭混合精度 (更慢但稳定)
--mixed_precision no
```

---

## 7. 微调检查清单

在开始微调前，确保完成以下检查：

### 环境准备
- [ ] 安装所有依赖 (`conda activate minedojo`)
- [ ] 下载预训练权重 (`steve1.weights`, `2x.model`)
- [ ] GPU显存 ≥ 24GB (推荐 RTX 3090/4090)
- [ ] 磁盘空间 ≥ 100GB

### 数据准备
- [ ] 准备训练数据（至少1小时游戏录像）
- [ ] 生成MineCLIP嵌入
- [ ] 创建采样配置文件
- [ ] 验证数据格式正确

### 训练配置
- [ ] 设置合理的学习率 (1e-5 for finetuning)
- [ ] 调整批量大小 (根据显存)
- [ ] 配置验证频率 (频繁验证防止过拟合)
- [ ] 设置检查点保存路径

### 监控与调试
- [ ] 启动TensorBoard监控
- [ ] 检查前几个batch的损失值
- [ ] 验证梯度范数正常 (< 10.0)
- [ ] 定期测试生成的行为

---

## 8. 参考资源

### 相关文档
- **STEVE-1脚本指南**: `docs/guides/STEVE1_SCRIPTS_USAGE_GUIDE.md`
- **MineCLIP策略**: `docs/design/UNIVERSAL_MINECLIP_STRATEGY.md`
- **VPT权重加载**: `docs/technical/VPT_WEIGHT_LOADING_EXPLAINED.md`

### 代码文件速查
| 功能 | 文件路径 |
|------|---------|
| 主训练循环 | `training/train.py` |
| 条件策略 | `embed_conditioned_policy.py` |
| 数据加载 | `data/minecraft_dataset.py` |
| Agent封装 | `MineRLConditionalAgent.py` |
| Prior VAE | `data/text_alignment/vae.py` |

### 外部资源
- **STEVE-1论文**: `docs/reference/STEVE-1: A Generative Model for Text-to-Behavior in Minecraft.pdf`
- **GitHub原始仓库**: https://github.com/Shalev-Lifshitz/STEVE-1

---

## 9. 总结

### 训练流程概括

```
VPT数据 → MineCLIP嵌入 → 事后重标记 → 行为克隆训练 → STEVE-1模型
   ↓                ↓                ↓                ↓
人类录像      条件信号          目标采样        条件策略学习
```

### 微调核心要点

1. **数据为王**: 高质量、任务相关的数据比大量无关数据更有效
2. **学习率减半**: 微调时使用比从头训练小5-10倍的学习率
3. **频繁验证**: 防止在小数据集上过拟合
4. **保留预训练**: 考虑冻结部分层以保留通用能力
5. **迭代优化**: 先快速实验，再逐步精调

### 下一步行动

**快速开始微调**:
```bash
# 1. 准备数据
bash prepare_custom_data.sh

# 2. 创建微调脚本
cp 3_train.sh 3_train_finetune.sh
# 修改参数...

# 3. 启动训练
bash 3_train_finetune.sh

# 4. 测试效果
bash 2_gen_vid_for_text_prompt.sh
```

**持续优化**:
- 监控训练指标 (loss, grad_norm)
- A/B测试不同超参数
- 收集更多高质量数据
- 结合强化学习进一步优化

---

**文档版本**: 1.0  
**维护者**: AIMC 项目组  
**反馈**: 如有问题或改进建议，请在项目中提issue

