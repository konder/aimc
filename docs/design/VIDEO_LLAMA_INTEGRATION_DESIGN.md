# Video-LLaMA集成方案设计

**创建日期**: 2025-11-28  
**目标**: 使用Video-LLaMA替代MineCLIP实现文本→目标画面→VPT对齐  
**预期改进**: 提高任务区分度50-80%

---

## 🎯 方案概述

### 当前架构（STEVE-1 + MineCLIP）

```
用户指令 (text)
    ↓
MineCLIP Text Encoder [512-d]
    ↓
Prior VAE (CVAE) [512-d → 512-d]
    ↓
z_goal (visual embedding) [512-d]
    ↓
VPT Policy (conditioned on z_goal)
    ↓
Minecraft Actions
```

**问题**：
- ❌ MineCLIP文本区分度低（1.3%）
- ❌ MineCLIP视觉相似度高（0.925）
- ❌ Prior VAE进一步collapse（0.873）

### 新架构（Video-LLaMA方案）

```
用户指令 (text)
    ↓
Video-LLaMA [7B/13B parameters]
    ├─ Text Encoder (LLM)
    └─ Visual Query Tokens
        ↓
    Goal Video Features [N×4096-d]
        ↓
    Alignment Module (Adapter/Projector) [4096-d → 512-d]
        ↓
    z_goal (VPT-compatible) [512-d]
        ↓
    VPT Policy (conditioned on z_goal)
        ↓
    Minecraft Actions
```

**优势**：
- ✅ Video-LLaMA强大的文本理解（基于LLaMA）
- ✅ 更好的视频-文本对齐
- ✅ 可以生成目标画面的详细描述
- ✅ 支持复杂指令理解

---

## 📐 详细架构设计

### 阶段1: Video-LLaMA基础架构

#### 1.1 Video-LLaMA组件

```python
class VideoLLaMAForGoalPrediction:
    """
    使用Video-LLaMA预测目标画面
    """
    def __init__(self):
        # 1. 视觉编码器 (Q-Former from BLIP-2)
        self.visual_encoder = VisionTransformer()  # EVA-CLIP ViT-g/14
        self.visual_qformer = QFormer()  # 32 query tokens
        
        # 2. 语言模型 (LLaMA)
        self.language_model = LLaMA_7B()  # 或 13B
        
        # 3. 视频适配器
        self.video_adapter = VideoQFormer()  # 时序建模
        
        # 4. 目标画面预测头
        self.goal_predictor = GoalPredictionHead()
    
    def forward(self, text_instruction):
        """
        输入: 文本指令 "chop tree, get a log"
        输出: 目标画面的visual features
        """
        # Step 1: 文本编码
        text_embeds = self.language_model.encode_text(text_instruction)
        # [1, seq_len, 4096]
        
        # Step 2: 生成visual query tokens
        # 使用LLaMA的输出作为条件，生成目标画面的query
        visual_queries = self.goal_predictor.generate_queries(text_embeds)
        # [1, 32, 768] - 32个query tokens
        
        # Step 3: 通过Q-Former得到visual features
        goal_visual_features = self.visual_qformer(visual_queries)
        # [1, 32, 768]
        
        return goal_visual_features
```

#### 1.2 目标画面预测头设计

```python
class GoalPredictionHead(nn.Module):
    """
    从文本预测目标画面的visual features
    
    灵感来源：BLIP-2的image generation，但这里是预测而非生成
    """
    def __init__(self, llama_dim=4096, query_dim=768, num_queries=32):
        super().__init__()
        
        # 1. Text-to-Query投影层
        self.text_to_query = nn.Sequential(
            nn.Linear(llama_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_queries * query_dim),
        )
        
        # 2. Query refinement（可选，使用Transformer）
        self.query_refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=query_dim, nhead=8),
            num_layers=4
        )
        
        # 3. 可学习的query embeddings（类似DETR的object queries）
        self.learnable_queries = nn.Parameter(
            torch.randn(num_queries, query_dim)
        )
    
    def generate_queries(self, text_embeds):
        """
        从文本embedding生成visual query tokens
        
        Args:
            text_embeds: [B, seq_len, 4096] - LLaMA的输出
        
        Returns:
            visual_queries: [B, 32, 768] - 目标画面的query tokens
        """
        B = text_embeds.shape[0]
        
        # 1. 池化文本特征
        text_pooled = text_embeds.mean(dim=1)  # [B, 4096]
        
        # 2. 投影到query space
        queries = self.text_to_query(text_pooled)  # [B, 32*768]
        queries = queries.view(B, 32, 768)
        
        # 3. 与可学习的queries相加（类似positional encoding）
        queries = queries + self.learnable_queries.unsqueeze(0)
        
        # 4. Refine queries
        queries = queries.transpose(0, 1)  # [32, B, 768] for Transformer
        refined_queries = self.query_refiner(queries)
        refined_queries = refined_queries.transpose(0, 1)  # [B, 32, 768]
        
        return refined_queries
```

### 阶段2: 对齐到VPT

#### 2.1 对齐模块设计

```python
class VideoLLaMAToVPTAligner(nn.Module):
    """
    将Video-LLaMA的visual features对齐到VPT的visual embedding空间
    
    关键：VPT使用的是IDM (Inverse Dynamics Model) 训练的visual encoder
    输出维度：512-d（与MineCLIP一致）
    """
    def __init__(
        self, 
        videollama_dim=768,  # Q-Former output dim
        num_queries=32,
        vpt_dim=512,
        use_attention=True
    ):
        super().__init__()
        
        # 方案A: 简单投影（类似MineCLIP Prior）
        if not use_attention:
            self.aligner = nn.Sequential(
                nn.Linear(num_queries * videollama_dim, 2048),
                nn.LayerNorm(2048),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, vpt_dim),
            )
            self.use_attention = False
        
        # 方案B: 注意力池化（推荐）
        else:
            # 1. 跨query的注意力池化
            self.query_attention = nn.MultiheadAttention(
                embed_dim=videollama_dim,
                num_heads=8,
                batch_first=True
            )
            
            # 2. 可学习的pooling query
            self.pool_query = nn.Parameter(torch.randn(1, 1, videollama_dim))
            
            # 3. 投影到VPT空间
            self.projection = nn.Sequential(
                nn.Linear(videollama_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, vpt_dim),
            )
            self.use_attention = True
    
    def forward(self, videollama_features):
        """
        Args:
            videollama_features: [B, 32, 768] - Video-LLaMA的visual features
        
        Returns:
            z_goal: [B, 512] - VPT兼容的visual embedding
        """
        B = videollama_features.shape[0]
        
        if not self.use_attention:
            # 方案A: 简单flatten + MLP
            features_flat = videollama_features.view(B, -1)  # [B, 32*768]
            z_goal = self.aligner(features_flat)  # [B, 512]
        
        else:
            # 方案B: 注意力池化（推荐）
            # 1. 使用可学习的query池化32个visual tokens
            pool_query = self.pool_query.expand(B, -1, -1)  # [B, 1, 768]
            pooled_features, _ = self.query_attention(
                query=pool_query,
                key=videollama_features,
                value=videollama_features
            )  # [B, 1, 768]
            
            # 2. 投影到VPT空间
            pooled_features = pooled_features.squeeze(1)  # [B, 768]
            z_goal = self.projection(pooled_features)  # [B, 512]
        
        return z_goal
```

#### 2.2 完整的文本→VPT流程

```python
class TextToVPTGoalPredictor(nn.Module):
    """
    完整的文本指令 → VPT goal embedding流程
    """
    def __init__(self):
        super().__init__()
        
        # 1. Video-LLaMA（预训练权重）
        self.videollama = VideoLLaMAForGoalPrediction()
        
        # 2. 对齐模块（需要训练）
        self.aligner = VideoLLaMAToVPTAligner(
            videollama_dim=768,
            num_queries=32,
            vpt_dim=512,
            use_attention=True
        )
        
        # 3. VPT Policy（预训练权重，frozen）
        self.vpt_policy = VPTPolicy()
        self.vpt_policy.eval()
        for param in self.vpt_policy.parameters():
            param.requires_grad = False
    
    def forward(self, text_instruction, current_observation):
        """
        Args:
            text_instruction: str - "chop tree, get a log"
            current_observation: [B, 3, H, W] - 当前游戏画面
        
        Returns:
            action: dict - Minecraft action
        """
        # Step 1: 文本 → Video-LLaMA visual features
        with torch.no_grad():  # Video-LLaMA可以frozen或微调
            videollama_features = self.videollama(text_instruction)
            # [B, 32, 768]
        
        # Step 2: 对齐到VPT空间
        z_goal = self.aligner(videollama_features)
        # [B, 512]
        
        # Step 3: VPT policy生成动作
        action = self.vpt_policy(
            observation=current_observation,
            goal_embedding=z_goal
        )
        
        return action, z_goal
```

---

## 🎓 训练策略

### 训练阶段1: Video-LLaMA在Minecraft上的预训练

#### 数据准备

```python
# 使用MineCLIP的730K YouTube数据
minecraft_youtube_data = {
    'video_paths': [...],  # 730K视频
    'transcripts': [...],  # 时间对齐的字幕
    'task_labels': [...],  # 可选：任务类别标注
}

# 数据增强
def prepare_videollama_data(video_path, transcript):
    """
    为Video-LLaMA准备训练数据
    
    策略1: 视频-字幕对比学习（类似MineCLIP）
    策略2: 视频摘要生成
    策略3: 目标画面预测
    """
    # 1. 提取视频片段（16帧）
    video_clip = extract_frames(video_path, num_frames=16)
    
    # 2. 提取对应的字幕
    caption = get_aligned_caption(transcript, video_clip.timestamp)
    
    # 3. 提取"成功时刻"的画面作为目标
    success_frames = extract_success_frames(video_path)
    
    return {
        'video': video_clip,
        'caption': caption,
        'goal_frames': success_frames,  # 用于训练目标预测
    }
```

#### 训练目标

```python
def train_videollama_minecraft(model, dataloader, epochs=10):
    """
    Video-LLaMA在Minecraft上的训练
    
    Loss组合:
    1. 对比学习loss (类似CLIP)
    2. 目标预测loss
    3. 语言建模loss（可选）
    """
    for epoch in range(epochs):
        for batch in dataloader:
            video = batch['video']  # [B, 16, 3, H, W]
            caption = batch['caption']  # [B, seq_len]
            goal_frames = batch['goal_frames']  # [B, 16, 3, H, W]
            
            # Loss 1: 视频-文本对比学习
            video_features = model.encode_video(video)
            text_features = model.encode_text(caption)
            contrastive_loss = clip_loss(video_features, text_features)
            
            # Loss 2: 目标预测
            # 从文本预测目标画面的visual features
            predicted_goal_features = model.predict_goal(caption)
            actual_goal_features = model.encode_video(goal_frames)
            goal_prediction_loss = mse_loss(
                predicted_goal_features, 
                actual_goal_features
            )
            
            # 总loss
            total_loss = contrastive_loss + 0.5 * goal_prediction_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
```

### 训练阶段2: 对齐模块训练

#### 方案A: 使用成功trials对齐（推荐）

```python
def train_aligner_with_success_trials(aligner, videollama, vpt_encoder):
    """
    使用成功的游戏trials训练对齐模块
    
    核心思想：
    1. 对于成功的trial，提取成功时刻的画面
    2. 用VPT的visual encoder编码这些画面
    3. 训练Video-LLaMA预测的features与VPT features对齐
    """
    # 数据：results/evaluation/all_tasks_*/
    success_trials = load_success_trials('results/evaluation/')
    
    for trial in success_trials:
        instruction = trial['instruction']  # "chop tree, get a log"
        success_frames = trial['success_frames']  # 成功时刻的16帧
        
        # 1. Video-LLaMA预测goal features
        videollama_features = videollama.predict_goal(instruction)
        # [1, 32, 768]
        
        # 2. VPT encoder编码实际成功画面
        with torch.no_grad():
            vpt_visual_embedding = vpt_encoder(success_frames)
            # [1, 512]
        
        # 3. 对齐
        predicted_embedding = aligner(videollama_features)
        # [1, 512]
        
        # Loss: 对齐预测的embedding和实际的VPT embedding
        alignment_loss = nn.MSELoss()(
            predicted_embedding, 
            vpt_visual_embedding
        )
        
        # 也可以用cosine similarity
        cosine_loss = 1 - F.cosine_similarity(
            predicted_embedding,
            vpt_visual_embedding
        ).mean()
        
        total_loss = alignment_loss + 0.1 * cosine_loss
        total_loss.backward()
```

#### 方案B: 强化学习fine-tune（可选）

```python
def finetune_with_rl(model, env, num_episodes=1000):
    """
    使用RL fine-tune整个pipeline
    
    核心思想：
    1. 用Video-LLaMA预测goal
    2. 用VPT执行
    3. 根据任务成功与否调整Video-LLaMA的预测
    """
    for episode in range(num_episodes):
        instruction = sample_instruction()
        obs = env.reset()
        
        # 1. 预测goal
        z_goal = model.predict_goal(instruction)
        
        # 2. VPT执行
        done = False
        total_reward = 0
        while not done:
            action = vpt_policy(obs, z_goal)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        # 3. 根据成功与否更新
        # 使用REINFORCE或其他PG算法
        if info['success']:
            # 增强这个goal prediction
            loss = -log_prob * total_reward
        else:
            # 惩罚这个prediction
            loss = log_prob * total_reward
        
        loss.backward()
```

---

## 📊 实施计划

### Phase 1: 准备阶段（1-2周）

**任务**：
1. ✅ 搭建Video-LLaMA环境
2. ✅ 下载预训练权重
3. ✅ 准备Minecraft数据
4. ✅ 实现基础架构代码

**代码**：
```bash
# 1. 安装Video-LLaMA
git clone https://github.com/DAMO-NLP-SG/Video-LLaMA
cd Video-LLaMA
pip install -r requirements.txt

# 2. 下载预训练权重
bash download_checkpoints.sh

# 3. 准备数据
python scripts/prepare_minecraft_data.py \
    --youtube_dir data/mineclip_youtube \
    --output_dir data/videollama_minecraft
```

### Phase 2: Video-LLaMA训练（2-4周）

**任务**：
1. 在Minecraft YouTube数据上预训练
2. 训练目标预测头
3. 评估视频-文本对齐质量

**硬件需求**：
- GPU: 8x A100 40GB（或 4x A100 80GB）
- 训练时间: 2-3周
- 估计成本: $5,000-10,000（云GPU）

**训练脚本**：
```bash
# 分布式训练
torchrun --nproc_per_node=8 train_videollama_minecraft.py \
    --data_dir data/videollama_minecraft \
    --model_size 7B \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 10 \
    --learning_rate 1e-5 \
    --output_dir checkpoints/videollama_minecraft_7b
```

### Phase 3: 对齐模块训练（1周）

**任务**：
1. 提取成功trials的VPT visual embeddings
2. 训练对齐模块
3. 验证对齐质量

**代码**：
```bash
# 1. 提取VPT embeddings
python scripts/extract_vpt_embeddings.py \
    --success_trials results/evaluation/all_tasks_* \
    --vpt_checkpoint data/weights/vpt_policy.pt \
    --output data/vpt_visual_embeddings.pkl

# 2. 训练对齐模块
python train_aligner.py \
    --videollama_checkpoint checkpoints/videollama_minecraft_7b \
    --vpt_embeddings data/vpt_visual_embeddings.pkl \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Phase 4: 集成和评估（1-2周）

**任务**：
1. 集成到现有STEVE-1系统
2. 运行Prior评估
3. 对比MineCLIP vs Video-LLaMA

**评估脚本**：
```bash
# 生成instruction-video pairs（使用Video-LLaMA）
python scripts/generate_pairs_videollama.py \
    --videollama_checkpoint checkpoints/videollama_minecraft_7b \
    --aligner_checkpoint checkpoints/aligner_best.pt \
    --eval_result_dir results/evaluation/all_tasks_* \
    --output_dir results/instruction_video_pairs_videollama

# 运行Prior评估
bash scripts/run_prior_evaluation.sh \
    --instruction-video-pairs results/instruction_video_pairs_videollama \
    --output-dir results/prior_evaluation_videollama

# 对比分析
python scripts/compare_models.py \
    --mineclip_results results/prior_evaluation/prior_eval_* \
    --videollama_results results/prior_evaluation_videollama \
    --output comparison_report.html
```

---

## 📈 预期改进

### 指标对比

| 指标 | MineCLIP | Video-LLaMA（预期） | 改进 |
|------|---------|------------------|------|
| **文本区分度** | 1.3% | 15-30% | ✅ +13-28% |
| **视觉相似度** | 0.925 | 0.70-0.80 | ✅ -12-22% |
| **Goal Accuracy** | 0.91-0.99 | 0.60-0.75 | ✅ 更真实 |
| **Discriminability** | 0.12 | 0.50-0.70 | ✅ +38-58% |
| **语义鲁棒性** | 0.96-0.99 | 0.85-0.92 | ✅ 更合理 |

### 定性改进

**MineCLIP问题**：
```
"kill pig" vs "chop tree": 0.854 ← 几乎一样
"kill pig" vs "build house": 0.878 ← 差异仅2.4%
```

**Video-LLaMA预期**：
```
"kill pig" vs "chop tree": 0.65-0.75 ← 明显不同
"kill pig" vs "build house": 0.45-0.55 ← 完全不同
```

---

## 🔧 代码实现示例

### 完整的推理流程

```python
class STEVE1WithVideoLLaMA:
    """
    集成Video-LLaMA的STEVE-1系统
    """
    def __init__(
        self,
        videollama_checkpoint,
        aligner_checkpoint,
        vpt_checkpoint
    ):
        # 1. 加载Video-LLaMA
        self.videollama = VideoLLaMAForGoalPrediction()
        self.videollama.load_state_dict(
            torch.load(videollama_checkpoint)
        )
        self.videollama.eval()
        
        # 2. 加载对齐模块
        self.aligner = VideoLLaMAToVPTAligner()
        self.aligner.load_state_dict(
            torch.load(aligner_checkpoint)
        )
        
        # 3. 加载VPT policy
        self.vpt_policy = load_vpt_policy(vpt_checkpoint)
    
    def play_minecraft(self, instruction, max_steps=1000):
        """
        根据文本指令玩Minecraft
        
        Args:
            instruction: str - "chop tree, get a log"
            max_steps: int - 最大步数
        
        Returns:
            success: bool
            trajectory: list
        """
        # 1. 从文本预测goal embedding
        with torch.no_grad():
            # Video-LLaMA预测
            videollama_features = self.videollama(instruction)
            
            # 对齐到VPT空间
            z_goal = self.aligner(videollama_features)
        
        # 2. 初始化环境
        env = gym.make('MineRLBasaltFindCave-v0')
        obs = env.reset()
        
        # 3. 执行
        trajectory = []
        for step in range(max_steps):
            # VPT生成动作
            action = self.vpt_policy(obs, z_goal)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            trajectory.append({
                'obs': obs,
                'action': action,
                'reward': reward,
            })
            
            if done or info.get('success'):
                break
        
        env.close()
        
        return info.get('success', False), trajectory
```

---

## ⚠️ 挑战和风险

### 技术挑战

1. **计算资源需求高**
   - Video-LLaMA 7B: ~28GB GPU内存
   - 训练需要8x A100
   - 推理速度可能较慢（vs MineCLIP）

2. **训练数据质量**
   - MineCLIP的YouTube数据可能不够精确
   - 需要人工标注"成功时刻"
   - 可能需要额外收集数据

3. **对齐难度**
   - Video-LLaMA和VPT的visual space可能差异大
   - 需要大量成功trials作为训练数据
   - 对齐质量直接影响最终性能

### 解决方案

1. **降低计算成本**
   ```python
   # 使用量化
   model = load_videollama_4bit()  # 4-bit量化
   
   # 使用LoRA微调
   from peft import get_peft_model, LoraConfig
   lora_config = LoraConfig(r=16, lora_alpha=32)
   model = get_peft_model(model, lora_config)
   ```

2. **数据增强**
   ```python
   # 利用现有成功trials
   # 使用数据增强生成更多样本
   # 可能需要人工标注100-500个高质量样本
   ```

3. **渐进式训练**
   ```python
   # Phase 1: 先用MineCLIP的visual embeddings训练对齐
   # Phase 2: 逐渐替换为VPT的visual embeddings
   # Phase 3: 端到端fine-tune
   ```

---

## 📚 参考实现

### 关键代码文件结构

```
src/
├── models/
│   ├── videollama_goal_predictor.py  # Video-LLaMA目标预测
│   ├── vpt_aligner.py                # 对齐模块
│   └── steve1_videollama.py          # 集成系统
├── training/
│   ├── train_videollama.py           # Video-LLaMA训练
│   ├── train_aligner.py              # 对齐模块训练
│   └── train_end2end.py              # 端到端训练
├── evaluation/
│   ├── eval_videollama_prior.py      # Prior评估
│   └── compare_models.py             # 模型对比
└── utils/
    ├── videollama_utils.py
    └── vpt_utils.py
```

---

## ✅ 总结

### 方案优势

1. **✅ 显著提高区分度**：预期15-30%文本区分度（vs 1.3%）
2. **✅ 更好的泛化能力**：Video-LLaMA基于7B LLM
3. **✅ 支持复杂指令**：可以理解长句、复杂描述
4. **✅ 可解释性强**：Video-LLaMA可以生成文本描述

### 实施建议

**短期（如果资源有限）**：
- 使用Video-LLaMA的预训练权重
- 只训练对齐模块（1周，单卡A100）
- 快速验证效果

**中期（推荐）**：
- 在Minecraft数据上fine-tune Video-LLaMA
- 训练高质量对齐模块
- 完整评估和对比

**长期**：
- 端到端fine-tune整个系统
- 收集更多高质量数据
- 建立Minecraft视频-文本benchmark

---

**下一步行动**：
1. 评估GPU资源可用性
2. 搭建Video-LLaMA环境
3. 实现对齐模块原型
4. 小规模测试验证

如有充足资源，这个方案预期能将任务区分度提升50-80%！🚀






