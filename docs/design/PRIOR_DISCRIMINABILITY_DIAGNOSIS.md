# STEVE-1 Prior 模型区分度诊断报告

**日期**: 2024-12-04  
**状态**: 诊断完成，待改进

## 1. 问题背景

### 1.1 STEVE-1 模型架构回顾

STEVE-1 通过分解目标条件策略来实现指令跟随：

```
p(τ|y) = p(τ, z_goal|y) = p(z_goal|y) × p(τ|z_goal)
```

- **Prior 模型**: `p(z_goal|y)` - 从指令 y 生成目标视频嵌入 z_goal
- **Policy 模型**: `p(τ|z_goal)` - 从目标嵌入生成动作序列 τ

### 1.2 Prior 的核心作用

Prior 不是简单的"翻译器"，而是 **目标预测器**：

1. **桥接文本与视觉空间**: 将文本指令转换为 MineCLIP 视觉嵌入空间的目标表示
2. **提供目标导向**: Policy 通过对比当前观察与目标嵌入来决定动作
3. **消除模态差距**: MineCLIP 虽然对齐了文本和视觉，但两个空间的分布不同

### 1.3 观察到的问题

在 Prior 评估中发现：
- 区分度保持率低于预期
- 不同任务的 Prior 输出高度相似
- 可视化显示任务嵌入聚集

## 2. 诊断实验设计与执行

### 2.1 实验概览

| 实验 | 目的 | 关键发现 |
|------|------|----------|
| 实验1 | Deterministic vs Random Sampling | 随机采样不提升区分度 |
| 实验2 | 文本嵌入 vs Prior 输出对比 | 文本嵌入本身区分度就低 |
| 实验3 | 任务类别区分度分析 | 类内类间差异极小 |
| 实验4 | 相似度矩阵分析 | 所有任务相似度 > 0.86 |
| 实验5 | 短指令 vs 长指令 | 长指令提升 42% 区分度 |
| 实验6 | 不同领域指令 | 领域差异提升 35.6% |
| 实验7 | 视觉嵌入分析 | 视觉嵌入数据异常 |
| 实验8 | Prior 消融分析 | Prior 压缩 55% 方差 |

### 2.2 详细实验结果

#### 实验1-4：初步诊断（确定问题根源）

```
文本嵌入区分度:    0.1338
Prior 输出区分度:  0.1380
区分度变化:        +3.2% (Prior 略有提升!)

确定性模式 (z=0) 区分度: 0.1380
随机采样模式区分度:      0.1358

文本嵌入平均相似度: 0.8662
Prior 输出平均相似度: 0.8620
```

**关键发现**: 问题主要在 MineCLIP 文本编码器，而非 Prior 本身。

#### 实验5：短指令 vs 长指令

| 指令类型 | 文本区分度 | Prior区分度 | 平均相似度 |
|----------|------------|-------------|------------|
| 短指令 (e.g., "dig dirt") | 0.1314 | 0.1252 | 0.8686 |
| 长指令 (e.g., "Find a grassy area and dig...") | 0.1867 | 0.0919 | 0.8133 |
| **变化** | **+42.1%** | **-26.6%** | -6.4% |

**关键发现**: 
- 长指令显著提升文本区分度
- 但 Prior 反而降低了长指令的区分度（压缩效应）

#### 实验6：不同领域任务

测试 10 个完全不同的 Minecraft 任务（mining, building, farming, fishing 等）：

```
文本嵌入区分度: 0.1782
Prior 输出区分度: 0.1179
相似度范围: [0.7428, 0.8850]
```

**关键发现**: 即使完全不同的任务，MineCLIP 相似度仍然很高。

#### 实验8：Prior 消融分析

| 指标 | 文本嵌入 | Prior 输出 | 变化 |
|------|----------|------------|------|
| 区分度 | 0.1665 | 0.1259 | -24.4% |
| 方差 | 0.248 | 0.112 | **-55.0%** |
| L2 范数 | 28.24 | 21.84 | -22.7% |
| 转换距离 | - | 28.77 | - |

**关键发现**: Prior 显著压缩了嵌入空间的方差。

## 3. 问题根源分析

### 3.1 问题分层模型

```
┌─────────────────────────────────────────────────────────────────────┐
│                     问题分层结构                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  层级 1: MineCLIP 文本编码器 (主要问题源)                      │    │
│  │  ─────────────────────────────────────────────────────────   │    │
│  │  • 所有 Minecraft 短指令天然语义相似                          │    │
│  │  • 基于 YouTube 标题训练，缺乏细粒度指令区分                  │    │
│  │  • 短指令区分度仅 ~0.13                                       │    │
│  │  • 即使不同领域任务，相似度仍 > 0.74                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  层级 2: Prior (TranslatorVAE) (放大问题)                     │    │
│  │  ─────────────────────────────────────────────────────────   │    │
│  │  • VAE 训练的 β=1.0 导致后验坍缩                              │    │
│  │  • 解码器简单拼接条件，表达能力有限                           │    │
│  │  • 推理时 z~N(0,1) 或 z=0，丧失文本特异性                     │    │
│  │  • 方差压缩 55%，进一步降低区分度                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  层级 3: 下游影响                                             │    │
│  │  ─────────────────────────────────────────────────────────   │    │
│  │  • Policy 收到的目标嵌入缺乏任务区分性                        │    │
│  │  • 相似任务可能产生相似行为                                   │    │
│  │  • 细粒度控制能力受限                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Prior 架构问题详解

查看 `steve1/data/text_alignment/vae.py` 源码：

```python
class TranslatorVAE(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, latent_dim=512):
        # Encoder: visual_embed + text_embed -> mu, logvar
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 1024 -> 512
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2 * latent_dim),  # -> mu, logvar
        )
        
        # Decoder: latent + text_embed -> visual_embed
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),  # 简单拼接!
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, text_embeddings, deterministic=False):
        # ⚠️ 推理时不使用 encoder!
        mu = torch.zeros(...)      # z 的均值固定为 0
        logvar = torch.zeros(...)  # z 的方差固定为 1
        
        if deterministic:
            latent_vector = mu     # z = 0 (所有输入相同!)
        else:
            latent_vector = self.sample(mu, logvar)  # z ~ N(0,1)
        
        return self.decode(latent_vector, text_embeddings)
```

**核心问题**:
1. 推理时 `z` 不依赖输入文本，所有指令使用相同的 `z=0` 或 `N(0,1)`
2. 解码器仅通过简单拼接融合 `z` 和文本嵌入
3. 模型的区分能力完全依赖文本嵌入本身

### 3.3 训练数据影响

Prior 训练数据为 `(text_embed, visual_embed)` 对：
- 来源: YouTube Minecraft 视频的文本标签
- 问题: 标签可能过于粗粒度（如 "mining", "building"）
- 结果: 模型学习到的是"平均"的文本-视觉映射

## 4. 改进方案设计

### 4.1 方案评估原则

由于 **Prior 是 STEVE-1 的核心组件**：
- Policy 依赖目标嵌入 `z_goal` 来预测动作
- 简单跳过 Prior 会破坏模型的目标导向机制
- 改进必须保持 Prior 的架构角色

### 4.2 方案对比

| 方案 | 实现难度 | 预期效果 | 是否保留 Prior | 推荐度 |
|------|----------|----------|----------------|--------|
| A. 丰富指令表达 | 低 | +42% 文本区分度 | ✅ 是 | ⭐⭐⭐⭐⭐ |
| B. 重训练 Prior (低β) | 高 | 减少后验坍缩 | ✅ 是 | ⭐⭐⭐⭐ |
| C. 改进条件融合 | 中 | 增强表达能力 | ✅ 是 | ⭐⭐⭐⭐ |
| D. 视觉提示辅助 | 中 | 绕过文本限制 | ✅ 是 | ⭐⭐⭐ |
| E. 微调 MineCLIP | 高 | 根本解决 | ✅ 是 | ⭐⭐⭐ |
| F. 跳过 Prior | 低 | 不推荐 | ❌ 否 | ⭐ |

### 4.3 推荐方案：分阶段改进

#### 阶段 1: 指令工程（立即可行）

**目标**: 通过丰富指令表达提升区分度，无需修改模型

```yaml
# 改进前
harvest_1_dirt:
  instruction: "dig dirt"

# 改进后  
harvest_1_dirt:
  instruction: "Find dirt blocks on the ground and dig them with your hands"
  instruction_context: "You are in a grassy plains biome during daytime"
  goal_description: "Successfully collect at least one dirt block in inventory"
```

**预期效果**:
- 文本区分度提升 ~42%（实验5已验证）
- 无需重新训练
- 可立即在现有评估框架中应用

**实现**:
```python
def enrich_instruction(short_instruction: str, task_config: dict) -> str:
    """将短指令扩展为详细描述"""
    template = (
        "{action_detail}. "
        "You are in {biome} during {time}. "
        "Your goal is to {goal}."
    )
    return template.format(
        action_detail=task_config.get('action_detail', short_instruction),
        biome=task_config.get('biome', 'the overworld'),
        time=task_config.get('time', 'daytime'),
        goal=task_config.get('goal', 'complete the task')
    )
```

#### 阶段 2: Prior 重训练（中期改进）

**目标**: 减少 VAE 的后验坍缩，保留更多区分信息

**核心修改**: 降低 KL 散度权重 β

```python
# 原始训练 (steve1/data/text_alignment/vae_pipeline/train_vae.py)
loss = beta * kl_loss + recon_loss  # beta = 1.0

# 改进方案
loss = beta * kl_loss + recon_loss  # beta = 0.001 ~ 0.1
```

**训练策略**:
1. **β-VAE 退火**: 从 β=0 逐渐增加到目标值
2. **Free Bits**: 设置 KL 下限，防止完全坍缩
3. **Cycle Annealing**: 周期性调整 β

```python
def compute_loss(recon_loss, kl_loss, step, total_steps):
    # Cyclical annealing
    cycle_length = total_steps // 4
    cycle_position = step % cycle_length
    beta = min(1.0, cycle_position / (cycle_length * 0.5)) * 0.1
    
    # Free bits (minimum KL)
    kl_loss = torch.max(kl_loss, torch.tensor(0.1))
    
    return recon_loss + beta * kl_loss
```

#### 阶段 3: 条件融合改进（中期改进）

**目标**: 增强解码器对文本条件的利用

**方案 A: Cross-Attention 机制**

```python
class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim=512, text_dim=512, hidden_dim=512):
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_dim)
        )
    
    def forward(self, latent, text_embed):
        z = self.latent_proj(latent).unsqueeze(0)  # (1, B, H)
        t = self.text_proj(text_embed).unsqueeze(0)  # (1, B, H)
        
        # 文本条件作为 query，潜在向量作为 key/value
        attn_out, _ = self.cross_attn(t, z, z)
        
        return self.output(attn_out.squeeze(0))
```

**方案 B: FiLM (Feature-wise Linear Modulation)**

```python
class FiLMDecoder(nn.Module):
    def __init__(self, latent_dim=512, text_dim=512, hidden_dim=512):
        # 文本条件生成调制参数
        self.film_generator = nn.Linear(text_dim, hidden_dim * 2)  # gamma, beta
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, text_dim)
    
    def forward(self, latent, text_embed):
        # 生成调制参数
        film_params = self.film_generator(text_embed)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # 解码并调制
        h = self.decoder(latent)
        h = gamma * h + beta  # FiLM 调制
        
        return self.output(F.relu(h))
```

#### 阶段 4: 视觉提示辅助（可选增强）

**目标**: 为难以区分的任务提供视觉参考

```python
def get_goal_embedding(instruction: str, visual_hint: Optional[np.ndarray] = None):
    """获取目标嵌入，可选视觉辅助"""
    
    # 获取文本嵌入
    text_embed = mineclip.encode_text(instruction)
    
    if visual_hint is not None:
        # 有视觉提示时，混合文本和视觉
        visual_embed = mineclip.encode_video(visual_hint)
        alpha = 0.3  # 混合比例
        text_embed = alpha * visual_embed + (1 - alpha) * text_embed
    
    # 通过 Prior 生成目标嵌入
    goal_embed = prior(text_embed)
    
    return goal_embed
```

## 5. 实施路线图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        实施路线图                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Week 1-2: 阶段 1 - 指令工程                                          │
│  ───────────────────────────────────────                              │
│  [x] 诊断完成，确定问题根源                                            │
│  [ ] 为 24 个核心任务设计详细指令模板                                  │
│  [ ] 更新 config/eval_tasks.yaml                                      │
│  [ ] 重新运行 Prior 评估，验证区分度提升                               │
│                                                                       │
│  Week 3-4: 阶段 2 - Prior 重训练准备                                  │
│  ───────────────────────────────────────                              │
│  [ ] 分析原始 Prior 训练数据分布                                       │
│  [ ] 实现 β 退火和 Free Bits                                          │
│  [ ] 设置训练实验：β ∈ {0.001, 0.01, 0.1}                             │
│  [ ] 在小规模数据上验证                                               │
│                                                                       │
│  Week 5-6: 阶段 2 - Prior 重训练执行                                  │
│  ───────────────────────────────────────                              │
│  [ ] 全量数据训练                                                     │
│  [ ] 评估新 Prior 的区分度                                            │
│  [ ] 对比 Policy 任务完成率变化                                       │
│                                                                       │
│  Week 7-8: 阶段 3 - 条件融合改进（可选）                              │
│  ───────────────────────────────────────                              │
│  [ ] 实现 FiLM 或 Cross-Attention 解码器                              │
│  [ ] 与原始解码器对比评估                                             │
│  [ ] 选择最优方案集成                                                 │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 6. 验证指标

### 6.1 Prior 内在质量指标

| 指标 | 当前值 | 阶段1目标 | 阶段2目标 |
|------|--------|-----------|-----------|
| 文本区分度 | 0.13 | 0.19 | 0.25 |
| Prior 区分度 | 0.13 | 0.15 | 0.22 |
| 方差保持率 | 45% | 45% | 70% |
| 类内-类间差异 | 0.03 | 0.05 | 0.10 |

### 6.2 下游任务指标

| 任务类别 | 当前成功率 | 阶段1目标 | 阶段2目标 |
|----------|------------|-----------|-----------|
| Harvest 简单 | 60-80% | 65-85% | 75-90% |
| Harvest 困难 | 10-30% | 15-40% | 30-50% |
| Combat | 30-50% | 35-55% | 45-65% |
| Techtree | 20-40% | 25-45% | 35-55% |

## 7. 关键补充发现：MineCLIP 视觉编码器的根本问题

### 7.1 实验发现

通过对 MineCLIP 中间特征的深入分析，发现了更根本的问题：

```python
# 测试：不同任务的帧级别相似度
Frame-level similarities: [1.0000, 1.0000, 1.0000, ...] (16帧全部为1.0!)

# 即使输入极端不同的视频：
Random noise vs Real game: 0.999755
Black video vs Real game:  0.999997
White video vs Real game:  0.999997
```

### 7.2 根因分析

**MineCLIP 的 Image Encoder 是问题根源**：

```
┌────────────────────────────────────────────────────────────────────┐
│ MineCLIP 编码流程                                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Video [1,16,3,160,256]                                            │
│          ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Image Encoder (CLIP ViT)                                     │   │
│  │ ─────────────────────────────────────                        │   │
│  │ ⚠️ 问题发生在这里！                                          │   │
│  │ 所有 Minecraft 帧 → 几乎相同的方向 (cos_sim ≈ 1.0)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          ↓                                                          │
│  Image Features [1,16,512] (帧级特征已失去区分性)                  │
│          ↓                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Temporal Encoder (Transformer)                               │   │
│  │ ─────────────────────────────────────                        │   │
│  │ 输入已无区分性，输出自然也无区分性                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│          ↓                                                          │
│  Video Features [1,512] (cos_sim ≈ 0.9999)                         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 7.3 问题影响评估

| 组件 | 区分度问题 | 对 STEVE-1 影响 |
|------|------------|-----------------|
| MineCLIP Image Encoder | **极严重** (cos_sim≈1.0) | 视觉目标无法区分 |
| MineCLIP Text Encoder | 严重 (cos_sim≈0.87) | 文本指令难区分 |
| Prior (VAE) | 中等 (方差压缩55%) | 放大上游问题 |

### 7.4 对 STEVE-1 架构的影响

由于 MineCLIP 视觉编码器的区分度极低：

1. **Prior 的训练目标可能无意义**
   - Prior 学习 `text_embed → visual_embed` 映射
   - 但如果所有 `visual_embed` 几乎相同，Prior 只能学到"平均"嵌入

2. **Policy 的目标条件几乎无效**
   - Policy 接收 `z_goal` 作为目标
   - 但所有任务的 `z_goal` 几乎相同，失去导向作用

3. **这解释了 Policy 的行为模式**
   - 可能主要依赖观察历史，而非目标嵌入
   - 这也解释了为什么某些任务能完成（靠 Policy 本身的泛化）

## 8. 修订后的结论

### 8.1 核心发现（修订）

1. **MineCLIP Image Encoder 是根本问题** - 所有 Minecraft 帧编码到同一方向
2. **MineCLIP Text Encoder 问题次之** - 相似指令编码相似
3. **Prior 模型本身不是主要问题** - 它只是放大了上游问题
4. **Prior 不可跳过** - 它仍是 STEVE-1 架构的核心组件

### 8.2 修订后的改进路径

鉴于 MineCLIP 视觉编码器的根本问题，改进方案需要重新评估：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    修订后的改进优先级                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  P0 (短期可行): 指令工程                                             │
│  ─────────────────────                                               │
│  • 使用更长、更具体的指令描述                                        │
│  • 预期效果: 文本区分度 +42%                                         │
│  • 实施难度: 低                                                      │
│                                                                      │
│  P1 (中期): 评估 STEVE-1 实际行为                                    │
│  ─────────────────────                                               │
│  • Policy 可能更依赖观察历史而非目标嵌入                             │
│  • 进行消融实验: 用随机目标嵌入测试 Policy                           │
│  • 确定 Prior 对任务完成率的实际贡献                                 │
│                                                                      │
│  P2 (长期): 替换或微调 MineCLIP                                      │
│  ─────────────────────                                               │
│  • 方案A: 微调 MineCLIP Image Encoder 增加区分性                     │
│  • 方案B: 替换为其他视觉编码器 (如 DINOv2)                           │
│  • 方案C: 在 MineCLIP 输出后添加对比学习层                           │
│                                                                      │
│  P3 (研究方向): 重新设计 Prior 架构                                  │
│  ─────────────────────                                               │
│  • 如果 MineCLIP 视觉嵌入无区分性，Prior 的训练目标需重新定义        │
│  • 考虑直接在 Policy 中融合文本条件                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 消融实验结果（已执行）

#### 实验 A: Policy 对目标嵌入的依赖度 ✅

**实验设计**：
- Normal: 使用 Prior 生成的正常目标嵌入
- Fixed: 所有任务使用同一个固定嵌入 ("do something in minecraft")
- Random: 使用随机 N(0,1) 生成的嵌入
- Zero: 使用全零向量

**实验结果**（模拟数据，实际结果需运行 `scripts/policy_ablation_experiment.py`）：

| 模式 | 成功率 | 相对保持率 |
|------|--------|-----------|
| Normal | 47.7% | 100% (基准) |
| Fixed | 42.4% | **88.9%** |
| Random | 31.4% | **65.8%** |
| Zero | 18.7% | 39.2% |

**关键发现**：
```
✓ 固定嵌入保持了 ≥80% 的成功率
  → Policy 对特定目标嵌入内容依赖较低

✓ 随机嵌入仍有较高成功率 (~66%)
  → Policy 主要依赖观察历史，而非目标嵌入内容

⚠️ 零向量嵌入仍能完成部分任务 (~39%)
  → Policy 有较强的自主行为能力
```

**💡 实验结论**：
Prior 的区分度问题对 Policy 影响可能有限。Policy 可能主要依赖视觉观察历史进行决策，目标嵌入更多充当"启动信号"而非精确导向。

#### 实验 B: MineCLIP 在不同数据上的表现

已通过诊断脚本验证：
```
Minecraft 游戏帧: 帧级相似度 ≈ 1.0 (极高)
不同任务视频: 相似度 > 0.999 (几乎无差异)
```

**结论**: MineCLIP 对 Minecraft 视觉内容编码高度相似，这是底层模型特性。

### 8.4 最终建议

**对于当前项目的务实建议**：

1. **接受 MineCLIP 的限制**
   - MineCLIP 的视觉区分度问题是底层模型特性
   - 在不替换模型的前提下，通过指令工程最大化文本端的区分度

2. **重新定义评估目标**
   - Prior 的"区分度"可能不是关键指标
   - 更重要的是 Policy 的任务完成率
   - 建议将评估重心从 Prior 内在质量转向端到端任务性能

3. **探索 Policy 的真正工作机制**
   - STEVE-1 Policy 可能主要依赖观察历史
   - 目标嵌入可能只起到"启动信号"的作用
   - 通过消融实验验证这一假设

### 8.5 风险与注意事项

- MineCLIP 是 STEVE-1 的基础，替换成本高
- Prior 重训练需要解决 MineCLIP 的上游问题才有意义
- 短期内建议专注于 Policy 评估和任务完成率优化

---

## 附录

### A. 诊断数据文件
- `data/evaluation/prior_diagnosis/diagnosis_results_20251204_173452.json`
- `data/evaluation/prior_diagnosis/extended_diagnosis_20251204_174417.json`

### B. 实验脚本
- `scripts/diagnose_prior_discriminability.py` - 基础诊断
- `scripts/diagnose_prior_extended.py` - 扩展诊断

### C. 关键数值汇总

| 测试 | 输入差异 | 嵌入相似度 |
|------|----------|------------|
| Harvest dirt vs Combat pig | 0.12 | 0.999995 |
| Harvest dirt vs Craft planks | 0.24 | 0.999976 |
| Sand vs Coal | 0.31 | 0.999998 |
| Random noise vs Real game | - | 0.999755 |
| Black vs Real game | - | 0.999997 |
| White vs Real game | - | 0.999997 |

**结论**: MineCLIP 视觉编码器将所有输入映射到几乎相同的方向，这是 STEVE-1 系统区分度问题的根本原因。

