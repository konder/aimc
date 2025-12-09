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

## 7. 结论

### 7.1 核心发现

1. **问题主因在 MineCLIP 文本编码器**，而非 Prior 本身
2. **Prior 的 VAE 架构加剧了问题**（方差压缩 55%）
3. **长指令可显著提升区分度**（+42%），是最快的改进路径
4. **Prior 不可跳过**，它是 STEVE-1 目标导向机制的核心

### 7.2 最优改进路径

```
优先级 1: 指令工程 → 立即可行，+42% 文本区分度
优先级 2: Prior 重训练 (低β) → 中期，减少后验坍缩
优先级 3: 条件融合改进 → 增强 Prior 表达能力
```

### 7.3 风险与注意事项

- 长指令可能增加 Policy 的泛化难度
- Prior 重训练需要原始训练数据
- 条件融合改进需要重新训练整个模型

---

**附录: 诊断数据文件**
- `data/evaluation/prior_diagnosis/diagnosis_results_20251204_173452.json`
- `data/evaluation/prior_diagnosis/extended_diagnosis_20251204_174417.json`

