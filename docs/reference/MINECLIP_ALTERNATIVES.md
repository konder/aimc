# MineCLIP替代方案调研

**创建日期**: 2025-11-28  
**目的**: 寻找MineCLIP的替代方案以提高任务区分度

---

## 🎯 问题背景

MineCLIP存在的问题：
- 文本编码器对Minecraft动词区分度低（1.3%）
- 视觉编码器对基础场景相似度高（0.925）
- 导致Prior评估指标虚高

**需求**：寻找能更好区分Minecraft任务的视频-文本模型

---

## 📊 模型分类

### 1️⃣ **通用视频-文本对比学习模型**

#### CLIP4Clip (2021)
- **论文**: [CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval](https://arxiv.org/abs/2104.08860)
- **特点**: 
  - 基于CLIP (ViT-B/32)
  - 专注视频-文本检索
  - 在MSR-VTT、MSVD等基准上SOTA
- **优势**: 
  - ✅ 比原始CLIP更适合视频
  - ✅ 端到端训练
- **劣势**:
  - ⚠️ 仍是通用模型，未针对Minecraft
  - ⚠️ 可能仍有区分度问题
- **GitHub**: https://github.com/ArrowLuo/CLIP4Clip

#### X-CLIP (2022)
- **论文**: [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/abs/2208.02816)
- **特点**:
  - 多粒度对比学习
  - 捕捉视频和文本的细粒度关系
  - Cross-frame communication
- **优势**:
  - ✅ 更好的细粒度理解
  - ✅ 时序建模能力强
- **劣势**:
  - ⚠️ 计算成本高
  - ⚠️ 仍未针对Minecraft
- **GitHub**: https://github.com/microsoft/VideoX/tree/master/X-CLIP

#### VideoCLIP (Microsoft, 2021)
- **特点**:
  - 对比学习 + 时序建模
  - 大规模预训练（180M video-text pairs）
- **优势**:
  - ✅ 训练数据规模大
  - ✅ 时序理解能力强
- **劣势**:
  - ⚠️ 需要大量计算资源
  - ⚠️ 未开源

---

### 2️⃣ **Minecraft特定模型**

#### MineCLIP (MineDojo, 2022)
- **现状**: 我们当前使用的模型
- **训练数据**: 730K YouTube视频
- **问题**: 
  - ❌ 文本区分度1.3%
  - ❌ 视觉相似度0.925
  - ❌ F1仅63.7-95.9%（论文Table A.5）
- **结论**: **不推荐继续使用**（除非fine-tune）

#### VPT (Video Pre-Training, 2022)
- **论文**: [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795)
- **特点**:
  - OpenAI开发
  - 从YouTube视频学习策略
  - 用于Minecraft的IDM (Inverse Dynamics Model)
- **优势**:
  - ✅ 专门为Minecraft设计
  - ✅ 学习动作策略而非仅embedding
- **劣势**:
  - ⚠️ 不是文本-视频对齐模型
  - ⚠️ 主要用于模仿学习，不适合Prior评估
- **GitHub**: https://github.com/openai/Video-Pre-Training

#### STEVE-1 (2023)
- **论文**: [STEVE-1: A Generative Model for Text-to-Behavior in Minecraft](https://arxiv.org/abs/2306.00937)
- **特点**:
  - 基于MineCLIP和VPT
  - 使用Prior VAE做文本到visual embedding映射
  - 这是我们当前的系统基础
- **问题**:
  - ❌ 继承了MineCLIP的所有问题
  - ❌ Prior VAE进一步增加了collapse
- **现状**: **这就是我们当前的系统**

---

### 3️⃣ **最新多模态大模型（2023-2024）**

#### Video-LLaMA (2023)
- **论文**: [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://arxiv.org/abs/2306.02858)
- **特点**:
  - 基于LLaMA的视频理解模型
  - 支持视频问答、描述生成
  - 音频-视觉-文本多模态
- **优势**:
  - ✅ 强大的视频理解能力
  - ✅ 可以fine-tune到Minecraft
  - ✅ 支持自然语言交互
- **劣势**:
  - ⚠️ 模型较大（7B/13B参数）
  - ⚠️ 需要大量GPU资源
  - ⚠️ 需要重新训练Prior
- **GitHub**: https://github.com/DAMO-NLP-SG/Video-LLaMA

#### VideoChat (2023)
- **论文**: [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/abs/2305.06355)
- **特点**:
  - 视频对话系统
  - 基于Vicuna (LLaMA-based)
  - 时空建模 + 指令微调
- **优势**:
  - ✅ 强大的视频理解
  - ✅ 可以处理长视频
- **劣势**:
  - ⚠️ 计算成本高
  - ⚠️ 需要适配Minecraft
- **GitHub**: https://github.com/OpenGVLab/Ask-Anything

#### LLaVA-NeXT (LLaVA-Video, 2024)
- **特点**:
  - 最新的视频-语言模型
  - 基于LLaVA架构扩展到视频
  - 支持多帧理解
- **优势**:
  - ✅ 最新技术
  - ✅ 性能优秀
- **劣势**:
  - ⚠️ 需要适配
- **GitHub**: https://github.com/LLaVA-VL/LLaVA-NeXT

---

### 4️⃣ **通用图像-文本模型（用于单帧）**

#### CLIP (OpenAI, 2021)
- **特点**: 4亿图像-文本对训练
- **优势**: 
  - ✅ 泛化能力强
  - ✅ 开源权重
- **劣势**:
  - ⚠️ 不支持视频序列
  - ⚠️ MineCLIP论文显示CLIP表现更差（Table A.5）
- **适用**: 可以用于单帧评估

#### OpenCLIP (2022)
- **特点**: CLIP的开源复现
- **优势**:
  - ✅ 多种规模（ViT-B/L/H/g）
  - ✅ 可以自己训练
- **GitHub**: https://github.com/mlfoundations/open_clip

#### SigLIP (Google, 2023)
- **论文**: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- **特点**:
  - 改进的loss function
  - 更好的zero-shot性能
- **优势**:
  - ✅ 比CLIP性能更好
  - ✅ 训练效率更高

---

## 🎯 针对我们问题的推荐方案

### 方案A: 短期改进（最简单）

**Fine-tune MineCLIP**

```python
# 思路：在Minecraft动作特定数据上微调
训练数据:
  - 收集不同动作的视频片段
  - 标注明确的动作类别
  - 对比学习loss: 拉开不同动作的距离

优势:
  ✅ 实现简单，基于现有模型
  ✅ 不需要改变架构
  
劣势:
  ⚠️ 需要标注数据
  ⚠️ 改进幅度可能有限
```

**实施步骤**:
1. 收集1000-5000个不同动作的视频片段
2. 标注动作类别（kill, chop, dig, mine, build等）
3. 使用对比学习fine-tune MineCLIP
4. 重新评估Prior

### 方案B: 中期改进（推荐）

**替换为CLIP4Clip + Fine-tune**

```python
# 思路：使用更强的视频-文本模型作为基础
模型: CLIP4Clip (ViT-B/32 或 ViT-L/14)
训练: 
  - 先在Minecraft YouTube数据上预训练
  - 然后在动作特定数据上fine-tune
  
优势:
  ✅ CLIP4Clip时序建模能力更强
  ✅ 开源，容易实现
  ✅ 性能可能明显提升
  
劣势:
  ⚠️ 需要重新训练（1-2周GPU时间）
  ⚠️ 需要重新训练Prior VAE
```

**实施步骤**:
1. 使用CLIP4Clip预训练权重
2. 在MineCLIP的730K YouTube数据上继续训练
3. Fine-tune on action-specific data
4. 重新训练Prior VAE
5. 评估改进效果

### 方案C: 长期改进（最彻底）

**使用Video-LLaMA/VideoChat + 重新设计Prior**

```python
# 思路：使用大模型替代MineCLIP
模型: Video-LLaMA 7B
架构改进:
  - 用Video-LLaMA的video encoder替代MineCLIP
  - Prior VAE输入改为Video-LLaMA的visual tokens
  - 可选: 用language model直接生成z_goal
  
优势:
  ✅ 最强的视频理解能力
  ✅ 可以处理复杂指令
  ✅ 未来可扩展性好
  
劣势:
  ⚠️ 需要大量GPU资源（A100 x 8）
  ⚠️ 训练时间长（数周）
  ⚠️ 架构改动大
```

### 方案D: 混合方案（实用）

**分层评估：简单任务用MineCLIP，复杂任务用人工/LLM**

```python
# 思路：根据任务复杂度选择评估方式
简单任务 (harvest, combat):
  - 继续使用MineCLIP
  - 接受高相似度的局限
  
复杂任务 (build, craft):
  - 使用Video-LLaMA生成描述
  - 用GPT-4评估成功与否
  - 或人工评估
  
优势:
  ✅ 无需重新训练
  ✅ 充分利用现有资源
  ✅ 灵活可扩展
```

---

## 📊 方案对比

| 方案 | 实施难度 | 时间成本 | GPU成本 | 预期改进 | 推荐度 |
|------|---------|---------|---------|---------|--------|
| **A. Fine-tune MineCLIP** | ⭐⭐ | 1-2周 | 中 | +10-20% | ⭐⭐⭐ |
| **B. CLIP4Clip** | ⭐⭐⭐ | 2-4周 | 高 | +30-50% | ⭐⭐⭐⭐ |
| **C. Video-LLaMA** | ⭐⭐⭐⭐⭐ | 1-2月 | 极高 | +50-80% | ⭐⭐⭐⭐⭐ |
| **D. 混合方案** | ⭐ | 1周 | 极低 | 灵活 | ⭐⭐⭐⭐ |

---

## 🚀 立即可行的行动

### 1. **快速验证（本周）**

测试OpenCLIP/SigLIP是否比MineCLIP更好：

```bash
# 安装OpenCLIP
pip install open_clip_torch

# 测试脚本
python scripts/test_openclip_similarity.py \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k
```

### 2. **Fine-tune MineCLIP（2周）**

收集动作特定数据并微调：

```python
# 数据收集
actions = ['kill_pig', 'chop_tree', 'dig_dirt', 'mine_coal', 'build_house']
for action in actions:
    collect_youtube_clips(action, num_clips=200)

# Fine-tune
train_contrastive(
    model=mineclip,
    data=action_clips,
    loss='triplet',  # 拉开不同动作距离
    epochs=10
)
```

### 3. **尝试CLIP4Clip（1个月）**

下载和适配：

```bash
git clone https://github.com/ArrowLuo/CLIP4Clip
cd CLIP4Clip

# 使用MineCLIP的YouTube数据训练
python train.py \
    --data_path /path/to/mineclip/youtube \
    --pretrained_clip_name ViT-B/32
```

---

## 📚 相关资源

### GitHub仓库
- CLIP4Clip: https://github.com/ArrowLuo/CLIP4Clip
- X-CLIP: https://github.com/microsoft/VideoX/tree/master/X-CLIP
- Video-LLaMA: https://github.com/DAMO-NLP-SG/Video-LLaMA
- OpenCLIP: https://github.com/mlfoundations/open_clip
- VideoChat: https://github.com/OpenGVLab/Ask-Anything

### 论文
- CLIP4Clip: https://arxiv.org/abs/2104.08860
- X-CLIP: https://arxiv.org/abs/2208.02816
- Video-LLaMA: https://arxiv.org/abs/2306.02858
- VPT: https://arxiv.org/abs/2206.11795
- STEVE-1: https://arxiv.org/abs/2306.00937

### Hugging Face Models
- CLIP4Clip: https://huggingface.co/ArrowLuo/CLIP4Clip
- OpenCLIP: https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K

---

## ✅ 总结与建议

### 当前状态
- ❌ MineCLIP区分度太低（文本1.3%，视觉0.925）
- ❌ 导致Prior评估指标虚高
- ✅ 但我们的使用和评估都正确

### 短期建议（1-2周）
1. **测试OpenCLIP/SigLIP**：快速验证是否有改进
2. **实施方案D**：对复杂任务使用GPT-4评估
3. **收集动作数据**：为fine-tune做准备

### 中期建议（1-2月）
1. **实施方案B**：采用CLIP4Clip + Fine-tune
2. **重新训练Prior VAE**：在更好的visual space中
3. **全面评估**：对比MineCLIP vs CLIP4Clip

### 长期建议（3-6月）
1. **考虑方案C**：如果有充足GPU资源
2. **探索Video-LLaMA**：用于更复杂的任务理解
3. **建立benchmark**：标准化Minecraft任务评估

---

**最终推荐**：
- 🥇 **首选**: 方案B (CLIP4Clip + Fine-tune) - 性价比最高
- 🥈 **备选**: 方案D (混合方案) - 快速可行
- 🥉 **探索**: 方案C (Video-LLaMA) - 未来方向

