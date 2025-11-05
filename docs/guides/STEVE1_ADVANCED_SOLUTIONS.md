# STEVE-1 进阶问题解决方案

> **问题1**: 如何支持中文语义？  
> **问题2**: 如何从网络视频生成训练数据？

---

## 问题1: 支持中文语义

### 1.1 当前问题

```
MineCLIP只支持英文:
  text("chop tree") ✅ → embed [0.23, -0.11, ...]
  text("砍树")      ❌ → embed [random...] 无法正确映射
```

### 1.2 解决方案

#### 方案A: 多语言MineCLIP（推荐）⭐

**核心思路**：在MineCLIP上添加多语言文本编码器

```python
# 1. 使用多语言CLIP模型作为基础
# 例如：multilingual-clip, Chinese-CLIP

from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# 2. 创建多语言MineCLIP适配器
class MultilingualMineCLIP:
    def __init__(self):
        # 原始MineCLIP（视觉编码器）
        self.visual_encoder = load_mineclip_visual()
        
        # 多语言文本编码器
        self.chinese_text_encoder = ChineseCLIPModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16"
        )
        
        # 对齐层：将Chinese-CLIP输出映射到MineCLIP空间
        self.alignment_layer = nn.Linear(512, 512)
    
    def encode_text(self, text, language='zh'):
        if language == 'zh':
            # 中文编码
            embed = self.chinese_text_encoder.encode_text(text)
            # 对齐到MineCLIP空间
            embed = self.alignment_layer(embed)
        elif language == 'en':
            # 英文编码（使用原始MineCLIP）
            embed = self.mineclip_text_encoder.encode_text(text)
        
        return embed
    
    def encode_visual(self, image):
        # 视觉编码保持不变
        return self.visual_encoder(image)
```

**训练对齐层**：

```python
# 准备中英文对照数据
pairs = [
    ("砍树", "chop tree"),
    ("挖矿", "mine ore"),
    ("建造", "build"),
    ...
]

# 训练目标：让中英文的嵌入接近
for zh_text, en_text in pairs:
    zh_embed = model.encode_text(zh_text, language='zh')
    en_embed = model.encode_text(en_text, language='en')
    
    # 对齐损失
    loss = F.mse_loss(zh_embed, en_embed)
    loss.backward()
    optimizer.step()
```

#### 方案B: 机器翻译桥接

**适用场景**：快速原型，不需要重新训练

```python
from transformers import MarianMTModel, MarianTokenizer

class TranslationBridge:
    def __init__(self):
        # 加载翻译模型
        self.translator = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-zh-en"
        )
        self.tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-zh-en"
        )
        
        # 原始MineCLIP
        self.mineclip = load_mineclip()
    
    def encode_chinese_text(self, chinese_text):
        # 1. 中译英
        translated = self.translate_zh_to_en(chinese_text)
        
        # 2. 用MineCLIP编码英文
        embed = self.mineclip.encode_text(translated)
        
        return embed
    
    def translate_zh_to_en(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.translator.generate(**inputs)
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

# 使用
bridge = TranslationBridge()

# 推理时
chinese_prompt = "砍树"
embed = bridge.encode_chinese_text(chinese_prompt)  # 翻译 → MineCLIP
action = policy(current_frame, embed)
```

**优缺点**：

```
优点:
  ✅ 无需重新训练MineCLIP
  ✅ 快速实现
  ✅ 可以立即使用

缺点:
  ❌ 依赖翻译质量
  ❌ Minecraft专业术语可能翻译不准
  ❌ 增加推理延迟
```

#### 方案C: 从头训练中文MineCLIP

**适用场景**：有足够的中文Minecraft数据和计算资源

```python
# 数据准备
chinese_minecraft_data = [
    {
        'video': 'bilibili_video_1.mp4',
        'title': '我的世界：如何快速砍树',
        'comments': ['砍树技巧', '木头收集', ...]
    },
    ...
]

# 训练Chinese-MineCLIP
for video, texts in chinese_minecraft_data:
    video_frames = extract_frames(video)
    video_embed = visual_encoder(video_frames)
    
    for text in texts:
        text_embed = chinese_text_encoder(text)
        
        # 对比学习损失
        loss = contrastive_loss(video_embed, text_embed)
        loss.backward()
```

**所需资源**：
- 数据：10万+ 中文Minecraft视频（B站、抖音）
- 计算：多GPU训练数周
- 标注：视频-文本对

### 1.3 推荐方案对比

| 方案 | 难度 | 效果 | 时间 | 成本 |
|------|------|------|------|------|
| 方案A: 多语言适配 | ⭐⭐ | ⭐⭐⭐⭐ | 数天 | 低 |
| 方案B: 机器翻译 | ⭐ | ⭐⭐⭐ | 数小时 | 极低 |
| 方案C: 从头训练 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 数周 | 高 |

**建议**：先用方案B快速验证，如果需要更好效果再实施方案A。

---

## 问题2: 从网络视频生成训练数据

### 2.1 核心问题

```
缺少专家演示数据，但有:
  ✅ 大量网络Minecraft视频（YouTube、B站）
  ❌ 没有对应的动作序列

需要: 从视频帧反推动作
```

### 2.2 这个想法可行吗？

**答案：可行，这正是VPT的做法！** ✅

VPT（Video Pre-Training）就是用这个方法：

```
VPT方法:
  1. 收集YouTube Minecraft视频
  2. 训练逆向动力学模型（IDM）
  3. 从视频帧预测动作
  4. 用预测的动作训练策略
```

### 2.3 完整实现方案

#### 步骤1: 收集网络视频

```python
# 1. 下载视频
import yt_dlp

def download_minecraft_videos(urls, output_dir):
    """下载Minecraft游戏视频"""
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            ydl.download([url])

# 2. 筛选高质量视频
def filter_quality_videos(video_path):
    """筛选标准"""
    checks = {
        'resolution': check_resolution(video_path) >= 720,  # ≥720p
        'fps': check_fps(video_path) >= 20,  # ≥20fps
        'gameplay': is_actual_gameplay(video_path),  # 实际游戏而非解说
        'pov': is_first_person(video_path),  # 第一人称视角
    }
    return all(checks.values())
```

#### 步骤2: 训练逆向动力学模型（IDM）

**IDM的作用**：从连续帧预测中间的动作

```python
# IDM模型架构
class InverseDynamicsModel(nn.Module):
    """
    输入: frame[t], frame[t+1]
    输出: action[t]
    """
    def __init__(self):
        super().__init__()
        # 图像编码器
        self.encoder = ImpalaCNN(outsize=256)
        
        # 动作预测器
        self.action_head = nn.Sequential(
            nn.Linear(256 * 2, 512),  # 拼接两帧特征
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, frame_t, frame_t1):
        # 编码两帧
        feat_t = self.encoder(frame_t)
        feat_t1 = self.encoder(frame_t1)
        
        # 拼接并预测动作
        feat = torch.cat([feat_t, feat_t1], dim=1)
        action_logits = self.action_head(feat)
        
        return action_logits
```

**训练IDM**：

```python
# 需要少量有标注的数据来训练IDM
# 可以使用VPT公开的IDM模型，或自己标注少量数据

# 方法1: 使用VPT预训练的IDM（推荐）
idm = load_vpt_idm()

# 方法2: 自己训练（需要标注数据）
# 标注1000-5000个视频片段的动作
for (frame_t, frame_t1), action_label in labeled_dataset:
    pred_action = idm(frame_t, frame_t1)
    loss = F.cross_entropy(pred_action, action_label)
    loss.backward()
```

#### 步骤3: 从视频预测动作序列

```python
def video_to_training_data(video_path, idm_model, output_dir):
    """
    将视频转换为训练数据
    
    输入: 视频文件
    输出: frames/ + actions.jsonl + embeds_attn.pkl
    """
    # 1. 提取视频帧
    frames = extract_frames(video_path, target_size=(128, 128))
    
    # 2. 用IDM预测动作
    actions = []
    for t in range(len(frames) - 1):
        frame_t = frames[t]
        frame_t1 = frames[t + 1]
        
        # 预测动作
        with torch.no_grad():
            action_logits = idm_model(frame_t, frame_t1)
            action = sample_action(action_logits)
        
        actions.append(action)
    
    # 3. 用MineCLIP编码每一帧
    mineclip = load_mineclip()
    embeds = []
    for frame in frames:
        embed = mineclip.encode_image(frame)
        embeds.append(embed)
    
    # 4. 保存为STEVE-1格式
    episode_dir = f"{output_dir}/episode_{video_id}/"
    os.makedirs(f"{episode_dir}/frames/", exist_ok=True)
    
    # 保存帧
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{episode_dir}/frames/{i:05d}.png", frame)
    
    # 保存动作
    with open(f"{episode_dir}/actions.jsonl", 'w') as f:
        for action in actions:
            f.write(json.dumps(action) + '\n')
    
    # 保存嵌入
    np.save(f"{episode_dir}/embeds_attn.pkl", np.array(embeds))
    
    return episode_dir
```

#### 步骤4: 质量控制

```python
def filter_low_quality_predictions(episode_dir):
    """过滤低质量的预测数据"""
    
    # 1. 检查动作连贯性
    actions = load_actions(episode_dir)
    
    # 异常检测：动作变化太频繁
    action_changes = count_action_changes(actions)
    if action_changes / len(actions) > 0.5:  # 超过50%帧动作都变化
        return False
    
    # 2. 检查物理合理性
    # 例如：不能在空中突然改变方向
    if not check_physics(actions):
        return False
    
    # 3. 检查视觉连贯性
    frames = load_frames(episode_dir)
    if not check_visual_continuity(frames):
        return False
    
    return True
```

### 2.4 完整工作流程

```bash
# 1. 下载视频
python download_videos.py \
    --source youtube \
    --keywords "minecraft gameplay" \
    --min_duration 300 \  # 至少5分钟
    --output_dir data/raw_videos/

# 2. 质量筛选
python filter_videos.py \
    --input_dir data/raw_videos/ \
    --output_dir data/filtered_videos/ \
    --min_resolution 720

# 3. 用IDM预测动作
python predict_actions.py \
    --idm_weights data/weights/vpt/inverse_dynamics.weights \
    --video_dir data/filtered_videos/ \
    --output_dir data/dataset_web/

# 4. 质量检查
python validate_predictions.py \
    --dataset_dir data/dataset_web/ \
    --remove_low_quality

# 5. 训练STEVE-1
cd src/training/steve1
bash 3_train.sh \
    --sampling web_videos \
    --sampling_dir ../../data/samplings/
```

### 2.5 IDM预测的准确性

```
IDM预测准确性（VPT论文）:
  ✅ 移动方向: >90% 准确
  ✅ 摄像机转向: >85% 准确
  ✅ 跳跃、攻击: >80% 准确
  ❌ 精细操作: ~60-70% 准确

建议:
  1. 先用IDM生成初步数据
  2. 训练基础策略
  3. 如果效果不佳，再收集少量高质量人工演示
  4. 混合训练：70% IDM数据 + 30% 人工数据
```

### 2.6 实用技巧

#### 技巧1: 使用VPT的预训练IDM

```bash
# VPT提供了预训练的IDM模型
# 下载地址: https://github.com/openai/Video-Pre-Training

# 使用方法
from vpt import load_inverse_dynamics_model

idm = load_inverse_dynamics_model(
    "data/weights/vpt/inverse_dynamics.weights"
)
```

#### 技巧2: 混合数据训练

```python
# 结合IDM预测数据和少量人工数据
dataset = CombinedDataset(
    idm_data_dir='data/dataset_web/',      # IDM预测（大量）
    human_data_dir='data/dataset_human/',  # 人工演示（少量）
    mix_ratio=0.7  # 70% IDM + 30% 人工
)
```

#### 技巧3: 渐进式训练

```python
# 阶段1: 用IDM数据预训练
train(dataset='idm_only', epochs=10)

# 阶段2: 加入少量人工数据微调
train(dataset='mixed', epochs=5, learning_rate=1e-5)

# 阶段3: 可选RL进一步优化
rl_finetune(base_policy='bc_pretrained', environment='minedojo')
```

---

## 3. 实施建议

### 3.1 中文支持优先级

```
第一阶段（快速验证）:
  ✅ 实施方案B（机器翻译）
  ✅ 验证系统可用性
  时间: 1-2天

第二阶段（性能优化）:
  ✅ 收集中英文对照数据
  ✅ 训练方案A（对齐层）
  时间: 1-2周

第三阶段（长期优化）:
  ✅ 收集中文视频数据
  ✅ 训练Chinese-MineCLIP
  时间: 1-2月
```

### 3.2 数据生成优先级

```
第一阶段（快速原型）:
  ✅ 使用VPT预训练IDM
  ✅ 处理100-1000个视频
  ✅ 训练基础策略
  时间: 1-2周

第二阶段（质量提升）:
  ✅ 筛选高质量预测
  ✅ 收集少量人工演示
  ✅ 混合训练
  时间: 2-4周

第三阶段（性能优化）:
  ✅ 扩大数据规模
  ✅ RL微调
  ✅ 任务特定优化
  时间: 持续改进
```

---

## 4. 代码实现参考

```bash
# 项目中已有的相关代码
src/training/steve1/
├─ data/generation/
│  ├─ convert_from_contractor.py  # VPT数据转换（参考）
│  └─ vpt_agents.py                # VPT agent使用
└─ VPT/
   └─ inverse_dynamics_model.py    # IDM实现

# 新建实现
tools/
├─ chinese_mineclip/
│  ├─ translation_bridge.py        # 方案B实现
│  └─ multilingual_adapter.py      # 方案A实现
└─ video_to_data/
   ├─ download_videos.py           # 视频下载
   ├─ predict_actions.py           # IDM预测
   └─ validate_quality.py          # 质量检查
```

---

## 5. 总结

### 问题1答案：中文支持

**可行性**: ✅ 完全可行

**推荐方案**:
1. 短期：机器翻译桥接（快速实现）
2. 中期：多语言MineCLIP适配（平衡效果和成本）
3. 长期：训练Chinese-MineCLIP（最佳效果）

### 问题2答案：从视频生成数据

**可行性**: ✅ 完全可行（VPT已验证）

**关键步骤**:
1. 使用预训练IDM模型
2. 从网络视频预测动作
3. 质量控制和筛选
4. 混合人工数据提升质量

**预期效果**:
- IDM数据可以训练出基础能力
- 结合少量人工数据效果更好
- 成本远低于纯人工标注

---

**相关文档**:
- VPT论文: https://arxiv.org/abs/2206.11795
- VPT代码: https://github.com/openai/Video-Pre-Training
- Chinese-CLIP: https://github.com/OFA-Sys/Chinese-CLIP

**最后更新**: 2025-11-05

