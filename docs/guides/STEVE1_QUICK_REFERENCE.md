# STEVE-1 快速参考卡片

## 🎯 核心公式

### 训练时
```
输入 x = {
    图像: 当前帧 [128×128×3]
    条件: 未来帧的MineCLIP视觉嵌入 [512维]
}

输出 f(x) = 动作分布 {
    buttons: [8641维] 离散按钮
    camera: [121维] 摄像机
}

损失 L = -log P(专家动作 | x)
```

### 推理时
```
输入 x = {
    图像: 当前帧 [128×128×3]
    条件: 文本提示的MineCLIP文本嵌入 [512维]
}

输出 f(x) = 动作分布 → 采样 → 执行
```

## 📊 数据流总览

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: 原始数据 (VPT数据集)                                  │
│   • 人类玩家录像 (无文本标注)                                │
│   • 每帧的游戏画面 + 对应动作                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段2: MineCLIP编码                                         │
│   frames[t] → MineCLIP视觉编码器 → embeds[t] [512维]       │
│   作用: 将每帧画面编码为语义向量                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段3: 事后重标记 (Hindsight Relabeling)                    │
│   随机采样未来帧作为"目标":                                  │
│   t=100 → 目标=t=150 (用embeds[150]作为条件)                │
│   构造训练样本: (frame[100], embeds[150]) → action[100]     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段4: 行为克隆训练                                          │
│   模型学习: P(动作 | 当前画面, 目标嵌入)                     │
│   优化: 最大化专家动作的对数似然                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段5: 推理 (跨模态迁移)                                     │
│   文本提示 → MineCLIP文本编码器 → text_embed [512维]        │
│   替换视觉嵌入为文本嵌入 → 生成动作                          │
│   有效原因: MineCLIP将文本和图像映射到同一空间               │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 关键技术点

### 1. MineCLIP的作用
- **训练阶段**: 在YouTube视频上用对比学习预训练
- **编码能力**: 
  - 文本 "chop tree" → [0.23, -0.11, ...] (512维)
  - 图像 [砍树画面] → [0.21, -0.09, ...] (512维，相近！)
- **关键属性**: 文本和视觉在同一语义空间

### 2. 事后重标记 (Hindsight Relabeling)
**为什么需要？** 
- VPT数据集只有录像，没有文本标签
- 如何训练"根据文本执行动作"的模型？

**解决方案**：
```python
# 将未来帧视为"隐含目标"
for t in range(total_timesteps):
    current_frame = frames[t]
    future_goal_embed = embeds[random_future_t]  # 随机15-200步后
    current_action = actions[t]
    
    # 训练样本: "为了达到future目标，在current时刻应执行current_action"
    train_data.append((current_frame, future_goal_embed, current_action))
```

### 3. Classifier-Free Guidance (CFG)
**训练时**: 10%概率将条件设为零向量
```python
if random() < 0.1:
    embeds = zeros(512)  # 无条件训练
```

**推理时**: 增强条件影响
```python
logits = (1 + scale) * logits_cond - scale * logits_uncond
#      = 7.0 * 有条件预测 - 6.0 * 无条件预测
# 效果: 让模型更"听"文本提示的话
```

## 📁 数据格式速查

### Episode存储结构
```
data/dataset_contractor/episode_0001/
├── frames/
│   ├── 00000.png          # 128×128×3, uint8
│   ├── 00001.png
│   └── ...
├── actions.jsonl          # VPT格式动作
│   {"camera": [0.5, -0.2], "buttons": {"forward": 1, "attack": 0, ...}}
└── embeds_attn.pkl        # [T, 512] float32
```

### 训练Batch形状
```python
obs = {
    'img': [B=12, T=640, 128, 128, 3],      # 批量×时间×高×宽×通道
    'mineclip_embed': [B=12, T=640, 512]    # 批量×时间×嵌入维度
}
actions = {
    'buttons': [B=12, T=640, 8641],         # 离散动作 (独热编码)
    'camera': [B=12, T=640, 121]            # 摄像机 (分层编码)
}
firsts = [B=12, T=640]                      # bool, True=序列开始
```

## 🛠️ 关键代码位置

| 功能 | 文件 | 行数 |
|------|------|------|
| 事后重标记 | `data/minecraft_dataset.py` | 58-124 |
| 条件策略网络 | `embed_conditioned_policy.py` | 91-210 |
| 嵌入融合 | `embed_conditioned_policy.py` | 199 |
| 训练主循环 | `training/train.py` | 全文 |
| MineCLIP加载 | `mineclip_code/load_mineclip.py` | 全文 |

## 🎓 理解检查清单

- [ ] 理解MineCLIP将文本和图像映射到同一空间
- [ ] 理解事后重标记如何生成"目标条件"数据
- [ ] 理解训练用视觉嵌入，推理用文本嵌入
- [ ] 理解CFG增强条件控制的机制
- [ ] 理解为什么不需要人工文本标注
- [ ] 知道如何准备自己的训练数据
- [ ] 能够解释 x 和 f(x) 的含义

## 📚 相关文档

- **完整原理**: `STEVE1_TRAINING_EXPLAINED.md` ← 深入理解推荐
- **微调指南**: `STEVE1_FINETUNING_QUICKSTART.md`
- **脚本使用**: `STEVE1_SCRIPTS_USAGE_GUIDE.md`
- **评估方法**: `STEVE1_EVALUATION_GUIDE.md`
- **训练分析**: `../technical/STEVE1_TRAINING_ANALYSIS.md`

## 🚀 快速开始

### 1. 检查数据结构
```bash
python scripts/inspect_steve1_data.py
```

### 2. 生成测试视频
```bash
cd src/training/steve1
bash 2_gen_vid_for_text_prompt.sh
```

### 3. 准备训练数据
```bash
bash 1_generate_dataset.sh  # 下载VPT数据
bash 2_create_sampling.sh   # 创建采样配置
```

### 4. 启动训练
```bash
bash 3_train.sh             # 从头训练
# 或
bash 3_train_finetune_template.sh  # 微调
```

---

**提示**: 运行 `python scripts/inspect_steve1_data.py` 可以查看实际数据的形状和内容！

