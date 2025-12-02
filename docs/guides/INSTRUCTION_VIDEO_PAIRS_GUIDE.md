# 指令->视频对数据集使用指南

## 📋 概述

本指南介绍如何从评估结果中提取和使用"指令->视频对"数据集。

---

## 🎯 数据格式

### 配对格式（Pair Format）

每个配对包含一个指令和对应的视觉嵌入：

```python
{
    'instruction': 'chop tree',              # 指令文本
    'visual_embed': np.ndarray,              # 视觉嵌入 [512]
    'task_id': 'harvest_1_log',              # 任务ID
    'trial_idx': 0,                          # Trial索引
    'source': 'eval_result',                 # 数据源
    'metadata': {                            # 元数据
        'embed_dim': 512,
        'n_frames': 16,
        'frame_paths': [...],
        'reward_frame_idx': 45,
        'total_frames': 120,
        'trial_success': True,
        'trial_reward': 1.0,
    }
}
```

---

## 🔧 提取数据

### 方法1: 使用Python API

```python
from pathlib import Path
from src.utils.success_visual_extractor import create_extractor

# 创建提取器
extractor = create_extractor(
    source_type='eval_result',
    last_n_frames=16,
    use_reward_moment=True
)

# 提取配对数据
eval_result_dir = Path('results/evaluation/all_tasks_20251121_214545')
pairs = extractor.extract_pairs(eval_result_dir)

# 查看数据
print(f"提取了 {len(pairs)} 个指令->视频对")
for pair in pairs[:3]:
    print(f"{pair['instruction']} -> {pair['visual_embed'].shape}")
```

**输出**:
```
提取了 42 个指令->视频对
chop tree -> (512,)
chop tree -> (512,)
dig dirt -> (512,)
```

### 方法2: 使用导出脚本

#### 基本使用

```bash
# 导出为JSON格式
python scripts/export_instruction_video_pairs.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/datasets/instruction_video_pairs
```

#### 导出多种格式

```bash
# 同时导出JSON、CSV、NPZ格式
python scripts/export_instruction_video_pairs.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/datasets/pairs \
    --formats json csv npz
```

**生成的文件**:
- `pairs.json` - 包含完整元数据（可视化、调试）
- `pairs.csv` - 表格格式（不含嵌入向量，便于查看）
- `pairs.npz` - Numpy压缩格式（训练、分析）

---

## 📊 数据分析

### 加载JSON数据

```python
import json
import numpy as np

with open('results/datasets/pairs.json', 'r') as f:
    pairs = json.load(f)

# 转换回numpy数组
for pair in pairs:
    pair['visual_embed'] = np.array(pair['visual_embed'])

print(f"任务数: {len(set(p['task_id'] for p in pairs))}")
print(f"指令数: {len(set(p['instruction'] for p in pairs))}")
```

### 加载NPZ数据

```python
import numpy as np

# 加载
data = np.load('results/datasets/pairs.npz', allow_pickle=True)

instructions = data['instructions']
visual_embeds = data['visual_embeds']
task_ids = data['task_ids']
trial_indices = data['trial_indices']

print(f"指令: {instructions.shape}")
print(f"嵌入: {visual_embeds.shape}")
print(f"示例: {instructions[0]} -> {visual_embeds[0].shape}")
```

### 统计分析

```python
import pandas as pd

# 加载CSV
df = pd.read_csv('results/datasets/pairs.csv')

# 按任务统计
task_counts = df['task_id'].value_counts()
print("每个任务的数据量:")
print(task_counts)

# 按指令统计
instruction_counts = df['instruction'].value_counts()
print("\n每个指令的数据量:")
print(instruction_counts)

# 可视化
import matplotlib.pyplot as plt
task_counts.plot(kind='bar', figsize=(12, 6))
plt.title('每个任务的指令->视频对数量')
plt.xlabel('任务ID')
plt.ylabel('数量')
plt.tight_layout()
plt.savefig('task_distribution.png')
```

---

## 🔍 数据探索

### 查找特定任务的数据

```python
# 查找所有"砍树"相关的数据
chop_tree_pairs = [p for p in pairs if 'chop' in p['instruction'].lower()]
print(f"砍树相关: {len(chop_tree_pairs)} 对")

# 查找特定任务
harvest_log_pairs = [p for p in pairs if p['task_id'] == 'harvest_1_log']
print(f"harvest_1_log: {len(harvest_log_pairs)} 对")
```

### 计算嵌入相似度

```python
from scipy.spatial.distance import cosine

# 比较两个指令的视觉嵌入相似度
pair1 = pairs[0]
pair2 = pairs[1]

similarity = 1 - cosine(pair1['visual_embed'], pair2['visual_embed'])
print(f"{pair1['instruction']} vs {pair2['instruction']}: {similarity:.4f}")
```

### 可视化嵌入空间

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 收集所有嵌入
embeds = np.stack([p['visual_embed'] for p in pairs])
labels = [p['task_id'] for p in pairs]

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
embeds_2d = tsne.fit_transform(embeds)

# 可视化
plt.figure(figsize=(12, 8))
for task_id in set(labels):
    mask = np.array(labels) == task_id
    plt.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], label=task_id, alpha=0.6)
plt.legend()
plt.title('指令->视频嵌入空间 (t-SNE)')
plt.savefig('embedding_space.png')
```

---

## 🚀 训练应用

### 1. 文本-视觉对比学习

```python
import torch
import torch.nn as nn

class TextVisualContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, text_embeds, visual_embeds):
        # 归一化
        text_embeds = F.normalize(text_embeds, dim=-1)
        visual_embeds = F.normalize(visual_embeds, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(text_embeds, visual_embeds.t()) / self.temperature
        
        # 对比学习损失
        labels = torch.arange(len(logits), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss

# 使用数据
for batch in dataloader:
    instructions = batch['instructions']
    visual_embeds = batch['visual_embeds']
    
    # 前向传播
    text_embeds = text_encoder(instructions)
    loss = contrastive_loss(text_embeds, visual_embeds)
    
    # 反向传播
    loss.backward()
    optimizer.step()
```

### 2. 指令检索

```python
def retrieve_visual(query_instruction, pairs, top_k=5):
    """根据指令检索最相似的视觉嵌入"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 编码查询指令（假设已有编码器）
    query_embed = encode_instruction(query_instruction)
    
    # 计算与所有视觉嵌入的相似度
    visual_embeds = np.stack([p['visual_embed'] for p in pairs])
    similarities = cosine_similarity([query_embed], visual_embeds)[0]
    
    # 返回Top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'pair': pairs[idx],
            'similarity': similarities[idx]
        })
    
    return results

# 使用
results = retrieve_visual("cut down tree", pairs, top_k=3)
for r in results:
    print(f"{r['pair']['instruction']} (相似度: {r['similarity']:.4f})")
```

---

## 📝 数据格式转换

### 转换为PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class InstructionVideoDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            'instruction': pair['instruction'],
            'visual_embed': torch.from_numpy(pair['visual_embed']).float(),
            'task_id': pair['task_id'],
        }

# 使用
dataset = InstructionVideoDataset(pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(batch['instruction'])
    print(batch['visual_embed'].shape)
    break
```

### 转换为HuggingFace Dataset

```python
from datasets import Dataset

# 准备数据
data_dict = {
    'instruction': [p['instruction'] for p in pairs],
    'visual_embed': [p['visual_embed'].tolist() for p in pairs],
    'task_id': [p['task_id'] for p in pairs],
}

# 创建Dataset
hf_dataset = Dataset.from_dict(data_dict)

# 保存
hf_dataset.save_to_disk('results/datasets/hf_instruction_video_pairs')

# 加载
from datasets import load_from_disk
loaded_dataset = load_from_disk('results/datasets/hf_instruction_video_pairs')
```

---

## ⚙️ 高级选项

### 自定义帧提取

```python
# 使用最后N帧（不使用奖励时刻）
extractor = create_extractor(
    source_type='eval_result',
    last_n_frames=16,
    use_reward_moment=False  # 使用最后16帧
)

# 提取更长的视频片段
extractor = create_extractor(
    source_type='eval_result',
    last_n_frames=32,  # 提取32帧
    use_reward_moment=True
)
```

### 过滤数据

```python
# 只保留特定任务
filtered_pairs = [p for p in pairs if p['task_id'].startswith('harvest')]

# 只保留高奖励的trial
high_reward_pairs = [
    p for p in pairs 
    if p['metadata'].get('trial_reward', 0) > 0.8
]

# 去重（相同指令只保留一个）
seen = set()
unique_pairs = []
for p in pairs:
    if p['instruction'] not in seen:
        seen.add(p['instruction'])
        unique_pairs.append(p)
```

---

## 🐛 常见问题

### Q: 为什么有些任务没有数据？

A: 只有**成功的trial**才会被提取。如果某个任务的所有trial都失败了，就不会有对应的数据。

### Q: 嵌入向量的含义是什么？

A: 嵌入向量是使用MineCLIP编码的16帧视频片段的512维表示，捕捉了视觉语义信息。

### Q: 如何获取原始视频帧？

A: 查看 `metadata['frame_paths']` 字段，它包含了所有使用的帧文件路径。

### Q: 数据量太大怎么办？

A: 使用NPZ格式（压缩），或只导出CSV（不含嵌入向量），或分批处理。

---

## 📚 相关文档

- **提取器源码**: `src/utils/success_visual_extractor.py`
- **导出脚本**: `scripts/export_instruction_video_pairs.py`
- **Prior评估指南**: `docs/guides/PRIOR_EVALUATION_GUIDE.md`

---

**版本**: 1.0  
**日期**: 2025-11-27  
**作者**: AI Assistant

