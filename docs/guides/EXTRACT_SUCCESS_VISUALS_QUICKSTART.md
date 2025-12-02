# 提取成功画面嵌入 - 快速入门

## 📋 概述

`extract_success_visuals.py` 是一个独立脚本，用于从评估结果中提取成功trial的视觉嵌入（MineCLIP编码）。

---

## 🚀 快速使用

### 方法1: 使用Bash脚本（推荐）

```bash
# 基本使用（分组格式）
bash scripts/run_extract_success_visuals.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_visuals.pkl

# 提取配对格式
bash scripts/run_extract_success_visuals.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_pairs.pkl \
    --pairs
```

### 方法2: 直接运行Python脚本

```bash
python scripts/extract_success_visuals.py \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_visuals.pkl
```

---

## 📊 输出格式

### 分组格式（默认）

用于评估框架，按task_id分组：

```python
import pickle

with open('results/success_visuals.pkl', 'rb') as f:
    data = pickle.load(f)

# 结构
{
    'harvest_1_log': {
        'instruction': 'chop tree',
        'success_visual_embeds': [ndarray[512], ndarray[512], ...],
        'n_success_trials': 2,
        'embed_dim': 512,
    },
    'harvest_1_dirt': { ... },
    ...
}
```

### 配对格式（--pairs）

用于数据分析，扁平化列表：

```python
import pickle

with open('results/success_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)

# 结构
[
    {
        'instruction': 'chop tree',
        'visual_embed': ndarray[512],
        'task_id': 'harvest_1_log',
        'trial_idx': 0,
        'source': 'eval_result',
        'metadata': {
            'embed_dim': 512,
            'n_frames': 16,
            'frame_paths': [...],
            'reward_frame_idx': 45,
            'trial_reward': 1.0,
            ...
        }
    },
    ...
]
```

---

## 🔧 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--eval-result-dir` | 评估结果目录 | `results/evaluation/all_tasks_20251121_214545` |
| `--output` | 输出文件路径（.pkl） | `results/success_visuals.pkl` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pairs` | False | 输出配对格式（否则为分组格式） |
| `--last-n-frames` | 16 | 提取视频帧数 |
| `--no-reward-moment` | False | 使用最后N帧（不基于奖励时刻） |
| `--verbose` | False | 详细输出 |

---

## 📝 使用示例

### 示例1: 提取用于评估的数据

```bash
bash scripts/run_extract_success_visuals.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_visuals.pkl
```

**输出**:
```
================================================================================
提取成功画面嵌入
================================================================================
评估结果目录: results/evaluation/all_tasks_20251121_214545
输出文件: results/success_visuals.pkl
输出格式: 分组格式
提取帧数: 16
使用奖励时刻: True
================================================================================

步骤1: 初始化提取器
--------------------------------------------------------------------------------
加载 MineCLIP...
✓ MineCLIP已加载
✓ 提取器已创建

步骤2: 提取成功画面嵌入
--------------------------------------------------------------------------------
找到 33 个任务目录
处理任务: 100%|████████████████████| 33/33 [00:15<00:00,  2.1it/s]

================================================================================
数据统计
================================================================================
任务数: 10
总嵌入数: 42
平均每任务嵌入数: 4.2

Top 5 任务（按嵌入数）:
  harvest_1_sand (dig sand): 10 个嵌入
  harvest_1_dirt (dig dirt): 9 个嵌入
  techtree_craft_planks (craft wooden planks): 8 个嵌入
  harvest_1_wool (shear sheep): 5 个嵌入
  harvest_1_log (chop tree): 2 个嵌入

嵌入维度: 512
================================================================================

步骤3: 保存数据
--------------------------------------------------------------------------------
✓ 数据已保存到: results/success_visuals.pkl
  文件大小: 0.17 MB

================================================================================
✓ 提取完成！
================================================================================
```

### 示例2: 提取配对格式用于数据分析

```bash
bash scripts/run_extract_success_visuals.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_pairs.pkl \
    --pairs
```

### 示例3: 自定义帧数

```bash
# 提取32帧（更长的视频片段）
bash scripts/run_extract_success_visuals.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_visuals_32frames.pkl \
    --last-n-frames 32
```

### 示例4: 使用最后N帧（不基于奖励时刻）

```bash
bash scripts/run_extract_success_visuals.sh \
    --eval-result-dir results/evaluation/all_tasks_20251121_214545 \
    --output results/success_visuals_lastframes.pkl \
    --no-reward-moment
```

---

## 💻 在Python中使用

### 加载分组格式

```python
import pickle
import numpy as np

# 加载数据
with open('results/success_visuals.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看任务
print(f"任务数: {len(data)}")
for task_id in list(data.keys())[:3]:
    task_data = data[task_id]
    print(f"\n任务: {task_id}")
    print(f"  指令: {task_data['instruction']}")
    print(f"  嵌入数: {len(task_data['success_visual_embeds'])}")
    print(f"  嵌入形状: {task_data['success_visual_embeds'][0].shape}")

# 获取特定任务的嵌入
harvest_log = data.get('harvest_1_log')
if harvest_log:
    embeds = harvest_log['success_visual_embeds']  # List[ndarray[512]]
    print(f"\nharvest_1_log 有 {len(embeds)} 个成功嵌入")
```

### 加载配对格式

```python
import pickle
import numpy as np

# 加载数据
with open('results/success_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)

# 查看数据
print(f"总对数: {len(pairs)}")
for pair in pairs[:3]:
    print(f"\n指令: {pair['instruction']}")
    print(f"  任务: {pair['task_id']}")
    print(f"  Trial: {pair['trial_idx']}")
    print(f"  嵌入形状: {pair['visual_embed'].shape}")
    print(f"  使用帧数: {pair['metadata']['n_frames']}")

# 按任务分组
from collections import defaultdict
by_task = defaultdict(list)
for pair in pairs:
    by_task[pair['task_id']].append(pair)

print(f"\n按任务分组: {len(by_task)} 个任务")
for task_id, task_pairs in list(by_task.items())[:3]:
    print(f"  {task_id}: {len(task_pairs)} 对")
```

### 转换为numpy数组

```python
import numpy as np

# 配对格式 -> numpy数组
instructions = [p['instruction'] for p in pairs]
visual_embeds = np.stack([p['visual_embed'] for p in pairs])
task_ids = [p['task_id'] for p in pairs]

print(f"指令数: {len(instructions)}")
print(f"嵌入矩阵: {visual_embeds.shape}")  # (N, 512)

# 保存为npz
np.savez(
    'results/success_visuals.npz',
    instructions=np.array(instructions, dtype=object),
    visual_embeds=visual_embeds,
    task_ids=np.array(task_ids, dtype=object)
)
```

---

## 🔍 与其他脚本的对比

| 脚本 | 输出格式 | 用途 | 元数据 |
|------|---------|------|-------|
| `extract_success_visuals.py` | PKL (分组/配对) | 通用提取 | 基本 |
| `export_instruction_video_pairs.py` | JSON/CSV/NPZ (配对) | 数据导出 | 丰富 |
| `prior_eval_framework.py` | PKL (分组，缓存) | Prior评估 | 评估专用 |

**选择建议**:
- 快速提取 → `extract_success_visuals.py`
- 数据分析 → `export_instruction_video_pairs.py`
- Prior评估 → `prior_eval_framework.py`（自动缓存）

---

## ❓ 常见问题

### Q: 输出文件很大怎么办？

A: 使用配对格式并只保留需要的字段：

```python
# 保存精简版
import pickle

pairs = pickle.load(open('results/success_pairs.pkl', 'rb'))
compact_pairs = [
    {
        'instruction': p['instruction'],
        'visual_embed': p['visual_embed'],
        'task_id': p['task_id'],
    }
    for p in pairs
]

with open('results/success_pairs_compact.pkl', 'wb') as f:
    pickle.dump(compact_pairs, f)
```

### Q: 如何并行处理多个评估结果？

A: 使用简单的bash循环：

```bash
for eval_dir in results/evaluation/*/; do
    output_name=$(basename "$eval_dir")
    bash scripts/run_extract_success_visuals.sh \
        --eval-result-dir "$eval_dir" \
        --output "results/visuals/${output_name}.pkl"
done
```

### Q: 可以提取其他数据源吗？

A: 目前只支持评估结果。未来版本将支持专家演示（Expert Demonstrations）。

---

## 📚 相关文档

- **提取器源码**: `src/utils/success_visual_extractor.py`
- **配对格式指南**: `docs/guides/INSTRUCTION_VIDEO_PAIRS_GUIDE.md`
- **Prior评估指南**: `docs/guides/PRIOR_EVALUATION_GUIDE.md`

---

**版本**: 1.0  
**日期**: 2025-11-27  
**作者**: AI Assistant

