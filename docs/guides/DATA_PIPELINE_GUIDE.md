# CLIP4MC 数据处理流水线使用指南

**统一工具**: `src/utils/clip4mc_data_pipeline.py`

整合了所有数据处理功能，支持从原始视频到训练数据的完整流程。

---

## 功能特性

✅ **完整流程**: 原始视频 → 切片 → 训练数据  
✅ **模块化**: 支持单独运行切片或数据准备  
✅ **多进程加速**: CPU 并行处理  
✅ **GPU 加速**: NVIDIA GPU 硬件解码  
✅ **断点续传**: 支持中断恢复  
✅ **进度监控**: 实时显示进度  

---

## 三种运行模式

### 1. **完整流程** (`--mode full`)

从原始视频到训练数据一步完成

```bash
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir /path/to/raw_videos \
    --info-csv /path/to/info.csv \
    --metadata /path/to/dataset.json \
    --output-dir /path/to/processed \
    --num-workers 32 \
    --split-mode all_train
```

**输出结构**:
```
output-dir/
├── clips/                      # 视频切片
│   ├── VID1_0_30.mp4
│   └── VID2_10_40.mp4
├── text_video_pairs.json       # 切片元数据
├── sample_000000_VID1/         # 训练样本
│   ├── video_input.pkl
│   ├── text_input.pkl
│   └── size.json
├── sample_000001_VID2/
│   └── ...
└── dataset_info.json           # 数据集划分
```

---

### 2. **仅切片** (`--mode clip`)

从原始视频生成切片（不进行数据准备）

```bash
python src/utils/clip4mc_data_pipeline.py \
    --mode clip \
    --videos-dir /path/to/raw_videos \
    --info-csv /path/to/info.csv \
    --metadata /path/to/dataset.json \
    --output-dir /path/to/output
```

**输出**:
- `output-dir/clips/` - 视频切片
- `output-dir/text_video_pairs.json` - 切片元数据

---

### 3. **仅数据准备** (`--mode process`)

从已有切片生成训练数据（跳过切片阶段）

```bash
python src/utils/clip4mc_data_pipeline.py \
    --mode process \
    --clips-dir /path/to/clips \
    --pairs-json /path/to/text_video_pairs.json \
    --output-dir /path/to/processed \
    --num-workers 32 \
    --split-mode all_train
```

---

## 参数详解

### 模式参数

| 参数 | 说明 | 必需 |
|------|------|------|
| `--mode` | 运行模式: `full`, `clip`, `process` | ✅ |

---

### 切片阶段参数 (mode=full/clip)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--videos-dir` | 原始视频目录 | - |
| `--info-csv` | info.csv 文件 (URL,filename) | - |
| `--metadata` | CLIP4MC 元数据 JSON | - |

**info.csv 格式**:
```csv
url,filename
https://www.youtube.com/watch?v=ABC123,Video Title.mp4
https://www.youtube.com/watch?v=DEF456,Another Video.mp4
```

---

### 数据准备阶段参数 (mode=full/process)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--clips-dir` | 视频切片目录 | - |
| `--pairs-json` | text_video_pairs.json | - |
| `--num-workers` | CPU 进程数 | CPU 核心数 |
| `--use-gpu` | 启用 GPU 加速 | False |
| `--gpu-ids` | GPU IDs (逗号分隔) | `0` |

---

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 输出目录 | **必需** |
| `--num-frames` | 每个视频提取帧数 | 16 |
| `--frame-height` | 帧高度 | 160 |
| `--frame-width` | 帧宽度 | 256 |
| `--split-mode` | 数据集划分: `random`, `all_train`, `all_test` | `random` |
| `--seed` | 随机种子 | 42 |
| `--max-samples` | 最大样本数 (测试用) | None |
| `--resume` | 启用断点续传 | False |
| `--checkpoint-file` | 检查点文件 | `checkpoint.json` |

---

## 使用示例

### 示例 1: 处理测试数据 (完整流程)

```bash
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir /Users/nanzhang/aimc/data/raw_videos/clip4mc_youtube/videos \
    --info-csv /Users/nanzhang/aimc/data/raw_videos/clip4mc_youtube/info.csv \
    --metadata /Users/nanzhang/aimc/data/raw_videos/clip4mc_youtube/.cache/dataset_test.json \
    --output-dir /Users/nanzhang/clip4mc/processed_data \
    --num-workers 8 \
    --split-mode all_test
```

**说明**:
- 从原始视频开始
- 使用 8 个 CPU 进程
- 全部样本作为测试集
- 输出到 CLIP4MC 训练目录

---

### 示例 2: 处理训练数据 (GPU 加速)

```bash
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir /mnt/nvme/raw_videos \
    --info-csv /mnt/nvme/info.csv \
    --metadata /mnt/nvme/dataset_train.json \
    --output-dir /mnt/nvme/processed \
    --use-gpu \
    --gpu-ids 0,1,2,3 \
    --workers-per-gpu 8 \
    --split-mode all_train \
    --resume
    # 注意：不要加 --gpu-encode-clip（会变慢）
```

**说明**:
- 使用 4 块 GPU 加速（仅用于帧提取，不用于切片编码）
- 全部样本作为训练集
- 启用断点续传
- ⚠️ **不要使用 `--gpu-encode-clip`**（GPU 编码器有并发限制，反而变慢）

**性能预估** (30万视频):
- 切片阶段: CPU 并行（3-4 clip/s）
- 帧提取阶段: GPU 加速（8-12 video/s per GPU）
- 4x 3090: **~4-6 小时**
- 8x 3090: **~2-3 小时**

---

### 示例 3: 仅切片 (准备数据)

```bash
# 1. 先切片
python src/utils/clip4mc_data_pipeline.py \
    --mode clip \
    --videos-dir /path/to/videos \
    --info-csv /path/to/info.csv \
    --metadata /path/to/dataset.json \
    --output-dir /path/to/output

# 2. 后续使用 GPU 处理
python src/utils/clip4mc_data_pipeline.py \
    --mode process \
    --clips-dir /path/to/output/clips \
    --pairs-json /path/to/output/text_video_pairs.json \
    --output-dir /path/to/processed \
    --use-gpu \
    --gpu-ids 0,1,2,3 \
    --resume
```

**优势**:
- 切片一次，多次使用
- 切片可在 CPU 机器完成，处理在 GPU 机器完成

---

### 示例 4: 断点续传

```bash
# 首次运行
python src/utils/clip4mc_data_pipeline.py \
    --mode process \
    --clips-dir /path/to/clips \
    --pairs-json /path/to/pairs.json \
    --output-dir /path/to/processed \
    --num-workers 32 \
    --resume \
    --checkpoint-file my_checkpoint.json

# 中断后恢复 (使用相同命令)
python src/utils/clip4mc_data_pipeline.py \
    --mode process \
    --clips-dir /path/to/clips \
    --pairs-json /path/to/pairs.json \
    --output-dir /path/to/processed \
    --num-workers 32 \
    --resume \
    --checkpoint-file my_checkpoint.json
```

**检查点机制**:
- 每 1000 个样本保存一次
- 中断后自动跳过已处理样本

---

## 数据集划分

### `--split-mode` 选项

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `random` | 随机 80/10/10 划分 | 小规模测试 |
| `all_train` | 全部作为训练集 | 处理官方训练数据 |
| `all_test` | 全部作为测试集 | 处理官方测试数据 |

**官方数据集处理**:

```bash
# 处理训练集
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir /path/to/train_videos \
    --info-csv train_info.csv \
    --metadata dataset_train_LocalCorrelationFilter.json \
    --output-dir /path/to/train_processed \
    --split-mode all_train \
    --use-gpu --gpu-ids 0,1,2,3

# 处理测试集
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir /path/to/test_videos \
    --info-csv test_info.csv \
    --metadata dataset_test.json \
    --output-dir /path/to/test_processed \
    --split-mode all_test \
    --use-gpu --gpu-ids 0,1,2,3

# 合并 dataset_info.json
python << 'EOF'
import json

with open('/path/to/train_processed/dataset_info.json') as f:
    train_info = json.load(f)

with open('/path/to/test_processed/dataset_info.json') as f:
    test_info = json.load(f)

merged = {
    "train": train_info['train'],
    "val": train_info['train'][:1000],  # 从训练集取 1000 个作为验证集
    "test": test_info['test']
}

with open('/path/to/final_dataset_info.json', 'w') as f:
    json.dump(merged, f, indent=2)
EOF
```

---

## 性能对比

| 模式 | 硬件 | 速度 | 30万视频耗时 |
|------|------|------|--------------|
| 单进程 | 1 CPU | 1x | 35 天 |
| CPU 并行 | 32 CPU | 30-60x | **1.2 天** |
| GPU 加速 | 4x 3090 | 100-200x | **4-6 小时** |

---

## 监控进度

### 实时进度

脚本会自动显示进度条：

```
阶段 1: 视频切片
🎬 视频切片: 100%|████████████| 1000/1000 [05:23<00:00, 3.09clip/s]

阶段 2: 数据准备 (GPU 加速)
🎬 GPU 处理: 65%|███████     | 650/1000 [01:15<00:35, 49.6video/s]
```

### 检查断点

```bash
# 查看检查点内容
python << 'EOF'
import json
with open('checkpoint.json') as f:
    ckpt = json.load(f)
print(f"已处理: {len(ckpt['processed_indices'])} 个样本")
EOF
```

### 监控 GPU

```bash
# 实时监控 GPU 使用
watch -n 1 nvidia-smi
```

---

## 故障排查

### 1. 找不到视频文件

**问题**: `文件不存在: /path/to/video.mp4`

**原因**:
- info.csv 中的文件名与实际文件名不匹配
- 视频文件确实不存在

**解决**:
```bash
# 1. 检查文件名是否匹配
ls /path/to/videos/ | head

# 2. 检查 info.csv 格式
head -5 info.csv

# 3. 使用 --mode clip 先测试切片阶段
```

### 2. GPU 解码失败

**问题**: GPU 解码回退到 CPU

**原因**:
- ffmpeg 未启用 NVDEC
- 视频格式不支持 (VP8, AV1)

**解决**:
```bash
# 检查 ffmpeg 支持
ffmpeg -hwaccels

# 应该看到:
# cuda
# nvdec

# 如果没有，重新编译 ffmpeg 或使用 CPU 模式
```

### 3. 内存不足

**问题**: `MemoryError` 或进程被杀死

**解决**:
```bash
# 减少并行进程数
--num-workers 16  # 原来 32

# 或分批处理
--max-samples 10000
```

---

## 与旧工具对比

| 功能 | 旧工具 | 新工具 (Pipeline) |
|------|--------|-------------------|
| 视频切片 | `video_clip_processor.py` | `--mode clip` |
| CPU 处理 | `prepare_clip4mc_data_parallel.py` | `--mode process` |
| GPU 处理 | `prepare_clip4mc_data_gpu.py` | `--mode process --use-gpu` |
| 完整流程 | 需要 2 个脚本 | `--mode full` |
| 断点续传 | ✅ | ✅ |
| 代码维护 | 3 个文件 | 1 个文件 |

**迁移示例**:

```bash
# 旧方式 (2 步)
python scripts/video_clip_processor.py --videos-dir ... --info-csv ... --metadata ... --output-dir ./output
python src/utils/prepare_clip4mc_data_parallel.py --clips-dir ./output/clips --pairs-json ./output/text_video_pairs.json --output-dir ./processed

# 新方式 (1 步)
python src/utils/clip4mc_data_pipeline.py --mode full --videos-dir ... --info-csv ... --metadata ... --output-dir ./processed
```

---

## 最佳实践

### 1. 生产环境 (30万视频)

```bash
# 使用 GPU 加速 + 断点续传
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir /mnt/nvme/videos \
    --info-csv /mnt/nvme/info.csv \
    --metadata /mnt/nvme/dataset_train.json \
    --output-dir /mnt/nvme/processed \
    --use-gpu \
    --gpu-ids 0,1,2,3 \
    --split-mode all_train \
    --resume \
    --checkpoint-file train_checkpoint.json
```

**建议配置**:
- 4-8 块 GPU
- NVMe SSD 存储
- 定期保存输出目录

### 2. 开发测试

```bash
# 小规模测试
python src/utils/clip4mc_data_pipeline.py \
    --mode full \
    --videos-dir ./videos \
    --info-csv ./info.csv \
    --metadata ./dataset.json \
    --output-dir ./test_output \
    --max-samples 100 \
    --num-workers 4
```

### 3. 分阶段处理

```bash
# 阶段 1: 切片 (CPU 机器)
python src/utils/clip4mc_data_pipeline.py \
    --mode clip \
    --videos-dir /data/videos \
    --info-csv /data/info.csv \
    --metadata /data/dataset.json \
    --output-dir /data/clips_output

# 拷贝到 GPU 机器
rsync -avz /data/clips_output/ gpu-server:/mnt/nvme/clips_output/

# 阶段 2: 数据准备 (GPU 机器)
python src/utils/clip4mc_data_pipeline.py \
    --mode process \
    --clips-dir /mnt/nvme/clips_output/clips \
    --pairs-json /mnt/nvme/clips_output/text_video_pairs.json \
    --output-dir /mnt/nvme/processed \
    --use-gpu \
    --gpu-ids 0,1,2,3 \
    --resume
```

---

## 总结

**推荐使用新工具** `clip4mc_data_pipeline.py`:

✅ **功能完整**: 一个工具完成所有任务  
✅ **灵活**: 3 种模式适应不同场景  
✅ **高效**: 多进程/GPU 加速  
✅ **可靠**: 断点续传、错误处理  
✅ **易用**: 统一接口，减少学习成本  

**旧工具保留**用于特殊场景或向后兼容。

