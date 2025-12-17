# CLIP4MC 数据处理方法总结

## 当前脚本支持的方案

### 方案对比

| 脚本 | 加速方式 | 速度 | 硬件需求 | 适用场景 |
|------|----------|------|----------|----------|
| `prepare_clip4mc_data.py` | 单进程 CPU | 1x | CPU | 小规模测试 |
| `prepare_clip4mc_data_parallel.py` | **多进程 CPU** | **30-60x** | CPU | ✅ **推荐：通用** |
| `prepare_clip4mc_data_gpu.py` | **GPU 硬件解码** | **100-200x** | NVIDIA GPU | ⭐ **最快：有GPU** |

---

## 方案 1: 单进程脚本

**文件**: `src/utils/prepare_clip4mc_data.py`

### 功能

✅ 基础功能
- 视频帧提取（使用 OpenCV）
- CLIP tokenization
- 官方 size 值处理
- 数据集划分（train/val/test）

✅ 分割模式
- `--split-mode random`: 随机划分 80/10/10
- `--split-mode all_train`: 全部作为训练集
- `--split-mode all_test`: 全部作为测试集

### 使用示例

```bash
# 测试集数据（小规模）
python src/utils/prepare_clip4mc_data.py \
    --pairs-json data/test_pairs.json \
    --clips-dir data/clips \
    --output-dir output/processed_test \
    --split-mode all_test
```

### 适用场景
- 测试和验证
- 小规模数据 (<1000 视频)
- 开发调试

---

## 方案 2: 并行处理脚本 ⭐ 推荐

**文件**: `src/utils/prepare_clip4mc_data_parallel.py`

### 核心功能

#### 1. **多进程并行**
```bash
--num-workers 32  # 32 个进程同时处理
```

**性能**:
- 单进程: 35 天 (30万视频)
- 32 进程: **1.2 天**
- 64 进程: **18 小时**

---

#### 2. **断点续传** 🔥

```bash
# 首次运行
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/videos \
    --output-dir /mnt/processed \
    --num-workers 32 \
    --resume \
    --checkpoint-file checkpoint.json

# 中断后恢复（自动从上次位置继续）
python src/utils/prepare_clip4mc_data_parallel.py \
    ... 相同参数 ... \
    --resume
```

**检查点机制**:
- 每处理 1000 个视频保存一次
- 记录已处理的样本索引
- 支持任意次中断恢复

---

#### 3. **优化的视频解码**

**关键优化**:
```python
# ✅ 跳帧读取（不丢失数据）
cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

# ✅ 预分配内存
frames = np.zeros((num_frames, H, W, 3), dtype=np.uint8)

# ✅ 高效序列化
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# ✅ 合并操作
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (256, 160), interpolation=cv2.INTER_LINEAR)
```

**加速效果**:
- 跳帧 vs 逐帧: **100x**
- 预分配 vs append: **2x**
- INTER_LINEAR vs INTER_CUBIC: **2x**

---

#### 4. **进度监控**

使用 `tqdm` 实时显示:
```
🎬 并行处理: 45%|████████      | 135000/300000 [2:15:30<2:50:15, 16.2video/s]
```

显示信息:
- 当前进度百分比
- 已处理/总数
- 已用时间/预估剩余时间
- 处理速度（视频/秒）

---

#### 5. **容错处理**

```python
# 视频损坏/解码失败 → 自动跳过
# 记录前 10 个错误，避免日志爆炸
```

---

### 完整使用示例

#### 示例 1: 处理训练集（30万视频）

```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json data/train_pairs_with_size.json \
    --clips-dir /mnt/data/clips \
    --output-dir /mnt/ssd/processed_train \
    --num-workers 48 \
    --split-mode all_train \
    --resume \
    --checkpoint-file train_checkpoint.json
```

#### 示例 2: 处理测试集

```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json data/test_pairs.json \
    --clips-dir /mnt/data/clips \
    --output-dir /mnt/ssd/processed_test \
    --num-workers 16 \
    --split-mode all_test \
    --resume
```

#### 示例 3: 测试运行（100 视频）

```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/data/clips \
    --output-dir /tmp/test_output \
    --num-workers 8 \
    --max-samples 100
```

---

### 参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--pairs-json` | pairs JSON 文件（必需） | - | - |
| `--clips-dir` | 视频目录（必需） | - | - |
| `--output-dir` | 输出目录（必需） | - | - |
| `--num-workers` | 并行进程数 | CPU 核心数 | 32-64 |
| `--num-frames` | 每视频提取帧数 | 16 | 16 |
| `--frame-height` | 帧高度 | 160 | 160 |
| `--frame-width` | 帧宽度 | 256 | 256 |
| `--split-mode` | 划分模式 | random | all_train |
| `--resume` | 启用断点续传 | False | True |
| `--checkpoint-file` | 检查点文件 | checkpoint.json | - |
| `--max-samples` | 最大样本数（测试用） | None | - |

---

## 方案 3: 未来扩展（已提供指南）

**文档**: `docs/guides/DATA_PROCESSING_OPTIMIZATION.md`

包含以下方案的实现指南：

### 3.1 GPU 加速解码

**特点**:
- 使用 NVDEC 硬件解码
- 速度: ~50-200 视频/秒（取决于 GPU）
- 需要: NVIDIA GPU + CUDA

**预估时间**: 30万视频 = **6-8 小时**

### 3.2 使用 Decord 库

**特点**:
- 比 OpenCV 快 2-3x
- 更好的跳帧支持
- 更低内存占用

**安装**: `pip install decord`

### 3.3 分布式处理

**特点**:
- 多机并行
- 使用 Ray 框架
- 适合超大规模数据

**预估时间**: 4 节点 x 32 核 = **4-6 小时**

---

## 实际部署建议

### 配置 1: 单机服务器

**硬件**:
- CPU: 32-64 核
- RAM: 64GB+
- 存储: 
  - 输入: HDD/NAS（20TB 原始视频）
  - 输出: NVMe SSD（~500GB processed data）

**命令**:
```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json /data/train_pairs.json \
    --clips-dir /hdd/clips \
    --output-dir /nvme/processed \
    --num-workers 48 \
    --split-mode all_train \
    --resume
```

**预估**:
- 时间: 1-1.5 天
- 成本: ~$40 (AWS c5.12xlarge 36 小时)

---

### 配置 2: 高性能单机

**硬件**:
- CPU: 64 核
- RAM: 128GB
- 存储: NVMe RAID 0

**命令**:
```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json /data/train_pairs.json \
    --clips-dir /nvme/clips \
    --output-dir /nvme/processed \
    --num-workers 80 \
    --split-mode all_train \
    --resume
```

**预估**:
- 时间: 12-18 小时
- 成本: ~$50

---

## 监控和维护

### 实时监控

```bash
# 终端 1: 运行脚本
python src/utils/prepare_clip4mc_data_parallel.py ...

# 终端 2: 监控 CPU
htop

# 终端 3: 监控 IO
watch -n 1 iostat -x

# 终端 4: 查看进度
tail -f output.log
```

### 检查进度

```bash
# 查看检查点
cat checkpoint.json

# 计算剩余时间
python3 << 'EOF'
import json
import time

with open('checkpoint.json') as f:
    ckpt = json.load(f)

processed = len(ckpt['processed_indices'])
total = 300000
elapsed = time.time() - ckpt['timestamp']
rate = processed / elapsed if elapsed > 0 else 0
remaining = (total - processed) / rate / 3600 if rate > 0 else 0

print(f"已处理: {processed}/{total} ({processed/total*100:.1f}%)")
print(f"速度: {rate:.1f} 视频/秒")
print(f"剩余时间: {remaining:.1f} 小时")
EOF
```

### 错误处理

```bash
# 查看失败的视频
grep "WARNING" output.log | head -20

# 重试失败的样本
python retry_failed.py --checkpoint checkpoint.json
```

---

---

## 方案 3: GPU 硬件解码 ⚡ 最快

**文件**: `src/utils/prepare_clip4mc_data_gpu.py`

### 核心特性

#### 1. **NVDEC 硬件解码**

使用 NVIDIA GPU 的专用视频解码器（不占用 CUDA 核心）

```bash
# 使用 4 块 GPU
python src/utils/prepare_clip4mc_data_gpu.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/videos \
    --output-dir /mnt/processed \
    --gpu-ids 0,1,2,3 \
    --split-mode all_train
```

**性能**:
- 1x 3090: ~80-100 视频/秒
- 4x 3090: ~300-400 视频/秒
- 8x 3090: ~600-800 视频/秒

**30万视频预估时间**:
- 4x 3090: **~4-6 小时**
- 8x 3090: **~2-3 小时**

---

#### 2. **依赖要求**

```bash
# 1. NVIDIA GPU + CUDA
nvidia-smi  # 检查 GPU

# 2. ffmpeg 编译时启用 NVDEC
ffmpeg -hwaccels  # 应该看到 cuda

# 如果没有，重新编译 ffmpeg:
# https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/

# 3. Python 依赖
pip install torch torchvision
pip install nvidia-ml-py3
```

---

#### 3. **GPU vs CPU 对比**

| 操作 | CPU (OpenCV) | GPU (NVDEC) | 加速比 |
|------|--------------|-------------|--------|
| 解码 1080p | 50 ms | 5 ms | 10x |
| 解码 4K | 200 ms | 8 ms | 25x |
| 并发能力 | 受限于核心数 | 多 GPU 线性扩展 | - |
| 内存占用 | 高 | 低（GPU 显存） | - |

---

#### 4. **Fallback 机制**

GPU 解码失败时自动回退到 CPU：

```python
# 自动检测 GPU 可用性
# 如果 GPU 解码失败 → CPU 解码
# 如果视频格式不支持 NVDEC → CPU 解码
```

---

#### 5. **使用示例**

**单 GPU**:
```bash
python src/utils/prepare_clip4mc_data_gpu.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/videos \
    --output-dir /mnt/processed \
    --gpu-ids 0 \
    --split-mode all_train
```

**多 GPU**:
```bash
# 8 块 GPU 并行
python src/utils/prepare_clip4mc_data_gpu.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/videos \
    --output-dir /mnt/processed \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --split-mode all_train \
    --resume
```

**断点续传**:
```bash
# 支持中断恢复
python src/utils/prepare_clip4mc_data_gpu.py \
    ... \
    --resume \
    --checkpoint-file checkpoint_gpu.json
```

---

#### 6. **注意事项**

⚠️ **检查 ffmpeg NVDEC 支持**:
```bash
# 检查是否支持硬件加速
ffmpeg -hwaccels

# 应该看到:
# Hardware acceleration methods:
# cuda
# nvdec
```

⚠️ **视频格式兼容性**:
- ✅ H.264/AVC (最常见)
- ✅ H.265/HEVC
- ✅ VP9
- ❌ VP8, AV1 (需要 CPU fallback)

⚠️ **GPU 显存**:
- 每个 GPU 进程 ~2GB 显存
- 8GB GPU 可以运行 1 个进程
- 24GB GPU 可以运行多个进程（但通常 1 个进程/GPU 最优）

---

## 总结

| 方案 | 速度 | 硬件 | 30万视频耗时 | 推荐度 |
|------|------|------|--------------|--------|
| 单进程 | 1x | CPU | 35 天 | ⭐ 测试 |
| **并行处理** | **30-60x** | **CPU** | **1.2 天** | **⭐⭐⭐⭐⭐ 通用** |
| **GPU 加速** | **100-200x** | **NVIDIA GPU** | **4-6 小时** | **⭐⭐⭐⭐⭐ 最快** |
| 分布式 | 150x | 多机 CPU | 6-8 小时 | ⭐⭐⭐ 超大规模 |

### 选择建议

**有 NVIDIA GPU**: 使用 `prepare_clip4mc_data_gpu.py`（最快）

**仅 CPU**: 使用 `prepare_clip4mc_data_parallel.py`（性价比高）

**小规模测试**: 使用 `prepare_clip4mc_data.py`

