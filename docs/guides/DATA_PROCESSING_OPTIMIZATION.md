# CLIP4MC 数据处理优化指南

## 性能对比

| 方案 | 速度 | 适用场景 |
|------|------|----------|
| 单进程 | 1x (35天) | 小规模测试 |
| 32进程 CPU | 30x (1.2天) | ✅ **推荐：通用** |
| GPU 解码 | 50x (17小时) | 有 NVIDIA GPU |
| 分布式 (4机x32核) | 120x (7小时) | 大规模生产 |

---

## 方案 1: 多进程 CPU (推荐)

### 使用方法

```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/videos \
    --output-dir /mnt/processed \
    --num-workers 32 \
    --split-mode all_train \
    --resume
```

### 最佳实践

1. **进程数选择**：
   - CPU 密集型：`num_workers = CPU 核心数`
   - IO 密集型：`num_workers = CPU 核心数 * 1.5`
   - 推荐：32-64 进程（视服务器而定）

2. **存储优化**：
   - 输入视频：机械硬盘（顺序读取）
   - 输出 pkl：SSD（大量随机写入）
   - 使用 NVMe SSD 可再提速 2-3x

3. **内存需求**：
   - 每进程 ~500MB
   - 32 进程 ~16GB RAM

---

## 方案 2: GPU 加速解码

### 安装依赖

```bash
# 需要 NVIDIA GPU + CUDA
pip install nvidia-ml-py3
pip install cupy-cuda11x  # 根据 CUDA 版本选择
```

### 核心代码修改

```python
# 使用 NVDEC 硬件解码
import pycuda.driver as cuda
import pycuda.autoinit

def extract_frames_gpu(video_path, num_frames=16):
    """使用 GPU 解码视频"""
    # 使用 ffmpeg-python + CUDA
    import ffmpeg
    
    probe = ffmpeg.probe(str(video_path))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    
    # NVDEC 解码
    out, _ = (
        ffmpeg
        .input(str(video_path), hwaccel='cuda')
        .filter('select', f'not(mod(n,{total_frames//num_frames}))')
        .filter('scale', 256, 160)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )
    
    return np.frombuffer(out, np.uint8).reshape([-1, 160, 256, 3])
```

### 性能

- 1080 Ti: ~50 视频/秒
- 3090: ~100 视频/秒
- A100: ~200 视频/秒

---

## 方案 3: 分布式处理

### 架构

```
主节点 (Coordinator)
├── 分发任务
└── 收集结果

工作节点 1 (32 进程)
工作节点 2 (32 进程)
工作节点 3 (32 进程)
工作节点 4 (32 进程)
```

### 使用 Ray 实现

```python
import ray

ray.init(address='auto')  # 连接到 Ray 集群

@ray.remote
def process_video_batch(batch, ...):
    """Ray remote 函数"""
    return process_batch_parallel(batch, ...)

# 分发任务
futures = []
batch_size = len(pairs) // num_nodes
for i in range(num_nodes):
    batch = pairs[i*batch_size:(i+1)*batch_size]
    futures.append(process_video_batch.remote(batch, ...))

# 收集结果
results = ray.get(futures)
```

---

## 方案 4: 使用 Decord (推荐替代)

### 安装

```bash
pip install decord
```

### 优势

- 比 OpenCV 快 2-3x
- 原生支持跳帧
- 更少内存占用

### 代码示例

```python
from decord import VideoReader, cpu

def extract_frames_decord(video_path, num_frames=16):
    """使用 Decord 提取帧（更快）"""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    
    # 均匀采样
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # 批量读取（比逐帧快）
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
    
    # Resize
    frames_resized = np.array([
        cv2.resize(frame, (256, 160)) for frame in frames
    ])
    
    return frames_resized
```

---

## 监控和调试

### 实时监控

```bash
# 监控 CPU
htop

# 监控 IO
iostat -x 1

# 监控进度
tail -f process.log

# 估算剩余时间
python -c "
import json
with open('checkpoint.json') as f:
    ckpt = json.load(f)
processed = len(ckpt['processed_indices'])
total = 300000
rate = processed / (time.time() - ckpt['timestamp'])
remaining = (total - processed) / rate / 3600
print(f'剩余: {remaining:.1f} 小时')
"
```

### 常见问题

**问题 1**: 进程卡死
```bash
# 可能是内存不足，减少 num_workers
--num-workers 16
```

**问题 2**: IO 瓶颈
```bash
# 增加 chunksize
pool.imap_unordered(func, tasks, chunksize=50)
```

**问题 3**: 视频解码失败
```bash
# 检查视频完整性
ffprobe video.mp4

# 跳过损坏的视频
--continue-on-error
```

---

## 最终推荐方案

### 配置 1: 单机 (32核 + 64GB RAM + NVMe SSD)

```bash
python src/utils/prepare_clip4mc_data_parallel.py \
    --pairs-json data/train_pairs.json \
    --clips-dir /mnt/nvme/clips \
    --output-dir /mnt/nvme/processed \
    --num-workers 48 \
    --split-mode all_train \
    --resume
```

**预估耗时**: ~1-1.5 天

### 配置 2: GPU 加速 (8x 3090)

```bash
# 每个 GPU 处理一部分数据
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python process_gpu.py \
        --shard $i --total-shards 8 &
done
```

**预估耗时**: ~6-8 小时

### 配置 3: 分布式 (4 节点 x 32 核)

```bash
# 主节点
ray start --head --port=6379

# 工作节点 (x3)
ray start --address='主节点IP:6379'

# 运行
python process_distributed.py \
    --num-nodes 4
```

**预估耗时**: ~4-6 小时

---

## 成本估算

| 方案 | 时间 | 云服务器成本 (AWS) |
|------|------|--------------------|
| 单进程 | 35 天 | ~$840 (c5.xlarge) |
| 32 进程 | 1.2 天 | ~$40 (c5.9xlarge) |
| GPU (8x3090) | 8 小时 | ~$50 (p4d.24xlarge) |
| 分布式 (4 节点) | 6 小时 | ~$80 (4x c5.9xlarge) |

**推荐**: 32 进程单机方案（性价比最高）

