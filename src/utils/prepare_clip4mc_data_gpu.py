#!/usr/bin/env python3
"""
CLIP4MC 数据准备脚本 - GPU 加速版本

使用 NVIDIA GPU 硬件解码 (NVDEC) 加速视频处理

依赖:
    pip install nvidia-ml-py3 torch torchvision

使用方法:
    python src/utils/prepare_clip4mc_data_gpu.py \
        --pairs-json data/train_pairs.json \
        --clips-dir data/clips \
        --output-dir /path/to/processed \
        --gpu-ids 0,1,2,3 \
        --batch-size 8
"""

import argparse
import json
import pickle
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import random
import time
import subprocess
from multiprocessing import Process, Queue, Manager
from queue import Empty

import numpy as np

try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch 未安装")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GPU-%(process)d] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """检查 GPU 可用性"""
    if not torch.cuda.is_available():
        return False, "CUDA 不可用"
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_info.append(f"GPU {i}: {name}")
        
        return True, "\n".join(gpu_info)
    except Exception as e:
        return True, f"检测到 {torch.cuda.device_count()} 个 GPU"


def extract_frames_gpu_ffmpeg(
    video_path: Path, 
    num_frames: int = 16,
    frame_height: int = 160, 
    frame_width: int = 256,
    gpu_id: int = 0
) -> Optional[np.ndarray]:
    """
    使用 ffmpeg + NVDEC 硬件解码提取帧
    
    注意: 需要 ffmpeg 编译时启用 NVDEC 支持
    """
    try:
        # 1. 获取视频总帧数
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        total_frames = int(result.stdout.strip())
        
        if total_frames == 0:
            return None
        
        # 2. 计算采样间隔
        if total_frames >= num_frames:
            step = total_frames / num_frames
        else:
            step = 1
        
        # 3. 使用 ffmpeg NVDEC 解码
        # -hwaccel cuda: 使用 CUDA 硬件加速
        # -hwaccel_device: 指定 GPU
        # -c:v h264_cuvid: 使用 NVDEC 解码器
        ffmpeg_cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{int(step)})),scale={frame_width}:{frame_height}',
            '-vsync', '0',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-frames:v', str(num_frames),
            'pipe:1'
        ]
        
        result = subprocess.run(
            ffmpeg_cmd, 
            capture_output=True, 
            timeout=30,
            check=False
        )
        
        if result.returncode != 0:
            # Fallback to CPU decoding
            return extract_frames_cpu_fallback(video_path, num_frames, frame_height, frame_width)
        
        # 4. 解析输出
        frame_size = frame_height * frame_width * 3
        frames_data = result.stdout
        
        if len(frames_data) < frame_size * num_frames:
            # 数据不完整，补齐
            frames = []
            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                if end <= len(frames_data):
                    frame = np.frombuffer(frames_data[start:end], dtype=np.uint8)
                    frame = frame.reshape((frame_height, frame_width, 3))
                    frames.append(frame)
                else:
                    # 使用最后一帧或黑帧
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
            return np.array(frames)
        
        frames = np.frombuffer(frames_data, dtype=np.uint8)
        frames = frames.reshape((-1, frame_height, frame_width, 3))
        
        return frames[:num_frames]
    
    except Exception as e:
        logger.debug(f"GPU 解码失败: {e}, 回退到 CPU")
        return extract_frames_cpu_fallback(video_path, num_frames, frame_height, frame_width)


def extract_frames_cpu_fallback(
    video_path: Path, 
    num_frames: int = 16,
    frame_height: int = 160, 
    frame_width: int = 256
) -> Optional[np.ndarray]:
    """CPU 解码 fallback"""
    if not HAS_CV2:
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # 均匀采样
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    
    frames = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_width, frame_height))
            frames[i] = frame
    
    cap.release()
    
    return frames


def tokenize_text_clip(text: str) -> np.ndarray:
    """CLIP tokenization"""
    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokens = tokenizer([text])
        return tokens[0].numpy()
    except:
        # Fallback
        tokens = np.zeros(77, dtype=np.int64)
        tokens[0] = 49406  # SOS
        tokens[-1] = 49407  # EOS
        return tokens


def gpu_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    clips_dir: Path,
    output_dir: Path,
    num_frames: int,
    frame_height: int,
    frame_width: int,
    stop_event
):
    """
    GPU 工作进程
    
    每个 GPU 一个进程，批量处理视频
    """
    logger.info(f"GPU {gpu_id} worker 启动")
    
    # 设置当前进程使用的 GPU
    if HAS_TORCH:
        torch.cuda.set_device(gpu_id)
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # 获取任务（超时 1 秒）
            task = task_queue.get(timeout=1)
            
            if task is None:  # 毒丸信号
                break
            
            idx, pair = task
            
            try:
                vid = pair['vid']
                transcript = pair.get('transcript', pair.get('transcript clip', ''))
                clip_path_str = pair['clip_path']
                
                clip_filename = Path(clip_path_str).name
                clip_path = clips_dir / clip_filename
                
                if not clip_path.exists():
                    result_queue.put((False, None, f"文件不存在: {clip_path}"))
                    continue
                
                # GPU 解码
                frames = extract_frames_gpu_ffmpeg(
                    clip_path, num_frames, frame_height, frame_width, gpu_id
                )
                
                if frames is None:
                    result_queue.put((False, None, f"解码失败: {clip_path}"))
                    continue
                
                # Tokenize
                tokens = tokenize_text_clip(transcript)
                
                # 保存
                sample_dir = output_dir / f"sample_{idx:06d}_{vid}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                with open(sample_dir / "video_input.pkl", "wb") as f:
                    pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(sample_dir / "text_input.pkl", "wb") as f:
                    pickle.dump({'tokens': tokens}, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Size
                if 'size' in pair and isinstance(pair['size'], list) and len(pair['size']) > 0:
                    size_values = pair['size']
                    if len(size_values) != num_frames:
                        if len(size_values) >= num_frames:
                            indices = np.linspace(0, len(size_values) - 1, num_frames, dtype=int)
                            size_values = [size_values[i] for i in indices]
                        else:
                            size_values = size_values + [size_values[-1]] * (num_frames - len(size_values))
                else:
                    size_values = [0.5] * num_frames
                
                with open(sample_dir / "size.json", "w") as f:
                    json.dump(size_values, f)
                
                result_queue.put((True, str(sample_dir), None))
                processed_count += 1
                
            except Exception as e:
                result_queue.put((False, None, f"处理异常: {str(e)}"))
        
        except Empty:
            continue
        except Exception as e:
            logger.error(f"Worker 异常: {e}")
            break
    
    logger.info(f"GPU {gpu_id} worker 完成，处理了 {processed_count} 个视频")


def process_with_gpu(
    pairs: List[Dict[str, Any]],
    clips_dir: Path,
    output_dir: Path,
    gpu_ids: List[int],
    num_frames: int = 16,
    frame_height: int = 160,
    frame_width: int = 256,
    resume_from: Optional[Path] = None
) -> List[str]:
    """
    使用多 GPU 并行处理
    """
    # 加载检查点
    processed_samples = set()
    if resume_from and resume_from.exists():
        with open(resume_from) as f:
            checkpoint = json.load(f)
            processed_samples = set(checkpoint.get('processed_indices', []))
        logger.info(f"从断点恢复: 已处理 {len(processed_samples)} 个样本")
    
    # 过滤未处理的样本
    tasks = [(i, pair) for i, pair in enumerate(pairs) if i not in processed_samples]
    
    if not tasks:
        logger.info("所有样本已处理完成")
        return []
    
    logger.info(f"待处理: {len(tasks)} 个样本")
    logger.info(f"使用 GPU: {gpu_ids}")
    
    # 创建队列
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    stop_event = manager.Event()
    
    # 填充任务队列
    for task in tasks:
        task_queue.put(task)
    
    # 为每个 GPU 添加毒丸信号
    for _ in gpu_ids:
        task_queue.put(None)
    
    # 启动 GPU workers
    workers = []
    for gpu_id in gpu_ids:
        p = Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, clips_dir, output_dir,
                  num_frames, frame_height, frame_width, stop_event)
        )
        p.start()
        workers.append(p)
    
    # 收集结果
    successful_dirs = []
    failed_count = 0
    
    if HAS_TQDM:
        pbar = tqdm(total=len(tasks), desc="🎬 GPU 处理", unit="video")
    
    checkpoint_interval = 1000
    processed_count = len(processed_samples)
    
    for _ in range(len(tasks)):
        try:
            success, sample_dir, error_msg = result_queue.get(timeout=60)
            
            if success:
                successful_dirs.append(sample_dir)
                processed_count += 1
                
                # 保存检查点
                if resume_from and processed_count % checkpoint_interval == 0:
                    with open(resume_from, 'w') as f:
                        json.dump({
                            'processed_indices': list(processed_samples) + 
                                               list(range(len(successful_dirs))),
                            'timestamp': time.time()
                        }, f)
            else:
                failed_count += 1
                if error_msg and failed_count <= 10:
                    logger.warning(error_msg)
            
            if HAS_TQDM:
                pbar.update(1)
        
        except Empty:
            logger.warning("结果队列超时")
            break
    
    if HAS_TQDM:
        pbar.close()
    
    # 等待所有 workers 完成
    for p in workers:
        p.join(timeout=10)
    
    logger.info(f"处理完成: 成功 {len(successful_dirs)}, 失败 {failed_count}")
    
    return successful_dirs


def main():
    parser = argparse.ArgumentParser(description="CLIP4MC 数据准备（GPU 加速版本）")
    
    parser.add_argument("--pairs-json", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    
    parser.add_argument("--gpu-ids", type=str, default="0", 
                       help="GPU IDs，逗号分隔，如 0,1,2,3")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-height", type=int, default=160)
    parser.add_argument("--frame-width", type=int, default=256)
    
    parser.add_argument("--split-mode", type=str, 
                       choices=['random', 'all_train', 'all_test'],
                       default='random')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--checkpoint-file", type=Path, 
                       default=Path("checkpoint_gpu.json"))
    
    args = parser.parse_args()
    
    # 检查 GPU
    available, gpu_info = check_gpu_availability()
    if not available:
        logger.error(f"GPU 不可用: {gpu_info}")
        logger.error("回退到 CPU 模式: python src/utils/prepare_clip4mc_data_parallel.py")
        sys.exit(1)
    
    logger.info(f"检测到 GPU:\n{gpu_info}")
    
    # 解析 GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    logger.info(f"将使用 GPU: {gpu_ids}")
    
    # 加载数据
    with open(args.pairs_json, encoding="utf-8") as f:
        pairs = json.load(f)
    
    logger.info(f"加载了 {len(pairs)} 个样本")
    
    if args.max_samples:
        pairs = pairs[:args.max_samples]
        logger.info(f"限制为 {args.max_samples} 个样本")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"GPU 加速处理配置:")
    logger.info(f"  GPU 数量: {len(gpu_ids)}")
    logger.info(f"  帧尺寸: {args.frame_height}x{args.frame_width}")
    logger.info(f"  断点续传: {'开启' if args.resume else '关闭'}")
    logger.info(f"{'='*60}\n")
    
    start_time = time.time()
    
    successful_dirs = process_with_gpu(
        pairs=pairs,
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
        gpu_ids=gpu_ids,
        num_frames=args.num_frames,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
        resume_from=args.checkpoint_file if args.resume else None
    )
    
    elapsed = time.time() - start_time
    
    if not successful_dirs:
        logger.error("没有成功处理任何样本")
        sys.exit(1)
    
    logger.info(f"\n总耗时: {elapsed/3600:.2f} 小时")
    logger.info(f"平均速度: {len(successful_dirs)/elapsed:.2f} 视频/秒")
    
    # 划分数据集
    n = len(successful_dirs)
    
    if args.split_mode == 'all_train':
        train_dirs, val_dirs, test_dirs = successful_dirs, [], []
    elif args.split_mode == 'all_test':
        train_dirs, val_dirs, test_dirs = [], [], successful_dirs
    else:
        random.seed(args.seed)
        random.shuffle(successful_dirs)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_test - n_val
        train_dirs = successful_dirs[:n_train]
        val_dirs = successful_dirs[n_train:n_train + n_val]
        test_dirs = successful_dirs[n_train + n_val:]
    
    logger.info(f"训练集: {len(train_dirs)}")
    logger.info(f"验证集: {len(val_dirs)}")
    logger.info(f"测试集: {len(test_dirs)}")
    
    # 保存 dataset_info.json
    dataset_info = {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs
    }
    
    with open(args.output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"\n✓ 完成")


if __name__ == "__main__":
    main()

