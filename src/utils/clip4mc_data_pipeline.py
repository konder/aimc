#!/usr/bin/env python3
"""
CLIP4MC 数据处理流水线 - 统一工具

功能:
1. 视频切片: raw videos + metadata -> clips
2. 数据准备: clips -> training data (pkl files)
3. 支持多进程/GPU加速、断点续传

使用方法:
    # 完整流程: 从原始视频到训练数据
    python src/utils/clip4mc_data_pipeline.py \
        --mode full \
        --videos-dir /path/to/raw_videos \
        --info-csv info.csv \
        --metadata dataset.json \
        --output-dir /path/to/processed \
        --num-workers 32

    # 仅切片
    python src/utils/clip4mc_data_pipeline.py \
        --mode clip \
        --videos-dir /path/to/raw_videos \
        --info-csv info.csv \
        --metadata dataset.json \
        --output-dir /path/to/output

    # 仅数据准备 (已有clips)
    python src/utils/clip4mc_data_pipeline.py \
        --mode process \
        --clips-dir /path/to/clips \
        --pairs-json text_video_pairs.json \
        --output-dir /path/to/processed \
        --num-workers 32

    # GPU 加速
    python src/utils/clip4mc_data_pipeline.py \
        --mode process \
        --clips-dir /path/to/clips \
        --pairs-json text_video_pairs.json \
        --output-dir /path/to/processed \
        --use-gpu \
        --gpu-ids 0,1,2,3
"""

import argparse
import csv
import json
import pickle
import logging
import sys
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import random
from multiprocessing import Pool, Manager, Process, Queue, cpu_count
from queue import Empty
from functools import partial

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 第一阶段: 视频切片
# ============================================================

def extract_video_id(url: str) -> Optional[str]:
    """从 YouTube URL 提取视频 ID"""
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def parse_info_csv(csv_path: Path) -> Dict[str, str]:
    """解析 info.csv -> {video_id: filename}"""
    vid_to_filename = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = 'url' in sample.lower() or 'http' not in sample[:100].lower()
        
        try:
            reader = csv.reader(f)
            if has_header:
                next(reader)
            
            for row in reader:
                if len(row) >= 2:
                    url = row[0].strip().strip('"')
                    filename = row[1].strip().strip('"')
                    vid = extract_video_id(url)
                    if vid:
                        vid_to_filename[vid] = filename
        except Exception as e:
            logger.warning(f"CSV 解析失败: {e}")
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',', 1)
                if len(parts) >= 2:
                    url = parts[0].strip().strip('"')
                    filename = parts[1].strip().strip('"')
                    vid = extract_video_id(url)
                    if vid:
                        vid_to_filename[vid] = filename
    
    logger.info(f"从 info.csv 解析了 {len(vid_to_filename)} 条记录")
    return vid_to_filename


def normalize_filename(name: str) -> str:
    """标准化文件名"""
    name = re.sub(r'\.(mp4|webm|mkv|avi|mov)$', '', name, flags=re.IGNORECASE)
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name


def build_file_index(videos_dir: Path) -> dict:
    """建立视频文件索引"""
    name_index = {}
    for f in videos_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
            normalized = normalize_filename(f.name)
            name_index[normalized] = f
    return name_index


def find_video_file(videos_dir: Path, filename: str, all_files: dict = None) -> Optional[Path]:
    """查找视频文件"""
    direct_path = videos_dir / filename
    if direct_path.exists():
        return direct_path
    
    if not filename.endswith('.mp4'):
        mp4_path = videos_dir / f"{filename}.mp4"
        if mp4_path.exists():
            return mp4_path
    
    if all_files is not None:
        normalized = normalize_filename(filename)
        if normalized in all_files:
            return all_files[normalized]
    
    return None


def extract_clip_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    use_gpu: bool = False,
    gpu_id: int = 0,
    preset: str = "ultrafast",
    crf: int = 28,
    use_copy: bool = False,
    target_height: int = 0,
    target_fps: float = 0
) -> bool:
    """
    使用 ffmpeg 提取视频片段（优化版）
    
    Args:
        use_gpu: 是否使用 GPU 加速编码
        gpu_id: GPU ID
        preset: 编码速度预设 (ultrafast/superfast/veryfast/fast/medium)
        crf: 质量控制 (18-30, 越大越快但质量越低)
        use_copy: 是否直接复制（不重新编码，极快但精度降低）
        target_height: 目标分辨率高度（0=保持原样）
        target_fps: 目标帧率（0=保持原样）
    
    性能优化说明:
        - 快速跳帧 (Input Seek): 已实现，100-300x 加速
        - preset=ultrafast: 2-3x 加速
        - crf=28: 1.3x 加速
        - 注意: CLIP4MC 原始视频已经是 360p 30fps，通常不需要降低分辨率/帧率
    """
    if output_path.exists():
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_time - start_time
    
    # 计算实际需要的帧率（确保至少 20 帧）
    if target_fps > 0:
        min_fps = 20.0 / max(duration, 1.0)  # 至少 20 帧
        actual_fps = max(target_fps, min_fps)
    else:
        actual_fps = 0
    
    if use_copy:
        # 方案 A: 直接复制（极快，10-50x，但精度 ±1秒）
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            "-avoid_negative_ts", "1",
            "-loglevel", "error",
            str(output_path)
        ]
    elif use_gpu:
        # 方案 B: GPU 编码（不推荐，有并发限制）
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-c:a", "aac",
            "-loglevel", "error",
            str(output_path)
        ]
    else:
        # 方案 C: CPU 编码（推荐，优化参数 + 分辨率/帧率优化）
        # 构建视频滤镜
        vf_filters = []
        if target_height > 0:
            vf_filters.append(f"scale=-2:{target_height}")  # 保持宽高比
        if actual_fps > 0:
            vf_filters.append(f"fps={actual_fps:.2f}")
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
        ]
        
        # 添加视频滤镜
        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
        
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,      # 优化: 可配置速度
            "-crf", str(crf),       # 优化: 可配置质量
            "-c:a", "aac",
            "-b:a", "64k",          # 优化: 降低音频码率（CLIP4MC 不用音频）
            "-ac", "1",             # 优化: 单声道
            "-movflags", "+faststart",
            "-loglevel", "error",
            str(output_path)
        ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        
        # 如果 GPU 编码失败，回退到 CPU
        if use_gpu and result.returncode != 0:
            cmd[cmd.index("h264_nvenc")] = "libx264"
            cmd.insert(cmd.index("libx264") + 1, "-preset")
            cmd.insert(cmd.index("libx264") + 2, preset)
            cmd.insert(cmd.index("libx264") + 3, "-crf")
            cmd.insert(cmd.index("libx264") + 4, str(crf))
            result = subprocess.run(cmd, capture_output=True, timeout=120)
        
        # 如果 copy 失败，回退到重新编码
        if use_copy and result.returncode != 0:
            return extract_clip_ffmpeg(
                input_path, output_path, start_time, end_time,
                use_gpu=False, preset=preset, crf=crf, use_copy=False
            )
        
        return result.returncode == 0 and output_path.exists()
    except Exception:
        return False


def clip_single_video(task: Tuple[int, Dict]) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """处理单个视频切片 (worker 函数)"""
    i, item = task
    
    try:
        vid = item['vid']
        begin = item['begin']
        end = item['end']
        clips_dir = item['clips_dir']
        use_gpu = item.get('use_gpu', False)
        gpu_id = item.get('gpu_id', 0)
        preset = item.get('preset', 'ultrafast')
        crf = item.get('crf', 28)
        use_copy = item.get('use_copy', False)
        target_height = item.get('target_height', 360)
        target_fps = item.get('target_fps', 2.0)
        
        clip_name = f"{vid}_{int(begin)}_{int(end)}.mp4"
        clip_path = clips_dir / clip_name
        
        if extract_clip_ffmpeg(
            item['video_path'], clip_path, begin, end,
            use_gpu=use_gpu, gpu_id=gpu_id,
            preset=preset, crf=crf, use_copy=use_copy,
            target_height=target_height, target_fps=target_fps
        ):
            result = {
                'vid': vid,
                'clip_path': str(clip_path),
                'transcript': item['transcript'],
                'begin_time': begin,
                'end_time': end,
                'duration': end - begin,
                'size': item.get('size', [])
            }
            return True, result, None
        else:
            return False, None, f"切片失败: {vid}"
    except Exception as e:
        return False, None, f"异常: {str(e)}"


def clip_videos(
    videos_dir: Path,
    info_csv: Path,
    metadata_json: Path,
    output_dir: Path,
    num_workers: int = 8,
    use_gpu: bool = False,
    gpu_ids: List[int] = [0],
    preset: str = "ultrafast",
    crf: int = 28,
    use_copy: bool = False,
    target_height: int = 0,
    target_fps: float = 0
) -> Tuple[List[Dict], Path]:
    """
    视频切片阶段 (支持并行 + 编码优化)
    
    Args:
        use_gpu: 是否使用 GPU 加速编码（不推荐，有并发限制）
        gpu_ids: GPU IDs 列表
        preset: 编码速度预设 (ultrafast/superfast/fast)
        crf: 质量控制 (18-30)
        use_copy: 直接复制模式（极快但精度降低）
        target_height: 目标分辨率高度（0=保持原样）
        target_fps: 目标帧率（0=保持原样）
    
    Returns:
        (pairs, clips_dir): 文本-视频对列表, 切片目录
    """
    logger.info("=" * 60)
    if use_copy:
        mode_desc = "copy模式（不转码）"
    elif use_gpu:
        mode_desc = "GPU编码（不推荐）"
    else:
        res_desc = f"{target_height}p" if target_height > 0 else "原始分辨率"
        fps_desc = f"{target_fps}fps" if target_fps > 0 else "原始帧率"
        mode_desc = f"CPU编码 (preset={preset}, crf={crf}, {res_desc}, {fps_desc})"
    logger.info(f"阶段 1: 视频切片 ({mode_desc})")
    logger.info("=" * 60)
    
    # 解析映射
    vid_to_filename = parse_info_csv(info_csv)
    
    # 加载元数据
    with open(metadata_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logger.info(f"加载了 {len(metadata)} 条元数据")
    
    # 建立文件索引
    logger.info(f"扫描视频目录: {videos_dir}")
    name_index = build_file_index(videos_dir)
    logger.info(f"找到 {len(name_index)} 个视频文件")
    
    # 统计可用视频
    available_videos = {}
    for vid, filename in vid_to_filename.items():
        video_path = find_video_file(videos_dir, filename, name_index)
        if video_path:
            available_videos[vid] = video_path
    
    logger.info(f"可用视频: {len(available_videos)} / {len(vid_to_filename)}")
    
    # 创建切片目录
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # 筛选可处理的元数据
    processable = []
    for i, item in enumerate(metadata):
        vid = item.get('vid', '')
        if vid in available_videos:
            # 为每个任务分配 GPU（轮流）
            gpu_id = gpu_ids[i % len(gpu_ids)] if use_gpu else 0
            processable.append({
                'vid': vid,
                'video_path': available_videos[vid],
                'transcript': item.get('transcript', item.get('transcript clip', '')),
                'begin': item.get('begin position', item.get('begin', 0)),
                'end': item.get('end position', item.get('end', 0)),
                'size': item.get('size', []),
                'clips_dir': clips_dir,
                'use_gpu': use_gpu,
                'gpu_id': gpu_id,
                'preset': preset,
                'crf': crf,
                'use_copy': use_copy,
                'target_height': target_height,
                'target_fps': target_fps
            })
    
    logger.info(f"可处理片段: {len(processable)} 条")
    logger.info(f"并行进程: {num_workers}")
    if use_gpu:
        logger.info(f"GPU 编码: {len(gpu_ids)} 块 GPU")
    
    if not processable:
        logger.error("没有可处理的片段")
        sys.exit(1)
    
    # 并行切片处理
    tasks = list(enumerate(processable))
    results = []
    failed_count = 0
    
    with Pool(num_workers) as pool:
        if HAS_TQDM:
            clip_results = tqdm(
                pool.imap_unordered(clip_single_video, tasks, chunksize=5),
                total=len(tasks),
                desc="🎬 视频切片",
                unit="clip"
            )
        else:
            clip_results = pool.imap_unordered(clip_single_video, tasks, chunksize=5)
        
        for success, result, error_msg in clip_results:
            if success:
                results.append(result)
            else:
                failed_count += 1
                if error_msg and failed_count <= 5:
                    logger.warning(error_msg)
    
    logger.info(f"切片完成: {len(results)} 个片段 (失败 {failed_count})")
    
    # 保存 pairs JSON
    pairs_json = output_dir / "text_video_pairs.json"
    with open(pairs_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"保存到: {pairs_json}")
    
    return results, clips_dir


# ============================================================
# 第二阶段: 数据准备 (帧提取 + Tokenization)
# ============================================================

def extract_all_frames_cv2(
    video_path: Path,
    frame_height: int = 160,
    frame_width: int = 256,
    max_frames: int = None
) -> Optional[np.ndarray]:
    """
    提取视频的所有帧（官方 CLIP4MC 格式）
    
    ⚠️ 重要: 官方 CLIP4MC 要求保存视频的所有帧，不是采样的16帧！
    DataLoader 会在加载时动态采样16帧。
    
    Args:
        video_path: 视频路径
        frame_height: 目标高度 (默认: 160)
        frame_width: 目标宽度 (默认: 256)
        max_frames: 最大帧数限制（可选，防止超长视频，如1000）
    
    Returns:
        np.ndarray: shape (N, H, W, 3) 其中 N 是总帧数
        
    Example:
        对于 10s × 30fps 的视频:
        返回: (300, 160, 256, 3)
    """
    if not HAS_CV2:
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    frames = []
    frame_count = 0
    
    while True:
        # 检查最大帧数限制
        if max_frames and frame_count >= max_frames:
            logger.warning(f"视频帧数超过限制 {max_frames}，截断")
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换颜色空间 BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize 到目标尺寸
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # 返回所有帧 (N, H, W, 3)
    return np.array(frames, dtype=np.uint8)


def extract_frames_fast_cv2(
    video_path: Path,
    num_frames: int = 16,
    frame_height: int = 160,
    frame_width: int = 256
) -> Optional[np.ndarray]:
    """快速提取帧 (CPU)"""
    if not HAS_CV2:
        return None
    
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
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
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            frames[i] = frame
    
    cap.release()
    return frames


def extract_frames_gpu_ffmpeg(
    video_path: Path,
    num_frames: int = 16,
    frame_height: int = 160,
    frame_width: int = 256,
    gpu_id: int = 0
) -> Optional[np.ndarray]:
    """GPU 硬件解码提取帧"""
    try:
        # 获取总帧数
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
        
        step = total_frames / num_frames if total_frames >= num_frames else 1
        
        # GPU 解码
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
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=30, check=False)
        
        if result.returncode != 0:
            return extract_frames_fast_cv2(video_path, num_frames, frame_height, frame_width)
        
        frame_size = frame_height * frame_width * 3
        frames_data = result.stdout
        
        if len(frames_data) < frame_size * num_frames:
            frames = []
            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                if end <= len(frames_data):
                    frame = np.frombuffer(frames_data[start:end], dtype=np.uint8)
                    frame = frame.reshape((frame_height, frame_width, 3))
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
            return np.array(frames)
        
        frames = np.frombuffer(frames_data, dtype=np.uint8)
        frames = frames.reshape((-1, frame_height, frame_width, 3))
        return frames[:num_frames]
    
    except Exception:
        return extract_frames_fast_cv2(video_path, num_frames, frame_height, frame_width)


# 全局 tokenizer 缓存（避免重复加载）
_global_tokenizer = None

def get_tokenizer():
    """获取或创建 tokenizer（单例模式）"""
    global _global_tokenizer
    if _global_tokenizer is None:
        try:
            import open_clip
            _global_tokenizer = open_clip.get_tokenizer('ViT-B-16')
        except:
            _global_tokenizer = None
    return _global_tokenizer


def tokenize_text_clip(text: str) -> np.ndarray:
    """CLIP tokenization（优化：复用 tokenizer）"""
    tokenizer = get_tokenizer()
    
    if tokenizer is not None:
        try:
            tokens = tokenizer([text])
            return tokens[0].numpy()
        except:
            pass
    
    # Fallback
    tokens = np.zeros(77, dtype=np.int64)
    tokens[0] = 49406  # SOS
    tokens[-1] = 49407  # EOS
    return tokens


def process_single_sample_cpu(
    task: Tuple[int, Dict[str, Any]],
    clips_dir: Path,
    output_dir: Path,
    frame_height: int,
    frame_width: int
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    处理单个样本 (CPU worker)
    
    提取视频的所有帧（官方 CLIP4MC 格式）
    """
    idx, pair = task
    
    try:
        vid = pair['vid']
        transcript = pair.get('transcript', pair.get('transcript clip', ''))
        clip_path_str = pair['clip_path']
        
        clip_filename = Path(clip_path_str).name
        clip_path = clips_dir / clip_filename
        
        if not clip_path.exists():
            return False, None, f"文件不存在: {clip_path}"
        
        # 提取所有帧（官方 CLIP4MC 格式）
        frames = extract_all_frames_cv2(clip_path, frame_height, frame_width)
        
        if frames is None:
            return False, None, f"解码失败: {clip_path}"
        
        # 获取实际帧数
        actual_num_frames = frames.shape[0]
        
        # Tokenize
        tokens = tokenize_text_clip(transcript)
        
        # 保存
        sample_dir = output_dir / f"sample_{idx:06d}_{vid}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        with open(sample_dir / "video_input.pkl", "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(sample_dir / "text_input.pkl", "wb") as f:
            pickle.dump({'tokens': tokens}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Size - 长度应该与实际帧数一致
        if 'size' in pair and isinstance(pair['size'], list) and len(pair['size']) > 0:
            size_values = pair['size']
            # 重采样到实际帧数
            if len(size_values) != actual_num_frames:
                if len(size_values) >= actual_num_frames:
                    indices = np.linspace(0, len(size_values) - 1, actual_num_frames, dtype=int)
                    size_values = [size_values[i] for i in indices]
                else:
                    size_values = size_values + [size_values[-1]] * (actual_num_frames - len(size_values))
        else:
            size_values = [0.5] * actual_num_frames
        
        with open(sample_dir / "size.json", "w") as f:
            json.dump(size_values, f)
        
        return True, str(sample_dir), None
    
    except Exception as e:
        return False, None, f"处理异常: {str(e)}"


def gpu_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    clips_dir: Path,
    output_dir: Path,
    frame_height: int,
    frame_width: int,
    stop_event
):
    """
    GPU 工作进程
    
    注意: GPU 模式使用 CPU 提取所有帧（官方 CLIP4MC 格式）
    GPU 仅用于其他加速，不用于帧提取
    """
    logger.info(f"GPU {gpu_id} worker 启动")
    
    if HAS_TORCH:
        torch.cuda.set_device(gpu_id)
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=1)
            
            if task is None:
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
                
                # 提取所有帧（使用 CPU，官方 CLIP4MC 格式）
                frames = extract_all_frames_cv2(clip_path, frame_height, frame_width)
                
                if frames is None:
                    result_queue.put((False, None, f"解码失败: {clip_path}"))
                    continue
                
                # 获取实际帧数
                actual_num_frames = frames.shape[0]
                
                tokens = tokenize_text_clip(transcript)
                
                sample_dir = output_dir / f"sample_{idx:06d}_{vid}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                with open(sample_dir / "video_input.pkl", "wb") as f:
                    pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(sample_dir / "text_input.pkl", "wb") as f:
                    pickle.dump({'tokens': tokens}, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Size - 长度应该与实际帧数一致
                if 'size' in pair and isinstance(pair['size'], list) and len(pair['size']) > 0:
                    size_values = pair['size']
                    if len(size_values) != actual_num_frames:
                        if len(size_values) >= actual_num_frames:
                            indices = np.linspace(0, len(size_values) - 1, actual_num_frames, dtype=int)
                            size_values = [size_values[i] for i in indices]
                        else:
                            size_values = size_values + [size_values[-1]] * (actual_num_frames - len(size_values))
                else:
                    size_values = [0.5] * actual_num_frames
                
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


def process_data_cpu(
    pairs: List[Dict[str, Any]],
    clips_dir: Path,
    output_dir: Path,
    num_workers: int,
    frame_height: int = 160,
    frame_width: int = 256,
    resume_from: Optional[Path] = None
) -> List[str]:
    """
    CPU 多进程处理（官方 CLIP4MC 格式：提取所有帧）
    """
    logger.info("=" * 60)
    logger.info("阶段 2: 数据准备 (CPU 多进程)")
    logger.info("=" * 60)
    
    # 断点续传
    processed_samples = set()
    if resume_from and resume_from.exists():
        with open(resume_from) as f:
            checkpoint = json.load(f)
            processed_samples = set(checkpoint.get('processed_indices', []))
        logger.info(f"从断点恢复: 已处理 {len(processed_samples)} 个样本")
    
    tasks = [(i, pair) for i, pair in enumerate(pairs) if i not in processed_samples]
    
    if not tasks:
        logger.info("所有样本已处理完成")
        return []
    
    logger.info(f"待处理: {len(tasks)} 个样本")
    logger.info(f"进程数: {num_workers}")
    
    process_func = partial(
        process_single_sample_cpu,
        clips_dir=clips_dir,
        output_dir=output_dir,
        frame_height=frame_height,
        frame_width=frame_width
    )
    
    successful_dirs = []
    failed_count = 0
    
    with Pool(num_workers) as pool:
        if HAS_TQDM:
            results = tqdm(
                pool.imap_unordered(process_func, tasks, chunksize=10),
                total=len(tasks),
                desc="🎬 并行处理",
                unit="video"
            )
        else:
            results = pool.imap_unordered(process_func, tasks, chunksize=10)
        
        checkpoint_interval = 1000
        processed_count = len(processed_samples)
        
        for success, sample_dir, error_msg in results:
            if success:
                successful_dirs.append(sample_dir)
                processed_count += 1
                
                if resume_from and processed_count % checkpoint_interval == 0:
                    with open(resume_from, 'w') as f:
                        json.dump({
                            'processed_indices': list(processed_samples) + list(range(len(successful_dirs))),
                            'timestamp': time.time()
                        }, f)
            else:
                failed_count += 1
                if error_msg and failed_count <= 10:
                    logger.warning(error_msg)
    
    logger.info(f"处理完成: 成功 {len(successful_dirs)}, 失败 {failed_count}")
    return successful_dirs


def process_data_gpu(
    pairs: List[Dict[str, Any]],
    clips_dir: Path,
    output_dir: Path,
    gpu_ids: List[int],
    num_workers_per_gpu: int = 4,
    frame_height: int = 160,
    frame_width: int = 256,
    resume_from: Optional[Path] = None
) -> List[str]:
    """
    GPU 多进程处理（官方 CLIP4MC 格式：提取所有帧）
    
    注意: 使用 CPU 提取所有帧，GPU 仅用于其他加速
    
    Args:
        num_workers_per_gpu: 每个 GPU 运行的 worker 数量
            - 增加可以提高 GPU 利用率
            - 但每个 worker 会占用 GPU 显存（~500MB）
            - 推荐: 2-8 个 worker/GPU
    """
    logger.info("=" * 60)
    logger.info("阶段 2: 数据准备 (GPU 加速)")
    logger.info("=" * 60)
    
    processed_samples = set()
    if resume_from and resume_from.exists():
        with open(resume_from) as f:
            checkpoint = json.load(f)
            processed_samples = set(checkpoint.get('processed_indices', []))
        logger.info(f"从断点恢复: 已处理 {len(processed_samples)} 个样本")
    
    tasks = [(i, pair) for i, pair in enumerate(pairs) if i not in processed_samples]
    
    if not tasks:
        logger.info("所有样本已处理完成")
        return []
    
    total_workers = len(gpu_ids) * num_workers_per_gpu
    logger.info(f"待处理: {len(tasks)} 个样本")
    logger.info(f"GPU 配置: {len(gpu_ids)} 块 GPU × {num_workers_per_gpu} workers = {total_workers} 并行进程")
    
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    stop_event = manager.Event()
    
    for task in tasks:
        task_queue.put(task)
    
    # 为每个 worker 添加毒丸信号
    for _ in range(total_workers):
        task_queue.put(None)
    
    # 为每个 GPU 创建多个 worker
    workers = []
    for gpu_id in gpu_ids:
        for worker_id in range(num_workers_per_gpu):
            p = Process(
                target=gpu_worker,
                args=(gpu_id, task_queue, result_queue, clips_dir, output_dir,
                      frame_height, frame_width, stop_event),
                name=f"GPU-{gpu_id}-Worker-{worker_id}"
            )
            p.start()
            workers.append(p)
    
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
    
    for p in workers:
        p.join(timeout=10)
    
    logger.info(f"处理完成: 成功 {len(successful_dirs)}, 失败 {failed_count}")
    return successful_dirs


def generate_dataset_info(
    successful_dirs: List[str],
    output_dir: Path,
    split_mode: str = 'random',
    seed: int = 42
):
    """生成 dataset_info.json"""
    n = len(successful_dirs)
    
    if split_mode == 'all_train':
        train_dirs, val_dirs, test_dirs = successful_dirs, [], []
        logger.info(f"[all_train] 全部 {n} 个样本作为训练集")
    elif split_mode == 'all_test':
        train_dirs, val_dirs, test_dirs = [], [], successful_dirs
        logger.info(f"[all_test] 全部 {n} 个样本作为测试集")
    else:
        random.seed(seed)
        random.shuffle(successful_dirs)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_test - n_val
        train_dirs = successful_dirs[:n_train]
        val_dirs = successful_dirs[n_train:n_train + n_val]
        test_dirs = successful_dirs[n_train + n_val:]
        logger.info(f"[random] 随机划分:")
    
    logger.info(f"  训练集: {len(train_dirs)}")
    logger.info(f"  验证集: {len(val_dirs)}")
    logger.info(f"  测试集: {len(test_dirs)}")
    
    dataset_info = {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs
    }
    
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"✓ dataset_info.json 已保存")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP4MC 数据处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 模式
    parser.add_argument("--mode", type=str, required=True,
                       choices=['full', 'clip', 'process'],
                       help="运行模式: full=完整流程, clip=仅切片, process=仅数据准备")
    
    # 切片阶段参数
    parser.add_argument("--videos-dir", type=Path, help="原始视频目录")
    parser.add_argument("--info-csv", type=Path, help="info.csv 文件")
    parser.add_argument("--metadata", type=Path, help="CLIP4MC 元数据 JSON")
    
    # 数据准备阶段参数
    parser.add_argument("--clips-dir", type=Path, help="视频切片目录")
    parser.add_argument("--pairs-json", type=Path, help="text_video_pairs.json")
    
    # 通用参数
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    
    # 处理参数
    parser.add_argument("--num-workers", type=int, default=None,
                       help=f"CPU 进程数或每GPU worker数 (默认: 物理核心数，当前系统: {cpu_count()})")
    parser.add_argument("--use-gpu", action='store_true', help="使用 GPU 加速")
    parser.add_argument("--gpu-ids", type=str, default="0",
                       help="GPU IDs (逗号分隔)")
    parser.add_argument("--workers-per-gpu", type=int, default=None,
                       help="每个GPU的worker数 (默认: --num-workers值，推荐4-8)")
    parser.add_argument("--gpu-encode-clip", action='store_true',
                       help="切片阶段使用 GPU 编码 (h264_nvenc，需配合 --use-gpu，不推荐)")
    
    # 切片优化参数
    parser.add_argument("--clip-preset", type=str, default="ultrafast",
                       choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'],
                       help="ffmpeg 编码速度预设 (默认: ultrafast，最快)")
    parser.add_argument("--clip-crf", type=int, default=28,
                       help="ffmpeg 质量控制 CRF (18-30，越大越快，默认: 28)")
    parser.add_argument("--clip-use-copy", action='store_true',
                       help="直接复制模式 (不重新编码，极快但精度降低)")
    parser.add_argument("--clip-height", type=int, default=0,
                       help="切片目标分辨率高度 (0=保持原样，默认: 0)")
    parser.add_argument("--clip-fps", type=float, default=0,
                       help="切片目标帧率 (0=保持原样，默认: 0)")
    
    # 帧提取参数（官方 CLIP4MC 格式：提取所有帧）
    parser.add_argument("--frame-height", type=int, default=160)
    parser.add_argument("--frame-width", type=int, default=256)
    
    parser.add_argument("--split-mode", type=str, default='random',
                       choices=['random', 'all_train', 'all_test'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--checkpoint-file", type=Path, default=Path("checkpoint.json"))
    
    args = parser.parse_args()
    
    # 智能设置 num_workers 默认值
    if args.num_workers is None:
        # 尝试获取物理核心数（避免超线程）
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores and physical_cores > 0:
                args.num_workers = physical_cores
            else:
                # psutil 返回 None，保守估计
                args.num_workers = max(4, cpu_count() // 4)
        except ImportError:
            # 如果没有 psutil，保守估计
            # 避免使用过多进程（超线程 + 资源竞争）
            logical_cores = cpu_count()
            if logical_cores >= 64:
                # 大型服务器，假设超线程，除以 4 更安全
                args.num_workers = max(16, logical_cores // 4)
            else:
                args.num_workers = max(4, logical_cores // 2)
        
        logger.info(f"自动设置 num_workers = {args.num_workers} (系统逻辑核心: {cpu_count()})")
    
    # 检查依赖
    if not HAS_CV2:
        logger.error("需要安装 opencv-python: pip install opencv-python")
        sys.exit(1)
    
    if args.use_gpu and not HAS_TORCH:
        logger.warning("未安装 PyTorch，回退到 CPU 模式")
        args.use_gpu = False
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    pairs = None
    clips_dir = None
    
    # 解析 GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    # 阶段 1: 切片
    if args.mode in ['full', 'clip']:
        if not args.videos_dir or not args.info_csv or not args.metadata:
            logger.error("--mode=clip/full 需要: --videos-dir, --info-csv, --metadata")
            sys.exit(1)
        
        # GPU 编码有并发限制，调整 workers
        clip_workers = args.num_workers
        use_gpu_encode = args.use_gpu and args.gpu_encode_clip
        
        if use_gpu_encode:
            # NVENC 只能同时处理 2-3 个会话，限制 workers
            max_gpu_encode_workers = len(gpu_ids) * 4  # 每个 GPU 最多 4 个并发
            if clip_workers > max_gpu_encode_workers:
                logger.warning(f"GPU 编码并发限制: workers {clip_workers} → {max_gpu_encode_workers}")
                logger.warning(f"建议: 不使用 --gpu-encode-clip (CPU 编码在并行场景下更快)")
                clip_workers = max_gpu_encode_workers
        
        pairs, clips_dir = clip_videos(
            args.videos_dir,
            args.info_csv,
            args.metadata,
            args.output_dir,
            num_workers=clip_workers,
            use_gpu=use_gpu_encode,
            gpu_ids=gpu_ids,
            preset=args.clip_preset,
            crf=args.clip_crf,
            use_copy=args.clip_use_copy,
            target_height=args.clip_height,
            target_fps=args.clip_fps
        )
    
    # 阶段 2: 数据准备
    if args.mode in ['full', 'process']:
        if args.mode == 'process':
            # 仅处理模式，加载已有的 pairs
            if not args.clips_dir or not args.pairs_json:
                logger.error("--mode=process 需要: --clips-dir, --pairs-json")
                sys.exit(1)
            
            clips_dir = args.clips_dir
            with open(args.pairs_json, encoding='utf-8') as f:
                pairs = json.load(f)
            logger.info(f"加载了 {len(pairs)} 个文本-视频对")
        
        if args.max_samples:
            pairs = pairs[:args.max_samples]
            logger.info(f"限制为 {args.max_samples} 个样本")
        
        # 选择处理方式
        if args.use_gpu:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            # 如果指定了 workers-per-gpu，使用它；否则使用 num-workers
            workers_per_gpu = args.workers_per_gpu if args.workers_per_gpu else args.num_workers
            successful_dirs = process_data_gpu(
                pairs, clips_dir, args.output_dir, gpu_ids,
                num_workers_per_gpu=workers_per_gpu,
                frame_height=args.frame_height,
                frame_width=args.frame_width,
                resume_from=args.checkpoint_file if args.resume else None
            )
        else:
            successful_dirs = process_data_cpu(
                pairs, clips_dir, args.output_dir, args.num_workers,
                args.frame_height, args.frame_width,
                args.checkpoint_file if args.resume else None
            )
        
        if not successful_dirs:
            logger.error("没有成功处理任何样本")
            sys.exit(1)
        
        # 生成 dataset_info.json
        generate_dataset_info(successful_dirs, args.output_dir, args.split_mode, args.seed)
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 处理完成")
    logger.info(f"  总耗时: {elapsed/3600:.2f} 小时")
    if args.mode in ['full', 'process']:
        logger.info(f"  平均速度: {len(successful_dirs)/elapsed:.2f} 视频/秒")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

