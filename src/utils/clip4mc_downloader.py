#!/usr/bin/env python3
"""
CLIP4MC YouTube 视频下载和预处理工具

从 CLIP4MC 数据集下载 Minecraft YouTube 视频并转换为训练格式。
基于 https://github.com/PKU-RL/CLIP4MC 的数据集。

使用方法:
    # 下载视频
    python -m src.utils.clip4mc_downloader download \
        --output-dir data/raw_videos/clip4mc_youtube \
        --max-samples 100 --cookies data/www.youtube.com_cookies.txt

    # 预处理为训练格式
    python -m src.utils.clip4mc_downloader preprocess \
        --samples-json data/raw_videos/clip4mc_youtube/samples.json \
        --output-dir data/training/clip4mc

    # 一键执行（下载+预处理）
    python -m src.utils.clip4mc_downloader all \
        --max-samples 100 --cookies data/www.youtube.com_cookies.txt
"""

import argparse
import json
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: 简单的进度显示
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.n = 0
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            print("", file=sys.stderr)
        
        def update(self, n=1):
            self.n += n
            if self.total:
                pct = self.n / self.total * 100
                print(f"\r{self.desc}: {pct:.0f}% ({self.n}/{self.total})", end="", file=sys.stderr)
        
        def set_postfix_str(self, s):
            pass

logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

# CLIP4MC 数据集 URL
CLIP4MC_DATASET_URLS = {
    "test": "https://huggingface.co/datasets/AnonymousUserCLIP4MC/CLIP4MC/raw/main/dataset_test.json",
    "train": "https://huggingface.co/datasets/AnonymousUserCLIP4MC/CLIP4MC/raw/main/dataset_train.json",
}

# 视频帧配置
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)

# 默认下载配置
DEFAULT_RESOLUTION = 360  # 360p
DEFAULT_FPS = 20  # Minecraft 默认帧率

# 默认路径
DEFAULT_RAW_DIR = Path("data/raw_videos/clip4mc_youtube")
DEFAULT_TRAINING_DIR = Path("data/training/clip4mc")


# ============================================================
# YouTube 下载功能
# ============================================================

def download_dataset_json(dataset_type: str, cache_dir: Path) -> List[Dict]:
    """下载 CLIP4MC 数据集 JSON 文件"""
    import requests
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"dataset_{dataset_type}.json"
    
    if cache_file.exists():
        logger.info(f"使用缓存: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)
    
    url = CLIP4MC_DATASET_URLS.get(dataset_type)
    if not url:
        raise ValueError(f"未知数据集类型: {dataset_type}")
    
    logger.info(f"下载数据集: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"下载完成，共 {len(data)} 条记录")
    return data


class DownloadAbortError(Exception):
    """403 错误，需要终止下载"""
    pass


def download_youtube_video(
    video_id: str, 
    output_path: Path, 
    timeout: int = 300,
    cookies_file: str = None,
    resolution: int = DEFAULT_RESOLUTION,
    fps: int = None,
) -> bool:
    """
    下载 YouTube 视频 (使用 yt-dlp)
    
    Args:
        video_id: YouTube 视频 ID
        output_path: 输出文件路径
        timeout: 超时时间（秒）
        cookies_file: cookies.txt 文件路径
        resolution: 最大分辨率 (360, 480, 720, 1080)
        fps: 最大帧率，None 表示不限制
    
    Raises:
        DownloadAbortError: 遇到 403 错误，需要终止整个下载任务
    """
    if output_path.exists():
        logger.debug(f"视频已存在: {output_path}")
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        import yt_dlp
        
        # 使用 HLS 流格式 (91=144p, 92=240p, 93=360p, 94=480p, 95=720p)
        # HLS 格式可以绕过 googlevideo.com 的 403 限制
        if resolution <= 144:
            hls_format = "91"
        elif resolution <= 240:
            hls_format = "92"
        elif resolution <= 360:
            hls_format = "93"
        elif resolution <= 480:
            hls_format = "94"
        else:
            hls_format = "95"
        
        ydl_opts = {
            'format': f'{hls_format}/91/best',  # HLS 流格式优先
            'outtmpl': str(output_path.with_suffix('.mp4')),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': timeout,
            'retries': 10,
            'fragment_retries': 10,
            'extractor_retries': 3,
            'proxy': '',  # 禁用代理
        }
        
        if cookies_file and Path(cookies_file).exists():
            ydl_opts['cookiefile'] = cookies_file
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 检查输出文件
        if output_path.exists():
            return True
        output_with_ext = output_path.with_suffix('.mp4')
        if output_with_ext != output_path and output_with_ext.exists():
            output_with_ext.rename(output_path)
            return True
        possible_files = list(output_path.parent.glob(f"{video_id}*.mp4"))
        if possible_files:
            possible_files[0].rename(output_path)
            return True
            
        logger.warning(f"下载完成但文件不存在: {output_path}")
        return False
            
    except ImportError:
        logger.error("请先安装 yt-dlp: pip install yt-dlp")
        return False
    except Exception as e:
        error_msg = str(e)
        if "Video unavailable" in error_msg:
            logger.warning(f"视频不可用: {video_id}")
            return False
        elif "Private video" in error_msg:
            logger.warning(f"私密视频: {video_id}")
            return False
        elif "403" in error_msg:
            logger.error(f"❌ 403 Forbidden: {video_id}")
            logger.error("⚠️ YouTube 拒绝访问，请刷新 cookies 后重试")
            raise DownloadAbortError("403 Forbidden - 需要刷新 cookies")
        else:
            logger.warning(f"下载失败 {video_id}: {error_msg[:100]}")
            return False


def extract_video_clip(
    input_path: Path, 
    output_path: Path, 
    start_time: float, 
    end_time: float
) -> bool:
    """使用 ffmpeg 提取视频片段"""
    if output_path.exists():
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_time - start_time
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-loglevel", "error",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and output_path.exists():
            return True
        logger.warning(f"ffmpeg 失败: {result.stderr[:200] if result.stderr else 'Unknown'}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg 超时")
        return False
    except FileNotFoundError:
        logger.error("请先安装 ffmpeg: brew install ffmpeg")
        return False
    except Exception as e:
        logger.warning(f"提取失败: {e}")
        return False


def download_videos(
    output_dir: Path,
    dataset_type: str = "test",
    max_samples: int = None,
    cookies_file: str = None,
    resume: bool = True,
    resolution: int = DEFAULT_RESOLUTION,
    fps: int = None,
    delete_original: bool = True,
) -> List[Dict]:
    """
    下载 CLIP4MC 数据集的视频片段
    
    Args:
        output_dir: 输出目录
        dataset_type: "test" 或 "train"
        max_samples: 最大样本数
        cookies_file: cookies.txt 文件路径
        resume: 是否从检查点恢复
        resolution: 下载分辨率 (360, 480, 720, 1080)
        fps: 最大帧率，None 表示不限制
        delete_original: 切片成功后是否删除原始完整视频
    
    Returns:
        处理成功的样本列表
    """
    import datetime
    
    output_dir = Path(output_dir)
    videos_dir = output_dir / "videos"
    clips_dir = output_dir / "clips"
    cache_dir = output_dir / ".cache"
    reports_dir = output_dir / "reports"
    
    # 创建目录
    for d in [videos_dir, clips_dir, cache_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 检查点文件
    checkpoint_file = output_dir / ".checkpoint.json"
    samples_file = output_dir / "samples.json"
    
    start_time = time.time()
    
    data = download_dataset_json(dataset_type, cache_dir)
    logger.info(f"数据集共 {len(data)} 条记录")
    
    # 过滤有效样本 (1-30秒)
    valid_samples = [
        s for s in data 
        if 1.0 <= (s.get("end position", 0) - s.get("begin position", 0)) <= 30.0
    ]
    logger.info(f"有效样本（1-30秒）: {len(valid_samples)} 条")
    
    if max_samples:
        samples_to_process = valid_samples[:max_samples]
    else:
        samples_to_process = valid_samples
    
    total = len(samples_to_process)
    
    # 加载检查点
    processed_samples = []
    failed_samples = []
    skipped_samples = []
    processed_ids = set()
    start_idx = 0
    
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            processed_samples = checkpoint.get("processed_samples", [])
            failed_samples = checkpoint.get("failed_samples", [])
            skipped_samples = checkpoint.get("skipped_samples", [])
            processed_ids = set(checkpoint.get("processed_ids", []))
            start_idx = checkpoint.get("last_idx", 0)
            
            logger.info(f"📂 从检查点恢复: 已处理 {len(processed_ids)} 个, 从第 {start_idx + 1} 个继续")
        except Exception as e:
            logger.warning(f"无法加载检查点: {e}")
    
    remaining = total - len(processed_ids)
    logger.info(f"将处理 {remaining} 个样本 (总计 {total})\n")
    
    def save_checkpoint(idx):
        """保存检查点"""
        checkpoint = {
            "last_idx": idx,
            "processed_ids": list(processed_ids),
            "processed_samples": processed_samples,
            "failed_samples": failed_samples,
            "skipped_samples": skipped_samples,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        # 同时更新 samples.json
        with open(samples_file, 'w') as f:
            json.dump(processed_samples, f, indent=2, ensure_ascii=False)
    
    # 使用固定在底部的进度条
    try:
        with tqdm(
            total=total,
            initial=len(processed_ids),
            desc=f"📥 {len(processed_ids)}/{total}",
            unit="video",
            position=0,
            leave=True,
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format='{desc} | {percentage:3.0f}%|{bar}| ✓{postfix} [{elapsed}<{remaining}]'
        ) as pbar:
            
            for idx, sample in enumerate(samples_to_process):
                video_id = sample.get("vid", "unknown")
                transcript = sample.get("transcript clip", "")
                begin_pos = sample.get("begin position", 0)
                end_pos = sample.get("end position", 0)
                
                # 生成唯一标识
                sample_id = f"{video_id}_{int(begin_pos)}_{int(end_pos)}"
                
                # 跳过已处理的
                if sample_id in processed_ids:
                    continue
                
                # 更新进度条描述
                pbar.set_description(f"📥 {idx+1}/{total}")
                pbar.set_postfix_str(f"{len(processed_samples)} ✗{len(failed_samples)}")
                
                if not video_id or not transcript:
                    skipped_samples.append({"video_id": video_id, "reason": "无效数据"})
                    processed_ids.add(sample_id)
                    pbar.update(1)
                    continue
                
                # 下载完整视频
                video_path = videos_dir / f"{video_id}.mp4"
                need_download = not video_path.exists()
                
                if need_download:
                    try:
                        if not download_youtube_video(
                            video_id, video_path, 
                            cookies_file=cookies_file,
                            resolution=resolution,
                            fps=fps
                        ):
                            failed_samples.append({
                                "video_id": video_id,
                                "reason": "下载失败",
                                "begin": begin_pos,
                                "end": end_pos,
                            })
                            processed_ids.add(sample_id)
                            pbar.update(1)
                            # 每10个失败保存检查点
                            if len(failed_samples) % 10 == 0:
                                save_checkpoint(idx)
                            time.sleep(0.5)
                            continue
                    except DownloadAbortError as e:
                        # 403 错误，终止整个下载任务
                        pbar.close()
                        save_checkpoint(idx)
                        logger.error(f"\n❌ 下载任务终止: {e}")
                        logger.error(f"已处理: {len(processed_samples)} 成功, {len(failed_samples)} 失败")
                        logger.error("请刷新 cookies 后使用 'status' 查看进度，'download' 继续下载")
                        return {
                            "success": False,
                            "aborted": True,
                            "reason": str(e),
                            "processed": len(processed_samples),
                            "failed": len(failed_samples),
                        }
                
                # 提取片段
                clip_name = f"{video_id}_{int(begin_pos)}_{int(end_pos)}.mp4"
                clip_path = clips_dir / clip_name
                
                if not clip_path.exists():
                    if not extract_video_clip(video_path, clip_path, begin_pos, end_pos):
                        failed_samples.append({
                            "video_id": video_id,
                            "reason": "片段提取失败",
                            "begin": begin_pos,
                            "end": end_pos,
                        })
                        processed_ids.add(sample_id)
                        pbar.update(1)
                        continue
                
                # 检查是否还有其他片段需要从这个视频提取
                # 如果启用 delete_original，在所有片段提取完成后删除原始视频
                if delete_original and video_path.exists():
                    # 检查是否还有待处理的片段来自同一视频
                    remaining_from_same_video = any(
                        s.get("vid") == video_id and 
                        f"{video_id}_{int(s.get('begin position', 0))}_{int(s.get('end position', 0))}" not in processed_ids
                        for s in samples_to_process[idx+1:]
                    )
                    if not remaining_from_same_video:
                        try:
                            video_path.unlink()
                            logger.debug(f"已删除原始视频: {video_path.name}")
                        except Exception as e:
                            logger.debug(f"删除失败: {e}")
                
                processed_samples.append({
                    "video_id": video_id,
                    "text": transcript.strip(),
                    "clip_path": str(clip_path),
                    "begin_time": begin_pos,
                    "end_time": end_pos,
                    "duration": end_pos - begin_pos,
                })
                processed_ids.add(sample_id)
                
                pbar.update(1)
                
                # 每20个成功保存检查点
                if len(processed_samples) % 20 == 0:
                    save_checkpoint(idx)
                
                time.sleep(0.3)
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ 下载被中断，保存检查点...")
        save_checkpoint(idx if 'idx' in dir() else 0)
        logger.info(f"✓ 检查点已保存，下次运行将自动恢复")
        raise
    
    elapsed_time = time.time() - start_time
    
    # 保存最终结果
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)
    
    # 删除检查点（下载完成）
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("✓ 下载完成，检查点已清除")
    
    # 生成下载报告
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_type": dataset_type,
        "elapsed_seconds": round(elapsed_time, 2),
        "elapsed_human": f"{int(elapsed_time//60)}分{int(elapsed_time%60)}秒",
        "summary": {
            "total": total,
            "success": len(processed_samples),
            "failed": len(failed_samples),
            "skipped": len(skipped_samples),
            "success_rate": f"{len(processed_samples)/total*100:.1f}%" if total > 0 else "0%",
        },
        "output_files": {
            "samples_json": str(samples_file),
            "videos_dir": str(videos_dir),
            "clips_dir": str(clips_dir),
        },
        "failed_samples": failed_samples[:50],  # 只保存前50个失败样本
        "skipped_samples": skipped_samples[:20],
    }
    
    report_file = reports_dir / f"download_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印最终报告
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"📊 CLIP4MC 下载报告", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  📅 时间: {report['timestamp'][:19]}", file=sys.stderr)
    print(f"  ⏱️  耗时: {report['elapsed_human']}", file=sys.stderr)
    print(f"  📦 数据集: {dataset_type}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  📈 统计:", file=sys.stderr)
    print(f"     总任务: {total}", file=sys.stderr)
    print(f"     ✓ 成功: {len(processed_samples)} ({report['summary']['success_rate']})", file=sys.stderr)
    print(f"     ✗ 失败: {len(failed_samples)}", file=sys.stderr)
    print(f"     ⊘ 跳过: {len(skipped_samples)}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  📁 输出文件:", file=sys.stderr)
    print(f"     样本: {samples_file}", file=sys.stderr)
    print(f"     视频: {videos_dir}", file=sys.stderr)
    print(f"     切片: {clips_dir}", file=sys.stderr)
    print(f"     报告: {report_file}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    return processed_samples


# ============================================================
# 数据预处理功能
# ============================================================

def extract_video_frames(video_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """从视频中均匀采样帧"""
    try:
        import cv2
    except ImportError:
        logger.error("请安装 opencv-python: pip install opencv-python")
        raise
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            frame = frames[-1].copy() if frames else np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, FRAME_SIZE)
        frames.append(frame)
    
    cap.release()
    return np.array(frames, dtype=np.uint8)


def get_tokenizer():
    """获取 CLIP tokenizer"""
    try:
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    except Exception:
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            
            class TokenizerWrapper:
                def __init__(self, tok):
                    self._tokenizer = tok
                
                def __call__(self, text, max_length=77, padding='max_length', 
                           truncation=True, return_tensors='np'):
                    tokens = self._tokenizer(text)
                    return {'input_ids': tokens.numpy() if return_tensors == 'np' else tokens}
            
            return TokenizerWrapper(tokenizer)
        except ImportError:
            logger.error("请安装 transformers 或 open-clip-torch")
            raise


def preprocess_samples(samples_json: Path, output_dir: Path) -> List[str]:
    """将视频片段转换为 CLIP4MC 训练格式"""
    import datetime
    
    start_time = time.time()
    
    logger.info("加载 CLIP tokenizer...")
    tokenizer = get_tokenizer()
    logger.info("✓ Tokenizer 已加载\n")
    
    with open(samples_json) as f:
        samples = json.load(f)
    
    if not samples:
        logger.error("没有可处理的样本")
        return []
    
    total = len(samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dirs = []
    failed_samples = []
    
    # 使用固定在底部的进度条
    with tqdm(
        total=total,
        desc=f"🔄 0/{total}",
        unit="sample",
        position=0,
        leave=True,
        file=sys.stderr,
        dynamic_ncols=True,
        bar_format='{desc} | {percentage:3.0f}%|{bar}| ✓{postfix} [{elapsed}<{remaining}]'
    ) as pbar:
        
        for idx, sample in enumerate(samples):
            clip_path = Path(sample['clip_path'])
            text = sample['text']
            video_id = sample.get('video_id', 'unknown')
            
            # 更新进度条描述
            pbar.set_description(f"🔄 {idx+1}/{total}")
            pbar.set_postfix_str(f"{len(processed_dirs)} ✗{len(failed_samples)}")
            
            if not clip_path.exists():
                failed_samples.append({"video_id": video_id, "reason": "文件不存在"})
                pbar.update(1)
                continue
            
            sample_dir = output_dir / f"data_dir_{idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 提取视频帧
                frames = extract_video_frames(clip_path)
                
                # Tokenize 文本
                tokens = tokenizer(text, max_length=77, padding='max_length', 
                                 truncation=True, return_tensors='np')
                text_input = {'tokens': tokens['input_ids'][0]}
                
                # 保存
                with open(sample_dir / "video_input.pkl", 'wb') as f:
                    pickle.dump(frames, f)
                
                with open(sample_dir / "text_input.pkl", 'wb') as f:
                    pickle.dump(text_input, f)
                
                with open(sample_dir / "size.json", 'w') as f:
                    json.dump({
                        "num_samples": 1,
                        "video_shape": list(frames.shape),
                        "text_length": len(text_input['tokens']),
                        "original_text": text,
                        "video_id": video_id,
                    }, f, indent=2)
                
                processed_dirs.append(str(sample_dir))
                
            except Exception as e:
                failed_samples.append({"video_id": video_id, "reason": str(e)[:100]})
            
            pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # 创建 log 文件
    log_data = None
    if processed_dirs:
        n_train = max(1, int(len(processed_dirs) * 0.8))
        log_data = {
            "train": processed_dirs[:n_train],
            "test": processed_dirs[n_train:] if n_train < len(processed_dirs) else processed_dirs[:1],
        }
        with open(output_dir / "data_log.json", 'w') as f:
            json.dump(log_data, f, indent=2)
    
    # 生成预处理报告
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_time, 2),
        "elapsed_human": f"{int(elapsed_time//60)}分{int(elapsed_time%60)}秒",
        "summary": {
            "total": total,
            "success": len(processed_dirs),
            "failed": len(failed_samples),
            "success_rate": f"{len(processed_dirs)/total*100:.1f}%" if total > 0 else "0%",
        },
        "training_split": {
            "train": len(log_data['train']) if log_data else 0,
            "test": len(log_data['test']) if log_data else 0,
        },
        "output_dir": str(output_dir),
        "failed_samples": failed_samples[:30],
    }
    
    report_file = output_dir / f"preprocess_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印最终报告
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"📊 CLIP4MC 预处理报告", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  📅 时间: {report['timestamp'][:19]}", file=sys.stderr)
    print(f"  ⏱️  耗时: {report['elapsed_human']}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  📈 统计:", file=sys.stderr)
    print(f"     总样本: {total}", file=sys.stderr)
    print(f"     ✓ 成功: {len(processed_dirs)} ({report['summary']['success_rate']})", file=sys.stderr)
    print(f"     ✗ 失败: {len(failed_samples)}", file=sys.stderr)
    print(f"", file=sys.stderr)
    if log_data:
        print(f"  📚 数据集划分:", file=sys.stderr)
        print(f"     训练集: {len(log_data['train'])}", file=sys.stderr)
        print(f"     测试集: {len(log_data['test'])}", file=sys.stderr)
        print(f"", file=sys.stderr)
    print(f"  📁 输出文件:", file=sys.stderr)
    print(f"     数据目录: {output_dir}", file=sys.stderr)
    print(f"     报告: {report_file}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    return processed_dirs


# ============================================================
# 命令行接口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLIP4MC YouTube 视频下载和预处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 test 集全部视频
  python -m src.utils.clip4mc_downloader download --dataset test --cookies data/www.youtube.com_cookies.txt

  # 下载 100 个样本
  python -m src.utils.clip4mc_downloader download --max-samples 100 --cookies data/www.youtube.com_cookies.txt

  # 预处理已下载的视频
  python -m src.utils.clip4mc_downloader preprocess

  # 一键执行（下载+预处理）
  python -m src.utils.clip4mc_downloader all --max-samples 100 --cookies data/www.youtube.com_cookies.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # download 命令
    dl_parser = subparsers.add_parser('download', help='下载 YouTube 视频')
    dl_parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_RAW_DIR, help="输出目录")
    dl_parser.add_argument("--dataset", "-d", choices=["test", "train"], default="test", help="数据集类型")
    dl_parser.add_argument("--max-samples", "-n", type=int, default=None, help="最大样本数，默认全部")
    dl_parser.add_argument("--cookies", "-c", type=str, default=None, help="cookies.txt 文件路径")
    dl_parser.add_argument("--no-resume", action="store_true", help="不从检查点恢复，重新开始")
    dl_parser.add_argument("--resolution", "-r", type=int, default=DEFAULT_RESOLUTION, 
                          choices=[360, 480, 720, 1080], help="下载分辨率 (默认: 360)")
    dl_parser.add_argument("--fps", type=int, default=None, help="最大帧率，默认不限制")
    dl_parser.add_argument("--keep-original", action="store_true", 
                          help="保留原始完整视频（默认：切片后删除）")
    
    # preprocess 命令
    pp_parser = subparsers.add_parser('preprocess', help='预处理为训练格式')
    pp_parser.add_argument("--samples-json", type=Path, default=DEFAULT_RAW_DIR / "samples.json")
    pp_parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    
    # all 命令
    all_parser = subparsers.add_parser('all', help='下载并预处理')
    all_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="原始视频目录")
    all_parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR, help="训练数据目录")
    all_parser.add_argument("--dataset", "-d", choices=["test", "train"], default="test")
    all_parser.add_argument("--max-samples", "-n", type=int, default=None)
    all_parser.add_argument("--cookies", "-c", type=str, default=None)
    all_parser.add_argument("--no-resume", action="store_true", help="不从检查点恢复")
    all_parser.add_argument("--resolution", "-r", type=int, default=DEFAULT_RESOLUTION,
                          choices=[360, 480, 720, 1080], help="下载分辨率 (默认: 360)")
    all_parser.add_argument("--fps", type=int, default=None, help="最大帧率")
    all_parser.add_argument("--keep-original", action="store_true", help="保留原始完整视频")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.command == 'download':
        download_videos(
            output_dir=args.output_dir,
            dataset_type=args.dataset,
            max_samples=args.max_samples,
            cookies_file=args.cookies,
            resume=not getattr(args, 'no_resume', False),
            resolution=args.resolution,
            fps=args.fps,
            delete_original=not getattr(args, 'keep_original', False),
        )
    
    elif args.command == 'preprocess':
        preprocess_samples(
            samples_json=args.samples_json,
            output_dir=args.output_dir,
        )
    
    elif args.command == 'all':
        logger.info("=" * 60)
        logger.info("步骤 1: 下载视频")
        logger.info("=" * 60)
        download_videos(
            output_dir=args.raw_dir,
            dataset_type=args.dataset,
            max_samples=args.max_samples,
            cookies_file=args.cookies,
            resume=not getattr(args, 'no_resume', False),
            resolution=args.resolution,
            fps=args.fps,
            delete_original=not getattr(args, 'keep_original', False),
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("步骤 2: 预处理为训练格式")
        logger.info("=" * 60)
        preprocess_samples(
            samples_json=args.raw_dir / "samples.json",
            output_dir=args.training_dir,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("全部完成！")
        logger.info("=" * 60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

