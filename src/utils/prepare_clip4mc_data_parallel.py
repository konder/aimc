#!/usr/bin/env python3
"""
CLIP4MC 数据准备脚本 - 并行版本

支持多进程并行、断点续传、进度监控

使用方法:
    python src/utils/prepare_clip4mc_data_parallel.py \
        --pairs-json data/train_pairs.json \
        --clips-dir data/clips \
        --output-dir /path/to/processed_data \
        --num-workers 32 \
        --split-mode all_train
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
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV 未安装")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames_fast(video_path: Path, num_frames: int = 16, 
                        frame_height: int = 160, frame_width: int = 256) -> Optional[np.ndarray]:
    """
    快速提取视频帧（优化版本）
    
    优化点:
    1. 使用 cv2.CAP_PROP_POS_FRAMES 跳帧
    2. 避免重复读取
    3. 预分配数组
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # 均匀采样帧索引
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    
    # 预分配数组
    frames = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # BGR -> RGB + Resize (合并操作)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            frames[i] = frame
        # 失败时使用零帧（已预分配）
    
    cap.release()
    
    return frames


def tokenize_text_clip_cached(text: str, tokenizer=None) -> np.ndarray:
    """
    使用缓存的 tokenizer（避免重复加载）
    """
    if tokenizer is None:
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
        except ImportError:
            # Fallback
            logger.warning("open_clip 未安装，使用简单 tokenization")
            return simple_tokenize(text)
    
    tokens = tokenizer([text])
    return tokens[0].numpy()


def simple_tokenize(text: str, context_length: int = 77) -> np.ndarray:
    """简单 tokenization fallback"""
    SOS_TOKEN, EOS_TOKEN = 49406, 49407
    tokens = [SOS_TOKEN]
    
    for char in text.lower()[:context_length - 2]:
        if char == ' ':
            tokens.append(267)
        elif char.isalpha():
            tokens.append(ord(char) - ord('a') + 320)
        elif char.isdigit():
            tokens.append(ord(char) - ord('0') + 410)
        else:
            tokens.append(256)
    
    tokens.append(EOS_TOKEN)
    while len(tokens) < context_length:
        tokens.append(0)
    
    return np.array(tokens[:context_length], dtype=np.int64)


def process_single_sample(
    task: Tuple[int, Dict[str, Any]],
    clips_dir: Path,
    output_dir: Path,
    num_frames: int,
    frame_height: int,
    frame_width: int,
    tokenizer=None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    处理单个样本（worker 函数）
    
    Returns:
        (success, sample_dir, error_msg)
    """
    idx, pair = task
    
    try:
        vid = pair['vid']
        transcript = pair.get('transcript', pair.get('transcript clip', ''))
        clip_path_str = pair['clip_path']
        
        # 构建 clip 路径
        clip_filename = Path(clip_path_str).name
        clip_path = clips_dir / clip_filename
        
        if not clip_path.exists():
            return False, None, f"视频不存在: {clip_path}"
        
        # 提取视频帧
        frames = extract_frames_fast(clip_path, num_frames, frame_height, frame_width)
        
        if frames is None:
            return False, None, f"视频解码失败: {clip_path}"
        
        # Tokenize 文本
        tokens = tokenize_text_clip_cached(transcript, tokenizer)
        
        # 创建输出目录
        sample_dir = output_dir / f"sample_{idx:06d}_{vid}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        with open(sample_dir / "video_input.pkl", "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(sample_dir / "text_input.pkl", "wb") as f:
            pickle.dump({'tokens': tokens}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 保存 size
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
        
        return True, str(sample_dir), None
    
    except Exception as e:
        return False, None, f"处理异常: {str(e)}"


def worker_init():
    """Worker 进程初始化（加载 tokenizer）"""
    global _tokenizer
    try:
        import open_clip
        _tokenizer = open_clip.get_tokenizer('ViT-B-16')
    except:
        _tokenizer = None


def process_batch_parallel(
    pairs: List[Dict[str, Any]],
    clips_dir: Path,
    output_dir: Path,
    num_workers: int,
    num_frames: int = 16,
    frame_height: int = 160,
    frame_width: int = 256,
    resume_from: Optional[Path] = None
) -> List[str]:
    """
    并行处理视频批次
    
    Args:
        pairs: 视频对列表
        num_workers: 并行进程数
        resume_from: 断点续传文件路径
    
    Returns:
        successful_dirs: 成功处理的样本目录列表
    """
    # 加载已处理的样本（断点续传）
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
    
    logger.info(f"待处理样本: {len(tasks)}")
    
    # 创建进程池
    process_func = partial(
        process_single_sample,
        clips_dir=clips_dir,
        output_dir=output_dir,
        num_frames=num_frames,
        frame_height=frame_height,
        frame_width=frame_width
    )
    
    successful_dirs = []
    failed_count = 0
    
    # 使用 imap_unordered 支持进度条
    with Pool(num_workers, initializer=worker_init) as pool:
        if HAS_TQDM:
            results = tqdm(
                pool.imap_unordered(process_func, tasks, chunksize=10),
                total=len(tasks),
                desc="🎬 并行处理",
                unit="video"
            )
        else:
            results = pool.imap_unordered(process_func, tasks, chunksize=10)
        
        # 定期保存检查点
        checkpoint_interval = 1000
        processed_count = len(processed_samples)
        
        for success, sample_dir, error_msg in results:
            if success:
                successful_dirs.append(sample_dir)
                processed_count += 1
                
                # 保存检查点
                if resume_from and processed_count % checkpoint_interval == 0:
                    with open(resume_from, 'w') as f:
                        json.dump({
                            'processed_indices': list(processed_samples) + list(range(len(successful_dirs))),
                            'timestamp': time.time()
                        }, f)
            else:
                failed_count += 1
                if error_msg and failed_count <= 10:  # 只打印前 10 个错误
                    logger.warning(error_msg)
    
    logger.info(f"处理完成: 成功 {len(successful_dirs)}, 失败 {failed_count}")
    
    return successful_dirs


def main():
    parser = argparse.ArgumentParser(description="CLIP4MC 数据准备（并行版本）")
    
    parser.add_argument("--pairs-json", type=Path, required=True, help="pairs JSON 文件")
    parser.add_argument("--clips-dir", type=Path, required=True, help="视频切片目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    
    parser.add_argument("--num-workers", type=int, default=cpu_count(), 
                       help=f"并行进程数 (默认: {cpu_count()})")
    parser.add_argument("--num-frames", type=int, default=16, help="每个视频提取的帧数")
    parser.add_argument("--frame-height", type=int, default=160, help="帧高度")
    parser.add_argument("--frame-width", type=int, default=256, help="帧宽度")
    
    parser.add_argument("--split-mode", type=str, choices=['random', 'all_train', 'all_test'],
                       default='random', help="数据集划分模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数（测试用）")
    
    parser.add_argument("--resume", action='store_true', help="从断点续传")
    parser.add_argument("--checkpoint-file", type=Path, 
                       default=Path("checkpoint.json"), help="检查点文件")
    
    args = parser.parse_args()
    
    # 检查依赖
    if not HAS_CV2:
        logger.error("需要安装 opencv-python")
        sys.exit(1)
    
    # 加载数据
    if not args.pairs_json.exists():
        logger.error(f"pairs JSON 文件不存在: {args.pairs_json}")
        sys.exit(1)
    
    with open(args.pairs_json, encoding="utf-8") as f:
        pairs = json.load(f)
    
    logger.info(f"加载了 {len(pairs)} 个文本-视频对")
    
    # 限制样本数（测试用）
    if args.max_samples and len(pairs) > args.max_samples:
        random.seed(args.seed)
        pairs = random.sample(pairs, args.max_samples)
        logger.info(f"限制为 {args.max_samples} 个样本")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"并行处理配置:")
    logger.info(f"  进程数: {args.num_workers}")
    logger.info(f"  帧尺寸: {args.frame_height}x{args.frame_width}")
    logger.info(f"  帧数: {args.num_frames}")
    logger.info(f"  断点续传: {'开启' if args.resume else '关闭'}")
    logger.info(f"{'='*60}\n")
    
    # 开始处理
    start_time = time.time()
    
    successful_dirs = process_batch_parallel(
        pairs=pairs,
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
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
        train_dirs = successful_dirs
        val_dirs = []
        test_dirs = []
        logger.info(f"[all_train] 全部 {n} 个样本作为训练集")
    elif args.split_mode == 'all_test':
        train_dirs = []
        val_dirs = []
        test_dirs = successful_dirs
        logger.info(f"[all_test] 全部 {n} 个样本作为测试集")
    else:
        random.seed(args.seed)
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
    
    # 生成 dataset_info.json
    dataset_info = {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs
    }
    
    dataset_info_path = args.output_dir / "dataset_info.json"
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✓ dataset_info.json 已保存: {dataset_info_path}")


if __name__ == "__main__":
    main()

