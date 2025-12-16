#!/usr/bin/env python3
"""
CLIP4MC 数据准备脚本

将切片好的视频和文本转换为官方 CLIP4MC 训练所需的格式：
- text_input.pkl: {'tokens': ndarray(77,)}
- video_input.pkl: 视频帧 (N, H, W, C) 尺寸 160x256
- size.json: [size_0, size_1, ..., size_N] 每帧的目标实体大小比例 (0-1)

使用方法:
    python src/utils/prepare_clip4mc_data.py \
        --pairs-json data/raw_videos/clip4mc_youtube/text_video_pairs.json \
        --clips-dir data/raw_videos/clip4mc_youtube/clips \
        --output-dir /Users/nanzhang/clip4mc/processed_data \
        --num-frames 16
"""

import argparse
import json
import pickle
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import random

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV 未安装，尝试使用 PIL + imageio")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames_cv2(video_path: Path, num_frames: int = 16, frame_height: int = 160, frame_width: int = 256) -> Optional[np.ndarray]:
    """
    使用 OpenCV 从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 提取帧数
        frame_height: 帧高度 (CLIP4MC 官方: 160)
        frame_width: 帧宽度 (CLIP4MC 官方: 256)
    
    Returns:
        frames: (N, H, W, C) numpy array, RGB 格式，或 None 如果失败
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.warning(f"无法打开视频: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        logger.warning(f"视频没有帧: {video_path}")
        cap.release()
        return None
    
    # 均匀采样帧索引
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # 帧数不足时，重复最后一帧
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            # 如果读取失败，使用黑帧
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整大小到 CLIP4MC 官方尺寸 (W, H)
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames, dtype=np.uint8)


def extract_frames_imageio(video_path: Path, num_frames: int = 16, frame_height: int = 160, frame_width: int = 256) -> Optional[np.ndarray]:
    """
    使用 imageio 从视频中提取帧
    """
    try:
        reader = imageio.get_reader(str(video_path))
        all_frames = []
        
        for frame in reader:
            all_frames.append(frame)
        
        reader.close()
        
        if len(all_frames) == 0:
            logger.warning(f"视频没有帧: {video_path}")
            return None
        
        total_frames = len(all_frames)
        
        # 均匀采样
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
        
        frames = []
        for idx in indices:
            frame = all_frames[idx]
            
            # 使用 PIL 调整大小
            if HAS_PIL:
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((frame_width, frame_height), Image.BILINEAR)
                frame = np.array(pil_img)
            else:
                # 简单的最近邻缩放
                h, w = frame.shape[:2]
                y_indices = np.linspace(0, h - 1, frame_height, dtype=int)
                x_indices = np.linspace(0, w - 1, frame_width, dtype=int)
                frame = frame[y_indices][:, x_indices]
            
            frames.append(frame)
        
        return np.array(frames, dtype=np.uint8)
    
    except Exception as e:
        logger.warning(f"imageio 读取视频失败: {video_path}, 错误: {e}")
        return None


def extract_frames(video_path: Path, num_frames: int = 16, frame_height: int = 160, frame_width: int = 256) -> Optional[np.ndarray]:
    """
    从视频中提取帧，自动选择可用的方法
    
    CLIP4MC 官方尺寸: 160x256 (H x W)
    """
    if HAS_CV2:
        return extract_frames_cv2(video_path, num_frames, frame_height, frame_width)
    elif HAS_IMAGEIO:
        return extract_frames_imageio(video_path, num_frames, frame_height, frame_width)
    else:
        logger.error("需要安装 opencv-python 或 imageio: pip install opencv-python imageio")
        return None


def tokenize_text_clip(text: str, context_length: int = 77) -> np.ndarray:
    """
    使用 CLIP tokenizer 对文本进行编码
    
    Args:
        text: 输入文本
        context_length: token 序列长度
    
    Returns:
        tokens: (context_length,) numpy array
    """
    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokens = tokenizer([text])
        return tokens[0].numpy()
    except ImportError:
        pass
    
    # Fallback: 简单的字符级 tokenization
    logger.warning("open_clip 未安装，使用简单 tokenization")
    
    # 简单的字符级 tokenization (仅用于测试)
    # 使用 ASCII 码作为 token，加上 SOS/EOS
    SOS_TOKEN = 49406  # CLIP 的 <|startoftext|>
    EOS_TOKEN = 49407  # CLIP 的 <|endoftext|>
    
    tokens = [SOS_TOKEN]
    
    for char in text.lower()[:context_length - 2]:
        # 简单映射：空格和标点符号映射到特定范围
        if char == ' ':
            tokens.append(267)  # 空格在 CLIP vocab 中的大致位置
        elif char.isalpha():
            tokens.append(ord(char) - ord('a') + 320)  # 字母映射
        elif char.isdigit():
            tokens.append(ord(char) - ord('0') + 410)  # 数字映射
        else:
            tokens.append(256)  # 其他字符
    
    tokens.append(EOS_TOKEN)
    
    # Padding
    while len(tokens) < context_length:
        tokens.append(0)
    
    return np.array(tokens[:context_length], dtype=np.int64)


def prepare_sample(
    pair: Dict[str, Any],
    clips_dir: Path,
    output_dir: Path,
    num_frames: int,
    frame_height: int,
    frame_width: int,
    sample_idx: int
) -> Optional[Path]:
    """
    准备单个样本（CLIP4MC 官方格式）
    
    输出格式:
    - text_input.pkl: {'tokens': ndarray(77,)}
    - video_input.pkl: ndarray(N, H, W, C) 尺寸 160x256
    - size.json: [size_0, ..., size_N] 每帧的目标实体大小比例
    
    Returns:
        样本输出目录，或 None 如果失败
    """
    vid = pair['vid']
    transcript = pair['transcript']
    clip_path_str = pair['clip_path']
    
    # 构建实际的 clip 路径
    clip_filename = Path(clip_path_str).name
    clip_path = clips_dir / clip_filename
    
    if not clip_path.exists():
        logger.warning(f"视频文件不存在: {clip_path}")
        return None
    
    # 提取视频帧 (CLIP4MC 官方尺寸: 160x256)
    frames = extract_frames(clip_path, num_frames, frame_height, frame_width)
    
    if frames is None:
        return None
    
    # Tokenize 文本
    tokens = tokenize_text_clip(transcript)
    
    # 创建输出目录
    sample_dir = output_dir / f"sample_{sample_idx:06d}_{vid}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 video_input.pkl (直接保存 ndarray)
    with open(sample_dir / "video_input.pkl", "wb") as f:
        pickle.dump(frames, f)
    
    # 保存 text_input.pkl (官方格式: {'tokens': array})
    with open(sample_dir / "text_input.pkl", "wb") as f:
        pickle.dump({'tokens': tokens}, f)
    
    # 保存 size.json (官方格式: 每帧的目标实体大小比例)
    # 优先使用官方数据中的 size，如果没有则使用占位值
    if 'size' in pair and isinstance(pair['size'], list) and len(pair['size']) > 0:
        # 使用官方提供的 size 值
        size_values = pair['size']
        size_source = 'official'
        # 确保 size 长度与帧数一致
        if len(size_values) != num_frames:
            # 重采样 size 值以匹配帧数
            if len(size_values) >= num_frames:
                indices = np.linspace(0, len(size_values) - 1, num_frames, dtype=int)
                size_values = [size_values[i] for i in indices]
            else:
                # 不足时重复最后一个值
                size_values = size_values + [size_values[-1]] * (num_frames - len(size_values))
    else:
        # 没有官方 size，使用占位值
        # size > 0.02 时不进行 swap
        size_values = [0.5] * num_frames
        size_source = 'placeholder'
    
    with open(sample_dir / "size.json", "w") as f:
        json.dump(size_values, f)
    
    # 保存元数据 (用于调试)
    meta = {
        'vid': vid,
        'transcript': transcript,
        'clip_path': str(clip_path),
        'num_frames': frames.shape[0],
        'frame_size': [frame_height, frame_width],
        'size_source': size_source,
        'size_values': size_values[:3] if len(size_values) > 3 else size_values,  # 只保存前几个用于调试
    }
    with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return sample_dir


def main():
    parser = argparse.ArgumentParser(description="CLIP4MC 数据准备")
    
    parser.add_argument(
        "--pairs-json",
        type=Path,
        default=Path("data/raw_videos/clip4mc_youtube/text_video_pairs.json"),
        help="text_video_pairs.json 文件路径"
    )
    
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/raw_videos/clip4mc_youtube/clips"),
        help="视频切片目录"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/nanzhang/clip4mc/processed_data"),
        help="输出目录 (CLIP4MC processed_data 目录)"
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="每个视频提取的帧数"
    )
    
    parser.add_argument(
        "--frame-height",
        type=int,
        default=160,
        help="帧高度 (CLIP4MC 官方: 160)"
    )
    
    parser.add_argument(
        "--frame-width",
        type=int,
        default=256,
        help="帧宽度 (CLIP4MC 官方: 256)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例 (仅当 --split-mode=random 时使用)"
    )
    
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=['random', 'all_train', 'all_test'],
        default='random',
        help="数据集划分模式: random=随机划分, all_train=全部作为训练集, all_test=全部作为测试集"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数 (用于测试)"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if not HAS_CV2 and not HAS_IMAGEIO:
        logger.error("需要安装 opencv-python 或 imageio")
        logger.error("  pip install opencv-python")
        logger.error("  或")
        logger.error("  pip install imageio imageio-ffmpeg")
        sys.exit(1)
    
    # 加载数据
    if not args.pairs_json.exists():
        logger.error(f"pairs JSON 文件不存在: {args.pairs_json}")
        sys.exit(1)
    
    with open(args.pairs_json, encoding="utf-8") as f:
        pairs = json.load(f)
    
    logger.info(f"加载了 {len(pairs)} 个文本-视频对")
    
    # 限制样本数
    if args.max_samples and len(pairs) > args.max_samples:
        random.seed(args.seed)
        pairs = random.sample(pairs, args.max_samples)
        logger.info(f"限制为 {args.max_samples} 个样本")
    
    # 创建输出目录 (直接在 output_dir 下创建样本，不再有 samples 子目录)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理样本
    successful_dirs: List[str] = []
    failed_count = 0
    
    logger.info(f"\n{'='*60}")
    logger.info("开始处理样本 (CLIP4MC 官方格式)...")
    logger.info(f"  帧尺寸: {args.frame_height}x{args.frame_width}")
    logger.info(f"  帧数: {args.num_frames}")
    logger.info(f"{'='*60}\n")
    
    iterator = enumerate(pairs)
    if tqdm:
        iterator = tqdm(list(iterator), desc="🎬 处理视频", unit="sample")
    
    for idx, pair in iterator:
        result = prepare_sample(
            pair=pair,
            clips_dir=args.clips_dir,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            frame_height=args.frame_height,
            frame_width=args.frame_width,
            sample_idx=idx
        )
        
        if result:
            successful_dirs.append(str(result))
        else:
            failed_count += 1
    
    logger.info(f"\n处理完成: 成功 {len(successful_dirs)}, 失败 {failed_count}")
    
    if not successful_dirs:
        logger.error("没有成功处理任何样本")
        sys.exit(1)
    
    # 划分训练集、验证集和测试集 (CLIP4MC 官方格式)
    random.seed(args.seed)
    random.shuffle(successful_dirs)
    
    n = len(successful_dirs)
    n_test = max(1, int(n * 0.1))
    n_val = max(1, int(n * 0.1))
    n_train = n - n_test - n_val
    
    train_dirs = successful_dirs[:n_train]
    val_dirs = successful_dirs[n_train:n_train + n_val]
    test_dirs = successful_dirs[n_train + n_val:]
    
    logger.info(f"训练集: {len(train_dirs)} 个样本")
    logger.info(f"验证集: {len(val_dirs)} 个样本")
    logger.info(f"测试集: {len(test_dirs)} 个样本")
    
    # 生成 dataset_info.json (CLIP4MC 官方格式)
    dataset_info = {
        "train": train_dirs,
        "val": val_dirs,
        "test": test_dirs
    }
    
    dataset_info_path = args.output_dir / "dataset_info.json"
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✓ dataset_info.json 已保存: {dataset_info_path}")
    
    # 打印使用说明
    logger.info(f"\n{'='*60}")
    logger.info("数据准备完成！使用以下命令开始训练:")
    logger.info(f"{'='*60}")
    logger.info(f"\ncd /Users/nanzhang/clip4mc")
    logger.info(f"conda activate minedojo-x86")
    logger.info(f"python train_ddp_clip4mc.py --batch_size 2 --epochs 5")


if __name__ == "__main__":
    main()

