#!/usr/bin/env python3
"""
生成 Decord Pipeline 所需的元数据文件

功能：
1. 读取 dataset_test.json 和 dataset_train_LocalCorrelationFilter.json
2. 读取 youtube_download_log.csv（vid 到文件名映射）
3. 匹配视频文件（支持网盘文件名兼容）
4. 生成 text_input.pkl（使用 transformers AutoTokenizer，与 CLIP4MC 官方一致）
5. 输出元数据 JSON 文件

使用示例：
    # 单进程模式（默认）
    python src/utils/generate_clip4mc_metadata.py \\
        --test-json data/training/dataset_test.json \\
        --download-log data/training/youtube_download_log.csv \\
        --videos-dir /path/to/videos \\
        --text-inputs-dir /path/to/text_inputs \\
        --output metadata.json \\
        --loose-match
    
    # 多进程模式（推荐，处理大量数据时）
    python src/utils/generate_clip4mc_metadata.py \\
        --test-json data/training/dataset_test.json \\
        --train-json data/training/dataset_train_LocalCorrelationFilter.json \\
        --download-log data/training/youtube_download_log.csv \\
        --videos-dir /path/to/videos \\
        --text-inputs-dir /path/to/text_inputs \\
        --output metadata.json \\
        --loose-match \\
        --num-workers 8 \\
        --unmatched-output unmatched.json

依赖：
    - transformers: pip install transformers
"""

import json
import csv
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoClip:
    """视频片段元数据"""
    vid: str
    transcript: str
    start_time: float
    end_time: float
    size: List[float]
    data_type: str  # 'train', 'test'


def normalize_title_for_filename(title: str, remove_punctuation: bool = False) -> str:
    """
    规范化标题以匹配文件名
    
    文件系统会移除或替换某些字符：
    - 移除 emoji 和其他非ASCII字符
    - 点号可能被空格替换
    - 网盘特殊字符替换
    
    Args:
        title: 标题
        remove_punctuation: 是否将标点符号替换为空格（用于宽松匹配）
    """
    import re
    
    # 移除 emoji 和其他非ASCII可打印字符（保留常见符号）
    # 只保留 ASCII 可打印字符 + 空格
    normalized = ''.join(
        c if (32 <= ord(c) < 127) else ' ' 
        for c in title
    )
    
    # 如果需要，将标点符号替换为空格（用于宽松匹配）
    if remove_punctuation:
        # 将常见标点符号替换为空格
        normalized = re.sub(r'[\.,:;!?\'"()\[\]{}\-_/\\|]', ' ', normalized)
    
    # 合并多个空格
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def normalize_for_ultra_loose_match(text: str) -> str:
    """
    超宽松匹配：只保留字母和数字
    
    用于处理文件名中移除了所有特殊字符的情况：
    - CASA *ALEX* → casa alex
    - $1,000 → 1 000
    - EXPÉRIENCE: (NFD) → experience (移除重音符号)
    
    Args:
        text: 原始文本
    
    Returns:
        只包含字母、数字和空格的小写文本
    """
    import re
    import unicodedata
    
    # 1. Unicode 规范化 (NFD) + 移除重音符号
    nfd = unicodedata.normalize('NFD', text)
    without_accents = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    # 2. 只保留字母、数字和空格
    normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', without_accents)
    
    # 3. 合并多个空格
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized.lower()


def normalize_netdisk_filename(name: str) -> str:
    """
    规范化网盘转存后的文件名（反向转换特殊字符）
    
    网盘（如百度网盘）在转存文件时会替换文件系统非法字符：
    - ⧸ (U+29F8) → /
    - ？ (U+FF1F) → ?
    - ： (U+FF1A) → :
    - &#39; → '
    - ｜ (U+FF5C) → |
    - ＊ (U+FF0A) → *
    - ＂ (U+FF02) → "
    """
    replacements = [
        ('⧸', '/'),   # BIG SOLIDUS → SOLIDUS
        ('？', '?'),   # FULLWIDTH QUESTION MARK → QUESTION MARK
        ('：', ':'),   # FULLWIDTH COLON → COLON
        ('&#39;', "'"), # HTML ENTITY → APOSTROPHE
        ('｜', '|'),   # FULLWIDTH VERTICAL LINE → VERTICAL LINE
        ('＊', '*'),   # FULLWIDTH ASTERISK → ASTERISK
        ('＂', '"'),   # FULLWIDTH QUOTATION MARK → QUOTATION MARK
    ]
    
    for old, new in replacements:
        name = name.replace(old, new)
    
    return name


def extract_vid_from_url(url: str) -> Optional[str]:
    """从 YouTube URL 中提取 video ID"""
    try:
        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc:
            query_params = parse_qs(parsed.query)
            return query_params.get('v', [None])[0]
        elif 'youtu.be' in parsed.netloc:
            return parsed.path.lstrip('/')
    except Exception as e:
        logger.warning(f"解析 URL 失败: {url} - {str(e)}")
    return None


def load_download_log(csv_path: Path) -> Dict[str, str]:
    """
    加载 youtube_download_log.csv，构建 vid -> title 映射
    
    Returns:
        {vid: title}
    """
    vid_to_title = {}
    
    # 使用 utf-8-sig 自动移除 BOM（Byte Order Mark）
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('url', '')
            title = row.get('title', '')
            status = row.get('status', 'False')
            
            if status != 'True':
                continue
            
            vid = extract_vid_from_url(url)
            if vid and title:
                vid_to_title[vid] = title
    
    logger.info(f"加载了 {len(vid_to_title)} 条下载记录")
    return vid_to_title


def find_video_file(vid: str, title: str, videos_dir: Path, use_loose_match: bool = False) -> Optional[Path]:
    """
    根据 vid 和 title 查找视频文件
    
    匹配策略：
    - 不使用宽松匹配（默认，3层）：
      1. 直接匹配 title.mp4
      2. 规范化后匹配（移除 emoji）
      3. vid 匹配（文件名包含 vid）
    
    - 使用宽松匹配（--loose-match，7层）：
      1. 直接匹配 title.mp4
      2. 规范化后匹配（移除 emoji）
      3. 宽松匹配（移除 emoji + 标点符号）
      4. 超宽松匹配（只保留字母和数字）
      5. 规范化 + 网盘字符替换
      6. vid 匹配（文件名包含 vid）
      7. 模糊匹配（title 前 30 个字符）
    
    Args:
        vid: 视频 ID
        title: 视频标题
        videos_dir: 视频目录
        use_loose_match: 是否使用宽松匹配（默认: False）
    """
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.mov', '.webm']
    
    # 收集所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f'*{ext}'))
    
    if not video_files:
        return None
    
    # 策略 1: 直接匹配（总是启用）
    for video_file in video_files:
        if video_file.stem == title:
            return video_file
    
    # 策略 2: 规范化后匹配（总是启用）
    normalized_title = normalize_title_for_filename(title, remove_punctuation=False)
    for video_file in video_files:
        normalized_filename = normalize_title_for_filename(video_file.stem, remove_punctuation=False)
        if normalized_filename == normalized_title:
            return video_file
    
    # 以下策略仅在 use_loose_match=True 时启用
    if use_loose_match:
        # 策略 3: 宽松匹配（移除 emoji + 标点符号）
        loose_title = normalize_title_for_filename(title, remove_punctuation=True)
        for video_file in video_files:
            loose_filename = normalize_title_for_filename(video_file.stem, remove_punctuation=True)
            if loose_filename == loose_title:
                return video_file
        
        # 策略 4: 超宽松匹配（只保留字母和数字）
        ultra_loose_title = normalize_for_ultra_loose_match(title)
        for video_file in video_files:
            ultra_loose_filename = normalize_for_ultra_loose_match(video_file.stem)
            if ultra_loose_filename == ultra_loose_title:
                return video_file
        
        # 策略 5: 规范化 + 网盘字符替换
        normalized_title_netdisk = normalize_netdisk_filename(normalized_title)
        for video_file in video_files:
            normalized_filename = normalize_title_for_filename(video_file.stem, remove_punctuation=False)
            normalized_filename_netdisk = normalize_netdisk_filename(normalized_filename)
            if normalized_filename_netdisk == normalized_title_netdisk:
                return video_file
        
        # 策略 6: vid 匹配
        for video_file in video_files:
            if vid in video_file.stem:
                return video_file
        
        # 策略 7: 模糊匹配（前 30 个字符）
        title_prefix = normalize_title_for_filename(title, remove_punctuation=True)[:30].lower()
        for video_file in video_files:
            file_prefix = normalize_title_for_filename(video_file.stem, remove_punctuation=True)[:30].lower()
            if title_prefix == file_prefix:
                return video_file
    else:
        # 默认模式：只使用 vid 匹配作为最后的尝试
        # 策略 3: vid 匹配
        for video_file in video_files:
            if vid in video_file.stem:
                return video_file
    
    return None


def load_dataset_clips(
    test_json_path: Optional[Path],
    train_json_path: Optional[Path]
) -> List[VideoClip]:
    """
    加载 dataset_test.json 和 dataset_train_LocalCorrelationFilter.json
    
    Returns:
        List[VideoClip]
    """
    clips = []
    
    # 加载 test 数据
    if test_json_path and test_json_path.exists():
        logger.info(f"加载 test 数据: {test_json_path}")
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        skipped_count = 0
        for item in test_data:
            # 获取时间，确保 None 被转换为默认值
            start_time = item.get('begin position')
            end_time = item.get('end position')
            
            # 跳过无效片段（时间为 None）
            if start_time is None or end_time is None:
                skipped_count += 1
                continue
            
            clip = VideoClip(
                vid=item.get('vid', ''),
                transcript=item.get('transcript clip', ''),
                start_time=start_time,
                end_time=end_time,
                size=item.get('size', []),
                data_type='test'
            )
            clips.append(clip)
        
        logger.info(f"加载了 {len(test_data)} 条 test 数据（有效: {len(clips)}, 跳过: {skipped_count}）")
    
    # 加载 train 数据
    if train_json_path and train_json_path.exists():
        logger.info(f"加载 train 数据: {train_json_path}")
        
        # train 数据可能是流式 JSON（每行一个 JSON 对象）
        train_data = []
        with open(train_json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 尝试直接解析为 JSON 数组
            if content.startswith('['):
                train_data = json.loads(content)
            else:
                # 流式 JSON，每行一个对象
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            train_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        train_skipped_count = 0
        train_valid_count = 0
        for item in train_data:
            # 获取时间，确保 None 被转换为默认值
            start_time = item.get('begin position')
            end_time = item.get('end position')
            
            # 跳过无效片段（时间为 None）
            if start_time is None or end_time is None:
                train_skipped_count += 1
                continue
            
            clip = VideoClip(
                vid=item.get('vid', ''),
                transcript=item.get('transcript clip', ''),
                start_time=start_time,
                end_time=end_time,
                size=item.get('size', []),
                data_type='train'
            )
            clips.append(clip)
            train_valid_count += 1
        
        logger.info(f"加载了 {len(train_data)} 条 train 数据（有效: {train_valid_count}, 跳过: {train_skipped_count}）")
    
    return clips


def generate_text_input_pkl(
    transcript: str,
    output_path: Path
) -> bool:
    """
    使用 transformers AutoTokenizer 生成 text_input.pkl（与 CLIP4MC 官方一致）
    
    Args:
        transcript: 文本描述
        output_path: 输出 .pkl 文件路径
    
    Returns:
        是否成功
    """
    try:
        from transformers import AutoTokenizer
        
        # 获取 tokenizer（单例，只初始化一次）
        # 使用与 CLIP4MC 官方相同的 tokenizer
        if not hasattr(generate_text_input_pkl, 'tokenizer'):
            generate_text_input_pkl.tokenizer = AutoTokenizer.from_pretrained(
                'openai/clip-vit-base-patch16'
            )
        
        tokenizer = generate_text_input_pkl.tokenizer
        
        # Tokenize（与官方格式一致）
        tokens = tokenizer(
            transcript,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tokens_np = tokens['input_ids'][0].numpy()
        
        # 保存为 CLIP4MC 格式
        with open(output_path, 'wb') as f:
            pickle.dump({'tokens': tokens_np}, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return True
    
    except Exception as e:
        logger.error(f"生成 text_input.pkl 失败: {str(e)}")
        return False


def process_single_clip(
    clip: VideoClip,
    vid_to_title: Dict[str, str],
    videos_dir: Path,
    text_inputs_dir: Path,
    use_loose_match: bool,
    video_prefix: str,
    text_prefix: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    处理单个视频片段（用于多进程）
    
    Args:
        clip: 视频片段
        vid_to_title: vid 到 title 的映射
        videos_dir: 视频目录
        text_inputs_dir: text_input.pkl 输出目录
        use_loose_match: 是否使用宽松匹配
        video_prefix: 视频路径前缀
        text_prefix: text_input.pkl 路径前缀
    
    Returns:
        (metadata_item, unmatched_item) 元组
        - 如果成功: (metadata_item, None)
        - 如果失败: (None, unmatched_item)
    """
    # 获取 title
    title = vid_to_title.get(clip.vid)
    if not title:
        return None, {
            'vid': clip.vid,
            'reason': 'no_download_record',
            'message': '未找到下载记录'
        }
    
    # 查找视频文件
    video_file = find_video_file(clip.vid, title, videos_dir, use_loose_match=use_loose_match)
    if not video_file:
        return None, {
            'vid': clip.vid,
            'title': title,
            'reason': 'video_not_found',
            'message': '未找到视频文件'
        }
    
    # 生成 text_input.pkl
    text_input_path = text_inputs_dir / f"{clip.vid}_text_input.pkl"
    
    if not text_input_path.exists():
        success = generate_text_input_pkl(clip.transcript, text_input_path)
        if not success:
            return None, {
                'vid': clip.vid,
                'title': title,
                'reason': 'tokenization_failed',
                'message': '生成 text_input.pkl 失败'
            }
    
    # 构建路径（使用 prefix）
    video_path = str(video_file.absolute())
    if video_prefix:
        video_path = video_prefix.rstrip('/') + '/' + video_file.name
    
    text_path = str(text_input_path.absolute())
    if text_prefix:
        text_path = text_prefix.rstrip('/') + '/' + text_input_path.name
    
    # 构建元数据项
    metadata_item = {
        'vid': clip.vid,
        'video_path': video_path,
        'start_time': clip.start_time,
        'end_time': clip.end_time,
        'transcript': clip.transcript,
        'size': clip.size,
        'data_type': clip.data_type,
        'text_input_path': text_path
    }
    
    return metadata_item, None


def main():
    parser = argparse.ArgumentParser(
        description="生成 Decord Pipeline 元数据文件",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 必需参数
    parser.add_argument("--download-log", type=Path, required=True,
                       help="youtube_download_log.csv 路径")
    parser.add_argument("--videos-dir", type=Path, required=True,
                       help="视频文件目录")
    parser.add_argument("--text-inputs-dir", type=Path, required=True,
                       help="text_input.pkl 输出目录")
    parser.add_argument("--output", type=Path, required=True,
                       help="输出元数据 JSON 文件路径")
    
    # 可选参数
    parser.add_argument("--test-json", type=Path,
                       help="dataset_test.json 路径")
    parser.add_argument("--train-json", type=Path,
                       help="dataset_train_LocalCorrelationFilter.json 路径")
    
    # 匹配参数
    parser.add_argument("--loose-match", action='store_true',
                       help="启用宽松匹配（移除特殊字符、emoji、标点符号）")
    
    # 路径前缀参数
    parser.add_argument("--video-prefix", type=str, default="",
                       help="视频路径前缀（例如：/mnt/data/）")
    parser.add_argument("--text-prefix", type=str, default="",
                       help="text_input.pkl 路径前缀（例如：/mnt/data/）")
    
    # 未匹配文件输出
    parser.add_argument("--unmatched-output", type=Path,
                       help="未匹配文件和 vid 清单输出路径（JSON 格式）")
    
    # 多进程参数
    parser.add_argument("--num-workers", type=int, default=1,
                       help="并行处理的进程数（默认: 1，建议: CPU核心数）")
    
    args = parser.parse_args()
    
    # 验证输入
    if not args.test_json and not args.train_json:
        logger.error("必须至少指定 --test-json 或 --train-json 之一")
        return 1
    
    if not args.download_log.exists():
        logger.error(f"下载日志不存在: {args.download_log}")
        return 1
    
    if not args.videos_dir.exists():
        logger.error(f"视频目录不存在: {args.videos_dir}")
        return 1
    
    # 创建输出目录
    args.text_inputs_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("开始生成 Decord Pipeline 元数据")
    logger.info("=" * 60)
    
    # 1. 加载下载日志
    logger.info("步骤 1: 加载下载日志...")
    vid_to_title = load_download_log(args.download_log)
    
    # 2. 加载数据集
    logger.info("步骤 2: 加载数据集...")
    clips = load_dataset_clips(args.test_json, args.train_json)
    logger.info(f"总共 {len(clips)} 条片段")
    
    # 3. 匹配视频文件并生成元数据
    logger.info("步骤 3: 匹配视频文件并生成 text_input.pkl...")
    logger.info(f"  使用 {args.num_workers} 个进程并行处理")
    
    metadata = []
    matched_count = 0
    failed_count = 0
    unmatched_items = []  # 收集未匹配的项
    
    if args.num_workers == 1:
        # 单进程模式
        for clip in tqdm(clips, desc="处理进度", unit="clip"):
            metadata_item, unmatched_item = process_single_clip(
                clip, vid_to_title, args.videos_dir, args.text_inputs_dir,
                args.loose_match, args.video_prefix, args.text_prefix
            )
            
            if metadata_item:
                metadata.append(metadata_item)
                matched_count += 1
            else:
                unmatched_items.append(unmatched_item)
                failed_count += 1
    else:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    process_single_clip,
                    clip, vid_to_title, args.videos_dir, args.text_inputs_dir,
                    args.loose_match, args.video_prefix, args.text_prefix
                ): clip for clip in clips
            }
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(clips), desc="处理进度", unit="clip") as pbar:
                for future in as_completed(futures):
                    try:
                        metadata_item, unmatched_item = future.result()
                        
                        if metadata_item:
                            metadata.append(metadata_item)
                            matched_count += 1
                        else:
                            unmatched_items.append(unmatched_item)
                            failed_count += 1
                    except Exception as e:
                        clip = futures[future]
                        logger.error(f"处理 clip {clip.vid} 失败: {str(e)}")
                        unmatched_items.append({
                            'vid': clip.vid,
                            'reason': 'processing_error',
                            'message': f'处理异常: {str(e)}'
                        })
                        failed_count += 1
                    
                    pbar.update(1)
    
    # 4. 保存元数据
    logger.info("步骤 4: 保存元数据...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 5. 保存未匹配清单（如果指定）
    if args.unmatched_output and unmatched_items:
        logger.info("步骤 5: 保存未匹配清单...")
        with open(args.unmatched_output, 'w', encoding='utf-8') as f:
            json.dump(unmatched_items, f, indent=2, ensure_ascii=False)
        logger.info(f"  未匹配清单: {args.unmatched_output}")
    
    # 完成
    logger.info("=" * 60)
    logger.info("元数据生成完成！")
    logger.info(f"  总片段数: {len(clips)}")
    logger.info(f"  匹配成功: {matched_count}")
    logger.info(f"  匹配失败: {failed_count}")
    logger.info(f"  输出文件: {args.output}")
    if args.unmatched_output and unmatched_items:
        logger.info(f"  未匹配文件: {args.unmatched_output} ({len(unmatched_items)} 项)")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

