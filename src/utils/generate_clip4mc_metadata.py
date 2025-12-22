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
    
    # 性能测试模式（跳过 text token 生成）
    python src/utils/generate_clip4mc_metadata.py \\
        --test-json data/training/dataset_test.json \\
        --download-log data/training/youtube_download_log.csv \\
        --videos-dir /path/to/videos \\
        --text-inputs-dir /tmp/text_inputs \\
        --output /tmp/metadata.json \\
        --loose-match \\
        --num-workers 16 \\
        --skip-text-generation

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
from multiprocessing import Pool
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class VideoClip:
    """视频片段元数据"""
    vid: str
    transcript: str
    start_time: float
    end_time: float
    size: List[float]
    data_type: str  # 'train', 'test'


# ============================================================
# 全局变量（用于进程池初始化）
# ============================================================
_worker_vid_to_title = None
_worker_video_files = None
_worker_text_inputs_dir = None
_worker_use_loose_match = None
_worker_video_prefix = None
_worker_text_prefix = None
_worker_skip_text_generation = None
_worker_video_index = None  # 预计算的视频文件索引（性能优化）


def _init_metadata_worker(
    vid_to_title: Dict[str, str],
    video_files: List[Path],
    text_inputs_dir: Path,
    use_loose_match: bool,
    video_prefix: str,
    text_prefix: str,
    skip_text_generation: bool
):
    """
    初始化 worker 进程（每个进程只调用一次）
    
    Args:
        vid_to_title: vid 到 title 的映射
        video_files: 视频文件列表
        text_inputs_dir: text_input.pkl 输出目录
        use_loose_match: 是否使用宽松匹配
        video_prefix: 视频路径前缀
        text_prefix: text_input.pkl 路径前缀
        skip_text_generation: 是否跳过 text token 生成
    """
    global _worker_vid_to_title, _worker_video_files, _worker_text_inputs_dir
    global _worker_use_loose_match, _worker_video_prefix, _worker_text_prefix
    global _worker_skip_text_generation, _worker_video_index
    
    _worker_vid_to_title = vid_to_title
    _worker_video_files = video_files
    _worker_text_inputs_dir = text_inputs_dir
    _worker_use_loose_match = use_loose_match
    _worker_video_prefix = video_prefix
    _worker_text_prefix = text_prefix
    _worker_skip_text_generation = skip_text_generation
    
    # 🚀 性能优化：预计算视频文件索引（O(n) → O(1) 查找）
    # 这将查找从线性遍历（2092次）变为字典查找（1次）
    _worker_video_index = _build_video_index(video_files, use_loose_match)


def _build_video_index(video_files: List[Path], use_loose_match: bool) -> Dict:
    """
    构建视频文件索引（预计算所有 normalized 版本）
    
    这是性能优化的关键：
    - 将 O(n) 线性查找优化为 O(1) 字典查找
    - 避免重复计算 normalized 版本
    - 在 worker 初始化时只计算一次
    
    重要：对所有文件名先应用 normalize_netdisk_filename，
    以处理云存储/网盘的特殊字符替换（全角符号、HTML实体等）
    
    Args:
        video_files: 视频文件列表
        use_loose_match: 是否使用宽松匹配
    
    Returns:
        索引字典，包含多个匹配策略的索引
    """
    index = {
        'direct': {},           # 策略 1: 直接匹配
        'normalized': {},       # 策略 2: 规范化匹配
        'vid_contains': [],     # 最后策略: vid 包含匹配（仍需遍历）
    }
    
    # 如果启用宽松匹配，添加额外索引
    if use_loose_match:
        index['loose'] = {}           # 策略 3: 宽松匹配
        index['ultra_loose'] = {}     # 策略 4: 超宽松匹配
    
    # 预计算所有文件的 normalized 版本（显示进度）
    for video_file in tqdm(video_files, desc="🔨 构建视频索引", unit="file", leave=False):
        stem = video_file.stem
        
        # 🔧 关键修复：先应用网盘字符规范化
        # 将全角符号、HTML实体等转换为标准字符
        stem_normalized_netdisk = normalize_netdisk_filename(stem)
        
        # 策略 1: 直接匹配索引（使用规范化后的文件名）
        index['direct'][stem_normalized_netdisk] = video_file
        
        # 策略 2: 规范化匹配索引
        normalized = normalize_title_for_filename(stem_normalized_netdisk, remove_punctuation=False)
        if normalized not in index['normalized']:  # 避免覆盖（保留第一个匹配）
            index['normalized'][normalized] = video_file
        
        # 宽松匹配索引（仅在启用时）
        if use_loose_match:
            # 策略 3: 宽松匹配（移除标点符号）
            loose = normalize_title_for_filename(stem_normalized_netdisk, remove_punctuation=True)
            if loose not in index['loose']:
                index['loose'][loose] = video_file
            
            # 策略 4: 超宽松匹配（只保留字母和数字）
            ultra_loose = normalize_for_ultra_loose_match(stem_normalized_netdisk)
            if ultra_loose not in index['ultra_loose']:
                index['ultra_loose'][ultra_loose] = video_file
        
        # vid 包含匹配（需要遍历，使用原始 stem 以保留 vid）
        index['vid_contains'].append((video_file, stem))
    
    return index


def _process_clip_worker(clip: VideoClip) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Worker 函数：处理单个 clip（使用全局变量）
    
    Args:
        clip: 视频片段
    
    Returns:
        (metadata_item, unmatched_item)
    """
    global _worker_vid_to_title, _worker_video_files, _worker_text_inputs_dir
    global _worker_use_loose_match, _worker_video_prefix, _worker_text_prefix
    global _worker_skip_text_generation
    
    return process_single_clip(
        clip,
        _worker_vid_to_title,
        _worker_video_files,
        _worker_text_inputs_dir,
        _worker_use_loose_match,
        _worker_video_prefix,
        _worker_text_prefix,
        _worker_skip_text_generation
    )


# ============================================================
# 辅助函数
# ============================================================


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
    
    网盘（如百度网盘）和云存储在转存文件时会替换文件系统非法字符。
    这个函数将这些替换字符转回原始字符，以便匹配。
    
    支持的转换：
    - 全角标点符号 → 半角
    - 特殊Unicode字符 → 标准字符
    - HTML实体 → 原始字符
    """
    replacements = [
        # 特殊Unicode字符（优先处理）
        ('⧸', '/'),    # BIG SOLIDUS (U+29F8) → SOLIDUS
        ('⁄', '/'),    # FRACTION SLASH (U+2044) → SOLIDUS
        ('∕', '/'),    # DIVISION SLASH (U+2215) → SOLIDUS
        
        # 全角标点符号 → 半角（最常见）
        ('！', '!'),   # FULLWIDTH EXCLAMATION MARK
        ('＂', '"'),   # FULLWIDTH QUOTATION MARK
        ('＃', '#'),   # FULLWIDTH NUMBER SIGN
        ('＄', '$'),   # FULLWIDTH DOLLAR SIGN
        ('％', '%'),   # FULLWIDTH PERCENT SIGN
        ('＆', '&'),   # FULLWIDTH AMPERSAND
        ('＇', "'"),   # FULLWIDTH APOSTROPHE
        ('（', '('),   # FULLWIDTH LEFT PARENTHESIS
        ('）', ')'),   # FULLWIDTH RIGHT PARENTHESIS
        ('＊', '*'),   # FULLWIDTH ASTERISK
        ('＋', '+'),   # FULLWIDTH PLUS SIGN
        ('，', ','),   # FULLWIDTH COMMA
        ('－', '-'),   # FULLWIDTH HYPHEN-MINUS
        ('．', '.'),   # FULLWIDTH FULL STOP
        ('／', '/'),   # FULLWIDTH SOLIDUS
        ('：', ':'),   # FULLWIDTH COLON
        ('；', ';'),   # FULLWIDTH SEMICOLON
        ('＜', '<'),   # FULLWIDTH LESS-THAN SIGN
        ('＝', '='),   # FULLWIDTH EQUALS SIGN
        ('＞', '>'),   # FULLWIDTH GREATER-THAN SIGN
        ('？', '?'),   # FULLWIDTH QUESTION MARK
        ('＠', '@'),   # FULLWIDTH COMMERCIAL AT
        ('［', '['),   # FULLWIDTH LEFT SQUARE BRACKET
        ('＼', '\\'),  # FULLWIDTH REVERSE SOLIDUS
        ('］', ']'),   # FULLWIDTH RIGHT SQUARE BRACKET
        ('＾', '^'),   # FULLWIDTH CIRCUMFLEX ACCENT
        ('＿', '_'),   # FULLWIDTH LOW LINE
        ('｀', '`'),   # FULLWIDTH GRAVE ACCENT
        ('｛', '{'),   # FULLWIDTH LEFT CURLY BRACKET
        ('｜', '|'),   # FULLWIDTH VERTICAL LINE
        ('｝', '}'),   # FULLWIDTH RIGHT CURLY BRACKET
        ('～', '~'),   # FULLWIDTH TILDE
        
        # HTML实体编码
        ('&#39;', "'"),  # APOSTROPHE
        ('&quot;', '"'), # QUOTATION MARK
        ('&amp;', '&'),  # AMPERSAND
        ('&lt;', '<'),   # LESS-THAN
        ('&gt;', '>'),   # GREATER-THAN
        
        # 其他常见替换
        (''', "'"),   # LEFT SINGLE QUOTATION MARK
        (''', "'"),   # RIGHT SINGLE QUOTATION MARK
        ('"', '"'),   # LEFT DOUBLE QUOTATION MARK
        ('"', '"'),   # RIGHT DOUBLE QUOTATION MARK
        ('—', '-'),   # EM DASH
        ('–', '-'),   # EN DASH
        ('…', '...'), # HORIZONTAL ELLIPSIS
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
    
    logger.info(f"  构建 {len(vid_to_title)} 条 vid→title 映射")
    return vid_to_title


def find_video_file(vid: str, title: str, video_files: List[Path], use_loose_match: bool = False) -> Optional[Path]:
    """
    根据 vid 和 title 查找视频文件（使用预计算索引优化）
    
    性能优化：
    - 在多进程环境下，使用 _worker_video_index 预计算索引（O(1) 查找）
    - 在单进程环境下，使用传统线性查找（向后兼容）
    
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
        video_files: 预先收集的视频文件列表
        use_loose_match: 是否使用宽松匹配（默认: False）
    """
    global _worker_video_index
    
    if not video_files:
        return None
    
    # 🚀 使用预计算索引（如果可用）
    if _worker_video_index is not None:
        return _find_video_file_with_index(vid, title, _worker_video_index, use_loose_match)
    
    # 传统线性查找（向后兼容，单进程模式）
    return _find_video_file_linear(vid, title, video_files, use_loose_match)


def _find_video_file_with_index(
    vid: str,
    title: str,
    index: Dict,
    use_loose_match: bool
) -> Optional[Path]:
    """
    使用预计算索引查找视频文件（O(1) 字典查找）
    
    重要：title 不需要应用 normalize_netdisk_filename，
    因为索引构建时已经对文件名应用了该函数。
    直接比较即可。
    
    Args:
        vid: 视频 ID
        title: 视频标题（来自 CSV，标准格式）
        index: 预计算的索引字典
        use_loose_match: 是否使用宽松匹配
    
    Returns:
        视频文件路径，未找到则返回 None
    """
    # 策略 1: 直接匹配 O(1)
    # title 和索引中的 key 都是标准格式，直接比较
    if title in index['direct']:
        return index['direct'][title]
    
    # 策略 2: 规范化匹配 O(1)
    normalized_title = normalize_title_for_filename(title, remove_punctuation=False)
    if normalized_title in index['normalized']:
        return index['normalized'][normalized_title]
    
    # 宽松匹配策略（仅在启用时）
    if use_loose_match:
        # 策略 3: 宽松匹配 O(1)
        loose_title = normalize_title_for_filename(title, remove_punctuation=True)
        if loose_title in index['loose']:
            return index['loose'][loose_title]
        
        # 策略 4: 超宽松匹配 O(1)
        ultra_loose_title = normalize_for_ultra_loose_match(title)
        if ultra_loose_title in index['ultra_loose']:
            return index['ultra_loose'][ultra_loose_title]
        
        # 策略 5: vid 包含匹配 O(n) - 仍需遍历，但已预存 stem
        for video_file, stem in index['vid_contains']:
            if vid in stem:
                return video_file
        
        # 策略 6: 模糊匹配（前 30 个字符）O(n)
        title_prefix = normalize_title_for_filename(title, remove_punctuation=True)[:30].lower()
        for video_file, stem in index['vid_contains']:
            # 对 stem 也应用网盘规范化后再比较
            stem_normalized = normalize_netdisk_filename(stem)
            file_prefix = normalize_title_for_filename(stem_normalized, remove_punctuation=True)[:30].lower()
            if title_prefix == file_prefix:
                return video_file
    else:
        # 默认模式：vid 包含匹配
        for video_file, stem in index['vid_contains']:
            if vid in stem:
                return video_file
    
    return None


def _find_video_file_linear(
    vid: str,
    title: str,
    video_files: List[Path],
    use_loose_match: bool
) -> Optional[Path]:
    """
    传统线性查找（向后兼容）
    
    重要：对文件名应用网盘字符规范化以匹配索引版本的行为
    
    Args:
        vid: 视频 ID
        title: 视频标题
        video_files: 视频文件列表
        use_loose_match: 是否使用宽松匹配
    
    Returns:
        视频文件路径，未找到则返回 None
    """
    # 策略 1: 直接匹配（总是启用）
    for video_file in video_files:
        stem_normalized = normalize_netdisk_filename(video_file.stem)
        if stem_normalized == title:
            return video_file
    
    # 策略 2: 规范化后匹配（总是启用）
    normalized_title = normalize_title_for_filename(title, remove_punctuation=False)
    for video_file in video_files:
        stem_normalized = normalize_netdisk_filename(video_file.stem)
        normalized_filename = normalize_title_for_filename(stem_normalized, remove_punctuation=False)
        if normalized_filename == normalized_title:
            return video_file
    
    # 以下策略仅在 use_loose_match=True 时启用
    if use_loose_match:
        # 策略 3: 宽松匹配（移除 emoji + 标点符号）
        loose_title = normalize_title_for_filename(title, remove_punctuation=True)
        for video_file in video_files:
            stem_normalized = normalize_netdisk_filename(video_file.stem)
            loose_filename = normalize_title_for_filename(stem_normalized, remove_punctuation=True)
            if loose_filename == loose_title:
                return video_file
        
        # 策略 4: 超宽松匹配（只保留字母和数字）
        ultra_loose_title = normalize_for_ultra_loose_match(title)
        for video_file in video_files:
            stem_normalized = normalize_netdisk_filename(video_file.stem)
            ultra_loose_filename = normalize_for_ultra_loose_match(stem_normalized)
            if ultra_loose_filename == ultra_loose_title:
                return video_file
        
        # 策略 5: vid 匹配
        for video_file in video_files:
            if vid in video_file.stem:
                return video_file
        
        # 策略 6: 模糊匹配（前 30 个字符）
        title_prefix = normalize_title_for_filename(title, remove_punctuation=True)[:30].lower()
        for video_file in video_files:
            stem_normalized = normalize_netdisk_filename(video_file.stem)
            file_prefix = normalize_title_for_filename(stem_normalized, remove_punctuation=True)[:30].lower()
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
        logger.info(f"  加载 test 数据: {test_json_path}")
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
        
        logger.info(f"    test: {len(clips)} 个有效片段（总计{len(test_data)}，跳过{skipped_count}个无效）")
    
    # 加载 train 数据
    if train_json_path and train_json_path.exists():
        logger.info(f"  加载 train 数据: {train_json_path}")
        
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
        train_start_idx = len(clips)
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
        
        logger.info(f"    train: {train_valid_count} 个有效片段（总计{len(train_data)}，跳过{train_skipped_count}个无效）")
    
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
    video_files: List[Path],
    text_inputs_dir: Path,
    use_loose_match: bool,
    video_prefix: str,
    text_prefix: str,
    skip_text_generation: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    处理单个视频片段（用于多进程）
    
    Args:
        clip: 视频片段
        vid_to_title: vid 到 title 的映射
        video_files: 预先收集的视频文件列表
        text_inputs_dir: text_input.pkl 输出目录
        use_loose_match: 是否使用宽松匹配
        video_prefix: 视频路径前缀
        text_prefix: text_input.pkl 路径前缀
        skip_text_generation: 是否跳过 text token 生成
    
    Returns:
        (metadata_item, failure_item) 元组
        - 如果成功: (metadata_item, None)
        - 如果失败: (None, failure_item)
    """
    # 获取 title
    title = vid_to_title.get(clip.vid)
    if not title:
        return None, {
            'vid': clip.vid,
            'data_type': clip.data_type,
            'transcript': clip.transcript,
            'start_time': clip.start_time,
            'end_time': clip.end_time,
            'duration': clip.end_time - clip.start_time,
            'title': None,
            'reason': 'no_download_record',
            'message': '在下载日志中未找到此vid的记录',
            'suggestion': '检查vid是否正确，或该视频是否在下载日志CSV中'
        }
    
    # 查找视频文件
    video_file = find_video_file(clip.vid, title, video_files, use_loose_match=use_loose_match)
    if not video_file:
        return None, {
            'vid': clip.vid,
            'data_type': clip.data_type,
            'transcript': clip.transcript,
            'start_time': clip.start_time,
            'end_time': clip.end_time,
            'duration': clip.end_time - clip.start_time,
            'title': title,
            'reason': 'video_not_found',
            'message': f'下载日志中有记录，但在视频目录中找不到对应文件',
            'suggestion': f'检查文件名是否匹配: 期望"{title}.mp4"或相似名称'
        }
    
    # 生成 text_input.pkl（如果需要）
    text_input_path = text_inputs_dir / f"{clip.vid}_text_input.pkl"
    
    if not skip_text_generation:
        if not text_input_path.exists():
            success = generate_text_input_pkl(clip.transcript, text_input_path)
            if not success:
                return None, {
                    'vid': clip.vid,
                    'data_type': clip.data_type,
                    'transcript': clip.transcript,
                    'start_time': clip.start_time,
                    'end_time': clip.end_time,
                    'duration': clip.end_time - clip.start_time,
                    'title': title,
                    'video_file': str(video_file),
                    'reason': 'tokenization_failed',
                    'message': '视频文件找到，但生成text_input.pkl失败',
                    'suggestion': '检查transcript内容是否有效，或transformers库是否正常'
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
        'text_input_path': text_path if not skip_text_generation else None
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
    
    # Text token 生成参数
    parser.add_argument("--skip-text-generation", action='store_true',
                       help="跳过 text_input.pkl 生成（仅测试视频匹配性能）")
    
    # 路径前缀参数
    parser.add_argument("--video-prefix", type=str, default="",
                       help="视频路径前缀（例如：/mnt/data/）")
    parser.add_argument("--text-prefix", type=str, default="",
                       help="text_input.pkl 路径前缀（例如：/mnt/data/）")
    
    # 失败分析报告输出
    parser.add_argument("--unmatched-output", type=Path,
                       help="失败分析报告输出路径（JSON格式，包含详细失败原因和统计）")
    
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
    
    # 1. 加载数据集（作为片段索引）
    logger.info("步骤 1: 加载片段索引...")
    clips = load_dataset_clips(args.test_json, args.train_json)
    logger.info(f"  需要处理 {len(clips)} 个片段")
    
    # 2. 加载下载日志（vid→title映射）
    logger.info("步骤 2: 加载下载日志...")
    vid_to_title = load_download_log(args.download_log)
    
    # 3. 扫描视频目录（实际存在的视频文件）
    logger.info("步骤 3: 扫描视频目录...")
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.mov', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(args.videos_dir.glob(f'*{ext}'))
    logger.info(f"  找到 {len(video_files)} 个视频文件（一个视频可对应多个片段）")
    
    # 4. 匹配视频文件并生成元数据
    if args.skip_text_generation:
        logger.info("步骤 4: 匹配视频文件（跳过 text_input.pkl 生成）...")
        logger.info("  ⚠️  性能测试模式：text token 生成已禁用")
    else:
        logger.info("步骤 4: 匹配视频文件并生成 text_input.pkl...")
    logger.info(f"  使用 {args.num_workers} 个进程并行处理")
    
    metadata = []
    matched_count = 0
    failed_count = 0
    failure_items = []  # 收集失败项（含详细信息）
    
    if args.num_workers == 1:
        # 单进程模式（也使用预计算索引优化性能）
        global _worker_video_index
        logger.info(f"  构建视频文件索引（{len(video_files)} 个文件）...")
        _worker_video_index = _build_video_index(video_files, args.loose_match)
        logger.info("  ✅ 索引构建完成")
        
        for clip in tqdm(clips, desc="处理进度", unit="clip"):
            metadata_item, failure_item = process_single_clip(
                clip, vid_to_title, video_files, args.text_inputs_dir,
                args.loose_match, args.video_prefix, args.text_prefix,
                args.skip_text_generation
            )
            
            if metadata_item:
                metadata.append(metadata_item)
                matched_count += 1
            else:
                failure_items.append(failure_item)
                failed_count += 1
    else:
        # 多进程模式（使用 Pool initializer 避免重复传递数据）
        logger.info(f"  每个 worker 进程将独立构建视频索引（{len(video_files)} 个文件）")
        with Pool(
            processes=args.num_workers,
            initializer=_init_metadata_worker,
            initargs=(
                vid_to_title,
                video_files,
                args.text_inputs_dir,
                args.loose_match,
                args.video_prefix,
                args.text_prefix,
                args.skip_text_generation
            )
        ) as pool:
            # 使用 tqdm 显示进度
            for metadata_item, failure_item in tqdm(
                pool.imap(_process_clip_worker, clips),
                total=len(clips),
                desc="处理进度",
                unit="clip"
            ):
                if metadata_item:
                    metadata.append(metadata_item)
                    matched_count += 1
                else:
                    failure_items.append(failure_item)
                    failed_count += 1
    
    # 5. 保存元数据
    logger.info("步骤 5: 保存元数据...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 6. 生成并保存失败分析报告（如果指定）
    if args.unmatched_output and failure_items:
        logger.info("步骤 6: 生成失败分析报告...")
        
        # 统计各种失败原因
        failure_stats = {}
        for item in failure_items:
            reason = item['reason']
            failure_stats[reason] = failure_stats.get(reason, 0) + 1
        
        # 按data_type分类
        failure_by_type = {'test': [], 'train': []}
        for item in failure_items:
            data_type = item.get('data_type', 'unknown')
            if data_type in failure_by_type:
                failure_by_type[data_type].append(item)
        
        # 构建完整报告
        failure_report = {
            'summary': {
                'total_clips': len(clips),
                'matched': matched_count,
                'failed': failed_count,
                'success_rate': f"{matched_count/len(clips)*100:.2f}%",
                'failure_breakdown': failure_stats,
                'failure_by_data_type': {
                    'test': len(failure_by_type['test']),
                    'train': len(failure_by_type['train'])
                }
            },
            'failure_analysis': {
                'no_download_record': {
                    'description': 'vid在下载日志CSV中不存在',
                    'count': failure_stats.get('no_download_record', 0),
                    'solution': '1. 检查vid是否正确\n2. 确认该视频是否已下载\n3. 检查CSV文件是否完整'
                },
                'video_not_found': {
                    'description': 'vid在下载日志中存在，但找不到对应视频文件',
                    'count': failure_stats.get('video_not_found', 0),
                    'solution': '1. 检查视频文件是否存在\n2. 检查文件名是否匹配\n3. 尝试使用--loose-match参数'
                },
                'tokenization_failed': {
                    'description': '视频文件找到，但生成text_input.pkl失败',
                    'count': failure_stats.get('tokenization_failed', 0),
                    'solution': '1. 检查transcript内容\n2. 检查transformers库是否正常\n3. 检查磁盘空间'
                }
            },
            'failed_clips': failure_items
        }
        
        with open(args.unmatched_output, 'w', encoding='utf-8') as f:
            json.dump(failure_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  失败分析报告: {args.unmatched_output}")
        logger.info(f"  失败原因统计:")
        for reason, count in failure_stats.items():
            logger.info(f"    - {reason}: {count} 个片段")
    
    # 完成
    logger.info("=" * 60)
    logger.info("✅ 元数据生成完成！")
    logger.info("=" * 60)
    logger.info(f"数据流概览:")
    logger.info(f"  1️⃣  片段索引 (dataset): {len(clips)} 个片段")
    logger.info(f"  2️⃣  下载记录 (csv): {len(vid_to_title)} 条映射")
    logger.info(f"  3️⃣  视频文件 (实际): {len(video_files)} 个文件")
    logger.info(f"  4️⃣  匹配结果:")
    logger.info(f"      ✅ 成功: {matched_count} 个片段")
    logger.info(f"      ❌ 失败: {failed_count} 个片段")
    logger.info(f"")
    logger.info(f"输出文件:")
    logger.info(f"  📄 metadata.json: {args.output}")
    if args.unmatched_output and failure_items:
        logger.info(f"  📄 失败分析报告: {args.unmatched_output}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

