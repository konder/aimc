#!/usr/bin/env python3
"""
视频切片处理脚本

根据已下载的视频和 CLIP4MC 元数据，生成文本-视频片段对。

使用方法:
    python video_clip_processor.py \
        --videos-dir ./videos \
        --info-csv ./info.csv \
        --metadata ./dataset_test.json \
        --output-dir ./clips
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    """从 YouTube URL 中提取视频 ID"""
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
    """
    解析 info.csv，返回 video_id -> filename 的映射
    
    CSV 格式: URL,filename (可能有双引号)
    """
    vid_to_filename = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # 尝试自动检测分隔符
        sample = f.read(2048)
        f.seek(0)
        
        # 检测是否有表头
        has_header = 'url' in sample.lower() or 'http' not in sample[:100].lower()
        
        try:
            # 尝试用 csv 模块解析
            reader = csv.reader(f)
            if has_header:
                next(reader)  # 跳过表头
            
            for row in reader:
                if len(row) >= 2:
                    url = row[0].strip().strip('"')
                    filename = row[1].strip().strip('"')
                    
                    vid = extract_video_id(url)
                    if vid:
                        vid_to_filename[vid] = filename
        except Exception as e:
            logger.warning(f"CSV 解析失败，尝试手动解析: {e}")
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 手动解析
                parts = line.split(',', 1)
                if len(parts) >= 2:
                    url = parts[0].strip().strip('"')
                    filename = parts[1].strip().strip('"')
                    
                    vid = extract_video_id(url)
                    if vid:
                        vid_to_filename[vid] = filename
    
    logger.info(f"从 info.csv 解析了 {len(vid_to_filename)} 条记录")
    return vid_to_filename


def find_video_file(videos_dir: Path, filename: str) -> Optional[Path]:
    """查找视频文件（支持模糊匹配）"""
    # 直接匹配
    direct_path = videos_dir / filename
    if direct_path.exists():
        return direct_path
    
    # 加上 .mp4 扩展名
    if not filename.endswith('.mp4'):
        mp4_path = videos_dir / f"{filename}.mp4"
        if mp4_path.exists():
            return mp4_path
    
    # 模糊匹配（文件名可能有细微差异）
    filename_lower = filename.lower().replace('.mp4', '')
    for f in videos_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp4', '.webm', '.mkv']:
            if filename_lower in f.stem.lower():
                return f
    
    return None


def extract_clip(
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        logger.warning(f"切片失败: {e}")
        return False


def process_clips(
    videos_dir: Path,
    info_csv: Path,
    metadata_json: Path,
    output_dir: Path,
    debug: bool = False,
) -> List[Dict]:
    """处理视频切片"""
    
    # 1. 解析 info.csv
    vid_to_filename = parse_info_csv(info_csv)
    
    # 2. 加载元数据
    with open(metadata_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logger.info(f"加载了 {len(metadata)} 条元数据记录")
    
    # 获取元数据中的唯一 video ID
    metadata_vids = set(item.get('vid', '') for item in metadata if item.get('vid'))
    logger.info(f"元数据中唯一视频 ID: {len(metadata_vids)} 个")
    
    # 3. 统计可用视频
    available_videos = {}
    missing_files = []
    for vid, filename in vid_to_filename.items():
        video_path = find_video_file(videos_dir, filename)
        if video_path:
            available_videos[vid] = video_path
        else:
            missing_files.append((vid, filename))
    
    logger.info(f"info.csv 中的视频: {len(vid_to_filename)} 个")
    logger.info(f"  - 找到文件: {len(available_videos)} 个")
    logger.info(f"  - 文件缺失: {len(missing_files)} 个")
    
    # 4. 分析匹配情况
    csv_vids = set(vid_to_filename.keys())
    matched_vids = csv_vids & metadata_vids
    
    logger.info(f"\n📊 匹配分析:")
    logger.info(f"  - info.csv 中的 video ID: {len(csv_vids)} 个")
    logger.info(f"  - dataset 中的 video ID: {len(metadata_vids)} 个")
    logger.info(f"  - 两者交集 (可处理): {len(matched_vids)} 个")
    
    if debug and len(matched_vids) < 20:
        logger.info(f"  - 匹配的 ID: {list(matched_vids)}")
    
    # 显示不匹配的原因
    unmatched_csv = csv_vids - metadata_vids
    if unmatched_csv and debug:
        logger.info(f"\n⚠️ info.csv 中有 {len(unmatched_csv)} 个视频不在 dataset 中")
        logger.info(f"   前 5 个: {list(unmatched_csv)[:5]}")
    
    # 5. 筛选可处理的元数据 (视频文件存在且在元数据中)
    processable = []
    for item in metadata:
        vid = item.get('vid', '')
        if vid in available_videos:
            processable.append({
                'vid': vid,
                'video_path': available_videos[vid],
                'transcript': item.get('transcript clip', ''),
                'begin': item.get('begin position', 0),
                'end': item.get('end position', 0),
            })
    
    logger.info(f"\n✅ 可处理的片段: {len(processable)} 条")
    
    if not processable:
        logger.warning("没有可处理的片段")
        return []
    
    # 5. 处理切片
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    results = []
    success_count = 0
    fail_count = 0
    
    for i, item in enumerate(processable):
        vid = item['vid']
        begin = item['begin']
        end = item['end']
        
        # 生成输出文件名
        clip_name = f"{vid}_{int(begin)}_{int(end)}.mp4"
        clip_path = clips_dir / clip_name
        
        # 进度显示
        print(f"\r处理中: {i+1}/{len(processable)} - {vid}", end='', flush=True)
        
        # 切片
        if extract_clip(item['video_path'], clip_path, begin, end):
            results.append({
                'vid': vid,
                'clip_path': str(clip_path),
                'transcript': item['transcript'],
                'begin_time': begin,
                'end_time': end,
                'duration': end - begin,
            })
            success_count += 1
        else:
            fail_count += 1
    
    print()  # 换行
    
    # 6. 保存结果
    output_json = output_dir / "text_video_pairs.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 7. 生成简单的文本文件（每行: clip_path<TAB>transcript）
    output_tsv = output_dir / "text_video_pairs.tsv"
    with open(output_tsv, 'w', encoding='utf-8') as f:
        for r in results:
            # 清理 transcript 中的换行
            text = r['transcript'].replace('\n', ' ').replace('\t', ' ')
            f.write(f"{r['clip_path']}\t{text}\n")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"处理完成!")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  JSON: {output_json}")
    logger.info(f"  TSV: {output_tsv}")
    logger.info(f"{'='*50}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="视频切片处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python video_clip_processor.py \\
        --videos-dir ./downloaded_videos \\
        --info-csv ./info.csv \\
        --metadata ./dataset_test.json \\
        --output-dir ./output
        """
    )
    
    parser.add_argument("--videos-dir", "-v", type=Path, required=True,
                       help="已下载视频的目录")
    parser.add_argument("--info-csv", "-i", type=Path, required=True,
                       help="info.csv 文件路径 (URL,filename)")
    parser.add_argument("--metadata", "-m", type=Path, required=True,
                       help="CLIP4MC 元数据 JSON 文件")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                       help="输出目录")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="显示详细调试信息")
    
    args = parser.parse_args()
    
    # 验证输入
    if not args.videos_dir.exists():
        logger.error(f"视频目录不存在: {args.videos_dir}")
        sys.exit(1)
    if not args.info_csv.exists():
        logger.error(f"info.csv 不存在: {args.info_csv}")
        sys.exit(1)
    if not args.metadata.exists():
        logger.error(f"元数据文件不存在: {args.metadata}")
        sys.exit(1)
    
    # 处理
    process_clips(
        videos_dir=args.videos_dir,
        info_csv=args.info_csv,
        metadata_json=args.metadata,
        output_dir=args.output_dir,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

