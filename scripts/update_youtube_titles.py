#!/usr/bin/env python3
"""
更新 youtube_download_log.csv 中缺失的视频标题

支持多种获取方式（按优先级）：
1. pytube
2. Invidious API
3. 网页抓取

使用示例：
    # 基本用法
    python scripts/update_youtube_titles.py \\
        --input data/training/youtube_download_log.csv \\
        --output data/training/youtube_download_log_updated.csv
    
    # 指定工具
    python scripts/update_youtube_titles.py \\
        --input input.csv \\
        --output output.csv \\
        --method pytube
    
    # 只更新状态为 True 但标题为空的记录
    python scripts/update_youtube_titles.py \\
        --input input.csv \\
        --output output.csv \\
        --only-missing
"""

import csv
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    """从 YouTube URL 中提取视频 ID"""
    try:
        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc:
            return parse_qs(parsed.query).get('v', [None])[0]
        elif 'youtu.be' in parsed.netloc:
            return parsed.path.lstrip('/')
    except Exception:
        pass
    return None


def get_title_pytube(url: str) -> Optional[str]:
    """使用 pytube 获取标题"""
    try:
        from pytube import YouTube
        yt = YouTube(url)
        return yt.title
    except ImportError:
        logger.warning("pytube 未安装，跳过此方法")
        return None
    except Exception as e:
        logger.debug(f"pytube 失败: {str(e)}")
        return None


def get_title_invidious(url: str) -> Optional[str]:
    """使用 Invidious API 获取标题"""
    try:
        import requests
        
        video_id = extract_video_id(url)
        if not video_id:
            return None
        
        # 尝试多个 Invidious 实例
        instances = [
            "https://invidious.snopyta.org",
            "https://yewtu.be",
            "https://invidious.kavin.rocks",
        ]
        
        for instance in instances:
            try:
                api_url = f"{instance}/api/v1/videos/{video_id}"
                response = requests.get(api_url, timeout=5)
                response.raise_for_status()
                data = response.json()
                return data.get('title')
            except Exception:
                continue
        
        return None
    
    except ImportError:
        logger.warning("requests 未安装，跳过 Invidious API")
        return None
    except Exception as e:
        logger.debug(f"Invidious API 失败: {str(e)}")
        return None


def get_title_scraping(url: str) -> Optional[str]:
    """使用网页抓取获取标题"""
    try:
        import requests
        import re
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 从 <title> 标签提取
        match = re.search(r'<title>(.+?)</title>', response.text)
        if match:
            title = match.group(1).replace(' - YouTube', '').strip()
            return title
        
        return None
    
    except ImportError:
        logger.warning("requests 未安装，跳过网页抓取")
        return None
    except Exception as e:
        logger.debug(f"网页抓取失败: {str(e)}")
        return None


def get_title_with_fallback(url: str, method: str = 'auto') -> tuple[Optional[str], str]:
    """
    使用多种方法获取标题（带回退机制）
    
    Args:
        url: YouTube URL
        method: 指定方法 ('auto', 'pytube', 'invidious', 'scraping')
    
    Returns:
        (title, method_used)
    """
    if method == 'pytube':
        title = get_title_pytube(url)
        if title:
            return title, 'pytube'
    
    elif method == 'invidious':
        title = get_title_invidious(url)
        if title:
            return title, 'invidious'
    
    elif method == 'scraping':
        title = get_title_scraping(url)
        if title:
            return title, 'scraping'
    
    elif method == 'auto':
        # 自动回退：pytube -> invidious -> scraping
        title = get_title_pytube(url)
        if title:
            return title, 'pytube'
        
        title = get_title_invidious(url)
        if title:
            return title, 'invidious'
        
        title = get_title_scraping(url)
        if title:
            return title, 'scraping'
    
    return None, 'failed'


def update_titles(
    input_csv: Path,
    output_csv: Path,
    method: str = 'auto',
    only_missing: bool = False,
    delay: float = 0.5
):
    """
    更新 CSV 中的标题
    
    Args:
        input_csv: 输入 CSV 文件
        output_csv: 输出 CSV 文件
        method: 获取方法 ('auto', 'pytube', 'invidious', 'scraping')
        only_missing: 只更新缺失的标题
        delay: 请求间隔（秒）
    """
    # 读取原始数据
    rows = []
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.info(f"加载了 {len(rows)} 条记录")
    
    # 筛选需要更新的记录
    if only_missing:
        to_update = [
            row for row in rows 
            if row.get('status') == 'True' and not row.get('title', '').strip()
        ]
        logger.info(f"需要更新标题: {len(to_update)} 条")
    else:
        to_update = [
            row for row in rows
            if row.get('status') == 'True'
        ]
        logger.info(f"需要检查标题: {len(to_update)} 条")
    
    if not to_update:
        logger.info("无需更新")
        return
    
    # 统计
    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'pytube': 0,
        'invidious': 0,
        'scraping': 0,
    }
    
    # 更新标题
    for row in tqdm(to_update, desc="更新标题"):
        url = row.get('url', '')
        current_title = row.get('title', '').strip()
        
        # 如果已有标题且只更新缺失的，跳过
        if only_missing and current_title:
            stats['skipped'] += 1
            continue
        
        # 获取新标题
        new_title, method_used = get_title_with_fallback(url, method)
        
        if new_title:
            row['title'] = new_title
            stats['success'] += 1
            stats[method_used] = stats.get(method_used, 0) + 1
            logger.debug(f"✅ {url} -> {new_title} (via {method_used})")
        else:
            stats['failed'] += 1
            logger.debug(f"❌ {url} -> 获取失败")
        
        # 延迟（避免请求过快）
        time.sleep(delay)
    
    # 保存结果
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['url', 'title', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # 打印统计
    logger.info("=" * 60)
    logger.info("更新完成！")
    logger.info(f"  总记录数: {len(rows)}")
    logger.info(f"  需要更新: {len(to_update)}")
    logger.info(f"  成功更新: {stats['success']}")
    logger.info(f"  更新失败: {stats['failed']}")
    logger.info(f"  跳过: {stats['skipped']}")
    logger.info("")
    logger.info("使用的方法:")
    logger.info(f"  pytube: {stats.get('pytube', 0)}")
    logger.info(f"  invidious: {stats.get('invidious', 0)}")
    logger.info(f"  scraping: {stats.get('scraping', 0)}")
    logger.info(f"  输出文件: {output_csv}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="更新 youtube_download_log.csv 中缺失的视频标题",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", type=Path, required=True,
                       help="输入 CSV 文件路径")
    parser.add_argument("--output", type=Path, required=True,
                       help="输出 CSV 文件路径")
    parser.add_argument("--method", type=str, 
                       choices=['auto', 'pytube', 'invidious', 'scraping'],
                       default='auto',
                       help="获取标题的方法（默认: auto，自动回退）")
    parser.add_argument("--only-missing", action='store_true',
                       help="只更新缺失的标题（status=True 但 title 为空）")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="请求间隔（秒，默认: 0.5）")
    parser.add_argument("--debug", action='store_true',
                       help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 验证输入文件
    if not args.input.exists():
        logger.error(f"输入文件不存在: {args.input}")
        return 1
    
    # 创建输出目录
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行更新
    try:
        update_titles(
            args.input,
            args.output,
            method=args.method,
            only_missing=args.only_missing,
            delay=args.delay
        )
        return 0
    except Exception as e:
        logger.error(f"更新失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

