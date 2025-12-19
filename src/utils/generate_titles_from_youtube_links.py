#!/usr/bin/env python3
"""
从 YouTube 链接文件生成 CSV（包含 title 和 status）

使用方法:
    python src/utils/update_youtube_titles_api.py \
        --input /Users/nanzhang/Downloads/youtube_links.txt \
        --output data/training/youtube_videos.csv \
        --api-key YOUR_API_KEY

环境变量:
    YOUTUBE_API_KEY: 如果不通过 --api-key 传入，可以设置环境变量
"""

import csv
import re
import os
import time
import logging
from pathlib import Path
from typing import List, Dict
import argparse

import requests
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeAPIClient:
    """YouTube Data API v3 客户端"""
    
    def __init__(self, api_key: str):
        """
        初始化 YouTube API 客户端
        
        Args:
            api_key: YouTube Data API v3 密钥
        """
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3/videos"
        self.quota_used = 0
        
        # API 配额信息
        # videos.list: 每次调用消耗 1 quota
        # 每天默认配额: 10,000
        logger.info("YouTube API v3 客户端初始化成功")
    
    def extract_video_id(self, url: str) -> str:
        """
        从 YouTube URL 提取 video ID
        
        Args:
            url: YouTube URL
        
        Returns:
            video_id: 视频 ID，失败返回空字符串
        """
        url = url.strip()
        
        # 支持多种 URL 格式
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return ""
    
    def get_videos_info(self, video_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """
        批量获取视频信息
        
        YouTube API v3 支持一次查询最多 50 个视频
        
        Args:
            video_ids: 视频 ID 列表 (最多 50 个)
        
        Returns:
            dict: {video_id: {'title': str, 'status': str}}
                status: 'True', 'private', 'unavailable'
        """
        if not video_ids:
            return {}
        
        if len(video_ids) > 50:
            logger.warning(f"视频数量超过 50，将只处理前 50 个")
            video_ids = video_ids[:50]
        
        # API 请求参数
        params = {
            'part': 'snippet,status',
            'id': ','.join(video_ids),
            'key': self.api_key,
            'maxResults': 50
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            self.quota_used += 1
            
            if response.status_code == 200:
                data = response.json()
                results = {}
                
                # 解析返回的视频信息
                for item in data.get('items', []):
                    video_id = item['id']
                    title = item['snippet']['title']
                    
                    # 判断视频状态
                    privacy_status = item['status']['privacyStatus']
                    
                    # 映射状态 (匹配 test2000_formatted.csv 的格式)
                    if privacy_status == 'private':
                        status = 'private'
                    elif privacy_status in ['public', 'unlisted']:
                        status = 'True'
                    else:
                        status = privacy_status
                    
                    results[video_id] = {
                        'title': title,
                        'status': status
                    }
                
                # 对于没有返回的视频，标记为 unavailable
                for vid in video_ids:
                    if vid not in results:
                        results[vid] = {
                            'title': '',
                            'status': 'unavailable'
                        }
                
                return results
            
            elif response.status_code == 403:
                logger.error("API 配额已用尽或 API_KEY 无效")
                raise Exception("API 配额已用尽或 API_KEY 无效")
            
            elif response.status_code == 400:
                logger.error(f"API 请求错误: {response.text}")
                # 返回空结果，标记为 unavailable
                return {vid: {'title': '', 'status': 'unavailable'} for vid in video_ids}
            
            else:
                logger.error(f"API 请求失败: {response.status_code}")
                return {vid: {'title': '', 'status': 'unavailable'} for vid in video_ids}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {str(e)}")
            return {vid: {'title': '', 'status': 'unavailable'} for vid in video_ids}
    
    def get_quota_info(self) -> Dict[str, int]:
        """获取配额使用信息"""
        return {
            'used': self.quota_used,
            'remaining_estimate': 10000 - self.quota_used
        }


def generate_csv_from_links(
    input_txt: Path,
    output_csv: Path,
    api_key: str,
    batch_size: int = 50,
    delay: float = 0.1
) -> Dict[str, int]:
    """
    从 YouTube 链接文件生成 CSV
    
    Args:
        input_txt: 输入文本文件（每行一个 YouTube URL）
        output_csv: 输出 CSV 文件
        api_key: YouTube API 密钥
        batch_size: 批量查询大小 (最大 50)
        delay: 每批之间的延迟（秒）
    
    Returns:
        统计信息字典
    """
    # 初始化 API 客户端
    client = YouTubeAPIClient(api_key)
    
    # 统计信息
    stats = {
        'total': 0,
        'available': 0,
        'unavailable': 0,
        'private': 0,
        'invalid_url': 0
    }
    
    # 读取 URL 列表
    logger.info(f"读取 URL 列表: {input_txt}")
    urls = []
    
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    
    stats['total'] = len(urls)
    logger.info(f"共 {stats['total']} 个 URL")
    
    # 批量处理
    batch_size = min(batch_size, 50)  # API 限制最多 50
    csv_rows = []
    
    logger.info(f"开始批量查询 (batch_size={batch_size})")
    
    with tqdm(total=len(urls), desc="🔄 获取视频信息") as pbar:
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            
            # 提取 video IDs
            url_to_vid = {}
            video_ids = []
            
            for url in batch_urls:
                vid = client.extract_video_id(url)
                if vid:
                    url_to_vid[url] = vid
                    video_ids.append(vid)
                else:
                    # 无法提取 video ID
                    csv_rows.append({
                        'url': url,
                        'title': '',
                        'status': 'unavailable'
                    })
                    stats['invalid_url'] += 1
            
            # 调用 API
            if video_ids:
                results = client.get_videos_info(video_ids)
                
                # 构建 CSV 行
                for url in batch_urls:
                    vid = url_to_vid.get(url)
                    if vid and vid in results:
                        info = results[vid]
                        csv_rows.append({
                            'url': url,
                            'title': info['title'],
                            'status': info['status']
                        })
                        
                        # 统计
                        if info['status'] == 'unavailable':
                            stats['unavailable'] += 1
                        elif info['status'] == 'True':
                            stats['available'] += 1
                        elif info['status'] == 'private':
                            stats['private'] += 1
            
            pbar.update(len(batch_urls))
            
            # 延迟（避免过快请求）
            if i + batch_size < len(urls):
                time.sleep(delay)
    
    # 保存结果
    logger.info(f"保存结果到: {output_csv}")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['url', 'title', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # 配额信息
    quota_info = client.get_quota_info()
    logger.info(f"API 配额使用: {quota_info['used']} 次调用")
    logger.info(f"预计剩余配额: {quota_info['remaining_estimate']}")
    
    return stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从 YouTube 链接文件生成 CSV（包含 title 和 status）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用 API_KEY 参数
    python src/utils/update_youtube_titles_api.py \\
        --input /Users/nanzhang/Downloads/youtube_links.txt \\
        --output data/training/youtube_videos.csv \\
        --api-key YOUR_API_KEY
    
    # 使用环境变量
    export YOUTUBE_API_KEY="YOUR_API_KEY"
    python src/utils/update_youtube_titles_api.py \\
        --input /Users/nanzhang/Downloads/youtube_links.txt \\
        --output data/training/youtube_videos.csv
    
    # 自定义批量大小和延迟
    python src/utils/update_youtube_titles_api.py \\
        --input /Users/nanzhang/Downloads/youtube_links.txt \\
        --output data/training/youtube_videos.csv \\
        --api-key YOUR_API_KEY \\
        --batch-size 50 \\
        --delay 0.5

配额说明:
    - 每次 API 调用消耗 1 quota
    - 每批最多查询 50 个视频
    - 默认每天配额: 10,000
    - 2196 个视频约需 44 次调用 (44 quota)
        """
    )
    
    parser.add_argument("--input", type=Path, required=True,
                       help="输入文本文件路径（每行一个 YouTube URL）")
    parser.add_argument("--output", type=Path, required=True,
                       help="输出 CSV 文件路径")
    parser.add_argument("--api-key", type=str,
                       help="YouTube Data API v3 密钥 (或设置环境变量 YOUTUBE_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="批量查询大小 (默认: 50，最大: 50)")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="每批之间的延迟（秒，默认: 0.1）")
    
    args = parser.parse_args()
    
    # 获取 API_KEY
    api_key = args.api_key or os.getenv('YOUTUBE_API_KEY')
    
    if not api_key:
        logger.error("未提供 API_KEY，请使用 --api-key 参数或设置环境变量 YOUTUBE_API_KEY")
        return 1
    
    # 检查输入文件
    if not args.input.exists():
        logger.error(f"输入文件不存在: {args.input}")
        return 1
    
    # 创建输出目录
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行生成
    logger.info("=" * 60)
    logger.info("开始使用 YouTube Data API v3 获取视频信息")
    logger.info("=" * 60)
    
    try:
        stats = generate_csv_from_links(
            input_txt=args.input,
            output_csv=args.output,
            api_key=api_key,
            batch_size=args.batch_size,
            delay=args.delay
        )
        
        # 打印统计
        logger.info("\n" + "=" * 60)
        logger.info("CSV 生成完成")
        logger.info("=" * 60)
        logger.info(f"总数:         {stats['total']}")
        logger.info(f"可用:         {stats['available']}")
        logger.info(f"不可用:       {stats['unavailable']}")
        logger.info(f"私密:         {stats['private']}")
        logger.info(f"无效URL:      {stats['invalid_url']}")
        logger.info("=" * 60)
        
        # 验证行数
        with open(args.output, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f) - 1  # 减去表头
        
        logger.info(f"\n✅ CSV 文件已生成: {args.output}")
        logger.info(f"✅ 总行数: {line_count} 行（不含表头）")
        
        if line_count >= 2196:
            logger.info(f"✅ 满足要求：至少 2196 行")
        else:
            logger.warning(f"⚠️  警告：只有 {line_count} 行，少于预期的 2196 行")
        
        return 0
    
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
