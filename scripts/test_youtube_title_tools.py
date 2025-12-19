#!/usr/bin/env python3
"""
测试不同的 YouTube 标题获取工具

支持的工具:
1. pytube
2. youtube-dl
3. YouTube Data API v3
4. Invidious API
5. 网页抓取
"""

import sys
import time
from pathlib import Path

# 测试 URL
TEST_URLS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo (第一个 YouTube 视频)
]


def test_pytube():
    """测试 pytube"""
    print("\n" + "=" * 60)
    print("测试工具: pytube")
    print("=" * 60)
    
    try:
        from pytube import YouTube
        
        for url in TEST_URLS:
            try:
                yt = YouTube(url)
                print(f"✅ {url}")
                print(f"   标题: {yt.title}")
                print(f"   作者: {yt.author}")
                print(f"   时长: {yt.length}秒")
            except Exception as e:
                print(f"❌ {url}")
                print(f"   错误: {str(e)}")
            print()
        
        return True
    
    except ImportError:
        print("❌ pytube 未安装")
        print("   安装命令: pip install pytube")
        return False


def test_youtube_dl():
    """测试 youtube-dl"""
    print("\n" + "=" * 60)
    print("测试工具: youtube-dl")
    print("=" * 60)
    
    try:
        import youtube_dl
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        for url in TEST_URLS:
            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    print(f"✅ {url}")
                    print(f"   标题: {info['title']}")
                    print(f"   作者: {info['uploader']}")
                    print(f"   时长: {info['duration']}秒")
            except Exception as e:
                print(f"❌ {url}")
                print(f"   错误: {str(e)}")
            print()
        
        return True
    
    except ImportError:
        print("❌ youtube-dl 未安装")
        print("   安装命令: pip install youtube-dl")
        return False


def test_youtube_api(api_key=None):
    """测试 YouTube Data API v3"""
    print("\n" + "=" * 60)
    print("测试工具: YouTube Data API v3")
    print("=" * 60)
    
    if not api_key:
        print("⚠️  需要 API Key")
        print("   1. 访问 https://console.cloud.google.com/")
        print("   2. 创建项目并启用 YouTube Data API v3")
        print("   3. 创建 API 密钥")
        print("   4. 运行: python test_youtube_title_tools.py --api-key YOUR_KEY")
        return False
    
    try:
        from googleapiclient.discovery import build
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # 提取视频 ID
        from urllib.parse import urlparse, parse_qs
        video_ids = []
        for url in TEST_URLS:
            parsed = urlparse(url)
            if 'youtube.com' in parsed.netloc:
                vid = parse_qs(parsed.query).get('v', [None])[0]
                if vid:
                    video_ids.append(vid)
        
        # 批量查询
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids)
        )
        response = request.execute()
        
        for item in response['items']:
            print(f"✅ Video ID: {item['id']}")
            print(f"   标题: {item['snippet']['title']}")
            print(f"   作者: {item['snippet']['channelTitle']}")
            print(f"   观看数: {item['statistics'].get('viewCount', 'N/A')}")
            print()
        
        print(f"📊 API 配额消耗: 1 单位")
        return True
    
    except ImportError:
        print("❌ google-api-python-client 未安装")
        print("   安装命令: pip install google-api-python-client")
        return False
    except Exception as e:
        print(f"❌ API 调用失败: {str(e)}")
        return False


def test_invidious():
    """测试 Invidious API"""
    print("\n" + "=" * 60)
    print("测试工具: Invidious API")
    print("=" * 60)
    
    try:
        import requests
        from urllib.parse import urlparse, parse_qs
        
        # 尝试多个 Invidious 实例
        instances = [
            "https://invidious.snopyta.org",
            "https://yewtu.be",
            "https://invidious.kavin.rocks",
        ]
        
        for url in TEST_URLS:
            # 提取视频 ID
            parsed = urlparse(url)
            video_id = None
            if 'youtube.com' in parsed.netloc:
                video_id = parse_qs(parsed.query).get('v', [None])[0]
            elif 'youtu.be' in parsed.netloc:
                video_id = parsed.path.lstrip('/')
            
            if not video_id:
                print(f"❌ 无法提取视频 ID: {url}")
                continue
            
            # 尝试不同实例
            success = False
            for instance in instances:
                try:
                    api_url = f"{instance}/api/v1/videos/{video_id}"
                    response = requests.get(api_url, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    print(f"✅ {url}")
                    print(f"   标题: {data['title']}")
                    print(f"   作者: {data['author']}")
                    print(f"   时长: {data['lengthSeconds']}秒")
                    print(f"   实例: {instance}")
                    success = True
                    break
                except Exception as e:
                    continue
            
            if not success:
                print(f"❌ {url}")
                print(f"   错误: 所有 Invidious 实例都失败")
            print()
        
        return True
    
    except ImportError:
        print("❌ requests 未安装")
        print("   安装命令: pip install requests")
        return False


def test_web_scraping():
    """测试网页抓取"""
    print("\n" + "=" * 60)
    print("测试工具: 网页抓取")
    print("=" * 60)
    
    try:
        import requests
        import re
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for url in TEST_URLS:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # 方法 1: 从 <title> 提取
                title_match = re.search(r'<title>(.+?)</title>', response.text)
                if title_match:
                    title = title_match.group(1).replace(' - YouTube', '').strip()
                    print(f"✅ {url}")
                    print(f"   标题: {title}")
                else:
                    print(f"⚠️  {url}")
                    print(f"   警告: 无法从 HTML 提取标题")
            except Exception as e:
                print(f"❌ {url}")
                print(f"   错误: {str(e)}")
            print()
        
        return True
    
    except ImportError:
        print("❌ requests 未安装")
        print("   安装命令: pip install requests")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="测试不同的 YouTube 标题获取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--api-key", type=str,
                       help="YouTube Data API v3 密钥（用于测试官方 API）")
    parser.add_argument("--tool", type=str, choices=['pytube', 'youtube-dl', 'api', 'invidious', 'scraping', 'all'],
                       default='all',
                       help="指定测试哪个工具（默认: all）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YouTube 标题获取工具测试")
    print("=" * 60)
    print(f"测试 URL 数量: {len(TEST_URLS)}")
    print()
    
    results = {}
    
    if args.tool in ['pytube', 'all']:
        results['pytube'] = test_pytube()
        time.sleep(1)
    
    if args.tool in ['youtube-dl', 'all']:
        results['youtube-dl'] = test_youtube_dl()
        time.sleep(1)
    
    if args.tool in ['api', 'all']:
        results['api'] = test_youtube_api(args.api_key)
        time.sleep(1)
    
    if args.tool in ['invidious', 'all']:
        results['invidious'] = test_invidious()
        time.sleep(1)
    
    if args.tool in ['scraping', 'all']:
        results['scraping'] = test_web_scraping()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for tool, success in results.items():
        status = "✅ 可用" if success else "❌ 不可用"
        print(f"{tool:20s}: {status}")
    
    # 推荐
    print("\n推荐使用:")
    if results.get('pytube'):
        print("  1. pytube (简单易用，无需 API Key)")
    if results.get('invidious'):
        print("  2. Invidious API (无需 API Key，隐私友好)")
    if results.get('api'):
        print("  3. YouTube Data API v3 (最稳定，需要 API Key)")
    
    if not any(results.values()):
        print("  ⚠️  所有工具都不可用，请检查网络连接或安装相关库")


if __name__ == "__main__":
    main()

