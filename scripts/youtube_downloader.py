#!/usr/bin/env python3
"""
独立 YouTube 视频下载脚本

依赖最少，适合临时部署到生产机器长时间下载。

依赖:
    pip install yt-dlp tqdm

使用方法:
    # 下载单个视频
    python youtube_downloader.py -u "https://www.youtube.com/watch?v=VIDEO_ID" -o ./videos

    # 从文件批量下载 (支持 URL 或 ID，每行一个)
    python youtube_downloader.py -f video_list.txt -o ./videos

    # 使用 Bright Data 代理 (推荐，避免 IP 被封)
    python youtube_downloader.py -f video_list.txt -o ./videos \\
        --proxy "http://USER:PASS@brd.superproxy.io:33335"

    # 使用环境变量配置代理
    export BRIGHT_PROXY="http://brd-customer-XXX:PASSWORD@brd.superproxy.io:33335"
    python youtube_downloader.py -f video_list.txt -o ./videos --proxy env
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ============================================================
# 配置
# ============================================================

DEFAULT_RESOLUTION = 360
DEFAULT_TIMEOUT = 300
MAX_403_RETRIES = 3

# Bright Data 代理配置 (可通过环境变量覆盖)
# 格式: http://USER:PASS@HOST:PORT
BRIGHT_DATA_PROXY_ENV = "BRIGHT_PROXY"

# YouTube URL 正则表达式
YOUTUBE_URL_PATTERNS = [
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
    r'^([a-zA-Z0-9_-]{11})$',  # 纯 ID
]


def extract_video_id(url_or_id: str) -> str:
    """从 URL 或直接输入中提取 YouTube 视频 ID"""
    url_or_id = url_or_id.strip()
    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # 如果没匹配到，假设就是 ID
    return url_or_id


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# 进度显示 (tqdm)
# ============================================================

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm 未安装，使用简单进度条。安装: pip install tqdm")


class SimpleProgress:
    """简单进度条，无外部依赖"""
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.success = 0
        self.failed = 0
    
    def update(self, success=True):
        self.current += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
        self._print()
    
    def _print(self):
        elapsed = time.time() - self.start_time
        pct = self.current / self.total * 100 if self.total > 0 else 0
        eta = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0
        
        bar_len = 30
        filled = int(bar_len * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        
        status = f"✓{self.success} ✗{self.failed}"
        time_str = f"{int(elapsed)}s/{int(eta)}s"
        
        print(f"\r{self.desc} |{bar}| {pct:5.1f}% {self.current}/{self.total} {status} [{time_str}]", 
              end='', file=sys.stderr, flush=True)
    
    def finish(self):
        print(file=sys.stderr)


class TqdmProgress:
    """使用 tqdm 的进度条"""
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.success = 0
        self.failed = 0
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit="video",
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format='{desc} |{bar}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    
    def update(self, success=True):
        if success:
            self.success += 1
        else:
            self.failed += 1
        self.pbar.set_postfix_str(f"✓{self.success} ✗{self.failed}")
        self.pbar.update(1)
    
    def finish(self):
        self.pbar.close()


def create_progress(total, desc="📥 下载"):
    """创建进度条"""
    if HAS_TQDM:
        return TqdmProgress(total, desc)
    return SimpleProgress(total, desc)


# ============================================================
# 下载异常
# ============================================================

class DownloadAbortError(Exception):
    """严重错误，需要终止下载"""
    pass


# ============================================================
# 核心下载功能
# ============================================================

def download_video(
    video_id: str,
    output_path: Path,
    cookies_file: str = None,
    resolution: int = DEFAULT_RESOLUTION,
    timeout: int = DEFAULT_TIMEOUT,
    debug: bool = False,
    use_impersonate: bool = True,
    proxy: str = None,
) -> bool:
    """
    下载 YouTube 视频
    
    Args:
        video_id: YouTube 视频 ID
        output_path: 输出文件路径
        cookies_file: cookies.txt 文件路径
        resolution: 最大分辨率
        timeout: 超时时间
        debug: 是否显示调试信息
        use_impersonate: 是否使用 curl_cffi impersonate
        proxy: 代理地址，格式 http://user:pass@host:port
    
    Returns:
        是否成功
        
    Raises:
        DownloadAbortError: 遇到 403 重试失败
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("请安装 yt-dlp: pip install yt-dlp")
        sys.exit(1)
    
    if output_path.exists():
        logger.debug(f"已存在: {output_path.name}")
        return True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # HLS 格式映射 (避免 403)
    hls_formats = {144: "91", 240: "92", 360: "93", 480: "94", 720: "95", 1080: "96"}
    hls_format = hls_formats.get(resolution, "93")
    
    ydl_opts = {
        'format': f'{hls_format}/91/92/93/best',
        'outtmpl': str(output_path.with_suffix('.mp4')),
        'noplaylist': True,
        'quiet': not debug,
        'no_warnings': not debug,
        'verbose': debug,
        'socket_timeout': timeout,
        'retries': 10,
        'fragment_retries': 20,
        'extractor_retries': 3,
        'sleep_interval': 1,
        'max_sleep_interval': 5,
        'noprogress': not debug,
    }
    
    # 配置代理
    if proxy:
        ydl_opts['proxy'] = proxy
    
    # 尝试使用 curl_cffi impersonate (代理模式下通常不需要)
    has_curl_cffi = False
    if use_impersonate and not proxy:
        try:
            import curl_cffi
            try:
                from yt_dlp.networking.impersonate import ImpersonateTarget
                ydl_opts['impersonate'] = ImpersonateTarget('chrome')
            except ImportError:
                ydl_opts['impersonate'] = 'chrome'
            has_curl_cffi = True
        except ImportError:
            pass
    
    if debug:
        if proxy:
            # 隐藏密码显示
            proxy_display = re.sub(r':([^:@]+)@', ':***@', proxy)
            logger.debug(f"proxy: ✓ {proxy_display}")
        else:
            logger.debug(f"proxy: ✗ 未配置")
        if use_impersonate and not proxy:
            logger.debug(f"impersonate: {'✓ 已启用' if has_curl_cffi else '✗ curl_cffi 未安装'}")
        logger.debug(f"cookies: {cookies_file if cookies_file else '未指定'}")
        logger.debug(f"format: {ydl_opts['format']}")
    
    if cookies_file and Path(cookies_file).exists():
        ydl_opts['cookiefile'] = cookies_file
    
    def try_download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 检查输出文件
        if output_path.exists():
            return True
        mp4_path = output_path.with_suffix('.mp4')
        if mp4_path.exists() and mp4_path != output_path:
            mp4_path.rename(output_path)
            return True
        # 查找可能的文件
        for f in output_path.parent.glob(f"{video_id}*.*"):
            if f.suffix in ['.mp4', '.webm', '.mkv']:
                f.rename(output_path)
                return True
        return False
    
    try:
        if try_download():
            return True
        logger.warning(f"下载完成但文件不存在: {video_id}")
        return False
        
    except Exception as e:
        error_msg = str(e).strip()
        
        # 如果错误信息为空，获取完整 traceback
        if not error_msg:
            import traceback
            error_msg = traceback.format_exc()
            if debug:
                logger.error(f"完整错误:\n{error_msg}")
        
        if "Video unavailable" in error_msg or "Private video" in error_msg:
            logger.warning(f"视频不可用: {video_id}")
            return False
        elif "Sign in to confirm" in error_msg or "age" in error_msg.lower():
            logger.warning(f"需要登录验证: {video_id} (尝试使用 cookies)")
            return False
        elif "404" in error_msg:
            logger.warning(f"视频不存在: {video_id}")
            return False
        elif "403" in error_msg:
            # 403 错误：指数退避重试
            for retry in range(MAX_403_RETRIES):
                wait_time = 60 * (2 ** retry)
                logger.warning(f"⚠️ 403 错误，等待 {wait_time}s 后重试 ({retry+1}/{MAX_403_RETRIES})")
                time.sleep(wait_time)
                
                try:
                    if try_download():
                        logger.info(f"✓ 重试成功: {video_id}")
                        return True
                except Exception as retry_e:
                    if "403" not in str(retry_e):
                        break
            
            logger.error(f"❌ 403 重试失败: {video_id}")
            raise DownloadAbortError(f"403 Forbidden - {video_id}")
        elif "DownloadError" in error_msg or "ExtractorError" in error_msg:
            # yt-dlp 特定错误，提取有用信息
            lines = error_msg.split('\n')
            for line in lines:
                if 'ERROR' in line:
                    logger.warning(f"下载失败 {video_id}: {line.strip()[:100]}")
                    break
            else:
                logger.warning(f"下载失败 {video_id}: {error_msg[:150]}")
            return False
        else:
            logger.warning(f"下载失败 {video_id}: {error_msg[:150]}")
            return False


# ============================================================
# 批量下载
# ============================================================

def load_checkpoint(checkpoint_file: Path) -> dict:
    """加载检查点"""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": [], "failed": [], "last_idx": 0}


def save_checkpoint(checkpoint_file: Path, data: dict):
    """保存检查点"""
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)


def download_batch(
    video_ids: list,
    output_dir: Path,
    cookies_file: str = None,
    resolution: int = DEFAULT_RESOLUTION,
    resume: bool = True,
    debug: bool = False,
    use_impersonate: bool = True,
    proxy: str = None,
) -> dict:
    """
    批量下载视频
    
    Args:
        video_ids: 视频 ID 列表，支持 str 或 dict 格式
        output_dir: 输出目录
        cookies_file: cookies 文件
        resolution: 分辨率
        resume: 是否从检查点恢复
        debug: 是否显示调试信息
        use_impersonate: 是否使用 curl_cffi impersonate
        proxy: 代理地址
    
    Returns:
        统计信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = output_dir / ".checkpoint.json"
    
    # 加载检查点
    checkpoint = load_checkpoint(checkpoint_file) if resume else {"completed": [], "failed": [], "last_idx": 0}
    completed_set = set(checkpoint["completed"])
    failed_set = set(checkpoint["failed"])
    
    # 解析视频列表
    tasks = []
    for item in video_ids:
        if isinstance(item, str):
            tasks.append({"vid": item})
        elif isinstance(item, dict):
            tasks.append(item)
    
    total = len(tasks)
    skip_count = len(completed_set) + len(failed_set)
    
    logger.info(f"📦 任务: {total} 个视频, 已完成: {skip_count}")
    
    progress = create_progress(total, "📥 下载")
    # 设置初始值
    if hasattr(progress, 'current'):
        progress.current = skip_count
    if hasattr(progress, 'pbar'):
        progress.pbar.n = skip_count
        progress.pbar.refresh()
    progress.success = len(completed_set)
    progress.failed = len(failed_set)
    
    stats = {"success": len(completed_set), "failed": len(failed_set), "skipped": 0}
    
    try:
        for idx, task in enumerate(tasks):
            vid = task.get("vid", "")
            if not vid:
                continue
            
            # 跳过已处理
            if vid in completed_set or vid in failed_set:
                continue
            
            # 下载
            video_path = output_dir / f"{vid}.mp4"
            
            try:
                success = download_video(
                    video_id=vid,
                    output_path=video_path,
                    cookies_file=cookies_file,
                    resolution=resolution,
                    debug=debug,
                    use_impersonate=use_impersonate,
                    proxy=proxy,
                )
                
                if success:
                    completed_set.add(vid)
                    stats["success"] += 1
                else:
                    failed_set.add(vid)
                    stats["failed"] += 1
                
                progress.update(success)
                
            except DownloadAbortError as e:
                # 403 错误，保存检查点并退出
                logger.error(f"\n❌ 下载终止: {e}")
                checkpoint["completed"] = list(completed_set)
                checkpoint["failed"] = list(failed_set)
                checkpoint["last_idx"] = idx
                save_checkpoint(checkpoint_file, checkpoint)
                logger.info(f"💾 检查点已保存，重新运行即可继续")
                progress.finish()
                return stats
            
            # 定期保存检查点
            if (stats["success"] + stats["failed"]) % 20 == 0:
                checkpoint["completed"] = list(completed_set)
                checkpoint["failed"] = list(failed_set)
                checkpoint["last_idx"] = idx
                save_checkpoint(checkpoint_file, checkpoint)
            
            # 避免请求过快
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ 用户中断，保存检查点...")
        checkpoint["completed"] = list(completed_set)
        checkpoint["failed"] = list(failed_set)
        save_checkpoint(checkpoint_file, checkpoint)
        logger.info("💾 检查点已保存")
        raise
    
    progress.finish()
    
    # 完成，清理检查点
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 下载完成")
    logger.info(f"   ✓ 成功: {stats['success']}")
    logger.info(f"   ✗ 失败: {stats['failed']}")
    logger.info(f"{'='*50}")
    
    return stats


# ============================================================
# 命令行接口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="YouTube 视频下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载单个视频 (推荐使用 URL，避免 - 开头的 ID 问题)
  python youtube_downloader.py -u "https://www.youtube.com/watch?v=-DC1yJYbkXY" -o ./videos

  # 用视频 ID (以 - 开头时用 = 连接)
  python youtube_downloader.py -v="-DC1yJYbkXY" -o ./videos

  # 从文件批量下载 (支持 URL 或 ID，每行一个)
  python youtube_downloader.py -f video_list.txt -o ./videos

  # 使用 Bright Data 代理 (推荐)
  python youtube_downloader.py -f video_list.txt -o ./videos --proxy "http://USER:PASS@brd.superproxy.io:33335"

  # 使用环境变量配置代理
  export BRIGHT_PROXY="http://brd-customer-XXX:PASSWORD@brd.superproxy.io:33335"
  python youtube_downloader.py -f video_list.txt -o ./videos --proxy env
        """
    )
    
    # 输入源 (互斥)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-u", "--url", help="YouTube URL (推荐，避免 - 开头的 ID 问题)")
    input_group.add_argument("-v", "--video-id", 
                            help="视频 ID (以-开头请用: -v='-xxx')")
    input_group.add_argument("-f", "--file", help="URL/ID 列表文件 (每行一个)")
    
    # 输出选项
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("-n", "--max-count", type=int, help="最大下载数量")
    
    # 下载选项
    parser.add_argument("-c", "--cookies", help="cookies.txt 文件 (账号被封后不需要)")
    parser.add_argument("-r", "--resolution", type=int, default=360,
                       choices=[360, 480, 720, 1080], help="分辨率 (默认 360)")
    parser.add_argument("--proxy", "-p", help="代理地址 (格式: http://user:pass@host:port, 或 'env' 使用环境变量)")
    parser.add_argument("--no-resume", action="store_true", help="不从检查点恢复")
    parser.add_argument("--no-impersonate", action="store_true", 
                       help="禁用 curl_cffi impersonate")
    
    # 调试
    parser.add_argument("--debug", action="store_true", help="显示调试信息")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    output_dir = Path(args.output)
    
    # 处理代理配置
    proxy = None
    if args.proxy:
        if args.proxy.lower() == 'env':
            proxy = os.environ.get(BRIGHT_DATA_PROXY_ENV)
            if not proxy:
                logger.error(f"环境变量 {BRIGHT_DATA_PROXY_ENV} 未设置")
                logger.error(f"请设置: export {BRIGHT_DATA_PROXY_ENV}=\"http://user:pass@host:port\"")
                sys.exit(1)
        else:
            proxy = args.proxy
        
        # 显示代理信息（隐藏密码）
        proxy_display = re.sub(r':([^:@]+)@', ':***@', proxy)
        logger.info(f"🔒 使用代理: {proxy_display}")
    
    # 确定视频列表
    video_ids = []
    
    if args.url:
        vid = extract_video_id(args.url)
        if vid:
            video_ids = [vid]
            logger.debug(f"从 URL 提取 ID: {vid}")
        else:
            logger.error(f"无法从 URL 提取视频 ID: {args.url}")
            sys.exit(1)
    
    elif args.video_id:
        video_ids = [args.video_id]
    
    elif args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 支持 URL 或 ID
                    vid = extract_video_id(line)
                    if vid:
                        video_ids.append(vid)
        logger.info(f"📂 从文件加载 {len(video_ids)} 个视频")
    
    # 限制数量
    if args.max_count and len(video_ids) > args.max_count:
        video_ids = video_ids[:args.max_count]
        logger.info(f"⚠️ 限制为前 {args.max_count} 个")
    
    if not video_ids:
        logger.error("没有找到视频 ID")
        sys.exit(1)
    
    # 开始下载
    logger.info(f"🚀 开始下载 {len(video_ids)} 个视频到 {output_dir}")
    
    download_batch(
        video_ids=video_ids,
        output_dir=output_dir,
        cookies_file=args.cookies,
        resolution=args.resolution,
        resume=not args.no_resume,
        debug=args.debug,
        use_impersonate=not args.no_impersonate,
        proxy=proxy,
    )


if __name__ == "__main__":
    main()

