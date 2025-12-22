#!/usr/bin/env python3
"""
诊断FFmpeg帧提取失败的原因

使用方法:
    python scripts/diagnose_ffmpeg_failure.py \
        --video-path "/path/to/video.mp4" \
        --start-time 636.4 \
        --duration 3.2
"""

import subprocess
import argparse
from pathlib import Path
import sys


def check_video_exists(video_path: Path):
    """检查视频文件是否存在"""
    print("1. 检查视频文件...")
    if not video_path.exists():
        print(f"   ❌ 文件不存在: {video_path}")
        return False
    
    size = video_path.stat().st_size
    print(f"   ✅ 文件存在")
    print(f"   大小: {size / 1024 / 1024:.2f} MB")
    return True


def check_video_info(video_path: Path):
    """获取视频信息"""
    print("\n2. 检查视频基本信息...")
    
    # 获取时长
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        if result.returncode == 0:
            duration = float(result.stdout.decode().strip())
            print(f"   ✅ 视频时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
            return duration
        else:
            print(f"   ❌ 无法获取视频信息")
            print(f"   错误: {result.stderr.decode()}")
            return None
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return None


def check_video_codec(video_path: Path):
    """检查视频编码格式"""
    print("\n3. 检查视频编码...")
    
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,pix_fmt',
        '-of', 'default=noprint_wrappers=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        if result.returncode == 0:
            info = result.stdout.decode()
            print(f"   ✅ 视频信息:")
            for line in info.strip().split('\n'):
                print(f"      {line}")
        else:
            print(f"   ❌ 无法获取编码信息")
    except Exception as e:
        print(f"   ❌ 异常: {e}")


def test_frame_extraction(video_path: Path, start_time: float, duration: float):
    """测试帧提取"""
    print("\n4. 测试CPU模式帧提取...")
    
    frame_width = 256
    frame_height = 160
    
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', str(video_path),
        '-t', str(duration),
        '-vf', f'scale={frame_width}:{frame_height}',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-loglevel', 'error',
        'pipe:1'
    ]
    
    print(f"   命令: {' '.join(cmd[:8])}...")
    
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        print(f"   返回码: {result.returncode}")
        
        if result.stderr:
            stderr = result.stderr.decode().strip()
            if stderr:
                print(f"   stderr: {stderr}")
        
        if result.returncode == 0:
            output_size = len(result.stdout)
            frame_size = frame_height * frame_width * 3
            num_frames = output_size // frame_size
            
            print(f"   ✅ 提取成功")
            print(f"   输出大小: {output_size / 1024 / 1024:.2f} MB")
            print(f"   帧大小: {frame_size} 字节")
            print(f"   提取帧数: {num_frames}")
            
            if num_frames == 0:
                print(f"   ⚠️  警告: 没有提取到帧！")
                return False
            
            return True
        else:
            print(f"   ❌ 提取失败")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ 超时（30秒）")
        print(f"   可能原因:")
        print(f"      - 视频文件损坏")
        print(f"      - 视频编码复杂，解码太慢")
        print(f"      - 磁盘IO问题")
        return False
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False


def test_specific_timestamp(video_path: Path, start_time: float):
    """测试特定时间点是否可以seek"""
    print(f"\n5. 测试时间点 {start_time} 秒是否有效...")
    
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', str(video_path),
        '-frames:v', '1',
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"   ✅ 时间点有效")
            return True
        else:
            print(f"   ❌ 无法seek到该时间点")
            stderr = result.stderr.decode()
            if stderr:
                print(f"   错误: {stderr[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="诊断FFmpeg帧提取失败的原因"
    )
    
    parser.add_argument("--video-path", type=Path, required=True,
                       help="视频文件路径")
    parser.add_argument("--start-time", type=float, required=True,
                       help="开始时间（秒）")
    parser.add_argument("--duration", type=float, default=3.2,
                       help="时长（秒），默认3.2")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FFmpeg 帧提取失败诊断工具")
    print("=" * 70)
    print(f"\n视频路径: {args.video_path}")
    print(f"时间范围: {args.start_time}s - {args.start_time + args.duration}s")
    print()
    
    # 运行诊断
    results = []
    
    # 1. 检查文件存在
    if not check_video_exists(args.video_path):
        print("\n❌ 诊断结束：文件不存在")
        return 1
    results.append(True)
    
    # 2. 获取视频信息
    duration = check_video_info(args.video_path)
    if duration is None:
        print("\n⚠️  警告：无法获取视频信息，但继续诊断...")
        results.append(False)
    else:
        # 检查时间范围是否有效
        if args.start_time + args.duration > duration:
            print(f"\n⚠️  警告：请求的时间范围 {args.start_time}~{args.start_time+args.duration}s")
            print(f"          超出视频时长 {duration}s")
        results.append(True)
    
    # 3. 检查编码格式
    check_video_codec(args.video_path)
    
    # 4. 测试时间点
    timestamp_ok = test_specific_timestamp(args.video_path, args.start_time)
    results.append(timestamp_ok)
    
    # 5. 测试帧提取
    extraction_ok = test_frame_extraction(args.video_path, args.start_time, args.duration)
    results.append(extraction_ok)
    
    # 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    
    if all(results):
        print("✅ 所有测试通过！")
        print("\n可能的问题：")
        print("  1. 生产环境的视频文件与本地不同（损坏/不完整）")
        print("  2. 生产环境的FFmpeg版本不同")
        print("  3. 生产环境的超时时间不够")
        print("  4. 路径编码问题")
        print("\n建议：")
        print("  1. 检查生产环境视频文件的MD5")
        print("  2. 检查生产环境的FFmpeg版本: ffmpeg -version")
        print("  3. 增加超时时间（目前30秒）")
        return 0
    else:
        print("❌ 发现问题！")
        if not results[0]:
            print("  - 文件不存在")
        if not results[1]:
            print("  - 无法获取视频信息（文件可能损坏）")
        if not results[2]:
            print("  - 无法seek到指定时间点")
        if not results[3]:
            print("  - FFmpeg无法提取帧")
        
        print("\n建议：")
        print("  1. 检查视频文件完整性")
        print("  2. 尝试重新下载视频")
        print("  3. 使用其他工具播放视频验证")
        return 1


if __name__ == "__main__":
    sys.exit(main())

