#!/usr/bin/env python3
"""
GPU 解码诊断工具

检查 ffmpeg GPU 解码是否正常工作
"""

import subprocess
import sys
from pathlib import Path

def check_ffmpeg_nvdec():
    """检查 ffmpeg 是否支持 NVDEC"""
    print("=" * 60)
    print("检查 ffmpeg NVDEC 支持")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-hwaccels'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        hwaccels = result.stdout
        print(hwaccels)
        
        if 'cuda' in hwaccels:
            print("✅ ffmpeg 支持 CUDA 硬件加速")
            return True
        else:
            print("❌ ffmpeg 不支持 CUDA 硬件加速")
            print("\n建议:")
            print("1. 重新编译 ffmpeg 并启用 NVDEC:")
            print("   参考: https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/")
            print("2. 或使用纯 CPU 模式（不加 --use-gpu）")
            return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def test_gpu_decode(video_path: str, gpu_id: int = 0):
    """测试 GPU 解码一个视频"""
    print("\n" + "=" * 60)
    print(f"测试 GPU 解码: {video_path}")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    # 测试 GPU 解码
    cmd_gpu = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-hwaccel_device', str(gpu_id),
        '-i', video_path,
        '-vf', 'select=not(mod(n\\,10)),scale=256:160',
        '-vsync', '0',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-frames:v', '16',
        '-y',
        '/dev/null'
    ]
    
    print("\n运行命令:")
    print(' '.join(cmd_gpu))
    print()
    
    try:
        result = subprocess.run(
            cmd_gpu,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ GPU 解码成功")
            return True
        else:
            print("❌ GPU 解码失败")
            print("\nSTDERR:")
            print(result.stderr)
            
            # 常见错误分析
            if 'No NVDEC capable devices found' in result.stderr:
                print("\n原因: GPU 不支持 NVDEC 或驱动问题")
            elif 'hwaccel' in result.stderr.lower():
                print("\n原因: ffmpeg 未正确启用硬件加速")
            elif 'codec' in result.stderr.lower():
                print("\n原因: 视频编码格式不支持 NVDEC")
                print("支持格式: H.264, H.265, VP9")
                print("不支持: VP8, AV1")
            
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_cpu_decode(video_path: str):
    """测试 CPU 解码（作为对比）"""
    print("\n" + "=" * 60)
    print(f"测试 CPU 解码: {video_path}")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    cmd_cpu = [
        'ffmpeg',
        '-i', video_path,
        '-vf', 'select=not(mod(n\\,10)),scale=256:160',
        '-vsync', '0',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-frames:v', '16',
        '-y',
        '/dev/null'
    ]
    
    try:
        result = subprocess.run(
            cmd_cpu,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ CPU 解码成功")
            return True
        else:
            print("❌ CPU 解码失败")
            print("\nSTDERR:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU 解码诊断工具")
    parser.add_argument("--video", type=str, help="测试视频路径")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    
    # 1. 检查 ffmpeg 支持
    ffmpeg_ok = check_ffmpeg_nvdec()
    
    # 2. 如果提供了视频，测试解码
    if args.video:
        cpu_ok = test_cpu_decode(args.video)
        
        if ffmpeg_ok:
            gpu_ok = test_gpu_decode(args.video, args.gpu_id)
            
            print("\n" + "=" * 60)
            print("诊断总结")
            print("=" * 60)
            print(f"ffmpeg NVDEC 支持: {'✅' if ffmpeg_ok else '❌'}")
            print(f"CPU 解码:          {'✅' if cpu_ok else '❌'}")
            print(f"GPU 解码:          {'✅' if gpu_ok else '❌'}")
            
            if not gpu_ok:
                print("\n⚠️ GPU 解码失败，数据流水线将回退到 CPU 模式")
                print("   这会导致 GPU 利用率很低（只有 tokenization 用 CPU）")
                print("\n建议:")
                print("1. 如果 GPU 解码不可用，使用纯 CPU 模式:")
                print("   python src/utils/clip4mc_data_pipeline.py --num-workers 16 \\")
                print("      (不加 --use-gpu)")
                print()
                print("2. 或修复 ffmpeg GPU 解码支持")
        else:
            print("\n⚠️ ffmpeg 不支持 NVDEC，无法进行 GPU 解码")
    else:
        print("\n使用方法:")
        print("  python scripts/diagnose_gpu_decode.py --video /path/to/video.mp4")


if __name__ == "__main__":
    main()

