#!/usr/bin/env python3
"""
测试 GPU 解码功能

验证 NVDEC 硬件解码是否正常工作，并与 CPU 解码性能对比。
"""

import sys
import time
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.clip4mc_data_pipeline import (
    extract_all_frames_cv2,
    extract_all_frames_gpu_ffmpeg
)

import numpy as np


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
        
        if 'cuda' in hwaccels or 'nvdec' in hwaccels:
            print("\n✅ ffmpeg 支持 NVDEC 硬件解码")
            return True
        else:
            print("\n❌ ffmpeg 不支持 NVDEC")
            print("需要重新编译 ffmpeg 并启用 --enable-cuda-nvdec")
            return False
    
    except Exception as e:
        print(f"\n❌ 检查失败: {e}")
        return False


def check_nvidia_gpu():
    """检查 NVIDIA GPU 和驱动"""
    print("\n" + "=" * 60)
    print("检查 NVIDIA GPU")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            info = result.stdout.strip()
            print(f"GPU 信息: {info}")
            print("✅ NVIDIA GPU 可用")
            return True
        else:
            print("❌ nvidia-smi 执行失败")
            return False
    
    except FileNotFoundError:
        print("❌ nvidia-smi 未找到（NVIDIA 驱动未安装？）")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def test_decode_performance(video_path: Path, gpu_id: int = 0, runs: int = 3):
    """测试 CPU vs GPU 解码性能"""
    print("\n" + "=" * 60)
    print(f"性能测试: {video_path.name}")
    print("=" * 60)
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 测试 CPU 解码
    print("\n🔵 CPU 解码测试 (cv2.VideoCapture)")
    cpu_times = []
    cpu_frames = None
    
    for i in range(runs):
        start = time.time()
        frames = extract_all_frames_cv2(video_path)
        elapsed = time.time() - start
        
        if frames is not None:
            cpu_times.append(elapsed)
            if cpu_frames is None:
                cpu_frames = frames
            print(f"  Run {i+1}: {elapsed:.3f}s, 形状: {frames.shape}")
        else:
            print(f"  Run {i+1}: 失败")
    
    if cpu_times:
        avg_cpu = np.mean(cpu_times)
        print(f"\n  平均: {avg_cpu:.3f}s")
    else:
        print("\n  ❌ CPU 解码全部失败")
        return
    
    # 测试 GPU 解码
    print("\n🟢 GPU 解码测试 (ffmpeg NVDEC)")
    gpu_times = []
    gpu_frames = None
    
    for i in range(runs):
        start = time.time()
        frames = extract_all_frames_gpu_ffmpeg(video_path, gpu_id=gpu_id)
        elapsed = time.time() - start
        
        if frames is not None:
            gpu_times.append(elapsed)
            if gpu_frames is None:
                gpu_frames = frames
            print(f"  Run {i+1}: {elapsed:.3f}s, 形状: {frames.shape}")
        else:
            print(f"  Run {i+1}: 失败（可能回退到 CPU）")
    
    if gpu_times:
        avg_gpu = np.mean(gpu_times)
        print(f"\n  平均: {avg_gpu:.3f}s")
    else:
        print("\n  ❌ GPU 解码全部失败")
        return
    
    # 性能对比
    print("\n" + "-" * 60)
    print("性能对比")
    print("-" * 60)
    print(f"CPU 平均: {avg_cpu:.3f}s")
    print(f"GPU 平均: {avg_gpu:.3f}s")
    print(f"加速比: {avg_cpu / avg_gpu:.2f}x")
    
    if avg_gpu < avg_cpu:
        print(f"✅ GPU 比 CPU 快 {avg_cpu / avg_gpu:.2f}x")
    else:
        print(f"⚠️ GPU 未比 CPU 快（可能回退到 CPU 或硬件未启用）")
    
    # 数据一致性检查
    if cpu_frames is not None and gpu_frames is not None:
        print("\n" + "-" * 60)
        print("数据一致性检查")
        print("-" * 60)
        
        if cpu_frames.shape == gpu_frames.shape:
            print(f"✅ 形状一致: {cpu_frames.shape}")
            
            # 检查像素差异（允许小误差，因为解码器可能略有不同）
            diff = np.abs(cpu_frames.astype(float) - gpu_frames.astype(float))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"最大像素差异: {max_diff:.1f}")
            print(f"平均像素差异: {mean_diff:.3f}")
            
            if max_diff < 10:
                print("✅ 像素差异在可接受范围内")
            else:
                print("⚠️ 像素差异较大（可能是解码器差异）")
        else:
            print(f"❌ 形状不一致!")
            print(f"  CPU: {cpu_frames.shape}")
            print(f"  GPU: {gpu_frames.shape}")


def check_nvdec_concurrent_limit(video_path: Path, max_workers: int = 32):
    """测试 NVDEC 并发限制"""
    print("\n" + "=" * 60)
    print("NVDEC 并发测试")
    print("=" * 60)
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    import concurrent.futures
    
    def decode_task(worker_id):
        """单个解码任务"""
        frames = extract_all_frames_gpu_ffmpeg(video_path, gpu_id=0)
        if frames is not None:
            return (worker_id, True, frames.shape[0])
        else:
            return (worker_id, False, 0)
    
    print(f"测试 {max_workers} 个并发 NVDEC 解码...")
    print("（监控提示：在另一个终端运行 'watch -n 1 nvidia-smi dmon -s u'）")
    
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(decode_task, i) for i in range(max_workers)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    elapsed = time.time() - start
    
    success_count = sum(1 for _, success, _ in results if success)
    total_frames = sum(frames for _, _, frames in results)
    
    print(f"\n结果:")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  成功数: {success_count}/{max_workers}")
    print(f"  失败数: {max_workers - success_count}")
    print(f"  吞吐量: {max_workers / elapsed:.2f} video/s")
    print(f"  总帧数: {total_frames:,}")
    
    if success_count == max_workers:
        print(f"\n✅ NVDEC 支持 {max_workers} 个并发会话")
    else:
        print(f"\n⚠️ 部分会话失败（{max_workers - success_count} 个）")
        print("提示：检查 nvidia-smi 输出或降低并发数")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 GPU 解码功能")
    parser.add_argument("--video", type=str, required=True, help="测试视频路径")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--runs", type=int, default=3, help="性能测试运行次数")
    parser.add_argument("--test-concurrent", action='store_true', help="测试并发限制")
    parser.add_argument("--max-workers", type=int, default=16, help="并发测试最大 worker 数")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    # 1. 检查环境
    has_gpu = check_nvidia_gpu()
    has_nvdec = check_ffmpeg_nvdec()
    
    if not has_gpu or not has_nvdec:
        print("\n❌ 环境不满足 GPU 解码要求")
        print("请确保:")
        print("  1. NVIDIA GPU 和驱动已安装")
        print("  2. ffmpeg 编译时启用了 NVDEC 支持")
        return 1
    
    # 2. 性能测试
    test_decode_performance(video_path, args.gpu_id, args.runs)
    
    # 3. 并发测试（可选）
    if args.test_concurrent:
        check_nvdec_concurrent_limit(video_path, args.max_workers)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

