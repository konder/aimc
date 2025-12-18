#!/usr/bin/env python3
"""
批量 GPU 解码性能测试

对比 CPU 和 GPU 在批量处理时的实际吞吐量。
这比单视频测试更能反映生产环境性能。
"""

import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.clip4mc_data_pipeline import (
    extract_all_frames_cv2,
    extract_all_frames_gpu_ffmpeg
)


def test_cpu_decode(video_path: Path) -> tuple:
    """CPU 解码单个视频"""
    start = time.time()
    frames = extract_all_frames_cv2(video_path)
    elapsed = time.time() - start
    
    if frames is not None:
        return (True, elapsed, frames.shape[0])
    else:
        return (False, elapsed, 0)


def test_gpu_decode(video_path: Path, gpu_id: int = 0) -> tuple:
    """GPU 解码单个视频"""
    start = time.time()
    frames = extract_all_frames_gpu_ffmpeg(video_path, gpu_id=gpu_id)
    elapsed = time.time() - start
    
    if frames is not None:
        return (True, elapsed, frames.shape[0])
    else:
        return (False, elapsed, 0)


def test_batch_cpu(video_paths: List[Path], num_workers: int = 1) -> dict:
    """批量 CPU 解码测试"""
    print(f"\n{'='*60}")
    print(f"CPU 批量测试 ({len(video_paths)} 个视频, {num_workers} workers)")
    print(f"{'='*60}")
    
    start_time = time.time()
    results = []
    
    if num_workers == 1:
        # 串行处理
        for video_path in video_paths:
            result = test_cpu_decode(video_path)
            results.append(result)
    else:
        # 并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(test_cpu_decode, vp) for vp in video_paths]
            for future in as_completed(futures):
                results.append(future.result())
    
    total_time = time.time() - start_time
    
    success_count = sum(1 for r in results if r[0])
    total_decode_time = sum(r[1] for r in results)
    total_frames = sum(r[2] for r in results)
    
    return {
        'method': 'CPU',
        'workers': num_workers,
        'total_videos': len(video_paths),
        'success_count': success_count,
        'total_time': total_time,
        'total_decode_time': total_decode_time,
        'total_frames': total_frames,
        'throughput': success_count / total_time if total_time > 0 else 0,
        'avg_per_video': total_time / len(video_paths) if len(video_paths) > 0 else 0
    }


def test_batch_gpu(video_paths: List[Path], num_workers: int = 8, gpu_id: int = 0) -> dict:
    """批量 GPU 解码测试"""
    print(f"\n{'='*60}")
    print(f"GPU 批量测试 ({len(video_paths)} 个视频, {num_workers} workers)")
    print(f"{'='*60}")
    
    start_time = time.time()
    results = []
    
    if num_workers == 1:
        # 串行处理
        for video_path in video_paths:
            result = test_gpu_decode(video_path, gpu_id)
            results.append(result)
    else:
        # 并行处理
        # 注意：GPU 解码需要在进程内部调用，避免 CUDA 上下文问题
        def gpu_decode_wrapper(vp):
            return test_gpu_decode(vp, gpu_id)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(gpu_decode_wrapper, vp) for vp in video_paths]
            for future in as_completed(futures):
                results.append(future.result())
    
    total_time = time.time() - start_time
    
    success_count = sum(1 for r in results if r[0])
    total_decode_time = sum(r[1] for r in results)
    total_frames = sum(r[2] for r in results)
    
    return {
        'method': 'GPU',
        'workers': num_workers,
        'total_videos': len(video_paths),
        'success_count': success_count,
        'total_time': total_time,
        'total_decode_time': total_decode_time,
        'total_frames': total_frames,
        'throughput': success_count / total_time if total_time > 0 else 0,
        'avg_per_video': total_time / len(video_paths) if len(video_paths) > 0 else 0
    }


def print_results(cpu_result: dict, gpu_result: dict):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print("批量处理性能对比")
    print(f"{'='*60}")
    
    print(f"\n{'指标':<20} {'CPU':<20} {'GPU':<20} {'GPU提升':<15}")
    print(f"{'-'*75}")
    
    print(f"{'总视频数':<20} {cpu_result['total_videos']:<20} {gpu_result['total_videos']:<20} -")
    print(f"{'成功数':<20} {cpu_result['success_count']:<20} {gpu_result['success_count']:<20} -")
    print(f"{'总时间':<20} {cpu_result['total_time']:.2f}s{'':<16} {gpu_result['total_time']:.2f}s{'':<16} {cpu_result['total_time']/gpu_result['total_time']:.2f}x")
    print(f"{'平均每视频':<20} {cpu_result['avg_per_video']:.3f}s{'':<15} {gpu_result['avg_per_video']:.3f}s{'':<15} {cpu_result['avg_per_video']/gpu_result['avg_per_video']:.2f}x")
    print(f"{'吞吐量':<20} {cpu_result['throughput']:.2f} video/s{'':<9} {gpu_result['throughput']:.2f} video/s{'':<9} {gpu_result['throughput']/cpu_result['throughput']:.2f}x")
    print(f"{'总帧数':<20} {cpu_result['total_frames']:,}{'':<15} {gpu_result['total_frames']:,}{'':<15} -")
    
    print(f"\n{'='*60}")
    if gpu_result['throughput'] > cpu_result['throughput']:
        speedup = gpu_result['throughput'] / cpu_result['throughput']
        print(f"✅ GPU 比 CPU 快 {speedup:.2f}x")
        print(f"   300,000 视频预计时间：")
        print(f"   - CPU: {300000 / cpu_result['throughput'] / 3600:.1f} 小时")
        print(f"   - GPU: {300000 / gpu_result['throughput'] / 3600:.1f} 小时")
        print(f"   - 节省: {(300000 / cpu_result['throughput'] - 300000 / gpu_result['throughput']) / 3600:.1f} 小时")
    else:
        print(f"⚠️ GPU 未显示加速优势")
        print(f"   可能原因：")
        print(f"   1. 视频太短（<10秒），GPU初始化开销占比大")
        print(f"   2. workers 数不够（尝试增加 --gpu-workers）")
        print(f"   3. ffmpeg 未正确使用 GPU（检查 nvidia-smi）")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="批量 GPU 解码性能测试")
    parser.add_argument("--clips-dir", type=str, required=True, help="视频目录")
    parser.add_argument("--num-videos", type=int, default=100, help="测试视频数量")
    parser.add_argument("--cpu-workers", type=int, default=8, help="CPU workers 数")
    parser.add_argument("--gpu-workers", type=int, default=16, help="GPU workers 数")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--skip-cpu", action='store_true', help="跳过 CPU 测试")
    parser.add_argument("--skip-gpu", action='store_true', help="跳过 GPU 测试")
    
    args = parser.parse_args()
    
    clips_dir = Path(args.clips_dir)
    
    if not clips_dir.exists():
        print(f"❌ 目录不存在: {clips_dir}")
        return 1
    
    # 收集视频文件
    video_files = list(clips_dir.glob("*.mp4"))
    if not video_files:
        print(f"❌ 目录中没有 .mp4 文件: {clips_dir}")
        return 1
    
    # 限制数量
    video_files = video_files[:args.num_videos]
    
    print(f"找到 {len(video_files)} 个测试视频")
    
    # CPU 测试
    if not args.skip_cpu:
        cpu_result = test_batch_cpu(video_files, args.cpu_workers)
        print(f"\n✅ CPU 测试完成")
        print(f"   总时间: {cpu_result['total_time']:.2f}s")
        print(f"   吞吐量: {cpu_result['throughput']:.2f} video/s")
    else:
        cpu_result = None
    
    # GPU 测试
    if not args.skip_gpu:
        gpu_result = test_batch_gpu(video_files, args.gpu_workers, args.gpu_id)
        print(f"\n✅ GPU 测试完成")
        print(f"   总时间: {gpu_result['total_time']:.2f}s")
        print(f"   吞吐量: {gpu_result['throughput']:.2f} video/s")
    else:
        gpu_result = None
    
    # 对比结果
    if cpu_result and gpu_result:
        print_results(cpu_result, gpu_result)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

