#!/usr/bin/env python3
"""
分析保存的训练画面
快速查看MineCLIP相似度峰值对应的画面
"""

import os
import glob
import re
from pathlib import Path


def analyze_saved_frames(frames_dir="logs/frames", top_n=10):
    """
    分析保存的画面文件
    
    Args:
        frames_dir: 画面保存目录
        top_n: 显示相似度最高的N个画面
    """
    print("=" * 80)
    print("MineCLIP 画面分析")
    print("=" * 80)
    
    if not os.path.exists(frames_dir):
        print(f"❌ 目录不存在: {frames_dir}")
        print(f"请先运行训练并启用 --save-frames")
        return
    
    # 获取所有图片文件
    pattern = os.path.join(frames_dir, "step_*.png")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ 未找到画面文件")
        print(f"请确保训练时使用了 --save-frames 参数")
        return
    
    print(f"\n✓ 找到 {len(files)} 个画面文件")
    print(f"  目录: {frames_dir}")
    print()
    
    # 解析文件名提取信息
    frame_data = []
    for filepath in files:
        filename = os.path.basename(filepath)
        # 文件名格式: step_000100_sim_0.6337_mc_+0.0018_reward_+0.0179.png
        match = re.match(
            r'step_(\d+)_sim_([\d.]+)_mc_([-+\d.]+)_reward_([-+\d.]+)\.png',
            filename
        )
        if match:
            step = int(match.group(1))
            similarity = float(match.group(2))
            mc_reward = float(match.group(3))
            total_reward = float(match.group(4))
            frame_data.append({
                'filepath': filepath,
                'filename': filename,
                'step': step,
                'similarity': similarity,
                'mc_reward': mc_reward,
                'total_reward': total_reward
            })
    
    if not frame_data:
        print("❌ 无法解析文件名")
        return
    
    print(f"✓ 成功解析 {len(frame_data)} 个文件")
    print()
    
    # 按相似度排序
    sorted_by_sim = sorted(frame_data, key=lambda x: x['similarity'], reverse=True)
    
    # 显示相似度最高的画面
    print(f"📊 相似度最高的 {top_n} 个画面:")
    print("-" * 80)
    print(f"{'排名':>4s} | {'Step':>8s} | {'相似度':>8s} | {'MC奖励':>10s} | {'总奖励':>10s} | 文件名")
    print("-" * 80)
    
    for i, data in enumerate(sorted_by_sim[:top_n], 1):
        print(f"{i:>4d} | {data['step']:>8d} | {data['similarity']:>8.4f} | "
              f"{data['mc_reward']:>+10.4f} | {data['total_reward']:>+10.4f} | {data['filename']}")
    
    # 按相似度排序（最低）
    print()
    print(f"📊 相似度最低的 {top_n} 个画面:")
    print("-" * 80)
    print(f"{'排名':>4s} | {'Step':>8s} | {'相似度':>8s} | {'MC奖励':>10s} | {'总奖励':>10s} | 文件名")
    print("-" * 80)
    
    sorted_by_sim_low = sorted(frame_data, key=lambda x: x['similarity'])
    for i, data in enumerate(sorted_by_sim_low[:top_n], 1):
        print(f"{i:>4d} | {data['step']:>8d} | {data['similarity']:>8.4f} | "
              f"{data['mc_reward']:>+10.4f} | {data['total_reward']:>+10.4f} | {data['filename']}")
    
    # 统计信息
    similarities = [d['similarity'] for d in frame_data]
    mc_rewards = [d['mc_reward'] for d in frame_data]
    
    print()
    print("=" * 80)
    print("统计信息:")
    print("=" * 80)
    print(f"相似度范围:  {min(similarities):.6f} ~ {max(similarities):.6f}")
    print(f"相似度均值:  {sum(similarities)/len(similarities):.6f}")
    print(f"相似度波动:  {max(similarities) - min(similarities):.6f}")
    print()
    print(f"MC奖励范围:  {min(mc_rewards):+.6f} ~ {max(mc_rewards):+.6f}")
    print(f"MC奖励均值:  {sum(mc_rewards)/len(mc_rewards):+.6f}")
    print()
    print("=" * 80)
    print("💡 建议:")
    print("=" * 80)
    print("1. 查看相似度最高的画面，看是否包含树木")
    print("2. 比较高相似度和低相似度画面的差异")
    print("3. 如果相似度波动小(<2%)，说明agent可能没看到树")
    print("4. 打开画面文件路径查看:")
    print(f"   {os.path.abspath(frames_dir)}/")
    print()
    
    # 生成快速查看命令（macOS）
    if sorted_by_sim:
        top_file = sorted_by_sim[0]['filepath']
        print("快速查看命令（查看相似度最高的画面）:")
        print(f"  open '{top_file}'")
        print()


if __name__ == "__main__":
    import sys
    
    frames_dir = "logs/frames"
    if len(sys.argv) > 1:
        frames_dir = sys.argv[1]
    
    analyze_saved_frames(frames_dir, top_n=10)

