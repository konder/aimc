#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证MineCLIP 16帧视频模式的效果
使用录制的帧序列或已有的logs/frames
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from mineclip import MineCLIP
from transformers import CLIPTokenizer
import matplotlib.pyplot as plt

def load_frames_from_directory(directory, max_frames=None):
    """
    从目录加载帧序列
    
    Args:
        directory: 帧目录
        max_frames: 最大加载帧数
    
    Returns:
        frames: List of [H, W, C] numpy arrays
        filenames: 对应的文件名
    """
    # 查找所有图像文件
    image_files = sorted(glob.glob(os.path.join(directory, "*.png")))
    
    if max_frames:
        image_files = image_files[:max_frames]
    
    print(f"  找到 {len(image_files)} 帧")
    
    frames = []
    filenames = []
    
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)  # [H, W, C]
        frames.append(img_array)
        filenames.append(os.path.basename(img_path))
    
    return frames, filenames

def compute_16frame_similarity(model, tokenizer, frames_16, text_prompt, device):
    """
    计算16帧视频与文本的相似度（官方方式）
    
    Args:
        model: MineCLIP模型
        tokenizer: CLIP tokenizer
        frames_16: List of 16 frames [H, W, C]
        text_prompt: 文本描述
        device: 设备
    
    Returns:
        similarity: 相似度分数
    """
    with torch.no_grad():
        # MineCraft官方归一化参数
        MC_MEAN = torch.tensor([0.3331, 0.3245, 0.3051], device=device).view(1, 1, 3, 1, 1)
        MC_STD = torch.tensor([0.2439, 0.2493, 0.2873], device=device).view(1, 1, 3, 1, 1)
        
        # 预处理帧
        processed = []
        for frame in frames_16:
            # 转tensor
            frame_t = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
            # HWC -> CHW
            frame_t = frame_t.permute(2, 0, 1)  # [C, H, W]
            processed.append(frame_t)
        
        # 堆叠为视频 [1, T, C, H, W]
        video = torch.stack(processed).unsqueeze(0).to(device)
        
        # MineCraft归一化
        video = (video - MC_MEAN) / MC_STD
        
        # 编码视频（官方完整流程）
        video_features = model.encode_video(video)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        # 编码文本
        tokens = tokenizer(text_prompt, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=77)
        text_features = model.encode_text(tokens['input_ids'].to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarity = (video_features @ text_features.T).item()
    
    return similarity

def analyze_sequence(sequence_dir, num_frames=16, stride=4, task_prompt="chopping a tree with hand"):
    """
    分析整个帧序列
    
    Args:
        sequence_dir: 帧序列目录
        num_frames: 每个视频片段的帧数
        stride: 滑动窗口步长
        task_prompt: 任务描述
    """
    device = "auto"
    
    print("=" * 80)
    print("MineCLIP 16帧视频模式验证")
    print("=" * 80)
    
    # 加载MineCLIP
    print("\n[1/4] 加载MineCLIP...")
    model = MineCLIP(
        arch="vit_base_p16_fz.v2.t2",
        hidden_dim=512,
        image_feature_dim=512,
        mlp_adapter_spec='v0-2.t0',
        pool_type="attn.d2.nh8.glusw",
        resolution=(160, 256)
    )
    model.load_ckpt("data/mineclip/attn.pth", strict=True)
    model.eval()
    model = model.to(device)
    
    tokenizer = CLIPTokenizer.from_pretrained("data/clip_tokenizer")
    print(f"  ✓ MineCLIP已加载")
    print(f"  任务描述: '{task_prompt}'")
    
    # 加载帧序列
    print(f"\n[2/4] 加载帧序列...")
    print(f"  目录: {sequence_dir}")
    
    frames, filenames = load_frames_from_directory(sequence_dir)
    print(f"  ✓ 加载了 {len(frames)} 帧")
    
    if len(frames) < num_frames:
        print(f"  ⚠️  帧数不足16帧，无法进行16帧分析")
        return
    
    # 滑动窗口分析
    print(f"\n[3/4] 滑动窗口分析...")
    print(f"  窗口大小: {num_frames}帧")
    print(f"  滑动步长: {stride}帧")
    
    similarities = []
    window_indices = []
    
    for i in range(0, len(frames) - num_frames + 1, stride):
        # 提取16帧
        frames_16 = frames[i:i+num_frames]
        
        # 计算相似度
        sim = compute_16frame_similarity(model, tokenizer, frames_16, task_prompt, device)
        
        similarities.append(sim)
        window_indices.append(i)
        
        print(f"  窗口 {len(similarities):3d} [帧{i:04d}-{i+num_frames-1:04d}]: 相似度 = {sim:.4f}", end='\r')
    
    print(f"\n  ✓ 分析了 {len(similarities)} 个窗口")
    
    # 统计分析
    print(f"\n[4/4] 统计分析...")
    print("=" * 80)
    
    similarities_np = np.array(similarities)
    
    print(f"\n📊 相似度统计:")
    print(f"  最小值: {similarities_np.min():.4f}")
    print(f"  最大值: {similarities_np.max():.4f}")
    print(f"  平均值: {similarities_np.mean():.4f}")
    print(f"  标准差: {similarities_np.std():.4f}")
    print(f"  变化范围: {similarities_np.max() - similarities_np.min():.4f}")
    
    # 找到最高和最低相似度的窗口
    max_idx = np.argmax(similarities_np)
    min_idx = np.argmin(similarities_np)
    
    print(f"\n🔝 最高相似度窗口:")
    print(f"  位置: 帧 {window_indices[max_idx]:04d}-{window_indices[max_idx]+num_frames-1:04d}")
    print(f"  相似度: {similarities_np[max_idx]:.4f}")
    print(f"  帧文件: {filenames[window_indices[max_idx]]} ~ {filenames[window_indices[max_idx]+num_frames-1]}")
    
    print(f"\n🔻 最低相似度窗口:")
    print(f"  位置: 帧 {window_indices[min_idx]:04d}-{window_indices[min_idx]+num_frames-1:04d}")
    print(f"  相似度: {similarities_np[min_idx]:.4f}")
    print(f"  帧文件: {filenames[window_indices[min_idx]]} ~ {filenames[window_indices[min_idx]+num_frames-1]}")
    
    # 可视化
    print(f"\n📈 生成可视化图表...")
    plt.figure(figsize=(12, 6))
    plt.plot(window_indices, similarities, marker='o', markersize=3, linewidth=1)
    plt.xlabel('起始帧索引')
    plt.ylabel('MineCLIP相似度')
    plt.title(f'MineCLIP 16帧视频相似度变化\n任务: "{task_prompt}"')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=similarities_np.mean(), color='r', linestyle='--', label=f'平均值: {similarities_np.mean():.4f}')
    plt.legend()
    
    # 保存图表
    plot_path = os.path.join(sequence_dir, "similarity_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {plot_path}")
    
    # 保存详细结果
    results_path = os.path.join(sequence_dir, "similarity_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"MineCLIP 16帧视频模式验证结果\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"任务描述: {task_prompt}\n")
        f.write(f"窗口大小: {num_frames}帧\n")
        f.write(f"滑动步长: {stride}帧\n")
        f.write(f"总帧数: {len(frames)}\n")
        f.write(f"分析窗口数: {len(similarities)}\n\n")
        f.write(f"统计结果:\n")
        f.write(f"  最小值: {similarities_np.min():.4f}\n")
        f.write(f"  最大值: {similarities_np.max():.4f}\n")
        f.write(f"  平均值: {similarities_np.mean():.4f}\n")
        f.write(f"  标准差: {similarities_np.std():.4f}\n")
        f.write(f"  变化范围: {similarities_np.max() - similarities_np.min():.4f}\n\n")
        f.write(f"详细结果:\n")
        for i, (idx, sim) in enumerate(zip(window_indices, similarities)):
            f.write(f"窗口 {i+1:3d} [帧{idx:04d}-{idx+num_frames-1:04d}]: {sim:.4f}\n")
    
    print(f"  ✓ 详细结果已保存: {results_path}")
    
    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)
    
    # 评估结论
    print(f"\n🎯 评估结论:")
    range_val = similarities_np.max() - similarities_np.min()
    
    if range_val > 0.05:
        print(f"  ✅ 16帧视频模式效果良好！变化范围达到 {range_val:.4f}")
        print(f"  → 建议使用16帧视频模式进行训练")
    elif range_val > 0.02:
        print(f"  ⚠️  16帧视频模式有一定效果，变化范围为 {range_val:.4f}")
        print(f"  → 可以尝试，但可能需要调整提示词")
    else:
        print(f"  ❌ 16帧视频模式区分度较低，变化范围仅 {range_val:.4f}")
        print(f"  → 考虑其他方案（任务分解、奖励塑形等）")
    
    print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="验证MineCLIP 16帧视频模式")
    parser.add_argument('--sequence-dir', type=str, default='logs/chopping_sequence',
                       help='帧序列目录')
    parser.add_argument('--num-frames', type=int, default=16,
                       help='每个视频片段的帧数')
    parser.add_argument('--stride', type=int, default=4,
                       help='滑动窗口步长')
    parser.add_argument('--task-prompt', type=str, default='chopping a tree with hand',
                       help='任务描述')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sequence_dir):
        print(f"❌ 目录不存在: {args.sequence_dir}")
        print(f"\n请先录制帧序列：")
        print(f"  python record_chopping_sequence.py --output-dir {args.sequence_dir}")
        return
    
    analyze_sequence(args.sequence_dir, args.num_frames, args.stride, args.task_prompt)

if __name__ == "__main__":
    main()

