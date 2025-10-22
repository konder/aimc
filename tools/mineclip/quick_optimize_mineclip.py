#!/usr/bin/env python3
"""
MineCLIP快速配置优化工具 - 只测试不同的text prompts
基于手动录制的砍树序列
"""

import os
import sys
import glob
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('/Users/nanzhang/aimc')

# Minecraft特定的归一化参数
MC_IMAGE_MEAN = torch.tensor([0.3331, 0.3245, 0.3051])
MC_IMAGE_STD = torch.tensor([0.2439, 0.2493, 0.2873])

def load_mineclip_model(device='mps'):
    """加载MineCLIP模型"""
    try:
        from mineclip import MineCLIP
        
        model = MineCLIP(
            arch="vit_base_p16_fz.v2.t2",
            pool_type="attn.d2.nh8.glusw",
            resolution=(160, 256),
            image_feature_dim=512,
            mlp_adapter_spec="v0-2.t0",
            hidden_dim=512
        ).to(device)
        
        # 加载权重
        weight_path = "/Users/nanzhang/aimc/data/mineclip/attn.pth"
        checkpoint = torch.load(weight_path, map_location=device)
        
        # 提取state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理键名
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[6:] if key.startswith('model.') else key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        print("✓ MineCLIP模型加载成功")
        return model
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_frames(sequence_dir):
    """加载帧序列"""
    frame_files = sorted(glob.glob(os.path.join(sequence_dir, "frame_*.png")))
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        frames.append(np.array(img))
    print(f"✓ 已加载 {len(frames)} 帧")
    return frames

def preprocess_frame(frame):
    """预处理单帧"""
    frame_resized = cv2.resize(frame, (256, 160))  # (W, H)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
    
    # MineCLIP归一化
    mean = MC_IMAGE_MEAN.view(3, 1, 1)
    std = MC_IMAGE_STD.view(3, 1, 1)
    frame_tensor = (frame_tensor - mean) / std
    
    return frame_tensor

def compute_similarity_for_prompt(model, tokenizer, frames, prompt, device='mps'):
    """计算单个prompt的相似度序列"""
    print(f"\n测试prompt: '{prompt}'")
    
    # 编码文本
    with torch.no_grad():
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )
        text_tokens = tokens['input_ids'].to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 处理每16帧
    similarities = []
    num_frames = 16
    stride = 1
    
    for i in tqdm(range(0, len(frames) - num_frames + 1, stride), desc="计算相似度"):
        video_frames = frames[i:i+num_frames]
        
        # 预处理帧
        processed_frames = [preprocess_frame(f) for f in video_frames]
        video_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)
        
        # 编码视频
        with torch.no_grad():
            video_features = model.encode_video(video_tensor)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (video_features @ text_features.T).squeeze().item()
            similarities.append(similarity)
    
    similarities = np.array(similarities)
    
    # 统计
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    range_sim = np.max(similarities) - np.min(similarities)
    
    print(f"  平均: {mean_sim:.4f}")
    print(f"  标准差: {std_sim:.4f}")
    print(f"  范围: {range_sim:.4f} ({range_sim/mean_sim*100:.2f}%)")
    
    return {
        'prompt': prompt,
        'similarities': similarities,
        'mean': mean_sim,
        'std': std_sim,
        'range': range_sim,
        'range_percent': range_sim / mean_sim * 100
    }

def main():
    print("\n" + "="*80)
    print("MineCLIP快速配置优化 - Prompt测试")
    print("="*80 + "\n")
    
    # 加载模型
    device = 'mps'
    model = load_mineclip_model(device)
    if model is None:
        return
    
    # 加载tokenizer
    from transformers import CLIPTokenizer
    tokenizer_path = "/Users/nanzhang/aimc/data/clip_tokenizer"
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    print("✓ Tokenizer加载成功")
    
    # 加载帧
    frames = load_frames("/Users/nanzhang/aimc/logs/my_chopping")
    
    # 测试不同的prompts
    test_prompts = [
        # 当前使用的
        "chopping a tree with hand",
        
        # 简单物体描述
        "a tree",
        "tree",
        "oak tree",
        "tree trunk",
        "wood blocks",
        
        # 视觉位置描述
        "looking at tree",
        "facing a tree",
        "tree in front",
        设计
        # 动作描述  
        "breaking tree",
        "punching tree",
        "mining wood",
        "cutting wood",
        
        # 场景描述
        "forest",
        "trees in minecraft",
    ]
    
    # 测试所有prompts
    results = []
    for prompt in test_prompts:
        try:
            result = compute_similarity_for_prompt(model, tokenizer, frames, prompt, device)
            results.append(result)
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    
    # 排序（按变化范围百分比）
    results.sort(key=lambda x: x['range_percent'], reverse=True)
    
    # 输出结果
    print("\n" + "="*80)
    print("测试结果汇总（按相似度变化范围排序）")
    print("="*80 + "\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {result['prompt']:30s} | "
              f"平均={result['mean']:.4f} | "
              f"范围={result['range']:.4f} ({result['range_percent']:.2f}%)")
    
    # 绘制对比图
    output_dir = "/Users/nanzhang/aimc/logs/mineclip_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Top 10 相似度曲线
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 变化范围对比（柱状图）
    prompts = [r['prompt'] for r in results]
    ranges = [r['range_percent'] for r in results]
    
    axes[0].barh(prompts, ranges, color='steelblue')
    axes[0].set_xlabel('相似度变化范围 (%)')
    axes[0].set_title('不同Prompt的MineCLIP相似度变化范围对比')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    
    # 2. Top 5 相似度曲线
    for i, result in enumerate(results[:5]):
        axes[1].plot(result['similarities'], 
                    label=f"{result['prompt'][:20]} ({result['range_percent']:.2f}%)",
                    linewidth=2)
    
    axes[1].set_xlabel('帧索引')
    axes[1].set_ylabel('MineCLIP相似度')
    axes[1].set_title('Top 5 Prompt的相似度变化曲线')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_optimization.png'), dpi=150)
    print(f"\n✓ 对比图已保存: {output_dir}/prompt_optimization.png")
    
    # 保存详细结果
    with open(os.path.join(output_dir, 'prompt_results.txt'), 'w', encoding='utf-8') as f:
        f.write("MineCLIP Prompt优化测试结果\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Prompt #{i}: {result['prompt']}\n")
            f.write(f"  平均相似度: {result['mean']:.4f}\n")
            f.write(f"  标准差: {result['std']:.4f}\n")
            f.write(f"  变化范围: {result['range']:.4f} ({result['range_percent']:.2f}%)\n")
            f.write("-"*80 + "\n\n")
    
    print(f"✓ 详细结果已保存: {output_dir}/prompt_results.txt")
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("="*80)

if __name__ == "__main__":
    main()

