#!/usr/bin/env python3
"""
分析MineCLIP相似度和奖励的分布
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log_line(line):
    """解析实时日志行"""
    # 匹配格式: 0 | 100 | ... | 0.0179 | 0.0018 | 9.9999 | 1.00 | 0.6337 | N/A
    pattern = r'\s+\d+\s+\|\s+(\d+,?\d*)\s+\|.*?\|\s+([-\d.]+)\s+\|\s+([-\d.]+)\s+\|.*?\|\s+([\d.]+)\s+\|'
    match = re.search(pattern, line)
    if match:
        step = int(match.group(1).replace(',', ''))
        total_reward = float(match.group(2))
        mineclip_reward = float(match.group(3))
        similarity = float(match.group(4))
        return step, total_reward, mineclip_reward, similarity
    return None

def analyze_similarity_distribution():
    """分析相似度分布"""
    print("=" * 80)
    print("MineCLIP 相似度分析")
    print("=" * 80)
    
    # 从用户提供的日志中手动提取数据（实际应从文件读取）
    print("\n请粘贴训练日志（包含相似度列），以空行结束：")
    print("（或者直接按回车使用示例数据）\n")
    
    lines = []
    try:
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
    except EOFError:
        pass
    
    if not lines:
        # 使用示例数据
        print("使用示例数据分析...\n")
        similarities = [
            0.6337, 0.6314, 0.6314, 0.6315, 0.6322, 0.6298, 0.6316, 0.6343,
            0.6327, 0.6314, 0.6346, 0.6339, 0.6347, 0.6312, 0.6329, 0.6333,
            0.6303, 0.6319, 0.6337, 0.6311, 0.6317, 0.6320, 0.6319, 0.6316,
            0.6315, 0.6315, 0.6342, 0.6308, 0.6360, 0.6336, 0.6315, 0.6324,
            0.6369, 0.6316, 0.6341, 0.6314, 0.6306, 0.6307, 0.6305, 0.6307
        ]
        mineclip_rewards = [
            0.0018, -0.0004, -0.0009, -0.0023, -0.0019, -0.0002, 0.0002, 0.0010,
            -0.0020, -0.0014, 0.0004, -0.0001, -0.0006, 0.0014, 0.0013, -0.0012,
            0.0003, -0.0000, 0.0019, 0.0010, 0.0009, 0.0013, 0.0004, 0.0002,
            0.0002, 0.0011, -0.0005, -0.0021, -0.0000, 0.0021, -0.0024, -0.0029,
            0.0001, -0.0004, -0.0032, 0.0002, -0.0021, -0.0005, -0.0012, -0.0007
        ]
    else:
        # 解析实际日志
        data = [parse_log_line(line) for line in lines]
        data = [d for d in data if d is not None]
        if not data:
            print("无法解析日志数据")
            return
        
        steps, total_rewards, mineclip_rewards, similarities = zip(*data)
    
    similarities = np.array(similarities)
    mineclip_rewards = np.array(mineclip_rewards)
    
    # 统计分析
    print(f"相似度统计:")
    print(f"  均值:    {similarities.mean():.6f}")
    print(f"  标准差:  {similarities.std():.6f}")
    print(f"  最小值:  {similarities.min():.6f}")
    print(f"  最大值:  {similarities.max():.6f}")
    print(f"  范围:    {similarities.max() - similarities.min():.6f}")
    print(f"  变异系数: {similarities.std()/similarities.mean()*100:.2f}%")
    
    print(f"\nMineCLIP奖励统计:")
    print(f"  均值:    {mineclip_rewards.mean():.6f}")
    print(f"  标准差:  {mineclip_rewards.std():.6f}")
    print(f"  最大正:  {mineclip_rewards.max():.6f}")
    print(f"  最大负:  {mineclip_rewards.min():.6f}")
    
    # 找出峰值
    threshold_high = similarities.mean() + similarities.std()
    threshold_low = similarities.mean() - similarities.std()
    
    print(f"\n高相似度阈值 (mean + 1σ): {threshold_high:.6f}")
    high_indices = np.where(similarities > threshold_high)[0]
    if len(high_indices) > 0:
        print(f"高相似度事件 ({len(high_indices)}次):")
        for idx in high_indices[:10]:  # 只显示前10个
            print(f"  步数 {(idx+1)*100}: 相似度={similarities[idx]:.6f}, 奖励={mineclip_rewards[idx]:+.6f}")
    
    print(f"\n低相似度阈值 (mean - 1σ): {threshold_low:.6f}")
    low_indices = np.where(similarities < threshold_low)[0]
    if len(low_indices) > 0:
        print(f"低相似度事件 ({len(low_indices)}次):")
        for idx in low_indices[:10]:
            print(f"  步数 {(idx+1)*100}: 相似度={similarities[idx]:.6f}, 奖励={mineclip_rewards[idx]:+.6f}")
    
    # 相关性分析
    correlation = np.corrcoef(similarities[:-1], mineclip_rewards[1:])[0, 1]
    print(f"\n相似度变化 vs MineCLIP奖励相关性: {correlation:.3f}")
    print("（MineCLIP奖励 = 当前相似度 - 上一步相似度）")
    
    print("\n" + "=" * 80)
    print("💡 解释:")
    print("=" * 80)
    print("相似度变化小 (< 2%) 通常意味着:")
    print("  1. agent主要看到的是普通场景（草地、天空）")
    print("  2. 还没有看到任务关键物体（树木）")
    print("  3. MineCLIP正确工作，但缺乏有意义的视觉刺激")
    print("\n建议:")
    print("  - 训练更长时间，让agent探索到树木")
    print("  - 使用更大的MineCLIP权重引导探索")
    print("  - 观察高相似度事件，那可能是agent短暂看到了树")

if __name__ == "__main__":
    analyze_similarity_distribution()

