#!/usr/bin/env python3
"""
简单分析微调前后的MineCLIP结果
"""
import numpy as np

# 加载数据
baseline_file = "tmp/mineclip_analysis_20_data.npy"
finetuned_file = "tmp/finetuned.npy"

baseline = np.load(baseline_file, allow_pickle=True)
finetuned = np.load(finetuned_file, allow_pickle=True)

print("="*70)
print("MineCLIP 基线 vs 微调 对比分析")
print("="*70)
print()

print(f"基线数据结构: {type(baseline)}, shape: {baseline.shape}")
print(f"微调数据结构: {type(finetuned)}, shape: {finetuned.shape}")
print()

# 检查是否是episode数组
if len(baseline) > 0:
    print(f"基线[0]类型: {type(baseline[0])}")
    if isinstance(baseline[0], dict):
        print(f"基线[0]keys: {baseline[0].keys()}")
print()

if len(finetuned) > 0:
    print(f"微调[0]类型: {type(finetuned[0])}")
    if isinstance(finetuned[0], dict):
        print(f"微调[0]keys: {finetuned[0].keys()}")
print()

# 提取相似度数据
print("="*70)
print("相似度统计")
print("="*70)
print()

baseline_sims = []
for ep in baseline:
    if isinstance(ep, dict) and 'similarities' in ep:
        baseline_sims.extend(ep['similarities'])
    elif isinstance(ep, (list, np.ndarray)):
        baseline_sims.extend(ep)

finetuned_sims = []
for ep in finetuned:
    if isinstance(ep, dict) and 'similarities' in ep:
        finetuned_sims.extend(ep['similarities'])
    elif isinstance(ep, (list, np.ndarray)):
        finetuned_sims.extend(ep)

if baseline_sims:
    baseline_sims = np.array(baseline_sims)
    print(f"基线MineCLIP:")
    print(f"  数据点: {len(baseline_sims)}")
    print(f"  平均相似度: {baseline_sims.mean():.4f}")
    print(f"  标准差: {baseline_sims.std():.4f}")
    print(f"  最小值: {baseline_sims.min():.4f}")
    print(f"  最大值: {baseline_sims.max():.4f}")
    print()
    
    # 前后对比
    mid = len(baseline_sims) // 2
    first_half = baseline_sims[:mid].mean()
    second_half = baseline_sims[mid:].mean()
    baseline_correlation = second_half - first_half
    print(f"  时序分析:")
    print(f"    前半段: {first_half:.4f}")
    print(f"    后半段: {second_half:.4f}")
    print(f"    进度相关性: {baseline_correlation:+.4f}")
    print()

if finetuned_sims:
    finetuned_sims = np.array(finetuned_sims)
    print(f"微调后MineCLIP:")
    print(f"  数据点: {len(finetuned_sims)}")
    print(f"  平均相似度: {finetuned_sims.mean():.4f}")
    print(f"  标准差: {finetuned_sims.std():.4f}")
    print(f"  最小值: {finetuned_sims.min():.4f}")
    print(f"  最大值: {finetuned_sims.max():.4f}")
    print()
    
    # 前后对比
    mid = len(finetuned_sims) // 2
    first_half = finetuned_sims[:mid].mean()
    second_half = finetuned_sims[mid:].mean()
    finetuned_correlation = second_half - first_half
    print(f"  时序分析:")
    print(f"    前半段: {first_half:.4f}")
    print(f"    后半段: {second_half:.4f}")
    print(f"    进度相关性: {finetuned_correlation:+.4f}")
    print()

if len(baseline_sims) > 0 and len(finetuned_sims) > 0:
    print("="*70)
    print("对比总结")
    print("="*70)
    print()
    
    mean_change = finetuned_sims.mean() - baseline_sims.mean()
    correlation_improvement = finetuned_correlation - baseline_correlation
    
    print(f"平均相似度变化: {baseline_sims.mean():.4f} → {finetuned_sims.mean():.4f} ({mean_change:+.4f})")
    print(f"进度相关性改善: {baseline_correlation:+.4f} → {finetuned_correlation:+.4f} ({correlation_improvement:+.4f})")
    print()
    
    # 评估
    print("="*70)
    print("最终评估")
    print("="*70)
    print()
    
    if abs(correlation_improvement) < 0.05:
        print("❌ 微调效果不明显")
        print(f"   进度相关性改善仅 {correlation_improvement:+.4f} < 0.05")
    elif abs(correlation_improvement) < 0.15:
        print("⚠️  微调效果一般")
        print(f"   进度相关性改善 {correlation_improvement:+.4f}")
    else:
        print("✅ 微调效果显著！")
        print(f"   进度相关性改善 {correlation_improvement:+.4f} > 0.15")
    
    print()
    print("查看可视化:")
    print(f"  基线: open tmp/mineclip_analysis_20.png")
    print(f"  微调: open tmp/finetuned.png")
    print()

