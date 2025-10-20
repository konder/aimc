#!/usr/bin/env python3
"""
测试MineCLIP对不同Minecraft场景的相似度评分

使用方法：
1. 运行脚本会创建MineDojo环境
2. 手动控制agent移动（如果可能）或随机探索
3. 记录不同场景下的相似度

这个脚本用于验证MineCLIP是否能区分：
- 空旷草地 vs 有树的场景
- 远处的树 vs 近处的树
- 砍树动作 vs 静止观看
"""

import minedojo
import torch
import numpy as np
from utils.mineclip_reward import MineCLIPRewardWrapper

def test_mineclip_sensitivity():
    """测试MineCLIP相似度对不同场景的敏感度"""
    
    print("=" * 80)
    print("MineCLIP 敏感度测试")
    print("=" * 80)
    print("\n创建环境...")
    
    # 创建环境
    env = minedojo.make(
        task_id="harvest_1_log",
        image_size=(160, 256)
    )
    
    # 包装MineCLIP
    env = MineCLIPRewardWrapper(
        env,
        task_prompt="chop down a tree and collect one wood log",
        model_path="data/mineclip/attn.pth",
        variant="attn",
        device="cpu",
        sparse_weight=10.0,
        mineclip_weight=10.0
    )
    
    print("\n开始测试...\n")
    print(f"{'步数':>6s} | {'动作描述':<30s} | {'相似度':>8s} | {'MineCLIP奖励':>12s}")
    print("-" * 70)
    
    obs = env.reset()
    
    # 测试场景
    actions_to_test = [
        ("原地不动", [0] * 2),        # noop
        ("向前走", [1, 0]),            # forward
        ("向左转", [0, 1]),            # turn left
        ("向右转", [0, 2]),            # turn right
        ("向后走", [2, 0]),            # backward
        ("跳跃", [4, 0]),              # jump
        ("向前走+跳跃", [1, 4]),       # forward + jump
    ]
    
    similarities = []
    rewards = []
    
    for step in range(100):
        # 使用预定义动作或随机动作
        if step < len(actions_to_test):
            action_name, action = actions_to_test[step]
            action = env.action_space.sample()  # 暂时用随机
        else:
            action_name = "随机动作"
            action = env.action_space.sample()
        
        obs, reward, done, info = env.step(action)
        
        similarity = info.get('mineclip_similarity', 0.0)
        mineclip_reward = info.get('mineclip_reward', 0.0)
        
        similarities.append(similarity)
        rewards.append(mineclip_reward)
        
        # 每10步打印一次
        if step % 10 == 0:
            print(f"{step:>6d} | {action_name:<30s} | {similarity:>8.6f} | {mineclip_reward:>+12.6f}")
        
        if done:
            obs = env.reset()
    
    env.close()
    
    # 统计
    similarities = np.array(similarities)
    rewards = np.array(rewards)
    
    print("\n" + "=" * 80)
    print("统计结果:")
    print("=" * 80)
    print(f"相似度范围: {similarities.min():.6f} ~ {similarities.max():.6f}")
    print(f"相似度均值: {similarities.mean():.6f} ± {similarities.std():.6f}")
    print(f"最大波动:   {similarities.max() - similarities.min():.6f}")
    print(f"\n奖励范围:   {rewards.min():+.6f} ~ {rewards.max():+.6f}")
    print(f"奖励均值:   {rewards.mean():+.6f} ± {rewards.std():.6f}")
    
    print("\n" + "=" * 80)
    print("💡 解读:")
    print("=" * 80)
    if similarities.max() - similarities.min() < 0.02:
        print("⚠️  相似度波动 < 2%")
        print("   → agent可能一直在看普通场景（草地、天空）")
        print("   → 建议：训练更长时间，或提高MineCLIP权重")
    elif similarities.max() > 0.70:
        print("✅ 检测到高相似度 (> 0.70)")
        print("   → agent可能看到了树木！")
    else:
        print("ℹ️  相似度在正常范围")
        print("   → MineCLIP正常工作，但需要更多探索")

if __name__ == "__main__":
    import os
    os.environ['JAVA_OPTS'] = '-Djava.awt.headless=true'
    test_mineclip_sensitivity()

