#!/usr/bin/env python3
"""
评估VPT BC微调后的模型

在MineDojo环境中运行多个episodes，统计成功率
"""

import os
import sys
import argparse
import numpy as np
import torch as th
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.vpt_policy_standalone import VPTPolicy, VPTMinedojoAdapter
import minedojo


def evaluate_policy(
    policy: VPTPolicy,
    task_id: str = "harvest_1_log",
    num_episodes: int = 20,
    max_steps: int = 1000,
    image_size: tuple = (160, 256),
    device: str = "cpu",
    render: bool = False
):
    """
    评估policy性能
    
    Args:
        policy: VPT policy模型
        task_id: 任务ID
        num_episodes: 评估episodes数
        max_steps: 每个episode最大步数
        image_size: 图像尺寸
        device: 设备
        render: 是否渲染（慎用，可能崩溃）
    
    Returns:
        results: 评估结果
    """
    # 创建adapter
    adapter = VPTMinedojoAdapter(
        model_path="dummy.model",
        weights_path="dummy.weights",  # 权重已在policy中
        device=device
    )
    adapter.policy = policy  # 使用传入的policy
    
    # 创建环境
    print(f"创建环境: {task_id}")
    print(f"  图像尺寸: {image_size}")
    print(f"  最大步数: {max_steps}")
    
    env = minedojo.make(
        task_id=task_id,
        image_size=image_size
    )
    
    # 评估
    results = {
        'episodes': [],
        'success_count': 0,
        'total_episodes': num_episodes
    }
    
    print(f"\n开始评估 ({num_episodes} episodes)...")
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        for step in range(max_steps):
            # 预测动作
            action = adapter.predict(obs, deterministic=True)
            
            # 执行
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # 判断成功（根据任务不同，这里简化为reward > 0）
        success = episode_reward > 0
        
        if success:
            results['success_count'] += 1
        
        results['episodes'].append({
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'success': success
        })
        
        # 实时显示
        if (episode + 1) % 5 == 0:
            current_success_rate = results['success_count'] / (episode + 1)
            print(f"  Progress: {episode+1}/{num_episodes}, "
                  f"Success Rate: {current_success_rate:.2%}")
    
    env.close()
    
    # 计算统计
    results['success_rate'] = results['success_count'] / num_episodes
    results['avg_reward'] = np.mean([ep['reward'] for ep in results['episodes']])
    results['avg_steps'] = np.mean([ep['steps'] for ep in results['episodes']])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="评估VPT BC模型")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="评估设备")
    
    # 评估参数
    parser.add_argument("--task", type=str, default="harvest_1_log",
                       help="任务ID")
    parser.add_argument("--num-episodes", type=int, default=20,
                       help="评估episodes数")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="每个episode最大步数")
    parser.add_argument("--image-size", type=int, nargs=2, default=[160, 256],
                       help="图像尺寸 (H W)")
    
    # 输出参数
    parser.add_argument("--output", type=str, default=None,
                       help="结果保存路径（.yaml）")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VPT BC模型评估")
    print("=" * 70)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    checkpoint = th.load(args.model, map_location=args.device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  Checkpoint epoch: {epoch}")
    else:
        state_dict = checkpoint
    
    # 创建policy并加载权重
    policy = VPTPolicy(obs_shape=(128, 128, 3))
    policy.load_state_dict(state_dict)
    policy.to(args.device)
    policy.eval()
    
    print(f"✓ 模型加载成功")
    print(f"  设备: {args.device}")
    print(f"  参数量: {sum(p.numel() for p in policy.parameters()):,}")
    
    # 评估
    image_size = tuple(args.image_size)
    results = evaluate_policy(
        policy=policy,
        task_id=args.task,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        image_size=image_size,
        device=args.device
    )
    
    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"任务: {args.task}")
    print(f"Episodes: {results['total_episodes']}")
    print(f"成功: {results['success_count']}")
    print(f"成功率: {results['success_rate']:.2%}")
    print(f"平均奖励: {results['avg_reward']:.4f}")
    print(f"平均步数: {results['avg_steps']:.1f}")
    
    # 成功的episodes
    successful_episodes = [ep for ep in results['episodes'] if ep['success']]
    if successful_episodes:
        print(f"\n成功episodes平均步数: {np.mean([ep['steps'] for ep in successful_episodes]):.1f}")
    
    print("=" * 70)
    
    # 保存结果
    if args.output:
        import yaml
        with open(args.output, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"\n结果保存到: {args.output}")
    
    # 对比基线
    print("\n对比:")
    print(f"  纯BC基线: <1% 成功率")
    print(f"  VPT微调: {results['success_rate']:.2%} 成功率")
    if results['success_rate'] > 0.01:
        improvement = results['success_rate'] / 0.01
        print(f"  提升: {improvement:.1f}x")


if __name__ == "__main__":
    main()

