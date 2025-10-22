#!/usr/bin/env python3
"""
策略评估工具

评估训练好的策略的成功率和性能指标

Usage:
    python tools/evaluate_policy.py \
        --model checkpoints/bc_round_0.zip \
        --episodes 20 \
        --task-id harvest_1_log
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import minedojo
from stable_baselines3 import PPO
from src.utils.env_wrappers import make_minedojo_env


def evaluate_policy(
    model_path,
    num_episodes,
    task_id="harvest_1_log",
    max_steps=1000,
    deterministic=True
):
    """
    评估策略
    
    Args:
        model_path: 策略模型路径
        num_episodes: 评估的episode数量
        task_id: MineDojo任务ID
        max_steps: 每个episode最大步数
        deterministic: 是否使用确定性策略
    
    Returns:
        dict: 评估结果统计
    """
    
    print(f"\n{'='*60}")
    print(f"策略评估")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"任务: {task_id}")
    print(f"Episode数: {num_episodes}")
    print(f"确定性: {deterministic}")
    print(f"{'='*60}\n")
    
    # 加载策略
    try:
        policy = PPO.load(model_path)
        print(f"✓ 策略加载成功\n")
    except Exception as e:
        print(f"✗ 策略加载失败: {e}")
        return None
    
    # 创建环境
    # fast_reset=False: 每个episode重新生成世界，确保多样性
    env = make_minedojo_env(
        task_id=task_id,
        use_camera_smoothing=False,
        max_episode_steps=max_steps,
        fast_reset=False
    )
    
    # 统计信息
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    print(f"开始评估...\n")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        print(f"Episode {ep+1}/{num_episodes} ", end="", flush=True)
        
        while not done and episode_length < max_steps:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if episode_length % 50 == 0:
                print(".", end="", flush=True)
        
        # 判断成功
        success = episode_reward > 0.5 or info.get('success', False)
        successes.append(success)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        status = "✓" if success else "✗"
        print(f" {status} | 步数:{episode_length:3d} | 奖励:{episode_reward:6.2f}")
    
    env.close()
    
    # 计算统计
    success_rate = np.mean(successes)
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_reward = np.std(episode_rewards)
    std_length = np.std(episode_lengths)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"成功率: {success_rate*100:.1f}% ({np.sum(successes)}/{num_episodes})")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"平均步数: {avg_length:.0f} ± {std_length:.0f}")
    print(f"{'='*60}\n")
    
    results = {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'std_length': std_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'successes': successes
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="策略评估工具 - 评估训练好的策略"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="策略模型路径（.zip文件）"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="评估的episode数量（默认: 20）"
    )
    
    parser.add_argument(
        "--task-id",
        type=str,
        default="harvest_1_log",
        help="MineDojo任务ID（默认: harvest_1_log）"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="每个episode最大步数（默认: 1000）"
    )
    
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机策略而非确定性策略"
    )
    
    args = parser.parse_args()
    
    # 验证模型文件存在
    if not os.path.exists(args.model):
        print(f"✗ 错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 评估策略
    results = evaluate_policy(
        model_path=args.model,
        num_episodes=args.episodes,
        task_id=args.task_id,
        max_steps=args.max_steps,
        deterministic=not args.stochastic
    )
    
    if results:
        # 保存结果（可选）
        model_dir = os.path.dirname(args.model)
        model_name = os.path.basename(args.model).replace('.zip', '')
        results_file = os.path.join(model_dir, f"{model_name}_eval_results.npy")
        
        np.save(results_file, results)
        print(f"✓ 评估结果已保存: {results_file}\n")


if __name__ == "__main__":
    main()

