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
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import minedojo
from stable_baselines3 import PPO
from src.envs import make_minedojo_env


def create_and_reset_env_with_retry(task_id, max_steps, max_retries=3):
    """
    创建环境并重置，支持重试
    
    Args:
        task_id: MineDojo任务ID
        max_steps: 每个episode最大步数
        max_retries: 最大重试次数
    
    Returns:
        tuple: (env, obs) 如果成功，否则 (None, None)
    """
    for attempt in range(max_retries):
        try:
            env = make_minedojo_env(
                task_id=task_id,
                max_episode_steps=max_steps,
                fast_reset=False
            )
            obs = env.reset()
            return env, obs
        except (EOFError, RuntimeError, Exception) as e:
            error_type = type(e).__name__
            print(f"\n  ✗ 环境启动失败 (尝试 {attempt+1}/{max_retries}): {error_type}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                print(f"  ⏳ 等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ 达到最大重试次数，跳过此episode")
                return None, None
    
    return None, None


def evaluate_policy(
    model_path,
    num_episodes,
    task_id="harvest_1_log",
    max_steps=1000,
    deterministic=True,
    device="auto"
):
    """
    评估策略
    
    Args:
        model_path: 策略模型路径
        num_episodes: 评估的episode数量
        task_id: MineDojo任务ID
        max_steps: 每个episode最大步数
        deterministic: 是否使用确定性策略
        device: 运行设备 (auto/cpu/cuda/mps)
    
    Returns:
        dict: 评估结果统计
    """
    
    # 设备检测
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"\n{'='*60}")
    print(f"策略评估")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"任务: {task_id}")
    print(f"Episode数: {num_episodes}")
    print(f"确定性: {deterministic}")
    print(f"设备: {device}")
    print(f"环境管理: 每个episode重新创建环境（避免内存泄漏）")
    print(f"{'='*60}\n")
    
    # 加载策略
    try:
        policy = PPO.load(model_path, device=device)
        print(f"✓ 策略加载成功")
        print(f"  策略类型: {type(policy)}")
        print(f"  设备: {policy.device}")
        print(f"  动作空间: {policy.action_space}")
        print(f"  观察空间: {policy.observation_space}")
        print()
    except Exception as e:
        print(f"✗ 策略加载失败: {e}")
        return None
    
    # 统计信息
    episode_rewards = []
    episode_lengths = []
    successes = []
    skipped_count = 0  # 环境启动失败跳过的episode数
    
    print(f"开始评估...\n")
    
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{num_episodes}")
        print(f"{'='*60}")
        
        # 每个回合创建新环境（带重试）
        print(f"  创建新环境...")
        env, obs = create_and_reset_env_with_retry(task_id, max_steps)
        
        # 如果环境创建失败，跳过此episode
        if env is None:
            skipped_count += 1
            print(f"  ⚠️  Episode {ep+1} 已跳过\n")
            continue
        
        print(f"  ✓ 环境已创建并重置")
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        # 动作统计
        action_counts = {
            'idle': 0,
            'forward': 0,
            'back': 0,
            'left': 0,
            'right': 0,
            'jump': 0,
            'attack': 0,
            'camera_move': 0
        }
        
        while not done and episode_length < max_steps:
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            # 统计动作
            is_idle = (action[0] == 0 and action[1] == 0 and action[2] == 0 and 
                      action[3] == 12 and action[4] == 12 and action[5] == 0)
            
            if is_idle:
                action_counts['idle'] += 1
            else:
                if action[0] == 1:
                    action_counts['forward'] += 1
                elif action[0] == 2:
                    action_counts['back'] += 1
                
                if action[1] == 1:
                    action_counts['left'] += 1
                elif action[1] == 2:
                    action_counts['right'] += 1
                
                if action[2] == 1:
                    action_counts['jump'] += 1
                
                if action[5] == 3:
                    action_counts['attack'] += 1
                
                if action[3] != 12 or action[4] != 12:
                    action_counts['camera_move'] += 1
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if episode_length % 100 == 0:
                print(".", end="", flush=True)
        
        # 判断成功
        success = episode_reward > 0.5 or info.get('success', False)
        successes.append(success)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        status = "✓" if success else "✗"
        
        # 打印动作统计
        total_actions = sum(action_counts.values())
        print(f"\n  动作统计 (共{episode_length}步):")
        print(f"    IDLE: {action_counts['idle']:4d} ({action_counts['idle']/episode_length*100:5.1f}%)")
        print(f"    前进: {action_counts['forward']:4d} ({action_counts['forward']/episode_length*100:5.1f}%)")
        print(f"    后退: {action_counts['back']:4d} ({action_counts['back']/episode_length*100:5.1f}%)")
        print(f"    攻击: {action_counts['attack']:4d} ({action_counts['attack']/episode_length*100:5.1f}%)")
        print(f"    跳跃: {action_counts['jump']:4d} ({action_counts['jump']/episode_length*100:5.1f}%)")
        print(f"    镜头: {action_counts['camera_move']:4d} ({action_counts['camera_move']/episode_length*100:5.1f}%)")
        
        print(f"\n {status} | 步数:{episode_length:3d} | 奖励:{episode_reward:6.2f}")
        
        # 关闭环境
        print(f"  关闭环境...")
        env.close()
        print(f"  ✓ 环境已关闭")
    
    # 计算统计
    completed_episodes = num_episodes - skipped_count
    
    if completed_episodes > 0:
        success_rate = np.mean(successes)
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        std_reward = np.std(episode_rewards)
        std_length = np.std(episode_lengths)
    else:
        success_rate = 0.0
        avg_reward = 0.0
        avg_length = 0.0
        std_reward = 0.0
        std_length = 0.0
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"总episode数: {num_episodes} | 完成: {completed_episodes} | 跳过: {skipped_count}")
    if completed_episodes > 0:
        print(f"成功率: {success_rate*100:.1f}% ({np.sum(successes)}/{completed_episodes})")
        print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"平均步数: {avg_length:.0f} ± {std_length:.0f}")
    else:
        print(f"⚠️  没有成功完成的episode")
    print(f"{'='*60}\n")
    
    results = {
        'num_episodes': num_episodes,
        'completed_episodes': completed_episodes,
        'skipped_count': skipped_count,
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
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="运行设备 (auto/cpu/cuda/mps，默认: auto)"
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
        deterministic=not args.stochastic,
        device=args.device
    )
    
    if results:
        # 保存结果（可选）
        model_dir = os.path.dirname(args.model)
        model_name = os.path.basename(args.model).replace('.zip', '')
        results_file = os.path.join(model_dir, f"{model_name}_eval_results.npy")
        
        np.save(results_file, results)
        print(f"✓ 评估结果已保存: {results_file}\n")
        print(f"下一步：\npython tools/run_policy_collect_states.py --model checkpoints/bc_baseline.zip --episodes 20 --output data/policy_states/iter_1/ --save-failures-only --task-id harvest_1_log")


if __name__ == "__main__":
    main()

