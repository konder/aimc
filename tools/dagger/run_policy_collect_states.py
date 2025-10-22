#!/usr/bin/env python3
"""
策略状态收集工具 - DAgger第一步

运行训练好的策略并收集访问的状态，用于后续人工标注。
重点收集失败的episode，这些是需要专家纠正的场景。

Usage:
    python tools/run_policy_collect_states.py \
        --model checkpoints/bc_round_0.zip \
        --episodes 20 \
        --output data/policy_states/iter_1/ \
        --save-failures-only
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


def collect_policy_states(
    model_path,
    num_episodes,
    output_dir,
    task_id="harvest_1_log",
    save_failures_only=False,
    max_steps=1000,
    deterministic=False
):
    """
    运行策略并收集访问的状态
    
    Args:
        model_path: 策略模型路径
        num_episodes: 收集的episode数量
        output_dir: 输出目录
        task_id: MineDojo任务ID
        save_failures_only: 是否只保存失败的episode
        max_steps: 每个episode最大步数
        deterministic: 是否使用确定性策略
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载策略
    print(f"\n{'='*60}")
    print(f"DAgger 状态收集工具")
    print(f"{'='*60}")
    print(f"策略模型: {model_path}")
    print(f"任务: {task_id}")
    print(f"收集episode数: {num_episodes}")
    print(f"只保存失败: {save_failures_only}")
    print(f"{'='*60}\n")
    
    try:
        policy = PPO.load(model_path)
        print(f"✓ 策略加载成功")
    except Exception as e:
        print(f"✗ 策略加载失败: {e}")
        return
    
    # 创建环境
    print(f"✓ 创建环境...")
    env = make_minedojo_env(
        task_id=task_id,
        use_camera_smoothing=False,  # 不需要相机平滑
        max_episode_steps=max_steps,
        fast_reset=False  # DAgger收集：每个episode独立环境
    )
    
    # 统计信息
    total_states = 0
    saved_episodes = 0
    success_count = 0
    failure_count = 0
    
    print(f"\n开始收集状态...\n")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        step_count = 0
        episode_reward = 0.0
        
        print(f"Episode {ep+1}/{num_episodes} ", end="", flush=True)
        
        while not done and step_count < max_steps:
            # 策略选择动作
            action, _ = policy.predict(obs, deterministic=deterministic)
            
            # 保存状态和动作
            episode_states.append({
                'observation': obs.copy(),
                'step': step_count,
                'episode': ep
            })
            episode_actions.append(action.copy())
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            episode_reward += reward
            step_count += 1
            
            # 每50步显示进度
            if step_count % 50 == 0:
                print(".", end="", flush=True)
        
        # 判断成功/失败
        # MineDojo harvest任务: reward > 0.5 通常表示成功获得物品
        episode_success = episode_reward > 0.5 or info.get('success', False)
        
        if episode_success:
            success_count += 1
            status = "✓ 成功"
        else:
            failure_count += 1
            status = "✗ 失败"
        
        print(f" {status} | 步数:{step_count:3d} | 奖励:{episode_reward:6.2f}")
        
        # 决定是否保存
        should_save = (not save_failures_only) or (not episode_success)
        
        if should_save:
            # 保存episode数据
            episode_data = {
                'states': episode_states,
                'actions': episode_actions,
                'rewards': episode_rewards,
                'total_reward': episode_reward,
                'success': episode_success,
                'num_steps': step_count,
                'episode_id': ep
            }
            
            success_tag = "success" if episode_success else "fail"
            filename = f"episode_{ep:03d}_{success_tag}_steps{step_count}.npy"
            filepath = os.path.join(output_dir, filename)
            
            np.save(filepath, episode_data)
            
            total_states += len(episode_states)
            saved_episodes += 1
    
    env.close()
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"收集完成！")
    print(f"{'='*60}")
    print(f"总episode数: {num_episodes}")
    print(f"成功: {success_count} ({success_count/num_episodes*100:.1f}%)")
    print(f"失败: {failure_count} ({failure_count/num_episodes*100:.1f}%)")
    print(f"保存episode数: {saved_episodes}")
    print(f"总状态数: {total_states}")
    print(f"平均每episode: {total_states/saved_episodes:.0f}步" if saved_episodes > 0 else "")
    print(f"\n输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    # 保存统计信息
    stats = {
        'num_episodes': num_episodes,
        'success_count': success_count,
        'failure_count': failure_count,
        'saved_episodes': saved_episodes,
        'total_states': total_states,
        'success_rate': success_count / num_episodes if num_episodes > 0 else 0.0
    }
    
    stats_file = os.path.join(output_dir, "collection_stats.npy")
    np.save(stats_file, stats)
    print(f"✓ 统计信息已保存: {stats_file}\n")
    print(f"下一步：\npython tools/label_states.py --states data/policy_states/iter_1/ --output data/expert_labels/iter_1.pkl --smart-sampling --failure-window 10")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="DAgger状态收集工具 - 运行策略并收集状态用于标注"
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
        help="收集的episode数量（默认: 20）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--task-id",
        type=str,
        default="harvest_1_log",
        help="MineDojo任务ID（默认: harvest_1_log）"
    )
    
    parser.add_argument(
        "--save-failures-only",
        action="store_true",
        help="只保存失败的episode（节省存储空间）"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="每个episode最大步数（默认: 1000）"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="使用确定性策略（默认: 随机策略）"
    )
    
    args = parser.parse_args()
    
    # 验证模型文件存在
    if not os.path.exists(args.model):
        print(f"✗ 错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 运行收集
    collect_policy_states(
        model_path=args.model,
        num_episodes=args.episodes,
        output_dir=args.output,
        task_id=args.task_id,
        save_failures_only=args.save_failures_only,
        max_steps=args.max_steps,
        deterministic=args.deterministic
    )


if __name__ == "__main__":
    main()

