"""
零样本评估完整版VPT Agent

在harvest_log任务上评估完整版VPT的零样本性能，对比简化版
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.training.vpt import VPTAgent
from src.envs.task_wrappers import HarvestLogWrapper


def create_env():
    """创建MineDojo环境"""
    import minedojo
    from src.envs.task_wrappers import HarvestLogWrapper
    
    base_env = minedojo.make(
        task_id="harvest_1_log",
        image_size=(160, 256),
        world_seed=42,
        seed=42,
    )
    env = HarvestLogWrapper(base_env, required_logs=1, verbose=False)
    return env


def evaluate_agent(agent, num_episodes=10, max_steps=1200, verbose=True, debug_actions=False):
    """
    评估agent在环境中的表现
    
    每个episode创建新环境，避免状态污染
    
    Args:
        agent: Agent实例
        num_episodes: 评估轮数
        max_steps: 每轮最大步数
        verbose: 是否打印详细信息
        debug_actions: 是否打印动作详情（调试用）
    
    Returns:
        stats: 统计信息字典
    """
    success_count = 0
    total_reward = 0
    episode_lengths = []
    episode_rewards = []
    
    for episode_idx in range(num_episodes):
        # 每个episode创建新环境
        env = None
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                env = create_env()
                obs = env.reset()
                break
            except Exception as e:
                retry_count += 1
                if env:
                    try:
                        env.close()
                    except:
                        pass
                    env = None
                if retry_count < max_retries:
                    if verbose:
                        print(f"  ⚠️  环境重置失败，重试 {retry_count}/{max_retries}...")
                        print(f"      错误: {str(e)[:100]}")
                    import time
                    time.sleep(2)  # 增加等待时间到2秒
                else:
                    if verbose:
                        print(f"  ❌ Episode {episode_idx + 1} 环境创建失败，跳过")
                        print(f"      最后错误: {str(e)[:100]}")
                    break  # 跳出while循环
        
        if env is None:
            continue  # 跳过此episode
        
        # 临时启用action调试
        original_debug = agent.debug_actions
        agent.debug_actions = debug_actions
        
        try:
            agent.reset()
            
            episode_reward = 0
            done = False
            step_count = 0
            
            if verbose:
                print(f"\n📍 Episode {episode_idx + 1}/{num_episodes}")
            
            while not done and step_count < max_steps:
                # 预测动作
                action = agent.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # 打印进度
                if verbose and step_count % 100 == 0:
                    print(f"  Step {step_count}/{max_steps}, Reward={episode_reward:.2f}")
                
                # 检查成功
                if reward > 0:
                    success_count += 1
                    if verbose:
                        print(f"  ✅ 成功！第{step_count}步获得奖励: {reward}")
                    break
            
            episode_lengths.append(step_count)
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            
            if verbose:
                status = "✅ 成功" if episode_reward > 0 else "❌ 失败"
                print(f"  {status} - 步数: {step_count}, 累积奖励: {episode_reward:.2f}")
        
        finally:
            # 恢复debug设置
            agent.debug_actions = original_debug
            # 关闭环境
            if env:
                try:
                    env.close()
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  环境关闭出错: {e}")
    
    stats = {
        'success_count': success_count,
        'success_rate': success_count / num_episodes,
        'avg_reward': total_reward / num_episodes,
        'total_reward': total_reward,
        'avg_steps': np.mean(episode_lengths),
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='零样本评估VPT Agent')
    parser.add_argument('--episodes', type=int, default=10,
                        help='评估轮数')
    parser.add_argument('--max_steps', type=int, default=1200,
                        help='每轮最大步数')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: cpu/cuda/mps/auto')
    parser.add_argument('--weights', type=str, 
                        default='data/pretrained/vpt/rl-from-early-game-2x.weights',
                        help='VPT权重路径')
    parser.add_argument('--debug-actions', action='store_true',
                        help='打印详细的动作转换日志（调试用）')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 VPT零样本评估")
    print("="*70)
    print(f"任务: harvest_1_log")
    print(f"评估轮数: {args.episodes}")
    print(f"最大步数: {args.max_steps}")
    print(f"设备: {args.device}")
    if args.debug_actions:
        print(f"调试模式: 启用动作日志")
    print("="*70 + "\n")
    
    # 评估VPT Agent
    print("="*70)
    print("📊 评估VPT Agent (完整版: pi_head + hidden state)")
    print("="*70)
    
    agent = VPTAgent(
        vpt_weights_path=args.weights,
        device=args.device,
        verbose=True,
        debug_actions=False  # 通过evaluate_agent参数控制
    )
    agent.eval()
    
    print("\n💡 提示: 每个episode会创建新环境，避免状态污染")
    print("="*70 + "\n")
    
    stats = evaluate_agent(
        agent,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=True,
        debug_actions=args.debug_actions
    )
    
    print("\n" + "-"*70)
    print("📈 评估统计结果:")
    print("-"*70)
    print(f"成功率: {stats['success_rate']*100:.1f}% ({stats['success_count']}/{args.episodes})")
    print(f"平均奖励: {stats['avg_reward']:.3f}")
    print(f"平均步数: {stats['avg_steps']:.1f}")
    print("-"*70 + "\n")
    
    print("✅ 评估完成！")


if __name__ == '__main__':
    main()

