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
import torch.nn as nn
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)

# 添加external路径
EXTERNAL_PATH = os.path.join(PROJECT_ROOT, 'external')
sys.path.insert(0, EXTERNAL_PATH)

from src.models.vpt import load_vpt_policy
from src.models.vpt.lib.policy import MinecraftPolicy
from src.envs import make_minedojo_env
import minedojo


class MinedojoActionAdapter(nn.Module):
    """
    适配VPT的输出到MineDojo action space
    """
    
    def __init__(self, vpt_policy: MinecraftPolicy):
        super().__init__()
        self.vpt_policy = vpt_policy
        
        # MineDojo action dimensions
        self.minedojo_action_dim = [3, 3, 4, 25, 25, 8, 244, 36]
        
        # 创建action heads
        hidden_dim = 2048
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in self.minedojo_action_dim
        ])
    
    def forward(self, obs):
        """
        Args:
            obs: (B, H, W, C) numpy array or tensor
        Returns:
            action_logits: list of tensors, one per action dimension
        """
        batch_size = obs.shape[0]
        
        # 转换为tensor (如果还不是)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        
        # 确保在正确的设备上
        obs = obs.to(next(self.vpt_policy.parameters()).device)
        
        # 添加时间维度：(B, H, W, C) -> (B, T=1, H, W, C)
        obs_vpt = obs.unsqueeze(1)  # (B, 1, H, W, C)
        
        # VPT前向传播
        x = self.vpt_policy.img_preprocess(obs_vpt)
        x = self.vpt_policy.img_process(x)
        x = x.squeeze(1)  # 移除时间维度
        latent = self.vpt_policy.lastlayer(x)
        
        # 通过action heads
        action_logits = [head(latent) for head in self.action_heads]
        
        return action_logits
    
    def predict(self, obs, deterministic=True):
        """
        预测动作
        
        Args:
            obs: (H, W, C) 或 (C, H, W) 单个观察
            deterministic: 是否使用确定性动作
        
        Returns:
            action: MineDojo action (list of ints)
        """
        # 检查并转换为HWC格式
        if len(obs.shape) == 3:
            # 检测CHW格式: (C, H, W)
            if obs.shape[0] == 3 or obs.shape[0] == 1:
                if obs.shape[0] < obs.shape[1] and obs.shape[0] < obs.shape[2]:
                    # (C, H, W) -> (H, W, C)
                    obs = np.transpose(obs, (1, 2, 0))
        
        # 处理数据类型和范围
        is_normalized = obs.dtype in [np.float32, np.float64] and obs.max() <= 1.0
        
        if not is_normalized:
            # uint8 [0,255] -> float32 [0,1]
            obs = obs.astype(np.float32) / 255.0
        else:
            # 确保是float32
            obs = obs.astype(np.float32)
        
        # Resize到128x128（VPT训练尺寸）
        import cv2
        if obs.shape[:2] != (128, 128):
            obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        # 添加batch维度
        obs = obs[np.newaxis, ...]  # (1, H, W, C)
        
        with th.no_grad():
            action_logits = self.forward(obs)
            
            # 采样或取argmax
            actions = []
            for logits in action_logits:
                if deterministic:
                    action = th.argmax(logits, dim=-1)
                else:
                    probs = th.softmax(logits, dim=-1)
                    action = th.multinomial(probs, 1).squeeze(-1)
                actions.append(action.item())
        
        return actions


def evaluate_policy(
    adapter: MinedojoActionAdapter,
    task_id: str = "harvest_1_log",
    num_episodes: int = 20,
    max_steps: int = 1000,
    image_size: tuple = (160, 256),  # 使用与录制一致的尺寸(160, 256)，在predict中resize到VPT的128x128
    device: str = "auto",
    render: bool = False
):
    """
    评估policy性能
    
    Args:
        adapter: MinedojoActionAdapter
        task_id: 任务ID
        num_episodes: 评估episodes数
        max_steps: 每个episode最大步数
        image_size: 图像尺寸（与录制时一致）
        device: 设备
        render: 是否渲染
    
    Returns:
        results: 评估结果字典
    """
    # 评估配置
    print(f"评估配置:")
    print(f"  任务: {task_id}")
    print(f"  图像尺寸: {image_size}")
    print(f"  最大步数: {max_steps}")
    print(f"  Episodes: {num_episodes}")
    
    # 评估结果
    results = {
        'episodes': [],
        'success_count': 0,
        'total_episodes': num_episodes,
        'action_stats': {
            'dim0_forward_back': [0, 0, 0],  # 3个值: [forward, back, noop]
            'dim1_left_right': [0, 0, 0],    # 3个值: [left, right, noop]
            'dim2_jump_sneak': [0, 0, 0, 0], # 4个值（修正）
            'dim3_pitch': [],                 # 25个值（连续统计）
            'dim4_yaw': [],                   # 25个值（连续统计）
            'dim5_functional': [0] * 8,       # 8个值（修正）
            'dim6': [0] * 244,                # 244个值
            'dim7': [0] * 36,                 # 36个值
            'total_actions': 0
        }
    }
    
    print(f"\n开始评估 ({num_episodes} episodes)...")
    print("  每个episode独立创建环境，结束后关闭")
    print("  📊 将统计动作分布以分析agent行为")
    
    adapter.eval()  # 设置为评估模式
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        # 为每个episode创建新环境
        env = make_minedojo_env(
            task_id=task_id,
            image_size=image_size,
            max_episode_steps=max_steps
        )
        
        try:
            # 重试reset（MineDojo环境启动可能失败）
            max_reset_retries = 3
            obs = None
            for retry in range(max_reset_retries):
                try:
                    obs = env.reset()
                    break
                except (EOFError, RuntimeError, Exception) as e:
                    if retry < max_reset_retries - 1:
                        print(f"\n  ⚠️  Episode {episode+1} reset失败（重试 {retry+1}/{max_reset_retries}）: {e}")
                        import time
                        time.sleep(2)  # 等待2秒后重试
                        # 关闭并重新创建环境
                        try:
                            env.close()
                        except:
                            pass
                        env = make_minedojo_env(
                            task_id=task_id,
                            image_size=image_size,
                            max_episode_steps=max_steps
                        )
                    else:
                        print(f"\n  ✗ Episode {episode+1} reset失败（已重试{max_reset_retries}次），跳过")
                        raise
            
            if obs is None:
                continue
            
            episode_reward = 0
            episode_steps = 0
            done = False
            episode_actions = []  # 记录episode的所有动作
            action_unchanged_count = 0  # 连续相同动作计数
            prev_action = None
            
            for step in range(max_steps):
                # 预测动作
                rgb_obs = obs['rgb'] if isinstance(obs, dict) else obs
                action = adapter.predict(rgb_obs, deterministic=True)
                
                # 记录动作
                episode_actions.append(action.copy())
                
                # 统计动作分布
                results['action_stats']['total_actions'] += 1
                results['action_stats']['dim0_forward_back'][action[0]] += 1
                results['action_stats']['dim1_left_right'][action[1]] += 1
                results['action_stats']['dim2_jump_sneak'][action[2]] += 1
                results['action_stats']['dim3_pitch'].append(action[3])
                results['action_stats']['dim4_yaw'].append(action[4])
                results['action_stats']['dim5_functional'][action[5]] += 1
                results['action_stats']['dim6'][action[6]] += 1
                results['action_stats']['dim7'][action[7]] += 1
                
                # 检测动作是否改变（检测"原地不动"）
                if prev_action is not None and np.array_equal(action, prev_action):
                    action_unchanged_count += 1
                else:
                    action_unchanged_count = 0
                prev_action = action.copy()
                
                # 执行
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # 判断成功
            success = done and episode_reward > 0
            
            if success:
                results['success_count'] += 1
            
            # 计算episode的动作统计
            episode_action_array = np.array(episode_actions)
            unique_actions = len(np.unique(episode_action_array, axis=0))
            action_diversity = unique_actions / max(episode_steps, 1)
            
            # 检测是否大部分时间重复相同动作（"卡住"）
            max_repeated = 1
            if episode_steps > 0:
                for i in range(episode_steps):
                    repeat_count = 1
                    for j in range(i + 1, episode_steps):
                        if np.array_equal(episode_actions[i], episode_actions[j-1]):
                            repeat_count += 1
                        else:
                            break
                    max_repeated = max(max_repeated, repeat_count)
            
            stuck_ratio = max_repeated / max(episode_steps, 1)
            is_stuck = stuck_ratio > 0.5  # 超过50%时间重复同一动作
            
            results['episodes'].append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'success': success,
                'done': done,
                'unique_actions': unique_actions,
                'action_diversity': action_diversity,
                'max_repeated': max_repeated,
                'stuck_ratio': stuck_ratio,
                'is_stuck': is_stuck
            })
            
            # 实时显示
            if (episode + 1) % 5 == 0 or episode == 0:
                current_success_rate = results['success_count'] / (episode + 1)
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"Success Rate = {current_success_rate:.2%}")
        
        finally:
            # 确保环境被关闭
            env.close()
    
    # 计算统计
    results['success_rate'] = results['success_count'] / num_episodes
    results['avg_reward'] = np.mean([ep['reward'] for ep in results['episodes']])
    results['avg_steps'] = np.mean([ep['steps'] for ep in results['episodes']])
    
    return results


def load_trained_model(checkpoint_path: str, device: str = 'cpu'):
    """
    加载训练好的VPT模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备
    
    Returns:
        adapter: MinedojoActionAdapter
    """
    print(f"加载模型: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = th.load(checkpoint_path, map_location=device)
    
    # 显示checkpoint信息
    print(f"\n📦 Checkpoint信息:")
    print(f"  训练Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  训练Loss: {checkpoint.get('loss', 'unknown'):.4f}" if isinstance(checkpoint.get('loss'), (int, float)) else f"  训练Loss: {checkpoint.get('loss', 'unknown')}")
    print(f"  VPT权重路径: {checkpoint.get('vpt_weights_path', 'unknown')}")
    
    # 创建VPT policy（显示详细信息）
    print(f"\n🤖 加载VPT预训练权重...")
    vpt_weights_path = checkpoint.get('vpt_weights_path', 'data/pretrained/vpt/rl-from-early-game-2x.weights')
    policy, load_result = load_vpt_policy(
        vpt_weights_path,
        device=device,
        verbose=True  # 显示详细加载信息
    )
    
    # 创建adapter
    adapter = MinedojoActionAdapter(policy)
    
    # 加载adapter权重（这是微调后的权重）
    print(f"\n📥 加载微调后的权重...")
    adapter.load_state_dict(checkpoint['model_state_dict'])
    adapter = adapter.to(device)
    adapter.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    
    print(f"\n✓ 模型加载成功")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return adapter


def main():
    parser = argparse.ArgumentParser(description="评估VPT BC模型")
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型checkpoint路径')
    parser.add_argument('--task-id', type=str, default='harvest_1_log',
                        help='MineDojo任务ID')
    parser.add_argument('--episodes', type=int, default=20,
                        help='评估episodes数')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='每个episode最大步数')
    parser.add_argument('--image-size', type=int, nargs=2, default=[160, 256],
                        help='图像尺寸 (H W) - 与录制时一致')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='设备')
    parser.add_argument('--render', action='store_true',
                        help='渲染环境（慎用）')
    
    args = parser.parse_args()
    
    print('=' * 70)
    print('VPT BC模型评估')
    print('=' * 70)
    print(f'模型: {args.model}')
    print(f'任务: {args.task_id}')
    print(f'Episodes: {args.episodes}')
    print(f'最大步数: {args.max_steps}')
    print(f'图像尺寸: {tuple(args.image_size)}')
    print(f'设备: {args.device}')
    print('=' * 70)
    
    # 加载模型
    adapter = load_trained_model(args.model, device=args.device)
    
    # 评估
    results = evaluate_policy(
        adapter=adapter,
        task_id=args.task_id,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        image_size=tuple(args.image_size),
        device=args.device,
        render=args.render
    )
    
    # 打印详细分析报告
    print('\n' + '=' * 70)
    print('📊 评估结果与动作分析')
    print('=' * 70)
    
    # 基本结果
    print(f'\n【基本统计】')
    print(f'总Episodes: {results["total_episodes"]}')
    print(f'成功: {results["success_count"]}/{results["total_episodes"]} '
          f'({results["success_rate"]:.2%})')
    print(f'平均奖励: {results["avg_reward"]:.2f}')
    print(f'平均步数: {results["avg_steps"]:.1f}')
    
    # 动作分布分析
    stats = results['action_stats']
    total = stats['total_actions']
    
    print(f'\n【动作分布统计】(总动作数: {total})')
    print('-' * 70)
    
    # 维度0: 前进/后退
    dim0 = stats['dim0_forward_back']
    print(f'移动（前/后）:')
    print(f'  前进: {dim0[0]:6d} ({dim0[0]/total*100:5.1f}%)')
    print(f'  后退: {dim0[1]:6d} ({dim0[1]/total*100:5.1f}%)')
    print(f'  不动: {dim0[2]:6d} ({dim0[2]/total*100:5.1f}%)  {"⚠️ 过高！" if dim0[2]/total > 0.5 else ""}')
    
    # 维度1: 左右移动
    dim1 = stats['dim1_left_right']
    print(f'\n平移（左/右）:')
    print(f'  左移: {dim1[0]:6d} ({dim1[0]/total*100:5.1f}%)')
    print(f'  右移: {dim1[1]:6d} ({dim1[1]/total*100:5.1f}%)')
    print(f'  不动: {dim1[2]:6d} ({dim1[2]/total*100:5.1f}%)  {"⚠️ 过高！" if dim1[2]/total > 0.5 else ""}')
    
    # 维度2: 跳跃/潜行等（4个值）
    dim2 = stats['dim2_jump_sneak']
    print(f'\n跳跃/潜行/其他（维度2，4个值）:')
    for i, count in enumerate(dim2):
        percentage = count/total*100 if total > 0 else 0
        warning = "  ⚠️ 过低！" if i == 0 and percentage < 10 else ""  # 假设动作0是跳跃
        print(f'  动作{i}: {count:6d} ({percentage:5.1f}%){warning}')
    
    # 维度3-4: 相机
    dim3_mean = np.mean(stats['dim3_pitch']) if stats['dim3_pitch'] else 0
    dim3_std = np.std(stats['dim3_pitch']) if stats['dim3_pitch'] else 0
    dim4_mean = np.mean(stats['dim4_yaw']) if stats['dim4_yaw'] else 0
    dim4_std = np.std(stats['dim4_yaw']) if stats['dim4_yaw'] else 0
    print(f'\n相机移动:')
    print(f'  Pitch (上下): mean={dim3_mean:6.2f}, std={dim3_std:6.2f}  {"⚠️ 过低！" if dim3_std < 0.01 else ""}')
    print(f'  Yaw (左右):   mean={dim4_mean:6.2f}, std={dim4_std:6.2f}  {"⚠️ 过低！" if dim4_std < 0.01 else ""}')
    
    # 维度5: 功能键（8个值）
    dim5 = stats['dim5_functional']
    print(f'\n功能键（维度5，8个值）:')
    for i, count in enumerate(dim5):
        if count > 0:  # 只打印有使用的
            percentage = count/total*100 if total > 0 else 0
            print(f'  功能{i}: {count:6d} ({percentage:5.1f}%)')
    
    # Episode行为分析
    print(f'\n【Episode行为分析】')
    print('-' * 70)
    
    stuck_episodes = [ep for ep in results['episodes'] if ep['is_stuck']]
    low_diversity = [ep for ep in results['episodes'] if ep['action_diversity'] < 0.1]
    
    print(f'卡住的Episodes: {len(stuck_episodes)}/{results["total_episodes"]} '
          f'({len(stuck_episodes)/results["total_episodes"]*100:.1f}%)')
    print(f'  定义: 超过50%时间重复相同动作')
    
    print(f'低动作多样性: {len(low_diversity)}/{results["total_episodes"]} '
          f'({len(low_diversity)/results["total_episodes"]*100:.1f}%)')
    print(f'  定义: 动作多样性<0.1 (10%独特动作)')
    
    avg_diversity = np.mean([ep['action_diversity'] for ep in results['episodes']])
    avg_stuck_ratio = np.mean([ep['stuck_ratio'] for ep in results['episodes']])
    print(f'\n平均动作多样性: {avg_diversity:.3f}  {"⚠️ 过低！" if avg_diversity < 0.2 else ""}')
    print(f'平均卡住比例: {avg_stuck_ratio:.3f}  {"⚠️ 过高！" if avg_stuck_ratio > 0.3 else ""}')
    
    # 诊断和建议
    print(f'\n【🔍 问题诊断】')
    print('-' * 70)
    
    issues = []
    
    # 检查各种问题
    if dim0[2]/total > 0.5:  # 不前进/后退
        issues.append("❌ 移动不足：超过50%时间不前进/后退")
    
    if dim1[2]/total > 0.5:  # 不左右移动
        issues.append("❌ 平移不足：超过50%时间不左右移动")
    
    if dim2[0]/total < 0.1:  # 跳跃太少
        issues.append("❌ 跳跃缺失：跳跃动作<10%（专家数据是85%！）")
    
    if dim3_std < 0.01 or dim4_std < 0.01:  # 相机不动
        issues.append("❌ 相机静止：相机几乎不移动")
    
    if avg_diversity < 0.2:
        issues.append("❌ 动作单一：动作多样性过低")
    
    if len(stuck_episodes) > results["total_episodes"] * 0.3:
        issues.append(f"❌ 频繁卡住：{len(stuck_episodes)}个episodes卡住")
    
    if results["success_rate"] < 0.2:
        issues.append(f"❌ 成功率极低：只有{results['success_rate']:.1%}")
    
    if issues:
        for issue in issues:
            print(f'  {issue}')
    else:
        print('  ✓ 未检测到明显问题')
    
    # 显示成功和失败的episodes
    successful_episodes = [ep for ep in results['episodes'] if ep['success']]
    failed_episodes = [ep for ep in results['episodes'] if not ep['success']]
    
    if successful_episodes:
        print(f'\n【✓ 成功的Episodes】')
        for ep in successful_episodes[:5]:
            print(f"  Episode {ep['episode']+1}: "
                  f"Reward={ep['reward']:.1f}, Steps={ep['steps']}, "
                  f"Diversity={ep['action_diversity']:.3f}")
    
    if failed_episodes:
        print(f'\n【✗ 失败的Episodes（前5个）】')
        for ep in failed_episodes[:5]:
            status = "🔒 卡住" if ep['is_stuck'] else "❓ 其他"
            print(f"  Episode {ep['episode']+1}: "
                  f"{status}, Steps={ep['steps']}, "
                  f"Diversity={ep['action_diversity']:.3f}, "
                  f"Stuck={ep['stuck_ratio']:.1%}")
    
    print('\n' + '=' * 70)
    
    return results


if __name__ == '__main__':
    main()
