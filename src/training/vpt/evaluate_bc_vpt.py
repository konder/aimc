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
    device: str = "cpu",
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
        'total_episodes': num_episodes
    }
    
    print(f"\n开始评估 ({num_episodes} episodes)...")
    print("  每个episode独立创建环境，结束后关闭")
    
    adapter.eval()  # 设置为评估模式
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        # 为每个episode创建新环境
        env = make_minedojo_env(
            task_id=task_id,
            image_size=image_size,
            max_episode_steps=max_steps
        )
        
        try:
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            for step in range(max_steps):
                # 预测动作
                rgb_obs = obs['rgb'] if isinstance(obs, dict) else obs
                action = adapter.predict(rgb_obs, deterministic=True)
                
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
            
            results['episodes'].append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'success': success,
                'done': done
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
    
    # 打印结果
    print('\n' + '=' * 70)
    print('评估结果')
    print('=' * 70)
    print(f'总Episodes: {results["total_episodes"]}')
    print(f'成功: {results["success_count"]}/{results["total_episodes"]} '
          f'({results["success_rate"]:.2%})')
    print(f'平均奖励: {results["avg_reward"]:.2f}')
    print(f'平均步数: {results["avg_steps"]:.1f}')
    print('=' * 70)
    
    # 显示成功的episodes
    successful_episodes = [ep for ep in results['episodes'] if ep['success']]
    if successful_episodes:
        print('\n成功的Episodes:')
        for ep in successful_episodes[:10]:  # 最多显示10个
            print(f"  Episode {ep['episode']}: "
                  f"Reward={ep['reward']:.1f}, Steps={ep['steps']}")
    
    return results


if __name__ == '__main__':
    main()
