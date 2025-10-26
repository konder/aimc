#!/usr/bin/env python3
"""
行为克隆(Behavioral Cloning)训练脚本

从专家演示数据训练策略。这是DAgger的基础，也可以单独使用。

Usage:
    # 从手动录制的演示训练
    python src/training/train_bc.py \
        --data data/expert_demos/round_0/ \
        --output checkpoints/bc_round_0.zip \
        --epochs 30

    # 从DAgger聚合数据训练
    python src/training/train_bc.py \
        --data data/dagger_combined/iter_1.pkl \
        --output checkpoints/dagger_iter_1.zip
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import minedojo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs import make_minedojo_env


class ExpertDataset(Dataset):
    """专家演示数据集"""
    
    def __init__(self, observations, actions):
        """
        Args:
            observations: numpy array of shape (N, C, H, W), uint8 [0, 255] 或 float32
            actions: numpy array of shape (N, 8) - MineDojo MultiDiscrete
        """
        # 归一化图像到[0, 1]
        if observations.dtype == np.uint8:
            # uint8 类型，需要归一化
            observations = observations.astype(np.float32) / 255.0
        elif observations.dtype in [np.float32, np.float64]:
            # float 类型，检查是否需要归一化
            if observations.max() > 1.5:
                # 值域在 [0, 255]，需要归一化
                print(f"  ⚠️  检测到未归一化的 float 数据 (范围: [{observations.min():.1f}, {observations.max():.1f}])，正在归一化...")
                observations = observations.astype(np.float32) / 255.0
            # 否则假设已经归一化到 [0, 1]
        
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def load_expert_demonstrations(data_path):
    """
    加载专家演示数据
    
    支持两种格式:
    1. 目录: 包含多个.npy文件（手动录制）
    2. .pkl文件: 单个pickle文件（DAgger标注）
    
    Returns:
        observations: (N, C, H, W)
        actions: (N, 8)
    """
    observations = []
    actions = []
    
    data_path = Path(data_path)
    
    if data_path.is_dir():
        # 从目录加载多个.npy文件
        print(f"从目录加载: {data_path}")
        
        # 1. 首先查找episode子目录（record_manual_chopping.py格式）
        episode_dirs = sorted(data_path.glob("episode_*/"))
        if episode_dirs:
            print(f"  找到 {len(episode_dirs)} 个episode目录")
            for ep_dir in episode_dirs:
                frame_files = sorted(ep_dir.glob("frame_*.npy"))
                if frame_files:
                    print(f"  [{ep_dir.name}] 加载 {len(frame_files)} 个帧...")
                    for file in frame_files:
                        try:
                            frame_data = np.load(file, allow_pickle=True).item()
                            obs = frame_data['observation']
                            action = frame_data['action']
                            observations.append(obs)
                            actions.append(action)
                        except Exception as e:
                            print(f"    ⚠️  {file.name}: 加载失败 - {e}")
                    print(f"    ✓ {ep_dir.name}: 成功加载 {len(frame_files)} 帧")
        
        # 2. 如果没有episode子目录，查找当前目录的episode_*.npy文件
        else:
            episode_files = sorted(data_path.glob("episode_*.npy"))
            frame_files = sorted(data_path.glob("frame_*.npy"))
            
            if episode_files:
                # run_policy_collect_states.py 格式
                for file in episode_files:
                    try:
                        episode_data = np.load(file, allow_pickle=True).item()
                        states = episode_data['states']
                        episode_actions = episode_data.get('actions', [])
                        
                        for state, action in zip(states, episode_actions):
                            obs = state['observation']
                            observations.append(obs)
                            actions.append(action)
                        
                        print(f"  ✓ {file.name}: {len(states)} 帧")
                    except Exception as e:
                        print(f"  ⚠️  {file.name}: 加载失败 - {e}")
            
            elif frame_files:
                # record_manual_chopping.py 格式（单episode目录）
                for file in frame_files:
                    try:
                        frame_data = np.load(file, allow_pickle=True).item()
                        obs = frame_data['observation']
                        action = frame_data['action']
                        observations.append(obs)
                        actions.append(action)
                    except Exception as e:
                        print(f"  ⚠️  {file.name}: 加载失败 - {e}")
                
                if observations:
                    print(f"  ✓ 加载了 {len(frame_files)} 个帧文件")
            
            else:
                print(f"  ⚠️  目录中未找到episode目录或frame文件")
    
    elif data_path.suffix == '.pkl':
        # 从pickle文件加载（DAgger标注格式）
        print(f"从pickle文件加载: {data_path}")
        
        with open(data_path, 'rb') as f:
            labeled_data = pickle.load(f)
        
        for item in labeled_data:
            observations.append(item['observation'])
            # 使用专家标注的动作，而非策略动作
            actions.append(item['expert_action'])
        
        print(f"  ✓ 加载了 {len(labeled_data)} 个标注")
    
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")
    
    if not observations:
        raise ValueError("未加载到任何数据！")
    
    # 转换为numpy数组
    observations = np.array(observations)
    actions = np.array(actions)
    
    # 转置观察数据: (N, H, W, C) -> (N, C, H, W)
    # 因为PyTorch CNN期望channel-first格式
    if len(observations.shape) == 4 and observations.shape[-1] == 3:
        print(f"  转置图像: (N, H, W, C) -> (N, C, H, W)")
        observations = np.transpose(observations, (0, 3, 1, 2))
    
    print(f"\n总计:")
    print(f"  观察: {observations.shape}")
    print(f"  动作: {actions.shape}")
    
    return observations, actions


def train_bc_with_ppo(
    observations,
    actions,
    output_path,
    task_id="harvest_1_log",
    learning_rate=3e-4,
    n_epochs=30,
    batch_size=64,
    device="auto"
):
    """
    使用PPO框架进行行为克隆训练
    
    策略: 通过预训练策略网络，然后用PPO微调
    
    Args:
        observations: 专家观察
        actions: 专家动作
        output_path: 输出模型路径
        task_id: MineDojo任务ID
        learning_rate: 学习率
        n_epochs: 训练轮数
        batch_size: 批次大小
        device: 计算设备
    """
    
    # MPS设备稳定性调整
    if device == "mps":
        print(f"\n⚠️  MPS设备检测到，应用稳定性优化:")
        print(f"  - 降低学习率: 3e-4 -> 1e-4")
        print(f"  - 启用梯度裁剪")
        print(f"  - 添加数值稳定性保护\n")
        learning_rate = 1e-4  # MPS上使用更低的学习率
    
    print(f"\n{'='*60}")
    print(f"行为克隆训练")
    print(f"{'='*60}")
    print(f"数据量: {len(observations)} 样本")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {n_epochs}")
    print(f"批次大小: {batch_size}")
    print(f"设备: {device}")
    print(f"{'='*60}\n")
    
    # 创建环境（用于获取observation_space和action_space）
    print("创建环境...")
    def make_env():
        return make_minedojo_env(
            task_id=task_id,
            max_episode_steps=1000
        )
    
    env = DummyVecEnv([make_env])
    
    # 创建PPO模型
    print("创建PPO模型...")
    # 注意: normalize_images=False 因为MinedojoWrapper已经归一化到[0,1]
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,  # PPO内部epoch
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=1,
        policy_kwargs=dict(normalize_images=False)  # 图像已在环境中归一化
    )
    
    # 预训练策略网络（行为克隆）
    print(f"\n开始行为克隆预训练...")
    print(f"{'='*60}\n")
    
    dataset = ExpertDataset(observations, actions)
    
    # 验证数据归一化（调试信息）
    sample_obs = dataset.observations[:4]
    print(f"数据集样本检查:")
    print(f"  形状: {sample_obs.shape}")
    print(f"  类型: {sample_obs.dtype}")
    print(f"  范围: [{sample_obs.min().item():.3f}, {sample_obs.max().item():.3f}]")
    print(f"  均值: {sample_obs.mean().item():.3f}")
    print(f"  标准差: {sample_obs.std().item():.3f}")
    
    if sample_obs.max() > 1.5:
        print(f"  🔴 错误: 数据未正确归一化！应该在[0,1]范围内")
        print(f"  → 这会导致训练失败，特别是在 MPS/CUDA 设备上")
        raise ValueError("数据归一化失败！请检查数据加载流程")
    elif sample_obs.max() < 0.01:
        print(f"  ⚠️  警告: 数据可能全为0或过暗")
    else:
        print(f"  ✓ 数据归一化正确\n")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # macOS上避免多进程问题
    )
    
    # 获取策略网络
    policy_net = model.policy
    optimizer = torch.optim.Adam(
        policy_net.parameters(), 
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        num_skipped = 0
        
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(model.device)
            batch_actions = batch_actions.to(model.device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # PPO策略输出
            features = policy_net.extract_features(batch_obs)
            latent_pi = policy_net.mlp_extractor.forward_actor(features)
            action_logits = policy_net.action_net(latent_pi)
            
            # MineDojo MultiDiscrete: 8个离散动作
            # action_logits shape: (batch_size, sum of action dimensions)
            # 我们需要分别计算每个维度的loss
            
            # 简化版本: 假设所有动作维度相同（实际上MineDojo不是这样）
            # 更精确的实现需要根据MultiDiscrete的nvec分割logits
            
            # 为了简化，我们用整体action预测
            # 注意: 这是一个简化版本，实际DAgger可能需要更精确的实现
            
            loss = 0.0
            acc = 0.0
            
            # 这里使用简化的损失计算
            # 实际应该根据MultiDiscrete的各个维度分别计算
            # 但对于初始BC训练，这个简化版本通常够用
            
            # 使用策略的distribution来计算log_prob
            actions_tensor = batch_actions
            distribution = policy_net.get_distribution(batch_obs)
            log_prob = distribution.log_prob(actions_tensor)
            
            # BC loss = -log_prob (最大化专家动作的概率)
            loss = -log_prob.mean()
            
            # 检查loss有效性（MPS稳定性保护）
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  警告: Batch {num_batches} loss无效 (NaN/Inf)，跳过")
                num_skipped += 1
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（关键！MPS设备必需）
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        
        status_msg = f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}"
        if num_skipped > 0:
            status_msg += f" | 跳过: {num_skipped} batches"
        print(status_msg)
    
    print(f"\n{'='*60}")
    print(f"预训练完成！")
    print(f"{'='*60}\n")
    
    # 保存模型
    print(f"保存模型到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    model.save(output_path)
    print(f"✓ 模型已保存\n")
    
    env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="行为克隆(BC)训练 - 从专家演示学习策略"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="专家演示数据路径（目录或.pkl文件）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出模型路径（.zip文件）"
    )
    
    parser.add_argument(
        "--task-id",
        type=str,
        default="harvest_1_log",
        help="MineDojo任务ID（默认: harvest_1_log）"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="学习率（默认: 3e-4）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="训练轮数（默认: 30）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批次大小（默认: 64）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto/cpu/cuda/mps，默认: auto)"
    )
    
    args = parser.parse_args()
    
    # 验证数据路径存在
    if not os.path.exists(args.data):
        print(f"✗ 错误: 数据路径不存在: {args.data}")
        sys.exit(1)
    
    # 加载专家演示
    try:
        observations, actions = load_expert_demonstrations(args.data)
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        sys.exit(1)
    
    # 训练BC
    train_bc_with_ppo(
        observations=observations,
        actions=actions,
        output_path=args.output,
        task_id=args.task_id,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"✓ BC训练完成！")
    print(f"\n下一步:")
    print(f"  1. 评估模型: python tools/evaluate_policy.py --model {args.output}")
    print(f"  2. 收集状态: python tools/run_policy_collect_states.py --model {args.output}\n")


if __name__ == "__main__":
    main()

