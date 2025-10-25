#!/usr/bin/env python3
"""
MineCLIP微调 - 数据加载器

从专家演示中加载数据并构造时序对比学习样本
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class ExpertDemosDataset(Dataset):
    """
    专家演示数据集
    
    构造时序对比样本：
    - Anchor: 第t帧
    - Positive: 第t+k帧（更接近目标）
    - Negative: 第t-k帧（离目标更远）或其他episode的帧
    """
    
    def __init__(
        self,
        expert_demos_dir,
        temporal_gap=10,
        max_episodes=None,
        frame_subsample=2,
        use_hard_negatives=True
    ):
        """
        初始化数据集
        
        Args:
            expert_demos_dir: 专家演示目录
            temporal_gap: 时序间隔（正样本距离anchor的帧数）
            max_episodes: 最多加载的episode数量
            frame_subsample: 帧采样间隔（减少数据量）
            use_hard_negatives: 是否使用困难负样本（其他episode的相似帧）
        """
        self.expert_demos_dir = Path(expert_demos_dir)
        self.temporal_gap = temporal_gap
        self.frame_subsample = frame_subsample
        self.use_hard_negatives = use_hard_negatives
        
        print(f"\n{'='*70}")
        print(f"加载专家演示数据")
        print(f"{'='*70}")
        print(f"目录: {expert_demos_dir}")
        print(f"时序间隔: {temporal_gap}")
        print(f"帧采样: 每{frame_subsample}帧")
        
        # 加载所有episodes
        self.episodes = self._load_episodes(max_episodes)
        
        # 构造训练样本
        self.samples = self._create_samples()
        
        print(f"\n✓ 数据加载完成:")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  训练样本: {len(self.samples)}")
        print(f"{'='*70}\n")
    
    def _load_episodes(self, max_episodes):
        """加载所有episodes的帧"""
        episode_dirs = sorted(self.expert_demos_dir.glob("episode_*"))
        
        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
        
        episodes = []
        
        print(f"加载 {len(episode_dirs)} 个episodes...")
        
        for i, ep_dir in enumerate(episode_dirs):
            if i % 10 == 0:
                print(f"  进度: {i}/{len(episode_dirs)}", end='\r')
            
            # 加载该episode的所有帧
            frame_files = sorted(ep_dir.glob("frame_*.npy"))
            
            # 采样帧（减少数据量）
            frame_files = frame_files[::self.frame_subsample]
            
            frames = []
            for frame_file in frame_files:
                try:
                    frame_data = np.load(frame_file, allow_pickle=True).item()
                    obs = frame_data['observation']
                    frames.append(obs)
                except:
                    continue
            
            if len(frames) > self.temporal_gap * 3:  # 确保有足够的帧
                episodes.append({
                    'name': ep_dir.name,
                    'frames': frames,
                    'length': len(frames)
                })
        
        print(f"\n  成功加载: {len(episodes)} episodes")
        return episodes
    
    def _create_samples(self):
        """
        构造时序对比样本
        
        每个样本包含：
        - anchor_episode_idx: anchor所在episode索引
        - anchor_frame_idx: anchor帧索引
        - positive_frame_idx: positive帧索引（同episode）
        - negative_episode_idx: negative所在episode索引
        - negative_frame_idx: negative帧索引
        """
        samples = []
        
        print("构造训练样本...")
        
        for ep_idx, episode in enumerate(self.episodes):
            frames = episode['frames']
            n_frames = len(frames)
            
            # 为每个可能的anchor创建样本
            for t in range(self.temporal_gap, n_frames - self.temporal_gap):
                # Positive: 后面的帧（更接近目标）
                pos_idx = min(t + self.temporal_gap, n_frames - 1)
                
                # Negative: 有两种策略
                if self.use_hard_negatives and len(self.episodes) > 1:
                    # 策略1: 从其他episode采样（困难负样本）
                    neg_ep_idx = (ep_idx + np.random.randint(1, len(self.episodes))) % len(self.episodes)
                    neg_episode = self.episodes[neg_ep_idx]
                    neg_idx = np.random.randint(0, len(neg_episode['frames']))
                else:
                    # 策略2: 从同episode的早期帧（简单负样本）
                    neg_ep_idx = ep_idx
                    neg_idx = max(0, t - self.temporal_gap)
                
                samples.append({
                    'anchor_ep': ep_idx,
                    'anchor_idx': t,
                    'positive_idx': pos_idx,
                    'negative_ep': neg_ep_idx,
                    'negative_idx': neg_idx,
                })
        
        print(f"  生成: {len(samples)} 个样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回一个训练样本
        
        Returns:
            dict: {
                'anchor': [C, H, W] tensor,
                'positive': [C, H, W] tensor,
                'negative': [C, H, W] tensor
            }
        """
        sample = self.samples[idx]
        
        # 获取anchor
        anchor_ep = self.episodes[sample['anchor_ep']]
        anchor = anchor_ep['frames'][sample['anchor_idx']]
        
        # 获取positive（同episode的后期帧）
        positive = anchor_ep['frames'][sample['positive_idx']]
        
        # 获取negative
        negative_ep = self.episodes[sample['negative_ep']]
        negative = negative_ep['frames'][sample['negative_idx']]
        
        # 转换为tensor并归一化
        anchor = self._to_tensor(anchor)
        positive = self._to_tensor(positive)
        negative = self._to_tensor(negative)
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }
    
    def _to_tensor(self, obs):
        """
        将观察转换为tensor
        
        Args:
            obs: numpy array, [C, H, W] or [H, W, C]
            
        Returns:
            torch.Tensor: [C, H, W], 范围[0, 1]
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        # 确保是 [C, H, W] 格式
        if obs.shape[0] != 3:  # 如果是 [H, W, C]
            obs = obs.permute(2, 0, 1)
        
        # 归一化到 [0, 1]
        if obs.max() > 1.0:
            obs = obs / 255.0
        
        return obs


def create_dataloader(
    expert_demos_dir,
    batch_size=16,
    temporal_gap=10,
    max_episodes=None,
    num_workers=4
):
    """
    创建数据加载器
    
    Args:
        expert_demos_dir: 专家演示目录
        batch_size: 批次大小
        temporal_gap: 时序间隔
        max_episodes: 最多加载的episode数量
        num_workers: 数据加载进程数
        
    Returns:
        DataLoader
    """
    dataset = ExpertDemosDataset(
        expert_demos_dir=expert_demos_dir,
        temporal_gap=temporal_gap,
        max_episodes=max_episodes,
        frame_subsample=2,  # 每2帧采样1帧
        use_hard_negatives=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器...")
    
    dataloader = create_dataloader(
        expert_demos_dir="data/tasks/harvest_1_log/expert_demos",
        batch_size=4,
        temporal_gap=10,
        max_episodes=10,
        num_workers=0  # 测试时用0
    )
    
    print(f"\n测试批次加载:")
    batch = next(iter(dataloader))
    
    print(f"  Anchor shape: {batch['anchor'].shape}")
    print(f"  Positive shape: {batch['positive'].shape}")
    print(f"  Negative shape: {batch['negative'].shape}")
    
    print(f"\n✓ 数据加载器测试成功")

