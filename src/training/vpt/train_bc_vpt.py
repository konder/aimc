#!/usr/bin/env python3
"""
使用VPT预训练权重进行BC训练

核心改进：
1. ✅ 使用真正的VPT MinecraftPolicy（从官方仓库）
2. ✅ 正确加载VPT权重（修复key前缀问题）
3. ✅ MineDojo action space适配
4. ✅ 专家数据加载和增强
"""

import os
import sys
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)

# 添加external路径（用于minerl依赖）
EXTERNAL_PATH = os.path.join(PROJECT_ROOT, 'external')
sys.path.insert(0, EXTERNAL_PATH)

from src.models.vpt.weights_loader import load_vpt_policy, create_vpt_policy
from src.models.vpt.lib.policy import MinecraftPolicy


class MinedojoActionAdapter(nn.Module):
    """
    适配VPT的MineRL action space到MineDojo
    
    MineRL: 复杂的hierarchical action space
    MineDojo: MultiDiscrete [3, 3, 4, 25, 25, 8, 244, 36]
    """
    
    def __init__(self, vpt_policy: MinecraftPolicy):
        super().__init__()
        self.vpt_policy = vpt_policy
        
        # MineDojo action dimensions
        self.minedojo_action_dim = [3, 3, 4, 25, 25, 8, 244, 36]
        
        # 创建action head将VPT的latent映射到MineDojo action space
        hidden_dim = 2048  # VPT的hidsize
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in self.minedojo_action_dim
        ])
        
        print(f"MinedojoActionAdapter创建:")
        print(f"  MineDojo action dims: {self.minedojo_action_dim}")
        print(f"  Action heads参数: {sum(p.numel() for p in self.action_heads.parameters()):,}")
    
    def forward(self, obs):
        """
        前向传播
        
        Args:
            obs: (B, H, W, C) 图像观察，范围[0, 1]
        
        Returns:
            action_logits: List of (B, action_dim[i])
        """
        batch_size = obs.shape[0]
        
        # 添加时间维度：(B, H, W, C) -> (B, T=1, H, W, C)
        # ImpalaCNN期望BHWC格式，会内部转换成BCHW
        obs_vpt = obs.unsqueeze(1)  # (B, 1, H, W, C)
        
        # 通过VPT的视觉encoder
        # img_preprocess: 归一化到[0, 1]（如果输入已经是[0,1]则不变）
        x = self.vpt_policy.img_preprocess(obs_vpt)  # (B, 1, H, W, C)
        
        # img_process: CNN特征提取 + linear
        # ImpalaCNN内部会reshape成 (B*T, C, H, W)，处理后reshape回 (B, T, feature_dim)
        x = self.vpt_policy.img_process(x)  # (B, 1, feature_dim)
        
        # 移除时间维度 (B, 1, feature_dim) -> (B, feature_dim)
        x = x.squeeze(1)
        
        # lastlayer: 映射到hidsize
        latent = self.vpt_policy.lastlayer(x)  # (B, hidsize=2048)
        
        # 通过MineDojo action heads
        action_logits = [head(latent) for head in self.action_heads]
        
        return action_logits


class ExpertDataset(Dataset):
    """专家数据集（从Web录制系统的数据格式加载）"""
    
    def __init__(self, expert_dir: str, target_size=(128, 128)):
        """
        Args:
            expert_dir: 专家演示目录（包含episode_000, episode_001等）
            target_size: 目标图像尺寸 (H, W)，VPT使用128x128
        """
        import cv2
        self.expert_dir = expert_dir
        self.target_size = target_size
        self.cv2 = cv2
        
        # 查找所有episode目录
        episode_dirs = []
        for item in sorted(os.listdir(expert_dir)):
            item_path = os.path.join(expert_dir, item)
            if os.path.isdir(item_path) and item.startswith('episode_'):
                episode_dirs.append(item_path)
        
        print(f"找到 {len(episode_dirs)} 个episode目录")
        
        # 预加载所有数据
        self.all_obs = []
        self.all_actions = []
        
        for ep_path in tqdm(episode_dirs, desc="Loading data"):
            # 查找所有frame文件
            frame_files = sorted([f for f in os.listdir(ep_path) 
                                if f.startswith('frame_') and f.endswith('.npy')])
            
            if len(frame_files) == 0:
                continue
            
            # 加载每个frame
            for frame_file in frame_files:
                frame_path = os.path.join(ep_path, frame_file)
                try:
                    # 加载.npy文件（包含observation和action）
                    data = np.load(frame_path, allow_pickle=True).item()
                    
                    # 提取观察和动作
                    obs = data['observation']  # RGB图像 (H, W, C)
                    action = data['action']    # MineDojo动作 (8,)
                    
                    self.all_obs.append(obs)
                    self.all_actions.append(action)
                    
                except Exception as e:
                    print(f"警告: 加载 {frame_path} 失败: {e}")
                    continue
        
        if len(self.all_obs) == 0:
            raise ValueError(f"未找到任何有效的训练数据！请检查目录: {expert_dir}")
        
        print(f"✓ 加载完成")
        print(f"  总样本数: {len(self.all_obs)}")
        print(f"  原始图像shape: {self.all_obs[0].shape}")
        print(f"  目标图像shape: {target_size}")
        print(f"  动作shape: {self.all_actions[0].shape}")
        print(f"  动作维度: {len(self.all_actions[0])}")
    
    def __len__(self):
        return len(self.all_obs)
    
    def __getitem__(self, idx):
        obs = self.all_obs[idx]  # 可能是 (C, H, W) 或 (H, W, C)
        action = self.all_actions[idx]  # (action_dim,)
        
        # 检查并转换为HWC格式
        if obs.shape[0] == 3 or obs.shape[0] == 1:  # 很可能是CHW格式
            if len(obs.shape) == 3 and obs.shape[0] < obs.shape[1] and obs.shape[0] < obs.shape[2]:
                # (C, H, W) -> (H, W, C)
                obs = np.transpose(obs, (1, 2, 0))
        
        # 确保是uint8类型
        if obs.dtype != np.uint8:
            if obs.max() <= 1.0:
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)
        
        # Resize图像到VPT期望的尺寸
        if obs.shape[:2] != self.target_size:
            obs = self.cv2.resize(obs, (self.target_size[1], self.target_size[0]), 
                                 interpolation=self.cv2.INTER_LINEAR)
        
        # 转换为tensor
        # 图像: (H, W, C) uint8 [0, 255] -> (H, W, C) float32 [0, 1]
        obs = th.from_numpy(obs).float() / 255.0
        
        # 动作: (action_dim,) -> long
        action = th.from_numpy(action).long()
        
        return obs, action


class BCTrainer:
    """BC训练器（使用VPT初始化）"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        device: str = "auto"
    ):
        if device == "auto":
            if th.cuda.is_available():
                device = "cuda"
            elif th.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = model.to(device)
        
        # 优化器 - 使用较小的学习率（因为是微调）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"BC Trainer初始化:")
        print(f"  设备: {device}")
        print(f"  学习率: {learning_rate}")
        print(f"  总参数: {param_count:,}")
        print(f"  可训练参数: {trainable_count:,}")
    
    def compute_loss(self, obs_batch, action_batch):
        """计算BC损失"""
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        
        # 前向传播
        action_logits = self.model(obs_batch)  # List of (B, action_dim[i])
        
        # 计算每个维度的损失
        losses = []
        for i, logits in enumerate(action_logits):
            target = action_batch[:, i]
            loss = self.criterion(logits, target)
            losses.append(loss)
        
        total_loss = sum(losses)
        return total_loss, losses
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = [0] * 8  # MineDojo action_dim
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for obs_batch, action_batch in pbar:
            # 计算损失
            loss, losses = self.compute_loss(obs_batch, action_batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_samples += obs_batch.size(0)
            
            # 计算准确率
            with th.no_grad():
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                action_logits = self.model(obs_batch)
                
                for i, logits in enumerate(action_logits):
                    pred = logits.argmax(dim=-1)
                    target = action_batch[:, i]
                    correct = (pred == target).sum().item()
                    total_correct[i] += correct
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (pbar.n + 1):.4f}"
            })
        
        # 计算平均指标
        avg_loss = total_loss / len(dataloader)
        accuracies = [correct / total_samples for correct in total_correct]
        avg_accuracy = np.mean(accuracies)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'accuracies': accuracies
        }
    
    def validate(self, dataloader: DataLoader) -> dict:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_correct = [0] * 8
        total_samples = 0
        
        with th.no_grad():
            for obs_batch, action_batch in tqdm(dataloader, desc="Validating"):
                loss, losses = self.compute_loss(obs_batch, action_batch)
                total_loss += loss.item()
                
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                action_logits = self.model(obs_batch)
                
                for i, logits in enumerate(action_logits):
                    pred = logits.argmax(dim=-1)
                    target = action_batch[:, i]
                    correct = (pred == target).sum().item()
                    total_correct[i] += correct
                
                total_samples += obs_batch.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracies = [correct / total_samples for correct in total_correct]
        avg_accuracy = np.mean(accuracies)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'accuracies': accuracies
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'vpt_weights_path': getattr(self, 'vpt_weights_path', None),
            'freeze_vpt': getattr(self, 'freeze_vpt', False)
        }
        th.save(checkpoint, path)
        print(f"✓ Checkpoint保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="VPT BC训练")
    
    # 数据参数
    parser.add_argument("--expert-dir", type=str, 
                       default="data/tasks/harvest_1_log/expert_demos",
                       help="专家演示目录")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="验证集比例")
    
    # VPT参数
    parser.add_argument("--vpt-weights", type=str,
                       default="data/pretrained/vpt/rl-from-early-game-2x.weights",
                       help="VPT weights文件")
    parser.add_argument("--no-pretrain", action="store_true",
                       help="不使用VPT预训练")
    parser.add_argument("--freeze-vpt", action="store_true",
                       help="冻结VPT视觉特征提取器（推荐，防止灾难性遗忘）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20,
                       help="训练epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="学习率（冻结VPT时建议用1e-4，全参数微调时建议用1e-5）")
    parser.add_argument("--device", type=str, default="auto",
                       help="训练设备")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="数据加载线程数")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str,
                       default="data/tasks/harvest_1_log/vpt_bc_model",
                       help="最终模型保存目录（best_model.pth, final_model.pth, config等）")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Checkpoint保存目录（如不指定，则使用output-dir）。可指向大容量磁盘")
    parser.add_argument("--save-freq", type=int, default=5,
                       help="保存checkpoint频率（设为0则不保存checkpoint，只保存best/final）")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                       help="保留的checkpoint数量（防止磁盘占用过大）")
    
    args = parser.parse_args()
    
    # 如果未指定checkpoint-dir，则使用output-dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output_dir
    
    print("=" * 70)
    print("🚀 VPT BC训练")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.output_dir, "train_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"配置保存: {config_path}")
    print(f"模型输出目录: {args.output_dir}")
    print(f"Checkpoint目录: {args.checkpoint_dir}")
    if args.checkpoint_dir != args.output_dir:
        print(f"  ℹ️  Checkpoint使用独立目录（大容量磁盘）")
    print()
    
    # 加载数据
    print("📂 加载专家数据...")
    full_dataset = ExpertDataset(args.expert_dir)
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=th.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  Batch大小: {args.batch_size}\n")
    
    # 创建VPT policy
    print("🤖 创建VPT Policy...")
    if not args.no_pretrain:
        print(f"  加载VPT预训练权重: {args.vpt_weights}")
        vpt_policy, result = load_vpt_policy(args.vpt_weights, device='cpu', verbose=False)
        print(f"  ✓ VPT权重加载成功 (Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)})")
    else:
        print("  使用随机初始化（--no-pretrain）")
        vpt_policy = create_vpt_policy(device='cpu')
    
    # 创建MineDojo适配器
    print("\n🔄 创建MineDojo适配器...")
    model = MinedojoActionAdapter(vpt_policy)
    
    # 参数冻结策略（防止灾难性遗忘）
    if not args.no_pretrain and args.freeze_vpt:
        print("\n❄️  冻结VPT参数...")
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            # 策略：冻结所有vpt_policy参数，只训练action_heads
            if 'vpt_policy' in name:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                # action_heads保持可训练
                trainable_params += param.numel()
        
        print(f"  冻结参数: {frozen_params:,} ({frozen_params/(frozen_params+trainable_params)*100:.1f}%)")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/(frozen_params+trainable_params)*100:.1f}%)")
        print(f"  策略: 冻结整个VPT模型，只训练MineDojo action heads")
        print(f"  优势: 完全保留VPT预训练知识（跳跃、移动、战斗等），只学习动作映射")
    
    # 创建训练器
    print("\n🎓 创建BC训练器...")
    trainer = BCTrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # 保存训练配置到trainer（用于checkpoint）
    trainer.vpt_weights_path = args.vpt_weights if not args.no_pretrain else None
    trainer.freeze_vpt = args.freeze_vpt
    
    # 训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # 验证
        val_metrics = trainer.validate(val_loader)
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # 详细准确率
        print(f"  各维度准确率: {[f'{acc:.3f}' for acc in val_metrics['accuracies']]}")
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = os.path.join(args.output_dir, "best_model.pth")
            trainer.save_checkpoint(best_path, epoch, val_metrics)
            print(f"  ✓ 新的最佳模型！")
        
        # 定期保存checkpoint（只保留最新的N个）
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            trainer.save_checkpoint(ckpt_path, epoch, val_metrics)
            
            # 清理旧的checkpoint，只保留最新的N个
            checkpoint_files = sorted([
                f for f in os.listdir(args.checkpoint_dir) 
                if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
            ], key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # 删除旧的checkpoint（保留最新N个）
            if len(checkpoint_files) > args.keep_checkpoints:
                for old_ckpt in checkpoint_files[:-args.keep_checkpoints]:
                    old_path = os.path.join(args.checkpoint_dir, old_ckpt)
                    try:
                        os.remove(old_path)
                        print(f"  🗑️  删除旧checkpoint: {old_ckpt}")
                    except Exception as e:
                        print(f"  ⚠️  删除失败: {e}")
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model.pth")
    trainer.save_checkpoint(final_path, args.epochs, val_metrics)
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, "train_history.yaml")
    with open(history_path, 'w') as f:
        yaml.dump(train_history, f, default_flow_style=False)
    
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print(f"最佳验证loss: {best_val_loss:.4f}")
    print(f"模型保存在: {args.output_dir}/")
    print("\n下一步:")
    print(f"  python src/training/evaluate_bc_vpt.py --model {args.output_dir}/best_model.pth")


if __name__ == "__main__":
    main()
