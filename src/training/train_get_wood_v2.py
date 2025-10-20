#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获得木头任务训练脚本 (MineCLIP 版本)

使用官方 MineCLIP 包和预训练权重
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

try:
    import minedojo
    MINEDOJO_AVAILABLE = True
except ImportError:
    MINEDOJO_AVAILABLE = False
    print("❌ MineDojo未安装")
    sys.exit(1)

from src.utils.realtime_logger import RealtimeLoggerCallback
from src.utils.env_wrappers import make_minedojo_env
from src.utils.mineclip_reward import MineCLIPRewardWrapper


def _detect_device(device_arg):
    """检测可用设备"""
    if device_arg != 'auto':
        return device_arg
    
    if torch.cuda.is_available():
        print("🚀 检测到CUDA GPU")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 检测到Apple Silicon MPS")
        return 'mps'
    else:
        print("💻 使用CPU")
        return 'cpu'


def create_harvest_log_env(use_mineclip=False, mineclip_model_path=None, 
                          mineclip_variant="attn", image_size=(160, 256)):
    """
    创建采集木头任务环境
    
    Args:
        use_mineclip: 是否使用MineCLIP密集奖励
        mineclip_model_path: MineCLIP模型权重路径
        mineclip_variant: MineCLIP变体 ("attn" 或 "avg")
        image_size: 图像尺寸
        
    Returns:
        MineDojo环境
    """
    print(f"创建环境: harvest_1_log (获得1个原木)")
    print(f"  图像尺寸: {image_size}")
    print(f"  MineCLIP: {'启用 (' + mineclip_variant + ')' if use_mineclip else '禁用'}")
    
    # 使用 env_wrappers 创建基础环境
    env = make_minedojo_env(
        task_id="harvest_1_log",
        image_size=image_size,
        use_frame_stack=False,
        use_discrete_actions=False
    )
    
    # 如果启用MineCLIP
    if use_mineclip:
        env = MineCLIPRewardWrapper(
            env,
            task_prompt="chop down a tree and collect one wood log",
            model_path=mineclip_model_path,
            variant=mineclip_variant,
            sparse_weight=10.0,
            mineclip_weight=0.1,
            device='auto'
        )
    
    return env


def train(args):
    """主训练函数"""
    
    # 检测设备
    device = _detect_device(args.device)
    
    print("=" * 70)
    print(f"MineDojo 获得木头训练 (harvest_1_log)")
    print("=" * 70)
    print(f"配置:")
    print(f"  总步数: {args.total_timesteps:,}")
    print(f"  设备: {device}")
    print(f"  MineCLIP: {'启用' if args.use_mineclip else '禁用'}")
    if args.use_mineclip:
        print(f"  MineCLIP模型: {args.mineclip_model}")
        print(f"  MineCLIP变体: {args.mineclip_variant}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  图像尺寸: {args.image_size}")
    print("=" * 70)
    print()
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建环境
    print("[1/4] 创建环境...")
    env_instance = create_harvest_log_env(
        use_mineclip=args.use_mineclip,
        mineclip_model_path=args.mineclip_model if args.use_mineclip else None,
        mineclip_variant=args.mineclip_variant,
        image_size=args.image_size
    )
    env = DummyVecEnv([lambda: env_instance])
    print("  ✓ 环境创建成功")
    print()
    
    # 创建模型
    print("[2/4] 创建PPO模型...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device=device,
        verbose=1,
        tensorboard_log=args.tensorboard_dir
    )
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  ✓ 模型创建成功 (参数量: {total_params:,})")
    print()
    
    # 设置回调
    print("[3/4] 设置训练回调...")
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_dir,
        name_prefix="get_wood"
    )
    
    logger_callback = RealtimeLoggerCallback(
        log_dir=args.log_dir,
        log_interval=100
    )
    
    callbacks = CallbackList([checkpoint_callback, logger_callback])
    print("  ✓ 回调设置完成")
    print()
    
    # 开始训练
    print("[4/4] 开始训练...")
    print("=" * 70)
    print()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(args.checkpoint_dir, "get_wood_final.zip")
        model.save(final_model_path)
        print(f"\n最终模型已保存: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n训练被中断")
        interrupted_path = os.path.join(args.checkpoint_dir, "get_wood_interrupted.zip")
        model.save(interrupted_path)
        print(f"当前模型已保存: {interrupted_path}")
    
    finally:
        env.close()
        print("环境已关闭")


def main():
    parser = argparse.ArgumentParser(description="MineDojo 获得木头训练 (MineCLIP版)")
    
    # 训练参数
    parser.add_argument('--total-timesteps', type=int, default=200000,
                       help='总训练步数')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='训练设备')
    
    # MineCLIP 参数
    parser.add_argument('--use-mineclip', action='store_true',
                       help='启用MineCLIP密集奖励')
    parser.add_argument('--mineclip-model', type=str, default='data/attn.pth',
                       help='MineCLIP模型权重路径')
    parser.add_argument('--mineclip-variant', type=str, default='attn',
                       choices=['attn', 'avg'],
                       help='MineCLIP变体')
    
    # 环境参数
    parser.add_argument('--image-size', type=int, nargs=2, default=[160, 256],
                       help='图像尺寸 (height width)')
    
    # 保存参数
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='保存检查点频率')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/get_wood',
                       help='检查点保存目录')
    parser.add_argument('--tensorboard-dir', type=str, default='logs/tensorboard',
                       help='TensorBoard日志目录')
    parser.add_argument('--log-dir', type=str, default='logs/training',
                       help='训练日志目录')
    
    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    
    train(args)


if __name__ == "__main__":
    main()
