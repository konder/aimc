#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
harvest_1_paper 任务训练脚本 (MVP版本)

使用PPO算法从头开始训练MineDojo智能体完成收集纸的任务

用法:
    python src/training/train_harvest_paper.py
    python src/training/train_harvest_paper.py --total-timesteps 1000000
"""

import os
import sys
import argparse
from datetime import datetime
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)

from src.utils.env_wrappers import make_minedojo_env


def _detect_device(device_arg):
    """
    检测并返回可用的设备
    
    Args:
        device_arg: 用户指定的设备参数
        
    Returns:
        str: 实际使用的设备
    """
    if device_arg != 'auto':
        return device_arg
    
    # 自动检测
    if torch.cuda.is_available():
        print("🚀 检测到 CUDA GPU，使用 cuda")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 检测到 Apple Silicon，使用 MPS 加速")
        return 'mps'
    else:
        print("💻 使用 CPU")
        return 'cpu'


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, 
            f"training_{timestamp}.log"
        )
        
        self.log("=" * 70)
        self.log("MineDojo harvest_1_paper 训练开始")
        self.log("=" * 70)
    
    def log(self, message):
        """
        记录日志
        
        Args:
            message: 日志消息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')


def create_env(task_id, image_size, rank=0):
    """
    创建环境的工厂函数
    
    Args:
        task_id: 任务ID
        image_size: 图像尺寸
        rank: 进程编号（用于并行环境）
        
    Returns:
        callable: 返回环境的函数
    """
    def _init():
        env = make_minedojo_env(
            task_id=task_id,
            image_size=image_size,
            use_frame_stack=False,  # MVP版本不使用帧堆叠
            use_discrete_actions=False  # 使用原始MultiDiscrete空间
        )
        # 注意: 不使用Monitor以避免gym/gymnasium兼容性问题
        # PPO已经有内置的日志记录功能
        return env
    return _init


def train(args):
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    # 检测并设置设备
    device = _detect_device(args.device)
    args.device = device
    
    # 初始化日志
    logger = TrainingLogger(args.log_dir)
    logger.log(f"训练配置:")
    logger.log(f"  任务: {args.task_id}")
    logger.log(f"  总步数: {args.total_timesteps:,}")
    logger.log(f"  并行环境数: {args.n_envs}")
    logger.log(f"  学习率: {args.learning_rate}")
    logger.log(f"  图像尺寸: {args.image_size}")
    logger.log(f"  设备: {args.device}")
    logger.log("")
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # ========================================================================
    # 1. 创建训练环境
    # ========================================================================
    logger.log("[1/5] 创建训练环境...")
    
    if args.n_envs > 1:
        # 多个并行环境
        logger.log(f"  使用 {args.n_envs} 个并行环境加速训练")
        env = SubprocVecEnv([
            create_env(args.task_id, args.image_size, i) 
            for i in range(args.n_envs)
        ])
    else:
        # 单个环境
        env = DummyVecEnv([create_env(args.task_id, args.image_size)])
    
    logger.log("  ✓ 训练环境创建成功")
    
    # ========================================================================
    # 2. 创建评估环境
    # ========================================================================
    # 注意: MineDojo在创建多个相同任务的环境时有bug，暂时禁用独立评估环境
    # logger.log("[2/5] 创建评估环境...")
    # eval_env = DummyVecEnv([create_env(args.task_id, args.image_size)])
    # logger.log("  ✓ 评估环境创建成功")
    eval_env = None  # 暂时禁用
    logger.log("[2/5] 跳过评估环境创建（避免MineDojo多环境bug）")
    
    # ========================================================================
    # 3. 创建PPO模型
    # ========================================================================
    logger.log("[3/5] 创建PPO模型...")
    logger.log("  注意: 模型从随机初始化开始（无预训练权重）")
    
    # PPO超参数
    # 注意: normalize_images=False 因为我们已经在wrapper中归一化了
    policy_kwargs = dict(
        normalize_images=False  # 重要！我们已经归一化了图像
    )
    
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=args.tensorboard_dir,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.log(f"  ✓ 模型创建成功 (参数量: {n_params:,})")
    
    # ========================================================================
    # 4. 设置回调函数
    # ========================================================================
    logger.log("[4/5] 设置训练回调...")
    
    # 检查点回调：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_dir,
        name_prefix="harvest_paper",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # 评估回调：定期评估模型性能
    # 注意: 由于禁用了独立评估环境，暂时不使用 EvalCallback
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=args.checkpoint_dir,
    #     log_path=args.log_dir,
    #     eval_freq=args.eval_freq,
    #     n_eval_episodes=args.n_eval_episodes,
    #     deterministic=True,
    #     render=False,
    # )
    
    # 组合回调（暂时只使用检查点回调）
    callback = checkpoint_callback
    
    logger.log("  ✓ 回调设置完成")
    logger.log(f"    - 每 {args.save_freq} 步保存检查点")
    # logger.log(f"    - 每 {args.eval_freq} 步评估模型")
    logger.log("")
    
    # ========================================================================
    # 5. 开始训练
    # ========================================================================
    logger.log("[5/5] 开始训练...")
    logger.log("=" * 70)
    logger.log("")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,  # 每10次rollout打印一次
            tb_log_name="ppo_harvest_paper",
            reset_num_timesteps=True,
        )
        
        # 保存最终模型
        final_model_path = os.path.join(
            args.checkpoint_dir, 
            "harvest_paper_final.zip"
        )
        model.save(final_model_path)
        
        logger.log("")
        logger.log("=" * 70)
        logger.log("✓ 训练完成!")
        logger.log(f"✓ 最终模型已保存到: {final_model_path}")
        logger.log("=" * 70)
        
    except KeyboardInterrupt:
        logger.log("")
        logger.log("=" * 70)
        logger.log("训练被用户中断")
        logger.log("=" * 70)
        
        # 保存中断时的模型
        interrupted_path = os.path.join(
            args.checkpoint_dir, 
            "harvest_paper_interrupted.zip"
        )
        model.save(interrupted_path)
        logger.log(f"当前模型已保存到: {interrupted_path}")
    
    finally:
        # 清理资源
        env.close()
        if eval_env is not None:
            eval_env.close()
        logger.log("环境已关闭")


def evaluate(args):
    """
    评估已训练的模型
    
    Args:
        args: 命令行参数
    """
    print("=" * 70)
    print("模型评估")
    print("=" * 70)
    
    # 加载模型
    print(f"\n[1/3] 加载模型: {args.model_path}")
    model = PPO.load(args.model_path)
    print("  ✓ 模型加载成功")
    
    # 创建环境
    print("\n[2/3] 创建环境...")
    env = make_minedojo_env(
        task_id=args.task_id,
        image_size=args.image_size,
        use_discrete_actions=False  # 使用原始MultiDiscrete空间
    )
    print("  ✓ 环境创建成功")
    
    # 评估
    print(f"\n[3/3] 运行 {args.n_eval_episodes} 个评估episodes...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(args.n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done and episode_length < args.max_episode_steps:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if reward > 0:
                success_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {episode + 1}/{args.n_eval_episodes}: "
              f"reward={episode_reward:.2f}, steps={episode_length}")
    
    # 统计结果
    print("\n" + "=" * 70)
    print("评估结果:")
    print("=" * 70)
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± "
          f"{np.std(episode_rewards):.2f}")
    print(f"平均步数: {np.mean(episode_lengths):.1f} ± "
          f"{np.std(episode_lengths):.1f}")
    print(f"成功率: {success_count}/{args.n_eval_episodes} "
          f"({100*success_count/args.n_eval_episodes:.1f}%)")
    print("=" * 70)
    
    env.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MineDojo harvest_1_paper 训练脚本"
    )
    
    # 模式选择
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train',
        choices=['train', 'eval'],
        help='运行模式: train=训练, eval=评估'
    )
    
    # 任务配置
    parser.add_argument(
        '--task-id',
        type=str,
        default='harvest_milk',  # 默认使用harvest_milk（更稳定）
        help='MineDojo任务ID'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[160, 256],
        help='图像尺寸 (height width)'
    )
    
    # 训练参数
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=500000,
        help='总训练步数'
    )
    parser.add_argument(
        '--n-envs',
        type=int,
        default=1,
        help='并行环境数量'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='学习率'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2048,
        help='每次更新收集的步数'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='批次大小'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=10,
        help='每次更新的epoch数'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='折扣因子'
    )
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='GAE lambda参数'
    )
    parser.add_argument(
        '--clip-range',
        type=float,
        default=0.2,
        help='PPO裁剪范围'
    )
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.01,
        help='熵系数'
    )
    parser.add_argument(
        '--vf-coef',
        type=float,
        default=0.5,
        help='价值函数系数'
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='梯度裁剪阈值'
    )
    
    # 保存和评估
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10000,
        help='保存检查点的频率'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=10000,
        help='评估模型的频率'
    )
    parser.add_argument(
        '--n-eval-episodes',
        type=int,
        default=5,
        help='评估的episode数量'
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=2000,
        help='每个episode的最大步数'
    )
    
    # 路径配置
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/training',
        help='日志目录'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/harvest_paper',
        help='检查点保存目录'
    )
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default='logs/tensorboard',
        help='TensorBoard日志目录'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='checkpoints/harvest_paper/best_model.zip',
        help='评估时加载的模型路径'
    )
    
    # 设备配置
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='训练设备 (auto/cpu/cuda/mps)'
    )
    
    args = parser.parse_args()
    
    # 转换image_size为元组
    args.image_size = tuple(args.image_size)
    
    # 运行
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

