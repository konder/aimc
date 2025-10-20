#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时训练日志回调
每 N 步打印当前训练状态
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time


class RealtimeLoggerCallback(BaseCallback):
    """
    实时日志回调：每 N 步打印训练状态
    
    Args:
        log_freq: 日志打印频率（步数）
        verbose: 是否打印详细信息
    """
    
    def __init__(self, log_freq=100, verbose=1):
        super(RealtimeLoggerCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []  # 完整episode的奖励列表
        self.episode_lengths = []  # 完整episode的长度列表
        # 当前步的值（不再累积）
        self.current_reward = 0.0
        self.current_mineclip_reward = 0.0
        self.current_similarity = 0.0
        self.current_mineclip_weight = 0.0
        self.sparse_weight = None
        self.current_episode = 0  # 当前回合数
        self.start_time = None
        self.last_log_step = 0
        
    def _on_training_start(self):
        """训练开始时调用"""
        self.start_time = time.time()
        print("\n" + "=" * 130)
        print("🚀 开始训练...")
        print("=" * 130)
        print(f"{'回合数':>8s} | {'步数':>10s} | {'总时间':>10s} | {'FPS':>8s} | "
              f"{'总奖励':>10s} | {'MineCLIP':>10s} | {'MC权重':>8s} | {'权重比':>8s} | {'相似度':>8s} | {'损失':>10s}")
        print("-" * 130)
        
    def _on_step(self) -> bool:
        """
        每步调用
        Returns:
            bool: 如果返回 False，训练将停止
        """
        # 只记录当前步的值（不累积）
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            if isinstance(rewards, np.ndarray):
                self.current_reward = float(rewards[0])  # 取第一个环境
            else:
                self.current_reward = float(rewards)
        
        # 从info中提取MineCLIP详细信息（只记录当前步）
        if 'infos' in self.locals:
            infos = self.locals['infos']
            # 处理多环境情况
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0]  # 取第一个环境
            elif isinstance(infos, dict):
                info = infos
            else:
                info = {}
            
            if isinstance(info, dict):
                self.current_mineclip_reward = float(info.get('mineclip_reward', 0.0))
                self.current_similarity = float(info.get('mineclip_similarity', 0.0))
                self.current_mineclip_weight = float(info.get('mineclip_weight', 0.0))
                if 'sparse_weight' in info and self.sparse_weight is None:
                    self.sparse_weight = float(info['sparse_weight'])
        
        # 检查是否需要打印日志
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_progress()
            self.last_log_step = self.num_timesteps
        
        return True
    
    def _on_rollout_end(self):
        """Rollout 结束时调用"""
        # 收集 episode 信息
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                    self.current_episode += 1  # 完成一个回合
                if 'l' in ep_info:
                    self.episode_lengths.append(ep_info['l'])
    
    def _log_progress(self):
        """打印当前训练进度"""
        # 计算时间和 FPS
        elapsed_time = time.time() - self.start_time
        fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        # 格式化时间
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # 使用当前步的值（不再计算平均）
        mean_reward = self.current_reward
        mean_mineclip = self.current_mineclip_reward
        mean_similarity = self.current_similarity
        current_weight = self.current_mineclip_weight
        
        # 计算权重比 (sparse_weight / mineclip_weight)
        if self.sparse_weight is not None and current_weight > 0:
            weight_ratio = self.sparse_weight / current_weight
        else:
            weight_ratio = 0.0
        
        # 获取损失信息（如果有）
        loss_str = "N/A"
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                # 尝试获取最近的损失值
                if hasattr(self.model.logger, 'name_to_value'):
                    losses = []
                    for key in ['train/loss', 'train/policy_loss', 'train/value_loss']:
                        if key in self.model.logger.name_to_value:
                            losses.append(self.model.logger.name_to_value[key])
                    if losses:
                        loss_str = f"{np.mean(losses):.4f}"
            except:
                pass
        
        # 打印日志（包含回合数和MineCLIP详细信息）
        mineclip_str = f"{mean_mineclip:>10.4f}" if mean_mineclip != 0.0 else "N/A".rjust(10)
        weight_str = f"{current_weight:>8.4f}" if current_weight != 0.0 else "N/A".rjust(8)
        ratio_str = f"{weight_ratio:>8.2f}" if weight_ratio > 0 else "N/A".rjust(8)
        similarity_str = f"{mean_similarity:>8.4f}" if mean_similarity != 0.0 else "N/A".rjust(8)
        
        print(f"{self.current_episode:>8,} | {self.num_timesteps:>10,} | {time_str:>10s} | {fps:>8.1f} | "
              f"{mean_reward:>10.4f} | {mineclip_str} | {weight_str} | {ratio_str} | {similarity_str} | {loss_str:>10s}")
    
    def _on_training_end(self):
        """训练结束时调用"""
        print("-" * 130)
        print("✓ 训练完成")
        
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"总时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"总步数: {self.num_timesteps:,}")
        
        if len(self.episode_rewards) > 0:
            print(f"平均奖励: {np.mean(self.episode_rewards):.2f}")
            print(f"最佳奖励: {np.max(self.episode_rewards):.2f}")
        
        print("=" * 80)


class DetailedLoggerCallback(BaseCallback):
    """
    详细日志回调：打印更详细的训练信息
    包括各种损失、学习率、熵等
    
    Args:
        log_freq: 日志打印频率（步数）
    """
    
    def __init__(self, log_freq=100):
        super(DetailedLoggerCallback, self).__init__()
        self.log_freq = log_freq
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_details()
            self.last_log_step = self.num_timesteps
        return True
    
    def _log_details(self):
        """打印详细训练信息"""
        print(f"\n[步数 {self.num_timesteps:,}]")
        
        # 从 logger 中获取所有可用信息
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                if hasattr(self.model.logger, 'name_to_value'):
                    name_to_value = self.model.logger.name_to_value
                    
                    # 打印训练指标
                    if any('train/' in k for k in name_to_value.keys()):
                        print("  训练指标:")
                        for key, value in sorted(name_to_value.items()):
                            if key.startswith('train/'):
                                metric_name = key.replace('train/', '')
                                print(f"    {metric_name:20s}: {value:.6f}")
                    
                    # 打印 rollout 指标
                    if any('rollout/' in k for k in name_to_value.keys()):
                        print("  Rollout 指标:")
                        for key, value in sorted(name_to_value.items()):
                            if key.startswith('rollout/'):
                                metric_name = key.replace('rollout/', '')
                                print(f"    {metric_name:20s}: {value:.2f}")
            except:
                pass

