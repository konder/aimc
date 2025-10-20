#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineDojo 环境包装器
用于适配Stable-Baselines3的训练需求
"""

import gym
import numpy as np
from collections import deque


class MinedojoWrapper(gym.Wrapper):
    """
    MineDojo环境包装器，将复杂的观察空间简化为适合训练的格式
    
    主要功能:
    1. 只使用RGB图像作为观察
    2. 归一化图像到[0, 1]
    
    注意: MineDojo已经返回channel-first格式(C, H, W)，不需要转置
    """
    
    def __init__(self, env):
        """
        初始化包装器
        
        Args:
            env: MineDojo环境实例
        """
        super().__init__(env)
        
        # 获取原始RGB图像形状 - MineDojo返回(C, H, W)格式
        orig_shape = env.observation_space['rgb'].shape
        c, h, w = orig_shape  # 注意: MineDojo使用channel-first
        
        # 定义新的观察空间: (C, H, W), 范围[0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, h, w),
            dtype=np.float32
        )
        
    def reset(self, **kwargs):
        """
        重置环境
        
        Returns:
            np.ndarray: 处理后的RGB图像
        """
        # MineDojo 的 reset 不接受参数
        obs_dict = self.env.reset()
        return self._process_obs(obs_dict)
    
    def step(self, action):
        """
        执行一步
        
        Args:
            action: 动作
            
        Returns:
            tuple: (观察, 奖励, 完成标志, 信息)
        """
        obs_dict, reward, done, info = self.env.step(action)
        obs = self._process_obs(obs_dict)
        return obs, reward, done, info
    
    def _process_obs(self, obs_dict):
        """
        处理观察数据
        
        Args:
            obs_dict: MineDojo返回的观察字典
            
        Returns:
            np.ndarray: 处理后的RGB图像 (C, H, W)
        """
        # 提取RGB图像 - MineDojo已经是(C, H, W)格式
        rgb = obs_dict['rgb']  # (C, H, W)
        
        # 归一化到[0, 1]
        rgb = rgb.astype(np.float32) / 255.0
        
        # 不需要转置 - MineDojo已经是channel-first
        return rgb


class FrameStack(gym.Wrapper):
    """
    帧堆叠包装器，将连续多帧堆叠作为观察
    这有助于模型学习时序信息（如物体移动方向）
    """
    
    def __init__(self, env, n_frames=4):
        """
        初始化帧堆叠包装器
        
        Args:
            env: 环境实例
            n_frames: 堆叠的帧数
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        # 更新观察空间维度
        obs_space = env.observation_space
        low = np.repeat(obs_space.low, n_frames, axis=0)
        high = np.repeat(obs_space.high, n_frames, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=obs_space.dtype
        )
    
    def reset(self, **kwargs):
        """重置环境并初始化帧缓存"""
        # MineDojo 的 reset 不接受参数
        obs = self.env.reset()
        # 用初始观察填充所有帧
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, action):
        """执行一步并更新帧缓存"""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        """获取堆叠后的观察"""
        # 沿通道维度堆叠
        return np.concatenate(list(self.frames), axis=0)


class ActionWrapper(gym.Wrapper):
    """
    简化MineDojo的动作空间
    MineDojo使用MultiDiscrete空间（8维数组），我们直接使用原始空间
    或者提供一个简化的离散动作映射
    """
    
    def __init__(self, env, use_discrete=False):
        """
        初始化动作包装器
        
        Args:
            env: 环境实例
            use_discrete: 是否使用简化的离散动作空间（暂不支持）
        """
        super().__init__(env)
        self.use_discrete = use_discrete
        
        # MVP版本：直接使用原始的MultiDiscrete空间
        # 未来可以添加离散化映射
        if use_discrete:
            # TODO: 实现离散动作映射
            # 目前直接使用原始空间
            print("警告: 离散动作空间映射尚未实现，使用原始MultiDiscrete空间")
            self.use_discrete = False
    
    def step(self, action):
        """
        执行动作，捕获无效动作错误
        
        Args:
            action: MultiDiscrete动作数组
            
        Returns:
            tuple: (观察, 奖励, 完成标志, 信息)
        """
        try:
            return self.env.step(action)
        except ValueError as e:
            # 捕获无效动作错误（如"Trying to place air"）
            # 返回当前观察和0奖励，不结束episode
            if "place air" in str(e) or "strict check" in str(e):
                # 使用no-op动作重试
                noop_action = self.env.action_space.no_op()
                return self.env.step(noop_action)
            else:
                # 其他错误继续抛出
                raise
    
    def reset(self, **kwargs):
        """重置环境"""
        # MineDojo 的 reset 不接受参数
        return self.env.reset()


def make_minedojo_env(task_id, image_size=(160, 256), use_frame_stack=False,
                      frame_stack_n=4, use_discrete_actions=False):
    """
    创建并包装MineDojo环境
    
    Args:
        task_id: 任务ID
        image_size: 图像尺寸 (height, width)
        use_frame_stack: 是否使用帧堆叠
        frame_stack_n: 堆叠帧数
        use_discrete_actions: 是否使用离散动作空间（暂不支持，使用MultiDiscrete）
        
    Returns:
        gym.Env: 包装后的环境
        
    Note:
        MineDojo使用MultiDiscrete(8)动作空间，直接由RL算法处理
    """
    import minedojo
    
    # 创建基础环境
    import random
    seed = random.randint(0, 2**31 - 1)
    
    env = minedojo.make(
        task_id=task_id,
        image_size=image_size,
        seed=seed,
        fast_reset=True,
    )
    
    # 应用包装器
    env = MinedojoWrapper(env)
    env = ActionWrapper(env, use_discrete=use_discrete_actions)
    
    if use_frame_stack:
        env = FrameStack(env, n_frames=frame_stack_n)
    
    return env

