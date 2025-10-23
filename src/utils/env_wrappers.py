#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineDojo 环境包装器
用于适配Stable-Baselines3的训练需求
"""

import gym
import numpy as np
from collections import deque
from src.utils.camera_wrappers import CameraSmoothingWrapper


class TimeLimitWrapper(gym.Wrapper):
    """
    添加每回合最大步数限制
    
    MineDojo的harvest任务默认没有超时机制，会导致：
    - 一个回合可能永远不结束（除非完成任务）
    - agent死亡也不会结束回合
    - 训练效率极低
    
    这个wrapper确保每个回合在max_steps步后强制结束
    """
    
    def __init__(self, env, max_steps=1000):
        """
        Args:
            env: 环境实例
            max_steps: 每回合最大步数（默认1000）
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.current_steps = 0
        
    def reset(self, **kwargs):
        """重置环境并重置步数计数器"""
        self.current_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """执行一步并检查是否超时"""
        obs, reward, done, info = self.env.step(action)
        self.current_steps += 1
        
        # 如果达到最大步数，强制结束回合
        if self.current_steps >= self.max_steps:
            done = True
            info['TimeLimit.truncated'] = True  # 标记为超时结束
        
        return obs, reward, done, info


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
                      frame_stack_n=4, use_discrete_actions=False, max_episode_steps=1000,
                      use_camera_smoothing=True, max_camera_change=12.0, fast_reset=True):
    """
    创建并包装MineDojo环境
    
    Args:
        task_id: 任务ID
        image_size: 图像尺寸 (height, width)
        use_frame_stack: 是否使用帧堆叠
        frame_stack_n: 堆叠帧数
        use_discrete_actions: 是否使用离散动作空间（暂不支持，使用MultiDiscrete）
        max_episode_steps: 每回合最大步数（默认1000）
                          MineDojo的harvest任务默认没有超时，必须手动设置
        use_camera_smoothing: 是否启用相机平滑（减少抖动，默认True）
        max_camera_change: 相机最大角度变化（度/步，默认12.0）
        fast_reset: 是否快速重置（默认True）
                   True: 重用世界，reset快但环境相同
                   False: 重新生成世界，reset慢但环境多样
        
    Returns:
        gym.Env: 包装后的环境
        
    Note:
        - MineDojo使用MultiDiscrete(8)动作空间，直接由RL算法处理
        - 无头模式通过外部JAVA_OPTS环境变量控制（在Shell脚本中设置）
        - harvest任务不会在agent死亡时结束，必须用TimeLimit wrapper
        - 相机平滑能显著减少视觉抖动，提升训练效率2-3倍
        - 评估时建议fast_reset=False，确保每个episode环境不同
    """
    import minedojo
    
    # 创建基础环境
    import random
    seed = random.randint(0, 2**31 - 1)
    
    env = minedojo.make(
        task_id=task_id,
        image_size=image_size,
        seed=seed,
        fast_reset=fast_reset,
    )
    
    # 应用包装器（顺序重要！）
    
    # 1. 任务特定Wrapper（harvest_log, hunt, craft等）
    from src.utils.task_wrappers import apply_task_wrapper
    env = apply_task_wrapper(env, task_id, verbose=True)
    
    # 2. 简化观察空间
    env = MinedojoWrapper(env)
    
    # 3. 添加超时限制（关键！MineDojo harvest任务默认没有超时）
    env = TimeLimitWrapper(env, max_steps=max_episode_steps)
    
    # 4. 处理动作空间
    env = ActionWrapper(env, use_discrete=use_discrete_actions)
    
    # 5. 相机平滑（可选，减少视觉抖动）
    if use_camera_smoothing:
        env = CameraSmoothingWrapper(
            env,
            max_pitch_change=max_camera_change,
            max_yaw_change=max_camera_change
        )
    
    # 6. 可选：帧堆叠
    if use_frame_stack:
        env = FrameStack(env, n_frames=frame_stack_n)
    
    # 注意：不在这里添加Monitor！
    # Monitor必须在MineCLIPRewardWrapper之后添加（如果使用的话）
    # 由train脚本负责在最外层添加Monitor
    
    return env

