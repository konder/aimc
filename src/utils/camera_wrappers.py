#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相机控制优化包装器
解决训练过程中镜头大幅抖动的问题
"""

import gym
import numpy as np


class CameraSmoothingWrapper(gym.Wrapper):
    """
    相机平滑包装器
    
    限制相邻步之间的相机角度变化，避免大幅抖动
    
    Args:
        env: 环境实例
        max_pitch_change: 最大pitch变化量（度/步）
        max_yaw_change: 最大yaw变化量（度/步）
        camera_indices: 相机动作在动作空间中的索引 (pitch_idx, yaw_idx)
    """
    
    def __init__(self, env, max_pitch_change=15.0, max_yaw_change=15.0, 
                 camera_indices=(3, 4)):  # ✅ 修正：MineDojo中相机是索引3,4
        super().__init__(env)
        self.max_pitch_change = max_pitch_change
        self.max_yaw_change = max_yaw_change
        self.pitch_idx, self.yaw_idx = camera_indices
        
        # MineDojo相机动作空间: [0-24], 12是中心(不动)
        # 每个离散值约等于10度角度变化
        self.camera_center = 12
        self.degrees_per_unit = 10.0  # 每个离散单位约10度
        
        # 记录上一步的相机动作
        self.last_pitch_action = None
        self.last_yaw_action = None
        
        # 打印配置
        print(f"  📷 相机平滑: 启用 (索引: pitch={self.pitch_idx}, yaw={self.yaw_idx})")
        print(f"     最大Pitch变化: ±{max_pitch_change}度/步")
        print(f"     最大Yaw变化: ±{max_yaw_change}度/步")
    
    def reset(self, **kwargs):
        """重置环境并重置相机状态"""
        # 重置为中心值，这样第一步可以自由探索
        # 而不是被随机的初始相机角度锁定
        self.last_pitch_action = self.camera_center
        self.last_yaw_action = self.camera_center
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        限制相机动作的变化幅度
        
        MineDojo相机动作是离散值[0-24]，12是中心(不动)
        我们将其映射到角度，限制变化，再映射回离散值
        """
        # 复制动作（避免修改原始动作）
        if isinstance(action, np.ndarray):
            action = action.copy()
        else:
            action = np.array(action)
        
        # 如果是第一步，直接通过
        if self.last_pitch_action is None:
            self.last_pitch_action = action[self.pitch_idx]
            self.last_yaw_action = action[self.yaw_idx]
        else:
            # 获取当前动作
            pitch_action = action[self.pitch_idx]
            yaw_action = action[self.yaw_idx]
            
            # 转换为角度差（相对于中心12）
            pitch_angle = (pitch_action - self.camera_center) * self.degrees_per_unit
            last_pitch_angle = (self.last_pitch_action - self.camera_center) * self.degrees_per_unit
            
            yaw_angle = (yaw_action - self.camera_center) * self.degrees_per_unit
            last_yaw_angle = (self.last_yaw_action - self.camera_center) * self.degrees_per_unit
            
            # 计算角度变化
            pitch_delta = pitch_angle - last_pitch_angle
            yaw_delta = yaw_angle - last_yaw_angle
            
            # 限制变化幅度
            if abs(pitch_delta) > self.max_pitch_change:
                # 限制到最大变化范围
                sign = 1 if pitch_delta > 0 else -1
                limited_pitch_angle = last_pitch_angle + sign * self.max_pitch_change
                # 转换回离散值
                limited_pitch_action = int(round(limited_pitch_angle / self.degrees_per_unit + self.camera_center))
                # 确保在有效范围内 [0, 24]
                limited_pitch_action = np.clip(limited_pitch_action, 0, 24)
                action[self.pitch_idx] = limited_pitch_action
            
            if abs(yaw_delta) > self.max_yaw_change:
                sign = 1 if yaw_delta > 0 else -1
                limited_yaw_angle = last_yaw_angle + sign * self.max_yaw_change
                limited_yaw_action = int(round(limited_yaw_angle / self.degrees_per_unit + self.camera_center))
                limited_yaw_action = np.clip(limited_yaw_action, 0, 24)
                action[self.yaw_idx] = limited_yaw_action
            
            # 更新记录
            self.last_pitch_action = action[self.pitch_idx]
            self.last_yaw_action = action[self.yaw_idx]
        
        return self.env.step(action)


class CameraConstraintWrapper(gym.Wrapper):
    """
    相机约束包装器
    
    限制相机只在合理范围内移动（例如：不要看天空太多）
    
    Args:
        env: 环境实例
        limit_pitch: 是否限制pitch范围
        min_pitch: 最小pitch（度，负数表示向下）
        max_pitch: 最大pitch（度，正数表示向上）
        camera_indices: 相机动作在动作空间中的索引
    """
    
    def __init__(self, env, limit_pitch=True, min_pitch=-45, max_pitch=45,
                 camera_indices=(5, 6)):
        super().__init__(env)
        self.limit_pitch = limit_pitch
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.pitch_idx, self.yaw_idx = camera_indices
        
        # 假设pitch动作编码：
        # MineDojo通常使用: 0=不动, 1=向上, 2=向下
        # 或类似的离散编码
        
        print(f"  📷 相机约束: 启用")
        if limit_pitch:
            print(f"     Pitch范围: {min_pitch}° 到 {max_pitch}°")
    
    def step(self, action):
        """
        根据当前pitch状态，限制向上/向下的动作
        
        注意：这需要跟踪当前的pitch角度，这在某些环境中可能不可用
        """
        # TODO: 实现pitch跟踪和约束
        # 这需要从observation中获取当前pitch，或自己维护状态
        
        return self.env.step(action)


class CameraRewardShapingWrapper(gym.Wrapper):
    """
    相机奖励塑形包装器
    
    通过奖励/惩罚来引导agent学习平滑的相机控制
    
    Args:
        env: 环境实例
        smooth_reward: 平滑移动的奖励
        jitter_penalty: 抖动的惩罚
        camera_indices: 相机动作在动作空间中的索引
    """
    
    def __init__(self, env, smooth_reward=0.001, jitter_penalty=0.01,
                 camera_indices=(5, 6)):
        super().__init__(env)
        self.smooth_reward = smooth_reward
        self.jitter_penalty = jitter_penalty
        self.pitch_idx, self.yaw_idx = camera_indices
        
        self.last_pitch_action = None
        self.last_yaw_action = None
        
        print(f"  📷 相机奖励塑形: 启用")
        print(f"     平滑奖励: +{smooth_reward}")
        print(f"     抖动惩罚: -{jitter_penalty}")
    
    def reset(self, **kwargs):
        """重置环境"""
        self.last_pitch_action = None
        self.last_yaw_action = None
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """添加相机控制的奖励塑形"""
        obs, reward, done, info = self.env.step(action)
        
        if isinstance(action, np.ndarray):
            pitch_action = action[self.pitch_idx]
            yaw_action = action[self.yaw_idx]
        else:
            pitch_action = action[self.pitch_idx]
            yaw_action = action[self.yaw_idx]
        
        # 计算相机奖励/惩罚
        camera_reward = 0.0
        
        if self.last_pitch_action is not None:
            # 检测抖动（连续大幅变化）
            pitch_change = abs(pitch_action - self.last_pitch_action)
            yaw_change = abs(yaw_action - self.last_yaw_action)
            
            # 如果变化小（平滑），给予小奖励
            if pitch_change <= 1 and yaw_change <= 1:
                camera_reward += self.smooth_reward
            
            # 如果变化大（抖动），给予惩罚
            if pitch_change >= 3 or yaw_change >= 3:
                camera_reward -= self.jitter_penalty
        
        # 更新记录
        self.last_pitch_action = pitch_action
        self.last_yaw_action = yaw_action
        
        # 添加到总奖励
        total_reward = reward + camera_reward
        
        # 记录到info
        info['camera_reward'] = camera_reward
        info['original_reward'] = reward
        
        return obs, total_reward, done, info


class SimpleCameraWrapper(gym.Wrapper):
    """
    简单相机控制包装器（推荐用于初期训练）
    
    强制相机动作为"不动"或"小幅移动"，禁用大幅移动
    
    Args:
        env: 环境实例
        allow_large_moves: 是否允许大幅移动（早期训练建议False）
        camera_indices: 相机动作在动作空间中的索引
    """
    
    def __init__(self, env, allow_large_moves=False, camera_indices=(5, 6)):
        super().__init__(env)
        self.allow_large_moves = allow_large_moves
        self.pitch_idx, self.yaw_idx = camera_indices
        
        print(f"  📷 简化相机控制: 启用")
        print(f"     允许大幅移动: {'是' if allow_large_moves else '否（仅小幅/不动）'}")
    
    def step(self, action):
        """
        限制相机动作范围
        
        假设MineDojo的相机动作编码（具体需要验证）：
        - 0: 不动
        - 1-2: 小幅移动
        - 3-4: 大幅移动
        
        如果不允许大幅移动，将3-4映射到1-2
        """
        if isinstance(action, np.ndarray):
            action = action.copy()
        else:
            action = np.array(action)
        
        if not self.allow_large_moves:
            # 限制pitch
            if action[self.pitch_idx] >= 3:
                action[self.pitch_idx] = min(action[self.pitch_idx], 2)
            
            # 限制yaw
            if action[self.yaw_idx] >= 3:
                action[self.yaw_idx] = min(action[self.yaw_idx], 2)
        
        return self.env.step(action)


