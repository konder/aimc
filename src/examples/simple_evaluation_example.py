#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的评估示例
展示如何评估训练好的模型
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from src.utils.env_wrappers import make_minedojo_env


def simple_evaluation():
    """
    最简单的评估示例
    """
    print("=" * 70)
    print("MineDojo 简单评估示例")
    print("=" * 70)
    
    # 1. 加载模型
    print("\n[1/3] 加载模型...")
    try:
        model = PPO.load("simple_model")
        print("  ✓ 模型加载成功")
    except FileNotFoundError:
        print("  ✗ 模型文件未找到: simple_model.zip")
        print("  请先运行: python src/examples/simple_training_example.py")
        return
    
    # 2. 创建环境
    print("\n[2/3] 创建环境...")
    env = make_minedojo_env(
        task_id="harvest_milk",
        image_size=(160, 256)
    )
    print("  ✓ 环境创建成功")
    
    # 3. 评估
    print("\n[3/3] 运行评估（3个episodes）...")
    
    episode_rewards = []
    n_episodes = 3
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, "
              f"steps={steps}")
    
    # 统计
    print("\n" + "=" * 70)
    print("评估结果:")
    print("=" * 70)
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"标准差: {np.std(episode_rewards):.2f}")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    simple_evaluation()

