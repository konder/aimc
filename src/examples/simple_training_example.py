#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的训练示例
展示如何使用最少的代码进行训练
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from src.utils.env_wrappers import make_minedojo_env


def simple_training():
    """
    最简单的训练示例
    适合快速测试和理解训练流程
    """
    print("=" * 70)
    print("MineDojo 简单训练示例")
    print("=" * 70)
    
    # 1. 创建环境
    print("\n[1/4] 创建环境...")
    env = make_minedojo_env(
        task_id="harvest_milk",
        image_size=(160, 256)
    )
    print("  ✓ 环境创建成功")
    
    # 2. 创建模型（从头开始，无预训练权重）
    print("\n[2/4] 创建PPO模型...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=0.0003,
        verbose=1,
    )
    print("  ✓ 模型创建成功（随机初始化）")
    
    # 3. 训练
    print("\n[3/4] 开始训练（10K步，快速演示）...")
    print("  注意: 这是从头训练，初始性能会很差")
    model.learn(total_timesteps=10000)
    print("  ✓ 训练完成")
    
    # 4. 保存模型
    print("\n[4/4] 保存模型...")
    model.save("simple_model")
    print("  ✓ 模型已保存到: simple_model.zip")
    
    # 清理
    env.close()
    
    print("\n" + "=" * 70)
    print("示例完成!")
    print("=" * 70)
    print("\n要评估模型:")
    print("  python src/examples/simple_evaluation_example.py")


if __name__ == "__main__":
    simple_training()

