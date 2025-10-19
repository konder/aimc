#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineDojo Hello World 示例
这是一个简单的MineDojo环境使用示例
"""

import minedojo
import numpy as np


def main():
    """
    MineDojo Hello World 主函数
    创建一个简单的环境并运行几个步骤
    """
    print("=" * 60)
    print("MineDojo Hello World!")
    print("=" * 60)
    
    # 创建一个简单的MineDojo环境
    # 使用harvest任务作为示例
    print("\n[1] 创建MineDojo环境...")
    try:
        env = minedojo.make(
            task_id="harvest_milk",  # 简单的收集牛奶任务
            image_size=(160, 256),   # 较小的图像尺寸以便快速运行
        )
        print("✓ 环境创建成功!")
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        print("\n尝试使用备用配置...")
        env = minedojo.make(
            task_id="open-ended",  # 使用开放式任务
            image_size=(160, 256),
        )
        print("✓ 使用备用配置创建成功!")
    
    # 重置环境
    print("\n[2] 重置环境...")
    obs = env.reset()
    print("✓ 环境重置成功!")
    print(f"   观察空间包含的键: {obs.keys()}")
    print(f"   RGB图像形状: {obs['rgb'].shape}")
    
    # 运行几个随机步骤
    print("\n[3] 执行随机动作...")
    num_steps = 100
    for step in range(num_steps):
        # 采样一个随机动作
        action = env.action_space.no_op()  # 使用 no-op 动作而不是随机动作
        
        try:
            # 执行动作
            obs, reward, done, info = env.step(action)
            print(f"   步骤 {step + 1}/{num_steps}: reward={reward:.4f}, done={done}")
            
            if done:
                print("   环境已结束，重置环境...")
                obs = env.reset()
        except ValueError as e:
            # 捕获无效动作错误（例如：尝试破坏空气）
            print(f"   步骤 {step + 1}/{num_steps}: 捕获到无效动作错误，继续...")
            continue
    
    # 关闭环境
    print("\n[4] 关闭环境...")
    env.close()
    print("✓ 环境已关闭!")
    
    print("\n" + "=" * 60)
    print("MineDojo Hello World 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

