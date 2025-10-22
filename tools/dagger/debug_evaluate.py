#!/usr/bin/env python3
"""
调试评估脚本 - 检查模型在实际环境中的行为
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from stable_baselines3 import PPO
from src.utils.env_wrappers import make_minedojo_env

def debug_evaluate():
    """调试评估"""
    
    model_path = "checkpoints/dagger/harvest_1_log/bc_baseline.zip"
    
    print("="*80)
    print("调试BC模型评估")
    print("="*80)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    policy = PPO.load(model_path)
    print(f"  设备: {policy.device}")
    print(f"  策略: {type(policy.policy)}")
    
    # 创建环境
    print(f"\n创建环境...")
    env = make_minedojo_env(
        task_id="harvest_1_log",
        use_camera_smoothing=False,
        max_episode_steps=1000,
        fast_reset=False
    )
    
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 重置环境
    print(f"\n重置环境...")
    obs = env.reset()
    print(f"  观察形状: {obs.shape}")
    print(f"  观察类型: {obs.dtype}")
    print(f"  观察范围: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # 测试前100步
    print(f"\n开始测试前100步...")
    
    action_counts = {
        'idle': 0,
        'forward': 0,
        'back': 0,
        'left': 0,
        'right': 0,
        'jump': 0,
        'attack': 0,
        'camera_move': 0
    }
    
    for step in range(100):
        # 预测动作
        action, _ = policy.predict(obs, deterministic=True)
        
        # 统计动作
        is_idle = (action[0] == 0 and action[1] == 0 and action[2] == 0 and 
                  action[3] == 12 and action[4] == 12 and action[5] == 0)
        
        if is_idle:
            action_counts['idle'] += 1
        else:
            if action[0] == 1:
                action_counts['forward'] += 1
            elif action[0] == 2:
                action_counts['back'] += 1
            
            if action[1] == 1:
                action_counts['left'] += 1
            elif action[1] == 2:
                action_counts['right'] += 1
            
            if action[2] == 1:
                action_counts['jump'] += 1
            
            if action[5] == 3:
                action_counts['attack'] += 1
            
            if action[3] != 12 or action[4] != 12:
                action_counts['camera_move'] += 1
        
        # 打印前10步和每10步
        if step < 10 or step % 10 == 0:
            action_str = "IDLE" if is_idle else str(action)
            print(f"  步骤{step:3d}: {action_str}")
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\n  ✅ Episode结束于步骤{step}")
            break
    
    env.close()
    
    # 打印统计
    print(f"\n{'='*80}")
    print(f"动作统计 (100步)")
    print(f"{'='*80}")
    total = sum(action_counts.values())
    for action_type, count in action_counts.items():
        pct = (count / 100) * 100
        print(f"  {action_type:12s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n总计: {total}")
    print(f"IDLE占比: {action_counts['idle']/100*100:.1f}%")
    print(f"有效动作: {100 - action_counts['idle']}")

if __name__ == "__main__":
    debug_evaluate()

