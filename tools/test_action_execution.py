#!/usr/bin/env python
"""
测试动作是否真的被执行
通过观察角色位置变化来验证
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import minedojo

def test_forward_action():
    """测试前进动作是否真的让角色移动"""
    print("="*60)
    print("测试：前进动作是否被执行？")
    print("="*60)
    
    # 创建环境
    env = minedojo.make(
        task_id="harvest_1_log",
        image_size=(160, 256),
        seed=42
    )
    
    obs = env.reset()
    
    # 获取初始位置
    if hasattr(env.unwrapped, 'agent_host'):
        # 尝试获取agent位置
        try:
            from minedojo.sim import InventoryItem
            world_state = env.unwrapped.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                obs_text = world_state.observations[-1].text
                import json
                obs_data = json.loads(obs_text)
                initial_pos = obs_data.get('XPos', 0), obs_data.get('YPos', 0), obs_data.get('ZPos', 0)
                print(f"初始位置: X={initial_pos[0]:.2f}, Y={initial_pos[1]:.2f}, Z={initial_pos[2]:.2f}")
        except Exception as e:
            print(f"无法获取初始位置: {e}")
            initial_pos = None
    else:
        initial_pos = None
    
    # 执行连续前进动作
    print("\n执行100步连续前进...")
    forward_action = np.array([1, 0, 0, 12, 12, 0, 0, 0])  # 前进
    
    for step in range(100):
        obs, reward, done, info = env.step(forward_action)
        
        if step % 20 == 0:
            print(f"  步骤 {step}:")
            print(f"    动作: {forward_action}")
            print(f"    奖励: {reward}")
            print(f"    Done: {done}")
            
            # 尝试获取当前位置
            if initial_pos and hasattr(env.unwrapped, 'agent_host'):
                try:
                    world_state = env.unwrapped.agent_host.getWorldState()
                    if world_state.number_of_observations_since_last_state > 0:
                        obs_text = world_state.observations[-1].text
                        import json
                        obs_data = json.loads(obs_text)
                        current_pos = obs_data.get('XPos', 0), obs_data.get('YPos', 0), obs_data.get('ZPos', 0)
                        
                        # 计算移动距离
                        dx = current_pos[0] - initial_pos[0]
                        dy = current_pos[1] - initial_pos[1]
                        dz = current_pos[2] - initial_pos[2]
                        distance = (dx**2 + dy**2 + dz**2)**0.5
                        
                        print(f"    位置: X={current_pos[0]:.2f}, Y={current_pos[1]:.2f}, Z={current_pos[2]:.2f}")
                        print(f"    移动距离: {distance:.2f} blocks")
                except Exception as e:
                    pass
        
        if done:
            print(f"\nEpisode结束于第{step}步")
            break
    
    env.close()
    
    print("\n"+"="*60)
    print("如果上面显示'移动距离'在增加，说明动作被正确执行")
    print("如果移动距离始终为0，说明动作没有被执行")
    print("="*60)


def test_different_actions():
    """测试不同动作的视觉效果"""
    print("\n"+"="*60)
    print("测试：不同动作的执行情况")
    print("="*60)
    
    env = minedojo.make(
        task_id="harvest_1_log",
        image_size=(160, 256),
        seed=42
    )
    
    obs = env.reset()
    
    actions_to_test = [
        ("IDLE", np.array([0, 0, 0, 12, 12, 0, 0, 0])),
        ("前进", np.array([1, 0, 0, 12, 12, 0, 0, 0])),
        ("后退", np.array([2, 0, 0, 12, 12, 0, 0, 0])),
        ("左移", np.array([0, 1, 0, 12, 12, 0, 0, 0])),
        ("右移", np.array([0, 2, 0, 12, 12, 0, 0, 0])),
        ("跳跃", np.array([0, 0, 1, 12, 12, 0, 0, 0])),
        ("前进+跳跃", np.array([1, 0, 1, 12, 12, 0, 0, 0])),
        ("攻击", np.array([0, 0, 0, 12, 12, 3, 0, 0])),
        ("视角向上", np.array([0, 0, 0, 10, 12, 0, 0, 0])),
        ("视角向下", np.array([0, 0, 0, 14, 12, 0, 0, 0])),
        ("视角向左", np.array([0, 0, 0, 12, 10, 0, 0, 0])),
        ("视角向右", np.array([0, 0, 0, 12, 14, 0, 0, 0])),
    ]
    
    for action_name, action in actions_to_test:
        print(f"\n测试动作: {action_name}")
        print(f"  动作数组: {action}")
        
        # 执行该动作10步
        for _ in range(10):
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                break
        
        # 观察结果
        print(f"  奖励: {reward}")
        print(f"  ✓ 动作已发送到环境")
    
    env.close()
    
    print("\n"+"="*60)
    print("注意：如果游戏窗口中角色没有动作，可能是以下原因：")
    print("1. 渲染延迟（MineDojo的渲染可能有延迟）")
    print("2. 游戏窗口没有焦点（需要点击窗口）")
    print("3. headless模式（没有窗口显示）")
    print("="*60)


if __name__ == "__main__":
    print("MineDojo动作执行测试\n")
    
    # 测试1: 验证前进动作
    test_forward_action()
    
    # 测试2: 测试各种动作
    test_different_actions()
    
    print("\n\n总结：")
    print("如果测试显示角色位置在变化，但游戏窗口看不到，可能是：")
    print("  1. 渲染帧率问题（MineDojo渲染较慢）")
    print("  2. 需要在录制模式下测试（render_mode='human'）")
    print("  3. 动作执行了但视觉上不明显")

