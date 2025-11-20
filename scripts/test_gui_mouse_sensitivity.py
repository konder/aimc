#!/usr/bin/env python3
"""
测试 GUI 鼠标灵敏度
通过发送固定的 camera 值，观察鼠标移动的实际距离
"""

import gym
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.minedojo_harvest import register_minedojo_biome_env

def test_mouse_sensitivity():
    """测试鼠标灵敏度"""
    print("\n" + "=" * 80)
    print("GUI 鼠标灵敏度测试")
    print("=" * 80)
    
    # 注册环境
    register_minedojo_biome_env()
    
    # 创建环境
    env = gym.make(
        'MineDojoHarvestEnv-v0',
        task_id='open-ended',
        image_size=[320, 640],
        specified_biome='plains',
        start_time=0,
        allow_mob_spawn=False,
        initial_inventory=[
            {'type': 'oak_planks', 'quantity': 4}
        ]
    )
    
    print("\n✓ 环境创建成功")
    
    # Reset
    obs = env.reset()
    print("✓ 环境重置完成")
    
    # 保存初始图像
    output_dir = Path("/tmp/gui_mouse_sensitivity_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img = Image.fromarray(obs['pov'])
    img.save(output_dir / "step_0_initial.png")
    print(f"✓ 保存初始图像: {output_dir / 'step_0_initial.png'}")
    
    # 测试序列
    test_cases = [
        {
            'name': '打开物品栏',
            'action': {
                'camera': np.array([0.0, 0.0]),
                'inventory': 1,
                'attack': 0,
                'forward': 0,
                'back': 0,
                'left': 0,
                'right': 0,
                'jump': 0,
                'sneak': 0,
                'sprint': 0,
                'use': 0,
                'drop': 0,
                'swapHands': 0,
                'pickItem': 0,
                'craft': 0,
                'nearbyCraft': 0,
                'nearbySmelt': 0,
                'place': 0,
                'equip': 0
            }
        },
        {
            'name': '鼠标向右移动 1.0°',
            'action': {
                'camera': np.array([0.0, 1.0]),  # yaw = 1.0
                'inventory': 0,
                'attack': 0,
                'forward': 0,
                'back': 0,
                'left': 0,
                'right': 0,
                'jump': 0,
                'sneak': 0,
                'sprint': 0,
                'use': 0,
                'drop': 0,
                'swapHands': 0,
                'pickItem': 0,
                'craft': 0,
                'nearbyCraft': 0,
                'nearbySmelt': 0,
                'place': 0,
                'equip': 0
            }
        },
        {
            'name': '鼠标向下移动 1.0°',
            'action': {
                'camera': np.array([1.0, 0.0]),  # pitch = 1.0
                'inventory': 0,
                'attack': 0,
                'forward': 0,
                'back': 0,
                'left': 0,
                'right': 0,
                'jump': 0,
                'sneak': 0,
                'sprint': 0,
                'use': 0,
                'drop': 0,
                'swapHands': 0,
                'pickItem': 0,
                'craft': 0,
                'nearbyCraft': 0,
                'nearbySmelt': 0,
                'place': 0,
                'equip': 0
            }
        },
        {
            'name': '鼠标向右移动 5.0°',
            'action': {
                'camera': np.array([0.0, 5.0]),  # yaw = 5.0
                'inventory': 0,
                'attack': 0,
                'forward': 0,
                'back': 0,
                'left': 0,
                'right': 0,
                'jump': 0,
                'sneak': 0,
                'sprint': 0,
                'use': 0,
                'drop': 0,
                'swapHands': 0,
                'pickItem': 0,
                'craft': 0,
                'nearbyCraft': 0,
                'nearbySmelt': 0,
                'place': 0,
                'equip': 0
            }
        },
        {
            'name': '鼠标向下移动 5.0°',
            'action': {
                'camera': np.array([5.0, 0.0]),  # pitch = 5.0
                'inventory': 0,
                'attack': 0,
                'forward': 0,
                'back': 0,
                'left': 0,
                'right': 0,
                'jump': 0,
                'sneak': 0,
                'sprint': 0,
                'use': 0,
                'drop': 0,
                'swapHands': 0,
                'pickItem': 0,
                'craft': 0,
                'nearbyCraft': 0,
                'nearbySmelt': 0,
                'place': 0,
                'equip': 0
            }
        }
    ]
    
    print("\n" + "=" * 80)
    print("开始测试序列")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n━━━ Step {i}: {test_case['name']} ━━━")
        
        obs, reward, done, info = env.step(test_case['action'])
        
        # 保存图像
        img = Image.fromarray(obs['pov'])
        filename = f"step_{i}_{test_case['name'].replace(' ', '_')}.png"
        img.save(output_dir / filename)
        print(f"✓ 保存图像: {output_dir / filename}")
        
        # 打印 camera 信息
        camera = test_case['action']['camera']
        print(f"  camera: pitch={camera[0]:.1f}°, yaw={camera[1]:.1f}°")
        
        if done:
            print("  ⚠️  环境结束")
            break
    
    env.close()
    print("\n" + "=" * 80)
    print(f"✅ 测试完成！图像保存在: {output_dir}")
    print("=" * 80)
    print("\n请对比图像中鼠标的位置变化，测量实际移动的像素距离。")

if __name__ == "__main__":
    test_mouse_sensitivity()

