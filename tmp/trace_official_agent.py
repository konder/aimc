"""追踪官方agent._env_obs_to_agent的完整流程"""
import numpy as np
import sys
from pathlib import Path

# 添加官方VPT路径
VPT_PATH = Path.cwd() / "src" / "models" / "Video-Pre-Training"
sys.path.insert(0, str(VPT_PATH))

from agent import MineRLAgent, resize_image, AGENT_RESOLUTION
import torch as th
import cv2

# 创建假环境
class FakeEnv:
    class FakeTaskSpec:
        fov_range = [70, 70]
        frameskip = 1
        gamma_range = [2, 2]
        guiscale_range = [1, 1]
        resolution = [640, 360]
        cursor_size_range = [16.0, 16.0]
    
    class FakeActionSpace:
        from gym import spaces
        spaces = {
            "ESC": spaces.Discrete(2), "attack": spaces.Discrete(2), "back": spaces.Discrete(2),
            "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)), "drop": spaces.Discrete(2),
            "forward": spaces.Discrete(2), "hotbar.1": spaces.Discrete(2), "hotbar.2": spaces.Discrete(2),
            "hotbar.3": spaces.Discrete(2), "hotbar.4": spaces.Discrete(2), "hotbar.5": spaces.Discrete(2),
            "hotbar.6": spaces.Discrete(2), "hotbar.7": spaces.Discrete(2), "hotbar.8": spaces.Discrete(2),
            "hotbar.9": spaces.Discrete(2), "inventory": spaces.Discrete(2), "jump": spaces.Discrete(2),
            "left": spaces.Discrete(2), "pickItem": spaces.Discrete(2), "right": spaces.Discrete(2),
            "sneak": spaces.Discrete(2), "sprint": spaces.Discrete(2), "swapHands": spaces.Discrete(2),
            "use": spaces.Discrete(2)
        }
    
    task = FakeTaskSpec()
    action_space = FakeActionSpace()

print("="*70)
print("追踪官方MineRLAgent._env_obs_to_agent")
print("="*70)

# 创建agent
agent = MineRLAgent(env=FakeEnv(), device='cpu')

# 测试输入
pov_hwc = np.random.randint(0, 255, (160, 256, 3), dtype=np.uint8)
print(f"\n输入pov (HWC): {pov_hwc.shape}")

# 手动执行_env_obs_to_agent的步骤
print("\n逐步追踪官方_env_obs_to_agent:")

# Step 1: resize
print(f"\n1. resize_image(pov, {AGENT_RESOLUTION})")
resized = resize_image(pov_hwc, AGENT_RESOLUTION)
print(f"   输出shape: {resized.shape}")

# Step 2: [None]
print(f"\n2. resized[None]")
with_batch = resized[None]
print(f"   输出shape: {with_batch.shape}")

# Step 3: th.from_numpy
print(f"\n3. th.from_numpy(with_batch)")
tensor = th.from_numpy(with_batch)
print(f"   输出shape: {tensor.shape}")

# Step 4: .to(device)
print(f"\n4. tensor.to('cpu')")
on_device = tensor.to('cpu')
print(f"   输出shape: {on_device.shape}")

# Final
print(f"\n最终输出: {'img': tensor with shape {on_device.shape}}")

print("\n" + "="*70)
print("结论: 官方代码从(H,W,C)->(1,H,W,C)，CNN期望(1,C,H,W)!")
print("官方代码缺少permute操作！")
print("="*70)
