"""检查官方VPT期望的输入格式"""
import numpy as np
import cv2

# 模拟MineRL观察
minerl_obs = {
    "pov": np.random.randint(0, 255, (640, 360, 3), dtype=np.uint8)
}

print("="*70)
print("官方VPT输入格式检查")
print("="*70)

# 模拟官方resize_image
AGENT_RESOLUTION = (128, 128)
img = minerl_obs["pov"]
print(f"\n原始图像:")
print(f"  shape: {img.shape}  # (H, W, C)")
print(f"  dtype: {img.dtype}")

# cv2.resize
img_resized = cv2.resize(img, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
print(f"\nresize后:")
print(f"  shape: {img_resized.shape}  # (H, W, C)")
print(f"  target_resolution: {AGENT_RESOLUTION}  # (W, H)")

# 添加batch维度
img_with_batch = img_resized[None]
print(f"\n添加batch维度后:")
print(f"  shape: {img_with_batch.shape}  # (B, H, W, C)")

# 转换为torch tensor
import torch as th
img_tensor = th.from_numpy(img_with_batch)
print(f"\n转换为tensor:")
print(f"  shape: {img_tensor.shape}  # (B, H, W, C) - channels last")

# PyTorch CNN需要channels first
img_tensor_chw = img_tensor.permute(0, 3, 1, 2)
print(f"\npermute(0,3,1,2)后:")
print(f"  shape: {img_tensor_chw.shape}  # (B, C, H, W) - channels first ✓")

print("\n" + "="*70)
print("结论：官方代码应该有permute操作，或者MineRL返回的就是CHW格式")
print("="*70)
