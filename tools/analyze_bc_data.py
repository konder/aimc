#!/usr/bin/env python
"""
分析BC训练数据的质量
检查静止帧、动作分布等
"""
import numpy as np
from pathlib import Path
import sys

def decode_action(action):
    """解码动作为人类可读格式"""
    parts = []
    
    # 前后移动
    if action[0] == 1: parts.append("Forward")
    elif action[0] == 2: parts.append("Back")
    
    # 左右移动
    if action[1] == 1: parts.append("Left")
    elif action[1] == 2: parts.append("Right")
    
    # 跳跃
    if action[2] == 1: parts.append("Jump")
    
    # 攻击
    if action[5] == 3: parts.append("Attack")
    
    # 视角
    pitch_delta = action[3] - 12
    yaw_delta = action[4] - 12
    if pitch_delta != 0 or yaw_delta != 0:
        parts.append(f"Camera(pitch={pitch_delta:+d}, yaw={yaw_delta:+d})")
    
    return " + ".join(parts) if parts else "IDLE"

base_dir = Path("data/expert_demos/harvest_1_log")
if not base_dir.exists():
    print(f"❌ 数据目录不存在: {base_dir}")
    sys.exit(1)

episodes = sorted(base_dir.glob("episode_*"))
print(f"找到 {len(episodes)} 个episode\n")

total_frames = 0
total_forward = 0
total_idle = 0
total_attack = 0
total_jump = 0
total_camera_move = 0

for ep_dir in episodes[:5]:  # 分析前5个
    print(f"═══ {ep_dir.name} ═══")
    
    frames = sorted(ep_dir.glob("frame_*.npy"))
    if not frames:
        print("  ⚠️  无帧数据\n")
        continue
    
    actions = []
    for frame_file in frames:
        try:
            data = np.load(frame_file, allow_pickle=True).item()
            actions.append(data['action'])
        except Exception as e:
            print(f"  ⚠️  读取帧失败: {frame_file.name} - {e}")
            continue
    
    if not actions:
        print("  ⚠️  无有效动作数据\n")
        continue
    
    actions = np.array(actions)
    ep_total = len(actions)
    ep_forward = np.sum(actions[:, 0] == 1)
    ep_idle = np.sum((actions[:, 0] == 0) & (actions[:, 1] == 0) & 
                     (actions[:, 2] == 0) & (actions[:, 5] == 0) &
                     (actions[:, 3] == 12) & (actions[:, 4] == 12))
    ep_attack = np.sum(actions[:, 5] == 3)
    ep_jump = np.sum(actions[:, 2] == 1)
    ep_camera = np.sum((actions[:, 3] != 12) | (actions[:, 4] != 12))
    
    total_frames += ep_total
    total_forward += ep_forward
    total_idle += ep_idle
    total_attack += ep_attack
    total_jump += ep_jump
    total_camera_move += ep_camera
    
    print(f"  总帧数: {ep_total}")
    print(f"  前进帧: {ep_forward} ({ep_forward/ep_total*100:.1f}%)")
    print(f"  静止帧: {ep_idle} ({ep_idle/ep_total*100:.1f}%)")
    print(f"  攻击帧: {ep_attack} ({ep_attack/ep_total*100:.1f}%)")
    print(f"  跳跃帧: {ep_jump} ({ep_jump/ep_total*100:.1f}%)")
    print(f"  视角移动帧: {ep_camera} ({ep_camera/ep_total*100:.1f}%)")
    
    # 显示前5帧和最后5帧
    print(f"  前5帧动作:")
    for i in range(min(5, len(actions))):
        action_str = decode_action(actions[i])
        print(f"    帧{i}: {action_str} -> {actions[i]}")
    
    print(f"  最后5帧动作:")
    for i in range(max(0, len(actions)-5), len(actions)):
        action_str = decode_action(actions[i])
        print(f"    帧{i}: {action_str} -> {actions[i]}")
    print()

if total_frames > 0:
    print(f"═══════════════════════════════════════════")
    print(f"═══ 总体统计 ({len(episodes)} episodes) ═══")
    print(f"═══════════════════════════════════════════")
    print(f"总帧数: {total_frames}")
    print(f"前进帧: {total_forward} ({total_forward/total_frames*100:.1f}%)")
    print(f"静止帧: {total_idle} ({total_idle/total_frames*100:.1f}%)")
    print(f"攻击帧: {total_attack} ({total_attack/total_frames*100:.1f}%)")
    print(f"跳跃帧: {total_jump} ({total_jump/total_frames*100:.1f}%)")
    print(f"视角移动帧: {total_camera_move} ({total_camera_move/total_frames*100:.1f}%)")
    
    print(f"\n═══ 数据质量评估 ═══")
    
    issues = []
    if total_idle / total_frames > 0.5:
        issues.append(f"⚠️  静止帧占比过高 ({total_idle/total_frames*100:.1f}%)")
        issues.append(f"   → BC模型可能学到了'大部分时间不动'的策略")
    
    if total_forward / total_frames < 0.3:
        issues.append(f"⚠️  前进帧占比过低 ({total_forward/total_frames*100:.1f}%)")
        issues.append(f"   → 模型可能学不到有效的移动策略")
    
    if total_attack / total_frames < 0.1:
        issues.append(f"⚠️  攻击帧占比过低 ({total_attack/total_frames*100:.1f}%)")
        issues.append(f"   → 对harvest_log任务，攻击是获取木头的关键")
    
    if issues:
        for issue in issues:
            print(issue)
        
        print(f"\n💡 建议:")
        print(f"1. 重新录制数据，使用 --skip-idle-frames")
        print(f"2. 录制时保持连续动作（少按.键）")
        print(f"3. 增加攻击和移动的比例")
    else:
        print(f"✅ 数据质量良好！")
        print(f"   前进帧充足，静止帧合理")
        
    # 理想分布
    print(f"\n📊 理想数据分布参考:")
    print(f"   前进帧: >60%")
    print(f"   静止帧: <15%")
    print(f"   攻击帧: 20-30%")
    print(f"   跳跃帧: 10-20%")
else:
    print("❌ 无有效数据")

