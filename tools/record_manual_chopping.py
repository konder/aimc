#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
录制手动砍树序列（键盘控制）
用于验证MineCLIP的16帧视频模式效果
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minedojo
import numpy as np
from PIL import Image
import time
import cv2

class KeyboardController:
    """
    键盘控制器 - MineDojo MultiDiscrete(8) 动作空间
    """
    
    def __init__(self, camera_delta=4):
        """
        初始化键盘控制器
        
        Args:
            camera_delta: 相机转动角度增量（默认1，范围1-12）
                         1 = 约15度，2 = 约30度，4 = 约60度
        """
        # 使用字典存储每个动作的状态，而不是按键
        self.actions = {
            'forward': False,
            'back': False,
            'left': False,
            'right': False,
            'jump': False,
            'pitch_up': False,
            'pitch_down': False,
            'yaw_left': False,
            'yaw_right': False,
            'attack': False,
        }
        self.running = True
        
        # 相机转动参数
        self.camera_delta = camera_delta
        
        # 键盘码映射
        self.key_map = {
            ord('w'): 'forward',
            ord('W'): 'forward',
            ord('s'): 'back',
            ord('S'): 'back',
            ord('a'): 'left',
            ord('A'): 'left',
            ord('d'): 'right',
            ord('D'): 'right',
            32: 'jump',  # Space
            ord('i'): 'pitch_up',
            ord('I'): 'pitch_up',
            ord('k'): 'pitch_down',
            ord('K'): 'pitch_down',
            ord('j'): 'yaw_left',
            ord('J'): 'yaw_left',
            ord('l'): 'yaw_right',
            ord('L'): 'yaw_right',
            ord('f'): 'attack',
            ord('F'): 'attack',
        }
        
        print("\n" + "=" * 80)
        print("🎮 键盘控制说明")
        print("=" * 80)
        print("\n移动控制:")
        print("  W - 前进")
        print("  S - 后退")
        print("  A - 左移")
        print("  D - 右移")
        print("  Space - 跳跃")
        print("\n相机控制:")
        print("  I - 向上看")
        print("  K - 向下看")
        print("  J - 向左看")
        print("  L - 向右看")
        print("\n动作:")
        print("  F - 攻击/挖掘（砍树）⭐")
        print("\n系统:")
        print("  Q - 停止录制并保存")
        print("  ESC - 紧急退出（不保存）")
        print("\n" + "=" * 80)
        print("提示: 点击OpenCV窗口，然后使用键盘控制")
        print("提示: 按住按键可以持续执行动作")
        print("=" * 80 + "\n")
    
    def update_action(self, key, press=True):
        """
        更新动作状态
        
        Args:
            key: 键盘码
            press: True=按下, False=释放
        """
        if key in self.key_map:
            action_name = self.key_map[key]
            self.actions[action_name] = press
    
    def get_action(self):
        """
        根据当前动作状态生成MineDojo动作
        
        Returns:
            action: 8维MultiDiscrete动作
        """
        # 初始化为中性动作
        action = np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32)
        
        # action[0]: forward/back (0=stay, 1=forward, 2=back)
        if self.actions['forward']:
            action[0] = 1
        elif self.actions['back']:
            action[0] = 2
        
        # action[1]: left/right (0=stay, 1=left, 2=right)
        if self.actions['left']:
            action[1] = 1
        elif self.actions['right']:
            action[1] = 2
        
        # action[2]: jump (0=no, 1=jump, 2=?, 3=sprint+jump)
        if self.actions['jump']:
            action[2] = 1
        
        # action[3]: pitch (12=center, range 0-24)
        # 简单模式：按一次就转一次
        if self.actions['pitch_up']:
            action[3] = 12 - self.camera_delta  # 向上看
        elif self.actions['pitch_down']:
            action[3] = 12 + self.camera_delta  # 向下看
        else:
            action[3] = 12  # 中心
        
        # action[4]: yaw (12=center, range 0-24)
        # 简单模式：按一次就转一次
        if self.actions['yaw_left']:
            action[4] = 12 - self.camera_delta  # 向左看
        elif self.actions['yaw_right']:
            action[4] = 12 + self.camera_delta  # 向右看
        else:
            action[4] = 12  # 中心
        
        # action[5]: functional (3=攻击，已验证 ✅)
        if self.actions['attack']:
            action[5] = 3  # 攻击动作（已确认有效）
        
        return action

def record_chopping_sequence(base_dir="data/expert_demos", max_frames=1000, camera_delta=4, max_episodes=10, fast_reset=False):
    """
    录制砍树过程（手动控制，支持多回合）
    
    Args:
        base_dir: 基础输出目录（会在下面创建episode_000, episode_001...）
        max_frames: 每回合最大帧数
        camera_delta: 相机转动角度增量（1-12，默认4约60度）
        max_episodes: 最大录制回合数（默认10）
        fast_reset: 是否使用快速重置（True=重用世界快速，False=重新生成世界慢但多样）
    """
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 自动检测下一个episode编号
    existing_episodes = sorted([d for d in os.listdir(base_dir) if d.startswith('episode_') and os.path.isdir(os.path.join(base_dir, d))])
    if existing_episodes:
        last_episode = existing_episodes[-1]
        last_num = int(last_episode.split('_')[1])
        start_episode = last_num + 1
        print(f"\n✓ 检测到已有 {len(existing_episodes)} 个episode，从 episode_{start_episode:03d} 开始")
    else:
        start_episode = 0
        print(f"\n✓ 目录为空，从 episode_000 开始")
    
    print("=" * 80)
    print("MineCLIP 砍树序列录制工具（多回合录制）")
    print("=" * 80)
    print(f"\n基础目录: {base_dir}")
    print(f"Episode范围: episode_{start_episode:03d} ~ episode_{start_episode + max_episodes - 1:03d}")
    print(f"每回合最大帧数: {max_frames}")
    print(f"Reset模式: {'快速模式(重用世界)' if fast_reset else '完整模式(重新生成世界)'}")
    if not fast_reset:
        print("  ⚠️  完整模式reset较慢(5-10秒)，但数据多样性高")
    
    # 创建环境
    print("\n[1/3] 创建MineDojo环境...")
    print("  任务: harvest_1_log_forest (森林中砍树)")
    
    env = minedojo.make(
        task_id="harvest_1_log_forest",
        image_size=(160, 256),
        world_seed=None,  # 每次随机种子，增加数据多样性
        fast_reset=fast_reset
    )
    print("  ✓ 环境创建成功")
    print(f"  动作空间: {env.action_space}")
    
    # 初始化键盘控制器
    controller = KeyboardController(camera_delta=camera_delta)
    print(f"\n⚙️  相机设置: delta={camera_delta} (约{camera_delta*15}度/次)")
    
    # 显示窗口
    window_name = "MineCraft - 多回合录制"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 640)
    
    # 全局统计
    completed_episodes = 0
    global_continue = True
    
    print("\n[2/3] 开始多回合录制...")
    print("\n" + "=" * 80)
    print("🎬 多回合录制模式")
    print("=" * 80)
    print("  ✅ 完成任务(done=True) → 自动保存当前回合，进入下一回合")
    print("  🔄 按Q键 → 不保存当前回合，重新录制当前回合")
    print("  ❌ 按ESC → 不保存当前回合，退出程序")
    print("=" * 80 + "\n")
    
    try:
        # 多回合循环
        episode_idx = start_episode
        while episode_idx < start_episode + max_episodes:
            if not global_continue:
                break
            
            # 重新录制标志
            retry_current_episode = False
                
            # 重置环境，开始新回合
            print(f"\n{'='*80}")
            print(f"🎮 Round {episode_idx}")
            print(f"{'='*80}")
            
            print(f"  重置环境中...")
            obs_dict = env.reset()
            obs = obs_dict['rgb']  # (C, H, W)
            
            # 转换为 (H, W, C) 格式
            if obs.shape[0] == 3:
                obs = obs.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            print(f"  ✓ 环境已重置，新的世界已生成")
            
            # 本回合数据
            frames = []
            actions_list = []  # 保存每一帧的action
            step_count = 0
            total_reward = 0
            task_completed = False
            
            # 显示初始画面，让用户看到新环境
            display_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            display_frame = cv2.resize(display_frame, (1024, 640))
            cv2.putText(display_frame, f"Round {episode_idx} - Ready! Press any key to start", 
                       (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(1000)  # 等待1秒，让用户看到新环境
            
            print(f"  开始录制 episode_{episode_idx:03d}...")
            print(f"  目标: 完成任务 (done=True)")
            print(f"  控制: Q=重录当前回合 | ESC=退出程序 | 完成=自动保存\n")
            
            # 本回合主循环
            while step_count < max_frames:
                # 先处理键盘事件，更新controller.actions
                keys_pressed = []
                for _ in range(10):  # 检测多次以捕获更多按键
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        keys_pressed.append(key)
                
                # 处理系统按键
                if ord('q') in keys_pressed or ord('Q') in keys_pressed:
                    print(f"\n🔄 重新录制 episode_{episode_idx:03d}（用户按下Q）")
                    print(f"   当前回合数据不保存，即将重置环境...")
                    retry_current_episode = True  # 标记需要重新录制当前round
                    frames = []  # 清空帧数据
                    actions_list = []  # 清空动作数据
                    break  # 跳出while循环，重新开始当前round
                elif 27 in keys_pressed:  # ESC
                    print(f"\n❌ 退出程序（用户按下ESC）")
                    print(f"   当前回合数据不保存")
                    global_continue = False  # 停止所有录制
                    frames = []  # 清空帧数据
                    actions_list = []  # 清空动作数据
                    break  # 跳出while循环并退出for循环
                
                # 更新动作状态（每帧重置，只保留当前检测到的按键）
                # 先重置所有动作
                for action_name in controller.actions:
                    controller.actions[action_name] = False
                
                # 然后设置当前检测到的按键
                if len(keys_pressed) > 0:
                    for key in keys_pressed:
                        controller.update_action(key, press=True)
                
                # 然后获取动作
                action = controller.get_action()
                
                # 执行动作
                obs_dict, reward, done, info = env.step(action)
                obs = obs_dict['rgb']  # (C, H, W)
                
                # 转换为 (H, W, C) 格式
                if obs.shape[0] == 3:
                    obs = obs.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                
                # 保存帧和动作
                frames.append(obs.copy())
                actions_list.append(action.copy())
                step_count += 1
                total_reward += reward
                
                # 显示当前帧
                display_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                display_frame = cv2.resize(display_frame, (1024, 640))
                
                # 添加信息overlay
                info_text = [
                    f"Round: {episode_idx} (目标: {start_episode + max_episodes - 1})",
                    f"Completed: {completed_episodes}",
                    f"Frame: {step_count}/{max_frames}",
                    f"Reward: {reward:.3f}",
                    f"Total: {total_reward:.3f}",
                    f"Status: {'DONE!' if task_completed else 'Recording...'}",
                    "",
                    "Q=retry | ESC=quit | Done=auto save&next"
                ]
                
                y_offset = 30
                for i, text in enumerate(info_text):
                    color = (0, 255, 0) if task_completed and i == 4 else (255, 255, 255)
                    cv2.putText(display_frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 添加动作状态显示
                active_actions = [name for name, active in controller.actions.items() if active]
                if active_actions:
                    action_text = f"Actions: {', '.join(active_actions)}"
                    cv2.putText(display_frame, action_text, (10, y_offset + 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
                
                # 检查任务是否完成（通过done信号）
                if done:
                    task_completed = True
                    print(f"\n🎉 episode_{episode_idx:03d}: 任务完成！已录制 {step_count} 帧")
                    # 检查是否是因为获得了目标物品
                    inventory = info.get('delta_inv', {})
                    if inventory:
                        print(f"    物品变化: {inventory}")
                    # 立即跳出循环，准备保存和reset
                    break
                
                # 控制帧率
                time.sleep(0.05)
            
            # 回合结束后的处理
            if retry_current_episode:
                # 按了Q键，重新录制当前round
                print(f"  准备重新录制 episode_{episode_idx:03d}...")
                # episode_idx不变，继续while循环
                continue
            
            # 正常结束：保存数据（只有done=True才保存）
            if task_completed and len(frames) > 0:
                # 创建round目录
                episode_dir = os.path.join(base_dir, f"episode_{episode_idx:03d}")
                os.makedirs(episode_dir, exist_ok=True)
                
                print(f"\n  💾 保存 episode_{episode_idx:03d} 数据...")
                
                # 1. 保存PNG图片（用于可视化验证）
                print(f"    [1/3] 保存PNG图片...")
                for i, frame in enumerate(frames):
                    img = Image.fromarray(frame)
                    filename = f"frame_{i:05d}.png"
                    filepath = os.path.join(episode_dir, filename)
                    img.save(filepath)
                
                # 2. 保存observation和action的numpy数据（用于BC训练）
                print(f"    [2/3] 保存BC训练数据...")
                for i, (obs, action) in enumerate(zip(frames, actions_list)):
                    frame_data = {
                        'observation': obs,  # (H, W, C) RGB uint8
                        'action': action     # (8,) int64
                    }
                    filename = f"frame_{i:05d}.npy"
                    filepath = os.path.join(episode_dir, filename)
                    np.save(filepath, frame_data)
                
                # 3. 保存回合元数据
                print(f"    [3/3] 保存元数据...")
                metadata_path = os.path.join(episode_dir, "metadata.txt")
                with open(metadata_path, 'w') as f:
                    f.write(f"Round: {episode_idx}\n")
                    f.write(f"Frames: {len(frames)}\n")
                    f.write(f"Actions: {len(actions_list)}\n")
                    f.write(f"Total Reward: {total_reward:.3f}\n")
                    f.write(f"Task Completed: True\n")
                    f.write(f"Recording Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"\nData Format:\n")
                    f.write(f"  - frame_XXXXX.png: 可视化图片 (H, W, 3) RGB\n")
                    f.write(f"  - frame_XXXXX.npy: BC训练数据 {{observation, action}}\n")
                    f.write(f"  - observation shape: {frames[0].shape}\n")
                    f.write(f"  - action shape: {actions_list[0].shape}\n")
                
                print(f"  ✓ episode_{episode_idx:03d} 已保存: {len(frames)} 帧 -> {episode_dir}")
                print(f"    - {len(frames)} PNG图片")
                print(f"    - {len(actions_list)} NPY文件（BC训练）")
                completed_episodes += 1
            elif not task_completed:
                print(f"\n  ⚠️  episode_{episode_idx:03d} 未完成 (done=False)，不保存")
                if not global_continue:
                    print("  用户按下ESC，退出录制")
                    break
            else:
                print(f"\n  ⚠️  episode_{episode_idx:03d} 没有录制任何帧，跳过")
            
            # 进入下一个round
            episode_idx += 1
    
    except KeyboardInterrupt:
        print("\n\n⏸️  录制停止（Ctrl+C）")
    
    finally:
        cv2.destroyAllWindows()
    
    # 最终统计
    print(f"\n\n{'='*80}")
    print("📊 录制完成统计")
    print(f"{'='*80}")
    
    if completed_episodes == 0:
        print("\n❌ 没有完成任何回合（done=True的回合数为0）")
        print("提示: 只有done=True时才会保存回合数据")
        env.close()
        return
    
    print(f"\n✅ 成功完成回合数: {completed_episodes}")
    print(f"Episode范围: episode_{start_episode:03d} ~ episode_{start_episode + completed_episodes - 1:03d}")
    print(f"\n保存位置: {base_dir}/")
    
    # 列出已保存的episode
    saved_episodes = sorted([d for d in os.listdir(base_dir) if d.startswith('episode_') and os.path.isdir(os.path.join(base_dir, d))])
    print(f"\n已保存的回合:")
    for ep in saved_episodes:
        ep_path = os.path.join(base_dir, ep)
        frame_count = len([f for f in os.listdir(ep_path) if f.endswith('.png')])
        print(f"  {ep}: {frame_count} 帧")
    
    # 保存全局元数据
    summary_path = os.path.join(base_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Total Completed Episodes: {completed_episodes}\n")
        f.write(f"Episode Range: episode_{start_episode:03d} ~ episode_{start_episode + completed_episodes - 1:03d}\n")
        f.write(f"Camera Delta: {camera_delta}\n")
        f.write(f"Max Frames per Episode: {max_frames}\n")
        f.write(f"Recording Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nSaved Episodes:\n")
        for ep in saved_episodes:
            ep_path = os.path.join(base_dir, ep)
            frame_count = len([f for f in os.listdir(ep_path) if f.endswith('.png')])
            f.write(f"  {ep}: {frame_count} frames\n")
    
    print(f"\n✓ 统计信息已保存到: {summary_path}")
    
    # 关闭环境
    env.close()
    
    print("\n" + "=" * 80)
    print("✅ 多回合录制完成！")
    print("=" * 80)
    print(f"\n继续录制提示:")
    print(f"  python tools/record_manual_chopping.py --base-dir {base_dir}")
    print(f"  (自动从 episode_{start_episode + completed_episodes:03d} 继续)")
    
    print(f"\n🔬 下一步: BC训练")
    print(f"  python src/training/train_bc.py --data {base_dir} --output checkpoints/bc_baseline.zip --epochs 50")
    print()
    
    return base_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="录制砍树序列用于BC训练（多回合录制）")
    parser.add_argument('--base-dir', type=str, default='data/expert_demos',
                       help='基础输出目录（会在下面创建episode_000, episode_001...，默认: data/expert_demos）')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='每回合最大录制帧数（默认: 1000）')
    parser.add_argument('--max-episodes', type=int, default=10,
                       help='最大录制回合数（默认: 10）')
    parser.add_argument('--camera-delta', type=int, default=1,
                       help='相机转动角度增量（1-12，默认1约15度，2约30度，4约60度）')
    parser.add_argument('--fast-reset', action='store_true',
                       help='使用快速重置模式（重用世界，快但数据多样性低）')
    parser.add_argument('--no-fast-reset', dest='fast_reset', action='store_false',
                       help='使用完整重置模式（重新生成世界，慢但数据多样性高，默认）')
    parser.set_defaults(fast_reset=False)
    
    args = parser.parse_args()
    
    # 验证camera_delta范围
    if args.camera_delta < 1 or args.camera_delta > 12:
        print(f"⚠️  警告: camera_delta={args.camera_delta} 超出推荐范围[1-12]，已调整为4")
        args.camera_delta = 4
    
    record_chopping_sequence(args.base_dir, args.max_frames, args.camera_delta, args.max_episodes, args.fast_reset)

