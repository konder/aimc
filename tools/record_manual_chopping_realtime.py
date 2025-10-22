#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
录制手动砍树序列（实时模式 - 使用pynput）
用于DAgger训练和MineCLIP验证

优势:
- 使用pynput后台监听，准确检测按住按键状态
- 按住W键时，每帧都能检测到前进动作
- 大幅减少静态帧，提高录制效率
- 支持多键同时检测（如W+F = 边前进边攻击）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import minedojo
import numpy as np
from PIL import Image
import time
import cv2
import argparse
import shutil
from pynput import keyboard

class RealtimeKeyController:
    """
    实时键盘控制器 - 使用pynput后台监听
    
    特点:
    - 后台线程监听所有按键事件
    - 准确追踪按键的按下/释放状态
    - 支持多个按键同时按下
    - 与OpenCV窗口无冲突
    """
    
    def __init__(self, camera_delta=4):
        """
        初始化实时键盘控制器
        
        Args:
            camera_delta: 相机转动角度增量（1-12）
                         1 = ~15度/帧, 4 = ~60度/帧
        """
        # 当前按下的所有按键（实时追踪）
        self.pressed_keys = set()
        
        # 相机转动参数
        self.camera_delta = camera_delta
        
        # 控制标志
        self.should_quit = False      # ESC退出
        self.should_retry = False     # Q重试当前回合
        
        # 启动后台监听器（非阻塞）
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        
        print("\n" + "=" * 80)
        print("🎮 实时录制模式 (pynput)")
        print("=" * 80)
        print("\n✅ 优势:")
        print("  - 按住按键时，每帧都能检测到动作")
        print("  - 大幅减少静态帧，提高数据质量")
        print("  - 支持多键同时按下（如W+F边前进边攻击）")
        print("\n📌 录制方式:")
        print("  - 录制以20 FPS速度进行（每帧50ms）")
        print("  - 按住W键会持续前进，不需要每帧按键")
        print("  - 松开按键后动作停止")
        print("\n移动控制:")
        print("  W - 前进 | S - 后退 | A - 左移 | D - 右移 | Space - 跳跃")
        print("\n相机控制:")
        print("  I - 向上看 | K - 向下看 | J - 向左看 | L - 向右看")
        print("\n动作:")
        print("  F - 攻击/挖掘（砍树）⭐")
        print("\n系统:")
        print("  Q - 重新录制当前回合（不保存）")
        print("  ESC - 退出程序（不保存当前回合）")
        print("\n" + "=" * 80)
        print(f"相机灵敏度: {camera_delta} (按一次转动 ~{camera_delta*15}度)")
        print("=" * 80 + "\n")
    
    def _on_press(self, key):
        """按键按下事件（后台线程调用）"""
        try:
            # 普通字符键
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                self.pressed_keys.add(char)
                
                # Q键：重试当前回合
                if char == 'q':
                    self.should_retry = True
        except AttributeError:
            # 特殊键
            if key == keyboard.Key.space:
                self.pressed_keys.add('space')
            elif key == keyboard.Key.esc:
                self.should_quit = True
    
    def _on_release(self, key):
        """按键释放事件（后台线程调用）"""
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.space:
                self.pressed_keys.discard('space')
    
    def get_action(self):
        """
        根据当前按键状态生成MineDojo动作
        
        Returns:
            action: np.array([8], dtype=np.int32)
                [0] forward_back: 0=stop, 1=forward, 2=back
                [1] left_right: 0=stop, 1=left, 2=right
                [2] jump: 0=noop, 1=jump
                [3] pitch: 12=center, <12=up, >12=down
                [4] yaw: 12=center, <12=left, >12=right
                [5] functional: 0=noop, 3=attack
                [6] craft_argument: 0=noop
                [7] inventory_argument: 0=noop
        """
        action = np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32)
        
        # 移动 (dimension 0: forward/back)
        if 'w' in self.pressed_keys:
            action[0] = 1  # forward
        elif 's' in self.pressed_keys:
            action[0] = 2  # back
        
        # 左右 (dimension 1: left/right)
        if 'a' in self.pressed_keys:
            action[1] = 1  # left
        elif 'd' in self.pressed_keys:
            action[1] = 2  # right
        
        # 跳跃 (dimension 2: jump)
        if 'space' in self.pressed_keys:
            action[2] = 1
        
        # 相机 pitch (dimension 3)
        if 'i' in self.pressed_keys:
            action[3] = 12 - self.camera_delta  # pitch up
        elif 'k' in self.pressed_keys:
            action[3] = 12 + self.camera_delta  # pitch down
        
        # 相机 yaw (dimension 4)
        if 'j' in self.pressed_keys:
            action[4] = 12 - self.camera_delta  # yaw left
        elif 'l' in self.pressed_keys:
            action[4] = 12 + self.camera_delta  # yaw right
        
        # 攻击 (dimension 5: functional)
        if 'f' in self.pressed_keys:
            action[5] = 3  # attack
        
        return action
    
    def decode_action(self, action):
        """将动作数组转换为可读描述"""
        parts = []
        
        # 移动
        if action[0] == 1:
            parts.append("Forward")
        elif action[0] == 2:
            parts.append("Back")
        
        if action[1] == 1:
            parts.append("Left")
        elif action[1] == 2:
            parts.append("Right")
        
        # 跳跃
        if action[2] == 1:
            parts.append("Jump")
        
        # 相机
        if action[3] != 12 or action[4] != 12:
            pitch_delta = action[3] - 12
            yaw_delta = action[4] - 12
            parts.append(f"Camera(pitch={pitch_delta:+d}, yaw={yaw_delta:+d})")
        
        # 攻击
        if action[5] == 3:
            parts.append("ATTACK")
        
        return " + ".join(parts) if parts else "IDLE"
    
    def reset_retry_flag(self):
        """重置重试标志"""
        self.should_retry = False
    
    def stop(self):
        """停止监听器"""
        self.listener.stop()


def record_chopping_sequence(
    base_dir="data/expert_demos",
    max_frames=1000,
    camera_delta=4,
    fast_reset=False,
    fps=20
):
    """
    录制手动砍树序列（实时模式）
    
    Args:
        base_dir: 保存目录
        max_frames: 每个回合的最大帧数
        camera_delta: 相机灵敏度
        fast_reset: 是否快速重置（True=同一世界，False=新世界）
        fps: 录制帧率（默认20 FPS）
    """
    # 检测已有episode
    os.makedirs(base_dir, exist_ok=True)
    existing_episodes = [d for d in os.listdir(base_dir) if d.startswith('episode_')]
    next_episode = len(existing_episodes)
    
    print(f"\n📁 保存目录: {base_dir}")
    print(f"📊 已有{len(existing_episodes)}个episode，将从episode_{next_episode:03d}开始")
    
    # 创建环境
    print(f"\n🌍 创建MineDojo环境...")
    print(f"   fast_reset={fast_reset} ({'同一世界' if fast_reset else '每次新世界'})")
    
    env = minedojo.make(
        task_id="harvest_1_log",
        image_size=(160, 256),
        seed=None,
        fast_reset=fast_reset,
    )
    
    # 初始化控制器
    controller = RealtimeKeyController(camera_delta=camera_delta)
    
    # 录制参数
    frame_delay = 1.0 / fps  # 50ms @ 20 FPS
    
    print(f"\n⏱️  录制帧率: {fps} FPS (每帧{frame_delay*1000:.0f}ms)")
    print(f"⚠️  注意: 录制将自动进行，按住按键即可持续动作")
    print(f"\n按Enter键开始录制第一个episode...")
    input()
    
    episode_idx = next_episode
    
    try:
        while True:
            # 重置环境
            print(f"\n{'='*80}")
            print(f"🎬 开始录制 Episode {episode_idx:03d}")
            print(f"{'='*80}")
            
            obs_dict = env.reset()
            obs = obs_dict['rgb']
            
            print(f"✅ 环境已重置")
            print(f"📝 目标: 砍树获得1个木头（或手动中断）")
            print(f"⏰ 最大帧数: {max_frames}")
            print(f"\n开始录制...\n")
            
            # 存储当前episode的数据
            frames = []
            actions_list = []
            
            # 重置控制器的重试标志
            controller.reset_retry_flag()
            
            # 帧计数
            frame_count = 0
            start_time = time.time()
            
            # 主循环
            done = False
            while frame_count < max_frames and not done:
                loop_start = time.time()
                
                # 获取当前动作（基于实时按键状态）
                action = controller.get_action()
                action_desc = controller.decode_action(action)
                
                # 执行动作
                obs_dict, reward, done, info = env.step(action)
                obs = obs_dict['rgb']
                
                # 保存数据
                frames.append(obs.copy())
                actions_list.append((action.copy(), action_desc))
                
                frame_count += 1
                
                # 准备显示
                display_obs = obs.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
                display_obs = cv2.cvtColor(display_obs, cv2.COLOR_RGB2BGR)
                
                # 放大显示
                display_obs = cv2.resize(display_obs, (512, 320))
                
                # 添加调试信息
                info_y = 30
                cv2.putText(display_obs, f"Episode: {episode_idx:03d} | Frame: {frame_count}/{max_frames}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                info_y += 25
                cv2.putText(display_obs, f"Action: {action_desc}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                info_y += 25
                cv2.putText(display_obs, f"Raw: {action}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                info_y += 25
                cv2.putText(display_obs, f"Reward: {reward:.3f} | Done: {done}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # 显示按键提示
                info_y += 30
                cv2.putText(display_obs, "Q: Retry | ESC: Exit",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 显示
                cv2.imshow('MineDojo Recording (Realtime Mode)', display_obs)
                cv2.waitKey(1)  # 只用于刷新窗口，不用于检测按键
                
                # 检查控制信号
                if controller.should_quit:
                    print(f"\n⚠️  用户按下ESC，退出录制")
                    print(f"⚠️  当前episode不保存")
                    env.close()
                    controller.stop()
                    cv2.destroyAllWindows()
                    return
                
                if controller.should_retry:
                    print(f"\n🔄 用户按下Q，重新录制episode {episode_idx:03d}")
                    print(f"⚠️  当前数据不保存")
                    break
                
                # 维持帧率
                elapsed = time.time() - loop_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
                # 实时统计
                if frame_count % 20 == 0 or done:
                    idle_count = sum(1 for _, desc in actions_list if desc == "IDLE")
                    idle_pct = (idle_count / frame_count) * 100 if frame_count > 0 else 0
                    elapsed_total = time.time() - start_time
                    actual_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                    
                    print(f"[{elapsed_total:6.1f}s] 帧{frame_count:4d}: {action_desc:<30} | "
                          f"IDLE: {idle_count}/{frame_count} ({idle_pct:4.1f}%) | "
                          f"FPS: {actual_fps:4.1f}")
            
            # 检查是否需要重试
            if controller.should_retry:
                controller.reset_retry_flag()
                continue
            
            # Episode完成
            if done:
                print(f"\n✅ 任务完成！ (用时 {time.time()-start_time:.1f}秒，共{frame_count}帧)")
            else:
                print(f"\n⏸️  达到最大帧数 {max_frames}")
            
            # 统计
            idle_count = sum(1 for _, desc in actions_list if desc == "IDLE")
            idle_pct = (idle_count / len(actions_list)) * 100 if actions_list else 0
            
            print(f"\n📊 Episode {episode_idx:03d} 统计:")
            print(f"   总帧数: {len(frames)}")
            print(f"   静态帧: {idle_count} ({idle_pct:.1f}%)")
            print(f"   动作帧: {len(frames) - idle_count} ({100-idle_pct:.1f}%)")
            
            # 保存数据
            episode_dir = os.path.join(base_dir, f"episode_{episode_idx:03d}")
            os.makedirs(episode_dir, exist_ok=True)
            
            print(f"\n💾 保存数据到 {episode_dir}...")
            
            # 保存所有帧（PNG + NPY）
            for i, (frame, (action, action_desc)) in enumerate(zip(frames, actions_list)):
                # PNG for visualization
                frame_img = Image.fromarray(frame.transpose(1, 2, 0))
                frame_img.save(os.path.join(episode_dir, f"frame_{i:04d}.png"))
                
                # NPY for BC training
                np.save(
                    os.path.join(episode_dir, f"frame_{i:04d}.npy"),
                    {'observation': frame, 'action': action}
                )
            
            # 保存metadata
            metadata_path = os.path.join(episode_dir, "metadata.txt")
            with open(metadata_path, 'w') as f:
                f.write(f"Episode: {episode_idx:03d}\n")
                f.write(f"Total Frames: {len(frames)}\n")
                f.write(f"IDLE Frames: {idle_count} ({idle_pct:.1f}%)\n")
                f.write(f"Action Frames: {len(frames) - idle_count} ({100-idle_pct:.1f}%)\n")
                f.write(f"Task Completed: {done}\n")
                f.write(f"Recording FPS: {fps}\n")
                f.write(f"Camera Delta: {camera_delta}\n")
            
            # 保存actions_log.txt
            actions_log_path = os.path.join(episode_dir, "actions_log.txt")
            with open(actions_log_path, 'w') as f:
                f.write(f"Episode {episode_idx:03d} - Action Log\n")
                f.write(f"Total Frames: {len(actions_list)}\n")
                f.write(f"IDLE Frames: {idle_count}\n")
                f.write(f"{'-'*80}\n\n")
                
                for i, (action, action_desc) in enumerate(actions_list):
                    f.write(f"Frame {i:04d}: {action} -> {action_desc}\n")
            
            print(f"✅ 保存完成:")
            print(f"   - {len(frames)} 个 .png 图像文件")
            print(f"   - {len(frames)} 个 .npy 数据文件")
            print(f"   - metadata.txt")
            print(f"   - actions_log.txt")
            
            # 询问是否继续
            print(f"\n{'='*80}")
            print(f"录制完成！")
            print(f"按Enter继续录制下一个episode，或按Ctrl+C退出...")
            print(f"{'='*80}\n")
            
            try:
                input()
                episode_idx += 1
            except KeyboardInterrupt:
                print(f"\n\n⚠️  用户中断，停止录制")
                break
    
    finally:
        env.close()
        controller.stop()
        cv2.destroyAllWindows()
        print(f"\n✅ 环境已关闭，录制结束")


def main():
    parser = argparse.ArgumentParser(description="录制手动砍树序列（实时模式 - pynput）")
    parser.add_argument("--base-dir", type=str, default="data/expert_demos",
                        help="保存目录（默认: data/expert_demos）")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="每个episode的最大帧数（默认: 1000）")
    parser.add_argument("--camera-delta", type=int, default=4,
                        help="相机灵敏度，范围1-12（默认: 4）")
    parser.add_argument("--fast-reset", action="store_true",
                        help="快速重置（同一世界）")
    parser.add_argument("--no-fast-reset", dest="fast_reset", action="store_false",
                        help="完全重置（每次新世界）")
    parser.add_argument("--fps", type=int, default=20,
                        help="录制帧率（默认: 20 FPS）")
    parser.set_defaults(fast_reset=False)
    
    args = parser.parse_args()
    
    # 验证参数
    if args.camera_delta < 1 or args.camera_delta > 12:
        print(f"⚠️  警告: camera_delta={args.camera_delta} 超出范围，已调整为4")
        args.camera_delta = 4
    
    if args.fps < 1 or args.fps > 60:
        print(f"⚠️  警告: fps={args.fps} 超出范围，已调整为20")
        args.fps = 20
    
    print("\n" + "=" * 80)
    print("🎬 MineDojo 实时录制工具 (pynput)")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - 保存目录: {args.base_dir}")
    print(f"  - 最大帧数: {args.max_frames}")
    print(f"  - 相机灵敏度: {args.camera_delta} (~{args.camera_delta*15}度/按键)")
    print(f"  - 录制帧率: {args.fps} FPS")
    print(f"  - 环境重置: {'同一世界 (fast_reset)' if args.fast_reset else '每次新世界 (slow reset)'}")
    print("=" * 80)
    
    record_chopping_sequence(
        base_dir=args.base_dir,
        max_frames=args.max_frames,
        camera_delta=args.camera_delta,
        fast_reset=args.fast_reset,
        fps=args.fps
    )


if __name__ == "__main__":
    main()

