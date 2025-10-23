#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
录制手动砍树序列（pygame实时模式）
使用pygame进行按键检测和画面显示 - 无需macOS辅助功能权限

优势:
- ✅ 无需macOS辅助功能权限
- ✅ 实时按键检测，按住W键每帧都检测到
- ✅ 支持多键同时检测（W+F边前进边攻击）
- ✅ 静态帧大幅减少（< 30%）
"""

import os
import sys
# 添加项目根目录到Python路径 (tools/dagger/xxx.py -> tools/dagger -> tools -> project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import minedojo
import numpy as np
from PIL import Image
import time
import pygame
import argparse

class PygameController:
    """
    Pygame实时控制器 - 无需特殊权限
    同时处理按键检测和画面显示
    """
    
    def __init__(self, camera_delta=4, display_size=(800, 600), mouse_sensitivity=0.2, fullscreen=False):
        """
        初始化pygame控制器
        
        Args:
            camera_delta: 相机转动角度增量（键盘）
            display_size: pygame窗口大小
            mouse_sensitivity: 鼠标灵敏度（0.1-2.0）
            fullscreen: 是否全屏显示（默认False）
        """
        # 初始化pygame
        pygame.init()
        
        # 设置显示模式
        self.fullscreen = fullscreen
        if fullscreen:
            # 全屏模式 - 使用当前屏幕分辨率
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.display_size = self.screen.get_size()
        else:
            # 窗口模式
            self.screen = pygame.display.set_mode(display_size)
            self.display_size = display_size
        
        pygame.display.set_caption("MineDojo Recording (Pygame+Mouse) - Press Q to retry, ESC to exit")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # 相机参数
        self.camera_delta = camera_delta
        self.mouse_sensitivity = mouse_sensitivity
        
        # 鼠标控制
        pygame.mouse.set_visible(True)  # 显示鼠标
        self.mouse_captured = False  # 鼠标是否被捕获
        self.last_mouse_pos = None
        self.mouse_initialized = False  # 新增：是否已初始化鼠标位置（避免首次移动被误读）
        
        # 启用鼠标锁定（限制鼠标在窗口内）
        pygame.event.set_grab(True)  # 锁定鼠标在窗口内
        print("🔒 鼠标已锁定在窗口内（按ESC或Q解除锁定）")
        
        # 控制标志
        self.should_quit = False
        self.should_retry = False
        
        print("\n" + "=" * 80)
        print("🎮 Pygame实时录制模式")
        print("=" * 80)
        print("\n✅ 优势: 无需macOS辅助功能权限！")
        print("\n📌 录制方式:")
        print("  - pygame窗口显示游戏画面")
        print("  - 按住W键会持续前进，每帧都检测")
        print("  - 支持多键同时按下（如W+左键）")
        print("  - 录制以20 FPS速度进行")
        print("  - 🔒 鼠标已锁定在窗口内（不会移出画面）")
        print("\n移动控制:")
        print("  W - 前进 | S - 后退 | A - 左移 | D - 右移 | Space - 跳跃")
        print("\n相机控制:")
        print("  鼠标移动 - 转动视角（快速、大角度）⭐")
        print("  方向键 ↑↓←→ - 转动视角（精确、小角度，角度=1°）🎯")
        print("\n攻击:")
        print("  鼠标左键 - 攻击/挖掘（砍树）")
        print("\n系统:")
        print("  Q - 重新录制当前episode")
        print("  ESC - 退出程序（会自动解除鼠标锁定）")
        if fullscreen:
            print("  F11 - 退出全屏")
        else:
            print("  F11 - 切换全屏")
        print("\n" + "=" * 80)
        if fullscreen:
            print(f"显示模式: 全屏 ({self.display_size[0]}x{self.display_size[1]})")
        else:
            print(f"显示模式: 窗口 ({self.display_size[0]}x{self.display_size[1]})")
        print(f"鼠标灵敏度: {mouse_sensitivity:.2f}")
        print(f"鼠标锁定: ✅ 已启用（鼠标不会移出窗口）")
        print("=" * 80 + "\n")
    
    def process_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.should_quit = True
                elif event.key == pygame.K_q:
                    self.should_retry = True
                elif event.key == pygame.K_F11:
                    # F11切换全屏
                    self.toggle_fullscreen()
    
    def toggle_fullscreen(self):
        """切换全屏/窗口模式"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            # 切换到全屏
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.display_size = self.screen.get_size()
            print(f"\n✅ 已切换到全屏模式 ({self.display_size[0]}x{self.display_size[1]})")
            print("   按F11退出全屏\n")
        else:
            # 切换到窗口模式
            default_size = (800, 600)
            self.screen = pygame.display.set_mode(default_size)
            self.display_size = default_size
            print(f"\n✅ 已切换到窗口模式 ({self.display_size[0]}x{self.display_size[1]})")
            print("   鼠标已锁定在窗口内，按F11切换回全屏\n")
        
        # 重置鼠标状态（切换显示模式后）
        self.reset_mouse_state()
        
        # 重新启用鼠标锁定（切换显示模式后需要重新设置）
        pygame.event.set_grab(True)
    
    def get_action(self):
        """
        根据当前按键和鼠标状态生成MineDojo动作
        
        Returns:
            action: np.array([8], dtype=np.int32)
        """
        action = np.array([0, 0, 0, 12, 12, 0, 0, 0], dtype=np.int32)
        
        # 获取当前所有按键状态
        keys = pygame.key.get_pressed()
        
        # 移动
        if keys[pygame.K_w]:
            action[0] = 1  # forward
        elif keys[pygame.K_s]:
            action[0] = 2  # back
        
        if keys[pygame.K_a]:
            action[1] = 1  # left
        elif keys[pygame.K_d]:
            action[1] = 2  # right
        
        # 跳跃
        if keys[pygame.K_SPACE]:
            action[2] = 1
        
        # === 方向键精确控制相机（小角度）===
        arrow_key_delta = 1  # 方向键移动角度（更小，更精确）
        arrow_key_used = False
        
        if keys[pygame.K_UP]:
            action[3] = 12 - arrow_key_delta  # 向上看
            arrow_key_used = True
        elif keys[pygame.K_DOWN]:
            action[3] = 12 + arrow_key_delta  # 向下看
            arrow_key_used = True
        
        if keys[pygame.K_LEFT]:
            action[4] = 12 - arrow_key_delta  # 向左看
            arrow_key_used = True
        elif keys[pygame.K_RIGHT]:
            action[4] = 12 + arrow_key_delta  # 向右看
            arrow_key_used = True
        
        # === 鼠标控制相机（仅在方向键未使用时）===
        # 优先级: 方向键 > 鼠标
        if not arrow_key_used:
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            # 首次获取鼠标位置，不计算移动（避免启动时的鼠标移动被误读）
            if not self.mouse_initialized:
                self.last_mouse_pos = mouse_pos
                self.mouse_initialized = True
            elif self.last_mouse_pos is not None:
                # 计算鼠标移动
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                
                # 将鼠标移动转换为相机动作
                # dx: 正值=向右看，负值=向左看
                # dy: 正值=向下看，负值=向上看
                
                # Yaw (左右) - dimension 4
                yaw_delta = int(dx * self.mouse_sensitivity)
                yaw_delta = max(-12, min(12, yaw_delta))  # 限制范围
                action[4] = 12 + yaw_delta
                
                # Pitch (上下) - dimension 3
                pitch_delta = int(dy * self.mouse_sensitivity)
                pitch_delta = max(-12, min(12, pitch_delta))  # 限制范围
                action[3] = 12 + pitch_delta
                
                # 更新鼠标位置
                self.last_mouse_pos = mouse_pos
            
            # 鼠标左键攻击
            if mouse_buttons[0]:  # 左键
                action[5] = 3  # attack
        else:
            # 如果使用了方向键，仍然允许鼠标左键攻击
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0]:
                action[5] = 3  # attack
        
        return action
    
    def decode_action(self, action):
        """将动作数组转换为可读描述"""
        parts = []
        
        if action[0] == 1:
            parts.append("Forward")
        elif action[0] == 2:
            parts.append("Back")
        
        if action[1] == 1:
            parts.append("Left")
        elif action[1] == 2:
            parts.append("Right")
        
        if action[2] == 1:
            parts.append("Jump")
        
        if action[3] != 12 or action[4] != 12:
            pitch_delta = action[3] - 12
            yaw_delta = action[4] - 12
            parts.append(f"Camera(p={pitch_delta:+d},y={yaw_delta:+d})")
        
        if action[5] == 3:
            parts.append("ATTACK")
        
        return " + ".join(parts) if parts else "IDLE"
    
    def display_frame(self, obs, episode_idx, frame_count, max_frames, action_desc, reward, done):
        """
        在pygame窗口中显示游戏画面和信息
        
        Args:
            obs: 观察图像 (C, H, W)
            episode_idx: Episode索引
            frame_count: 当前帧数
            max_frames: 最大帧数
            action_desc: 动作描述
            reward: 奖励
            done: 是否完成
        """
        # 清屏
        self.screen.fill((30, 30, 30))
        
        # 转换并显示游戏画面
        # MineDojo: (C, H, W) -> (H, W, C) for pygame
        game_img = obs.transpose(1, 2, 0)  # (160, 256, 3)
        
        # 放大到合适大小
        scale_factor = 3
        game_surface = pygame.surfarray.make_surface(game_img.transpose(1, 0, 2))  # pygame需要(W,H,C)
        game_surface = pygame.transform.scale(game_surface, 
                                              (game_img.shape[1] * scale_factor, 
                                               game_img.shape[0] * scale_factor))
        
        # 显示游戏画面（居中上方）
        game_rect = game_surface.get_rect(center=(self.screen.get_width() // 2, 240))
        self.screen.blit(game_surface, game_rect)
        
        # 显示信息
        y = 10
        
        # Episode和帧数信息
        info_text = self.font_large.render(f"Episode: {episode_idx:03d} | Frame: {frame_count}/{max_frames}", 
                                           True, (0, 255, 0))
        self.screen.blit(info_text, (10, y))
        y += 40
        
        # 当前动作
        action_text = self.font_small.render(f"Action: {action_desc}", True, (0, 255, 255))
        self.screen.blit(action_text, (10, y))
        y += 30
        
        # 奖励和完成状态
        status_text = self.font_small.render(f"Reward: {reward:.3f} | Done: {done}", 
                                            True, (255, 255, 0))
        self.screen.blit(status_text, (10, y))
        
        # 控制提示（底部）
        y = self.screen.get_height() - 60
        hint_text = self.font_small.render("Q: Retry | ESC: Exit | Keep pygame window focused!", 
                                          True, (255, 255, 255))
        self.screen.blit(hint_text, (10, y))
        
        # 刷新显示
        pygame.display.flip()
    
    def reset_retry_flag(self):
        """重置重试标志"""
        self.should_retry = False
    
    def reset_mouse_state(self):
        """重置鼠标状态（每个episode开始时调用）"""
        self.mouse_initialized = False
        self.last_mouse_pos = None
    
    def quit(self):
        """退出pygame"""
        # 解除鼠标锁定
        pygame.event.set_grab(False)
        pygame.quit()
        print("🔓 鼠标锁定已解除")


def record_chopping_sequence(
    base_dir="data/expert_demos",
    max_frames=1000,
    camera_delta=4,
    mouse_sensitivity=0.2,
    fast_reset=False,
    fps=20,
    skip_idle_frames=True,
    fullscreen=False
):
    """
    录制手动砍树序列（pygame实时模式）
    
    Args:
        base_dir: 保存目录
        max_frames: 每个episode的最大帧数
        camera_delta: 相机灵敏度（键盘）
        mouse_sensitivity: 鼠标灵敏度
        fast_reset: 是否快速重置
        fps: 录制帧率
        skip_idle_frames: 是否跳过静止帧（不保存IDLE帧）
        fullscreen: 是否全屏显示（默认False）
    """
    # 检测已有episode
    os.makedirs(base_dir, exist_ok=True)
    existing_episodes = [d for d in os.listdir(base_dir) if d.startswith('episode_')]
    next_episode = len(existing_episodes)
    
    print(f"\n📁 保存目录: {base_dir}")
    print(f"📊 已有{len(existing_episodes)}个episode，将从episode_{next_episode:03d}开始")
    
    # 创建环境
    print(f"\n🌍 创建MineDojo环境...")
    env = minedojo.make(
        task_id="harvest_1_log",
        image_size=(160, 256),
        seed=None,
        fast_reset=fast_reset,
    )
    
    # 添加任务特定wrapper（支持所有类型原木）
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.task_wrappers import HarvestLogWrapper
    env = HarvestLogWrapper(env, required_logs=1, verbose=True)
    
    # 初始化pygame控制器
    controller = PygameController(
        camera_delta=camera_delta, 
        mouse_sensitivity=mouse_sensitivity,
        fullscreen=fullscreen
    )
    
    # 录制参数
    frame_delay = 1.0 / fps
    
    print(f"\n⏱️  录制帧率: {fps} FPS (每帧{frame_delay*1000:.0f}ms)")
    print(f"⚠️  注意: 请保持pygame窗口为焦点状态")
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
            print(f"📝 目标: 砍树获得1个木头")
            print(f"⏰ 最大帧数: {max_frames}")
            print(f"\n开始录制...\n")
            
            # 存储数据
            frames = []
            actions_list = []
            
            # 重置控制器标志
            controller.reset_retry_flag()
            controller.reset_mouse_state()  # 重置鼠标状态，避免记录启动时的鼠标移动
            
            # 帧计数
            frame_count = 0
            start_time = time.time()
            
            # 主循环
            done = False
            while frame_count < max_frames and not done:
                loop_start = time.time()
                
                # 处理pygame事件
                controller.process_events()
                
                # 检查退出信号
                if controller.should_quit:
                    print(f"\n⚠️  用户按下ESC，退出录制")
                    env.close()
                    controller.quit()
                    return
                
                if controller.should_retry:
                    print(f"\n🔄 用户按下Q，重新录制episode {episode_idx:03d}")
                    break
                
                # 获取当前动作
                action = controller.get_action()
                action_desc = controller.decode_action(action)
                
                # 执行动作
                obs_dict, reward, done, info = env.step(action)
                obs = obs_dict['rgb']
                
                # 保存数据（根据skip_idle_frames设置）
                is_idle = (action_desc == "IDLE")
                
                if not skip_idle_frames or not is_idle:
                    # 不跳过，或者不是IDLE帧 -> 保存
                    frames.append(obs.copy())
                    actions_list.append((action.copy(), action_desc))
                
                frame_count += 1
                
                # 显示画面
                controller.display_frame(obs, episode_idx, frame_count, max_frames, 
                                       action_desc, reward, done)
                
                # 维持帧率
                controller.clock.tick(fps)
                
                # 实时统计
                if frame_count % 20 == 0 or done:
                    idle_count = sum(1 for _, desc in actions_list if desc == "IDLE")
                    idle_pct = (idle_count / frame_count) * 100
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"[{elapsed:6.1f}s] 帧{frame_count:4d}: {action_desc:<30} | "
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
            saved_idle_count = sum(1 for _, desc in actions_list if desc == "IDLE")
            saved_idle_pct = (saved_idle_count / len(actions_list)) * 100 if actions_list else 0
            
            print(f"\n📊 Episode {episode_idx:03d} 统计:")
            print(f"   执行总帧数: {frame_count}")
            print(f"   保存帧数: {len(frames)}")
            if skip_idle_frames:
                skipped_frames = frame_count - len(frames)
                print(f"   跳过静止帧: {skipped_frames} (未保存)")
                print(f"   保存的静止帧: {saved_idle_count} ({saved_idle_pct:.1f}%)")
            else:
                print(f"   静态帧: {saved_idle_count} ({saved_idle_pct:.1f}%)")
            print(f"   动作帧: {len(frames) - saved_idle_count}")
            
            # 保存数据
            episode_dir = os.path.join(base_dir, f"episode_{episode_idx:03d}")
            os.makedirs(episode_dir, exist_ok=True)
            
            print(f"\n💾 保存数据到 {episode_dir}...")
            
            # 保存所有帧
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
                f.write(f"Executed Frames: {frame_count}\n")
                f.write(f"Saved Frames: {len(frames)}\n")
                if skip_idle_frames:
                    f.write(f"Skipped IDLE Frames: {frame_count - len(frames)} (not saved)\n")
                    f.write(f"Saved IDLE Frames: {saved_idle_count} ({saved_idle_pct:.1f}%)\n")
                else:
                    f.write(f"IDLE Frames: {saved_idle_count} ({saved_idle_pct:.1f}%)\n")
                f.write(f"Action Frames: {len(frames) - saved_idle_count}\n")
                f.write(f"Task Completed: {done}\n")
                f.write(f"Recording FPS: {fps}\n")
                f.write(f"Mouse Sensitivity: {mouse_sensitivity}\n")
                f.write(f"Skip IDLE Frames: {skip_idle_frames}\n")
                f.write(f"Method: pygame+mouse (no macOS permission needed)\n")
            
            # 保存actions_log
            actions_log_path = os.path.join(episode_dir, "actions_log.txt")
            with open(actions_log_path, 'w') as f:
                f.write(f"Episode {episode_idx:03d} - Action Log\n")
                f.write(f"Saved Frames: {len(actions_list)}\n")
                if skip_idle_frames:
                    f.write(f"Note: IDLE frames were skipped during recording\n")
                f.write(f"Saved IDLE Frames: {saved_idle_count}\n")
                f.write(f"{'-'*80}\n\n")
                
                for i, (action, action_desc) in enumerate(actions_list):
                    f.write(f"Frame {i:04d}: {action} -> {action_desc}\n")
            
            print(f"✅ 保存完成:")
            print(f"   - {len(frames)} 个 .png 图像文件")
            print(f"   - {len(frames)} 个 .npy 数据文件")
            print(f"   - metadata.txt")
            print(f"   - actions_log.txt")
            
            # 自动继续下一个episode
            print(f"\n{'='*80}")
            print(f"✅ Episode {episode_idx:03d} 录制完成！")
            print(f"{'='*80}")
            print(f"⏭️  准备录制下一个episode...")
            print(f"💡 提示: 按ESC可随时退出录制\n")
            
            # 等待2秒，让用户看到提示
            time.sleep(2)
            episode_idx += 1
    
    finally:
        env.close()
        controller.quit()
        print(f"\n✅ 环境已关闭，录制结束")


def main():
    parser = argparse.ArgumentParser(description="录制手动砍树序列（pygame实时模式）")
    parser.add_argument("--base-dir", type=str, default="data/expert_demos",
                        help="保存目录（默认: data/expert_demos）")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="每个episode的最大帧数（默认: 1000）")
    parser.add_argument("--camera-delta", type=int, default=4,
                        help="相机灵敏度（键盘），范围1-12（默认: 4）")
    parser.add_argument("--mouse-sensitivity", type=float, default=0.2,
                        help="鼠标灵敏度，范围0.1-2.0（默认: 0.2，已降低）")
    parser.add_argument("--skip-idle-frames", action="store_true", default=True,
                        help="跳过静止帧（不保存IDLE帧，默认: True）")
    parser.add_argument("--no-skip-idle-frames", dest="skip_idle_frames", action="store_false",
                        help="保存所有帧（包括IDLE帧）")
    parser.add_argument("--fast-reset", action="store_true",
                        help="快速重置（同一世界）")
    parser.add_argument("--no-fast-reset", dest="fast_reset", action="store_false",
                        help="完全重置（每次新世界）")
    parser.add_argument("--fps", type=int, default=20,
                        help="录制帧率（默认: 20 FPS）")
    parser.add_argument("--fullscreen", action="store_true",
                        help="全屏显示（解决鼠标移出窗口问题，推荐！）")
    parser.add_argument("--no-fullscreen", dest="fullscreen", action="store_false",
                        help="窗口模式（默认）")
    parser.set_defaults(fast_reset=False, fullscreen=False)
    
    args = parser.parse_args()
    
    # 验证参数
    if args.camera_delta < 1 or args.camera_delta > 12:
        print(f"⚠️  警告: camera_delta={args.camera_delta} 超出范围，已调整为4")
        args.camera_delta = 4
    
    if args.fps < 1 or args.fps > 60:
        print(f"⚠️  警告: fps={args.fps} 超出范围，已调整为20")
        args.fps = 20
    
    if args.mouse_sensitivity < 0.1 or args.mouse_sensitivity > 2.0:
        print(f"⚠️  警告: mouse_sensitivity={args.mouse_sensitivity} 超出范围，已调整为0.2")
        args.mouse_sensitivity = 0.2
    
    print("\n" + "=" * 80)
    print("🎬 MineDojo Pygame实时录制工具 (鼠标+键盘)")
    print("=" * 80)
    print(f"\n✅ 无需macOS辅助功能权限！")
    print(f"\n配置:")
    print(f"  - 保存目录: {args.base_dir}")
    print(f"  - 最大帧数: {args.max_frames}")
    print(f"  - 显示模式: {'全屏 (推荐！)' if args.fullscreen else '窗口'}")
    print(f"  - 鼠标灵敏度: {args.mouse_sensitivity} (已优化)")
    print(f"  - 录制帧率: {args.fps} FPS")
    print(f"  - 跳过静止帧: {'是 (不保存IDLE帧)' if args.skip_idle_frames else '否 (保存所有帧)'}")
    print(f"  - 环境重置: {'同一世界' if args.fast_reset else '每次新世界'}")
    if not args.fullscreen:
        print(f"\n💡 提示: 鼠标容易移出窗口？试试 --fullscreen 参数")
    print("=" * 80)
    
    record_chopping_sequence(
        base_dir=args.base_dir,
        max_frames=args.max_frames,
        camera_delta=args.camera_delta,
        mouse_sensitivity=args.mouse_sensitivity,
        fast_reset=args.fast_reset,
        fps=args.fps,
        skip_idle_frames=args.skip_idle_frames,
        fullscreen=args.fullscreen
    )


if __name__ == "__main__":
    main()

