#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本录制工具 - 仅支持 MineRL 环境
Sample Recording Tool - MineRL Only

用途: 手动录制训练样本，用于提取目标视觉嵌入和动作序列
支持: 从 config 读取任务配置，仅支持 MineRL 环境
输出: 标准化目录结构 data/train_samples/{task_id}/trial{num}/
      自动生成 visual_embeds.pkl

优势:
- ✅ 无需macOS辅助功能权限（pygame控制）
- ✅ 输出格式与评估结果统一
- ✅ 简化代码，专注于 MineRL 环境
- ✅ 自动创建标准化目录结构
- ✅ 录制完成后自动生成 visual_embeds.pkl
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import yaml
import numpy as np
from PIL import Image
import pygame
import gym
import logging
import torch as th

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 注册自定义MineRL环境
try:
    from src.envs.minerl_wrappers import register_minerl_harvest_default_env
    register_minerl_harvest_default_env()
    
    # 启用 minerl_harvest_default 模块的日志输出（查看奖励计算）
    minerl_logger = logging.getLogger('src.envs.minerl_harvest_default')
    minerl_logger.setLevel(logging.INFO)
    # 如果没有 handler，添加一个控制台 handler
    if not minerl_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[WRAPPER] %(message)s')
        console_handler.setFormatter(formatter)
        minerl_logger.addHandler(console_handler)
    
except ImportError:
    pass  # 如果没有MineRL环境定义，忽略


class PygameController:
    """
    Pygame实时控制器 - 专用于 MineRL 环境
    处理按键检测和画面显示
    """
    
    @staticmethod
    def _find_chinese_font():
        """
        查找系统中可用的中文字体
        
        Returns:
            str or None: 字体文件路径，如果找不到则返回None
        """
        # macOS 常见中文字体路径
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # 苹方（macOS默认）
            "/System/Library/Fonts/STHeiti Light.ttc",  # 华文黑体
            "/System/Library/Fonts/Supplemental/Songti.ttc",  # 宋体
            "/Library/Fonts/Arial Unicode.ttf",  # Arial Unicode MS
            "/System/Library/Fonts/Hiragino Sans GB.ttc",  # 冬青黑体
        ]
        
        # Linux 常见中文字体路径
        font_paths.extend([
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Droid Sans Fallback
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
        ])
        
        # Windows 常见中文字体路径
        font_paths.extend([
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ])
        
        # 尝试查找可用字体
        for font_path in font_paths:
            if Path(font_path).exists():
                return font_path
        
        return None
    
    def __init__(self, camera_delta=4, display_size=(800, 600), mouse_sensitivity=0.5, fullscreen=False):
        """
        初始化pygame控制器
        
        Args:
            camera_delta: 相机转动角度增量（键盘）
            display_size: pygame窗口大小
            mouse_sensitivity: 鼠标灵敏度（0.1-2.0，默认0.5）
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
        
        pygame.display.set_caption("Expert Demo Recording - Press Q to retry, ESC to exit")
        self.clock = pygame.time.Clock()
        
        # 加载中文字体
        chinese_font = self._find_chinese_font()
        if chinese_font:
            try:
                self.font_large = pygame.font.Font(chinese_font, 36)
                self.font_small = pygame.font.Font(chinese_font, 24)
                print(f"✓ 已加载中文字体: {Path(chinese_font).name}")
            except Exception as e:
                print(f"⚠️  加载中文字体失败: {e}，使用默认字体")
                self.font_large = pygame.font.Font(None, 36)
                self.font_small = pygame.font.Font(None, 24)
        else:
            print(f"⚠️  未找到中文字体，中文字符可能显示为方块")
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        
        # 相机参数
        self.camera_delta = camera_delta
        self.mouse_sensitivity = mouse_sensitivity
        
        # 鼠标控制
        pygame.mouse.set_visible(True)
        self.mouse_captured = False
        self.last_mouse_pos = None
        self.mouse_initialized = False
        
        # 启用鼠标锁定
        pygame.event.set_grab(True)
        
        # 控制标志
        self.should_quit = False
        self.should_retry = False
        
        print("\n" + "=" * 80)
        print("🎮 Pygame实时录制模式")
        print("=" * 80)
        print("\n✅ 优势: 无需macOS辅助功能权限！")
        print("\n📌 控制说明:")
        print("  移动: W/A/S/D | 跳跃: Space | 攻击: F/左键 | 使用: R/右键")
        print("  物品栏: E 打开/关闭 | 数字键 1-9 切换快捷栏")
        print("  相机: 鼠标移动 | 方向键（精确）")
        print(f"  鼠标灵敏度: {self.mouse_sensitivity:.2f} (可用--mouse-sensitivity调整)")
        print("  重录: Q | 退出: ESC | 全屏: F11")
        print("\n🔒 鼠标已锁定在窗口内")
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
                    self.toggle_fullscreen()
    
    def toggle_fullscreen(self):
        """切换全屏/窗口模式"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.display_size = self.screen.get_size()
            print(f"\n✅ 已切换到全屏模式 ({self.display_size[0]}x{self.display_size[1]})\n")
        else:
            default_size = (800, 600)
            self.screen = pygame.display.set_mode(default_size)
            self.display_size = default_size
            print(f"\n✅ 已切换到窗口模式 ({self.display_size[0]}x{self.display_size[1]})\n")
        
        self.reset_mouse_state()
        pygame.event.set_grab(True)
    
    def get_action(self):
        """
        根据当前按键和鼠标状态生成 MineRL 格式的动作
        
        Returns:
            dict: MineRL 动作字典
        """
        # 获取当前所有按键状态
        keys = pygame.key.get_pressed()
        
        # 初始化动作字典
        action = {
            'forward': 0,
            'back': 0,
            'left': 0,
            'right': 0,
            'jump': 0,
            'sneak': 0,
            'sprint': 0,
            'attack': 0,
            'use': 0,        # 使用/右键
            'inventory': 0,  # 物品栏
            'camera': [0.0, 0.0]  # [pitch, yaw]
        }
        
        # 移动
        if keys[pygame.K_w]:
            action['forward'] = 1
        if keys[pygame.K_s]:
            action['back'] = 1
        if keys[pygame.K_a]:
            action['left'] = 1
        if keys[pygame.K_d]:
            action['right'] = 1
        
        # 跳跃
        if keys[pygame.K_SPACE]:
            action['jump'] = 1
        
        # 攻击（F键或鼠标左键）
        if keys[pygame.K_f]:
            action['attack'] = 1
        
        # 使用（R键）
        if keys[pygame.K_r]:
            action['use'] = 1
        
        # 物品栏（E键）
        if keys[pygame.K_e]:
            action['inventory'] = 1
        
        # 方向键精确控制相机
        arrow_key_delta = 2.0  # MineRL使用度数
        arrow_key_used = False
        
        if keys[pygame.K_UP]:
            action['camera'][0] = -arrow_key_delta  # 向上看
            arrow_key_used = True
        elif keys[pygame.K_DOWN]:
            action['camera'][0] = arrow_key_delta  # 向下看
            arrow_key_used = True
        
        if keys[pygame.K_LEFT]:
            action['camera'][1] = -arrow_key_delta  # 向左看
            arrow_key_used = True
        elif keys[pygame.K_RIGHT]:
            action['camera'][1] = arrow_key_delta  # 向右看
            arrow_key_used = True
        
        # 鼠标控制相机
        if not arrow_key_used:
            mouse_buttons = pygame.mouse.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            
            if not self.mouse_initialized:
                self.last_mouse_pos = mouse_pos
                self.mouse_initialized = True
            elif self.last_mouse_pos is not None:
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                
                # MineRL相机：[-180, 180] 度
                # 放大系数降低到2以减少灵敏度
                yaw_delta = dx * self.mouse_sensitivity * 2
                pitch_delta = dy * self.mouse_sensitivity * 2
                
                # 限制范围
                action['camera'][0] = max(-90, min(90, pitch_delta))
                action['camera'][1] = max(-180, min(180, yaw_delta))
                
                self.last_mouse_pos = mouse_pos
            
            # 鼠标左键攻击
            if mouse_buttons[0]:
                action['attack'] = 1
            
            # 鼠标右键使用
            if mouse_buttons[2]:
                action['use'] = 1
        
        # 物品栏切换 (数字键 1-9)
        if keys[pygame.K_1]:
            action['hotbar.1'] = 1
        if keys[pygame.K_2]:
            action['hotbar.2'] = 1
        if keys[pygame.K_3]:
            action['hotbar.3'] = 1
        if keys[pygame.K_4]:
            action['hotbar.4'] = 1
        if keys[pygame.K_5]:
            action['hotbar.5'] = 1
        if keys[pygame.K_6]:
            action['hotbar.6'] = 1
        if keys[pygame.K_7]:
            action['hotbar.7'] = 1
        if keys[pygame.K_8]:
            action['hotbar.8'] = 1
        if keys[pygame.K_9]:
            action['hotbar.9'] = 1
        else:
            # 方向键时仍允许鼠标左键攻击和右键使用
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0]:
                action['attack'] = 1
            if mouse_buttons[2]:
                action['use'] = 1
        
        return action
    
    def decode_action(self, action):
        """将 MineRL 动作转换为可读描述"""
        parts = []
        
        if action['forward']:
            parts.append("Forward")
        if action['back']:
            parts.append("Back")
        if action['left']:
            parts.append("Left")
        if action['right']:
            parts.append("Right")
        if action['jump']:
            parts.append("Jump")
        if action['sneak']:
            parts.append("Sneak")
        if action['sprint']:
            parts.append("Sprint")
        if action['attack']:
            parts.append("ATTACK")
        if action.get('use', 0):
            parts.append("USE")
        if action.get('inventory', 0):
            parts.append("INVENTORY")
        
        # 相机
        pitch, yaw = action['camera']
        if abs(pitch) > 0.1 or abs(yaw) > 0.1:
            parts.append(f"Camera(p={pitch:+.1f},y={yaw:+.1f})")
        
        return " + ".join(parts) if parts else "IDLE"
    
    def _draw_health_bar(self, x, y, health_value, max_health=20.0):
        """
        绘制血量条（Minecraft风格）
        
        Args:
            x: X坐标
            y: Y坐标
            health_value: 当前血量
            max_health: 最大血量（默认20，即10颗心）
        """
        # 确保血量在有效范围内
        health_value = max(0.0, min(health_value, max_health))
        health_ratio = health_value / max_health
        
        # 血量条尺寸
        bar_width = 300
        bar_height = 25
        
        # 绘制背景（灰色）
        bg_rect = pygame.Rect(x, y, bar_width, bar_height)
        pygame.draw.rect(self.screen, (60, 60, 60), bg_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), bg_rect, 2)  # 边框
        
        # 绘制血量条（根据血量值使用不同颜色）
        if health_ratio > 0:
            health_bar_width = int(bar_width * health_ratio)
            health_rect = pygame.Rect(x, y, health_bar_width, bar_height)
            
            # 血量颜色：高血量=绿色，中等=黄色，低血量=红色
            if health_ratio > 0.6:
                color = (0, 200, 0)  # 绿色
            elif health_ratio > 0.3:
                color = (255, 200, 0)  # 黄色
            else:
                color = (255, 50, 50)  # 红色
            
            pygame.draw.rect(self.screen, color, health_rect)
        
        # 绘制血量文字
        hearts = health_value / 2.0  # Minecraft: 1颗心 = 2点血量
        health_text = f"❤️ {hearts:.1f} / {max_health/2.0:.0f} ({health_value:.1f}/{max_health:.0f})"
        
        # 选择文字颜色（根据血量）
        if health_ratio > 0.6:
            text_color = (100, 255, 100)
        elif health_ratio > 0.3:
            text_color = (255, 255, 100)
        else:
            text_color = (255, 100, 100)
        
        text_surface = self.font_small.render(health_text, True, text_color)
        # 文字显示在血量条中央
        text_rect = text_surface.get_rect(center=(x + bar_width // 2, y + bar_height // 2))
        self.screen.blit(text_surface, text_rect)
    
    def display_frame(self, obs_dict, task_id, trial_idx, frame_count, max_frames, action_desc, done):
        """在pygame窗口中显示游戏画面和信息"""
        self.screen.fill((30, 30, 30))
        
        # 提取图像观察
        if isinstance(obs_dict, dict):
            obs = obs_dict.get('pov', obs_dict.get('rgb'))
            inventory = obs_dict.get('inventory', {})
            
            # 提取生命值（MineRL中在嵌套的 'life_stats' 字典中）
            health = None
            if 'life_stats' in obs_dict and isinstance(obs_dict['life_stats'], dict):
                # MineRL标准格式：obs_dict['life_stats']['life']
                health = obs_dict['life_stats'].get('life', None)
            elif 'life' in obs_dict:
                # 备用格式：直接在顶层
                health = obs_dict['life']
            elif 'health' in obs_dict:
                # 备用格式：health字段
                health = obs_dict['health']
        else:
            obs = obs_dict
            inventory = {}
            health = None
        
        # 转换并显示游戏画面
        # 处理不同的图像格式：MineDojo (C, H, W) vs MineRL (H, W, C)
        if obs.shape[0] == 3:
            # MineDojo 格式: (C, H, W) -> (H, W, C)
            game_img = obs.transpose(1, 2, 0)
        elif len(obs.shape) == 3 and obs.shape[2] == 3:
            # MineRL 格式: (H, W, C) - 已经是正确格式
            game_img = obs
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
        
        scale_factor = 3
        game_surface = pygame.surfarray.make_surface(game_img.transpose(1, 0, 2))
        game_surface = pygame.transform.scale(game_surface, 
                                              (game_img.shape[1] * scale_factor, 
                                               game_img.shape[0] * scale_factor))
        
        game_rect = game_surface.get_rect(center=(self.screen.get_width() // 2, 240))
        self.screen.blit(game_surface, game_rect)
        
        # 显示信息
        y = 10
        
        # 任务和试验信息
        info_text = self.font_large.render(f"Task: {task_id} | Trial: {trial_idx} | Frame: {frame_count}/{max_frames}", 
                                           True, (0, 255, 0))
        self.screen.blit(info_text, (10, y))
        y += 40
        
        # 显示血量（如果可用）
        if health is not None:
            # 处理numpy类型
            if hasattr(health, 'item'):
                health_value = health.item()
            else:
                health_value = float(health)
            
            # 血量条显示
            self._draw_health_bar(10, y, health_value)
            y += 35
        else:
            # 如果没有血量数据，显示提示（仅在combat任务中提示）
            if 'combat' in task_id.lower():
                no_health_text = self.font_small.render("❤️ 血量: N/A (环境未提供)", True, (150, 150, 150))
                self.screen.blit(no_health_text, (10, y))
                y += 30
        
        # 当前动作
        action_text = self.font_small.render(f"Action: {action_desc}", True, (0, 255, 255))
        self.screen.blit(action_text, (10, y))
        y += 30
        
        # 完成状态
        status_text = self.font_small.render(f"Done: {done}", True, (255, 255, 0))
        self.screen.blit(status_text, (10, y))
        y += 30
        
        # 显示库存信息（所有非零物品，优先显示武器）
        if inventory:
            items_to_show = []
            # 优先显示武器和工具
            priority_items = ['stone_sword', 'wooden_sword', 'iron_sword', 'diamond_sword',
                             'stone_pickaxe', 'wooden_pickaxe', 'iron_pickaxe', 'diamond_pickaxe',
                             'shield', 'bow', 'arrow', 'shears']
            
            # 先显示武器/工具
            for item in priority_items:
                if item in inventory:
                    count = inventory[item]
                    if hasattr(count, 'item'):
                        count = count.item()
                    count = int(count)
                    if count > 0:
                        items_to_show.append(f"⚔️{item}:{count}")
            
            # 再显示其他有数量的物品
            for item, count in inventory.items():
                if item not in priority_items:
                    if hasattr(count, 'item'):
                        count = count.item()
                    count = int(count)
                    if count > 0:
                        items_to_show.append(f"{item}:{count}")
            
            if items_to_show:
                inventory_text = self.font_large.render(f"📦 {', '.join(items_to_show[:5])}", 
                                                       True, (255, 200, 0))
                self.screen.blit(inventory_text, (10, y))
                y += 35
                # 装备提示
                equip_hint = self.font_small.render("💡 按 1-9 切换物品栏 | F 攻击", True, (200, 200, 100))
                self.screen.blit(equip_hint, (10, y))
                y += 25
            else:
                empty_text = self.font_small.render(f"📦 Inventory: (empty)", 
                                                   True, (150, 150, 150))
                self.screen.blit(empty_text, (10, y))
                y += 30
        
        # 控制提示
        y = self.screen.get_height() - 60
        hint_text = self.font_small.render("Q: Retry | ESC: Exit | Keep window focused!", 
                                          True, (255, 255, 255))
        self.screen.blit(hint_text, (10, y))
        
        pygame.display.flip()
    
    def reset_retry_flag(self):
        """重置重试标志"""
        self.should_retry = False
    
    def reset_mouse_state(self):
        """重置鼠标状态"""
        self.mouse_initialized = False
        self.last_mouse_pos = None
    
    def quit(self):
        """退出pygame"""
        pygame.event.set_grab(False)
        pygame.quit()
        print("🔓 鼠标锁定已解除")


class SampleRecorder:
    """训练样本录制器"""
    
    def __init__(self, config_path: str, base_dir: str = "data/train_samples"):
        """
        初始化录制器
        
        Args:
            config_path: 配置文件路径
            base_dir: 输出基础目录（默认 data/train_samples）
        """
        self.config_path = config_path
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 读取全局配置（带默认值）
        global_config = self.config.get('global', {})
        self.default_image_size = global_config.get('image_size', [320, 640])
        self.default_max_steps = global_config.get('max_steps', 6000)
        self.default_fps = global_config.get('fps', 20)
        
        # 当前任务会话的目录和数据（用于生成result.json）
        self.current_task_dir = None
        self.current_task_id = None
        self.current_instruction = None
        self.recorded_trials = []  # 存储所有已录制的trial结果
        
        # MineCLIP（延迟加载）
        self._mineclip = None
        
        print(f"\n📄 已加载配置文件: {config_path}")
        print(f"⚙️  全局参数: image_size={self.default_image_size}, max_steps={self.default_max_steps}, fps={self.default_fps}")
    
    def get_available_tasks(self) -> List[str]:
        """获取所有可用任务ID"""
        if 'tasks' in self.config:
            return list(self.config['tasks'].keys())
        return []
    
    def get_task_config(self, task_id: str) -> Dict[str, Any]:
        """获取任务配置"""
        if 'tasks' not in self.config or task_id not in self.config['tasks']:
            raise ValueError(f"Task {task_id} not found in config")
        
        task_config = self.config['tasks'][task_id].copy()
        
        # 兼容不同的指令字段名称 (en_instruction 或 instruction)
        if 'en_instruction' in task_config and 'instruction' not in task_config:
            task_config['instruction'] = task_config['en_instruction']
        
        # 确保有必要的字段
        if 'instruction' not in task_config:
            raise ValueError(f"Task {task_id} missing 'instruction' or 'en_instruction' field")
        
        if 'metadata' not in task_config or 'env_name' not in task_config['metadata']:
            # 默认使用 MineDojo task_id 作为环境
            task_config.setdefault('metadata', {})['env_name'] = task_id
        
        return task_config
    
    def create_environment(self, task_id: str, task_config: Dict[str, Any]):
        """创建 MineRL 环境（仅支持 MineRL）"""
        # 获取环境名称（优先从顶层读取，兼容旧配置从metadata读取）
        env_name = task_config.get('env_name', task_config.get('metadata', {}).get('env_name', None))
        
        if not env_name or not env_name.startswith('MineRL'):
            raise ValueError(
                f"\n❌ 此录制工具仅支持 MineRL 环境！\n"
                f"任务 '{task_id}' 的 env_name='{env_name}' 不是有效的 MineRL 环境\n"
                f"请确保配置中 env_name 以 'MineRL' 开头"
            )
        
        # 获取环境配置
        env_config = task_config.get('env_config', {})
        
        # 🔄 导入物品名称转换工具
        from src.envs.env_bridge import convert_initial_inventory, convert_reward_config
        
        # 使用配置的图像尺寸
        image_height, image_width = self.default_image_size
        
        print(f"\n🌍 创建 MineRL 环境: {env_name}")
        print(f"📐 图像尺寸: {image_height}x{image_width}")
        
        # 显示环境配置
        if env_config:
            if 'specified_biome' in env_config:
                print(f"🌲 指定群系: {env_config['specified_biome']}")
        
        try:
            # 检查是否是自定义MineRL环境（需要特殊参数）
            if env_name == 'MineRLHarvestDefaultEnv-v0':
                # 自定义harvest环境
                
                # 构建环境参数
                env_kwargs = {
                    'image_size': self.default_image_size,
                    'max_episode_steps': self.default_max_steps,
                }
                
                # 添加群系配置
                if 'specified_biome' in env_config:
                    env_kwargs['specified_biome'] = env_config['specified_biome']
                
                # 添加时间条件 (作为字典传递)
                if 'time_condition' in env_config:
                    env_kwargs['time_condition'] = env_config['time_condition']
                    start_time = env_config['time_condition'].get('start_time', 6000)
                    if start_time >= 13000:
                        print(f"🌙 夜间模式: start_time={start_time}")
                    else:
                        print(f"☀️ 白天模式: start_time={start_time}")
                
                # 添加生成条件 (作为字典传递)
                if 'spawning_condition' in env_config:
                    env_kwargs['spawning_condition'] = env_config['spawning_condition']
                    if env_config['spawning_condition'].get('allow_spawning', True):
                        print(f"🐾 怪物生成: 已启用")
                
                # 添加初始库存配置
                if 'initial_inventory' in env_config:
                    # 🔄 转换物品名称：MineDojo 格式 → MineRL 格式（如 planks → oak_planks）
                    initial_inventory = convert_initial_inventory(
                        env_config['initial_inventory'], 
                        target_env='minerl'
                    )
                    env_kwargs['initial_inventory'] = initial_inventory
                    print(f"🎒 初始库存: {initial_inventory}")
                
                # 添加奖励配置（用于自动检测任务完成）
                if 'reward_config' in env_config:
                    # 🔄 转换物品名称：MineDojo 格式 → MineRL 格式
                    reward_config = convert_reward_config(
                        env_config['reward_config'],
                        target_env='minerl'
                    )
                    env_kwargs['reward_config'] = reward_config
                    # 从配置读取 reward_rule，默认为 'any'
                    reward_rule = env_config.get('reward_rule', 'any')
                    env_kwargs['reward_rule'] = reward_rule
                    print(f"🎯 奖励配置: {reward_config}")
                    print(f"📋 完成规则: {reward_rule}（{'任意一个目标完成' if reward_rule == 'any' else '所有目标都要完成' if reward_rule == 'all' else '无自动完成'}）")
                else:
                    # 没有奖励配置，手动完成任务
                    env_kwargs['reward_config'] = None
                    print(f"⚙️  手动录制模式（无自动奖励检测）")
                
                env = gym.make(env_name, **env_kwargs)
            else:
                # 标准MineRL环境
                env = gym.make(env_name)
            
            print(f"✅ 已创建 MineRL 环境: {env_name}")
            return env
            
        except Exception as e:
            raise ValueError(
                f"\n❌ 创建 MineRL 环境失败！\n"
                f"环境名称: {env_name}\n"
                f"错误信息: {e}\n\n"
                f"请检查:\n"
                f"  1. 环境名称是否正确\n"
                f"  2. 自定义环境是否已注册\n"
                f"  3. 环境参数是否正确"
            )
    
    def _try_create_minedojo_env(self, task_id: str, env_name: str):
        """尝试创建 MineDojo 环境（带多种fallback策略）"""
        # 使用配置的图像尺寸
        image_height, image_width = self.default_image_size
        
        # 策略1: 尝试使用 env_name
        if env_name != task_id and not env_name.startswith('MineRL'):
            try:
                env = minedojo.make(
                    task_id=env_name,
                    image_size=(image_height, image_width),
                    seed=None,
                    fast_reset=False
                )
                print(f"✅ 已创建 MineDojo 环境: {env_name}")
                return env
            except ValueError:
                print(f"⚠️  环境 {env_name} 不存在，尝试其他策略...")
        
        # 策略2: 尝试使用 task_id
        try:
            env = minedojo.make(
                task_id=task_id,
                image_size=(image_height, image_width),
                seed=None,
                fast_reset=False
            )
            print(f"✅ 已创建 MineDojo 环境: {task_id}")
            return env
        except ValueError:
            print(f"⚠️  任务 {task_id} 不是MineDojo内置任务")
        
        # 策略3: 转换为 MineDojo 任务名（harvest_1_xxx -> harvest_xxx）
        if '_1_' in task_id:
            simplified_id = task_id.replace('_1_', '_')
            print(f"⚠️  尝试简化的任务ID: {simplified_id}")
            try:
                env = minedojo.make(
                    task_id=simplified_id,
                    image_size=(image_height, image_width),
                    seed=None,
                    fast_reset=False
                )
                print(f"✅ 已创建 MineDojo 环境: {simplified_id}")
                print(f"⚠️  注意: 使用 {simplified_id} 环境代替 {task_id}")
                return env
            except ValueError:
                print(f"⚠️  简化的任务ID {simplified_id} 也不存在")
        
        # 策略4: 使用通用类别环境
        if task_id.startswith('harvest_'):
            print(f"⚠️  尝试使用通用 'harvest' 环境")
            try:
                env = minedojo.make(
                    task_id="harvest",
                    image_size=(image_height, image_width),
                    seed=None,
                    fast_reset=False
                )
                print(f"✅ 已创建通用 MineDojo harvest 环境")
                print(f"⚠️  注意: 通用环境可能与 {task_id} 的目标不完全匹配")
                print(f"⚠️  建议: 手动完成任务目标（如挖掘gravel）")
                return env
            except Exception:
                pass
        elif task_id.startswith('combat_'):
            print(f"⚠️  尝试使用通用 'combat' 环境")
            try:
                env = minedojo.make(
                    task_id="combat",
                    image_size=(image_height, image_width),
                    seed=None,
                    fast_reset=False
                )
                print(f"✅ 已创建通用 MineDojo combat 环境")
                print(f"⚠️  注意: 通用环境可能与 {task_id} 的目标不完全匹配")
                return env
            except Exception:
                pass
        
        # 所有策略都失败
        raise ValueError(
            f"\n❌ 无法创建环境！\n\n"
            f"尝试过的策略:\n"
            f"  1. MineRL 环境: {env_name} (失败)\n"
            f"  2. MineDojo 任务: {task_id} (失败)\n"
            f"  3. 简化任务ID: {task_id.replace('_1_', '_') if '_1_' in task_id else 'N/A'} (失败)\n"
            f"  4. 通用环境: harvest/combat (失败)\n\n"
            f"建议:\n"
            f"  - 检查 config/eval_tasks_prior.yaml 中的 env_name 配置\n"
            f"  - 使用 MineDojo 内置任务（如 harvest_milk, harvest_wool, combat_cow 等）\n"
            f"  - 或者创建自定义环境wrapper\n\n"
            f"MineDojo 内置任务列表: https://docs.minedojo.org/sections/core_api/task_specs.html"
        )
    
    
    def _extract_observation(self, obs_data):
        """
        从环境返回的数据中提取观察图像
        支持不同环境的返回格式
        
        Args:
            obs_data: 环境返回的观察数据
            
        Returns:
            np.ndarray: 图像数据，格式为 (C, H, W) 或 (H, W, C)
        """
        # 如果是字典，尝试提取图像
        if isinstance(obs_data, dict):
            # MineDojo 使用 'rgb'
            if 'rgb' in obs_data:
                return obs_data['rgb']
            # MineRL 使用 'pov'
            elif 'pov' in obs_data:
                return obs_data['pov']
            else:
                print(f"⚠️  观察字典中没有找到 'rgb' 或 'pov' 键，可用键: {list(obs_data.keys())}")
                return None
        # 如果直接是数组，直接返回
        elif isinstance(obs_data, np.ndarray):
            return obs_data
        else:
            print(f"⚠️  未知的观察类型: {type(obs_data)}")
            return None
    
    def _get_mineclip(self):
        """延迟加载 MineCLIP"""
        if self._mineclip is None:
            print("加载 MineCLIP 用于生成视觉嵌入...")
            from src.utils.steve1_mineclip_agent_env_utils import load_mineclip_wconfig
            self._mineclip = load_mineclip_wconfig()
            print("✓ MineCLIP 已加载")
        return self._mineclip
    
    def _create_task_directory(self, task_id: str) -> Path:
        """
        创建任务目录（使用任务ID命名，不带时间戳和语言后缀）
        
        Args:
            task_id: 任务ID
        
        Returns:
            Path: 任务目录路径
        """
        # 直接使用 task_id 作为目录名，不再添加语言后缀和时间戳
        task_dir = self.base_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        return task_dir
    
    def get_next_trial_number(self) -> int:
        """获取当前任务的下一个trial编号"""
        if not self.current_task_dir or not self.current_task_dir.exists():
            return 1
        
        # 查找已有的 trial* 目录
        existing_trials = [
            d.name for d in self.current_task_dir.iterdir() 
            if d.is_dir() and d.name.startswith('trial')
        ]
        
        if not existing_trials:
            return 1
        
        # 提取trial编号
        trial_nums = []
        for trial_name in existing_trials:
            # trial{num}
            try:
                num = int(trial_name.replace('trial', ''))
                trial_nums.append(num)
            except ValueError:
                continue
        
        return max(trial_nums) + 1 if trial_nums else 1
    
    def record_task(
        self,
        task_id: str,
        max_frames: int = 1000,
        fps: int = 20,
        camera_delta: int = 4,
        mouse_sensitivity: float = 0.5,
        fullscreen: bool = False
    ):
        """
        录制指定任务的训练样本
        
        Args:
            task_id: 任务ID
            max_frames: 最大帧数
            fps: 录制帧率
            camera_delta: 相机灵敏度（键盘）
            mouse_sensitivity: 鼠标灵敏度（默认0.5，可用--mouse-sensitivity调整）
            fullscreen: 是否全屏
        """
        # 获取任务配置
        task_config = self.get_task_config(task_id)
        instruction = task_config['instruction']
        
        # 创建任务目录（使用 task_id 命名）
        self.current_task_dir = self._create_task_directory(task_id)
        self.current_task_id = task_id
        self.current_instruction = instruction
        self.recorded_trials = []  # 重置trial列表
        
        print("\n" + "=" * 80)
        print(f"🎬 专家演示录制")
        print("=" * 80)
        print(f"\n任务ID: {task_id}")
        print(f"语言: {language}")
        print(f"指令: {instruction}")
        print(f"最大帧数: {max_frames}")
        print(f"录制帧率: {fps} FPS")
        print(f"输出目录: {self.current_task_dir}")
        print("=" * 80 + "\n")
        
        # 创建 MineRL 环境
        env = self.create_environment(task_id, task_config)
        
        # 初始化pygame控制器（MineRL 专用）
        controller = PygameController(
            camera_delta=camera_delta,
            mouse_sensitivity=mouse_sensitivity,
            fullscreen=fullscreen
        )
        
        try:
            while True:
                # 获取下一个trial编号
                trial_idx = self.get_next_trial_number()
                
                print(f"\n{'='*80}")
                print(f"🎬 开始录制 Trial {trial_idx}")
                print(f"{'='*80}")
                print(f"📝 目标: {instruction}")
                print(f"⏰ 最大帧数: {max_frames}")
                print(f"\n按Enter键开始录制...")
                input()
                
                # 重置环境
                obs_dict = env.reset()
                obs = self._extract_observation(obs_dict)
                
                if obs is None:
                    print("❌ 无法获取观察图像")
                    break
                
                print(f"\n✅ 环境已重置")
                print(f"📐 观察形状: {obs.shape} (格式: {'(C,H,W)' if obs.shape[0] == 3 else '(H,W,C)'})")
                print(f"开始录制...\n")
                
                # 存储数据
                frames = []
                actions = []  # 保存动作序列
                
                # 重置控制器
                controller.reset_retry_flag()
                controller.reset_mouse_state()
                
                # 初始化库存追踪
                prev_inventory = {}
                if isinstance(obs_dict, dict) and 'inventory' in obs_dict:
                    for item, count in obs_dict['inventory'].items():
                        if hasattr(count, 'item'):
                            count = count.item()
                        prev_inventory[item] = int(count)
                
                # 录制循环
                frame_count = 0
                start_time = time.time()
                done = False
                total_reward = 0.0
                
                while frame_count < max_frames and not done:
                    # 处理pygame事件
                    controller.process_events()
                    
                    # 检查退出
                    if controller.should_quit:
                        print(f"\n⚠️  用户按下ESC，退出录制")
                        env.close()
                        controller.quit()
                        return
                    
                    # 检查重试
                    if controller.should_retry:
                        print(f"\n🔄 用户按下Q，重新录制trial {trial_idx}")
                        break
                    
                    # 获取动作
                    action = controller.get_action()
                    action_desc = controller.decode_action(action)
                    
                    # 保存动作（保存原始动作，用于生成 actions.json）
                    actions.append(action.copy() if isinstance(action, dict) else action)
                    
                    # 执行动作
                    step_result = env.step(action)
                    if len(step_result) == 4:
                        obs_dict, reward, done, info = step_result
                    else:
                        obs_dict, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    
                    obs = self._extract_observation(obs_dict)
                    
                    # 累计奖励
                    total_reward += reward
                    
                    # 调试：显示库存和奖励信息
                    if isinstance(obs_dict, dict) and 'inventory' in obs_dict:
                        inventory = obs_dict['inventory']
                        
                        # 检查库存变化
                        inventory_changed = False
                        changes = {}
                        for item, count in inventory.items():
                            # 处理 numpy 类型
                            if hasattr(count, 'item'):
                                count = count.item()
                            count = int(count)
                            
                            prev_count = prev_inventory.get(item, 0)
                            if count != prev_count:
                                inventory_changed = True
                                changes[item] = f"{prev_count}→{count}"
                                prev_inventory[item] = count
                        
                        # 获取当前库存摘要（显示所有非零物品）
                        current_items = {}
                        for item, count in inventory.items():
                            if hasattr(count, 'item'):
                                count = count.item()
                            count = int(count)
                            if count > 0:
                                current_items[item] = count
                        
                        # 如果库存变化，立即打印
                        if inventory_changed:
                            print(f"\n[INVENTORY] 📦 库存变化: {changes}")
                            print(f"[INVENTORY] 📦 当前库存: {current_items}")
                            if reward > 0:
                                print(f"[REWARD] 🎉 获得奖励: {reward:.1f} (累计: {total_reward:.1f})")
                            print(f"[STATUS] Done={done}, Frame={frame_count}\n")
                    
                    # 显示done状态变化
                    if done:
                        print(f"\n[DONE] ✅ 任务完成！")
                        print(f"[DONE] Frame: {frame_count}")
                        print(f"[DONE] 总奖励: {total_reward:.1f}")
                        if isinstance(obs_dict, dict) and 'inventory' in obs_dict:
                            print(f"[DONE] 最终库存: {current_items}\n")
                    
                    # 保存帧
                    frames.append(obs.copy())
                    frame_count += 1
                    
                    # 显示画面（传递完整的 obs_dict 以显示库存信息）
                    controller.display_frame(obs_dict, task_id, trial_idx, frame_count, max_frames, 
                                           action_desc, done)
                    
                    # 维持帧率
                    controller.clock.tick(fps)
                    
                    # 实时统计
                    if frame_count % 20 == 0 or done:
                        elapsed = time.time() - start_time
                        actual_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"[{elapsed:6.1f}s] 帧{frame_count:4d}: {action_desc:<30} | FPS: {actual_fps:4.1f} | Done: {done}")
                
                # 检查是否需要重试
                if controller.should_retry:
                    controller.reset_retry_flag()
                    continue
                
                # 录制完成
                elapsed_time = time.time() - start_time
                
                if done:
                    print(f"\n✅ 任务完成！ (用时 {elapsed_time:.1f}秒，共{frame_count}帧)")
                    success = True
                else:
                    print(f"\n⏸️  达到最大帧数 {max_frames} (用时 {elapsed_time:.1f}秒)")
                    # 询问是否保存
                    print(f"\n任务未完成，是否仍要保存？(y/n)")
                    save_choice = input().strip().lower()
                    if save_choice != 'y':
                        print("❌ 未保存，准备重录...")
                        continue
                    success = False
                
                # 保存数据
                trial_data = self._save_trial(trial_idx, frames, actions, success, frame_count, elapsed_time)
                self.recorded_trials.append(trial_data)
                
                # 生成/更新 result.json
                self._generate_result_json()
                
                # 询问是否继续录制
                print(f"\n{'='*80}")
                print(f"✅ Trial {trial_idx} 录制完成！")
                print(f"{'='*80}")
                print(f"\n是否继续录制下一个trial? (y/n)")
                continue_choice = input().strip().lower()
                
                if continue_choice != 'y':
                    print("\n录制结束")
                    break
        
        finally:
            # 生成最终的 result.json（确保即使中断也能生成）
            if self.recorded_trials:
                self._generate_result_json()
                print(f"\n📊 最终结果已保存: {self.current_task_dir / 'result.json'}")
            
            try:
                env.close()
            except Exception as e:
                print(f"⚠️  环境关闭时出错: {e}")
            
            try:
                controller.quit()
            except Exception as e:
                print(f"⚠️  控制器退出时出错: {e}")
    
    def _save_trial(self, trial_idx: int, frames: List[np.ndarray], 
                    actions: List[dict], success: bool, steps: int, time_seconds: float) -> dict:
        """
        保存trial数据并生成视觉嵌入
        
        目录结构: {task_id}/trial{num}/
            - frames/step_0000.png, step_0001.png, ...
            - actions.json
            - visual_embeds.pkl
            - trial_info.json
        
        Args:
            trial_idx: trial编号
            frames: 帧列表
            actions: 动作列表
            success: 是否成功
            steps: 步数
            time_seconds: 用时（秒）
        
        Returns:
            dict: trial数据（用于生成result.json）
        """
        # 创建 trial 目录: trial{num}
        trial_dir = self.current_task_dir / f"trial{trial_idx}"
        frames_dir = trial_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 保存数据到 {self.current_task_id}/trial{trial_idx}...")
        
        # 保存帧图像
        for i, frame in enumerate(frames):
            # 转换为 (H, W, C)
            if frame.shape[0] == 3:  # (C, H, W)
                frame_img = frame.transpose(1, 2, 0)
            else:
                frame_img = frame
            
            img = Image.fromarray(frame_img.astype(np.uint8))
            img.save(frames_dir / f"step_{i:04d}.png")
        
        # 保存 actions.json（MineRL格式）
        actions_json = self._convert_actions_to_json(actions)
        actions_path = trial_dir / "actions.json"
        with open(actions_path, 'w', encoding='utf-8') as f:
            json.dump(actions_json, f, indent=2, ensure_ascii=False)
        
        # 保存 trial_info.json
        trial_info = {
            "task_id": self.current_task_id,
            "instruction": self.current_instruction,
            "success": success,
            "steps": steps,
            "time_seconds": time_seconds
        }
        with open(trial_dir / "trial_info.json", 'w', encoding='utf-8') as f:
            json.dump(trial_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 保存完成:")
        print(f"   - {len(frames)} 帧图像 (.png)")
        print(f"   - actions.json ({len(actions)} 个动作)")
        print(f"   - trial_info.json")
        
        # 生成 visual_embeds.pkl
        if len(frames) >= 16:
            print("🔄 生成视觉嵌入...")
            embed = self._generate_visual_embed(frames)
            if embed is not None:
                embed_path = trial_dir / "visual_embeds.pkl"
                with open(embed_path, 'wb') as f:
                    pickle.dump(embed, f)
                print(f"   - visual_embeds.pkl (shape: {embed.shape})")
            else:
                print(f"   ⚠️ 视觉嵌入生成失败")
        else:
            print(f"   ⚠️ 帧数不足16帧，跳过视觉嵌入生成")
        
        # 返回trial数据
        return {
            "trial_idx": trial_idx,
            "success": success,
            "steps": steps,
            "time_seconds": time_seconds,
            "has_visual_embed": len(frames) >= 16
        }
    
    def _generate_visual_embed(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        从帧序列生成视觉嵌入（使用最后16帧）
        
        Args:
            frames: 帧列表 (H, W, C) 或 (C, H, W)
            
        Returns:
            视觉嵌入 [512] 或 None
        """
        try:
            from src.utils.device import DEVICE
            mineclip = self._get_mineclip()
            
            # 取最后16帧
            last_frames = frames[-16:]
            
            # 预处理帧
            processed_frames = []
            for frame in last_frames:
                # 转换为 (H, W, C)
                if frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
                
                # 调整大小到 MineCLIP 期望的 (160, 256)
                img = Image.fromarray(frame.astype(np.uint8))
                img = img.resize((256, 160), Image.Resampling.LANCZOS)
                
                # 转换为 float32，保持 [0, 255] 范围
                img_array = np.array(img).astype(np.float32)
                
                # 转换为 CHW 格式 [3, 160, 256]
                img_array = np.transpose(img_array, (2, 0, 1))
                processed_frames.append(img_array)
            
            # 堆叠为视频张量 [16, 3, 160, 256]
            video_array = np.stack(processed_frames, axis=0)
            
            # 转换为 torch tensor 并添加 batch 维度 [1, 16, 3, 160, 256]
            video_tensor = th.from_numpy(video_array).unsqueeze(0).float().to(DEVICE)
            
            # 使用 MineCLIP 编码视频
            with th.no_grad():
                video_embed = mineclip.encode_video(video_tensor)
            
            # 转换为 numpy
            video_embed = video_embed.cpu().numpy().squeeze()
            
            return video_embed
            
        except Exception as e:
            print(f"生成视觉嵌入失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_result_json(self):
        """
        生成/更新任务级别的 result.json
        
        格式与 eval_framework.py 保持一致
        """
        if not self.recorded_trials:
            return
        
        # 计算统计数据
        success_count = sum(1 for t in self.recorded_trials if t['success'])
        success_rate = success_count / len(self.recorded_trials) if self.recorded_trials else 0.0
        avg_steps = sum(t['steps'] for t in self.recorded_trials) / len(self.recorded_trials) if self.recorded_trials else 0.0
        avg_time = sum(t['time_seconds'] for t in self.recorded_trials) / len(self.recorded_trials) if self.recorded_trials else 0.0
        
        # 构建result数据
        result_data = {
            "task_id": self.current_task_id,
            "instruction": self.current_instruction,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_time": avg_time,
            "trials": self.recorded_trials
        }
        
        # 保存到任务目录
        result_path = self.current_task_dir / "result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"   - 已更新 result.json (成功率: {success_rate*100:.1f}%)")
    
    def _convert_actions_to_json(self, actions: List[dict]) -> List[dict]:
        """
        将MineRL动作转换为标准JSON格式
        
        Args:
            actions: MineRL动作列表
        
        Returns:
            符合评估格式的动作JSON列表
        """
        # MineRL动作空间的所有键
        action_keys = [
            'attack', 'back', 'forward', 'jump', 'left', 'right',
            'sneak', 'sprint', 'use', 'drop', 'inventory',
            'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5',
            'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9',
            'camera'
        ]
        
        actions_json = []
        for step, action in enumerate(actions):
            # 创建标准格式的动作
            action_dict = {}
            
            for key in action_keys:
                if key == 'camera':
                    # camera 是 [pitch, yaw] 的嵌套列表
                    camera_val = action.get('camera', [0.0, 0.0])
                    if isinstance(camera_val, (list, tuple)):
                        action_dict['camera'] = [[float(camera_val[0]), float(camera_val[1])]]
                    else:
                        action_dict['camera'] = [[0.0, 0.0]]
                else:
                    # 其他键是包含单个整数的列表
                    val = action.get(key, 0)
                    action_dict[key] = [int(val)]
            
            actions_json.append({
                "step": step,
                "action": action_dict
            })
        
        return actions_json


def main():
    parser = argparse.ArgumentParser(
        description="样本录制工具（基于配置文件）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 列出所有可用任务
  python src/evaluation/record_samples.py --list-tasks
  
  # 录制指定任务
  python src/evaluation/record_samples.py --task harvest_1_log
  
  # 录制时使用全屏模式
  python src/evaluation/record_samples.py --task combat_chicken --fullscreen
  
  # 指定不同的配置文件
  python src/evaluation/record_samples.py --config config/eval_tasks.yaml --task harvest_1_dirt
        """
    )
    
    parser.add_argument('--config', type=str, default='config/eval_tasks.yaml',
                        help='配置文件路径（默认: config/eval_tasks.yaml）')
    parser.add_argument('--task', type=str,
                        help='任务ID（如 harvest_1_log）')
    parser.add_argument('--list-tasks', action='store_true',
                        help='列出所有可用任务')
    parser.add_argument('--base-dir', type=str, default='data/train_samples',
                        help='输出基础目录（默认: data/train_samples）')
    parser.add_argument('--max-frames', type=int, default=1000,
                        help='最大帧数（默认: 1000）')
    parser.add_argument('--fps', type=int, default=20,
                        help='录制帧率（默认: 20）')
    parser.add_argument('--camera-delta', type=int, default=4,
                        help='相机灵敏度（键盘，默认: 4）')
    parser.add_argument('--mouse-sensitivity', type=float, default=0.2,
                        help='鼠标灵敏度（默认: 0.2）')
    parser.add_argument('--fullscreen', action='store_true',
                        help='全屏显示（推荐）')
    
    args = parser.parse_args()
    
    # 创建录制器
    recorder = SampleRecorder(args.config, args.base_dir)
    
    # 列出任务
    if args.list_tasks:
        tasks = recorder.get_available_tasks()
        print("\n" + "=" * 80)
        print("📋 可用任务列表")
        print("=" * 80)
        for i, task_id in enumerate(tasks, 1):
            task_config = recorder.get_task_config(task_id)
            instruction = task_config.get('instruction', 'N/A')
            print(f"{i:3d}. {task_id:<30} - {instruction}")
        print("=" * 80)
        print(f"\n总计: {len(tasks)} 个任务\n")
        return
    
    # 录制任务
    if not args.task:
        print("❌ 错误: 请指定任务ID (--task) 或使用 --list-tasks 查看可用任务")
        return
    
    # 使用配置文件的默认值（如果命令行未明确指定）
    # 注意：argparse 的 default 会覆盖，所以我们需要检查用户是否真的指定了
    # 简化方案：直接使用配置文件的值，除非明确在命令行修改
    max_frames = args.max_frames if args.max_frames != 1000 else recorder.default_max_steps
    fps = args.fps if args.fps != 20 else recorder.default_fps
    
    print("\n" + "=" * 80)
    print("🎬 训练样本录制工具")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"任务ID: {args.task}")
    print(f"输出目录: {args.base_dir}/{args.task}/")
    print(f"图像尺寸: {recorder.default_image_size[0]}x{recorder.default_image_size[1]} (H x W)")
    print(f"最大帧数: {max_frames}")
    print(f"录制帧率: {fps} FPS")
    print(f"显示模式: {'全屏' if args.fullscreen else '窗口'}")
    print(f"✨ 录制完成后自动生成 visual_embeds.pkl")
    print("=" * 80)
    
    recorder.record_task(
        task_id=args.task,
        max_frames=max_frames,
        fps=fps,
        camera_delta=args.camera_delta,
        mouse_sensitivity=args.mouse_sensitivity,
        fullscreen=args.fullscreen
    )


if __name__ == "__main__":
    main()

