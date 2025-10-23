#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineDojo 任务特定的环境包装器

这个文件包含针对特定MineDojo任务的自定义Wrapper，
用于修改或增强任务的判断条件、奖励函数等。

与 env_wrappers.py 的区别:
- env_wrappers.py: 通用的环境包装器（观察空间、动作空间、时间限制等）
- task_wrappers.py: 任务特定的包装器（harvest、combat、craft等）

使用方法:
    from src.utils.task_wrappers import HarvestLogWrapper
    
    env = minedojo.make(task_id="harvest_1_log")
    env = HarvestLogWrapper(env, required_logs=1)
"""

import gym
import numpy as np
import re


class HarvestLogWrapper(gym.Wrapper):
    """
    Harvest Log 任务专用包装器
    
    功能:
    1. 支持所有6种原木类型（oak, birch, spruce, dark_oak, jungle, acacia）
    2. 自动识别目标数量（如harvest_8_log -> 8个原木）
    3. 实时反馈获得的原木类型和数量
    
    适用任务:
    - harvest_1_log (获得1个原木)
    - harvest_8_log (获得8个原木)
    - harvest_64_log (获得64个原木)
    
    问题背景:
    MineDojo的harvest_log任务可能只识别特定类型的原木（如Oak Log），
    不识别Dark Oak Log等其他类型。用户报告"获得黑色的木头没有奖励"。
    
    解决方案:
    这个Wrapper检测所有6种原木类型，任意一种都计入任务完成条件。
    
    Example:
        >>> env = minedojo.make(task_id="harvest_1_log")
        >>> env = HarvestLogWrapper(env, required_logs=1)
        >>> # 现在获得任何类型的原木都会触发任务完成
    """
    
    def __init__(self, env, required_logs=1, verbose=True):
        """
        Args:
            env: MineDojo环境实例
            required_logs: 需要的原木数量（默认1）
            verbose: 是否打印详细信息（默认True）
        """
        super().__init__(env)
        self.required_logs = required_logs
        self.verbose = verbose
        
        # Minecraft所有原木类型
        self.log_types = [
            "oak_log",       # 橡木（最常见）
            "birch_log",     # 白桦木
            "spruce_log",    # 云杉木
            "dark_oak_log",  # 深色橡木（用户报告的"黑色木头"）
            "jungle_log",    # 丛林木（稀有）
            "acacia_log"     # 金合欢木（稀有）
        ]
        
        # 跟踪状态
        self.last_log_count = 0
        self.task_completed = False
        
        if self.verbose:
            print(f"  ✓ HarvestLogWrapper已启用")
            print(f"    - 支持所有6种原木类型")
            print(f"    - 目标数量: {required_logs}个")
    
    def reset(self, **kwargs):
        """重置环境并重置状态"""
        self.last_log_count = 0
        self.task_completed = False
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """执行一步并检查是否获得了任意类型的原木"""
        obs, reward, done, info = self.env.step(action)
        
        # 如果任务已完成，直接返回
        if self.task_completed:
            return obs, reward, done, info
        
        # 检查库存中的所有原木类型
        if 'inventory' in info:
            inventory = info['inventory']
            total_logs = 0
            obtained_log_types = []
            
            # 调试：打印所有包含"log"的物品
            if self.verbose and self.last_log_count == 0:
                log_items = {k: v for k, v in inventory.items() if 'log' in k.lower()}
                if log_items:
                    print(f"  [DEBUG] 库存中的原木物品: {log_items}")
            
            # 遍历所有原木类型
            for log_type in self.log_types:
                count = self._get_item_count(inventory, log_type)
                if count > 0:
                    total_logs += count
                    obtained_log_types.append(f"{log_type}({count})")
            
            # 如果获得了足够的原木，且任务还未完成
            if total_logs >= self.required_logs and not done:
                done = True
                reward = 1.0  # 给予成功奖励
                info['success'] = True
                self.task_completed = True
                
                # 只在新增原木时打印（避免重复）
                if total_logs > self.last_log_count and self.verbose:
                    log_info = ", ".join(obtained_log_types)
                    print(f"\n✓ 获得原木！总数: {total_logs} | 类型: {log_info}")
                    print(f"  任务成功！(需要{self.required_logs}个)\n")
            
            self.last_log_count = total_logs
        
        return obs, reward, done, info
    
    def _get_item_count(self, inventory, item_name):
        """
        从库存中获取物品数量
        
        支持多种物品ID格式:
        - "oak_log"
        - "minecraft:oak_log"
        
        Args:
            inventory: 库存字典
            item_name: 物品名称（不含minecraft:前缀）
        
        Returns:
            int: 物品数量
        """
        # 尝试多种可能的物品ID格式
        for item_id in [item_name, f"minecraft:{item_name}"]:
            if item_id in inventory:
                return inventory[item_id]
        return 0


class HarvestWheatWrapper(gym.Wrapper):
    """
    Harvest Wheat 任务专用包装器
    
    类似HarvestLogWrapper，但用于小麦收获任务。
    可以扩展支持不同成熟度的小麦等。
    
    适用任务:
    - harvest_1_wheat
    - harvest_8_wheat
    
    TODO: 根据实际需求实现
    """
    
    def __init__(self, env, required_wheat=1, verbose=True):
        super().__init__(env)
        self.required_wheat = required_wheat
        self.verbose = verbose
        
        if self.verbose:
            print(f"  ✓ HarvestWheatWrapper已启用（目标: {required_wheat}个）")
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # TODO: 实现小麦特定的判断逻辑
        return self.env.step(action)


class CombatWrapper(gym.Wrapper):
    """
    Combat 任务专用包装器
    
    用于狩猎/战斗任务，可以自定义：
    - 击杀奖励
    - 受伤惩罚
    - 死亡处理
    
    适用任务:
    - hunt_cow
    - hunt_pig
    - combat_spider
    
    TODO: 根据实际需求实现
    """
    
    def __init__(self, env, target_mob="cow", required_kills=1, verbose=True):
        super().__init__(env)
        self.target_mob = target_mob
        self.required_kills = required_kills
        self.verbose = verbose
        
        if self.verbose:
            print(f"  ✓ CombatWrapper已启用（目标: 击杀{required_kills}个{target_mob}）")
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # TODO: 实现战斗任务特定的判断逻辑
        return self.env.step(action)


class CraftWrapper(gym.Wrapper):
    """
    Craft 任务专用包装器
    
    用于合成任务，可以自定义：
    - 合成表验证
    - 材料检测
    - 合成奖励
    
    适用任务:
    - craft_planks
    - craft_stick
    - craft_crafting_table
    
    TODO: 根据实际需求实现
    """
    
    def __init__(self, env, target_item="planks", required_count=1, verbose=True):
        super().__init__(env)
        self.target_item = target_item
        self.required_count = required_count
        self.verbose = verbose
        
        if self.verbose:
            print(f"  ✓ CraftWrapper已启用（目标: 合成{required_count}个{target_item}）")
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # TODO: 实现合成任务特定的判断逻辑
        return self.env.step(action)


# ============================================================================
# 工具函数：自动选择合适的任务Wrapper
# ============================================================================

def get_task_wrapper(task_id, verbose=True):
    """
    根据任务ID自动选择合适的任务Wrapper
    
    Args:
        task_id: MineDojo任务ID（如"harvest_1_log"）
        verbose: 是否打印详细信息
    
    Returns:
        wrapper_class: Wrapper类
        wrapper_kwargs: Wrapper参数字典
        
    Returns None, {} if no specific wrapper is needed.
    
    Example:
        >>> wrapper_class, kwargs = get_task_wrapper("harvest_8_log")
        >>> env = minedojo.make(task_id="harvest_8_log")
        >>> env = wrapper_class(env, **kwargs)
    """
    # harvest_X_log 任务
    if "harvest" in task_id and "log" in task_id:
        # 提取目标数量（如harvest_8_log -> 8）
        match = re.search(r'harvest_(\d+)_log', task_id)
        required_logs = int(match.group(1)) if match else 1
        
        return HarvestLogWrapper, {
            'required_logs': required_logs,
            'verbose': verbose
        }
    
    # harvest_X_wheat 任务
    if "harvest" in task_id and "wheat" in task_id:
        match = re.search(r'harvest_(\d+)_wheat', task_id)
        required_wheat = int(match.group(1)) if match else 1
        
        return HarvestWheatWrapper, {
            'required_wheat': required_wheat,
            'verbose': verbose
        }
    
    # hunt/combat 任务
    if "hunt" in task_id or "combat" in task_id:
        # TODO: 解析目标生物和击杀数量
        return CombatWrapper, {
            'target_mob': 'unknown',
            'required_kills': 1,
            'verbose': verbose
        }
    
    # craft 任务
    if "craft" in task_id:
        # TODO: 解析目标物品和数量
        return CraftWrapper, {
            'target_item': 'unknown',
            'required_count': 1,
            'verbose': verbose
        }
    
    # 其他任务：不需要特殊Wrapper
    return None, {}


def apply_task_wrapper(env, task_id, verbose=True):
    """
    便捷函数：自动应用合适的任务Wrapper
    
    Args:
        env: MineDojo环境实例
        task_id: 任务ID
        verbose: 是否打印详细信息
    
    Returns:
        gym.Env: 包装后的环境（如果有对应Wrapper）或原环境
    
    Example:
        >>> env = minedojo.make(task_id="harvest_1_log")
        >>> env = apply_task_wrapper(env, task_id="harvest_1_log")
        # 自动应用 HarvestLogWrapper
    """
    wrapper_class, wrapper_kwargs = get_task_wrapper(task_id, verbose=verbose)
    
    if wrapper_class is not None:
        return wrapper_class(env, **wrapper_kwargs)
    else:
        return env

