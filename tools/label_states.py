#!/usr/bin/env python3
"""
交互式状态标注工具 - DAgger第二步

显示策略收集的状态，接受键盘输入作为专家标注，保存标注数据。
支持智能采样（只标注关键状态）以节省时间。

Usage:
    # 标注所有状态
    python tools/label_states.py \
        --states data/policy_states/iter_1/ \
        --output data/expert_labels/iter_1.pkl

    # 智能采样（只标注失败episode + 20%随机）
    python tools/label_states.py \
        --states data/policy_states/iter_1/ \
        --output data/expert_labels/iter_1.pkl \
        --smart-sampling \
        --failure-window 10
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pickle
from pathlib import Path
from collections import deque

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StateLabeler:
    """交互式状态标注工具"""
    
    def __init__(self, failure_window=10, random_sample_rate=0.1):
        """
        Args:
            failure_window: 失败前标注的步数
            random_sample_rate: 成功episode的随机采样率
        """
        self.failure_window = failure_window
        self.random_sample_rate = random_sample_rate
        
        # MineDojo动作映射 (8维MultiDiscrete)
        # [forward/back, left/right, jump, pitch, yaw, functional, sprint, sneak]
        
        # 动作解读映射
        self.action_descriptions = {
            0: {0: "后退", 1: "静止", 2: "前进"},
            1: {0: "右移", 1: "静止", 2: "左移"},
            2: {0: "静止", 1: "跳跃"},
            5: {0: "无", 1: "使用", 2: "丢弃", 3: "攻击", 4: "合成", 5: "装备", 6: "放置", 7: "破坏"}
        }
        
        self.action_mapping = {
            # 移动
            'w': [1, 0, 0, 12, 12, 0, 0, 0],  # 前进
            's': [2, 0, 0, 12, 12, 0, 0, 0],  # 后退
            'a': [0, 1, 0, 12, 12, 0, 0, 0],  # 左移
            'd': [0, 2, 0, 12, 12, 0, 0, 0],  # 右移
            
            # 视角控制
            'i': [0, 0, 0, 8, 12, 0, 0, 0],   # 向上看
            'k': [0, 0, 0, 16, 12, 0, 0, 0],  # 向下看
            'j': [0, 0, 0, 12, 8, 0, 0, 0],   # 向左看
            'l': [0, 0, 0, 12, 16, 0, 0, 0],  # 向右看
            
            # 动作
            'f': [0, 0, 0, 12, 12, 3, 0, 0],  # 攻击（砍树）
            ' ': [0, 0, 1, 12, 12, 0, 0, 0],  # 跳跃（空格）
            
            # 组合动作
            'q': [1, 0, 0, 12, 12, 3, 0, 0],  # 前进+攻击
            'e': [0, 0, 0, 8, 12, 3, 0, 0],   # 向上看+攻击
            
            # 特殊
            'n': None,  # 跳过此状态
            'z': 'undo',  # 撤销上一个标注
            'x': 'quit',  # 退出标注
        }
        
        self.labeled_data = []
        self.undo_stack = deque(maxlen=10)
    
    def load_episodes(self, states_dir):
        """加载所有episode文件"""
        episode_files = sorted(Path(states_dir).glob("episode_*.npy"))
        
        if not episode_files:
            print(f"✗ 错误: 在 {states_dir} 中未找到episode文件")
            return []
        
        episodes = []
        for file in episode_files:
            try:
                episode_data = np.load(file, allow_pickle=True).item()
                episodes.append({
                    'file': str(file),
                    'data': episode_data,
                    'success': episode_data.get('success', False),
                    'num_steps': episode_data.get('num_steps', 0)
                })
            except Exception as e:
                print(f"⚠️  加载失败: {file} - {e}")
        
        return episodes
    
    def decode_action(self, action):
        """
        解读动作数组为人类可读的文本（英文，避免OpenCV中文渲染问题）
        
        Args:
            action: numpy array [8] - MineDojo MultiDiscrete动作
            
        Returns:
            str: 人类可读的动作描述（英文）
        """
        parts = []
        
        # 维度0: 前后移动 (0=停止, 1=前进, 2=后退)
        if action[0] == 1:
            parts.append("Forward")
        elif action[0] == 2:
            parts.append("Back")
        
        # 维度1: 左右移动 (0=停止, 1=左移, 2=右移)
        if action[1] == 1:
            parts.append("Left")
        elif action[1] == 2:
            parts.append("Right")
        
        # 维度2: 跳跃
        if action[2] == 1:
            parts.append("Jump")
        
        # 维度3-4: 相机 (pitch, yaw)
        pitch = action[3]
        yaw = action[4]
        if pitch < 11:
            parts.append("Look Up")
        elif pitch > 13:
            parts.append("Look Down")
        
        if yaw < 11:
            parts.append("Turn Left")
        elif yaw > 13:
            parts.append("Turn Right")
        
        # 维度5: 功能动作
        functional = action[5]
        if functional == 3:
            parts.append("Attack")
        elif functional == 1:
            parts.append("Use")
        elif functional == 6:
            parts.append("Place")
        elif functional == 7:
            parts.append("Destroy")
        
        if not parts:
            return "Idle"
        
        return " + ".join(parts)
    
    def smart_sample_states(self, episodes):
        """
        智能采样状态用于标注
        
        策略:
        1. 失败episode: 标注最后N步（失败前的关键决策）
        2. 成功episode: 随机采样10%（用于数据增强）
        
        Returns:
            List of (episode_idx, state_idx, state, action)
        """
        states_to_label = []
        
        for ep_idx, episode in enumerate(episodes):
            episode_data = episode['data']
            states = episode_data['states']
            actions = episode_data.get('actions', [])
            success = episode['success']
            
            if not success:
                # 失败episode: 标注最后N步
                start_idx = max(0, len(states) - self.failure_window)
                for state_idx in range(start_idx, len(states)):
                    if state_idx < len(actions):
                        states_to_label.append({
                            'episode_idx': ep_idx,
                            'state_idx': state_idx,
                            'state': states[state_idx],
                            'policy_action': actions[state_idx],
                            'priority': 'high'  # 高优先级
                        })
            else:
                # 成功episode: 随机采样
                num_samples = max(1, int(len(states) * self.random_sample_rate))
                sample_indices = np.random.choice(
                    len(states), 
                    size=min(num_samples, len(states)), 
                    replace=False
                )
                for state_idx in sample_indices:
                    if state_idx < len(actions):
                        states_to_label.append({
                            'episode_idx': ep_idx,
                            'state_idx': state_idx,
                            'state': states[state_idx],
                            'policy_action': actions[state_idx],
                            'priority': 'low'  # 低优先级
                        })
        
        return states_to_label
    
    def display_state(self, state_info, current_idx, total):
        """显示状态图像"""
        obs = state_info['state']['observation']
        
        # MineDojo观察: (C, H, W) → (H, W, C)
        if obs.shape[0] == 3:  # (3, H, W)
            display_img = obs.transpose(1, 2, 0)
        else:
            display_img = obs
        
        # RGB → BGR for OpenCV
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        
        # 放大显示
        display_img = cv2.resize(display_img, (640, 480))
        
        # 添加信息叠加
        episode_idx = state_info['episode_idx']
        state_idx = state_info['state_idx']
        priority = state_info['priority']
        
        # 背景框
        cv2.rectangle(display_img, (5, 5), (635, 90), (0, 0, 0), -1)
        cv2.rectangle(display_img, (5, 5), (635, 90), (255, 255, 255), 2)
        
        # 解读策略动作
        policy_action = state_info['policy_action']
        action_description = self.decode_action(policy_action)
        
        # 文本信息
        info_lines = [
            f"Progress: {current_idx + 1}/{total}  |  Priority: {priority.upper()}",
            f"Episode: {episode_idx}  |  Step: {state_idx}",
            f"Policy Action: {policy_action}",
            f">>> {action_description}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(
                display_img, line, (10, 25 + i*20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        cv2.imshow('State Labeling - Expert Annotation', display_img)
    
    def label_states(self, states_to_label):
        """交互式标注状态"""
        print(f"\n{'='*60}")
        print(f"开始标注 ({len(states_to_label)} 个状态)")
        print(f"{'='*60}")
        print("控制:")
        print("  W/S/A/D    - 前进/后退/左/右")
        print("  I/K/J/L    - 视角 上/下/左/右")
        print("  F          - 攻击（砍树）")
        print("  Space      - 跳跃")
        print("  Q          - 前进+攻击")
        print("  E          - 向上看+攻击")
        print("  N          - 跳过此状态")
        print("  Z          - 撤销上一个标注")
        print("  X/ESC      - 完成标注")
        print(f"{'='*60}\n")
        
        current_idx = 0
        
        while current_idx < len(states_to_label):
            state_info = states_to_label[current_idx]
            
            # 显示当前状态
            self.display_state(state_info, current_idx, len(states_to_label))
            
            # 等待键盘输入
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                print("\n⚠️  ESC pressed - 退出标注")
                break
            
            key_char = chr(key).lower()
            
            if key_char == 'x':
                print(f"\n✓ 标注完成 ({len(self.labeled_data)} 个标注)")
                break
            
            elif key_char == 'z':
                # 撤销
                if self.undo_stack:
                    removed = self.undo_stack.pop()
                    self.labeled_data.remove(removed)
                    current_idx = max(0, current_idx - 1)
                    print(f"  ↶ 撤销标注 (剩余: {len(self.labeled_data)})")
                else:
                    print(f"  ⚠️  无法撤销")
            
            elif key_char in self.action_mapping:
                action = self.action_mapping[key_char]
                
                if action is not None:
                    # 保存标注
                    labeled_item = {
                        'observation': state_info['state']['observation'],
                        'expert_action': np.array(action),
                        'policy_action': state_info['policy_action'],
                        'episode_idx': state_info['episode_idx'],
                        'state_idx': state_info['state_idx'],
                        'priority': state_info['priority']
                    }
                    
                    self.labeled_data.append(labeled_item)
                    self.undo_stack.append(labeled_item)
                    
                    action_name = self._get_action_name(key_char)
                    print(f"  ✓ [{current_idx+1}/{len(states_to_label)}] {action_name}")
                
                current_idx += 1
            
            else:
                print(f"  ⚠️  未知按键: '{key_char}' (code: {key})")
        
        cv2.destroyAllWindows()
        
        return self.labeled_data
    
    def _get_action_name(self, key):
        """获取动作名称"""
        action_names = {
            'w': '前进', 's': '后退', 'a': '左移', 'd': '右移',
            'i': '上看', 'k': '下看', 'j': '左看', 'l': '右看',
            'f': '攻击', ' ': '跳跃', 'q': '前进攻击', 'e': '上看攻击'
        }
        return action_names.get(key, '未知')


def main():
    parser = argparse.ArgumentParser(
        description="DAgger交互式标注工具 - 标注策略收集的状态"
    )
    
    parser.add_argument(
        "--states",
        type=str,
        required=True,
        help="状态目录路径（包含episode_*.npy文件）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径（.pkl）"
    )
    
    parser.add_argument(
        "--smart-sampling",
        action="store_true",
        help="启用智能采样（只标注关键状态）"
    )
    
    parser.add_argument(
        "--failure-window",
        type=int,
        default=10,
        help="失败前标注的步数（默认: 10）"
    )
    
    parser.add_argument(
        "--random-sample-rate",
        type=float,
        default=0.1,
        help="成功episode的随机采样率（默认: 0.1 = 10%%）"
    )
    
    args = parser.parse_args()
    
    # 验证states目录存在
    if not os.path.exists(args.states):
        print(f"✗ 错误: 状态目录不存在: {args.states}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建标注器
    labeler = StateLabeler(
        failure_window=args.failure_window,
        random_sample_rate=args.random_sample_rate
    )
    
    # 加载episode
    print(f"\n加载episode数据...")
    episodes = labeler.load_episodes(args.states)
    
    if not episodes:
        print(f"✗ 没有可用的episode数据")
        sys.exit(1)
    
    print(f"✓ 加载了 {len(episodes)} 个episode")
    
    # 统计
    success_count = sum(1 for ep in episodes if ep['success'])
    failure_count = len(episodes) - success_count
    
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {failure_count}")
    
    # 采样状态
    if args.smart_sampling:
        print(f"\n智能采样状态...")
        states_to_label = labeler.smart_sample_states(episodes)
        print(f"✓ 选择了 {len(states_to_label)} 个关键状态")
    else:
        print(f"\n加载所有状态...")
        states_to_label = []
        for ep_idx, episode in enumerate(episodes):
            states = episode['data']['states']
            actions = episode['data'].get('actions', [])
            for state_idx, state in enumerate(states):
                if state_idx < len(actions):
                    states_to_label.append({
                        'episode_idx': ep_idx,
                        'state_idx': state_idx,
                        'state': state,
                        'policy_action': actions[state_idx],
                        'priority': 'medium'
                    })
        print(f"✓ 共 {len(states_to_label)} 个状态")
    
    # 开始标注
    labeled_data = labeler.label_states(states_to_label)
    
    # 保存标注数据
    if labeled_data:
        with open(args.output, 'wb') as f:
            pickle.dump(labeled_data, f)
        
        print(f"\n{'='*60}")
        print(f"标注完成！")
        print(f"{'='*60}")
        print(f"总标注数: {len(labeled_data)}")
        print(f"高优先级: {sum(1 for d in labeled_data if d['priority'] == 'high')}")
        print(f"输出文件: {args.output}")
        print(f"{'='*60}\n")
    else:
        print(f"\n⚠️  未保存任何标注")


if __name__ == "__main__":
    main()

