#!/usr/bin/env python3
"""
改进的智能采样策略 - 基于奖励变化识别关键决策点

核心思想：
1. 分析整个episode的奖励序列
2. 识别"奖励下降/停滞"的关键时刻
3. 在这些关键时刻周围进行密集采样
4. 对于"前期正确"的episode，避免过度采样正确行为
"""

import numpy as np
from scipy.signal import find_peaks


def analyze_episode_trajectory(episode_data):
    """
    分析episode轨迹，识别关键决策点
    
    Args:
        episode_data: episode数据，包含states, actions, rewards
    
    Returns:
        dict: 分析结果
            - error_regions: 错误决策的时间区间列表 [(start, end), ...]
            - critical_points: 关键决策点索引列表
            - trajectory_type: 轨迹类型 ('early_error', 'late_error', 'mixed')
    """
    rewards = episode_data.get('rewards', [])
    num_steps = len(rewards)
    
    if num_steps == 0:
        return {
            'error_regions': [],
            'critical_points': [],
            'trajectory_type': 'unknown'
        }
    
    # 计算累积奖励
    cumulative_rewards = np.cumsum(rewards)
    
    # 计算奖励的移动平均（平滑噪声）
    window_size = min(10, num_steps // 5)
    if window_size > 0:
        smoothed_rewards = np.convolve(
            cumulative_rewards, 
            np.ones(window_size)/window_size, 
            mode='same'
        )
    else:
        smoothed_rewards = cumulative_rewards
    
    # 计算奖励变化率（一阶导数）
    reward_velocity = np.gradient(smoothed_rewards)
    
    # 识别"奖励停滞/下降"的区间
    # 这些区间通常对应错误的决策
    error_threshold = np.percentile(reward_velocity, 25)  # 下四分位数
    error_mask = reward_velocity < error_threshold
    
    # 将连续的错误点合并为区间
    error_regions = []
    in_error = False
    start_idx = 0
    
    for i, is_error in enumerate(error_mask):
        if is_error and not in_error:
            start_idx = i
            in_error = True
        elif not is_error and in_error:
            error_regions.append((start_idx, i))
            in_error = False
    
    if in_error:  # 如果最后还在错误区间
        error_regions.append((start_idx, num_steps))
    
    # 识别关键转折点（奖励变化率的极值点）
    # 负的峰值 = 奖励突然下降 = 可能的错误决策
    negative_velocity = -reward_velocity
    peaks, _ = find_peaks(negative_velocity, distance=5, prominence=0.1)
    critical_points = peaks.tolist()
    
    # 判断轨迹类型
    first_third = num_steps // 3
    last_third = 2 * num_steps // 3
    
    early_errors = sum(1 for i in critical_points if i < first_third)
    late_errors = sum(1 for i in critical_points if i > last_third)
    
    if early_errors > late_errors:
        trajectory_type = 'early_error'
    elif late_errors > early_errors:
        trajectory_type = 'late_error'
    else:
        trajectory_type = 'mixed'
    
    return {
        'error_regions': error_regions,
        'critical_points': critical_points,
        'trajectory_type': trajectory_type,
        'reward_velocity': reward_velocity,
        'cumulative_rewards': cumulative_rewards
    }


def smart_sample_states_improved(episodes, failure_window=10, random_sample_rate=0.1):
    """
    改进的智能采样策略
    
    策略：
    1. 成功episode: 随机采样（保持不变）
    2. 失败episode:
       a. 分析奖励轨迹，识别错误区间
       b. 在错误区间内密集采样
       c. 在关键转折点周围采样
       d. 避免过度采样已经正确的行为
    
    Args:
        episodes: episode列表
        failure_window: 每个错误区间采样的窗口大小
        random_sample_rate: 成功episode的采样率
    
    Returns:
        List of states to label
    """
    states_to_label = []
    
    for ep_idx, episode in enumerate(episodes):
        episode_data = episode['data']
        states = episode_data['states']
        actions = episode_data.get('actions', [])
        rewards = episode_data.get('rewards', [])
        success = episode['success']
        
        if success:
            # 成功episode: 随机采样（保持原策略）
            num_samples = max(1, int(len(states) * random_sample_rate))
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
                        'priority': 'low'
                    })
        else:
            # 失败episode: 智能采样
            analysis = analyze_episode_trajectory(episode_data)
            
            sampled_indices = set()
            
            # 策略1: 在错误区间内采样
            for start, end in analysis['error_regions']:
                # 在每个错误区间内均匀采样
                region_length = end - start
                num_samples_in_region = min(failure_window, region_length)
                
                if region_length > 0:
                    step = max(1, region_length // num_samples_in_region)
                    for i in range(start, end, step):
                        sampled_indices.add(i)
            
            # 策略2: 在关键转折点周围采样
            for critical_point in analysis['critical_points']:
                # 在转折点前后各采样几步
                window = 3
                for offset in range(-window, window + 1):
                    idx = critical_point + offset
                    if 0 <= idx < len(states):
                        sampled_indices.add(idx)
            
            # 策略3: 如果采样点太少，补充最后几步
            if len(sampled_indices) < failure_window // 2:
                start_idx = max(0, len(states) - failure_window)
                for i in range(start_idx, len(states)):
                    sampled_indices.add(i)
            
            # 策略4: 限制总采样数（避免过度标注）
            max_samples = min(failure_window * 2, len(states))
            sampled_indices = sorted(sampled_indices)[:max_samples]
            
            # 添加到标注列表
            for state_idx in sampled_indices:
                if state_idx < len(actions):
                    # 根据轨迹类型设置优先级
                    if analysis['trajectory_type'] == 'early_error':
                        priority = 'critical' if state_idx < len(states) // 3 else 'high'
                    elif analysis['trajectory_type'] == 'late_error':
                        priority = 'critical' if state_idx > 2 * len(states) // 3 else 'high'
                    else:
                        priority = 'high'
                    
                    states_to_label.append({
                        'episode_idx': ep_idx,
                        'state_idx': state_idx,
                        'state': states[state_idx],
                        'policy_action': actions[state_idx],
                        'priority': priority,
                        'trajectory_type': analysis['trajectory_type']
                    })
    
    return states_to_label


def visualize_sampling_strategy(episode_data, sampled_indices):
    """
    可视化采样策略（用于调试和分析）
    
    Args:
        episode_data: episode数据
        sampled_indices: 采样的状态索引
    """
    import matplotlib.pyplot as plt
    
    rewards = episode_data.get('rewards', [])
    cumulative_rewards = np.cumsum(rewards)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制累积奖励
    plt.subplot(2, 1, 1)
    plt.plot(cumulative_rewards, label='Cumulative Reward', linewidth=2)
    plt.scatter(sampled_indices, cumulative_rewards[sampled_indices], 
                color='red', s=50, label='Sampled States', zorder=5)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Episode Trajectory & Sampling Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制奖励变化率
    plt.subplot(2, 1, 2)
    reward_velocity = np.gradient(cumulative_rewards)
    plt.plot(reward_velocity, label='Reward Velocity', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.scatter(sampled_indices, reward_velocity[sampled_indices], 
                color='red', s=50, label='Sampled States', zorder=5)
    plt.xlabel('Step')
    plt.ylabel('Reward Change Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sampling_strategy_visualization.png')
    print("✓ 可视化已保存: sampling_strategy_visualization.png")


# 示例：如何使用改进的采样策略
if __name__ == "__main__":
    # 模拟一些测试数据
    print("改进的智能采样策略示例\n")
    
    # 场景A: 前期正确，后期错误
    episode_a = {
        'data': {
            'states': [{'obs': i} for i in range(100)],
            'actions': [0] * 100,
            'rewards': [0.1] * 80 + [0.0] * 10 + [-0.1] * 10  # 前80步有正奖励
        },
        'success': False
    }
    
    analysis_a = analyze_episode_trajectory(episode_a['data'])
    print("场景A - 前期正确，后期错误:")
    print(f"  轨迹类型: {analysis_a['trajectory_type']}")
    print(f"  错误区间: {analysis_a['error_regions']}")
    print(f"  关键点数: {len(analysis_a['critical_points'])}")
    
    # 场景B: 从头到尾都错
    episode_b = {
        'data': {
            'states': [{'obs': i} for i in range(100)],
            'actions': [0] * 100,
            'rewards': [0.0] * 100  # 全程无奖励
        },
        'success': False
    }
    
    analysis_b = analyze_episode_trajectory(episode_b['data'])
    print("\n场景B - 从头到尾都错:")
    print(f"  轨迹类型: {analysis_b['trajectory_type']}")
    print(f"  错误区间: {analysis_b['error_regions']}")
    print(f"  关键点数: {len(analysis_b['critical_points'])}")
    
    print("\n改进策略的优势:")
    print("✓ 自动识别错误发生的时间段")
    print("✓ 在关键决策点周围密集采样")
    print("✓ 避免过度标注已经正确的行为")
    print("✓ 适应不同的失败模式")

