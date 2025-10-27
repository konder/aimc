"""
VPT权重加载工具

解决VPT预训练权重的key前缀问题：
- VPT权重文件中有'net.'前缀
- MinecraftPolicy不需要这个前缀
"""

import torch
from typing import Dict
import sys
import os

# 添加external路径（用于minerl依赖）
EXTERNAL_PATH = os.path.join(os.path.dirname(__file__), '../../../external')
if EXTERNAL_PATH not in sys.path:
    sys.path.insert(0, EXTERNAL_PATH)


def load_vpt_weights(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    加载VPT权重并修复key前缀问题
    
    Args:
        weights_path: VPT权重文件路径
    
    Returns:
        修复后的state_dict
    """
    print(f'加载VPT权重: {weights_path}')
    
    # 加载原始权重
    raw_weights = torch.load(weights_path, map_location='cpu')
    print(f'  原始keys数量: {len(raw_weights)}')
    
    # 分析key结构
    has_net_prefix = any(k.startswith('net.') for k in raw_weights.keys())
    has_pi_head = any(k.startswith('pi_head') for k in raw_weights.keys())
    has_value_head = any(k.startswith('value_head') for k in raw_weights.keys())
    
    print(f'  结构分析:')
    print(f'    - net.前缀: {has_net_prefix}')
    print(f'    - pi_head: {has_pi_head}')
    print(f'    - value_head: {has_value_head}')
    
    # 修复权重keys
    fixed_weights = {}
    skipped_keys = []
    
    for key, value in raw_weights.items():
        # 去掉net.前缀
        if key.startswith('net.'):
            new_key = key[4:]  # 去掉'net.'
            fixed_weights[new_key] = value
        # 跳过RL专用的head（BC训练不需要）
        elif key.startswith('pi_head') or key.startswith('value_head') or key.startswith('aux_value_head'):
            skipped_keys.append(key)
        else:
            fixed_weights[key] = value
    
    print(f'  修复后keys数量: {len(fixed_weights)}')
    print(f'  跳过的keys: {len(skipped_keys)} (pi_head, value_head等)')
    
    return fixed_weights


def create_vpt_policy(device='cpu'):
    """
    创建VPT MinecraftPolicy
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        MinecraftPolicy实例
    """
    from .lib.policy import MinecraftPolicy
    
    # VPT标准参数
    policy = MinecraftPolicy(
        recurrence_type='transformer',
        attention_heads=16,
        attention_mask_style='clipped_causal',
        attention_memory_size=256,
        diff_mlp_embedding=False,
        hidsize=2048,
        img_shape=[128, 128, 3],
        impala_chans=[16, 32, 32],
        impala_kwargs={'post_pool_groups': 1},
        impala_width=8,
        init_norm_kwargs={'batch_norm': False, 'group_norm_groups': 1},
        n_recurrence_layers=4,
        only_img_input=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        recurrence_is_residual=True,
        timesteps=128,
        use_pointwise_layer=True,
        use_pre_lstm_ln=False,
    )
    
    return policy.to(device)


def load_vpt_policy(weights_path: str, device='cpu', verbose=True):
    """
    创建VPT policy并加载预训练权重
    
    Args:
        weights_path: VPT权重文件路径
        device: 'cpu' or 'cuda'
        verbose: 是否打印详细信息
    
    Returns:
        加载了预训练权重的MinecraftPolicy
    """
    # 创建policy
    policy = create_vpt_policy(device)
    
    if verbose:
        param_count = sum(p.numel() for p in policy.parameters())
        print(f'Policy参数量: {param_count:,}')
    
    # 加载并修复权重
    fixed_weights = load_vpt_weights(weights_path)
    
    # 检查初始权重
    if verbose:
        initial_weight = policy.img_process.cnn.stacks[0].firstconv.layer.weight.clone()
        print(f'初始权重: mean={initial_weight.mean():.6f}, std={initial_weight.std():.6f}')
    
    # 加载权重
    result = policy.load_state_dict(fixed_weights, strict=False)
    
    if verbose:
        print(f'权重加载结果:')
        print(f'  Missing keys: {len(result.missing_keys)}')
        print(f'  Unexpected keys: {len(result.unexpected_keys)}')
        
        # 验证权重已改变
        loaded_weight = policy.img_process.cnn.stacks[0].firstconv.layer.weight
        print(f'加载后权重: mean={loaded_weight.mean():.6f}, std={loaded_weight.std():.6f}')
        
        weights_changed = not torch.allclose(initial_weight, loaded_weight)
        if weights_changed:
            print('✓ VPT预训练权重加载成功！')
        else:
            print('✗ 警告：权重未改变，可能加载失败！')
    
    return policy, result


if __name__ == '__main__':
    """测试权重加载"""
    print('=' * 70)
    print('测试VPT权重加载工具')
    print('=' * 70)
    
    weights_path = 'data/pretrained/vpt/rl-from-early-game-2x.weights'
    
    # 测试加载
    policy, result = load_vpt_policy(weights_path, verbose=True)
    
    print('=' * 70)
    print('✅ 测试完成')

