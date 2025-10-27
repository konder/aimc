#!/usr/bin/env python
"""
验证VPT模型是否正确使用了预训练权重
"""

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def verify_checkpoint(checkpoint_path):
    """
    验证checkpoint是否包含VPT信息
    
    Args:
        checkpoint_path: checkpoint文件路径
    """
    print('=' * 70)
    print('🔍 VPT模型验证')
    print('=' * 70)
    print(f'Checkpoint: {checkpoint_path}')
    print()
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f'✗ 无法加载checkpoint: {e}')
        return False
    
    print('✓ Checkpoint加载成功')
    print()
    
    # 检查关键信息
    print('📋 Checkpoint内容:')
    print('-' * 70)
    
    # 1. 检查是否有VPT权重路径
    vpt_weights_path = checkpoint.get('vpt_weights_path', None)
    if vpt_weights_path:
        print(f'✓ VPT权重路径: {vpt_weights_path}')
        if os.path.exists(vpt_weights_path):
            print(f'  ✓ 权重文件存在')
        else:
            print(f'  ✗ 警告: 权重文件不存在')
    else:
        print(f'✗ 未找到VPT权重路径')
        print(f'  这可能是一个纯BC模型，没有使用VPT预训练')
        return False
    
    print()
    
    # 2. 训练信息
    print('📊 训练信息:')
    print(f'  Epoch: {checkpoint.get("epoch", "unknown")}')
    
    loss = checkpoint.get('loss', None)
    if isinstance(loss, (int, float)):
        print(f'  Loss: {loss:.4f}')
    else:
        print(f'  Loss: {loss}')
    
    print()
    
    # 3. 模型结构
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print('📦 模型结构:')
        print(f'  参数数量: {len(state_dict)} 个tensors')
        
        # 检查是否有VPT相关的层
        vpt_layers = [k for k in state_dict.keys() if 'vpt_policy' in k]
        action_layers = [k for k in state_dict.keys() if 'action_heads' in k]
        
        print(f'  VPT相关层: {len(vpt_layers)} 个')
        print(f'  Action heads: {len(action_layers)} 个')
        
        # 显示一些关键层
        print()
        print('  关键层（前10个）:')
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape
            print(f'    {i+1}. {key}: {shape}')
        
        if len(state_dict) > 10:
            print(f'    ... 还有 {len(state_dict) - 10} 个')
    
    print()
    
    # 4. 总参数量
    if 'model_state_dict' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        print(f'📈 总参数量: {total_params:,}')
        print(f'   预期VPT模型约231M参数')
        
        if total_params > 200_000_000:
            print(f'   ✓ 参数量符合VPT模型规模')
        else:
            print(f'   ⚠️  参数量偏少，可能不是完整的VPT模型')
    
    print()
    print('=' * 70)
    print('✅ 验证完成')
    print('=' * 70)
    
    return True


def compare_with_baseline(vpt_checkpoint, bc_checkpoint=None):
    """
    对比VPT+BC模型和纯BC模型
    
    Args:
        vpt_checkpoint: VPT+BC checkpoint路径
        bc_checkpoint: 纯BC checkpoint路径（可选）
    """
    print()
    print('=' * 70)
    print('📊 模型对比')
    print('=' * 70)
    
    # 加载VPT模型
    print('VPT+BC模型:')
    vpt_ckpt = torch.load(vpt_checkpoint, map_location='cpu')
    vpt_params = sum(p.numel() for p in vpt_ckpt['model_state_dict'].values())
    print(f'  参数量: {vpt_params:,}')
    print(f'  Epoch: {vpt_ckpt.get("epoch", "unknown")}')
    print(f'  Loss: {vpt_ckpt.get("loss", "unknown")}')
    print(f'  有VPT权重: {"✓" if vpt_ckpt.get("vpt_weights_path") else "✗"}')
    
    print()
    
    # 如果提供了BC模型，进行对比
    if bc_checkpoint and os.path.exists(bc_checkpoint):
        print('纯BC模型:')
        bc_ckpt = torch.load(bc_checkpoint, map_location='cpu')
        bc_params = sum(p.numel() for p in bc_ckpt['model_state_dict'].values())
        print(f'  参数量: {bc_params:,}')
        print(f'  Epoch: {bc_ckpt.get("epoch", "unknown")}')
        print(f'  Loss: {bc_ckpt.get("loss", "unknown")}')
        print(f'  有VPT权重: {"✓" if bc_ckpt.get("vpt_weights_path") else "✗"}')
        
        print()
        print('对比:')
        print(f'  参数量差异: {abs(vpt_params - bc_params):,}')
        
        if vpt_params > bc_params * 10:
            print(f'  ✓ VPT模型明显更大，使用了预训练权重')
        else:
            print(f'  ⚠️  两个模型大小相近，可能都是同类型')
    else:
        print('未提供纯BC模型进行对比')
    
    print('=' * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='验证VPT模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='VPT+BC模型checkpoint路径')
    parser.add_argument('--baseline', type=str, default=None,
                        help='纯BC模型checkpoint路径（可选，用于对比）')
    
    args = parser.parse_args()
    
    # 验证checkpoint
    success = verify_checkpoint(args.checkpoint)
    
    # 如果提供了baseline，进行对比
    if success and args.baseline:
        compare_with_baseline(args.checkpoint, args.baseline)

