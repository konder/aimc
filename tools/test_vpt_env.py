#!/usr/bin/env python
"""
快速验证VPT环境是否正常
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.vpt import load_vpt_policy


def test_vpt_environment():
    print('=' * 70)
    print('🔍 VPT环境验证')
    print('=' * 70)
    
    weights_path = 'data/pretrained/vpt/rl-from-early-game-2x.weights'
    
    try:
        print(f'\n1. 加载VPT权重: {weights_path}')
        policy, result = load_vpt_policy(weights_path, device='cpu', verbose=True)
        
        print(f'\n2. 验证加载结果')
        missing = len(result.missing_keys)
        unexpected = len(result.unexpected_keys)
        
        if missing > 0:
            print(f'   ✗ Missing keys: {missing}')
            print(f'     {result.missing_keys[:5]}...')
            return False
            
        if unexpected > 0:
            print(f'   ✗ Unexpected keys: {unexpected}')
            print(f'     {result.unexpected_keys[:5]}...')
            return False
        
        print(f'   ✓ 权重加载完美: Missing=0, Unexpected=0')
        
        print(f'\n3. 统计参数')
        total_params = sum(p.numel() for p in policy.parameters())
        print(f'   ✓ 总参数: {total_params:,}')
        
        print('\n' + '=' * 70)
        print('✅ VPT环境验证成功！可以开始训练了')
        print('=' * 70)
        return True
        
    except Exception as e:
        print(f'\n✗ VPT环境验证失败: {e}')
        print('\n请检查:')
        print('  1. VPT权重文件是否存在')
        print('  2. 依赖是否正确安装')
        print('  3. Python路径是否正确')
        return False


if __name__ == '__main__':
    success = test_vpt_environment()
    sys.exit(0 if success else 1)
