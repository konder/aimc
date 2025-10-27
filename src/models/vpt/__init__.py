"""
VPT (Video Pre-Training) 模型相关模块

核心功能：
- weights_loader: 加载和管理VPT预训练权重
"""

from .weights_loader import load_vpt_policy, create_vpt_policy, load_vpt_weights

__all__ = [
    'load_vpt_policy',
    'create_vpt_policy',
    'load_vpt_weights',
]

