"""
CLIP4MC 训练模块

用于训练 RL-Friendly Vision-Language Model for Minecraft。
基于 https://github.com/PKU-RL/CLIP4MC
"""

from .train import SimpleCLIP4MC, SimpleClip4MCDataset

__all__ = ['SimpleCLIP4MC', 'SimpleClip4MCDataset']
