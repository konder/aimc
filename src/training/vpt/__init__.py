"""
VPT (Video PreTraining) 模块

包含VPT相关的训练、评估和Agent实现。

官方VPT参考：
- GitHub: https://github.com/openai/Video-Pre-Training
- 本地官方代码: src/models/Video-Pre-Training/
"""

from .vpt_agent import VPTAgent, MineRLToMinedojoConverter

__all__ = [
    'VPTAgent',
    'MineRLToMinedojoConverter',
]
