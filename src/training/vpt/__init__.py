"""
VPT (Video PreTraining) 模块

包含VPT相关的训练、评估和Agent实现。

架构层次：
1. src/models/vpt/agent.py::MineRLAgent - 官方VPT实现
2. src/models/vpt/minedojo_agent.py::MineDojoAgent - MineDojo适配
3. src/training/vpt/vpt_agent.py::VPTAgent - 训练/评估接口

官方VPT参考：
- GitHub: https://github.com/openai/Video-Pre-Training
- 本地VPT代码: src/models/vpt/
"""

from .vpt_agent import VPTAgent

__all__ = [
    'VPTAgent',
]
