"""
Agent基类模块

定义所有Agent的统一接口。

具体Agent实现：
- VPTAgent: src.training.vpt.vpt_agent
"""

from .agent_base import AgentBase

__all__ = [
    'AgentBase',
]
