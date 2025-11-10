"""
自定义 MineRL 任务环境
"""

from .minerl_harvest import (
    MineRLHarvestEnvSpec,
    MineRLHarvestWrapper,
    register_minerl_harvest_env,
)

# 自动注册环境
register_minerl_harvest_env()

__all__ = [
    'MineRLHarvestEnvSpec',
    'MineRLHarvestWrapper',
    'register_minerl_harvest_env',
]
