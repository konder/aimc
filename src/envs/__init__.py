"""
自定义 MineRL 和 MineDojo 任务环境
"""

import logging

# MineRL 环境
from .minerl_harvest_default import (
    MineRLHarvestDefaultEnvSpec,
    register_minerl_harvest_default_env,
)

# MineDojo 环境
from .minedojo_harvest import (
    MineDojoBiomeEnvSpec,
    register_minedojo_biome_env,
)

logger = logging.getLogger(__name__)

# 自动注册环境
register_minerl_harvest_default_env()
register_minedojo_biome_env()

__all__ = [
    'MineRLHarvestDefaultEnvSpec',
    'register_minerl_harvest_default_env',
    'MineDojoBiomeEnvSpec',
    'register_minedojo_biome_env',
]
