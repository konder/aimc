"""
自定义 MineRL 任务环境
"""

import gym
import logging

# Default 环境（DefaultWorldGenerator）
from .minerl_harvest_default import (
    MineRLHarvestDefaultEnvSpec,
    register_minerl_harvest_default_env,
    _minerl_harvest_default_env_entrypoint,
)

logger = logging.getLogger(__name__)

# 自动注册环境
register_minerl_harvest_default_env()

# 注册旧环境名（向后兼容）
try:
    gym.register(
        id='MineRLHarvestEnv-v0',
        entry_point='src.envs.minerl_harvest_default:_minerl_harvest_default_env_entrypoint'
    )
    logger.info("✓ MineRLHarvestEnv-v0 已注册（别名，指向 Default 环境）")
except gym.error.Error:
    pass  # 已注册

__all__ = [
    'MineRLHarvestDefaultEnvSpec',
    'register_minerl_harvest_default_env',
]
