"""
自定义 MineRL 和 MineDojo 任务环境
==================================

目录结构：
- minerl_wrappers.py:    MineRL 自定义环境和 Wrappers
- minedojo_wrappers.py:  MineDojo 自定义环境和 Wrappers
- env_bridge.py:         MineRL ↔ MineDojo 配置转换工具
"""

import logging

logger = logging.getLogger(__name__)

# 延迟导入以避免循环依赖和缺少依赖的问题
_minerl_registered = False
_minedojo_registered = False


def _try_register_minerl():
    """尝试注册 MineRL 环境"""
    global _minerl_registered
    if _minerl_registered:
        return True
    
    try:
        from .minerl_wrappers import register_minerl_harvest_default_env
        register_minerl_harvest_default_env()
        _minerl_registered = True
        logger.debug("✓ MineRL 环境已注册")
        return True
    except ImportError as e:
        logger.debug(f"MineRL 环境注册跳过: {e}")
        return False


def _try_register_minedojo():
    """尝试注册 MineDojo 环境"""
    global _minedojo_registered
    if _minedojo_registered:
        return True
    
    try:
        from .minedojo_wrappers import register_minedojo_biome_env
        register_minedojo_biome_env()
        _minedojo_registered = True
        logger.debug("✓ MineDojo 环境已注册")
        return True
    except ImportError as e:
        logger.debug(f"MineDojo 环境注册跳过: {e}")
        return False


# 环境桥接工具（总是可用）
from .env_bridge import (
    minerl_to_minedojo,
    minedojo_to_minerl,
    normalize_env_config,
    convert_initial_inventory,
    convert_reward_config,
    normalize_entity_name,
    get_entity_name_variants,
)

# 尝试自动注册环境
_try_register_minerl()
_try_register_minedojo()


def get_minerl_wrappers():
    """获取 MineRL Wrappers（按需导入）"""
    from .minerl_wrappers import (
        MineRLHarvestWrapper,
        MineRLHarvestDefaultEnvSpec,
        register_minerl_harvest_default_env,
    )
    return {
        'MineRLHarvestWrapper': MineRLHarvestWrapper,
        'MineRLHarvestDefaultEnvSpec': MineRLHarvestDefaultEnvSpec,
        'register_minerl_harvest_default_env': register_minerl_harvest_default_env,
    }


def get_minedojo_wrappers():
    """获取 MineDojo Wrappers（按需导入）"""
    from .minedojo_wrappers import (
        MineDojoBiomeEnvSpec,
        MineDojoBiomeWrapper,
        register_minedojo_biome_env,
    )
    return {
        'MineDojoBiomeEnvSpec': MineDojoBiomeEnvSpec,
        'MineDojoBiomeWrapper': MineDojoBiomeWrapper,
        'register_minedojo_biome_env': register_minedojo_biome_env,
    }


__all__ = [
    # 桥接工具
    'minerl_to_minedojo',
    'minedojo_to_minerl',
    'normalize_env_config',
    'convert_initial_inventory',
    'convert_reward_config',
    'normalize_entity_name',
    'get_entity_name_variants',
    
    # 按需导入函数
    'get_minerl_wrappers',
    'get_minedojo_wrappers',
]
