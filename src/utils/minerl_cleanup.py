"""
MineRL Saves 自动清理工具（简化版）

用途：防止 MCP-Reborn/saves/ 目录积累，自动清理旧存档
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def get_minerl_saves_dir() -> Path:
    """
    获取 MineRL saves 目录路径
    
    Returns:
        Path: MineRL saves 目录路径
    """
    try:
        import minerl
        minerl_path = Path(minerl.__file__).parent
        saves_dir = minerl_path / "MCP-Reborn" / "saves"
        return saves_dir
    except Exception as e:
        logger.error(f"无法获取 MineRL saves 目录: {e}")
        return None


def clean_minerl_saves() -> Tuple[int, float]:
    """
    清理 MineRL saves 目录中的所有存档
    
    Returns:
        Tuple[int, float]: (删除的存档数量, 释放的空间 MB)
    """
    saves_dir = get_minerl_saves_dir()
    
    if not saves_dir or not saves_dir.exists():
        logger.debug("MineRL saves 目录不存在，跳过清理")
        return 0, 0.0
    
    removed_count = 0
    freed_bytes = 0
    
    try:
        # 遍历所有世界存档目录
        for world_dir in saves_dir.iterdir():
            if world_dir.is_dir():
                try:
                    # 计算目录大小
                    dir_size = sum(
                        f.stat().st_size 
                        for f in world_dir.rglob('*') 
                        if f.is_file()
                    )
                    
                    # 删除目录
                    shutil.rmtree(world_dir)
                    removed_count += 1
                    freed_bytes += dir_size
                    
                except Exception as e:
                    logger.warning(f"删除存档 {world_dir.name} 失败: {e}")
        
        freed_mb = freed_bytes / (1024 * 1024)
        
        if removed_count > 0:
            logger.debug(f"已清理 {removed_count} 个 MineRL 存档，释放 {freed_mb:.1f} MB 空间")
        
        return removed_count, freed_mb
    
    except Exception as e:
        logger.error(f"清理 MineRL saves 时出错: {e}")
        return 0, 0.0

