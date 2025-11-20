"""
环境配置标准化器
Config Normalizer

功能：将不同格式的环境配置统一为标准格式
"""

from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def normalize_image_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    统一 image_size 和 resolution 配置
    
    支持的格式：
    - image_size: [height, width] (MineDojo 格式)
    - resolution: (width, height) (MineRL 格式)
    
    统一为：image_size = (height, width)
    
    Args:
        config: 环境配置字典
    
    Returns:
        (height, width) 元组
    """
    # 优先使用 image_size
    if 'image_size' in config:
        image_size = config['image_size']
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            # 确保是 (height, width) 格式
            return tuple(image_size)
    
    # 兼容 resolution (MineRL 格式: (width, height))
    if 'resolution' in config:
        resolution = config.pop('resolution')  # 移除旧配置
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            # 转换: (width, height) → (height, width)
            width, height = resolution
            logger.info(f"🔄 配置转换: resolution=({width}, {height}) → image_size=({height}, {width})")
            config['image_size'] = (height, width)
            return (height, width)
    
    # 默认值
    default_size = (160, 256)  # (height, width)
    if 'image_size' not in config:
        config['image_size'] = default_size
    return default_size


def normalize_initial_inventory(config: Dict[str, Any]) -> None:
    """
    统一 initial_inventory 配置，确保使用 'type' 字段
    
    支持的格式：
    - type: "item_name" (标准格式)
    - name: "item_name" (MineDojo 旧格式)
    - entity: "item_name" (非标准格式)
    
    统一为：type = "item_name"
    
    Args:
        config: 环境配置字典（会被修改）
    """
    if 'initial_inventory' not in config:
        return
    
    inventory = config['initial_inventory']
    if not isinstance(inventory, list):
        return
    
    for item in inventory:
        if not isinstance(item, dict):
            continue
        
        # 如果已经有 'type'，跳过
        if 'type' in item:
            continue
        
        # 从 'name' 或 'entity' 复制到 'type'
        if 'name' in item:
            item['type'] = item.pop('name')
            logger.debug(f"🔄 initial_inventory: 'name' → 'type' ({item['type']})")
        elif 'entity' in item:
            item['type'] = item.pop('entity')
            logger.debug(f"🔄 initial_inventory: 'entity' → 'type' ({item['type']})")


def normalize_reward_config(config: Dict[str, Any]) -> None:
    """
    统一 reward_config 配置，使用 'entity' 和 'amount' 字段
    
    MineDojo 格式：
    - target_names: ["item1", "item2"]
    - target_quantities: [1, 2]
    - reward_weights: {"item1": 100, "item2": 50}
    
    统一为 MineRL 格式：
    - reward_config: [
        {"entity": "item1", "amount": 1, "reward": 100},
        {"entity": "item2", "amount": 2, "reward": 50}
      ]
    
    Args:
        config: 环境配置字典（会被修改）
    """
    # 如果已经有 reward_config，确保使用 'entity' 和 'amount'
    if 'reward_config' in config:
        reward_config = config['reward_config']
        if isinstance(reward_config, list):
            for item in reward_config:
                if not isinstance(item, dict):
                    continue
                
                # 统一物品名称字段为 'entity'
                if 'type' in item and 'entity' not in item:
                    item['entity'] = item.pop('type')
                    logger.debug(f"🔄 reward_config: 'type' → 'entity' ({item['entity']})")
                elif 'name' in item and 'entity' not in item:
                    item['entity'] = item.pop('name')
                    logger.debug(f"🔄 reward_config: 'name' → 'entity' ({item['entity']})")
                
                # 统一数量字段为 'amount'
                if 'quantity' in item and 'amount' not in item:
                    item['amount'] = item.pop('quantity')
                    logger.debug(f"🔄 reward_config: 'quantity' → 'amount'")
        return
    
    # 如果有 MineDojo 格式的配置，转换为统一格式
    if 'target_names' in config:
        target_names = config.pop('target_names')
        target_quantities = config.pop('target_quantities', [1] * len(target_names))
        reward_weights = config.pop('reward_weights', {})
        
        # 转换为 reward_config 列表
        reward_config = []
        for i, name in enumerate(target_names):
            reward_config.append({
                'entity': name,
                'amount': target_quantities[i] if i < len(target_quantities) else 1,
                'reward': reward_weights.get(name, 1.0) if isinstance(reward_weights, dict) else 1.0
            })
        
        config['reward_config'] = reward_config
        logger.info(f"🔄 配置转换: target_names/target_quantities → reward_config ({len(reward_config)} 项)")


def normalize_world_generation(config: Dict[str, Any]) -> None:
    """
    统一世界生成配置
    
    移除：
    - world_generator (MineRL 专用)
    - generate_world_type (MineDojo 内部使用)
    
    保留：
    - specified_biome: 指定生物群系（仅 MineDojo 支持）
    
    逻辑：
    - 如果 specified_biome 为空 → generate_world_type = "default"
    - 如果 specified_biome 不为空 → generate_world_type = "specified_biome"
    
    Args:
        config: 环境配置字典（会被修改）
    """
    # 移除 world_generator
    if 'world_generator' in config:
        world_gen = config.pop('world_generator')
        logger.info(f" 移除配置: world_generator (MineRL 专用)")
    
    # 移除显式的 generate_world_type（由 specified_biome 自动决定）
    if 'generate_world_type' in config:
        old_type = config.pop('generate_world_type')
        logger.debug(f" 移除配置: generate_world_type={old_type} (自动推断)")
    
    # 根据 specified_biome 自动设置 generate_world_type
    # 注意：这个字段只在 MineDojo 环境内部使用，不在配置中显式设置


def normalize_spawn_and_time(config: Dict[str, Any]) -> None:
    """
    统一生成和时间配置
    
    移除：
    - time_condition (MineRL 嵌套格式)
    - spawning_condition (MineRL 嵌套格式)
    - allow_passage_of_time (MineRL 格式)
    - allow_time_passage (MineDojo 格式) - 保留这个
    - allow_spawning (MineRL 格式)
    
    保留：
    - start_time: 起始时间
    - allow_mob_spawn: 是否允许生物生成（统一名称）
    
    Args:
        config: 环境配置字典（会被修改）
    """
    # 处理 time_condition (MineRL 嵌套格式)
    if 'time_condition' in config:
        time_cond = config.pop('time_condition')
        logger.info(f" 移除配置: time_condition (使用扁平化配置)")
        
        # 提取 start_time
        if 'start_time' in time_cond and 'start_time' not in config:
            config['start_time'] = time_cond['start_time']
        
        # 提取 allow_passage_of_time
        if 'allow_passage_of_time' in time_cond:
            # 注意：不再使用这个配置，时间默认不流逝
            pass
    
    # 移除 allow_passage_of_time 和 allow_time_passage
    if 'allow_passage_of_time' in config:
        config.pop('allow_passage_of_time')
        logger.info(f" 移除配置: allow_passage_of_time (时间默认不流逝)")
    
    if 'allow_time_passage' in config:
        config.pop('allow_time_passage')
        logger.info(f" 移除配置: allow_time_passage (时间默认不流逝)")
    
    # 处理 spawning_condition (MineRL 嵌套格式)
    if 'spawning_condition' in config:
        spawn_cond = config.pop('spawning_condition')
        logger.info(f" 移除配置: spawning_condition (使用扁平化配置)")
        
        # 提取 allow_spawning
        if 'allow_spawning' in spawn_cond and 'allow_mob_spawn' not in config:
            config['allow_mob_spawn'] = spawn_cond['allow_spawning']
    
    # 统一为 allow_mob_spawn
    if 'allow_spawning' in config:
        if 'allow_mob_spawn' not in config:
            config['allow_mob_spawn'] = config['allow_spawning']
        config.pop('allow_spawning')
        logger.debug(f"🔄 配置转换: allow_spawning → allow_mob_spawn")


def normalize_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化环境配置
    
    统一所有配置项为标准格式：
    1. image_size: (height, width)
    2. initial_inventory: [{type, quantity}, ...]
    3. reward_config: [{entity, amount, reward}, ...]
    4. specified_biome: str (可选)
    5. start_time: int
    6. allow_mob_spawn: bool
    
    Args:
        config: 原始环境配置
    
    Returns:
        标准化后的配置
    """
    # 创建副本，避免修改原配置
    config = config.copy()
    
    #logger.info("📋 开始标准化环境配置...")
    
    # 1. 统一图像尺寸
    normalize_image_size(config)
    
    # 2. 统一初始物品栏
    normalize_initial_inventory(config)
    
    # 3. 统一奖励配置
    normalize_reward_config(config)
    
    # 4. 统一世界生成
    normalize_world_generation(config)
    
    # 5. 统一生成和时间配置
    normalize_spawn_and_time(config)
    
    #logger.info("✓ 环境配置标准化完成")
    
    return config


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 80)
    print("环境配置标准化测试")
    print("=" * 80)
    
    # 测试 1: MineRL 格式配置
    print("\n### 测试 1: MineRL 格式配置")
    minerl_config = {
        'resolution': (640, 320),  # (width, height)
        'initial_inventory': [
            {'name': 'oak_planks', 'quantity': 2},
            {'type': 'stick', 'quantity': 4}
        ],
        'target_names': ['oak_planks', 'stick'],
        'target_quantities': [1, 4],
        'reward_weights': {'oak_planks': 100, 'stick': 50},
        'world_generator': {'force_reset': True},
        'time_condition': {
            'allow_passage_of_time': False,
            'start_time': 6000
        },
        'spawning_condition': {
            'allow_spawning': True
        }
    }
    
    normalized = normalize_env_config(minerl_config)
    print("\n标准化后的配置:")
    for key, value in normalized.items():
        print(f"  {key}: {value}")
    
    # 测试 2: MineDojo 格式配置
    print("\n### 测试 2: MineDojo 格式配置")
    minedojo_config = {
        'image_size': [320, 640],  # [height, width]
        'initial_inventory': [
            {'type': 'planks', 'quantity': 2}
        ],
        'reward_config': [
            {'entity': 'planks', 'amount': 1, 'reward': 100}
        ],
        'specified_biome': 'forest',
        'generate_world_type': 'specified_biome',
        'start_time': 6000,
        'allow_time_passage': False,
        'allow_mob_spawn': False
    }
    
    normalized = normalize_env_config(minedojo_config)
    print("\n标准化后的配置:")
    for key, value in normalized.items():
        print(f"  {key}: {value}")
    
    # 测试 3: 混合格式配置
    print("\n### 测试 3: 混合格式配置")
    mixed_config = {
        'resolution': (640, 320),
        'initial_inventory': [
            {'name': 'oak_planks', 'quantity': 2},
            {'type': 'stick', 'quantity': 4}
        ],
        'reward_config': [
            {'type': 'oak_planks', 'quantity': 1, 'reward': 100}
        ],
        'specified_biome': 'plains',
        'allow_spawning': True,
        'start_time': 12000
    }
    
    normalized = normalize_env_config(mixed_config)
    print("\n标准化后的配置:")
    for key, value in normalized.items():
        print(f"  {key}: {value}")

