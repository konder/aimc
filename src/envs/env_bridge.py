"""
环境桥接工具 (Environment Bridge)
=================================

在 MineRL 和 MineDojo 环境之间进行配置转换和物品名称映射

功能：
1. 配置标准化 - 统一不同格式的环境配置
2. 物品名称映射 - MineRL ↔ MineDojo 物品名称转换
3. 实体名称规范化 - 统一实体名称格式

MineRL 使用格式：
  - 物品带前缀：'minecraft:oak_planks', 'minecraft:stick'
  - 配置嵌套：time_condition, spawning_condition

MineDojo 使用格式：
  - 物品不带前缀：'planks', 'stick', 'log'
  - 配置扁平化：start_time, allow_mob_spawn

作者: AI Assistant
日期: 2025-12-03
"""

from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 物品名称映射表
# =============================================================================

# MineRL → MineDojo 物品名称映射
MINERL_TO_MINEDOJO_ITEM_MAP = {
    # 木材类 (Wood)
    "oak_planks": "planks",
    "spruce_planks": "planks",
    "birch_planks": "planks",
    "jungle_planks": "planks",
    "acacia_planks": "planks",
    "dark_oak_planks": "planks",
    
    # 原木类 (Logs)
    "oak_log": "log",
    "spruce_log": "log",
    "birch_log": "log",
    "jungle_log": "log",
    "acacia_log": "log",
    "dark_oak_log": "log",
    "log": "log",
    
    # 木棍 (Stick)
    "stick": "stick",
    
    # 工具 - 木制
    "wooden_pickaxe": "wooden_pickaxe",
    "wooden_axe": "wooden_axe",
    "wooden_shovel": "wooden_shovel",
    "wooden_sword": "wooden_sword",
    "wooden_hoe": "wooden_hoe",
    
    # 工具 - 石制
    "stone_pickaxe": "stone_pickaxe",
    "stone_axe": "stone_axe",
    "stone_shovel": "stone_shovel",
    "stone_sword": "stone_sword",
    "stone_hoe": "stone_hoe",
    
    # 工具 - 铁制
    "iron_pickaxe": "iron_pickaxe",
    "iron_axe": "iron_axe",
    "iron_shovel": "iron_shovel",
    "iron_sword": "iron_sword",
    "iron_hoe": "iron_hoe",
    
    # 工具 - 金制
    "golden_pickaxe": "golden_pickaxe",
    "golden_axe": "golden_axe",
    "golden_shovel": "golden_shovel",
    "golden_sword": "golden_sword",
    "golden_hoe": "golden_hoe",
    
    # 工具 - 钻石
    "diamond_pickaxe": "diamond_pickaxe",
    "diamond_axe": "diamond_axe",
    "diamond_shovel": "diamond_shovel",
    "diamond_sword": "diamond_sword",
    "diamond_hoe": "diamond_hoe",
    
    # 方块 - 基础
    "dirt": "dirt",
    "cobblestone": "cobblestone",
    "stone": "stone",
    "sand": "sand",
    "gravel": "gravel",
    "clay": "clay",
    
    # 矿石
    "coal_ore": "coal_ore",
    "iron_ore": "iron_ore",
    "gold_ore": "gold_ore",
    "diamond_ore": "diamond_ore",
    "redstone_ore": "redstone_ore",
    "lapis_ore": "lapis_ore",
    "emerald_ore": "emerald_ore",
    
    # 矿物
    "coal": "coal",
    "iron_ingot": "iron_ingot",
    "gold_ingot": "gold_ingot",
    "diamond": "diamond",
    "redstone": "redstone",
    "lapis_lazuli": "dye",
    "emerald": "emerald",
    
    # 食物
    "apple": "apple",
    "bread": "bread",
    "cooked_beef": "cooked_beef",
    "cooked_porkchop": "cooked_porkchop",
    "cooked_chicken": "cooked_chicken",
    "cooked_mutton": "cooked_mutton",
    "beef": "beef",
    "porkchop": "porkchop",
    "chicken": "chicken",
    "mutton": "mutton",
    
    # 动物掉落物
    "leather": "leather",
    "feather": "feather",
    "wool": "wool",
    "white_wool": "wool",
    
    # 容器
    "bucket": "bucket",
    "water_bucket": "water_bucket",
    "lava_bucket": "lava_bucket",
    "milk_bucket": "milk_bucket",
    
    # 合成物品
    "crafting_table": "crafting_table",
    "furnace": "furnace",
    "chest": "chest",
    "torch": "torch",
    
    # 农作物
    "wheat": "wheat",
    "wheat_seeds": "wheat_seeds",
    "carrot": "carrot",
    "potato": "potato",
    "beetroot": "beetroot",
    "beetroot_seeds": "beetroot_seeds",
    
    # 植物
    "sapling": "sapling",
    "oak_sapling": "sapling",
    "spruce_sapling": "sapling",
    "birch_sapling": "sapling",
    "jungle_sapling": "sapling",
    "acacia_sapling": "sapling",
    "dark_oak_sapling": "sapling",
    
    # 花朵
    "dandelion": "yellow_flower",
    "poppy": "red_flower",
    "blue_orchid": "red_flower",
    "allium": "red_flower",
    "azure_bluet": "red_flower",
    "red_tulip": "red_flower",
    "orange_tulip": "red_flower",
    "white_tulip": "red_flower",
    "pink_tulip": "red_flower",
    "oxeye_daisy": "red_flower",
    
    # 蘑菇
    "brown_mushroom": "brown_mushroom",
    "red_mushroom": "red_mushroom",
    
    # 其他
    "snowball": "snowball",
    "snow": "snow",
    "ice": "ice",
    "sugar_cane": "reeds",
    "pumpkin": "pumpkin",
}

# MineDojo → MineRL 物品名称映射（反向映射）
MINEDOJO_TO_MINERL_ITEM_MAP = {
    # 基础映射（1对1）
    "stick": "stick",
    "dirt": "dirt",
    "cobblestone": "cobblestone",
    "stone": "stone",
    "coal": "coal",
    "iron_ingot": "iron_ingot",
    "gold_ingot": "gold_ingot",
    "diamond": "diamond",
    "bucket": "bucket",
    "milk_bucket": "milk_bucket",
    "crafting_table": "crafting_table",
    
    # 通用名称 → 具体变体（默认使用 oak）
    "planks": "oak_planks",
    "log": "oak_log",
    "sapling": "oak_sapling",
    
    # 特殊映射
    "dye": "lapis_lazuli",
    "reeds": "sugar_cane",
    "yellow_flower": "dandelion",
    "red_flower": "poppy",
    "wool": "white_wool",
}


# =============================================================================
# 实体名称映射（用于战斗任务）
# =============================================================================

# 标准实体名称（小写）
ENTITY_NAMES = {
    # 敌对生物
    "zombie": ["zombie", "Zombie", "minecraft:zombie"],
    "skeleton": ["skeleton", "Skeleton", "minecraft:skeleton"],
    "spider": ["spider", "Spider", "minecraft:spider"],
    "creeper": ["creeper", "Creeper", "minecraft:creeper"],
    "enderman": ["enderman", "Enderman", "minecraft:enderman"],
    "witch": ["witch", "Witch", "minecraft:witch"],
    "slime": ["slime", "Slime", "minecraft:slime"],
    
    # 被动生物
    "chicken": ["chicken", "Chicken", "minecraft:chicken"],
    "cow": ["cow", "Cow", "minecraft:cow"],
    "pig": ["pig", "Pig", "minecraft:pig"],
    "sheep": ["sheep", "Sheep", "minecraft:sheep"],
    "rabbit": ["rabbit", "Rabbit", "minecraft:rabbit"],
    
    # 中立生物
    "wolf": ["wolf", "Wolf", "minecraft:wolf"],
    "iron_golem": ["iron_golem", "IronGolem", "minecraft:iron_golem"],
}


# =============================================================================
# 物品名称转换函数
# =============================================================================

def strip_minecraft_prefix(item_name: str) -> str:
    """移除 'minecraft:' 前缀"""
    if item_name.startswith("minecraft:"):
        return item_name[len("minecraft:"):]
    return item_name


def minerl_to_minedojo(item_name: str) -> str:
    """
    将 MineRL 物品名称转换为 MineDojo 物品名称
    
    Examples:
        >>> minerl_to_minedojo('minecraft:oak_planks')
        'planks'
        >>> minerl_to_minedojo('oak_planks')
        'planks'
    """
    item_name = strip_minecraft_prefix(item_name)
    
    if item_name in MINERL_TO_MINEDOJO_ITEM_MAP:
        return MINERL_TO_MINEDOJO_ITEM_MAP[item_name]
    
    return item_name


def minedojo_to_minerl(item_name: str) -> str:
    """
    将 MineDojo 物品名称转换为 MineRL 物品名称
    
    Examples:
        >>> minedojo_to_minerl('planks')
        'oak_planks'
    """
    if item_name in MINEDOJO_TO_MINERL_ITEM_MAP:
        return MINEDOJO_TO_MINERL_ITEM_MAP[item_name]
    
    return item_name


def normalize_entity_name(entity_name: str) -> str:
    """
    规范化实体名称为小写标准格式
    
    Examples:
        >>> normalize_entity_name('Zombie')
        'zombie'
        >>> normalize_entity_name('minecraft:skeleton')
        'skeleton'
    """
    entity_name = strip_minecraft_prefix(entity_name)
    return entity_name.lower()


def get_entity_name_variants(entity_name: str) -> List[str]:
    """
    获取实体名称的所有可能变体
    
    用于在观察空间中查找击杀统计
    """
    normalized = normalize_entity_name(entity_name)
    
    if normalized in ENTITY_NAMES:
        return ENTITY_NAMES[normalized]
    
    # 生成常见变体
    return [
        normalized,
        normalized.capitalize(),
        normalized.title(),
        f"minecraft:{normalized}",
        normalized.upper(),
    ]


# =============================================================================
# 配置转换函数
# =============================================================================

def convert_item_config(item_config: dict, target_env: str) -> dict:
    """
    转换物品配置（用于 initial_inventory 和 reward_config）
    
    Args:
        item_config: 物品配置字典
        target_env: 目标环境 ('minerl' 或 'minedojo')
    """
    result = item_config.copy()
    
    item_key = 'type' if 'type' in result else 'entity' if 'entity' in result else 'name'
    
    if item_key not in result:
        return result
    
    item_name = result[item_key]
    
    if target_env == 'minedojo':
        result[item_key] = minerl_to_minedojo(item_name)
    elif target_env == 'minerl':
        result[item_key] = minedojo_to_minerl(item_name)
    
    return result


def convert_initial_inventory(inventory_list: list, target_env: str) -> list:
    """转换初始物品栏配置"""
    return [convert_item_config(item, target_env) for item in inventory_list]


def convert_reward_config(reward_list: list, target_env: str) -> list:
    """转换奖励配置"""
    return [convert_item_config(item, target_env) for item in reward_list]


# =============================================================================
# 配置标准化函数
# =============================================================================

def normalize_image_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    统一 image_size 和 resolution 配置
    
    支持格式：
    - image_size: [height, width] (MineDojo 格式)
    - resolution: (width, height) (MineRL 格式)
    
    统一为：image_size = (height, width)
    """
    if 'image_size' in config:
        image_size = config['image_size']
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            return tuple(image_size)
    
    if 'resolution' in config:
        resolution = config.pop('resolution')
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            width, height = resolution
            logger.info(f"🔄 配置转换: resolution=({width}, {height}) → image_size=({height}, {width})")
            config['image_size'] = (height, width)
            return (height, width)
    
    default_size = (160, 256)
    if 'image_size' not in config:
        config['image_size'] = default_size
    return default_size


def normalize_initial_inventory(config: Dict[str, Any]) -> None:
    """
    统一 initial_inventory 配置，确保使用 'type' 字段
    
    支持格式：
    - type: "item_name" (标准格式)
    - name: "item_name" (MineDojo 旧格式)
    - entity: "item_name" (非标准格式)
    """
    if 'initial_inventory' not in config:
        return
    
    inventory = config['initial_inventory']
    if not isinstance(inventory, list):
        return
    
    for item in inventory:
        if not isinstance(item, dict):
            continue
        
        if 'type' in item:
            continue
        
        if 'name' in item:
            item['type'] = item.pop('name')
            #logger.debug(f"🔄 initial_inventory: 'name' → 'type' ({item['type']})")
        elif 'entity' in item:
            item['type'] = item.pop('entity')
            #logger.debug(f"🔄 initial_inventory: 'entity' → 'type' ({item['type']})")


def normalize_reward_config(config: Dict[str, Any]) -> None:
    """
    统一 reward_config 配置
    
    MineDojo 格式 → MineRL 格式：
    - target_names/target_quantities/reward_weights → reward_config
    """
    if 'reward_config' in config:
        reward_config = config['reward_config']
        if isinstance(reward_config, list):
            for item in reward_config:
                if not isinstance(item, dict):
                    continue
                
                if 'type' in item and 'entity' not in item:
                    item['entity'] = item.pop('type')
                elif 'name' in item and 'entity' not in item:
                    item['entity'] = item.pop('name')
                
                if 'quantity' in item and 'amount' not in item:
                    item['amount'] = item.pop('quantity')
        return
    
    if 'target_names' in config:
        target_names = config.pop('target_names')
        target_quantities = config.pop('target_quantities', [1] * len(target_names))
        reward_weights = config.pop('reward_weights', {})
        
        reward_config = []
        for i, name in enumerate(target_names):
            reward_config.append({
                'entity': name,
                'amount': target_quantities[i] if i < len(target_quantities) else 1,
                'reward': reward_weights.get(name, 1.0) if isinstance(reward_weights, dict) else 1.0
            })
        
        config['reward_config'] = reward_config
        logger.info(f"🔄 配置转换: target_names → reward_config ({len(reward_config)} 项)")


def normalize_world_generation(config: Dict[str, Any]) -> None:
    """统一世界生成配置，移除 MineRL 专用配置"""
    if 'world_generator' in config:
        config.pop('world_generator')
        logger.info(f"移除配置: world_generator (MineRL 专用)")
    
    if 'generate_world_type' in config:
        config.pop('generate_world_type')


def normalize_spawn_and_time(config: Dict[str, Any]) -> None:
    """
    统一生成和时间配置
    
    移除嵌套格式，统一为扁平化配置：
    - start_time: int
    - allow_mob_spawn: bool
    """
    # 处理 time_condition (MineRL 嵌套格式)
    if 'time_condition' in config:
        time_cond = config.pop('time_condition')
        logger.info(f"移除配置: time_condition (使用扁平化配置)")
        
        if 'start_time' in time_cond and 'start_time' not in config:
            config['start_time'] = time_cond['start_time']
    
    # 移除时间流逝配置
    if 'allow_passage_of_time' in config:
        config.pop('allow_passage_of_time')
    
    if 'allow_time_passage' in config:
        config.pop('allow_time_passage')
    
    # 处理 spawning_condition (MineRL 嵌套格式)
    if 'spawning_condition' in config:
        spawn_cond = config.pop('spawning_condition')
        logger.info(f"移除配置: spawning_condition (使用扁平化配置)")
        
        if 'allow_spawning' in spawn_cond and 'allow_mob_spawn' not in config:
            config['allow_mob_spawn'] = spawn_cond['allow_spawning']
    
    # 统一为 allow_mob_spawn
    if 'allow_spawning' in config:
        if 'allow_mob_spawn' not in config:
            config['allow_mob_spawn'] = config['allow_spawning']
        config.pop('allow_spawning')


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
    """
    config = config.copy()
    
    normalize_image_size(config)
    normalize_initial_inventory(config)
    normalize_reward_config(config)
    normalize_world_generation(config)
    normalize_spawn_and_time(config)
    
    return config


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 80)
    print("环境桥接工具测试")
    print("=" * 80)
    
    # 测试物品名称转换
    print("\n### 物品名称转换测试")
    test_items = ["minecraft:oak_planks", "oak_planks", "stick", "oak_log"]
    for item in test_items:
        print(f"  {item:25} → MineDojo: {minerl_to_minedojo(item)}")
    
    # 测试实体名称变体
    print("\n### 实体名称变体测试")
    for entity in ["zombie", "skeleton", "chicken"]:
        variants = get_entity_name_variants(entity)
        print(f"  {entity}: {variants}")
    
    # 测试配置标准化
    print("\n### 配置标准化测试")
    test_config = {
        'resolution': (640, 320),
        'initial_inventory': [{'name': 'oak_planks', 'quantity': 2}],
        'time_condition': {'start_time': 13000, 'allow_passage_of_time': False},
        'spawning_condition': {'allow_spawning': True}
    }
    normalized = normalize_env_config(test_config)
    print(f"  标准化后: {normalized}")

