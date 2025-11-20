"""
物品名称映射器
Item Name Mapper

功能：在 MineRL 和 MineDojo 环境之间转换物品名称

MineRL 使用格式：
  - 带前缀：'minecraft:oak_planks', 'minecraft:stick'
  - 不带前缀：'oak_planks', 'stick'

MineDojo 使用格式：
  - 不带前缀：'planks', 'stick', 'log'
  - 使用通用名称（如 'planks' 而不是 'oak_planks'）

参考：
  - MineRL: https://github.com/minerllabs/minerl/blob/cdeae668c2f334e3c9117adf651b5a94436b45f8/minerl/herobraine/hero/mc.py#L535
  - MineDojo: https://github.com/MineDojo/MineDojo/blob/2731bc27394269643b43828d9db8ab3a364601f0/minedojo/sim/mc_meta/mc.py#L4
"""

# MineRL → MineDojo 物品名称映射
# 格式: {minerl_name: minedojo_name}
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
    "log": "log",  # 通用
    
    # 木棍 (Stick)
    "stick": "stick",
    
    # 工具 - 木制 (Wooden Tools)
    "wooden_pickaxe": "wooden_pickaxe",
    "wooden_axe": "wooden_axe",
    "wooden_shovel": "wooden_shovel",
    "wooden_sword": "wooden_sword",
    "wooden_hoe": "wooden_hoe",
    
    # 工具 - 石制 (Stone Tools)
    "stone_pickaxe": "stone_pickaxe",
    "stone_axe": "stone_axe",
    "stone_shovel": "stone_shovel",
    "stone_sword": "stone_sword",
    "stone_hoe": "stone_hoe",
    
    # 工具 - 铁制 (Iron Tools)
    "iron_pickaxe": "iron_pickaxe",
    "iron_axe": "iron_axe",
    "iron_shovel": "iron_shovel",
    "iron_sword": "iron_sword",
    "iron_hoe": "iron_hoe",
    
    # 工具 - 金制 (Golden Tools)
    "golden_pickaxe": "golden_pickaxe",
    "golden_axe": "golden_axe",
    "golden_shovel": "golden_shovel",
    "golden_sword": "golden_sword",
    "golden_hoe": "golden_hoe",
    
    # 工具 - 钻石 (Diamond Tools)
    "diamond_pickaxe": "diamond_pickaxe",
    "diamond_axe": "diamond_axe",
    "diamond_shovel": "diamond_shovel",
    "diamond_sword": "diamond_sword",
    "diamond_hoe": "diamond_hoe",
    
    # 方块 - 基础 (Basic Blocks)
    "dirt": "dirt",
    "cobblestone": "cobblestone",
    "stone": "stone",
    "sand": "sand",
    "gravel": "gravel",
    "clay": "clay",
    
    # 矿石 (Ores)
    "coal_ore": "coal_ore",
    "iron_ore": "iron_ore",
    "gold_ore": "gold_ore",
    "diamond_ore": "diamond_ore",
    "redstone_ore": "redstone_ore",
    "lapis_ore": "lapis_ore",
    "emerald_ore": "emerald_ore",
    
    # 矿物 (Minerals)
    "coal": "coal",
    "iron_ingot": "iron_ingot",
    "gold_ingot": "gold_ingot",
    "diamond": "diamond",
    "redstone": "redstone",
    "lapis_lazuli": "dye",  # MineDojo 使用 'dye'
    "emerald": "emerald",
    
    # 食物 (Food)
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
    
    # 动物掉落物 (Animal Drops)
    "leather": "leather",
    "feather": "feather",
    "wool": "wool",
    "white_wool": "wool",
    
    # 容器 (Containers)
    "bucket": "bucket",
    "water_bucket": "water_bucket",
    "lava_bucket": "lava_bucket",
    "milk_bucket": "milk_bucket",
    
    # 合成物品 (Crafted Items)
    "crafting_table": "crafting_table",
    "furnace": "furnace",
    "chest": "chest",
    "torch": "torch",
    
    # 农作物 (Crops)
    "wheat": "wheat",
    "wheat_seeds": "wheat_seeds",
    "carrot": "carrot",
    "potato": "potato",
    "beetroot": "beetroot",
    "beetroot_seeds": "beetroot_seeds",
    
    # 植物 (Plants)
    "sapling": "sapling",
    "oak_sapling": "sapling",
    "spruce_sapling": "sapling",
    "birch_sapling": "sapling",
    "jungle_sapling": "sapling",
    "acacia_sapling": "sapling",
    "dark_oak_sapling": "sapling",
    
    # 花朵 (Flowers)
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
    
    # 蘑菇 (Mushrooms)
    "brown_mushroom": "brown_mushroom",
    "red_mushroom": "red_mushroom",
    
    # 其他 (Others)
    "snowball": "snowball",
    "snow": "snow",
    "ice": "ice",
    "sugar_cane": "reeds",  # MineDojo 使用 'reeds'
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
    "dye": "lapis_lazuli",  # 默认映射到青金石
    "reeds": "sugar_cane",
    "yellow_flower": "dandelion",
    "red_flower": "poppy",
    "wool": "white_wool",
}


def strip_minecraft_prefix(item_name: str) -> str:
    """
    移除 'minecraft:' 前缀
    
    Args:
        item_name: 物品名称，可能带有 'minecraft:' 前缀
    
    Returns:
        不带前缀的物品名称
    
    Examples:
        >>> strip_minecraft_prefix('minecraft:oak_planks')
        'oak_planks'
        >>> strip_minecraft_prefix('oak_planks')
        'oak_planks'
    """
    if item_name.startswith("minecraft:"):
        return item_name[len("minecraft:"):]
    return item_name


def minerl_to_minedojo(item_name: str) -> str:
    """
    将 MineRL 物品名称转换为 MineDojo 物品名称
    
    Args:
        item_name: MineRL 物品名称（可能带有 'minecraft:' 前缀）
    
    Returns:
        MineDojo 物品名称
    
    Examples:
        >>> minerl_to_minedojo('minecraft:oak_planks')
        'planks'
        >>> minerl_to_minedojo('oak_planks')
        'planks'
        >>> minerl_to_minedojo('stick')
        'stick'
    
    Raises:
        ValueError: 如果物品名称无法映射
    """
    # 移除前缀
    item_name = strip_minecraft_prefix(item_name)
    
    # 查找映射
    if item_name in MINERL_TO_MINEDOJO_ITEM_MAP:
        return MINERL_TO_MINEDOJO_ITEM_MAP[item_name]
    
    # 如果没有映射，尝试直接使用（可能已经是 MineDojo 格式）
    # 或者物品名称在两个环境中相同
    return item_name


def minedojo_to_minerl(item_name: str) -> str:
    """
    将 MineDojo 物品名称转换为 MineRL 物品名称
    
    Args:
        item_name: MineDojo 物品名称
    
    Returns:
        MineRL 物品名称（不带 'minecraft:' 前缀）
    
    Examples:
        >>> minedojo_to_minerl('planks')
        'oak_planks'
        >>> minedojo_to_minerl('stick')
        'stick'
    
    Raises:
        ValueError: 如果物品名称无法映射
    """
    # 查找映射
    if item_name in MINEDOJO_TO_MINERL_ITEM_MAP:
        return MINEDOJO_TO_MINERL_ITEM_MAP[item_name]
    
    # 如果没有映射，尝试直接使用
    return item_name


def convert_item_config(item_config: dict, target_env: str) -> dict:
    """
    转换物品配置（用于 initial_inventory 和 reward_config）
    
    Args:
        item_config: 物品配置字典，包含 'type' 或 'entity' 和 'quantity' 或 'amount'
        target_env: 目标环境 ('minerl' 或 'minedojo')
    
    Returns:
        转换后的物品配置
    
    Examples:
        >>> convert_item_config({'type': 'oak_planks', 'quantity': 2}, 'minedojo')
        {'type': 'planks', 'quantity': 2}
        >>> convert_item_config({'entity': 'planks', 'amount': 1}, 'minerl')
        {'entity': 'oak_planks', 'amount': 1}
    """
    result = item_config.copy()
    
    # 确定物品名称字段
    item_key = 'type' if 'type' in result else 'entity' if 'entity' in result else 'name'
    
    if item_key not in result:
        return result
    
    item_name = result[item_key]
    
    # 转换物品名称
    if target_env == 'minedojo':
        result[item_key] = minerl_to_minedojo(item_name)
    elif target_env == 'minerl':
        result[item_key] = minedojo_to_minerl(item_name)
    else:
        raise ValueError(f"Unknown target environment: {target_env}")
    
    return result


def convert_initial_inventory(inventory_list: list, target_env: str) -> list:
    """
    转换初始物品栏配置
    
    Args:
        inventory_list: 物品列表，每个元素是包含 'type' 和 'quantity' 的字典
        target_env: 目标环境 ('minerl' 或 'minedojo')
    
    Returns:
        转换后的物品列表
    
    Examples:
        >>> convert_initial_inventory([{'type': 'oak_planks', 'quantity': 2}], 'minedojo')
        [{'type': 'planks', 'quantity': 2}]
    """
    return [convert_item_config(item, target_env) for item in inventory_list]


def convert_reward_config(reward_list: list, target_env: str) -> list:
    """
    转换奖励配置
    
    Args:
        reward_list: 奖励列表，每个元素是包含 'entity' 和 'amount' 的字典
        target_env: 目标环境 ('minerl' 或 'minedojo')
    
    Returns:
        转换后的奖励列表
    
    Examples:
        >>> convert_reward_config([{'entity': 'oak_planks', 'amount': 1, 'reward': 100}], 'minedojo')
        [{'entity': 'planks', 'amount': 1, 'reward': 100}]
    """
    return [convert_item_config(item, target_env) for item in reward_list]


# 测试代码
if __name__ == "__main__":
    # 测试 MineRL → MineDojo
    print("=== MineRL → MineDojo ===")
    test_items = [
        "minecraft:oak_planks",
        "oak_planks",
        "stick",
        "bucket",
        "milk_bucket",
        "oak_log",
        "wooden_pickaxe",
    ]
    for item in test_items:
        print(f"{item:30} → {minerl_to_minedojo(item)}")
    
    print("\n=== MineDojo → MineRL ===")
    test_items = [
        "planks",
        "stick",
        "log",
        "bucket",
        "milk_bucket",
    ]
    for item in test_items:
        print(f"{item:30} → {minedojo_to_minerl(item)}")
    
    print("\n=== 配置转换测试 ===")
    # 测试 initial_inventory 转换
    inventory = [
        {'type': 'oak_planks', 'quantity': 2},
        {'type': 'stick', 'quantity': 4},
    ]
    print("Initial Inventory (MineRL → MineDojo):")
    print(convert_initial_inventory(inventory, 'minedojo'))
    
    # 测试 reward_config 转换
    rewards = [
        {'entity': 'oak_planks', 'amount': 1, 'reward': 100},
        {'entity': 'stick', 'amount': 4, 'reward': 50},
    ]
    print("\nReward Config (MineRL → MineDojo):")
    print(convert_reward_config(rewards, 'minedojo'))

