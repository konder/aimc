"""
MineRL Harvest Default 环境配置
使用 DefaultWorldGenerator（默认世界生成，无群系控制）
"""

import gym
import logging
from typing import List, Dict, Optional
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.mc import ALL_ITEMS
import minerl.herobraine.hero.handlers as handlers

logger = logging.getLogger(__name__)

MS_PER_STEP = 50


class MineRLHarvestWrapper(gym.Wrapper):
    """
    MineRL Harvest 环境的 Wrapper
    在 Python 端计算自定义奖励，支持动态配置
    """
    
    def __init__(self, env, reward_config: List[Dict], reward_rule: str = "any"):
        """
        Args:
            env: MineRL 环境实例
            reward_config: 奖励配置列表，支持两种格式：
                - Harvest/TechTree: [{"entity": "oak_log", "amount": 1, "reward": 100}, ...]
                - Combat: [{"event": "kill_entity", "entity_type": "chicken", "reward": 100}, ...]
            reward_rule: 完成规则 ("any", "all", "none")
        """
        super().__init__(env)
        self.reward_config = reward_config
        self.reward_rule = reward_rule
        
        # 检测任务类型并初始化追踪变量
        self.task_type = self._detect_task_type(reward_config)
        
        if self.task_type == "inventory":
            # Harvest/TechTree任务：追踪物品数量
            self.prev_inventory = {cfg["entity"]: 0 for cfg in reward_config}
            self.item_targets = {cfg["entity"]: cfg.get("amount", 1) for cfg in reward_config}
            self.item_rewards = {cfg["entity"]: cfg["reward"] for cfg in reward_config}
            self.item_completed = {cfg["entity"]: False for cfg in reward_config}
            logger.info(f"MineRLHarvestWrapper 初始化 (Inventory模式)")
            logger.info(f"  监控物品: {[cfg['entity'] for cfg in reward_config]}")
        else:
            # Combat任务：追踪事件
            self.event_targets = {cfg.get("entity_type", cfg.get("event")): cfg for cfg in reward_config}
            self.event_completed = {key: False for key in self.event_targets.keys()}
            self.prev_stats = {}
            logger.info(f"MineRLHarvestWrapper 初始化 (Event模式)")
            logger.info(f"  监控事件: {list(self.event_targets.keys())}")
        
        self.task_done = False
        logger.info(f"  完成规则: {reward_rule}")
    
    def _detect_task_type(self, reward_config: List[Dict]) -> str:
        """
        检测任务类型
        
        Returns:
            "inventory": 基于物品数量的任务 (harvest/techtree)
            "event": 基于事件的任务 (combat)
        """
        if not reward_config:
            return "inventory"
        
        # 检查第一个配置项
        first_cfg = reward_config[0]
        if "event" in first_cfg or "entity_type" in first_cfg:
            return "event"
        else:
            return "inventory"
    
    def reset(self, **kwargs):
        """重置环境和追踪状态"""
        obs = self.env.reset(**kwargs)
        
        # 根据任务类型重置追踪状态
        if self.task_type == "inventory":
            self.prev_inventory = {cfg["entity"]: 0 for cfg in self.reward_config}
            self.item_completed = {cfg["entity"]: False for cfg in self.reward_config}
        else:
            self.event_completed = {key: False for key in self.event_targets.keys()}
            self.prev_stats = {}
        
        self.task_done = False
        
        return obs
    
    def step(self, action):
        """执行动作并计算自定义奖励"""
        obs, reward, done, info = self.env.step(action)
        
        # 计算自定义奖励（忽略环境原始的 reward）
        custom_reward = self._calculate_reward(obs)
        
        # 检查任务是否完成
        task_done = self._check_task_done()
        
        return obs, custom_reward, done or task_done, info
    
    def _calculate_reward(self, obs) -> float:
        """
        根据 reward_config 计算增量奖励
        支持两种模式：inventory（物品数量）和event（事件触发）
        
        Returns:
            float: 本步的奖励值
        """
        if self.task_done:
            # 任务已完成，不再给予奖励
            return 0.0
        
        if self.task_type == "inventory":
            return self._calculate_inventory_reward(obs)
        else:
            return self._calculate_event_reward(obs)
    
    def _calculate_inventory_reward(self, obs) -> float:
        """
        基于物品数量计算奖励 (Harvest/TechTree任务)
        
        Returns:
            float: 本步的奖励值
        """
        current_inventory = obs.get('inventory', {})
        total_reward = 0.0
        
        # 遍历奖励配置，计算增量奖励
        for config in self.reward_config:
            entity = config["entity"]
            target_amount = config.get("amount", 1)
            reward_per_item = config["reward"]
            
            # 获取当前和之前的数量
            current_count = current_inventory.get(entity, 0)
            # 处理 numpy array
            if hasattr(current_count, 'item'):
                current_count = current_count.item()
            current_count = int(current_count)
            
            prev_count = self.prev_inventory.get(entity, 0)
            
            # 计算增量
            increment = current_count - prev_count
            
            if increment > 0:
                # 按比例给予奖励
                item_reward = (reward_per_item / target_amount) * increment
                total_reward += item_reward
                
                logger.info(f"💰 获得 {entity} x{increment}, 奖励: +{item_reward:.1f}")
                
                # 更新追踪
                self.prev_inventory[entity] = current_count
                
                # 检查是否完成目标
                if current_count >= target_amount and not self.item_completed[entity]:
                    self.item_completed[entity] = True
                    logger.info(f"✅ {entity} 达到目标 ({current_count}/{target_amount})")
        
        return total_reward
    
    def _calculate_event_reward(self, obs) -> float:
        """
        基于事件触发计算奖励 (Combat任务)
        
        Returns:
            float: 本步的奖励值
        """
        total_reward = 0.0
        
        # 遍历配置的事件
        for entity_type, cfg in self.event_targets.items():
            if self.event_completed[entity_type]:
                continue
            
            event_type = cfg.get("event", "kill_entity")
            reward_value = cfg["reward"]
            
            # 检查对应的观察空间
            if event_type == "kill_entity":
                # 检查kill_entity统计
                kill_stats = obs.get('kill_entity', {})
                current_kills = kill_stats.get(entity_type, 0)
                prev_kills = self.prev_stats.get(f"kill_{entity_type}", 0)
                
                if current_kills > prev_kills:
                    logger.info(f"💰 击杀 {entity_type}: {prev_kills} → {current_kills}")
                    total_reward += reward_value
                    self.event_completed[entity_type] = True
                    logger.info(f"✅ 击杀 {entity_type} 目标达成! 获得奖励: {reward_value}")
                
                self.prev_stats[f"kill_{entity_type}"] = current_kills
        
        return total_reward
    
    def _check_task_done(self) -> bool:
        """
        检查任务是否完成
        支持inventory和event两种模式
        
        Returns:
            bool: 任务是否完成
        """
        if self.task_done:
            return True
        
        # 根据任务类型选择完成检查的字典
        if self.task_type == "inventory":
            completed_dict = self.item_completed
        else:
            completed_dict = self.event_completed
        
        if self.reward_rule == "any":
            # 任意一个目标完成即可
            if any(completed_dict.values()):
                self.task_done = True
                completed_items = [k for k, v in completed_dict.items() if v]
                logger.info(f"任务完成！(reward_rule=any, 完成: {completed_items})")
                return True
        
        elif self.reward_rule == "all":
            # 所有目标都要完成
            if all(completed_dict.values()):
                self.task_done = True
                logger.info(f"任务完成！(reward_rule=all)")
                return True
        
        return False


class MineRLHarvestDefaultEnvSpec(HumanControlEnvSpec):
    """
    HarvestEnv Default 任务规范
    
    使用 DefaultWorldGenerator（默认世界生成，无群系控制）
    适用于需要树木、动物、植物等自然生成的任务
    """
    
    def __init__(
        self, 
        resolution=(640, 320), 
        image_size=None,  # 新增：支持 image_size 参数
        max_episode_steps=2000,
        time_condition: Optional[Dict] = None,
        spawning_condition: Optional[Dict] = None,
        initial_inventory: Optional[List[Dict]] = None,
        specified_biome: Optional[str] = None,  # 新增：指定 biome
        **kwargs
    ):
        """
        Args:
            resolution: 分辨率 (width, height) - 已弃用，使用 image_size
            image_size: 图像尺寸 (height, width) - 优先使用此参数
            max_episode_steps: 最大步数
            time_condition: 时间条件 (如 {"allow_passage_of_time": False, "start_time": 6000})
            spawning_condition: 生成条件 (如 {"allow_spawning": True})
            initial_inventory: 初始物品 (如 [{"type": "bucket", "quantity": 1}])
            specified_biome: 指定 biome (如 "desert", "jungle", "forest" 等)
        """
        # 🔄 统一 image_size 和 resolution
        # image_size 优先（格式: [height, width]）
        # resolution 作为备选（格式: (width, height)）
        if image_size is not None:
            if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                height, width = image_size
                resolution = (width, height)  # 转换为 MineRL 格式
                logger.info(f"🔄 MineRL 配置转换: image_size=[{height}, {width}] → resolution=({width}, {height})")
        else:
            logger.info(f"📺 MineRL 使用默认分辨率: resolution={resolution}")
        
        # 设置环境名称
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLHarvestDefaultEnv-v0'
        
        # 设置 episode 长度
        if 'max_episode_steps' not in kwargs:
            kwargs['max_episode_steps'] = max_episode_steps
        
        # 在父类初始化之前设置这些属性
        self.episode_len = kwargs['max_episode_steps']
        self.reward_threshold = 100.0
        
        # 保存配置参数
        self.time_condition = time_condition or {
            "allow_passage_of_time": False,
            "start_time": 6000  # 默认白天
        }
        self.spawning_condition = spawning_condition or {
            "allow_spawning": True  # 默认允许生成动物
        }
        self.initial_inventory = initial_inventory or []  # 默认空手
        self.specified_biome = specified_biome  # 保存 biome 配置
        
        # 调用父类初始化
        super().__init__(
            resolution=resolution,
            **kwargs
        )
    
    def create_observables(self) -> List[Handler]:
        """定义观察空间"""
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS),
            handlers.EquippedItemObservation(
                items=ALL_ITEMS,
                mainhand=True,
                offhand=True,
                armor=True,
                _default="air",
                _other="air",
            ),
            handlers.ObservationFromLifeStats(),
            handlers.ObservationFromCurrentLocation(),
            handlers.ObserveFromFullStats("use_item"),
            handlers.ObserveFromFullStats("drop"),
            handlers.ObserveFromFullStats("pickup"),
            handlers.ObserveFromFullStats("break_item"),
            handlers.ObserveFromFullStats("craft_item"),
            handlers.ObserveFromFullStats("mine_block"),
            handlers.ObserveFromFullStats("damage_dealt"),
            handlers.ObserveFromFullStats("entity_killed_by"),
            handlers.ObserveFromFullStats("kill_entity"),
            handlers.ObserveFromFullStats(None),
        ]
    
    def create_agent_handlers(self) -> List[Handler]:
        """定义 Agent handlers"""
        return []
    
    def create_rewardables(self) -> List[Handler]:
        """定义奖励 - 返回空，因为奖励由 Wrapper 计算"""
        return []
    
    def create_agent_start(self) -> List[Handler]:
        """定义初始位置和初始物品"""
        agent_start_handlers = super().create_agent_start()
        
        # 如果有初始物品配置，添加 SimpleInventoryAgentStart
        if self.initial_inventory:
            #logger.info(f"✓ 初始物品库存: {self.initial_inventory}")
            agent_start_handlers.append(
                handlers.SimpleInventoryAgentStart(self.initial_inventory)
            )
        
        return agent_start_handlers
    
    def create_server_world_generators(self) -> List[Handler]:
        """世界生成器 - 根据配置使用 BiomeGenerator 或 DefaultWorldGenerator"""
        
        # Biome ID 映射表 (Minecraft 1.16)
        BIOME_ID_MAP = {
            # 温暖
            "desert": 2, "savanna": 35, "badlands": 37,
            # 温带
            "plains": 1, "forest": 4, "flower_forest": 132,
            "birch_forest": 27, "dark_forest": 29, "swamp": 6,
            # 寒冷
            "taiga": 5, "snowy_taiga": 30, "snowy_tundra": 12,
            # 海洋
            "ocean": 0, "deep_ocean": 24, "frozen_ocean": 10,
            "warm_ocean": 44, "lukewarm_ocean": 45, "cold_ocean": 46,
            # 丛林
            "jungle": 21, "bamboo_jungle": 168,
            # 山地
            "mountains": 3, "snowy_mountains": 13, "wooded_mountains": 34,
            # 其他
            "beach": 16, "snowy_beach": 26, "mushroom_fields": 14,
            "river": 7, "frozen_river": 11,
        }
        
        if self.specified_biome:
            # 使用 BiomeGenerator 创建单一 biome 世界
            biome_id = BIOME_ID_MAP.get(self.specified_biome.lower())
            
            if biome_id is not None:
                logger.info(f"🌍 使用 BiomeGenerator: {self.specified_biome} (ID: {biome_id})")
                return [
                    handlers.BiomeGenerator(
                        biome_id=biome_id,
                        force_reset=True
                    )
                ]
            else:
                logger.warning(f"⚠️ 未知的 biome: {self.specified_biome}，使用默认世界")
        
        # 使用 DefaultWorldGenerator（默认世界）
        logger.info(f"🌍 使用 DefaultWorldGenerator（默认世界）")
        return [
            handlers.DefaultWorldGenerator(
                force_reset=True,
                generator_options=''  # 留空，使用默认世界
            )
        ]
    
    def create_server_quit_producers(self) -> List[Handler]:
        """服务器退出条件"""
        return [
            handlers.ServerQuitFromTimeUp(self.episode_len * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]
    
    def create_server_decorators(self) -> List[Handler]:
        """定义服务器装饰器"""
        return []
    
    def create_server_initial_conditions(self) -> List[Handler]:
        """初始条件"""
        allow_passage_of_time = self.time_condition.get("allow_passage_of_time", False)
        start_time = self.time_condition.get("start_time", 6000)
        allow_spawning = self.spawning_condition.get("allow_spawning", True)
        
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=allow_passage_of_time,
                start_time=start_time
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=allow_spawning
            )
        ]
    
    def determine_success_from_rewards(self, rewards: list) -> bool:
        """根据奖励判断任务是否成功"""
        return False
    
    def is_from_folder(self, folder: str) -> bool:
        """判断是否来自指定文件夹"""
        return folder == 'none'
    
    def get_docstring(self):
        """获取文档字符串"""
        return """
        MineRL Harvest Default Environment
        使用 DefaultWorldGenerator 生成默认世界（有树木、动物、植物）。
        适用于依赖自然生成的任务。
        """


def _minerl_harvest_default_env_entrypoint(
    reward_config: Optional[List[Dict]] = None,
    reward_rule: str = "any",
    time_condition: Optional[Dict] = None,
    spawning_condition: Optional[Dict] = None,
    initial_inventory: Optional[List[Dict]] = None,
    specified_biome: Optional[str] = None,
    max_episode_steps: int = 2000,
    **kwargs
):
    """
    自定义 entry point，创建环境并应用 Wrapper
    
    Args:
        reward_config: 奖励配置
        reward_rule: 完成规则
        time_condition: 时间条件
        spawning_condition: 生成条件
        initial_inventory: 初始物品配置
        specified_biome: 指定 biome
        max_episode_steps: 最大步数
    """
    # 创建 env_spec
    env_spec = MineRLHarvestDefaultEnvSpec(
        max_episode_steps=max_episode_steps,
        time_condition=time_condition,
        spawning_condition=spawning_condition,
        initial_inventory=initial_inventory,
        specified_biome=specified_biome,
        **kwargs
    )
    
    # 创建基础环境
    from minerl.env._singleagent import _SingleAgentEnv
    env = _SingleAgentEnv(env_spec=env_spec)
    
    # 如果有 reward_config，应用 Wrapper
    if reward_config:
        env = MineRLHarvestWrapper(env, reward_config, reward_rule)
    
    return env


def register_minerl_harvest_default_env():
    """注册 MineRL Harvest Default 环境"""
    try:
        gym.register(
            id='MineRLHarvestDefaultEnv-v0',
            entry_point='src.envs.minerl_harvest_default:_minerl_harvest_default_env_entrypoint'
        )
        logger.info("✓ MineRLHarvestDefaultEnv-v0 已注册（DefaultWorldGenerator）")
    except gym.error.Error:
        pass

