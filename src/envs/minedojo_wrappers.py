"""
MineDojo 环境定义和 Wrapper

主要功能:
1. 创建 MineDojo 环境并支持 Biome 定制
2. 观察空间转换: MineDojo → MineRL (只保留 POV)
3. 动作空间转换: MineRL → MineDojo
4. 直接使用 MineDojo 的奖励和任务结束判断
"""

import gym
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

# 导入环境桥接工具
from src.envs.env_bridge import (
    convert_initial_inventory,
    convert_reward_config,
    minerl_to_minedojo,
    normalize_env_config,
)

logger = logging.getLogger(__name__)


class MineDojoBiomeEnvSpec:
    """
    MineDojo Biome 环境规格
    
    支持三种世界生成类型:
    - "default": 默认世界
    - "flat": 平坦世界
    - "specified_biome": 指定 Biome 的世界
    """
    
    def __init__(
        self,
        task_id: str = "open-ended",
        image_size: Tuple[int, int] = (160, 256),
        specified_biome: Optional[str] = None,
        world_seed: Optional[str] = None,
        start_time: int = 6000,
        allow_mob_spawn: bool = False,
        spawn_in_village: bool = False,
        initial_inventory: Optional[list] = None,
        break_speed_multiplier: float = 1.0,
        **kwargs
    ):
        """
        Args:
            task_id: MineDojo 任务 ID
            image_size: 图像尺寸 (height, width) 或 (width, height) - 自动识别
            specified_biome: 指定的 Biome (如 "plains", "forest") - 仅 MineDojo 支持
            world_seed: 世界种子
            start_time: 起始时间 (0-24000, 6000=正午)
            allow_mob_spawn: 是否允许生物生成
            spawn_in_village: 是否在村庄生成
            initial_inventory: 初始物品列表 (使用 'type' 字段)
            break_speed_multiplier: 破坏速度倍数
            **kwargs: 其他参数（会被标准化）
        """
        # 🔄 标准化所有配置
        all_config = {
            'task_id': task_id,
            'image_size': image_size,
            'specified_biome': specified_biome,
            'world_seed': world_seed,
            'start_time': start_time,
            'allow_mob_spawn': allow_mob_spawn,
            'spawn_in_village': spawn_in_village,
            'initial_inventory': initial_inventory or [],
            'break_speed_multiplier': break_speed_multiplier,
            **kwargs
        }
        
        # 标准化配置
        normalized_config = normalize_env_config(all_config)
        
        # 提取标准化后的参数
        self.task_id = normalized_config.pop('task_id', task_id)
        self.image_size = normalized_config.pop('image_size', image_size)
        self.specified_biome = normalized_config.pop('specified_biome', None)
        self.world_seed = normalized_config.pop('world_seed', None) or "minedojo_biome"
        self.start_time = normalized_config.pop('start_time', 6000)
        self.allow_mob_spawn = normalized_config.pop('allow_mob_spawn', False)
        self.spawn_in_village = normalized_config.pop('spawn_in_village', False)
        self.initial_inventory = normalized_config.pop('initial_inventory', [])
        self.break_speed_multiplier = normalized_config.pop('break_speed_multiplier', 1.0)
        
        # 根据 specified_biome 自动设置 generate_world_type
        if self.specified_biome:
            self.generate_world_type = "specified_biome"
            logger.info(f"🌍 自动设置: generate_world_type='specified_biome' (biome={self.specified_biome})")
        else:
            self.generate_world_type = "default"
            logger.debug(f"🌍 自动设置: generate_world_type='default'")
        
        # 剩余参数
        self.kwargs = normalized_config
    
    def create_env(self):
        """创建 MineDojo 环境"""
        import minedojo
        from minedojo.sim import InventoryItem
        
        env_config = {
            "image_size": self.image_size,
            "world_seed": self.world_seed,
            "break_speed_multiplier": self.break_speed_multiplier,
        }
        
        # 只有 open-ended 任务支持这些参数
        if self.task_id == "open-ended":
            env_config["start_time"] = self.start_time
            env_config["allow_time_passage"] = False  # 时间默认不流逝
            env_config["allow_mob_spawn"] = self.allow_mob_spawn
            
            # 世界生成类型（只有 open-ended 支持）
            if self.generate_world_type:
                env_config["generate_world_type"] = self.generate_world_type
            
            # 指定 Biome（只有 open-ended 支持）
            if self.generate_world_type == "specified_biome" and self.specified_biome:
                env_config["specified_biome"] = self.specified_biome
                logger.info(f"🌍 MineDojo 指定 Biome: {self.specified_biome}")
        
        # 村庄生成
        if self.spawn_in_village:
            env_config["spawn_in_village"] = True
        
        # 初始物品 - 转换为 InventoryItem 对象
        if self.initial_inventory:
            # 🔄 自动转换 MineRL 物品名称为 MineDojo 格式
            #logger.info(f"📦 处理初始物品栏 ({len(self.initial_inventory)} 项)...")
            converted_inventory = convert_initial_inventory(
                self.initial_inventory, 
                target_env='minedojo'
            )
            
            inventory_items = []
            for idx, item in enumerate(converted_inventory):
                if isinstance(item, dict):
                    # 从 dict 创建 InventoryItem
                    # 支持两种格式：'name' (MineDojo) 或 'type' (MineRL)
                    item_name = item.get('name') or item.get('type')
                    if not item_name:
                        logger.warning(f"⚠️ 跳过无效物品（缺少 name/type）: {item}")
                        continue
                    
                    # 记录转换信息
                    original_item = self.initial_inventory[idx]
                    original_name = original_item.get('name') or original_item.get('type')
                    if original_name != item_name:
                        logger.info(f"  🔄 物品名称转换: {original_name} → {item_name}")
                    
                    inventory_items.append(
                        InventoryItem(
                            slot=item.get('slot', idx),  # 如果没有指定 slot，使用索引
                            name=item_name,
                            variant=item.get('variant'),
                            quantity=item.get('quantity', 1)
                        )
                    )
                else:
                    # 已经是 InventoryItem 对象
                    inventory_items.append(item)
            
            if inventory_items:
                env_config["initial_inventory"] = inventory_items
                logger.info(f"✓ 初始物品: {len(inventory_items)} 项")
                for item in inventory_items:
                    logger.info(f"  - slot {item.slot}: {item.name} x{item.quantity}")
        
        # ⚠️ 关键：设置 cam_interval 为 0.01，获得连续相机控制
        # 这会使 MineDojo 的 camera bins 从 25 增加到 36001
        # 从而实现与 MineRL 相同的高精度相机控制
        cam_interval = self.kwargs.pop('cam_interval', 0.01)  # 从 kwargs 中移除，避免重复
        
        # 🔄 处理 reward_config（MineRL 格式）→ target_names + reward_weights（MineDojo 格式）
        if 'reward_config' in self.kwargs:
            logger.info(f"🎯 处理奖励配置...")
            reward_config = self.kwargs.pop('reward_config')
            
            # 转换 MineRL 的 reward_config 列表为 MineDojo 格式
            if isinstance(reward_config, list):
                # 转换物品名称
                converted_rewards = convert_reward_config(reward_config, target_env='minedojo')
                
                # 提取 target_names, target_quantities, reward_weights
                target_names = []
                target_quantities = []
                reward_weights_list = []
                
                for item in converted_rewards:
                    item_name = item.get('entity') or item.get('type') or item.get('name')
                    if item_name:
                        target_names.append(item_name)
                        target_quantities.append(item.get('amount', 1))
                        reward_weights_list.append(item.get('reward', 1.0))
                        
                        # 记录转换信息
                        original_item = reward_config[converted_rewards.index(item)]
                        original_name = original_item.get('entity') or original_item.get('type') or original_item.get('name')
                        if original_name != item_name:
                            logger.info(f"  🔄 奖励物品转换: {original_name} → {item_name}")
                
                # 设置 MineDojo 格式的参数
                self.kwargs['target_names'] = target_names
                self.kwargs['target_quantities'] = target_quantities
                
                # reward_weights 转换为字典
                reward_weights_dict = {
                    name: weight 
                    for name, weight in zip(target_names, reward_weights_list)
                }
                self.kwargs['reward_weights'] = reward_weights_dict
                
                logger.info(f"✓ 奖励配置转换完成:")
                logger.info(f"  target_names: {target_names}")
                logger.info(f"  target_quantities: {target_quantities}")
                logger.info(f"  reward_weights: {reward_weights_dict}")
        
        # 处理已有的 reward_weights（harvest 任务需要）
        elif 'reward_weights' in self.kwargs:
            reward_weights = self.kwargs.pop('reward_weights')
            target_names = self.kwargs.get('target_names')
            
            # 如果 target_names 存在，转换物品名称
            if target_names:
                converted_names = [minerl_to_minedojo(name) for name in target_names]
                if converted_names != target_names:
                    logger.info(f"🔄 目标物品名称转换:")
                    for orig, conv in zip(target_names, converted_names):
                        if orig != conv:
                            logger.info(f"  {orig} → {conv}")
                    self.kwargs['target_names'] = converted_names
            
            # 如果是列表，转换为字典
            if isinstance(reward_weights, list) and target_names:
                if len(reward_weights) == len(target_names):
                    reward_weights_dict = {
                        name: weight 
                        for name, weight in zip(converted_names if target_names else target_names, reward_weights)
                    }
                    self.kwargs['reward_weights'] = reward_weights_dict
                    logger.info(f"🎯 奖励权重转换: {reward_weights} → {reward_weights_dict}")
                else:
                    logger.warning(f"⚠️ reward_weights 长度与 target_names 不匹配，忽略")
            else:
                # 已经是字典或其他格式，直接使用
                self.kwargs['reward_weights'] = reward_weights
        
        # 其他参数
        env_config.update(self.kwargs)
        
        #logger.info(f"创建 MineDojo 环境:")
        #logger.info(f"  task_id: {self.task_id}")
        logger.info(f"  generate_world_type: {self.generate_world_type}")
        if self.specified_biome:
            logger.info(f"  specified_biome: {self.specified_biome}")
        #logger.info(f"  world_seed: {self.world_seed}")
        logger.info(f"  cam_interval: {cam_interval} (连续相机控制)")
        
        env = minedojo.make(
            task_id=self.task_id,
            cam_interval=cam_interval,
            **env_config
        )
        #logger.info(f"✓ MineDojo 环境创建完成")
        #logger.info(f"  Camera bins: {env.action_space.nvec[3]} (pitch), {env.action_space.nvec[4]} (yaw)")
        
        return env


class MineDojoBiomeWrapper(gym.Wrapper):
    """
    MineDojo → MineRL 兼容 Wrapper
    
    主要功能:
    1. 观察空间转换: MineDojo → MineRL (只保留 POV)
    2. 动作空间转换: MineRL → MineDojo
    3. 直接使用 MineDojo 的奖励和任务结束判断
    """
    
    def __init__(self, env):
        """
        Args:
            env: MineDojo 环境（已通过 cam_interval=0.01 配置）
        """
        super().__init__(env)
        
        # MineDojo 的动作空间（cam_interval=0.01）：
        # MultiDiscrete([3, 3, 4, 36001, 36001, 8, 244, 36])
        # Index 3, 4 是 camera，范围是 [0, 36000]，共 36001 个 bins
        # 
        # 转换公式（MineDojo 内部）:
        # continuous_angle = discrete_bin * cam_interval + (-180)
        # 
        # 因此:
        # - Bin 0 = -180°
        # - Bin 18000 = 0° (中心)
        # - Bin 36000 = +180°
        self.n_camera_bins = 36001  # 与 MineRL 相同的高精度
        self.camera_center = (self.n_camera_bins - 1) // 2  # 18000
        self.cam_interval = 0.01  # 与 MineDojo 的 cam_interval 一致
        
        
        #logger.info("✓ MineDojoBiomeWrapper 初始化完成")
        #logger.info(f"  相机 bins: {self.n_camera_bins} (与 MineRL 相同)")
        #logger.info(f"  相机中心: {self.camera_center}")
        #logger.info(f"  相机精度: {self.cam_interval}° per bin (连续控制)")
        #logger.info(f"  覆盖范围: ±180° (完整范围)")
        #logger.info(f"  Inventory Toggle: 启用")
    
    def reset(self):
        """
        重置环境
        
        Returns:
            obs: MineRL 格式的观察 (只包含 POV)
        """
        minedojo_obs = self.env.reset()
        minerl_obs = self._convert_obs_to_minerl(minedojo_obs)
        return minerl_obs
    
    def step(self, minerl_action: Dict):
        """
        执行动作
        
        Args:
            minerl_action: MineRL 格式的动作 (Dict)
        
        Returns:
            obs: MineRL 格式的观察
            reward: 奖励 (直接使用 MineDojo 的)
            done: 是否结束 (直接使用 MineDojo 的)
            info: 额外信息
        """
        # 转换动作: MineRL → MineDojo (MultiDiscrete 数组)
        minedojo_action = self._convert_action_to_minedojo(minerl_action)
        
        # 执行动作
        minedojo_obs, reward, done, info = self.env.step(minedojo_action)
        
        # 转换观察: MineDojo → MineRL
        minerl_obs = self._convert_obs_to_minerl(minedojo_obs)
        
        return minerl_obs, reward, done, info
    
    def _convert_obs_to_minerl(self, minedojo_obs: Dict) -> Dict:
        """
        转换观察空间: MineDojo → MineRL
        
        MineDojo 观察空间 (参考文档):
        - rgb: (3, H, W) uint8
        - inventory: dict with name, quantity, etc.
        - equipment: dict
        - location_stats: dict (包括 biome_id)
        - ... 等更多
        
        MineRL 观察空间:
        - pov: (H, W, 3) uint8  # 注意维度顺序不同！
        
        Args:
            minedojo_obs: MineDojo 观察 (Dict)
        
        Returns:
            minerl_obs: MineRL 观察 (Dict, 只包含 pov)
        """
        # MineDojo: (3, H, W) → MineRL: (H, W, 3)
        rgb = minedojo_obs["rgb"]  # (3, H, W)
        pov = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
        
        minerl_obs = {
            "pov": pov
        }
        
        return minerl_obs
    
    def _convert_action_to_minedojo(self, minerl_action: Dict) -> np.ndarray:
        """
        转换动作空间: MineRL → MineDojo
        
        MineRL 动作空间 (Dict):
        - forward, back, left, right: Discrete(2)
        - jump, sneak, sprint: Discrete(2)
        - camera: Box([-180, 180], shape=(2,))
        - attack, use, drop: Discrete(2)
        - swapHands, pickItem: Discrete(2)
        - hotbar.1 - hotbar.9: Discrete(2)
        - inventory: Discrete(2) - 需要特殊处理 (toggle 机制, E键)
        - swapHands: Discrete(2) - 交换主手和副手物品 (F键)
        - pickItem: Discrete(2) - 中键点击复制方块 (鼠标中键)
        
        MineDojo 动作空间 (MultiDiscrete):
        [0]: Forward/Back (0: noop, 1: forward, 2: back)
        [1]: Left/Right (0: noop, 1: left, 2: right)
        [2]: Jump/Sneak/Sprint (0: noop, 1: jump, 2: sneak, 3: sprint)
        [3]: Camera Pitch (0: -180°, 18000: 0°, 36000: +180°)
        [4]: Camera Yaw (0: -180°, 18000: 0°, 36000: +180°)
        [5]: Functional (0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy, 8: inventory, 9: swapHands, 10: pickItem)
        [6]: Craft Argument (0-243)
        [7]: Inventory/Equip/Place/Destroy Argument (0-35)
        
        参考: https://docs.minedojo.org/sections/core_api/action_space.html
        
        Args:
            minerl_action: MineRL 动作 (Dict)
        
        Returns:
            minedojo_action: MineDojo 动作 (np.ndarray, shape=(8,))
        """
        # 初始化 MineDojo 动作数组
        minedojo_action = np.zeros(8, dtype=np.int32)
        
        # 1. Forward/Back (index 0)
        if minerl_action.get('forward', 0):
            minedojo_action[0] = 1
        elif minerl_action.get('back', 0):
            minedojo_action[0] = 2
        
        # 2. Left/Right (index 1)
        if minerl_action.get('left', 0):
            minedojo_action[1] = 1
        elif minerl_action.get('right', 0):
            minedojo_action[1] = 2
        
        # 3. Jump/Sneak/Sprint (index 2)
        if minerl_action.get('jump', 0):
            minedojo_action[2] = 1
        elif minerl_action.get('sneak', 0):
            minedojo_action[2] = 2
        elif minerl_action.get('sprint', 0):
            minedojo_action[2] = 3
        
        # 4. Camera (index 3, 4)
        # ⚠️ 重要：MineRL 的 camera 是相对移动（delta），不是绝对角度！
        # 
        # MineRL 语义:
        # - camera=[pitch_delta, yaw_delta]
        # - 每帧的 camera 值是相对于当前视角的增量
        # - 例如: camera=[1.0, 0.0] 表示向下看 1°
        # 
        # MineDojo 语义:
        # - MineDojo 的 camera bins 也是相对移动
        # - center bin (18000) = 不移动
        # - bin > 18000 = 正向移动
        # - bin < 18000 = 负向移动
        # 
        # 转换公式:
        # - delta_angle = 0° → bin = 18000 (center, 不移动)
        # - delta_angle = +1° → bin = 18100 (向正方向移动 1°)
        # - delta_angle = -1° → bin = 17900 (向负方向移动 1°)
        
        camera_raw = minerl_action.get('camera', np.array([0.0, 0.0]))
        
        # 确保 camera 是 numpy 数组
        if isinstance(camera_raw, (list, tuple)):
            camera = np.array(camera_raw)
        elif not isinstance(camera_raw, np.ndarray):
            camera = np.array([0.0, 0.0])
        else:
            camera = camera_raw
        
        # 展平嵌套数组
        camera = np.asarray(camera).flatten()
        
        # 确保是 2D 向量
        if camera.size == 0:
            camera = np.array([0.0, 0.0])
        elif camera.size == 1:
            camera = np.array([float(camera[0]), 0.0])
        elif camera.size >= 2:
            camera = np.array([float(camera[0]), float(camera[1])])
        else:
            camera = np.array([0.0, 0.0])
        
        pitch_delta = float(camera[0])  # 俯仰角增量（上下）
        yaw_delta = float(camera[1])    # 偏航角增量（左右）
        
        # 转换为离散值（相对移动）
        # cam_interval=0.01 → 36001 bins (0 to 36000)
        # center = 18000 → delta = 0° (不移动)
        # 
        # 公式: bin = center + delta / cam_interval
        pitch_discrete = int(round(self.camera_center + pitch_delta / self.cam_interval))
        yaw_discrete = int(round(self.camera_center + yaw_delta / self.cam_interval))
        
        # 裁剪到有效范围
        pitch_final = int(np.clip(pitch_discrete, 0, self.n_camera_bins - 1))
        yaw_final = int(np.clip(yaw_discrete, 0, self.n_camera_bins - 1))
        
        minedojo_action[3] = pitch_final
        minedojo_action[4] = yaw_final
        
        # 5. Functional (index 5)
        # 0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy, 8: inventory, 9: swapHands
        # 优先级: inventory > swapHands > attack > use > drop
        
        # Inventory 处理
        # 直接传递 MineRL 的 inventory 值，不做任何状态管理
        # Java 层负责所有的 toggle 逻辑
        # 
        # MineRL 语义:
        # - inventory=0: 不改变状态
        # - inventory=1: toggle（切换状态）
        #
        # MineDojo action[5] 映射:
        # - 0: noop
        # - 8: inventory
        
        current_inventory_input = minerl_action.get('inventory', 0)
        if current_inventory_input == 1:
            minedojo_action[5] = 8  # 发送 inventory 命令
        else:
            minedojo_action[5] = 0  # noop
        
        # 其他功能动作（只在 inventory 未触发时执行）
        if minedojo_action[5] == 0:
            if minerl_action.get('swapHands', 0):
                minedojo_action[5] = 9  # swapHands
            elif minerl_action.get('pickItem', 0):
                minedojo_action[5] = 10  # pickItem
            elif minerl_action.get('attack', 0):
                minedojo_action[5] = 3  # attack
            elif minerl_action.get('use', 0):
                minedojo_action[5] = 1  # use
            elif minerl_action.get('drop', 0):
                minedojo_action[5] = 2  # drop
        
        # 6. Craft Argument (index 6) - VPT 不使用
        minedojo_action[6] = 0
        
        # 7. Inventory/Equip/Place/Destroy Argument (index 7)
        # 检查 hotbar.1 到 hotbar.9 (对应 inventory slot 0-8)
        for i in range(1, 10):
            if minerl_action.get(f'hotbar.{i}', 0):
                minedojo_action[7] = i - 1  # hotbar.1 → slot 0
                break
        
        return minedojo_action


def register_minedojo_biome_env():
    """注册 MineDojo Biome 环境"""
    try:
        gym.register(
            id='MineDojoHarvestEnv-v0',
            entry_point='src.envs.minedojo_wrappers:_minedojo_harvest_env_entrypoint',
            max_episode_steps=2000,
        )
        logger.info("✓ MineDojoHarvestEnv-v0 已注册")
    except gym.error.Error:
        # 已注册
        pass


def _minedojo_harvest_env_entrypoint(
    generate_world_type: str = "default",
    specified_biome: Optional[str] = None,
    world_seed: Optional[str] = None,
    task_id: str = "open-ended",
    image_size: Tuple[int, int] = (160, 256),
    start_time: int = 6000,
    allow_time_passage: bool = False,
    allow_mob_spawn: bool = False,
    spawn_in_village: bool = False,
    initial_inventory: Optional[list] = None,
    max_episode_steps: int = 2000,
    **kwargs
):
    """
    MineDojo Harvest 环境入口
    
    Args:
        generate_world_type: 世界类型 ("default", "flat", "specified_biome")
        specified_biome: 指定 Biome (如 "plains", "extreme_hills", "desert")
        world_seed: 世界种子
        task_id: MineDojo 任务 ID
        image_size: 图像尺寸
        start_time: 起始时间
        allow_time_passage: 是否允许时间流逝
        allow_mob_spawn: 是否允许怪物生成
        spawn_in_village: 是否在村庄生成
        initial_inventory: 初始物品
        max_episode_steps: 最大步数
    
    Returns:
        env: 包装后的 MineDojo 环境
    """
    # 创建环境规格
    env_spec = MineDojoBiomeEnvSpec(
        task_id=task_id,
        image_size=image_size,
        generate_world_type=generate_world_type,
        specified_biome=specified_biome,
        world_seed=world_seed,
        start_time=start_time,
        allow_time_passage=allow_time_passage,
        allow_mob_spawn=allow_mob_spawn,
        spawn_in_village=spawn_in_village,
        initial_inventory=initial_inventory,
        **kwargs
    )
    
    # 创建基础 MineDojo 环境
    base_env = env_spec.create_env()
    
    # 应用 Wrapper (观察空间和动作空间转换)
    env = MineDojoBiomeWrapper(base_env)
    
    logger.info("✓ MineDojo Harvest 环境创建完成")
    
    return env

