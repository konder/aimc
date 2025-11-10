"""
HarvestEnv - 自定义获取木头任务
参考 MineRL ObtainDiamondShovel 实现，使用 Wrapper 模式自定义奖励
"""

import gym
from typing import List
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero.mc import ALL_ITEMS
from minerl.env import _singleagent

MS_PER_STEP = 50


class MineRLHarvestWrapper(gym.Wrapper):
    """
    自定义奖励 Wrapper
    监控库存中木头的变化，每获得一个木头给予奖励
    """
    
    def __init__(self, env):
        super().__init__(env)
        # 木头类型列表（与库存观察中的名称一致）
        self.log_types = [
            'oak_log', 'birch_log', 'spruce_log', 
            'jungle_log', 'acacia_log', 'dark_oak_log'
        ]
        self.reward_per_log = 100.0
        self.target_logs = 1  # 获得1个木头后结束
        self.total_logs = 0
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        self.episode_over = False
    
    def step(self, action: dict):
        """重写 step 方法，添加自定义奖励逻辑"""
        if self.episode_over:
            raise RuntimeError("Expected `reset` after episode terminated, not `step`.")
        
        # 执行原始的 step
        observation, reward, done, info = super().step(action)
        
        # 检查库存中的木头数量
        if 'inventory' in observation:
            current_total = 0
            for log_type in self.log_types:
                if log_type in observation['inventory']:
                    count = observation['inventory'][log_type]
                    # 处理 numpy array
                    if hasattr(count, 'item'):
                        count = count.item()
                    current_total += int(count)
            
            # 如果木头数量增加，给予奖励
            if current_total > self.total_logs:
                new_logs = current_total - self.total_logs
                reward = new_logs * self.reward_per_log
                self.total_logs = current_total
                
                # 达到目标数量，结束任务
                if self.total_logs >= self.target_logs:
                    done = True
                    print(f"✅ 任务完成！获得 {self.total_logs} 个木头，总奖励: {self.total_logs * self.reward_per_log}")
        
        # 超时检查
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        
        self.episode_over = done
        return observation, reward, done, info
    
    def reset(self):
        """重置状态"""
        self.total_logs = 0
        self.num_steps = 0
        self.episode_over = False
        obs = super().reset()
        return obs


class MineRLHarvestEnvSpec(HumanControlEnvSpec):
    """
    HarvestEnv 任务规范
    
    目标：获得 1 个木头（任何类型）
    奖励：每获得1个木头奖励100分
    终止：获得1个木头后结束，或超时
    """
    
    def __init__(self, resolution=(640, 320), max_episode_steps=2000, *args, **kwargs):
        # 设置环境名称
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLHarvestEnv-v0'
        
        # 设置 episode 长度为 2000 步
        if 'max_episode_steps' not in kwargs:
            kwargs['max_episode_steps'] = max_episode_steps
        
        # 保存 episode_len（在父类 __init__ 之前）
        self.episode_len = kwargs['max_episode_steps']
        self.reward_threshold = 100.0
        
        # 调用父类构造函数
        super().__init__(*args, resolution=resolution, **kwargs)

    def create_observables(self) -> List[Handler]:
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

    def create_rewardables(self) -> List[Handler]:
        return []    
    
    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start()
    
    def create_agent_handlers(self) -> List[Handler]:
        return []
    
    def create_server_world_generators(self) -> List[Handler]:
        """世界生成：平原群系"""
        return [
            handlers.DefaultWorldGenerator(force_reset=True, generator_options='{"biome":"plains"}'),
        ]
    
    def create_server_quit_producers(self) -> List[Handler]:
        """服务器退出条件"""
        return [
            handlers.ServerQuitFromTimeUp(self.episode_len * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]
    
    def create_server_decorators(self) -> List[Handler]:
        return []
    
    def create_server_initial_conditions(self) -> List[Handler]:
        """初始条件：白天"""
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=6000),
            handlers.SpawningInitialCondition(allow_spawning=True)
        ]
    
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return True

    def is_from_folder(self, folder: str) -> bool:
        return True

    def get_docstring(self):
        return ""


def _minerl_harvest_env_gym_entrypoint(env_spec, fake=False):
    """
    自定义 entry point，返回包装后的环境
    参考 MineRL ObtainDiamondShovel 的实现
    """
    if fake:
        # 如果需要 fake 环境（通常用于测试）
        from minerl.env import _fake
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        # 创建真实环境
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    
    # 应用自定义 Wrapper
    env = MineRLHarvestWrapper(env)
    return env


# Entry point 字符串（用于 gym.make）
MINE_RL_HARVEST_ENV_ENTRY_POINT = "src.envs.minerl_harvest:_minerl_harvest_env_gym_entrypoint"


def register_minerl_harvest_env():
    """注册 MineRLHarvestEnv 环境到 gym"""
    try:
        env_spec = MineRLHarvestEnvSpec()
        
        # 使用自定义的 entry_point 注册
        gym.envs.registration.register(
            id='MineRLHarvestEnv-v0',
            entry_point=MINE_RL_HARVEST_ENV_ENTRY_POINT,
            kwargs={'env_spec': env_spec, 'fake': False}
        )
        print("✓ MineRLHarvestEnv-v0 已注册（使用自定义奖励 Wrapper）")
    except gym.error.Error:
        # 环境已存在，跳过
        pass
