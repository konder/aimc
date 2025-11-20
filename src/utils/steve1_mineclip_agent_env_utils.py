"""
本地版本的 steve1 工具函数
支持自定义 MineRL 环境
"""

import pickle

import gym
import torch

from steve1.MineRLConditionalAgent import MineRLConditionalAgent
from steve1.VPT.agent import ENV_KWARGS
from steve1.config import MINECLIP_CONFIG, PRIOR_INFO
from steve1.mineclip_code.load_mineclip import load
from steve1.data.text_alignment.vae import TranslatorVAE

from .device import DEVICE

import logging
import time
    
logger = logging.getLogger(__name__)


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_mineclip_wconfig():
    logger.info('Loading MineClip...')
    return load(MINECLIP_CONFIG, device=DEVICE)


def make_env(seed, env_name='MineRLBasaltFindCave-v0', env_config=None):
    """
    创建环境
    
    Args:
        seed: 随机种子
        env_name: 环境名称
            - 使用官方环境: 'MineRLBasaltFindCave-v0', 'HumanSurvival' 等
            - 使用自定义环境: 'MineRLHarvestEnv-v0' 等
        env_config: 环境配置（包含 reward_config、reward_rule、max_episode_steps 等）
    
    Returns:
        env: MineRL 环境（可能被 Wrapper 包装）
    """
    logger.info(f'Loading MineRL environment: {env_name}...')
    
    # MineRL 自定义环境
    minerl_custom_envs = ['MineRLHarvestDefaultEnv-v0']
    # MineDojo 自定义环境
    minedojo_custom_envs = ['MineDojoHarvestEnv-v0']
    
    if env_name in minerl_custom_envs and env_config:
        # MineRL 环境配置
        from src.envs.item_name_mapper import convert_initial_inventory, convert_reward_config
        
        reward_config = env_config.get('reward_config')
        reward_rule = env_config.get('reward_rule', 'any')
        time_condition = env_config.get('time_condition')
        spawning_condition = env_config.get('spawning_condition')
        initial_inventory = env_config.get('initial_inventory')
        specified_biome = env_config.get('specified_biome')  # 新增：读取 specified_biome
        image_size = env_config.get('image_size')  # 新增：读取 image_size
        max_episode_steps = env_config.get('max_episode_steps', 2000)
        
        # 🔄 转换物品名称：MineDojo 格式 → MineRL 格式
        if initial_inventory:
            initial_inventory = convert_initial_inventory(initial_inventory, target_env='minerl')
            logger.info(f"  🔄 initial_inventory 转换为 MineRL 格式")
        
        if reward_config:
            reward_config = convert_reward_config(reward_config, target_env='minerl')
            logger.info(f"  🔄 reward_config 转换为 MineRL 格式")
        
        logger.info(f"{'='*30}")
        logger.info(f"创建 MineRL Harvest 环境")
        logger.info(f"{'='*30}")
        logger.info(f"  reward_config: {len(reward_config)} 项" if reward_config else "  reward_config: None")
        logger.info(f"  reward_rule: {reward_rule}")
        logger.info(f"  initial_inventory: {initial_inventory}" if initial_inventory else "  initial_inventory: None")
        if specified_biome:
            logger.info(f"  specified_biome: {specified_biome}")
        if image_size:
            logger.info(f"  image_size: {image_size}")
        logger.info(f"  max_episode_steps: {max_episode_steps}")
        
        # 创建环境并传递所有配置
        env = gym.make(
            env_name,
            reward_config=reward_config,
            reward_rule=reward_rule,
            time_condition=time_condition,
            spawning_condition=spawning_condition,
            initial_inventory=initial_inventory,
            specified_biome=specified_biome,  # 新增：传递 specified_biome
            image_size=image_size,  # 新增：传递 image_size
            max_episode_steps=max_episode_steps
        )
    elif env_name in minedojo_custom_envs and env_config:
        # MineDojo 环境配置
        generate_world_type = env_config.get('generate_world_type', 'default')
        specified_biome = env_config.get('specified_biome')
        world_seed = env_config.get('world_seed')
        task_id = env_config.get('task_id', 'open-ended')
        image_size = env_config.get('image_size', (160, 256))
        start_time = env_config.get('start_time', 6000)
        allow_time_passage = env_config.get('allow_time_passage', False)
        allow_mob_spawn = env_config.get('allow_mob_spawn', False)
        spawn_in_village = env_config.get('spawn_in_village', False)
        initial_inventory = env_config.get('initial_inventory')
        max_episode_steps = env_config.get('max_episode_steps', 2000)
        
        logger.info(f"{'='*30}")
        logger.info(f"创建 MineDojo Harvest 环境")
        logger.info(f"{'='*30}")
        logger.info(f"  task_id: {task_id}")
        logger.info(f"  generate_world_type: {generate_world_type}")
        if specified_biome:
            logger.info(f"  specified_biome: {specified_biome}")
        logger.info(f"  world_seed: {world_seed}" if world_seed else "  world_seed: (随机)")
        logger.info(f"  image_size: {image_size}")
        logger.info(f"  max_episode_steps: {max_episode_steps}")
        
        # 创建 MineDojo 环境
        env = gym.make(
            env_name,
            generate_world_type=generate_world_type,
            specified_biome=specified_biome,
            world_seed=world_seed,
            task_id=task_id,
            image_size=image_size,
            start_time=start_time,
            allow_time_passage=allow_time_passage,
            allow_mob_spawn=allow_mob_spawn,
            spawn_in_village=spawn_in_village,
            initial_inventory=initial_inventory,
            max_episode_steps=max_episode_steps
        )
    else:
        # 创建标准环境
        env = gym.make(env_name)
    
    # 不在这里 reset，留给 _run_single_trial 时首次调用
    # （避免双重初始化：一次在组件加载时，一次在 trial 开始时）
    logger.info('Environment created (will be reset on first trial)...')
    
    if seed is not None:
        logger.info(f'Setting seed to {seed}...')
        env.seed(seed)
    
    return env


def make_agent(in_model, in_weights, cond_scale):
    logger.info(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    
    # 🔧 修复dtype问题: 确保模型所有参数和buffers都是float32（针对4090等支持混合精度的GPU）
    # 将agent的policy网络及其所有子模块转为float32，避免与float16嵌入混用时出错
    if hasattr(agent, 'policy'):
        # 转换所有参数和buffers为float32
        agent.policy.float()
        # 递归转换所有子模块
        for module in agent.policy.modules():
            if hasattr(module, 'float'):
                module.float()
        logger.info('  Agent policy 及所有子模块已转换为 float32')
    
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent


def load_mineclip_agent_env(in_model, in_weights, seed, cond_scale, env_name='MineRLBasaltFindCave-v0', env_config=None):
    """
    加载 MineCLIP, Agent 和环境
    
    Args:
        in_model: VPT 模型路径
        in_weights: STEVE-1 权重路径
        seed: 随机种子
        cond_scale: CFG scale
        env_name: 环境名称（支持自定义环境）
        env_config: 环境配置（用于自定义环境）
    
    Returns:
        agent: MineRLConditionalAgent
        mineclip: MineCLIP 模型
        env: MineRL 环境
    """
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed, env_name=env_name, env_config=env_config)
    return agent, mineclip, env


def load_vae_model(vae_info):
    """
    加载 VAE Prior 模型（支持所有设备）
    
    Args:
        vae_info: 模型配置字典，包含：
            - mineclip_dim: MineCLIP 维度
            - latent_dim: 潜在维度
            - hidden_dim: 隐藏维度
            - model_path 或 prior_weights: 模型权重路径
    
    Returns:
        model: TranslatorVAE 模型
    """
    mineclip_dim = vae_info['mineclip_dim']
    latent_dim = vae_info['latent_dim']
    hidden_dim = vae_info['hidden_dim']
    model_path = vae_info.get('model_path') or vae_info.get('prior_weights')
    
    # 使用全局 STEVE1_DEVICE
    device = torch.device(DEVICE)
    
    model = TranslatorVAE(input_dim=mineclip_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model