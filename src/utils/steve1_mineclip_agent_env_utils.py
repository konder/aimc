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


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_mineclip_wconfig():
    print('Loading MineClip...')
    return load(MINECLIP_CONFIG, device=DEVICE)


def make_env(seed, env_name='MineRLBasaltFindCave-v0'):
    """
    创建环境
    
    Args:
        seed: 随机种子
        env_name: 环境名称
            - 使用官方环境: 'MineRLBasaltFindCave-v0', 'HumanSurvival' 等
            - 使用自定义环境: 'MineRLHarvestEnv-v0' 等
    
    Returns:
        env: MineRL 环境
    """
    print(f'Loading MineRL environment: {env_name}...')
    
    # 如果是 HumanSurvival，使用原始方式创建
    if env_name == 'HumanSurvival':
        from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
        env = HumanSurvival(**ENV_KWARGS).make()
    else:
        # 使用 gym.make 创建环境（支持所有已注册的环境）
        env = gym.make(env_name)
    
    print('Starting new env...')
    env.reset()
    if seed is not None:
        print(f'Setting seed to {seed}...')
        env.seed(seed)
    return env


def make_agent(in_model, in_weights, cond_scale):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent


def load_mineclip_agent_env(in_model, in_weights, seed, cond_scale, env_name='MineRLBasaltFindCave-v0'):
    """
    加载 MineCLIP, Agent 和环境
    
    Args:
        in_model: VPT 模型路径
        in_weights: STEVE-1 权重路径
        seed: 随机种子
        cond_scale: CFG scale
        env_name: 环境名称（支持自定义环境）
    
    Returns:
        agent: MineRLConditionalAgent
        mineclip: MineCLIP 模型
        env: MineRL 环境
    """
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed, env_name=env_name)
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