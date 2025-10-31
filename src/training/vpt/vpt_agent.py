"""
VPT Agent for MineDojo Training/Evaluation

架构层次：
1. src/models/vpt/agent.py::MineRLAgent - 官方VPT实现（不修改）
2. src/models/vpt/minedojo_agent.py::MineDojoAgent - MineDojo适配层（重载obs/action转换）
3. src/training/vpt/vpt_agent.py::VPTAgent - 训练/评估接口（本文件，实现AgentBase）

VPTAgent职责：
- 多重继承: MineDojoAgent（提供VPT功能）+ AgentBase（提供统一接口）
- 实现 predict() 方法（AgentBase要求）
- 实现 reset() 方法（AgentBase要求）
- 提供训练/评估所需的统一接口

官方VPT参考：
- GitHub: https://github.com/openai/Video-Pre-Training
- 本地VPT代码：src/models/vpt/
"""

import sys
import numpy as np
from pathlib import Path

# 添加 VPT 代码路径
VPT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "vpt"
if not VPT_PATH.exists():
    raise FileNotFoundError(f"VPT代码未找到: {VPT_PATH}")

vpt_path_str = str(VPT_PATH)
if vpt_path_str not in sys.path:
    sys.path.insert(0, vpt_path_str)

try:
    from minedojo_agent import MineDojoAgent
    from agent import ENV_KWARGS
finally:
    if vpt_path_str in sys.path:
        sys.path.remove(vpt_path_str)

from ..agent.agent_base import AgentBase


class VPTAgent(MineDojoAgent, AgentBase):
    """
    VPT Agent for MineDojo Training/Evaluation
    
    多重继承：
    - MineDojoAgent: 提供VPT核心功能和MineDojo适配
    - AgentBase: 提供统一的训练/评估接口
    
    使用方式：
        agent = VPTAgent(
            vpt_weights_path='path/to/weights.weights',
            device='auto',
            cam_interval=0.01
        )
        
        # 训练/评估
        obs = env.reset()
        action = agent.predict(obs, deterministic=True)
        
        # 重置
        agent.reset()
    """
    
    def __init__(
        self,
        vpt_weights_path: str,
        device: str = 'auto',
        cam_interval: float = 0.01,
        verbose: bool = False
    ):
        """
        初始化VPT Agent
        
        Args:
            vpt_weights_path: VPT预训练权重路径
            device: 'cpu', 'cuda', 'mps', or 'auto'
            cam_interval: MineDojo环境的camera间隔（度数）
                - 0.01 = 0.01度精度（推荐，匹配VPT的[-10,10]范围）
                - 1.0 = 1度精度
                - 15.0 = 15度精度（MineDojo默认）
            verbose: 是否打印详细信息
        """
        # 保存参数
        self.vpt_weights_path = vpt_weights_path
        self.verbose = verbose
        
        # 处理设备
        import torch as th
        if device == 'auto':
            if th.cuda.is_available():
                device_str = 'cuda'
            elif hasattr(th.backends, 'mps') and th.backends.mps.is_available():
                device_str = 'mps'
            else:
                device_str = 'cpu'
        else:
            device_str = device
        
        if self.verbose:
            print("\n" + "="*70)
            print("🤖 初始化 VPT Agent for MineDojo")
            print("="*70)
            print(f"权重: {vpt_weights_path}")
            print(f"设备: {device_str}")
            print(f"Camera精度: {cam_interval}度/单位")
            print("="*70)
        
        # 创建临时 MineRL 环境用于初始化验证
        from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
        minerl_env = HumanSurvival(**ENV_KWARGS).make()
        
        try:
            # 初始化 MineDojoAgent（会调用MineRLAgent.__init__）
            MineDojoAgent.__init__(
                self,
                env=minerl_env,
                device=device_str,
                policy_kwargs=None,
                pi_head_kwargs=None,
                cam_interval=cam_interval
            )
            
            # 初始化 AgentBase
            AgentBase.__init__(self, device=device_str, verbose=verbose)
            
            # 加载权重
            if self.verbose:
                print(f"\n📥 加载VPT权重...")
            
            self.load_weights(vpt_weights_path)
            
            if self.verbose:
                print(f"✅ VPT Agent初始化完成!")
                print("="*70)
        
        finally:
            # 关闭临时环境
            minerl_env.close()
    
    def predict(self, obs, deterministic: bool = False) -> np.ndarray:
        """
        预测动作（AgentBase接口要求）
        
        Args:
            obs: MineDojo观察 {'rgb': [C, H, W]}
            deterministic: 是否使用确定性策略（VPT忽略此参数，总是确定性）
        
        Returns:
            MineDojo动作数组 [8]
        """
        # 调用 get_action (继承自MineRLAgent)
        # 内部会自动调用重载的 _env_obs_to_agent 和 _agent_action_to_env
        return self.get_action(obs)
    
    def reset(self):
        """
        重置Agent状态（AgentBase接口要求）
        
        调用 MineRLAgent.reset() 来重置隐藏状态
        """
        # MineRLAgent.reset() 继承自父类
        super(MineDojoAgent, self).reset()  # 调用 MineRLAgent.reset()
        
        if self.verbose:
            print("🔄 VPT Agent reset")
    
    def __repr__(self):
        return (
            f"VPTAgent(\n"
            f"  weights='{self.vpt_weights_path}',\n"
            f"  device='{self.device}',\n"
            f"  cam_interval={self.cam_interval}\n"
            f")"
        )


# 向后兼容：导出旧的类名
MineRLToMinedojoConverter = None  # 已废弃，保留以避免破坏性更改
