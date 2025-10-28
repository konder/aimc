"""
VPT Agent Wrapper for MineDojo

架构设计：
1. 组合官方VPT的MineRLAgent（不修改官方代码）
2. VPTAgent只负责MineDojo观察/动作的转换
3. 所有VPT逻辑完全委托给官方agent

官方VPT参考：
- https://github.com/openai/Video-Pre-Training
- 本地官方代码：src/models/Video-Pre-Training/
- 已修改：Video-Pre-Training/lib/actions.py使用本地mc.py（去除minerl依赖）
"""

import sys
import numpy as np
from pathlib import Path

# 添加官方VPT代码路径
VPT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "Video-Pre-Training"
if not VPT_PATH.exists():
    raise FileNotFoundError(f"官方VPT代码未找到: {VPT_PATH}")

vpt_path_str = str(VPT_PATH)
if vpt_path_str not in sys.path:
    sys.path.insert(0, vpt_path_str)

try:
    # 导入官方VPT Agent和配置
    from agent import MineRLAgent, ENV_KWARGS
finally:
    # 移除路径（保持sys.path干净）
    if vpt_path_str in sys.path:
        sys.path.remove(vpt_path_str)

from ..agent.agent_base import AgentBase


class MineRLActionToMineDojo:
    """MineRL Action -> MineDojo Action Converter"""

    def __init__(self, conflict_strategy='priority'):
        """
        Args:
            conflict_strategy: 冲突处理策略
                - 'priority': 优先级策略（forward > back, left > right等）
                - 'cancel': 冲突时取消
        """
        self.conflict_strategy = conflict_strategy
    
    def convert(self, minerl_action: dict, debug=False) -> np.ndarray:
        """
        将MineRL动作转换为MineDojo动作
        
        Args:
            minerl_action: MineRL格式的动作字典
            debug: 是否打印调试信息
            
        Returns:
            MineDojo格式的动作数组 [8]
        """
        minedojo_action = np.zeros(8, dtype=np.int32)
        
        # 前后移动
        forward = minerl_action.get('forward', 0)
        back = minerl_action.get('back', 0)
        if forward and back:
            if self.conflict_strategy == 'priority':
                minedojo_action[0] = 1
        elif forward:
            minedojo_action[0] = 1
        elif back:
            minedojo_action[0] = 2
        
        # 左右移动
        left = minerl_action.get('left', 0)
        right = minerl_action.get('right', 0)
        if left and right:
            if self.conflict_strategy == 'priority':
                minedojo_action[1] = 1
        elif left:
            minedojo_action[1] = 1
        elif right:
            minedojo_action[1] = 2
        
        # 跳跃/潜行/疾跑
        if minerl_action.get('jump', 0):
            minedojo_action[2] = 1
        elif minerl_action.get('sneak', 0):
            minedojo_action[2] = 2
        elif minerl_action.get('sprint', 0):
            minedojo_action[2] = 3
        
        # 相机
        camera = minerl_action.get('camera', np.array([0.0, 0.0]))
        # camera是一个numpy数组，可能是[[x, y]]或[x, y]
        camera = np.asarray(camera).flatten()  # 展平为1D数组
        camera_x = float(camera[0])
        camera_y = float(camera[1])
        minedojo_action[3] = int(np.clip(camera_x, -10, 10)) + 12
        minedojo_action[4] = int(np.clip(camera_y, -10, 10)) + 12
        
        # 攻击
        if minerl_action.get('attack', 0):
            minedojo_action[5] = 1
        
        # 使用
        if minerl_action.get('use', 0):
            minedojo_action[6] = 1
        
        if debug:
            print(f"\n  🔍 MineRL Action:")
            active_keys = [k for k, v in minerl_action.items() if np.any(v != 0)]
            for key in active_keys:
                print(f"    {key}: {minerl_action[key]}")
            
            print(f"\n  🎯 MineDojo Action: {minedojo_action}")
        
        return minedojo_action


class VPTAgent(AgentBase):
    """
    VPT Agent for MineDojo - 组合模式
    
    设计：
    1. 持有一个官方MineRLAgent实例
    2. 所有VPT操作委托给官方agent
    3. 只实现MineDojo <-> MineRL的转换
    
    官方VPT参考：
    - GitHub: https://github.com/openai/Video-Pre-Training
    - 本地: src/models/Video-Pre-Training/agent.py
    """
    
    def __init__(
        self,
        vpt_weights_path: str,
        device: str = 'auto',
        conflict_strategy: str = 'priority',
        verbose: bool = False,
        debug_actions: bool = False
    ):
        """
        初始化VPT Agent
        
        Args:
            vpt_weights_path: VPT预训练权重路径
            device: 'cpu', 'cuda', 'mps', or 'auto'
            conflict_strategy: MineRL→MineDojo转换的冲突处理策略
            verbose: 是否打印详细信息
            debug_actions: 是否打印动作转换调试信息
        """
        super().__init__(device, verbose)
        
        self.vpt_weights_path = vpt_weights_path
        self.debug_actions = debug_actions
        
        if self.verbose:
            print("\n" + "="*70)
            print("🤖 初始化VPT Agent (组合官方MineRLAgent)")
            print("="*70)
            print(f"权重: {vpt_weights_path}")
            print(f"设备: {self.device}")
            print(f"官方Agent: src/models/Video-Pre-Training/agent.py")
            print(f"适配层: MineDojo观察/动作转换")
        
        # 处理设备
        import torch as th
        if self.device == 'auto':
            if th.cuda.is_available():
                device_str = 'cuda'
            elif hasattr(th.backends, 'mps') and th.backends.mps.is_available():
                device_str = 'mps'
            else:
                device_str = 'cpu'
        else:
            device_str = self.device
        
        # ========================================
        # 创建官方MineRLAgent（组合模式）
        # ========================================
        # 官方agent需要一个MineRL环境来validate
        # 使用MineRL 1.0+的HumanSurvival环境（与官方run_agent.py相同）
        from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
        
        # 创建MineRL环境（只用于初始化验证）
        # 使用官方ENV_KWARGS确保环境配置正确
        minerl_env = HumanSurvival(**ENV_KWARGS).make()
        
        if self.verbose:
            print("\n📦 创建官方MineRLAgent...")
        
        # 创建官方agent（完全使用官方代码）
        self.vpt_agent = MineRLAgent(
            env=minerl_env,
            device=device_str,
            policy_kwargs=None,  # 使用官方默认配置
            pi_head_kwargs=None
        )
        
        # 关闭MineRL环境（只用于验证，不再需要）
        minerl_env.close()
        
        # 加载权重（调用官方load_weights方法）
        if self.verbose:
            print(f"📥 加载VPT权重（调用官方load_weights）...")
        
        self.vpt_agent.load_weights(vpt_weights_path)
        
        if self.verbose:
            print(f"  ✓ 官方MineRLAgent创建完成")
            print(f"  ✓ policy.act会自动添加时间维度T")
        
        # ========================================
        # 创建MineDojo适配层
        # ========================================
        self.action_converter = MineRLActionToMineDojo(conflict_strategy)
        
        if self.verbose:
            print("\n✅ VPT Agent初始化完成!")
            print("="*70)
    
    def reset(self):
        """
        重置Agent - 直接调用官方agent.reset()
        """
        self.vpt_agent.reset()
        if self.verbose:
            print("🔄 VPT Agent reset (调用官方agent.reset)")
    
    def _convert_obs_to_minerl(self, minedojo_obs) -> dict:
        """
        Convert MineDojo observation to MineRL format
        
        MineDojo returns CHW format (C, H, W), but MineRL/VPT expects HWC format (H, W, C).
        This method handles the conversion.
        
        Args:
            minedojo_obs: MineDojo observation
                - Can be dict with 'rgb' key: {'rgb': [C, H, W]}
                - Or numpy array: [C, H, W] uint8
        
        Returns:
            MineRL observation dict: {"pov": [H, W, C] uint8}
        """
        # Extract POV from observation
        if isinstance(minedojo_obs, dict):
            pov = minedojo_obs['rgb']
        else:
            pov = minedojo_obs
        
        # Validate POV is numpy array
        if not isinstance(pov, np.ndarray):
            raise TypeError(f"POV must be numpy array, got {type(pov)}")
        
        # Convert CHW to HWC if needed
        # MineDojo: (C, H, W) -> MineRL: (H, W, C)
        if pov.shape[0] == 3 and len(pov.shape) == 3:
            pov = np.transpose(pov, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        
        # Ensure contiguous array for cv2.resize
        pov = np.ascontiguousarray(pov)
        
        return {"pov": pov}
    
    def predict(self, minedojo_obs, deterministic: bool = False) -> np.ndarray:
        """
        Predict MineDojo action using VPT agent
        
        Pipeline:
        1. Convert MineDojo obs -> MineRL obs (adapter layer)
        2. Call official agent.get_action()
        3. Convert MineRL action -> MineDojo action (adapter layer)
        
        Args:
            minedojo_obs: MineDojo observation
                - Can be dict with 'rgb' key: {'rgb': [C, H, W]}
                - Or numpy array: [C, H, W] uint8
            deterministic: Whether to use deterministic prediction (VPT defaults to stochastic)
            
        Returns:
            MineDojo action [8] int32
        """
        # Step 1: Convert MineDojo obs -> MineRL obs
        minerl_obs = self._convert_obs_to_minerl(minedojo_obs)
        
        # Step 2: Call official agent.get_action()
        minerl_action = self.vpt_agent.get_action(minerl_obs)
        
        # Step 3: Convert MineRL action -> MineDojo action
        minedojo_action = self.action_converter.convert(
            minerl_action,
            debug=self.debug_actions
        )
        
        return minedojo_action


# For backward compatibility
MineRLToMinedojoConverter = MineRLActionToMineDojo
