"""
Agent抽象基类

定义统一的agent接口，用于与MineDojo环境交互
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np


class AgentBase(ABC):
    """
    Agent基类，定义统一接口
    
    所有agent必须实现predict方法，接收MineDojo observation，返回MineDojo action
    """
    
    def __init__(self, device: str = 'auto', verbose: bool = False):
        """
        初始化Agent基类
        
        Args:
            device: 设备 ('cpu', 'cuda', 'mps', 或 'auto')
            verbose: 是否打印详细信息
        """
        # 处理 'auto' device
        if device == 'auto':
            import torch as th
            if th.cuda.is_available():
                device = 'cuda'
            elif hasattr(th.backends, 'mps') and th.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.verbose = verbose
    
    @abstractmethod
    def predict(self, obs: Union[np.ndarray, Dict], deterministic: bool = True) -> List[int]:
        """
        根据观察预测动作
        
        Args:
            obs: MineDojo环境返回的observation
                 可能是np.ndarray (图像) 或 dict (包含'rgb'等键)
            deterministic: 是否使用确定性策略
        
        Returns:
            action: MineDojo动作空间格式 [dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7]
                   dim0: 前进/后退/不动 (0/1/2)
                   dim1: 左移/右移/不动 (0/1/2)
                   dim2: 跳跃/潜行/疾跑/不动 (0/1/2/3)
                   dim3: pitch (0-24)
                   dim4: yaw (0-24)
                   dim5: 功能键 (0-7)
                   dim6: (0-243)
                   dim7: (0-35)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        重置agent状态（如果有的话）
        
        用于在新的episode开始时重置内部状态（如LSTM hidden state）
        """
        pass
    
    def eval(self):
        """设置为评估模式（如果需要的话）"""
        pass
    
    def train(self):
        """设置为训练模式（如果需要的话）"""
        pass

