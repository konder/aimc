#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineCLIP 官方奖励包装器
使用已安装的 MineCLIP 模型计算密集奖励
"""

import os
import gym
import numpy as np
import torch


class MineCLIPRewardWrapper(gym.Wrapper):
    """
    使用官方 MineCLIP 模型的奖励包装器
    
    特性：
    - 使用预训练的 MineCLIP 模型（attn 或 avg）
    - 计算当前画面与任务描述的相似度
    - 提供连续密集奖励（每一步都有反馈）
    """
    
    def __init__(self, env, task_prompt, 
                 model_path=None,
                 variant="attn",
                 sparse_weight=10.0, 
                 mineclip_weight=0.1,
                 device="auto"):
        """
        初始化 MineCLIP 奖励包装器
        
        Args:
            env: 基础环境
            task_prompt: 任务描述（英文），如 "chop down a tree and collect one wood log"
            model_path: MineCLIP 模型权重路径（.pth 文件）
            variant: MineCLIP 变体 ("attn" 或 "avg")
            sparse_weight: 稀疏奖励权重
            mineclip_weight: MineCLIP 密集奖励权重
            device: 运行设备 ("cpu", "cuda", "mps", 或 "auto")
        """
        super().__init__(env)
        
        self.task_prompt = task_prompt
        self.sparse_weight = sparse_weight
        self.mineclip_weight = mineclip_weight
        self.variant = variant
        
        # 检测设备
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"  MineCLIP 奖励包装器:")
        print(f"    任务描述: {task_prompt}")
        print(f"    模型变体: {variant}")
        print(f"    稀疏权重: {sparse_weight}")
        print(f"    MineCLIP权重: {mineclip_weight}")
        print(f"    设备: {self.device}")
        
        # 加载 MineCLIP 模型
        self.mineclip_available = self._load_mineclip(model_path)
        
        if self.mineclip_available:
            print(f"    状态: ✓ MineCLIP 模型已加载")
            # 预计算任务文本的特征
            self.task_features = self._encode_text(task_prompt)
        else:
            print(f"    状态: ✗ MineCLIP 不可用，使用稀疏奖励")
        
        # 初始化状态
        self.previous_similarity = 0.0
    
    def _load_mineclip(self, model_path):
        """
        加载 MineCLIP 模型
        
        Args:
            model_path: 模型权重路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 导入 MineCLIP
            from mineclip import MineCLIP
            
            # MineCLIP 配置（attn 和 avg 的区别在 pool_type）
            configs = {
                "attn": {
                    "arch": "vit_base_p16_fz.v2.t2",
                    "pool_type": "attn.d2.nh8.glusw",
                    "resolution": (160, 256),
                    "image_feature_dim": 512,
                    "mlp_adapter_spec": "v0-2.t0",
                    "hidden_dim": 512
                },
                "avg": {
                    "arch": "vit_base_p16_fz.v2.t2",
                    "pool_type": "avg",
                    "resolution": (160, 256),
                    "image_feature_dim": 512,
                    "mlp_adapter_spec": "v0-2.t0",
                    "hidden_dim": 512
                }
            }
            
            if self.variant not in configs:
                print(f"    ✗ 未知的 variant: {self.variant}")
                return False
            
            # 创建模型
            print(f"    正在加载 MineCLIP {self.variant} 模型...")
            config = configs[self.variant]
            self.model = MineCLIP(**config).to(self.device)
            
            # 加载预训练权重
            if model_path and os.path.exists(model_path):
                print(f"    从 {model_path} 加载权重...")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # checkpoint 可能是字典或直接是 state_dict
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 处理键名：去掉 'model.' 前缀（如果存在）
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        new_key = key[6:]  # 去掉 'model.' 前缀
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                self.model.load_state_dict(new_state_dict)
                print(f"    ✓ 权重加载成功")
            else:
                print(f"    ⚠️ 未指定模型路径，使用随机初始化（性能会很差）")
                print(f"    请下载预训练模型: https://github.com/MineDojo/MineCLIP")
            
            self.model.eval()
            
            # 加载 tokenizer（优先使用本地，避免每次访问 HuggingFace）
            from transformers import CLIPTokenizer
            
            # 本地 tokenizer 路径
            local_tokenizer_path = "data/clip_tokenizer"
            
            if os.path.exists(local_tokenizer_path):
                print(f"    使用本地 tokenizer: {local_tokenizer_path}")
                self.tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)
            else:
                print(f"    本地 tokenizer 不存在，从 HuggingFace 下载...")
                print(f"    提示: 运行 'python scripts/download_clip_tokenizer.py' 可离线使用")
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            
            return True
            
        except ImportError as e:
            print(f"    ✗ MineCLIP 未安装: {e}")
            print(f"    安装命令: pip install git+https://github.com/MineDojo/MineCLIP")
            return False
        except Exception as e:
            print(f"    ✗ MineCLIP 加载失败: {e}")
            return False
    
    def _encode_text(self, text):
        """
        编码文本为特征向量
        
        Args:
            text: 文本描述
            
        Returns:
            torch.Tensor: 文本特征向量
        """
        with torch.no_grad():
            # Tokenize 文本（固定长度 77）
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True
            )
            
            # 移到设备
            token_ids = tokens['input_ids'].to(self.device)
            
            # MineCLIP 编码文本
            text_features = self.model.encode_text(token_ids)
            return text_features
    
    def _encode_image(self, image):
        """
        编码图像为特征向量
        
        Args:
            image: 图像（numpy array, shape: (C, H, W), 范围 [0, 1]）
            
        Returns:
            torch.Tensor: 图像特征向量
        """
        with torch.no_grad():
            # 转换为 tensor 并添加 batch 维度
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            
            # 确保范围在 [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
            
            # 添加 batch 维度
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # 移到设备
            image = image.to(self.device)
            
            # MineCLIP 编码图像（使用 forward_image_features）
            image_features = self.model.forward_image_features(image)
            return image_features
    
    def _compute_similarity(self, image):
        """
        计算图像与任务的相似度
        
        Args:
            image: 当前观察图像
            
        Returns:
            float: 相似度分数（0-1之间）
        """
        if not self.mineclip_available:
            return 0.0
        
        try:
            # 编码图像
            image_features = self._encode_image(image)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(
                image_features, 
                self.task_features, 
                dim=-1
            )
            
            # 转换为 [0, 1] 范围
            # cosine similarity 范围是 [-1, 1]，映射到 [0, 1]
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity.item())
        
        except Exception as e:
            print(f"    ⚠️ 相似度计算失败: {e}")
            return 0.0
    
    def reset(self, **kwargs):
        """重置环境"""
        # MineDojo 的 reset 不接受参数
        obs = self.env.reset()
        
        if self.mineclip_available:
            # 计算初始相似度
            self.previous_similarity = self._compute_similarity(obs)
        
        return obs
    
    def step(self, action):
        """
        执行一步，返回增强的奖励
        
        MineCLIP 密集奖励机制：
        1. 计算当前画面与任务的相似度（0-1）
        2. 奖励 = 当前相似度 - 上一步相似度（进步量）
        3. 每一步都有连续的反馈信号
        
        Args:
            action: 动作
            
        Returns:
            tuple: (观察, 奖励, 完成标志, 信息)
        """
        obs, sparse_reward, done, info = self.env.step(action)
        
        if self.mineclip_available:
            # 计算当前相似度
            current_similarity = self._compute_similarity(obs)
            
            # MineCLIP 密集奖励 = 进步量
            mineclip_reward = current_similarity - self.previous_similarity
            self.previous_similarity = current_similarity
            
            # 组合奖励
            total_reward = (
                sparse_reward * self.sparse_weight + 
                mineclip_reward * self.mineclip_weight
            )
            
            # 记录详细信息
            info['sparse_reward'] = sparse_reward
            info['mineclip_reward'] = mineclip_reward
            info['mineclip_similarity'] = current_similarity
            info['total_reward'] = total_reward
        else:
            # MineCLIP 不可用，只使用稀疏奖励
            total_reward = sparse_reward
        
        return obs, total_reward, done, info
    
    def close(self):
        """关闭环境"""
        return self.env.close()


def create_mineclip_wrapper(env, task_id, model_path=None, variant="attn"):
    """
    为环境添加 MineCLIP 奖励包装器的便捷函数
    
    Args:
        env: MineDojo 环境
        task_id: 任务 ID（用于生成任务描述）
        model_path: MineCLIP 模型路径
        variant: MineCLIP 变体 ("attn" 或 "avg")
        
    Returns:
        包装后的环境
    """
    # 根据任务 ID 生成任务描述
    task_prompts = {
        "harvest_1_log": "chop down a tree and collect one wood log",
        "harvest_1_paper": "collect one piece of paper",
        "hunt_1_cow": "hunt and kill one cow",
        # 可以添加更多任务
    }
    
    prompt = task_prompts.get(task_id, f"complete task {task_id}")
    
    return MineCLIPRewardWrapper(
        env,
        task_prompt=prompt,
        model_path=model_path,
        variant=variant,
        sparse_weight=10.0,
        mineclip_weight=0.1
    )

