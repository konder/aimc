#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineCLIP 官方奖励包装器
使用已安装的 MineCLIP 模型计算密集奖励
支持单帧和16帧视频模式
"""

import os
import gym
import numpy as np
import torch
from collections import deque


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
                 device="auto",
                 use_dynamic_weight=False,
                 weight_decay_steps=50000,
                 min_weight=0.01,
                 use_video_mode=True,
                 num_frames=16,
                 compute_frequency=4):
        """
        初始化 MineCLIP 奖励包装器
        
        Args:
            env: 基础环境
            task_prompt: 任务描述（英文），如 "chopping a tree with hand"
            model_path: MineCLIP 模型权重路径（.pth 文件）
            variant: MineCLIP 变体 ("attn" 或 "avg")
            sparse_weight: 稀疏奖励权重
            mineclip_weight: MineCLIP 密集奖励初始权重
            device: 运行设备 ("cpu", "cuda", "mps", 或 "auto")
            use_dynamic_weight: 是否使用动态权重调整（课程学习）
            weight_decay_steps: 权重衰减到最小值所需的步数
            min_weight: MineCLIP权重的最小值
            use_video_mode: 是否使用16帧视频模式（推荐）
            num_frames: 视频帧数（默认16，符合MineCLIP官方）
            compute_frequency: 每N步计算一次MineCLIP（减少开销）
        """
        super().__init__(env)
        
        self.task_prompt = task_prompt
        self.sparse_weight = sparse_weight
        self.initial_mineclip_weight = mineclip_weight
        self.mineclip_weight = mineclip_weight
        self.variant = variant
        
        # 动态权重调整参数
        self.use_dynamic_weight = use_dynamic_weight
        self.weight_decay_steps = weight_decay_steps
        self.min_weight = min_weight
        self.step_count = 0
        
        # 视频模式参数
        self.use_video_mode = use_video_mode
        self.num_frames = num_frames
        self.compute_frequency = compute_frequency
        self.frame_buffer = deque(maxlen=num_frames) if use_video_mode else None
        
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
        print(f"    运行模式: {'🎬 16帧视频模式' if use_video_mode else '🖼️  单帧模式'}")
        if use_video_mode:
            print(f"    视频帧数: {num_frames}帧")
            print(f"    计算频率: 每{compute_frequency}步")
        print(f"    稀疏权重: {sparse_weight}")
        print(f"    MineCLIP权重: {mineclip_weight} (初始值)")
        if use_dynamic_weight:
            print(f"    动态权重: 启用 (衰减步数: {weight_decay_steps}, 最小值: {min_weight})")
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
            image_features: 图像特征向量
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
            
            # CLIP标准归一化（ImageNet均值和标准差）
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            image = (image - mean) / std
            
            # MineCLIP 编码图像（使用 forward_image_features）
            image_features = self.model.forward_image_features(image)
            
            return image_features
    
    def _compute_similarity(self, image):
        """
        计算图像与任务的相似度
        
        Args:
            image: 当前观察图像
            
        Returns:
            similarity: 相似度分数（0-1之间）
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
    
    def _update_mineclip_weight(self):
        """
        根据训练步数动态更新MineCLIP权重（课程学习）
        
        策略：使用余弦衰减，从初始权重逐渐衰减到最小权重
        - 早期：高权重，agent依赖MineCLIP引导
        - 后期：低权重，agent更多依赖稀疏奖励和自身策略
        """
        if not self.use_dynamic_weight:
            return
        
        # 计算衰减进度 [0, 1]
        progress = min(self.step_count / self.weight_decay_steps, 1.0)
        
        # 余弦衰减：从1.0平滑下降到0.0
        decay_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        # 计算当前权重
        weight_range = self.initial_mineclip_weight - self.min_weight
        self.mineclip_weight = self.min_weight + weight_range * decay_factor
    
    def _encode_video(self, frames):
        """
        编码16帧视频序列（MineCLIP官方方式）
        
        Args:
            frames: List of [H, W, C] numpy arrays
            
        Returns:
            video_features: 视频特征向量
        """
        if not self.mineclip_available:
            return None
        
        try:
            with torch.no_grad():
                # MineCraft官方归一化参数
                MC_MEAN = torch.tensor([0.3331, 0.3245, 0.3051], device=self.device).view(1, 1, 3, 1, 1)
                MC_STD = torch.tensor([0.2439, 0.2493, 0.2873], device=self.device).view(1, 1, 3, 1, 1)
                
                # 预处理帧序列
                processed_frames = []
                for frame in frames:
                    # 转换为tensor
                    if isinstance(frame, np.ndarray):
                        frame_tensor = torch.from_numpy(frame).float()
                    else:
                        frame_tensor = frame.float()
                    
                    # 确保是 [H, W, C] 格式
                    if frame_tensor.dim() == 3 and frame_tensor.shape[0] == 3:
                        frame_tensor = frame_tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                    
                    # 归一化到 [0, 1]
                    if frame_tensor.max() > 1.0:
                        frame_tensor = frame_tensor / 255.0
                    
                    # [H, W, C] -> [C, H, W]
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                    
                    processed_frames.append(frame_tensor)
                
                # 堆叠为 [T, C, H, W]
                video_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
                
                # MineCraft归一化
                video_tensor = (video_tensor - MC_MEAN) / MC_STD
                
                # 使用MineCLIP的encode_video（完整官方流程）
                video_features = self.model.encode_video(video_tensor)
                
                return video_features
        
        except Exception as e:
            print(f"    ⚠️ 视频编码失败: {e}")
            return None
    
    def _compute_video_similarity(self, frames):
        """
        计算16帧视频与任务的相似度
        
        Args:
            frames: List of 16 frames
            
        Returns:
            similarity: 相似度分数（0-1之间）
        """
        if not self.mineclip_available or len(frames) < self.num_frames:
            return 0.0
        
        try:
            # 编码视频
            video_features = self._encode_video(frames)
            if video_features is None:
                return 0.0
            
            # 归一化
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = (video_features @ self.task_features.T).item()
            
            # 转换为 [0, 1] 范围
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity)
        
        except Exception as e:
            print(f"    ⚠️ 视频相似度计算失败: {e}")
            return 0.0
    
    def reset(self, **kwargs):
        """重置环境"""
        # MineDojo 的 reset 不接受参数
        obs = self.env.reset()
        
        # 清空帧缓冲
        if self.use_video_mode and self.frame_buffer is not None:
            self.frame_buffer.clear()
            # 用初始帧填充缓冲区
            for _ in range(self.num_frames):
                self.frame_buffer.append(obs.copy())
        
        if self.mineclip_available:
            # 计算初始相似度
            if self.use_video_mode and len(self.frame_buffer) == self.num_frames:
                self.previous_similarity = self._compute_video_similarity(list(self.frame_buffer))
            else:
                self.previous_similarity = self._compute_similarity(obs)
        
        return obs
    
    def step(self, action):
        """
        执行一步，返回增强的奖励
        
        MineCLIP 密集奖励机制：
        - 单帧模式：计算当前画面与任务的相似度
        - 16帧视频模式：累积16帧，每N步计算视频与任务的相似度
        
        Args:
            action: 动作
            
        Returns:
            tuple: (观察, 奖励, 完成标志, 信息)
        """
        obs, sparse_reward, done, info = self.env.step(action)
        
        # 更新步数计数器
        self.step_count += 1
        
        # 添加帧到缓冲区（视频模式）
        if self.use_video_mode and self.frame_buffer is not None:
            self.frame_buffer.append(obs.copy())
        
        if self.mineclip_available:
            # 更新MineCLIP权重（如果启用动态调整）
            self._update_mineclip_weight()
            
            # 计算当前相似度
            current_similarity = 0.0
            should_compute = False
            
            if self.use_video_mode:
                # 16帧视频模式：每N步计算一次
                if self.step_count % self.compute_frequency == 0 and len(self.frame_buffer) == self.num_frames:
                    current_similarity = self._compute_video_similarity(list(self.frame_buffer))
                    should_compute = True
                else:
                    # 非计算步，保持上一次的相似度
                    current_similarity = self.previous_similarity
            else:
                # 单帧模式：每步都计算
                current_similarity = self._compute_similarity(obs)
                should_compute = True
            
            # MineCLIP 密集奖励 = 相似度进步量
            if should_compute:
                mineclip_reward = current_similarity - self.previous_similarity
                self.previous_similarity = current_similarity
            else:
                mineclip_reward = 0.0
            
            # 组合奖励
            total_reward = (
                sparse_reward * self.sparse_weight + 
                mineclip_reward * self.mineclip_weight
            )
            
            # 记录详细信息
            info['sparse_reward'] = sparse_reward
            info['mineclip_reward'] = mineclip_reward
            info['mineclip_similarity'] = current_similarity
            info['mineclip_weight'] = self.mineclip_weight
            info['sparse_weight'] = self.sparse_weight
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

