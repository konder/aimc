#!/usr/bin/env python3
"""
MineCLIP微调 - 训练器 v2

改进版：结合文本prompt的时序对比学习
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm


class TaskAwareTemporalContrastiveLoss(nn.Module):
    """
    任务感知的时序对比损失
    
    改进：同时优化两个目标
    1. 时序关系：后期帧 > 早期帧
    2. 任务相关：后期帧应该更接近任务文本描述
    """
    
    def __init__(self, margin=0.2, task_weight=0.5):
        """
        Args:
            margin: 边界值
            task_weight: 任务对比损失的权重（0-1之间）
        """
        super().__init__()
        self.margin = margin
        self.task_weight = task_weight
        self.temporal_weight = 1.0 - task_weight
    
    def forward(self, anchor_features, positive_features, negative_features, 
                task_features):
        """
        计算对比损失
        
        Args:
            anchor_features: [B, D] anchor的特征
            positive_features: [B, D] positive的特征
            negative_features: [B, D] negative的特征
            task_features: [D] 任务文本的特征（共享）
            
        Returns:
            loss: scalar
        """
        # 归一化特征
        anchor_features = F.normalize(anchor_features, dim=1)
        positive_features = F.normalize(positive_features, dim=1)
        negative_features = F.normalize(negative_features, dim=1)
        
        # task_features需要detach（不参与梯度计算）
        task_features = F.normalize(task_features.detach(), dim=0)
        
        # 损失1: 时序对比（图像-图像）
        sim_pos = torch.sum(anchor_features * positive_features, dim=1)
        sim_neg = torch.sum(anchor_features * negative_features, dim=1)
        temporal_loss = F.relu(self.margin - (sim_pos - sim_neg))
        
        # 损失2: 任务对比（图像-文本）
        # positive应该比negative更接近任务描述
        task_features = task_features.unsqueeze(0)  # [1, D]
        
        sim_pos_task = torch.sum(positive_features * task_features, dim=1)
        sim_neg_task = torch.sum(negative_features * task_features, dim=1)
        task_loss = F.relu(self.margin - (sim_pos_task - sim_neg_task))
        
        # 组合损失
        total_loss = (
            self.temporal_weight * temporal_loss.mean() +
            self.task_weight * task_loss.mean()
        )
        
        return total_loss, sim_pos.mean(), sim_neg.mean(), sim_pos_task.mean()


class MineCLIPFineTunerV2:
    """
    MineCLIP微调器 v2 - 任务感知版本
    """
    
    def __init__(
        self,
        base_model_path,
        task_prompt="chop down a tree and collect one wood log",
        device="auto",
        freeze_ratio=0.8,
        learning_rate=1e-5,
        weight_decay=1e-4,
        task_weight=0.5
    ):
        """
        初始化微调器
        
        Args:
            base_model_path: 预训练MineCLIP模型路径
            task_prompt: 任务描述（重要！）
            device: 运行设备
            freeze_ratio: 冻结底层的比例
            learning_rate: 学习率
            weight_decay: 权重衰减
            task_weight: 任务对比损失的权重
        """
        self.base_model_path = base_model_path
        self.task_prompt = task_prompt
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_ratio = freeze_ratio
        self.task_weight = task_weight
        
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
        
        print(f"\n{'='*70}")
        print(f"MineCLIP微调器 v2（任务感知版本）")
        print(f"{'='*70}")
        print(f"基础模型: {base_model_path}")
        print(f"任务描述: {task_prompt}")  # ← 新增
        print(f"设备: {self.device}")
        print(f"冻结比例: {freeze_ratio*100:.0f}%")
        print(f"学习率: {learning_rate}")
        print(f"任务权重: {task_weight}")  # ← 新增
        print(f"{'='*70}\n")
        
        # 加载模型
        self.model, self.tokenizer = self._load_model()
        
        # 预计算任务文本特征（只计算一次）
        self.task_features = self._encode_text(task_prompt)
        print(f"✓ 任务文本特征已预计算")
        
        # 冻结底层
        self._freeze_bottom_layers()
        
        # 损失函数
        self.criterion = TaskAwareTemporalContrastiveLoss(
            margin=0.2,
            task_weight=task_weight
        )
        
        # 优化器
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params_count:,} ({trainable_params_count/total_params*100:.1f}%)")
        print()
    
    def _load_model(self):
        """加载预训练MineCLIP模型"""
        try:
            from mineclip import MineCLIP
            from transformers import CLIPTokenizer
            
            config = {
                "arch": "vit_base_p16_fz.v2.t2",
                "pool_type": "attn.d2.nh8.glusw",
                "resolution": (160, 256),
                "image_feature_dim": 512,
                "mlp_adapter_spec": "v0-2.t0",
                "hidden_dim": 512
            }
            
            print(f"加载MineCLIP模型...")
            model = MineCLIP(**config).to(self.device)
            
            # 加载预训练权重
            if os.path.exists(self.base_model_path):
                checkpoint = torch.load(self.base_model_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key[6:] if key.startswith('model.') else key
                    new_state_dict[new_key] = value
                
                model.load_state_dict(new_state_dict)
                print(f"  ✓ 预训练权重加载成功")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.base_model_path}")
            
            # 加载tokenizer
            local_tokenizer_path = "data/clip_tokenizer"
            if os.path.exists(local_tokenizer_path):
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)
            else:
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            
            return model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _encode_text(self, text):
        """编码文本为特征向量"""
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True
            )
            token_ids = tokens['input_ids'].to(self.device)
            text_features = self.model.encode_text(token_ids)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0)  # [D]
    
    def _freeze_bottom_layers(self):
        """冻结底层参数"""
        all_params = list(self.model.parameters())
        n_params = len(all_params)
        n_freeze = int(n_params * self.freeze_ratio)
        
        for i, param in enumerate(all_params):
            if i < n_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def _encode_frames(self, frames):
        """编码帧为特征向量"""
        MC_MEAN = torch.tensor([0.3331, 0.3245, 0.3051], device=self.device).view(1, 3, 1, 1)
        MC_STD = torch.tensor([0.2439, 0.2493, 0.2873], device=self.device).view(1, 3, 1, 1)
        
        frames = (frames - MC_MEAN) / MC_STD
        features = self.model.forward_image_features(frames)
        
        return features
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_sim_pos = 0
        total_sim_neg = 0
        total_sim_task = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # 前向传播
            anchor_features = self._encode_frames(anchor)
            positive_features = self._encode_frames(positive)
            negative_features = self._encode_frames(negative)
            
            # 计算损失（包含任务对比）
            loss, sim_pos, sim_neg, sim_task = self.criterion(
                anchor_features,
                positive_features,
                negative_features,
                self.task_features  # ← 任务文本特征
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_sim_pos += sim_pos.item()
            total_sim_neg += sim_neg.item()
            total_sim_task += sim_task.item()
            n_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sim+': f'{sim_pos.item():.3f}',
                'sim-': f'{sim_neg.item():.3f}',
                'task': f'{sim_task.item():.3f}'  # ← 新增
            })
        
        stats = {
            'loss': total_loss / n_batches,
            'sim_pos': total_sim_pos / n_batches,
            'sim_neg': total_sim_neg / n_batches,
            'sim_task': total_sim_task / n_batches,  # ← 新增
            'margin': (total_sim_pos - total_sim_neg) / n_batches
        }
        
        return stats
    
    def save_checkpoint(self, output_path, epoch, stats):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': stats,
            'config': {
                'base_model_path': self.base_model_path,
                'task_prompt': self.task_prompt,  # ← 保存prompt
                'freeze_ratio': self.freeze_ratio,
                'learning_rate': self.learning_rate,
                'task_weight': self.task_weight,
            }
        }
        
        torch.save(checkpoint, output_path)
        print(f"✓ 检查点已保存: {output_path}")

