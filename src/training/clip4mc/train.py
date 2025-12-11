#!/usr/bin/env python3
"""
CLIP4MC 简化训练脚本 (单 GPU 测试版)

基于 https://github.com/PKU-RL/CLIP4MC 的训练代码简化版本，
用于验证数据格式和训练流程。

使用方法:
    python -m src.training.clip4mc.train \
        --data-log data/training/clip4mc/data_log.json \
        --pretrain-model data/weights/vit-b-16-clip/ViT-B-16.pt \
        --epochs 5 --batch-size 1
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Literal
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleClip4MCDataset(Dataset):
    """简化的 CLIP4MC 数据集"""
    
    def __init__(self, data_dirs: List[str], dataset_type: str = "train"):
        self.data_dirs = data_dirs
        self.dataset_type = dataset_type
        
        # Minecraft 图像归一化参数
        self.image_mean = torch.tensor([0.3331, 0.3245, 0.3051]).view(3, 1, 1)
        self.image_std = torch.tensor([0.2439, 0.2493, 0.2873]).view(3, 1, 1)
    
    def __len__(self):
        return len(self.data_dirs)
    
    def __getitem__(self, idx):
        data_dir = Path(self.data_dirs[idx])
        
        # 加载文本
        with open(data_dir / "text_input.pkl", "rb") as f:
            text_data = pickle.load(f)
        
        if isinstance(text_data, dict):
            text_tokens = text_data.get('tokens', text_data)
        else:
            text_tokens = text_data
        
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        # 加载视频
        with open(data_dir / "video_input.pkl", "rb") as f:
            video_data = pickle.load(f)
        
        # (N, H, W, C) -> (N, C, H, W) 并归一化
        video = np.array(video_data)
        if video.ndim == 4 and video.shape[-1] == 3:
            video = video.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # 转换为 float 并归一化到 [0, 1]
        video = torch.tensor(video, dtype=torch.float32)
        if video.max() > 1.0:
            video = video / 255.0
        
        # 标准化
        video = (video - self.image_mean) / self.image_std
        
        # 确保是 16 帧
        if video.shape[0] != 16:
            indices = np.linspace(0, video.shape[0] - 1, 16, dtype=int)
            video = video[indices]
        
        return text_tokens, video


class SimpleCLIP4MC(nn.Module):
    """简化的 CLIP4MC 模型 (用于验证)"""
    
    def __init__(self, pretrain_path: str = None):
        super().__init__()
        
        self.clip_model = None
        self.tokenizer = None
        
        # 尝试从本地文件加载 CLIP
        if pretrain_path and Path(pretrain_path).exists():
            try:
                import open_clip
                # 创建模型结构
                self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16', pretrained=''  # 不下载权重
                )
                # 加载本地权重
                state_dict = torch.load(pretrain_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.clip_model.load_state_dict(state_dict, strict=False)
                self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
                logger.info(f"✓ 从本地加载 CLIP 权重: {pretrain_path}")
            except Exception as e:
                logger.warning(f"无法加载本地权重: {e}")
                self.clip_model = None
        
        # 如果本地加载失败，尝试在线加载
        if self.clip_model is None:
            try:
                import open_clip
                self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16', pretrained='openai'
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
                logger.info("✓ 使用 OpenAI CLIP 预训练权重")
            except Exception as e:
                logger.warning(f"无法加载 open_clip: {e}")
                logger.info("使用简化的随机初始化模型")
        
        # 视频时序聚合
        self.video_adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        编码视频帧
        
        Args:
            video: (B, T, C, H, W)
        
        Returns:
            视频嵌入 (B, 512)
        """
        B, T, C, H, W = video.shape
        
        # 展平 batch 和 time
        video_flat = video.view(B * T, C, H, W)
        
        # 使用 CLIP 视觉编码器
        if self.clip_model is not None:
            with torch.no_grad():
                frame_features = self.clip_model.encode_image(video_flat)
        else:
            # Fallback: 简单线性投影
            frame_features = video_flat.view(B * T, -1)
            if not hasattr(self, '_fallback_proj'):
                in_dim = C * H * W
                self._fallback_proj = nn.Linear(in_dim, 512).to(video_flat.device)
            frame_features = self._fallback_proj(frame_features)
        
        # 重塑为 (B, T, D)
        frame_features = frame_features.view(B, T, -1)
        
        # 时序聚合 (简单平均)
        video_features = frame_features.mean(dim=1)
        
        # 适配器
        video_features = self.video_adapter(video_features)
        
        # L2 归一化
        video_features = F.normalize(video_features, dim=-1)
        
        return video_features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        编码文本
        
        Args:
            text: (B, L) token ids
        
        Returns:
            文本嵌入 (B, 512)
        """
        if self.clip_model is not None:
            text_features = self.clip_model.encode_text(text)
        else:
            # Fallback: 使用嵌入层
            if not hasattr(self, '_text_embed'):
                self._text_embed = nn.Embedding(50000, 512).to(text.device)
            text_embed = self._text_embed(text.clamp(0, 49999))
            text_features = text_embed.mean(dim=1)
        
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def forward(self, text: torch.Tensor, video: torch.Tensor):
        """
        前向传播，计算对比损失
        
        Args:
            text: (B, L) token ids
            video: (B, T, C, H, W)
        
        Returns:
            loss, logits_per_video, logits_per_text
        """
        video_features = self.encode_video(video)
        text_features = self.encode_text(text)
        
        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()
        
        # 对比损失
        batch_size = video.shape[0]
        labels = torch.arange(batch_size, device=video.device)
        
        loss_v = F.cross_entropy(logits_per_video, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_v + loss_t) / 2
        
        return loss, logits_per_video, logits_per_text


def train_epoch(model, dataloader, optimizer, device, epoch, pbar=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (text, video) in enumerate(dataloader):
        text = text.to(device)
        video = video.to(device)
        
        optimizer.zero_grad()
        
        loss, logits_v, logits_t = model(text, video)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if pbar:
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
            pbar.update(1)
        elif batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="CLIP4MC 简化训练")
    
    parser.add_argument(
        "--data-log",
        type=Path,
        default=Path("data/training/clip4mc/data_log.json"),
        help="数据 log 文件"
    )
    
    parser.add_argument(
        "--pretrain-model",
        type=Path,
        default=Path("data/weights/vit-b-16-clip/ViT-B-16.pt"),
        help="预训练 CLIP 权重"
    )
    
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--output", type=Path, default=None, help="模型保存路径")
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    if not args.data_log.exists():
        logger.error(f"数据 log 文件不存在: {args.data_log}")
        return
    
    with open(args.data_log) as f:
        data_log = json.load(f)
    
    train_dirs = data_log.get("train", [])
    test_dirs = data_log.get("test", [])
    
    logger.info(f"训练集: {len(train_dirs)} 个样本")
    logger.info(f"测试集: {len(test_dirs)} 个样本")
    
    if not train_dirs:
        logger.error("没有训练数据")
        return
    
    # 创建数据集和加载器
    train_dataset = SimpleClip4MCDataset(train_dirs, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = SimpleCLIP4MC(str(args.pretrain_model))
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # 训练
    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)
    
    total_steps = args.epochs * len(train_loader)
    
    if tqdm:
        with tqdm(
            total=total_steps,
            desc="🎯 训练进度",
            unit="step",
            position=0,
            leave=True,
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{postfix}] [{elapsed}<{remaining}]'
        ) as pbar:
            for epoch in range(1, args.epochs + 1):
                avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, pbar)
                logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
    else:
        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
            avg_loss = train_epoch(model, train_loader, optimizer, device, epoch)
            logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
    
    # 保存模型
    output_path = args.output or args.data_log.parent / "clip4mc_trained.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
    }, output_path)
    
    logger.info(f"\n模型已保存到: {output_path}")
    logger.info("✓ 训练完成！")


if __name__ == "__main__":
    main()

