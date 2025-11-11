# 方案B：多语言MineCLIP适配实施方案

> **目标**: 训练对齐层，让中文指令直接映射到MineCLIP空间  
> **前提**: 方案A（翻译桥接）已实现并验证  
> **优势**: 消除翻译依赖，提升中文理解质量和推理速度  
>
> **设计日期**: 2025-11-10

---

## 📊 现状分析

### 1.1 已有基础设施（方案A）

```
✅ 评估框架完整
   ├─ EvaluationFramework (任务调度)
   ├─ STEVE1Evaluator (执行器)
   ├─ TaskLoader (任务加载)
   ├─ ReportGenerator (报告生成)
   └─ Metrics (评估指标)

✅ 中文支持模块
   ├─ ChineseTranslator (翻译器)
   ├─ chinese_terms.json (100+术语)
   └─ 自动中英文检测

✅ 数据资源
   ├─ eval_tasks.yaml (多任务配置)
   ├─ 中英文对照术语 (100对)
   └─ 评估基线数据

✅ 模型和环境
   ├─ MineCLIP (原生英文)
   ├─ STEVE-1 (VPT + 权重)
   ├─ Prior (VAE)
   └─ MineRLHarvestEnv (自定义环境)
```

### 1.2 当前架构流程

```python
# 方案A: 翻译桥接
中文指令 "砍树"
    ↓
ChineseTranslator (翻译)
    ↓
英文 "chop tree"
    ↓
MineCLIP.encode_text()
    ↓
512维嵌入
    ↓
STEVE-1策略 → 动作
```

### 1.3 方案B目标架构

```python
# 方案B: 对齐层
中文指令 "砍树"                      英文指令 "chop tree"
    ↓                                    ↓
Chinese-CLIP                          MineCLIP
    ↓                                    ↓
512维中文嵌入                         512维英文嵌入
    ↓                                    ↓
对齐层 (Alignment Layer) ─────→   MineCLIP空间
    ↓
512维对齐嵌入
    ↓
STEVE-1策略 → 动作

训练目标:
  loss = ||对齐层(Chinese-CLIP("砍树")) - MineCLIP("chop tree")||²
```

---

## 🏗️ 核心组件设计

### 2.1 MultilingualMineCLIP 类

**文件**: `src/models/multilingual_mineclip.py`（新建）

```python
"""
多语言MineCLIP适配器
Multilingual MineCLIP Adapter for Chinese-English Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
import logging

from transformers import ChineseCLIPModel, ChineseCLIPProcessor

logger = logging.getLogger(__name__)


class AlignmentLayer(nn.Module):
    """
    对齐层: Chinese-CLIP嵌入 → MineCLIP空间
    
    架构设计:
        - 输入: 512维 (Chinese-CLIP 输出)
        - 输出: 512维 (MineCLIP 空间)
        - 结构: MLP (2层 + ReLU + Dropout)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化为接近恒等映射（便于训练初期）
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)
        nn.init.zeros_(self.net[3].bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 512] Chinese-CLIP嵌入
        Returns:
            aligned: [batch_size, 512] 对齐后的嵌入
        """
        return self.net(x)


class MultilingualMineCLIP:
    """
    多语言MineCLIP适配器
    
    职责:
    1. 加载和管理 MineCLIP (英文) 和 Chinese-CLIP (中文)
    2. 提供统一的 encode_text() 接口（自动检测语言）
    3. 使用对齐层映射中文嵌入到MineCLIP空间
    
    使用示例:
        model = MultilingualMineCLIP(
            mineclip_config="data/weights/mineclip/config.pth",
            alignment_path="data/weights/alignment_layer.pth"
        )
        
        # 自动检测语言
        embed_zh = model.encode_text("砍树")
        embed_en = model.encode_text("chop tree")
        
        # 两个嵌入都在MineCLIP空间，可以直接用于STEVE-1
    """
    
    def __init__(
        self,
        mineclip_model=None,  # 如果已加载则传入
        mineclip_config: Optional[str] = None,
        chinese_clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        alignment_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化多语言MineCLIP
        
        Args:
            mineclip_model: 已加载的MineCLIP模型（优先使用，避免重复加载）
            mineclip_config: MineCLIP配置路径（如果mineclip_model=None）
            chinese_clip_model: Chinese-CLIP模型名称或路径
            alignment_path: 对齐层权重路径（如果已训练）
            device: 设备
        """
        self.device = device
        
        # 1. 加载MineCLIP（英文编码器）
        if mineclip_model is not None:
            logger.info("使用已加载的MineCLIP模型")
            self.mineclip = mineclip_model
        else:
            logger.info(f"从配置加载MineCLIP: {mineclip_config}")
            from steve1.mineclip_code.load_mineclip import load
            self.mineclip = load(mineclip_config, device=device)
        
        # 2. 加载Chinese-CLIP（中文编码器）
        logger.info(f"加载Chinese-CLIP: {chinese_clip_model}")
        self.chinese_clip = ChineseCLIPModel.from_pretrained(
            chinese_clip_model,
            cache_dir="data/huggingface_cache"  # 统一缓存目录
        ).to(device)
        self.chinese_clip.eval()  # 固定参数
        
        self.chinese_processor = ChineseCLIPProcessor.from_pretrained(
            chinese_clip_model,
            cache_dir="data/huggingface_cache"
        )
        
        # 3. 创建对齐层
        self.alignment_layer = AlignmentLayer().to(device)
        
        # 4. 加载对齐层权重（如果已训练）
        if alignment_path:
            logger.info(f"加载对齐层权重: {alignment_path}")
            self.alignment_layer.load_state_dict(
                torch.load(alignment_path, map_location=device)
            )
            self.alignment_layer.eval()
        else:
            logger.warning("未加载对齐层权重，使用随机初始化（需要训练）")
    
    def encode_text(
        self,
        text: str,
        language: Literal['auto', 'zh', 'en'] = 'auto',
        return_numpy: bool = True
    ):
        """
        编码文本为MineCLIP嵌入
        
        Args:
            text: 文本指令（中文或英文）
            language: 语言
                - 'auto': 自动检测
                - 'zh': 强制中文
                - 'en': 强制英文
            return_numpy: 是否返回numpy数组（STEVE-1需要）
        
        Returns:
            embed: [512] 维MineCLIP嵌入
        """
        # 1. 语言检测
        if language == 'auto':
            language = self._detect_language(text)
        
        # 2. 编码
        with torch.no_grad():
            if language == 'en':
                # 英文: 直接用MineCLIP
                embed = self._encode_english(text)
            else:  # language == 'zh'
                # 中文: Chinese-CLIP + 对齐层
                embed = self._encode_chinese(text)
        
        # 3. 转换格式
        if return_numpy:
            return embed.cpu().numpy()
        else:
            return embed
    
    def _encode_english(self, text: str) -> torch.Tensor:
        """使用MineCLIP编码英文"""
        return self.mineclip.encode_text(text)
    
    def _encode_chinese(self, text: str) -> torch.Tensor:
        """使用Chinese-CLIP + 对齐层编码中文"""
        # 1. Chinese-CLIP编码
        inputs = self.chinese_processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # 获取文本嵌入
        outputs = self.chinese_clip.get_text_features(**inputs)
        zh_embed = outputs / outputs.norm(dim=-1, keepdim=True)  # 归一化
        
        # 2. 对齐层映射到MineCLIP空间
        aligned_embed = self.alignment_layer(zh_embed)
        
        # 3. 归一化（保持和MineCLIP一致）
        aligned_embed = aligned_embed / aligned_embed.norm(dim=-1, keepdim=True)
        
        return aligned_embed.squeeze(0)  # [512]
    
    def _detect_language(self, text: str) -> Literal['zh', 'en']:
        """
        自动检测语言
        
        规则:
            - 包含中文字符 → 'zh'
            - 否则 → 'en'
        """
        import re
        # 检测是否包含中文字符
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        else:
            return 'en'
    
    def encode_image(self, image):
        """
        编码图像（使用MineCLIP视觉编码器）
        
        注意: 视觉编码器保持不变，只有文本编码器需要多语言支持
        """
        return self.mineclip.encode_image(image)
```

---

### 2.2 对齐层训练器

**文件**: `src/training/alignment_trainer.py`（新建）

```python
"""
对齐层训练器
Alignment Layer Trainer for Multilingual MineCLIP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class ChineseEnglishPairDataset(Dataset):
    """
    中英文对照数据集
    
    数据格式:
        [
            {"zh": "砍树", "en": "chop tree"},
            {"zh": "挖矿", "en": "mine"},
            ...
        ]
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: JSON文件路径（中英文对照）
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"加载数据集: {len(self.data)} 对")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['zh'], item['en']


class AlignmentTrainer:
    """
    对齐层训练器
    
    训练目标:
        让 Align(Chinese-CLIP("砍树")) ≈ MineCLIP("chop tree")
    
    损失函数:
        L2距离: ||aligned_embed - target_embed||²
        余弦相似度: 1 - cosine_similarity(aligned, target)
    """
    
    def __init__(
        self,
        model,  # MultilingualMineCLIP实例
        train_data_path: str,
        val_data_path: Optional[str] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化训练器
        
        Args:
            model: MultilingualMineCLIP实例
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            batch_size: 批大小
            learning_rate: 学习率
            device: 设备
        """
        self.model = model
        self.device = device
        
        # 数据集
        self.train_dataset = ChineseEnglishPairDataset(train_data_path)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        if val_data_path:
            self.val_dataset = ChineseEnglishPairDataset(val_data_path)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        else:
            self.val_loader = None
        
        # 优化器（只优化对齐层）
        self.optimizer = optim.AdamW(
            model.alignment_layer.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # epochs
            eta_min=1e-6
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.alignment_layer.train()
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cos = 0.0
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for zh_texts, en_texts in pbar:
                # 1. 编码中文（Chinese-CLIP + 未训练的对齐层）
                zh_embeds = []
                for zh_text in zh_texts:
                    # 需要梯度，用于反向传播
                    inputs = self.model.chinese_processor(
                        text=[zh_text],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.chinese_clip.get_text_features(**inputs)
                        zh_embed = outputs / outputs.norm(dim=-1, keepdim=True)
                    
                    zh_embeds.append(zh_embed)
                
                zh_embeds = torch.cat(zh_embeds, dim=0)  # [batch, 512]
                
                # 2. 对齐层（需要梯度）
                aligned_embeds = self.model.alignment_layer(zh_embeds)
                
                # 归一化
                aligned_embeds = aligned_embeds / aligned_embeds.norm(dim=-1, keepdim=True)
                
                # 3. 编码英文（MineCLIP，作为目标）
                en_embeds = []
                with torch.no_grad():
                    for en_text in en_texts:
                        en_embed = self.model.mineclip.encode_text(en_text)
                        en_embeds.append(en_embed)
                
                en_embeds = torch.stack(en_embeds, dim=0)  # [batch, 512]
                
                # 4. 计算损失
                mse = self.mse_loss(aligned_embeds, en_embeds)
                cos = 1 - self.cos_sim(aligned_embeds, en_embeds).mean()
                
                # 组合损失（MSE主导 + 余弦相似度辅助）
                loss = mse + 0.1 * cos
                
                # 5. 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.alignment_layer.parameters(),
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # 6. 记录
                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_cos += cos.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{mse.item():.4f}',
                    'cos': f'{cos.item():.4f}'
                })
        
        # 平均损失
        n_batches = len(self.train_loader)
        metrics = {
            'train_loss': epoch_loss / n_batches,
            'train_mse': epoch_mse / n_batches,
            'train_cos': epoch_cos / n_batches
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.alignment_layer.eval()
        
        val_loss = 0.0
        val_mse = 0.0
        val_cos = 0.0
        
        with torch.no_grad():
            for zh_texts, en_texts in tqdm(self.val_loader, desc="Validation"):
                # 编码中文
                zh_embeds = []
                for zh_text in zh_texts:
                    embed = self.model.encode_text(zh_text, language='zh', return_numpy=False)
                    zh_embeds.append(embed)
                zh_embeds = torch.stack(zh_embeds, dim=0)
                
                # 编码英文
                en_embeds = []
                for en_text in en_texts:
                    embed = self.model.encode_text(en_text, language='en', return_numpy=False)
                    en_embeds.append(embed)
                en_embeds = torch.stack(en_embeds, dim=0)
                
                # 损失
                mse = self.mse_loss(zh_embeds, en_embeds)
                cos = 1 - self.cos_sim(zh_embeds, en_embeds).mean()
                loss = mse + 0.1 * cos
                
                val_loss += loss.item()
                val_mse += mse.item()
                val_cos += cos.item()
        
        n_batches = len(self.val_loader)
        metrics = {
            'val_loss': val_loss / n_batches,
            'val_mse': val_mse / n_batches,
            'val_cos': val_cos / n_batches
        }
        
        return metrics
    
    def train(
        self,
        epochs: int,
        save_dir: str = "data/weights/alignment",
        save_every: int = 10
    ):
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            save_dir: 权重保存目录
            save_every: 每N个epoch保存一次
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始训练对齐层，共{epochs}轮")
        logger.info(f"训练数据: {len(self.train_dataset)} 对")
        if self.val_loader:
            logger.info(f"验证数据: {len(self.val_dataset)} 对")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])
            
            # 学习率调度
            self.scheduler.step()
            
            # 日志
            logger.info(
                f"Epoch {self.current_epoch}/{epochs} - "
                f"train_loss: {train_metrics['train_loss']:.4f}, "
                f"train_mse: {train_metrics['train_mse']:.4f}, "
                f"train_cos: {train_metrics['train_cos']:.4f}"
            )
            
            if val_metrics:
                logger.info(
                    f"  val_loss: {val_metrics['val_loss']:.4f}, "
                    f"val_mse: {val_metrics['val_mse']:.4f}, "
                    f"val_cos: {val_metrics['val_cos']:.4f}"
                )
                
                # 保存最佳模型
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    best_path = save_dir / "alignment_best.pth"
                    torch.save(
                        self.model.alignment_layer.state_dict(),
                        best_path
                    )
                    logger.info(f"  保存最佳模型: {best_path}")
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                checkpoint_path = save_dir / f"alignment_epoch_{epoch+1}.pth"
                torch.save(
                    self.model.alignment_layer.state_dict(),
                    checkpoint_path
                )
                logger.info(f"  保存checkpoint: {checkpoint_path}")
        
        # 保存最终模型
        final_path = save_dir / "alignment_final.pth"
        torch.save(
            self.model.alignment_layer.state_dict(),
            final_path
        )
        logger.info(f"训练完成，保存最终模型: {final_path}")
        
        # 保存训练曲线
        self._save_training_curves(save_dir)
    
    def _save_training_curves(self, save_dir: Path):
        """保存训练曲线"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            ax.plot(self.val_losses, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Alignment Layer Training Curves')
        ax.legend()
        ax.grid(True)
        
        curve_path = save_dir / "training_curves.png"
        plt.savefig(curve_path)
        plt.close()
        
        logger.info(f"保存训练曲线: {curve_path}")
```

---

## 📁 文件结构与代码适配

### 3.1 新增文件

```
src/
├── models/                          # 新增目录
│   ├── __init__.py
│   └── multilingual_mineclip.py     # 核心: MultilingualMineCLIP
│
├── training/
│   └── alignment_trainer.py         # 新增: 对齐层训练器
│
└── scripts/
    ├── train_alignment.py           # 新增: 训练脚本
    └── evaluate_alignment.py        # 新增: 评估脚本

data/
├── alignment_pairs/                 # 新增目录
│   ├── train.json                   # 训练数据（中英文对照）
│   ├── val.json                     # 验证数据
│   └── README.md                    # 数据格式说明
│
└── weights/
    └── alignment/                   # 新增目录
        ├── alignment_best.pth       # 最佳权重
        ├── alignment_final.pth      # 最终权重
        └── training_curves.png      # 训练曲线
```

### 3.2 需要修改的现有文件

#### 3.2.1 `src/evaluation/steve1_evaluator.py`

**修改点**: 支持使用MultilingualMineCLIP替代翻译

```python
# 原有代码
from ..translation.translator import ChineseTranslator

# 新增导入
from ..models.multilingual_mineclip import MultilingualMineCLIP

class STEVE1Evaluator:
    def __init__(
        self,
        # ... 其他参数
        use_multilingual: bool = False,  # 新增参数
        alignment_path: Optional[str] = None  # 新增参数
    ):
        # ... 原有代码
        
        # 修改: 根据配置选择翻译器或多语言模型
        if use_multilingual:
            logger.info("使用多语言MineCLIP（对齐层方案）")
            # 注意: MineCLIP已在_lazy_load_models中加载
            # 创建多语言适配器时复用已加载的MineCLIP
            self._multilingual_mineclip = None  # 延迟加载
            self.alignment_path = alignment_path
            self.translator = None  # 不使用翻译器
        else:
            logger.info("使用翻译桥接方案")
            self.translator = ChineseTranslator(
                term_dict_path="data/chinese_terms.json",
                method="term_dict"
            )
            self._multilingual_mineclip = None
    
    def _lazy_load_models(self):
        """延迟加载模型"""
        if self._agent is None or self._mineclip is None:
            logger.info("延迟加载 STEVE-1 Agent 和 MineCLIP...")
            self._agent, self._mineclip, _ = load_mineclip_agent_env(
                # ... 参数
            )
        
        # 如果使用多语言模式，加载MultilingualMineCLIP
        if self._multilingual_mineclip is None and self.translator is None:
            logger.info("加载 MultilingualMineCLIP...")
            from ..models.multilingual_mineclip import MultilingualMineCLIP
            self._multilingual_mineclip = MultilingualMineCLIP(
                mineclip_model=self._mineclip,  # 复用已加载的MineCLIP
                alignment_path=self.alignment_path
            )
    
    def _run_single_trial(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        trial_idx: int,
        n_trials: int
    ) -> TrialResult:
        """运行单次试验"""
        # ... 前面代码不变
        
        # 修改: 根据模式选择编码方式
        if self._multilingual_mineclip:
            # 方案B: 多语言MineCLIP（自动检测语言）
            logger.debug(f"  使用多语言MineCLIP编码: '{instruction}'")
            with th.no_grad():
                prompt_embed = self._multilingual_mineclip.encode_text(
                    instruction,
                    language='auto',
                    return_numpy=False  # 返回Tensor
                )
                # 使用Prior
                prompt_embed = get_prior_embed(
                    instruction,
                    self._multilingual_mineclip,  # 传入多语言模型
                    self._prior,
                    DEVICE
                )
                prompt_embed_np = prompt_embed.cpu().numpy()
        else:
            # 方案A: 翻译桥接
            if self._is_chinese(instruction):
                translated_instruction = self.translator.translate(instruction)
                logger.debug(f"  翻译: {instruction} → {translated_instruction}")
                instruction_to_encode = translated_instruction
            else:
                instruction_to_encode = instruction
            
            logger.debug(f"  使用 MineCLIP 编码: '{instruction_to_encode}'")
            with th.no_grad():
                prompt_embed = get_prior_embed(
                    instruction_to_encode,
                    self._mineclip,
                    self._prior,
                    DEVICE
                )
                prompt_embed_np = prompt_embed.cpu().numpy()
        
        # ... 后续代码不变
```

#### 3.2.2 `src/evaluation/eval_framework.py`

**修改点**: 添加配置参数传递

```python
@dataclass
class EvaluationConfig:
    # ... 原有参数
    
    # 新增: 多语言支持配置
    use_multilingual: bool = False
    alignment_path: Optional[str] = None

class EvaluationFramework:
    def __init__(self, config: Optional[EvaluationConfig] = None, ...):
        # ...
        
        if evaluator is None:
            self.evaluator = STEVE1Evaluator(
                # ... 原有参数
                use_multilingual=self.config.use_multilingual,  # 新增
                alignment_path=self.config.alignment_path  # 新增
            )
```

#### 3.2.3 `src/utils/steve1_mineclip_agent_env_utils.py`

**修改点**: 可选支持MultilingualMineCLIP（如果需要在其他地方使用）

```python
def load_multilingual_mineclip(
    mineclip_config,
    alignment_path: Optional[str] = None,
    device=DEVICE
):
    """
    加载多语言MineCLIP
    
    Args:
        mineclip_config: MineCLIP配置路径
        alignment_path: 对齐层权重路径
        device: 设备
    
    Returns:
        MultilingualMineCLIP实例
    """
    from ..models.multilingual_mineclip import MultilingualMineCLIP
    
    return MultilingualMineCLIP(
        mineclip_config=mineclip_config,
        alignment_path=alignment_path,
        device=device
    )
```

---

## 🎯 数据准备方案

### 4.1 数据来源

```python
# 1. 从现有术语词典扩展
from data/chinese_terms.json → data/alignment_pairs/

策略:
  - 展开嵌套结构: {"基础动作": {"砍树": "chop tree"}} → {"zh": "砍树", "en": "chop tree"}
  - 添加同义词变体: "伐木" → "chop tree", "获取木头" → "chop tree"
  - 当前可生成: ~200对

# 2. MineDojo任务集
from config/eval_tasks.yaml → 提取中英文指令对

策略:
  - 每个任务的 zh 和 en 指令配对
  - 不同任务变体
  - 可生成: ~50-100对

# 3. 组合指令生成
模板方法:
  - "砍 + 树" → "chop tree"
  - "制作 + 木镐" → "craft wooden pickaxe"
  - "找到 + 洞穴" → "find cave"
  - 可生成: ~500-1000对

# 4. 人工扩充（可选）
  - Minecraft论坛/社区常用表述
  - B站/抖音视频标题/评论
  - 目标: 1000-2000对

总计: 2000-3000对（足够训练）
训练集:验证集 = 8:2
```

### 4.2 数据格式

```json
// data/alignment_pairs/train.json
[
  {
    "zh": "砍树",
    "en": "chop tree",
    "category": "basic_action",
    "source": "term_dict"
  },
  {
    "zh": "伐木",
    "en": "chop tree",
    "category": "basic_action",
    "source": "synonym"
  },
  {
    "zh": "制作木镐",
    "en": "craft wooden pickaxe",
    "category": "composite_task",
    "source": "generated"
  },
  {
    "zh": "找到洞穴并进入",
    "en": "find cave and enter",
    "category": "complex_task",
    "source": "eval_tasks"
  }
]
```

### 4.3 数据生成脚本

**文件**: `scripts/generate_alignment_data.py`（新建）

```python
"""
生成对齐层训练数据
从现有资源生成中英文对照数据集
"""

import json
from pathlib import Path
from typing import List, Dict
import yaml
import random


def load_term_dict(path: str = "data/chinese_terms.json") -> List[Dict]:
    """从术语词典生成pairs"""
    with open(path, 'r', encoding='utf-8') as f:
        terms = json.load(f)
    
    pairs = []
    for category, items in terms.items():
        if isinstance(items, dict):
            for zh, en in items.items():
                pairs.append({
                    "zh": zh,
                    "en": en,
                    "category": category,
                    "source": "term_dict"
                })
    
    return pairs


def load_eval_tasks(path: str = "config/eval_tasks.yaml") -> List[Dict]:
    """从评估任务生成pairs"""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    pairs = []
    for task in config.get('tasks', []):
        en = task.get('instructions', {}).get('en')
        zh_list = task.get('instructions', {}).get('zh', [])
        
        if en and zh_list:
            for zh in zh_list:
                pairs.append({
                    "zh": zh,
                    "en": en,
                    "category": task.get('category', 'task'),
                    "source": "eval_tasks",
                    "task_id": task.get('task_id')
                })
    
    return pairs


def generate_composite_pairs() -> List[Dict]:
    """生成组合指令pairs"""
    actions = [
        ("砍", "chop"),
        ("挖", "mine"),
        ("制作", "craft"),
        ("建造", "build"),
        ("找到", "find"),
        ("攻击", "attack")
    ]
    
    objects = [
        ("树", "tree"),
        ("木头", "wood"),
        ("石头", "stone"),
        ("矿石", "ore"),
        ("房子", "house"),
        ("工作台", "crafting table")
    ]
    
    pairs = []
    for zh_action, en_action in actions:
        for zh_obj, en_obj in objects:
            pairs.append({
                "zh": f"{zh_action}{zh_obj}",
                "en": f"{en_action} {en_obj}",
                "category": "composite",
                "source": "generated"
            })
    
    return pairs


def main():
    """生成并保存数据集"""
    # 收集所有pairs
    all_pairs = []
    
    print("加载术语词典...")
    all_pairs.extend(load_term_dict())
    
    print("加载评估任务...")
    all_pairs.extend(load_eval_tasks())
    
    print("生成组合指令...")
    all_pairs.extend(generate_composite_pairs())
    
    print(f"总计: {len(all_pairs)} 对")
    
    # 去重（基于zh-en对）
    unique_pairs = []
    seen = set()
    for pair in all_pairs:
        key = (pair['zh'], pair['en'])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    
    print(f"去重后: {len(unique_pairs)} 对")
    
    # 打乱顺序
    random.shuffle(unique_pairs)
    
    # 分割训练集和验证集 (8:2)
    split_idx = int(len(unique_pairs) * 0.8)
    train_pairs = unique_pairs[:split_idx]
    val_pairs = unique_pairs[split_idx:]
    
    # 保存
    output_dir = Path("data/alignment_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"训练集: {len(train_pairs)} 对 → {output_dir / 'train.json'}")
    print(f"验证集: {len(val_pairs)} 对 → {output_dir / 'val.json'}")


if __name__ == "__main__":
    main()
```

---

## 🚀 训练流程

### 5.1 训练脚本

**文件**: `scripts/train_alignment.py`（新建）

```python
"""
训练对齐层
Train Alignment Layer for Multilingual MineCLIP
"""

import argparse
import logging
from pathlib import Path

import torch

from src.models.multilingual_mineclip import MultilingualMineCLIP
from src.training.alignment_trainer import AlignmentTrainer
from steve1.config import MINECLIP_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="训练对齐层")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, 
                        default="data/alignment_pairs/train.json",
                        help="训练数据路径")
    parser.add_argument("--val_data", type=str,
                        default="data/alignment_pairs/val.json",
                        help="验证数据路径")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    
    # 模型参数
    parser.add_argument("--chinese_clip_model", type=str,
                        default="OFA-Sys/chinese-clip-vit-base-patch16",
                        help="Chinese-CLIP模型")
    
    # 保存参数
    parser.add_argument("--save_dir", type=str,
                        default="data/weights/alignment",
                        help="权重保存目录")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每N个epoch保存一次")
    
    # 设备参数
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not Path(args.train_data).exists():
        logger.error(f"训练数据不存在: {args.train_data}")
        logger.info("请先运行: python scripts/generate_alignment_data.py")
        return
    
    # 创建多语言MineCLIP
    logger.info("初始化MultilingualMineCLIP...")
    model = MultilingualMineCLIP(
        mineclip_config=MINECLIP_CONFIG,
        chinese_clip_model=args.chinese_clip_model,
        alignment_path=None,  # 从头训练
        device=args.device
    )
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = AlignmentTrainer(
        model=model,
        train_data_path=args.train_data,
        val_data_path=args.val_data if Path(args.val_data).exists() else None,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=args.save_every
    )
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main()
```

### 5.2 使用方法

```bash
# 1. 生成训练数据
python scripts/generate_alignment_data.py

# 2. 训练对齐层
python scripts/train_alignment.py \
    --train_data data/alignment_pairs/train.json \
    --val_data data/alignment_pairs/val.json \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir data/weights/alignment

# 3. 监控训练（可选）
# 查看训练曲线
ls data/weights/alignment/training_curves.png
```

---

## 📊 评估对比方案

### 6.1 对比评估脚本

**文件**: `scripts/evaluate_alignment.py`（新建）

```python
"""
评估对齐层性能
Compare Alignment Layer vs Translation Bridge
"""

import argparse
import logging
from pathlib import Path

from src.evaluation.eval_framework import EvaluationFramework, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="对比评估：对齐层 vs 翻译桥接")
    
    parser.add_argument("--alignment_path", type=str,
                        default="data/weights/alignment/alignment_best.pth",
                        help="对齐层权重路径")
    parser.add_argument("--task_config", type=str,
                        default="config/eval_tasks.yaml",
                        help="任务配置文件")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="每个任务的试验次数")
    
    args = parser.parse_args()
    
    # 检查权重文件
    if not Path(args.alignment_path).exists():
        logger.error(f"对齐层权重不存在: {args.alignment_path}")
        logger.info("请先运行训练: python scripts/train_alignment.py")
        return
    
    # ===== 方案A: 翻译桥接 =====
    logger.info("="*80)
    logger.info("评估方案A: 翻译桥接")
    logger.info("="*80)
    
    config_translation = EvaluationConfig(
        task_config_path=args.task_config,
        n_trials=args.n_trials,
        use_multilingual=False,  # 使用翻译
        results_dir="results/evaluation/translation"
    )
    
    framework_translation = EvaluationFramework(config=config_translation)
    results_translation = framework_translation.evaluate_all_tasks()
    
    # ===== 方案B: 对齐层 =====
    logger.info("="*80)
    logger.info("评估方案B: 对齐层")
    logger.info("="*80)
    
    config_alignment = EvaluationConfig(
        task_config_path=args.task_config,
        n_trials=args.n_trials,
        use_multilingual=True,  # 使用对齐层
        alignment_path=args.alignment_path,
        results_dir="results/evaluation/alignment"
    )
    
    framework_alignment = EvaluationFramework(config=config_alignment)
    results_alignment = framework_alignment.evaluate_all_tasks()
    
    # ===== 对比分析 =====
    logger.info("="*80)
    logger.info("对比结果")
    logger.info("="*80)
    
    # 计算平均成功率
    def calc_avg_success(results):
        en_rates = [r.success_rate for r in results if r.language == 'en']
        zh_rates = [r.success_rate for r in results if r.language == 'zh']
        return {
            'en_avg': sum(en_rates) / len(en_rates) if en_rates else 0,
            'zh_avg': sum(zh_rates) / len(zh_rates) if zh_rates else 0
        }
    
    stats_translation = calc_avg_success(results_translation)
    stats_alignment = calc_avg_success(results_alignment)
    
    logger.info(f"翻译桥接: EN={stats_translation['en_avg']:.1%}, ZH={stats_translation['zh_avg']:.1%}, Gap={(stats_translation['en_avg']-stats_translation['zh_avg']):.1%}")
    logger.info(f"对齐层:   EN={stats_alignment['en_avg']:.1%}, ZH={stats_alignment['zh_avg']:.1%}, Gap={(stats_alignment['en_avg']-stats_alignment['zh_avg']):.1%}")
    
    # 改进分析
    zh_improvement = stats_alignment['zh_avg'] - stats_translation['zh_avg']
    gap_improvement = (stats_translation['en_avg'] - stats_translation['zh_avg']) - (stats_alignment['en_avg'] - stats_alignment['zh_avg'])
    
    logger.info(f"中文成功率提升: {zh_improvement:+.1%}")
    logger.info(f"中英文Gap缩小: {gap_improvement:+.1%}")
    
    if zh_improvement > 0:
        logger.info("✅ 对齐层方案优于翻译桥接")
    else:
        logger.info("⚠️  对齐层方案未达预期，建议检查训练")


if __name__ == "__main__":
    main()
```

---

## 📅 实施时间表

### 第1周：环境准备和数据生成

```
Day 1-2: 代码框架搭建
  ✓ 创建 src/models/ 目录
  ✓ 实现 MultilingualMineCLIP 类
  ✓ 实现 AlignmentLayer 类
  ✓ 测试Chinese-CLIP加载

Day 3-4: 训练器实现
  ✓ 实现 AlignmentTrainer 类
  ✓ 实现数据加载
  ✓ 测试训练循环

Day 5-7: 数据准备
  ✓ 实现数据生成脚本
  ✓ 从术语词典提取pairs
  ✓ 从评估任务提取pairs
  ✓ 生成组合指令
  ✓ 数据验证和清洗
```

### 第2周：训练和调优

```
Day 8-10: 模型训练
  ✓ 运行完整训练（100 epochs）
  ✓ 监控训练曲线
  ✓ 调整超参数

Day 11-12: 评估验证
  ✓ 实现对比评估脚本
  ✓ 运行方案A vs 方案B对比
  ✓ 分析性能差异

Day 13-14: 优化迭代
  ✓ 根据评估结果调整
  ✓ 补充数据（如有必要）
  ✓ 重新训练
```

### 第3-4周：集成和部署（可选）

```
Day 15-18: 系统集成
  ✓ 修改评估框架
  ✓ 更新配置管理
  ✓ 完整测试

Day 19-21: 文档和总结
  ✓ 编写使用文档
  ✓ 更新项目README
  ✓ 性能报告

Day 22-28: 持续优化
  ✓ 收集失败case
  ✓ 补充训练数据
  ✓ 微调模型
```

---

## 🎯 成功标准

### 训练阶段目标

```
必达指标:
  ✅ 训练损失收敛（<0.05）
  ✅ 验证损失稳定
  ✅ 余弦相似度 > 0.90

期望指标:
  ⭐ 训练损失 < 0.01
  ⭐ 余弦相似度 > 0.95
  ⭐ 中英文嵌入L2距离 < 0.1
```

### 评估阶段目标

```
必达指标（vs 翻译方案）:
  ✅ 中文成功率提升 > 0%
  ✅ 中英文gap缩小 > 0%
  ✅ 推理速度提升 > 20%（消除翻译延迟）

期望指标:
  ⭐ 中文成功率提升 > 5%
  ⭐ 中英文gap < 5%（vs 翻译方案 10-15%）
  ⭐ 基础任务中文成功率 > 85%
```

### 生产部署标准

```
必达指标:
  ✅ 模型稳定（无崩溃）
  ✅ 推理速度可接受（<50ms）
  ✅ 内存占用合理（<4GB）

期望指标:
  ⭐ 支持实时推理（<20ms）
  ⭐ 批处理优化
  ⭐ 可配置fallback（对齐层失败时用翻译）
```

---

## 🔄 方案对比总结

| 维度 | 方案A: 翻译桥接 | 方案B: 对齐层 | 改进 |
|-----|---------------|-------------|-----|
| **实现难度** | ⭐ 简单 | ⭐⭐⭐ 中等 | 需要训练 |
| **开发时间** | 1-2天 | 1-2周 | +10倍 |
| **训练需求** | 无 | 需要（1-3天GPU） | 有成本 |
| **数据需求** | 100术语 | 2000-3000对 | +20倍 |
| **中文理解** | 依赖翻译质量 | 直接语义理解 | ✅ 更准确 |
| **推理速度** | ~150ms | ~30ms | ✅ 快5倍 |
| **成功率gap** | 10-15% | 目标<5% | ✅ 缩小gap |
| **维护成本** | 需维护词典 | 模型固定 | ✅ 更低 |
| **扩展性** | 难扩展 | 易扩展（增加数据重训） | ✅ 更好 |

---

## 💡 关键技术要点

### 1. 为什么只训练对齐层？

```
固定部分（不训练）:
  ✅ MineCLIP视觉编码器 - 已在大规模Minecraft数据上训练
  ✅ MineCLIP文本编码器 - 英文理解能力强
  ✅ Chinese-CLIP - 中文理解能力强

训练部分:
  🎯 对齐层 - 学习中文→MineCLIP空间的映射

优势:
  ⚡ 训练数据需求小（2000对 vs 10万+视频）
  ⚡ 训练时间短（1-3天 vs 数周）
  ⚡ 保持原模型性能（不破坏MineCLIP和Chinese-CLIP）
```

### 2. 损失函数设计

```python
# 组合损失
loss = MSE(aligned, target) + 0.1 * (1 - CosineSim(aligned, target))

原因:
  1. MSE - 保证嵌入数值接近
  2. CosineSim - 保证语义方向一致
  3. 0.1权重 - MSE主导，余弦辅助
```

### 3. 数据增强策略

```python
# 1. 同义词增强
("砍树", "chop tree")
("伐木", "chop tree")  # 同一目标
("获取木头", "chop tree")

# 2. 短语变体
("找到洞穴", "find cave")
("寻找洞穴", "find cave")
("进入洞穴", "find cave")

# 3. 上下文增强
("砍树", "chop tree")
("砍一棵树", "chop a tree")  # 更自然的表述

效果: 提升模型鲁棒性
```

---

## 🚨 潜在风险与应对

### 风险1: Chinese-CLIP嵌入维度不匹配

```
问题: Chinese-CLIP输出可能不是512维
应对:
  ✓ 检查模型配置
  ✓ 如果不是512维，修改对齐层输入维度
  ✓ 或使用投影层: Linear(chinese_dim, 512)
```

### 风险2: 训练数据不足

```
问题: 2000对可能不够
应对:
  ✓ 优先使用现有数据训练baseline
  ✓ 根据评估结果决定是否需要更多数据
  ✓ 可使用数据增强（回译、同义词替换）
```

### 风险3: 对齐层过拟合

```
问题: 验证集loss不下降
应对:
  ✓ 增加Dropout
  ✓ 使用更简单的网络（单层Linear）
  ✓ Early stopping
  ✓ 数据增强
```

### 风险4: 性能未达预期

```
问题: 对齐层方案不如翻译
应对:
  ✓ 分析失败case
  ✓ 检查训练曲线
  ✓ 尝试不同的网络结构
  ✓ Fallback到翻译方案
```

---

## ✅ 检查清单

### 开始前检查

- [ ] 方案A（翻译桥接）已完成并验证
- [ ] 评估框架可正常运行
- [ ] GPU可用（建议，CPU也可以但较慢）
- [ ] Chinese-CLIP模型可正常下载

### 实施中检查

- [ ] MultilingualMineCLIP类实现正确
- [ ] AlignmentTrainer类实现正确
- [ ] 训练数据生成成功（>2000对）
- [ ] 训练损失正常收敛
- [ ] 验证集性能合理

### 完成后检查

- [ ] 对比评估完成
- [ ] 性能报告生成
- [ ] 代码文档完整
- [ ] 权重文件已保存
- [ ] 使用指南已更新

---

## 📚 参考资源

**已有代码**:
- `src/evaluation/steve1_evaluator.py` - 评估器（需修改）
- `src/translation/translator.py` - 翻译器（参考语言检测）
- `src/utils/steve1_mineclip_agent_env_utils.py` - MineCLIP加载

**外部资源**:
- Chinese-CLIP: https://github.com/OFA-Sys/Chinese-CLIP
- MineCLIP论文: https://arxiv.org/abs/2206.08853
- STEVE-1论文: https://arxiv.org/abs/2306.00937

**项目文档**:
- `docs/design/CHINESE_AIMC_AGENT_TECHNICAL_PLAN.md` - 整体方案
- `docs/guides/STEVE1_EVALUATION_GUIDE.md` - 评估方法
- `docs/design/UNIVERSAL_MINECLIP_STRATEGY.md` - MineCLIP通用策略

---

**方案版本**: v1.0  
**设计日期**: 2025-11-10  
**状态**: 待实施  
**预计完成**: 2周

**下一步**: 
1. 审阅本设计文档
2. 确认实施计划
3. 开始代码实现

