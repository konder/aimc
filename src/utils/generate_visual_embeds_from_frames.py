#!/usr/bin/env python
"""
从成功帧图像生成视觉嵌入向量（16帧视频模式）
==================================

读取成功帧图像，按trial目录分组，每个trial的16帧作为一个视频片段，
使用MineCLIP的encode_video生成视觉嵌入，保存为pkl文件。

目录结构:
    data/success_visuals/
      task_id/
        trial1/
          frame_000.png
          ...
          frame_015.png
        trial2/
          frame_000.png
          ...
        visual_embeds.pkl

用法:
    python src/utils/generate_visual_embeds_from_frames.py \
        --frames-dir data/success_visuals \
        --output-report data/success_visuals/generation_report.txt

作者: AI Assistant
日期: 2025-12-02
"""

import os
import sys
import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

import numpy as np
import torch as th
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入MineCLIP
from src.utils.steve1_mineclip_agent_env_utils import load_mineclip_wconfig
from src.utils.device import DEVICE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualEmbedGenerator:
    """视觉嵌入生成器（16帧视频模式）"""
    
    def __init__(self, frames_dir: str):
        """
        初始化生成器
        
        Args:
            frames_dir: 成功帧根目录
        """
        self.frames_dir = Path(frames_dir)
        
        logger.info("=" * 80)
        logger.info("初始化视觉嵌入生成器（16帧视频模式）")
        logger.info("=" * 80)
        
        # 加载MineCLIP
        logger.info("加载MineCLIP...")
        self.mineclip = load_mineclip_wconfig()
        logger.info("✓ MineCLIP已加载")
        
        logger.info(f"帧目录: {self.frames_dir}")
        logger.info("=" * 80)
    
    def find_task_directories(self) -> List[Tuple[str, Path]]:
        """
        查找所有任务目录
        
        Returns:
            [(task_id, task_dir), ...]
        """
        task_dirs = []
        
        for task_dir in sorted(self.frames_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            
            task_id = task_dir.name
            
            # 检查是否有trial目录
            trial_dirs = list(task_dir.glob("trial*"))
            if trial_dirs:
                task_dirs.append((task_id, task_dir))
        
        return task_dirs
    
    def group_frames_by_trial(self, task_dir: Path) -> Dict[int, List[Path]]:
        """
        按trial分组帧文件（新结构：每个trial一个目录）
        
        Args:
            task_dir: 任务目录
            
        Returns:
            {trial_num: [frame_paths...], ...}
        """
        trial_frames = {}
        
        # 遍历所有trial目录
        for trial_dir in sorted(task_dir.glob("trial*")):
            if not trial_dir.is_dir():
                continue
            
            # 从目录名提取trial编号
            try:
                trial_num = int(trial_dir.name.replace("trial", ""))
            except ValueError:
                logger.warning(f"  无法解析trial编号: {trial_dir.name}")
                continue
            
            # 获取该trial的所有帧（文件命名：frame_000.png）
            frames = sorted(trial_dir.glob("frame_*.png"))
            
            if frames:
                trial_frames[trial_num] = frames
        
        return trial_frames
    
    def load_and_preprocess_frame(self, image_path: Path) -> Optional[np.ndarray]:
        """
        加载并预处理单帧图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像数组 [3, 160, 256]
        """
        try:
            # 加载图像并确保是RGB格式
            img = Image.open(image_path)
            
            # 强制转换为RGB（处理RGBA、灰度等格式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # MineCLIP预期输入: [16, 3, 160, 256]
            # 调整大小
            img = img.resize((256, 160), Image.Resampling.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(img).astype(np.float32)
            
            # 验证形状
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                logger.error(f"图像形状异常 {image_path}: {img_array.shape}，期望 (160, 256, 3)")
                return None
            
            # 归一化到 [0, 1]
            img_array = img_array / 255.0
            
            # 转换为 CHW 格式 [3, 160, 256]
            img_array = np.transpose(img_array, (2, 0, 1))
            
            return img_array
            
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            return None
    
    def generate_video_embed_from_frames(
        self, 
        frame_paths: List[Path],
        trial_num: int
    ) -> Optional[np.ndarray]:
        """
        从16帧生成视频嵌入
        
        Args:
            frame_paths: 16帧图像路径列表
            trial_num: trial编号（用于日志）
            
        Returns:
            视觉嵌入 [512] 或 None
        """
        if len(frame_paths) != 16:
            logger.warning(f"    Trial {trial_num}: 帧数 {len(frame_paths)} != 16，跳过")
            return None
        
        try:
            # 加载所有帧
            frames = []
            for frame_path in frame_paths:
                frame = self.load_and_preprocess_frame(frame_path)
                if frame is None:
                    logger.warning(f"    Trial {trial_num}: 加载帧失败 {frame_path.name}")
                    return None
                frames.append(frame)
            
            # 堆叠为视频张量 [16, 3, 160, 256]
            video_array = np.stack(frames, axis=0)
            
            # 转换为torch tensor并添加batch维度 [1, 16, 3, 160, 256]
            video_tensor = th.from_numpy(video_array).unsqueeze(0).float().to(DEVICE)
            
            # 使用MineCLIP编码视频
            with th.no_grad():
                video_embed = self.mineclip.encode_video(video_tensor)
            
            # 转换为numpy
            video_embed = video_embed.cpu().numpy().squeeze()
            
            logger.info(f"    ✓ Trial {trial_num}: 生成嵌入 shape={video_embed.shape}")
            return video_embed
            
        except Exception as e:
            logger.error(f"    Trial {trial_num}: 生成嵌入失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def generate_embeds_for_task(
        self,
        task_id: str,
        task_dir: Path
    ) -> Dict[int, bool]:
        """
        为单个任务的所有trial生成视频嵌入（每个trial独立保存）
        
        Args:
            task_id: 任务ID
            task_dir: 任务目录
            
        Returns:
            {trial_num: success, ...}
        """
        logger.info(f"\n处理任务: {task_id}")
        logger.info(f"  任务目录: {task_dir}")
        
        # 按trial分组帧
        trial_frames = self.group_frames_by_trial(task_dir)
        
        if not trial_frames:
            logger.warning(f"  未找到任何trial目录")
            return {}
        
        logger.info(f"  找到 {len(trial_frames)} 个trials")
        
        # 为每个trial生成并保存嵌入
        results = {}
        
        for trial_num in sorted(trial_frames.keys()):
            frame_paths = trial_frames[trial_num]
            logger.info(f"  处理 Trial {trial_num}: {len(frame_paths)} 帧")
            
            # 生成嵌入
            embed = self.generate_video_embed_from_frames(frame_paths, trial_num)
            
            if embed is None:
                results[trial_num] = False
                continue
            
            # 保存到trial目录
            trial_dir = task_dir / f"trial{trial_num}"
            output_path = trial_dir / "visual_embeds.pkl"
            
            try:
                with open(output_path, 'wb') as f:
                    # 保存为单个嵌入（不是列表）
                    pickle.dump(embed, f)
                logger.info(f"    ✓ 保存到: {output_path}")
                results[trial_num] = True
            except Exception as e:
                logger.error(f"    ✗ 保存失败: {e}")
                results[trial_num] = False
        
        successful_count = sum(1 for v in results.values() if v)
        logger.info(f"  ✓ 成功生成 {successful_count}/{len(results)} 个嵌入")
        
        return results
    
    
    def run(self) -> Dict[str, bool]:
        """
        运行完整的嵌入生成流程
        
        Returns:
            {task_id: success, ...}
        """
        logger.info("\n" + "=" * 80)
        logger.info("开始生成视觉嵌入")
        logger.info("=" * 80)
        
        # 查找所有任务
        task_dirs = self.find_task_directories()
        
        if not task_dirs:
            logger.error("未找到任何任务目录")
            return {}
        
        logger.info(f"找到 {len(task_dirs)} 个任务")
        
        # 处理每个任务
        results = {}
        
        for task_id, task_dir in task_dirs:
            # 生成并保存嵌入（每个trial独立）
            trial_results = self.generate_embeds_for_task(task_id, task_dir)
            
            # 任务级别的成功判定：至少有一个trial成功
            results[task_id] = any(trial_results.values()) if trial_results else False
        
        # 生成报告
        logger.info("\n" + "=" * 80)
        logger.info("生成报告")
        logger.info("=" * 80)
        
        successful_tasks = [tid for tid, success in results.items() if success]
        failed_tasks = [tid for tid, success in results.items() if not success]
        
        logger.info(f"\n成功: {len(successful_tasks)}/{len(results)}")
        for task_id in successful_tasks:
            logger.info(f"  ✓ {task_id}")
        
        if failed_tasks:
            logger.info(f"\n失败: {len(failed_tasks)}/{len(results)}")
            for task_id in failed_tasks:
                logger.info(f"  ✗ {task_id}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从成功帧生成视觉嵌入（16帧视频模式）"
    )
    parser.add_argument(
        '--frames-dir',
        type=str,
        required=True,
        help='成功帧根目录（如 data/success_visuals）'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default=None,
        help='输出报告路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = VisualEmbedGenerator(
        frames_dir=args.frames_dir
    )
    
    # 运行
    results = generator.run()
    
    # 保存报告
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("视觉嵌入生成报告（16帧视频模式）\n")
            f.write("=" * 80 + "\n\n")
            
            successful_tasks = [tid for tid, success in results.items() if success]
            failed_tasks = [tid for tid, success in results.items() if not success]
            
            f.write(f"总任务数: {len(results)}\n")
            f.write(f"成功: {len(successful_tasks)}\n")
            f.write(f"失败: {len(failed_tasks)}\n\n")
            
            if successful_tasks:
                f.write("成功任务:\n")
                for task_id in successful_tasks:
                    f.write(f"  ✓ {task_id}\n")
                f.write("\n")
            
            if failed_tasks:
                f.write("失败任务:\n")
                for task_id in failed_tasks:
                    f.write(f"  ✗ {task_id}\n")
                f.write("\n")
        
        logger.info(f"\n报告已保存: {report_path}")
    
    # 返回状态码
    if all(results.values()):
        logger.info("\n✓ 所有任务成功")
        return 0
    elif any(results.values()):
        logger.warning("\n⚠ 部分任务失败")
        return 1
    else:
        logger.error("\n✗ 所有任务失败")
        return 2


if __name__ == '__main__':
    sys.exit(main())
