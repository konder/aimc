#!/usr/bin/env python
"""
从评估结果中提取成功帧图像（v5 - 按trial目录组织）
==================================

从评估结果中提取**真正成功获得奖励**的trials的最后16帧。
每个trial存放在独立目录中，便于管理。

策略：
1. 严格过滤：只提取result.json中success=true的trials
2. 提取最后16帧：成功trial的完成动作序列（用于16帧视频嵌入）
3. 按trial目录组织：每个trial一个目录，包含16帧

输出结构:
    data/success_visuals/
      task_id/
        trial1/
          frame_000.png
          ...
          frame_015.png
        trial2/
          frame_000.png
          ...

用法:
    # 基础模式（跳过已存在的完整trial）
    python src/utils/extract_success_frames_from_eval.py \
        --eval-dir data/evaluation/run1 \
        --output-dir data/success_visuals
    
    # 增量模式（自动递增trial编号）
    python src/utils/extract_success_frames_from_eval.py \
        --eval-dir data/evaluation/run1 \
        --output-dir data/success_visuals \
        --incremental
    
    # 强制覆盖模式
    python src/utils/extract_success_frames_from_eval.py \
        --eval-dir data/evaluation/run1 \
        --output-dir data/success_visuals \
        --no-skip

作者: AI Assistant
日期: 2025-12-02
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SuccessFrameExtractor:
    """从评估结果提取成功帧（最后16帧模式）"""
    
    def __init__(self, eval_dir: str, output_dir: str, top_n: int = 16, 
                 incremental: bool = False, skip_existing: bool = True):
        """
        Args:
            eval_dir: 评估结果目录
            output_dir: 输出目录
            top_n: 提取最后N帧
            incremental: 增量模式（自动递增trial编号，避免覆盖）
            skip_existing: 跳过已存在的完整trial（需要incremental=False）
        """
        self.eval_dir = Path(eval_dir)
        self.output_dir = Path(output_dir)
        self.top_n = top_n
        self.incremental = incremental
        self.skip_existing = skip_existing if not incremental else False
        
        # 用于增量模式：记录每个task的最大trial编号
        self.task_trial_offsets = {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"评估结果目录: {self.eval_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"每个trial提取最后 {self.top_n} 帧")
        if self.incremental:
            logger.info(f"🔄 增量模式: 启用（自动递增trial编号）")
        elif self.skip_existing:
            logger.info(f"⏭️  跳过模式: 启用（跳过已存在的完整trial）")
        logger.info("=" * 80)
        logger.info("策略: 只提取success=true的trials，每个trial提取最后16帧")
        logger.info("=" * 80)
    
    def extract_task_id_from_dirname(self, dirname: str) -> Optional[str]:
        """从目录名提取任务ID"""
        parts = dirname.split('_en_')
        if len(parts) >= 2:
            return parts[0]
        return None
    
    def find_task_directories(self) -> List[Tuple[str, Path]]:
        """查找所有任务目录"""
        task_dirs = []
        
        if not self.eval_dir.exists():
            logger.error(f"评估目录不存在: {self.eval_dir}")
            return task_dirs
        
        for subdir in self.eval_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            task_id = self.extract_task_id_from_dirname(subdir.name)
            if task_id:
                task_dirs.append((task_id, subdir))
        
        logger.info(f"找到 {len(task_dirs)} 个任务目录")
        return task_dirs
    
    def load_result_json(self, task_dir: Path) -> Optional[Dict]:
        """加载result.json（必须）"""
        result_file = task_dir / "result.json"
        
        if not result_file.exists():
            logger.warning(f"未找到result.json: {task_dir}")
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"加载result.json失败 {result_file}: {e}")
            return None
    
    def get_success_trial_indices(self, result_data: Dict) -> Set[int]:
        """
        获取成功的trial索引集合
        
        严格过滤：只返回success=true的trials
        """
        success_indices = set()
        
        trials = result_data.get('trials', [])
        for trial in trials:
            # 严格检查：必须success=true
            if trial.get('success') is True:
                trial_idx = trial.get('trial_idx')
                if trial_idx is not None:
                    success_indices.add(trial_idx)
        
        return success_indices
    
    def get_max_trial_num(self, task_id: str) -> int:
        """
        获取task已有的最大trial编号
        
        Returns:
            最大trial编号（如果不存在则返回0）
        """
        task_output_dir = self.output_dir / task_id
        
        if not task_output_dir.exists():
            return 0
        
        max_num = 0
        for trial_dir in task_output_dir.glob("trial*"):
            if trial_dir.is_dir():
                try:
                    num = int(trial_dir.name.replace("trial", ""))
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        return max_num
    
    def trial_exists_and_complete(self, task_id: str, trial_num: int) -> bool:
        """
        检查trial是否已存在且完整（包含16帧）
        
        Returns:
            True if trial已存在且完整
        """
        trial_dir = self.output_dir / task_id / f"trial{trial_num}"
        
        if not trial_dir.exists():
            return False
        
        # 检查是否包含16帧
        frames = sorted(trial_dir.glob("frame_*.png"))
        return len(frames) >= self.top_n
    
    def extract_frames_from_trial_dir(
        self,
        trial_dir: Path,
        trial_num: int
    ) -> List[Path]:
        """
        从trial目录提取帧
        
        策略：提取最后16帧（成功trial的完成动作序列）
        """
        frames_dir = trial_dir / "frames"
        
        if not frames_dir.exists():
            return []
        
        # 获取所有帧（帧文件命名为 step_*.png）
        all_frames = sorted(frames_dir.glob("step_*.png"))
        
        if not all_frames:
            return []
        
        # 提取最后16帧（成功trial的完成序列）
        total_frames = len(all_frames)
        
        if total_frames <= self.top_n:
            # 如果总帧数不足16，全部使用
            selected_frames = all_frames
            logger.debug(f"    Trial {trial_num}: 总帧数 {total_frames} <= {self.top_n}，使用全部帧")
        else:
            # 提取最后16帧
            selected_frames = all_frames[-self.top_n:]
            logger.debug(f"    Trial {trial_num}: 从 {total_frames} 帧中提取最后 {self.top_n} 帧")
        
        return selected_frames
    
    def extract_frames_for_task(
        self,
        task_id: str,
        task_dir: Path
    ) -> int:
        """为单个任务提取成功帧（严格模式）"""
        logger.info(f"\n处理任务: {task_id}")
        logger.info(f"  任务目录: {task_dir}")
        
        # 加载result.json（必须）
        result_data = self.load_result_json(task_dir)
        if not result_data:
            logger.warning(f"  ✗ 跳过（无result.json）")
            return 0
        
        # 获取成功的trial索引（严格模式）
        success_trial_indices = self.get_success_trial_indices(result_data)
        
        if not success_trial_indices:
            logger.warning(f"  ✗ 无成功trials (success_rate = 0%)")
            return 0
        
        # 获取成功率和成功次数
        success_rate = result_data.get('success_rate', 0.0)
        logger.info(f"  成功率: {success_rate*100:.1f}% ({len(success_trial_indices)}/10)")
        logger.info(f"  成功trials: {sorted(success_trial_indices)}")
        
        # 查找所有trial目录
        all_trial_dirs = sorted(task_dir.glob("report_*_trial*"))
        
        if not all_trial_dirs:
            logger.warning(f"  ✗ 未找到trial目录")
            return 0
        
        # 创建任务输出目录
        task_output_dir = self.output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 增量模式：获取已有的最大trial编号
        if self.incremental:
            if task_id not in self.task_trial_offsets:
                max_existing = self.get_max_trial_num(task_id)
                self.task_trial_offsets[task_id] = max_existing
                logger.info(f"  增量模式: 已有 {max_existing} 个trials，新trials从 {max_existing + 1} 开始")
            trial_offset = self.task_trial_offsets[task_id]
        else:
            trial_offset = 0
        
        # 只从成功的trials中提取帧（每个trial一个目录）
        copied_count = 0
        extracted_trials = []
        skipped_trials = []
        
        for trial_dir in all_trial_dirs:
            # 提取trial编号
            trial_name = trial_dir.name
            trial_parts = trial_name.split('_trial')
            if len(trial_parts) >= 2:
                try:
                    trial_num = int(trial_parts[-1])
                except ValueError:
                    continue
            else:
                continue
            
            # 严格检查：只处理成功的trials
            if trial_num not in success_trial_indices:
                logger.debug(f"  跳过失败trial: {trial_name}")
                continue
            
            # 计算输出trial编号（增量模式 vs 普通模式）
            if self.incremental:
                # 增量模式：递增编号
                trial_offset += 1
                output_trial_num = trial_offset
                self.task_trial_offsets[task_id] = trial_offset
            else:
                # 普通模式：保持原编号
                output_trial_num = trial_num
                
                # 跳过已存在的完整trial
                if self.skip_existing and self.trial_exists_and_complete(task_id, output_trial_num):
                    logger.info(f"  ⏭️  Trial {trial_num}: 已存在且完整，跳过")
                    skipped_trials.append(trial_num)
                    continue
            
            trial_frames = self.extract_frames_from_trial_dir(trial_dir, trial_num)
            
            if not trial_frames:
                logger.warning(f"  {trial_name}: 无帧")
                continue
            
            logger.info(f"  {trial_name}: 提取 {len(trial_frames)} 帧 → trial{output_trial_num}")
            extracted_trials.append(output_trial_num)
            
            # 为每个trial创建独立目录
            trial_output_dir = task_output_dir / f"trial{output_trial_num}"
            trial_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 为每个成功trial完整提取16帧
            for frame_local_idx, src_path in enumerate(trial_frames):
                if not src_path.exists():
                    continue
                
                # 目标文件名：简化为 frame_000.png
                dst_filename = f"frame_{frame_local_idx:03d}.png"
                dst_path = trial_output_dir / dst_filename
                
                # 复制文件
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    logger.error(f"    复制失败 {src_path}: {e}")
        
        if copied_count > 0:
            logger.info(f"  ✓ 提取 {copied_count} 帧 (来自trials: {extracted_trials}) -> {task_output_dir}")
            if skipped_trials:
                logger.info(f"  ⏭️  跳过 {len(skipped_trials)} 个已存在的trials: {skipped_trials}")
        else:
            if skipped_trials:
                logger.info(f"  ⏭️  全部跳过（{len(skipped_trials)} 个trials已存在）")
            else:
                logger.warning(f"  ✗ 未能提取任何帧（尽管有成功trials）")
        
        return copied_count
    
    def run(self) -> Dict[str, int]:
        """运行提取流程（最后16帧模式）"""
        logger.info("\n" + "=" * 80)
        logger.info("开始提取成功帧 (最后16帧模式)")
        logger.info("=" * 80)
        
        task_dirs = self.find_task_directories()
        
        if not task_dirs:
            logger.error("未找到任何任务目录")
            return {}
        
        results = {}
        
        for task_id, task_dir in task_dirs:
            try:
                frame_count = self.extract_frames_for_task(task_id, task_dir)
                results[task_id] = frame_count
            except Exception as e:
                logger.error(f"处理任务 {task_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                results[task_id] = 0
        
        # 打印总结
        logger.info("\n" + "=" * 80)
        logger.info("提取完成 - 总结")
        logger.info("=" * 80)
        
        success_tasks = [(k, v) for k, v in results.items() if v > 0]
        failed_tasks = [k for k, v in results.items() if v == 0]
        
        logger.info(f"\n✅ 成功提取: {len(success_tasks)} 个任务")
        for task_id, frame_count in sorted(success_tasks, key=lambda x: -x[1]):
            logger.info(f"  ✓ {task_id}: {frame_count} 帧")
        
        if failed_tasks:
            logger.info(f"\n✗ 未提取: {len(failed_tasks)} 个任务 (success_rate = 0%)")
            logger.debug(f"  失败任务列表: {failed_tasks[:10]}...")
        
        logger.info(f"\n输出目录: {self.output_dir}")
        logger.info("=" * 80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="从评估结果中提取成功帧图像（最后16帧模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--eval-dir',
        type=str,
        required=True,
        help='评估结果目录路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/success_visuals',
        help='输出目录路径（默认: data/success_visuals）'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=16,
        help='提取最后N帧（默认: 16）'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='增量模式：自动递增trial编号，避免覆盖已有数据'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的trial（会覆盖）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 运行提取器
    extractor = SuccessFrameExtractor(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        incremental=args.incremental,
        skip_existing=not args.no_skip  # --no-skip 表示不跳过
    )
    
    results = extractor.run()
    
    # 返回状态码
    if not results:
        sys.exit(1)
    
    success_count = sum(1 for v in results.values() if v > 0)
    if success_count == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
