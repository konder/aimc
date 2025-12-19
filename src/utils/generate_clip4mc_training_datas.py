#!/usr/bin/env python3
"""
Decord Video Processing Pipeline - DALI-like Architecture
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass
from collections import OrderedDict
import time
from multiprocessing import Pool
from functools import partial

import numpy as np
import decord
from decord import VideoReader, cpu, gpu
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class VideoSegment:
    """视频片段元数据"""
    vid: str                    # 视频 ID
    video_path: Path            # 视频文件路径
    start_time: float           # 开始时间（秒）
    end_time: float             # 结束时间（秒）
    transcript: str             # 文本描述
    size: List[float] = None    # 目标物体大小（16 个值）
    data_type: str = 'train'    # 数据类型：'train', 'val', 'test'
    text_input_path: str = None # text_input.pkl 文件路径（预生成）
    
    @property
    def duration(self) -> float:
        """片段时长"""
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"VideoSegment(vid={self.vid}, {self.start_time:.1f}s-{self.end_time:.1f}s, type={self.data_type})"


@dataclass
class ProcessedSample:
    """处理后的样本"""
    index: int
    vid: str
    frames: np.ndarray          # (N, H, W, 3) uint8
    tokens: np.ndarray          # (77,) int64
    size: List[float]           # [16] float
    sample_dir: Path
    success: bool
    data_type: str = 'train'    # 数据类型：'train', 'val', 'test'
    error_msg: Optional[str] = None
    
    @property
    def num_frames(self) -> int:
        return self.frames.shape[0] if self.frames is not None else 0


# ============================================================
# Pipeline Components
# ============================================================

class VideoDataSource:
    """
    视频数据源 (Data Source)
    
    负责：
    1. 加载预处理好的元数据 JSON
    2. 生成 VideoSegment 列表
    
    元数据格式：
    [
        {
            "vid": "abc123",
            "video_path": "/path/to/video.mp4",
            "start_time": 10.5,
            "end_time": 15.3,
            "transcript": "player mining diamond",
            "size": [0.3, 0.4, 0.5, ...],  # 16 个值
            "data_type": "train",  # 'train', 'val', 'test'
            "text_input_path": "/path/to/abc123_text_input.pkl"  # 预生成的 text_input.pkl
        },
        ...
    ]
    """
    
    def __init__(self, metadata_path: Path):
        """
        Args:
            metadata_path: 预处理好的元数据 JSON 文件
                          包含 vid, video_path, start_time, end_time, transcript, size
        """
        self.metadata_path = Path(metadata_path)
        self.segments: List[VideoSegment] = []
        
        self._load_data()
    
    def _load_data(self):
        """加载元数据并生成 VideoSegment 列表"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"加载了 {len(metadata)} 条元数据")
        
        # 直接从元数据创建 VideoSegment
        success_count = 0
        for item in metadata:
            vid = item.get('vid', '')
            video_path_str = item.get('video_path', '')
            
            if not vid or not video_path_str:
                logger.warning(f"跳过无效条目: vid={vid}, video_path={video_path_str}")
                continue
            
            video_path = Path(video_path_str)
            
            # 检查文件是否存在
            if not video_path.exists():
                logger.warning(f"视频文件不存在: {video_path}")
                continue
            
            segment = VideoSegment(
                vid=vid,
                video_path=video_path,
                start_time=item.get('start_time', 0),
                end_time=item.get('end_time', 0),
                transcript=item.get('transcript', ''),
                size=item.get('size', []),
                data_type=item.get('data_type', 'train'),  # 默认为 train
                text_input_path=item.get('text_input_path', None)
            )
            self.segments.append(segment)
            success_count += 1
        
        logger.info(f"加载成功: {success_count}/{len(metadata)} 个视频片段")
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, index: int) -> VideoSegment:
        return self.segments[index]
    
    def __iter__(self) -> Iterator[VideoSegment]:
        return iter(self.segments)


class DecordProcessor:
    """
    Decord 视频处理器 (Processor)
    
    负责：
    1. 使用 Decord 提取视频帧
    2. Resize 到目标尺寸
    3. 文本 Tokenization
    """
    
    def __init__(
        self,
        frame_height: int = 160,
        frame_width: int = 256,
        device_id: int = 0,
        use_gpu: bool = True
    ):
        """
        Args:
            frame_height: 目标帧高度
            frame_width: 目标帧宽度
            device_id: GPU ID
            use_gpu: 是否使用 GPU
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.device_id = device_id
        self.use_gpu = use_gpu
        
        self.ctx = gpu(device_id) if use_gpu else cpu(0)
        
        logger.info(f"DecordProcessor 初始化: {'GPU' if use_gpu else 'CPU'} (device={device_id})")
    
    def extract_frames(self, segment: VideoSegment) -> Optional[np.ndarray]:
        """
        提取视频帧 (核心方法)
        
        Args:
            segment: 视频片段信息
        
        Returns:
            frames: (N, H, W, 3) uint8, RGB 格式
        """
        try:
            # 创建 VideoReader (Decord 支持初始化时指定 resize)
            vr = VideoReader(
                str(segment.video_path),
                ctx=self.ctx,
                width=self.frame_width,
                height=self.frame_height
            )
            
            # 获取视频信息
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            
            if total_frames == 0:
                return None
            
            # 计算帧范围
            start_frame = int(segment.start_time * fps)
            end_frame = int(segment.end_time * fps)
            
            # 确保索引有效
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            # 批量读取帧（Decord 的核心优势）
            frame_indices = list(range(start_frame, end_frame))
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Decord 输出: (N, H, W, 3) RGB uint8
            return frames.astype(np.uint8)
        
        except Exception as e:
            logger.error(f"{segment.vid}: Decord 解码失败 - {str(e)}")
            return None
    
    def process_segment(self, index: int, segment: VideoSegment) -> ProcessedSample:
        """
        处理单个视频片段
        
        Returns:
            ProcessedSample: 包含处理结果的数据结构
        """
        try:
            # 1. 提取帧
            frames = self.extract_frames(segment)
            if frames is None or len(frames) == 0:
                return ProcessedSample(
                    index=index,
                    vid=segment.vid,
                    frames=None,
                    tokens=None,
                    size=None,
                    sample_dir=None,
                    success=False,
                    data_type=segment.data_type,
                    error_msg="帧提取失败"
                )
            
            # 2. 检查 text_input_path
            if not segment.text_input_path:
                return ProcessedSample(
                    index=index,
                    vid=segment.vid,
                    frames=None,
                    tokens=None,
                    size=None,
                    sample_dir=None,
                    success=False,
                    data_type=segment.data_type,
                    error_msg="元数据缺少 text_input_path"
                )
            
            text_input_path = Path(segment.text_input_path)
            if not text_input_path.exists():
                return ProcessedSample(
                    index=index,
                    vid=segment.vid,
                    frames=None,
                    tokens=None,
                    size=None,
                    sample_dir=None,
                    success=False,
                    data_type=segment.data_type,
                    error_msg=f"text_input.pkl 文件不存在: {text_input_path}"
                )
            
            # 3. 处理 size 数据
            # 注意：官方 CLIP4MC 的 size 数组固定为 16 个值
            # 这些值对应 DataLoader 动态采样的 16 帧，而非实际提取的所有帧
            # 因此直接使用元数据中的 size，不做任何处理
            size_values = segment.size if segment.size else []
            
            if not size_values or len(size_values) == 0:
                # 元数据未提供 size，使用占位符（16 个值）
                size_values = [0] * 16
            
            return ProcessedSample(
                index=index,
                vid=segment.vid,
                frames=frames,
                tokens=text_input_path,  # 存储 text_input.pkl 路径，稍后拷贝
                size=size_values,
                sample_dir=None,  # 稍后由 Saver 填充
                success=True,
                data_type=segment.data_type
            )
        
        except Exception as e:
            return ProcessedSample(
                index=index,
                vid=segment.vid,
                frames=None,
                tokens=None,
                size=None,
                sample_dir=None,
                success=False,
                data_type=segment.data_type,
                error_msg=f"处理异常: {str(e)}"
            )


class SampleSaver:
    """
    样本保存器 (Saver)
    
    负责：
    1. 保存 video_input.pkl
    2. 拷贝预生成的 text_input.pkl
    3. 保存 size.json
    """
    
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_sample(self, sample: ProcessedSample) -> bool:
        """
        保存单个样本
        
        Returns:
            bool: 是否成功
        """
        if not sample.success:
            return False
        
        try:
            import shutil
            
            # 创建样本目录
            sample_dir = self.output_dir / f"sample_{sample.index:06d}_{sample.vid}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 video_input.pkl
            with open(sample_dir / "video_input.pkl", "wb") as f:
                pickle.dump(sample.frames, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 拷贝预生成的 text_input.pkl
            # sample.tokens 现在存储的是 text_input.pkl 的路径
            text_input_src = sample.tokens  # 这是一个 Path 对象
            if isinstance(text_input_src, (str, Path)):
                text_input_src = Path(text_input_src)
                if text_input_src.exists():
                    shutil.copy2(text_input_src, sample_dir / "text_input.pkl")
                else:
                    logger.error(f"text_input.pkl 源文件不存在: {text_input_src}")
                    return False
            else:
                logger.error(f"text_input.pkl 路径格式错误: {type(text_input_src)}")
                return False
            
            # 保存 size.json
            with open(sample_dir / "size.json", "w") as f:
                json.dump(sample.size, f)
            
            # 更新 sample_dir
            sample.sample_dir = sample_dir
            
            return True
        
        except Exception as e:
            logger.error(f"保存失败 {sample.vid}: {str(e)}")
            sample.success = False
            sample.error_msg = f"保存失败: {str(e)}"
            return False


def _process_single_segment_worker(
    args: Tuple[int, VideoSegment, Path, Tuple[int, int], int, bool]
) -> Dict[str, Any]:
    """
    多进程 worker 函数：处理单个视频片段
    
    Args:
        args: (index, segment, output_dir, frame_size, device_id, use_gpu)
    
    Returns:
        dict: 处理结果
    """
    index, segment, output_dir, frame_size, device_id, use_gpu = args
    
    # 每个 worker 初始化自己的 processor 和 saver
    processor = DecordProcessor(
        frame_height=frame_size[0],
        frame_width=frame_size[1],
        device_id=device_id,
        use_gpu=use_gpu
    )
    
    saver = SampleSaver(output_dir=output_dir)
    
    # 处理
    sample = processor.process_segment(index, segment)
    
    # 保存
    if sample.success:
        success = saver.save_sample(sample)
        if not success:
            sample.success = False
    
    # 返回结果
    return {
        'index': index,
        'vid': segment.vid,
        'success': sample.success,
        'error_msg': sample.error_msg,
        'sample_dir': str(sample.sample_dir) if sample.sample_dir else None,
        'data_type': segment.data_type
    }


# ============================================================
# Pipeline & Iterator
# ============================================================

class DecordPipeline:
    """
    Decord 视频处理 Pipeline (类似 DALI Pipeline)
    
    架构：
        VideoDataSource -> DecordProcessor -> SampleSaver -> Iterator
    
    特点：
        - Pipeline 模式设计
        - 支持批量迭代
        - 进度跟踪
        - 统计信息
    """
    
    def __init__(
        self,
        metadata_path: Path,
        output_dir: Path,
        batch_size: int = 1,
        frame_size: Tuple[int, int] = (160, 256),
        device_id: int = 0,
        use_gpu: bool = True,
        num_workers: int = 1,
        show_progress: bool = True
    ):
        """
        Args:
            metadata_path: 预处理好的元数据 JSON（包含 vid, video_path, start_time, end_time, size, data_type, text_input_path）
            output_dir: 输出目录
            batch_size: 批量大小（用于进度显示，实际仍是逐个处理）
            frame_size: (height, width) 目标帧尺寸
            device_id: GPU ID
            use_gpu: 是否使用 GPU
            num_workers: 并行进程数（默认: 1，单线程）
            show_progress: 是否显示进度条
        """
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.num_workers = num_workers
        self.frame_size = frame_size
        self.device_id = device_id
        self.use_gpu = use_gpu
        
        # 1. 初始化组件
        logger.info("=" * 60)
        logger.info("初始化 Decord Pipeline")
        logger.info("=" * 60)
        
        self.data_source = VideoDataSource(metadata_path=metadata_path)
        
        # 单进程模式：初始化 processor
        if num_workers == 1:
            self.processor = DecordProcessor(
                frame_height=frame_size[0],
                frame_width=frame_size[1],
                device_id=device_id,
                use_gpu=use_gpu
            )
        else:
            # 多进程模式：不在主进程初始化 processor（每个子进程会初始化自己的）
            self.processor = None
        
        self.saver = SampleSaver(output_dir=output_dir)
        
        # 2. 统计信息
        self.stats = {
            'total': len(self.data_source),
            'processed': 0,
            'success': 0,
            'failed': 0,
            'failed_samples': [],
            'successful_samples': [],  # 存储成功的样本信息（包含 data_type）
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"待处理: {self.stats['total']} 个视频片段")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"设备: {'GPU' if use_gpu else 'CPU'} (ID={device_id})")
        logger.info(f"帧尺寸: {frame_size[0]}x{frame_size[1]}")
        logger.info(f"并行进程: {num_workers}")
        logger.info("=" * 60)
    
    def __len__(self) -> int:
        """Pipeline 中的样本总数"""
        return len(self.data_source)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        迭代处理器（支持多进程）
        
        Yields:
            batch_result: 包含批次处理结果的字典
        """
        self.stats['start_time'] = time.time()
        
        if self.num_workers == 1:
            # 单进程模式：原有逻辑
            yield from self._iter_single_process()
        else:
            # 多进程模式：使用 Pool
            yield from self._iter_multi_process()
        
        self.stats['end_time'] = time.time()
    
    def _iter_single_process(self) -> Iterator[Dict[str, Any]]:
        """单进程迭代器"""
        # 创建进度条
        if self.show_progress:
            pbar = tqdm(
                total=len(self.data_source),
                desc="🎬 Decord Pipeline (单进程)",
                unit="video"
            )
        else:
            pbar = None
        
        # 逐个处理
        for index, segment in enumerate(self.data_source):
            # 1. 处理
            sample = self.processor.process_segment(index, segment)
            
            # 2. 保存
            if sample.success:
                save_success = self.saver.save_sample(sample)
                if save_success:
                    self.stats['success'] += 1
                    self.stats['successful_samples'].append({
                        'sample_dir': str(sample.sample_dir),
                        'data_type': sample.data_type,
                        'vid': sample.vid
                    })
                else:
                    self.stats['failed'] += 1
                    self.stats['failed_samples'].append({
                        'vid': sample.vid,
                        'error': sample.error_msg
                    })
            else:
                self.stats['failed'] += 1
                self.stats['failed_samples'].append({
                    'vid': sample.vid,
                    'error': sample.error_msg
                })
                if self.stats['failed'] <= 10:
                    logger.warning(f"处理失败: {sample.vid} - {sample.error_msg}")
            
            self.stats['processed'] += 1
            
            # 3. 更新进度
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'success': self.stats['success'],
                    'failed': self.stats['failed']
                })
            
            # 4. Yield 批次结果
            yield {
                'index': index,
                'vid': segment.vid,
                'success': sample.success,
                'num_frames': sample.frames.shape[0] if sample.success and sample.frames is not None else 0,
                'sample_dir': str(sample.sample_dir) if sample.sample_dir else None,
                'error_msg': sample.error_msg,
                'batch_size': 1,
                'num_success': self.stats['success'],
                'num_failed': self.stats['failed']
            }
        
        if pbar:
            pbar.close()
    
    def _iter_multi_process(self) -> Iterator[Dict[str, Any]]:
        """多进程迭代器"""
        # 创建进度条
        if self.show_progress:
            pbar = tqdm(
                total=len(self.data_source),
                desc=f"🎬 Decord Pipeline ({self.num_workers}进程)",
                unit="video"
            )
        else:
            pbar = None
        
        # 准备参数列表
        args_list = [
            (
                index,
                segment,
                self.saver.output_dir,
                self.frame_size,
                self.device_id,
                self.use_gpu
            )
            for index, segment in enumerate(self.data_source)
        ]
        
        # 使用 Pool.imap 进行并行处理（保持顺序）
        with Pool(processes=self.num_workers) as pool:
            for result in pool.imap(_process_single_segment_worker, args_list):
                # 统计
                if result['success']:
                    self.stats['success'] += 1
                    self.stats['successful_samples'].append({
                        'sample_dir': result['sample_dir'],
                        'data_type': result['data_type'],
                        'vid': result['vid']
                    })
                else:
                    self.stats['failed'] += 1
                    self.stats['failed_samples'].append({
                        'vid': result['vid'],
                        'error': result['error_msg']
                    })
                    if self.stats['failed'] <= 10:
                        logger.warning(f"处理失败: {result['vid']} - {result['error_msg']}")
                
                self.stats['processed'] += 1
                
                # 更新进度
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': self.stats['success'],
                        'failed': self.stats['failed']
                    })
                
                # Yield 结果（添加额外字段保持兼容）
                result['batch_size'] = 1
                result['num_success'] = self.stats['success']
                result['num_failed'] = self.stats['failed']
                yield result
        
        if pbar:
            pbar.close()
    
    def run(self) -> Dict[str, Any]:
        """
        运行整个 pipeline（便捷方法）
        
        Returns:
            stats: 统计信息字典
        """
        for _ in self:
            pass  # 迭代完成
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            elapsed = stats['end_time'] - stats['start_time']
            stats['elapsed_time'] = elapsed
            stats['speed'] = stats['success'] / elapsed if elapsed > 0 else 0
        
        return stats
    
    def summary(self):
        """打印摘要"""
        stats = self.get_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline 处理完成")
        logger.info("=" * 60)
        logger.info(f"总数: {stats['total']}")
        logger.info(f"成功: {stats['success']}")
        logger.info(f"失败: {stats['failed']}")
        
        if 'elapsed_time' in stats:
            logger.info(f"耗时: {stats['elapsed_time']/60:.2f} 分钟")
            logger.info(f"速度: {stats['speed']:.2f} 视频/秒")
        
        # 保存失败列表
        if stats['failed_samples']:
            failed_json = self.saver.output_dir / "failed_samples.json"
            with open(failed_json, 'w', encoding='utf-8') as f:
                json.dump(stats['failed_samples'], f, indent=2, ensure_ascii=False)
            logger.info(f"失败列表: {failed_json}")
        
        # 生成 dataset_info.json
        self._generate_dataset_info(stats['successful_samples'])
        
        logger.info("=" * 60)
    
    def _generate_dataset_info(self, successful_samples: List[Dict[str, str]]):
        """
        生成 dataset_info.json
        
        根据元数据中的 data_type 字段分配样本到 train/val/test
        
        Args:
            successful_samples: 成功处理的样本列表
                [
                    {
                        'sample_dir': '/path/to/sample_000000_xxx',
                        'data_type': 'train',
                        'vid': 'xxx'
                    },
                    ...
                ]
        """
        dataset_info_path = self.saver.output_dir / "dataset_info.json"
        
        # 根据 data_type 分组
        dataset_info = {
            "train": [],
            "val": [],
            "test": []
        }
        
        for sample in successful_samples:
            sample_dir = sample['sample_dir']
            data_type = sample.get('data_type', 'train')
            
            # 确保 data_type 有效
            if data_type not in ['train', 'val', 'test']:
                logger.warning(f"无效的 data_type: {data_type}，默认为 train")
                data_type = 'train'
            
            dataset_info[data_type].append(sample_dir)
        
        # 保存
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        # 打印统计
        logger.info(f"dataset_info.json 生成成功:")
        logger.info(f"  train: {len(dataset_info['train'])} 个样本")
        logger.info(f"  val:   {len(dataset_info['val'])} 个样本")
        logger.info(f"  test:  {len(dataset_info['test'])} 个样本")


# ============================================================
# Command Line Interface
# ============================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Decord Video Processing Pipeline (DALI-like Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本使用
    python src/utils/generate_clip4mc_training_datas.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output

    # GPU 加速
    python src/utils/decord_pipeline.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output \\
        --use-gpu \\
        --device-id 0

    # 多进程加速（10 个进程）
    python src/utils/generate_clip4mc_training_datas.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output \\
        --num-workers 10

    # 自定义参数
    python src/utils/generate_clip4mc_training_datas.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output \\
        --use-gpu \\
        --device-id 0 \\
        --num-workers 4 \\
        --frame-height 160 \\
        --frame-width 256

元数据 JSON 格式:
    [
        {
            "vid": "abc123",
            "video_path": "/path/to/video.mp4",
            "start_time": 10.5,
            "end_time": 15.3,
            "transcript": "player mining diamond",
            "size": [0.3, 0.4, 0.5, ...],  # 16 个值
            "data_type": "train",  # 'train', 'val', 'test'
            "text_input_path": "/path/to/abc123_text_input.pkl"  # 预生成的 text_input.pkl
        },
        ...
    ]
        """
    )
    
    # 必需参数
    parser.add_argument("--metadata", type=Path, required=True,
                       help="预处理好的元数据 JSON 文件（包含 vid, video_path, start_time, end_time, transcript, size）")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="输出目录")
    
    # 可选参数
    parser.add_argument("--frame-height", type=int, default=160,
                       help="目标帧高度 (默认: 160)")
    parser.add_argument("--frame-width", type=int, default=256,
                       help="目标帧宽度 (默认: 256)")
    parser.add_argument("--device-id", type=int, default=0,
                       help="GPU ID (默认: 0)")
    parser.add_argument("--use-gpu", action='store_true',
                       help="使用 GPU 加速")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="并行进程数 (默认: 1，单线程)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="批量大小（暂仅用于进度显示）")
    parser.add_argument("--no-progress", action='store_true',
                       help="禁用进度条")
    
    args = parser.parse_args()
    
    # 创建 pipeline
    pipeline = DecordPipeline(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        frame_size=(args.frame_height, args.frame_width),
        device_id=args.device_id,
        use_gpu=args.use_gpu,
        num_workers=args.num_workers,
        show_progress=not args.no_progress
    )
    
    # 运行 pipeline
    pipeline.run()
    
    # 打印摘要
    pipeline.summary()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

