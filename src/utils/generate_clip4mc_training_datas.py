#!/usr/bin/env python3
"""
FFmpeg Video Processing Pipeline - Pure FFmpeg Implementation
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
import subprocess
import tempfile

import numpy as np
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


class FFmpegProcessor:
    """
    FFmpeg 视频处理器 (Processor) - 纯 FFmpeg 实现
    
    负责：
    1. 使用 FFmpeg 提取视频帧
    2. 支持 GPU 硬件加速 (NVDEC)
    3. Resize 到目标尺寸
    """
    
    def __init__(
        self,
        frame_height: int = 160,
        frame_width: int = 256,
        device_id: int = 0,
        use_gpu: bool = False
    ):
        """
        Args:
            frame_height: 目标帧高度
            frame_width: 目标帧宽度
            device_id: GPU ID（用于 NVDEC）
            use_gpu: 是否使用 GPU 硬件加速
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.device_id = device_id
        self.use_gpu = use_gpu
        
        # 检查 ffmpeg 是否支持 GPU 加速
        self.gpu_available = False
        if use_gpu:
            self.gpu_available = self._check_gpu_support()
            if not self.gpu_available:
                logger.warning("GPU 加速不可用，将使用 CPU 模式")
        
        mode = "GPU (NVDEC)" if (use_gpu and self.gpu_available) else "CPU"
        logger.info(f"FFmpegProcessor 初始化: {mode} (device={device_id})")
    
    def _check_gpu_support(self) -> bool:
        """
        检查 FFmpeg 是否支持 GPU 硬件加速
        
        Returns:
            bool: 是否支持 GPU
        """
        try:
            # 检查 ffmpeg 是否支持 cuda hwaccel
            result = subprocess.run(
                ['ffmpeg', '-hwaccels'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True
            )
            
            if 'cuda' in result.stdout.lower():
                return True
            
            return False
        except Exception:
            return False
    
    def extract_frames(self, segment: VideoSegment) -> Optional[np.ndarray]:
        """
        使用 FFmpeg 提取视频帧 (核心方法)
        
        Args:
            segment: 视频片段信息
        
        Returns:
            frames: (N, H, W, 3) uint8, RGB 格式
        """
        # GPU 模式：先尝试 GPU，失败则回退 CPU
        if self.use_gpu and self.gpu_available:
            frames = self._try_gpu_decode(segment)
            if frames is not None:
                return frames
            # GPU 失败，回退到 CPU（静默回退，不打印警告）
        
        # CPU 模式
        return self._extract_frames_cpu(segment)
    
    def _try_gpu_decode(self, segment: VideoSegment) -> Optional[np.ndarray]:
        """
        尝试使用 GPU 解码（NVDEC）
        
        策略:
        1. 先尝试 hwaccel cuda + scale (GPU 解码 + CPU 缩放)
        2. 失败则返回 None，由主流程回退到纯 CPU
        
        Returns:
            frames: (N, H, W, 3) uint8, RGB 格式，失败返回 None
        """
        try:
            duration = segment.end_time - segment.start_time
            
            # GPU 解码 + CPU 缩放（最兼容的方案）
            # -ss 放在 -i 前面（输入前 seek，更快）
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(self.device_id),
                '-hwaccel_output_format', 'cuda',  # 明确指定输出格式
                '-ss', str(segment.start_time),
                '-i', str(segment.video_path),
                '-t', str(duration),
                '-vf', f'hwdownload,format=nv12,scale={self.frame_width}:{self.frame_height}',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-loglevel', 'error',
                'pipe:1'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            if result.returncode == 0 and len(result.stdout) > 0:
                raw_data = result.stdout
                frame_size = self.frame_height * self.frame_width * 3
                
                if len(raw_data) >= frame_size:
                    num_frames = len(raw_data) // frame_size
                    if num_frames > 0:
                        frames = np.frombuffer(raw_data[:num_frames * frame_size], dtype=np.uint8)
                        frames = frames.reshape((num_frames, self.frame_height, self.frame_width, 3))
                        return frames
            
            # GPU 解码失败，返回 None
            return None
        
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
    
    def _extract_frames_cpu(self, segment: VideoSegment) -> Optional[np.ndarray]:
        """
        使用 CPU 模式提取视频帧（GPU 失败时的备选）
        
        Returns:
            frames: (N, H, W, 3) uint8, RGB 格式
        """
        try:
            duration = segment.end_time - segment.start_time
            
            cmd = [
                'ffmpeg',
                '-ss', str(segment.start_time),
                '-i', str(segment.video_path),
                '-t', str(duration),
                '-vf', f'scale={self.frame_width}:{self.frame_height}',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-loglevel', 'error',
                'pipe:1'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            if result.returncode != 0:
                return None
            
            raw_data = result.stdout
            frame_size = self.frame_height * self.frame_width * 3
            
            if len(raw_data) < frame_size:
                return None
            
            num_frames = len(raw_data) // frame_size
            if num_frames == 0:
                return None
            
            frames = np.frombuffer(raw_data[:num_frames * frame_size], dtype=np.uint8)
            frames = frames.reshape((num_frames, self.frame_height, self.frame_width, 3))
            
            return frames
        
        except Exception as e:
            logger.error(f"{segment.vid}: CPU 解码失败 - {str(e)}")
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


# 全局变量（用于进程池初始化）
_worker_processor = None
_worker_saver = None


def _init_worker(output_dir: Path, frame_size: Tuple[int, int], device_id: int, use_gpu: bool):
    """
    初始化 worker 进程（每个进程只调用一次）
    
    Args:
        output_dir: 输出目录
        frame_size: (height, width)
        device_id: 设备 ID
        use_gpu: 是否使用 GPU 硬件加速
    """
    global _worker_processor, _worker_saver
    
    _worker_processor = FFmpegProcessor(
        frame_height=frame_size[0],
        frame_width=frame_size[1],
        device_id=device_id,
        use_gpu=use_gpu
    )
    
    _worker_saver = SampleSaver(output_dir=output_dir)


def _process_single_segment_worker(args: Tuple[int, VideoSegment]) -> Dict[str, Any]:
    """
    多进程 worker 函数：处理单个视频片段
    
    Args:
        args: (index, segment)
    
    Returns:
        dict: 处理结果
    """
    global _worker_processor, _worker_saver
    
    index, segment = args
    
    # 使用已初始化的 processor 和 saver
    sample = _worker_processor.process_segment(index, segment)
    
    # 保存
    if sample.success:
        success = _worker_saver.save_sample(sample)
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

class FFmpegPipeline:
    """
    FFmpeg 视频处理 Pipeline (类似 DALI Pipeline)
    
    架构：
        VideoDataSource -> FFmpegProcessor -> SampleSaver -> Iterator
    
    特点：
        - 纯 FFmpeg 实现
        - 支持 GPU 硬件加速 (NVDEC)
        - 支持所有视频格式 (H.264, VP9, HEVC, etc.)
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
        use_gpu: bool = False,
        num_workers: int = 1,
        show_progress: bool = True
    ):
        """
        Args:
            metadata_path: 预处理好的元数据 JSON（包含 vid, video_path, start_time, end_time, size, data_type, text_input_path）
            output_dir: 输出目录
            batch_size: 批量大小（用于进度显示，实际仍是逐个处理）
            frame_size: (height, width) 目标帧尺寸
            device_id: GPU ID（用于 FFmpeg NVDEC）
            use_gpu: 是否使用 GPU 硬件加速（FFmpeg NVDEC）
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
            self.processor = FFmpegProcessor(
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
        logger.info(f"解码器: FFmpeg (纯实现)")
        logger.info(f"硬件加速: {'GPU (NVDEC)' if use_gpu else 'CPU'} (ID={device_id})")
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
            mode = "GPU" if self.use_gpu else "CPU"
            pbar = tqdm(
                total=len(self.data_source),
                desc=f"🎬 FFmpeg Pipeline ({mode})",
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
            mode = "GPU" if self.use_gpu else "CPU"
            pbar = tqdm(
                total=len(self.data_source),
                desc=f"🎬 FFmpeg Pipeline ({mode}, {self.num_workers}进程)",
                unit="video"
            )
        else:
            pbar = None
        
        # 准备参数列表（只传递 index 和 segment）
        args_list = [
            (index, segment)
            for index, segment in enumerate(self.data_source)
        ]
        
        # 使用 Pool.imap 进行并行处理（保持顺序）
        # 使用 initializer 让每个进程只初始化一次
        with Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.saver.output_dir, self.frame_size, self.device_id, self.use_gpu)
        ) as pool:
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
        description="FFmpeg Video Processing Pipeline (Pure FFmpeg Implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本使用 (CPU 模式)
    python src/utils/generate_clip4mc_training_datas.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output \\
        --num-workers 8

    # GPU 加速 (NVDEC)
    python src/utils/generate_clip4mc_training_datas.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output \\
        --use-gpu \\
        --device-id 0 \\
        --num-workers 4

    # 自定义参数
    python src/utils/generate_clip4mc_training_datas.py \\
        --metadata preprocessed_metadata.json \\
        --output-dir /path/to/output \\
        --num-workers 8 \\
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
                       help="使用 GPU 硬件加速 (FFmpeg NVDEC)")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="并行进程数 (默认: 1，单线程)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="批量大小（暂仅用于进度显示）")
    parser.add_argument("--no-progress", action='store_true',
                       help="禁用进度条")
    
    args = parser.parse_args()
    
    # 创建 pipeline
    pipeline = FFmpegPipeline(
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

