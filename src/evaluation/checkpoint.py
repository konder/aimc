"""
检查点管理模块
用于支持评估任务的中断恢复
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .metrics import TrialResult, TaskResult

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: Path):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def get_checkpoint_path(self, task_id: str) -> Path:
        """获取任务的检查点文件路径"""
        return self.checkpoint_dir / f"checkpoint_{task_id}.json"
    
    def save_checkpoint(
        self,
        task_id: str,
        completed_trials: List[TrialResult],
        total_trials: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        保存检查点
        
        Args:
            task_id: 任务ID
            completed_trials: 已完成的trial结果列表
            total_trials: 总trial数
            metadata: 额外的元数据
        """
        checkpoint_path = self.get_checkpoint_path(task_id)
        
        checkpoint_data = {
            "task_id": task_id,
            "total_trials": total_trials,
            "completed_trials_count": len(completed_trials),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "trials": [
                {
                    "task_id": trial.task_id,
                    "language": trial.language,
                    "instruction": trial.instruction,
                    "success": trial.success,
                    "steps": trial.steps,
                    "time_seconds": trial.time_seconds,
                    "final_inventory": trial.final_inventory,
                }
                for trial in completed_trials
            ]
        }
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 检查点已保存: {checkpoint_path}")
            logger.info(f"   进度: {len(completed_trials)}/{total_trials} trials")
        except Exception as e:
            logger.error(f"⚠️ 保存检查点失败: {e}")
    
    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        加载检查点
        
        Args:
            task_id: 任务ID
            
        Returns:
            检查点数据，如果不存在则返回None
        """
        checkpoint_path = self.get_checkpoint_path(task_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"📥 检查点已加载: {checkpoint_path}")
            logger.info(f"   进度: {checkpoint_data['completed_trials_count']}/{checkpoint_data['total_trials']} trials")
            logger.info(f"   时间: {checkpoint_data['timestamp']}")
            
            return checkpoint_data
        except Exception as e:
            logger.error(f"⚠️ 加载检查点失败: {e}")
            return None
    
    def restore_trials(self, checkpoint_data: Dict[str, Any]) -> List[TrialResult]:
        """
        从检查点数据恢复trial结果
        
        Args:
            checkpoint_data: 检查点数据
            
        Returns:
            恢复的trial结果列表
        """
        trials = []
        for trial_data in checkpoint_data.get("trials", []):
            trial = TrialResult(
                task_id=trial_data["task_id"],
                language=trial_data["language"],
                instruction=trial_data["instruction"],
                success=trial_data["success"],
                steps=trial_data["steps"],
                time_seconds=trial_data["time_seconds"],
                final_inventory=trial_data.get("final_inventory", {}),
                trajectory=[]  # 轨迹数据不保存到检查点
            )
            trials.append(trial)
        
        return trials
    
    def delete_checkpoint(self, task_id: str):
        """
        删除检查点文件
        
        Args:
            task_id: 任务ID
        """
        checkpoint_path = self.get_checkpoint_path(task_id)
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"🗑️ 检查点已删除: {checkpoint_path}")
            except Exception as e:
                logger.error(f"⚠️ 删除检查点失败: {e}")
    
    def has_checkpoint(self, task_id: str) -> bool:
        """
        检查是否存在检查点
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否存在检查点
        """
        return self.get_checkpoint_path(task_id).exists()
    
    def get_all_checkpoints(self) -> List[str]:
        """
        获取所有检查点的任务ID列表
        
        Returns:
            任务ID列表
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            # 从文件名提取任务ID: checkpoint_task_id.json
            task_id = checkpoint_file.stem.replace("checkpoint_", "")
            checkpoints.append(task_id)
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_latest: int = 10):
        """
        清理旧的检查点文件
        
        Args:
            keep_latest: 保留最新的N个检查点
        """
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # 删除超过保留数量的旧检查点
        for old_checkpoint in checkpoint_files[keep_latest:]:
            try:
                old_checkpoint.unlink()
                logger.info(f"🗑️ 清理旧检查点: {old_checkpoint.name}")
            except Exception as e:
                logger.error(f"⚠️ 清理检查点失败: {e}")
    
    def get_taskset_checkpoint_path(self, task_set_name: str) -> Path:
        """获取task-set的检查点文件路径"""
        return self.checkpoint_dir / f"taskset_{task_set_name}.json"
    
    def save_taskset_checkpoint(
        self,
        task_set_name: str,
        all_task_ids: List[str],
        completed_task_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        保存task-set检查点（记录已完成的任务）
        
        Args:
            task_set_name: 任务集名称
            all_task_ids: 所有任务ID列表
            completed_task_ids: 已完成的任务ID列表
            metadata: 额外的元数据
        """
        checkpoint_path = self.get_taskset_checkpoint_path(task_set_name)
        
        checkpoint_data = {
            "task_set_name": task_set_name,
            "all_task_ids": all_task_ids,
            "completed_task_ids": completed_task_ids,
            "total_tasks": len(all_task_ids),
            "completed_tasks_count": len(completed_task_ids),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Task-set检查点已保存: {checkpoint_path}")
            logger.info(f"   进度: {len(completed_task_ids)}/{len(all_task_ids)} tasks")
        except Exception as e:
            logger.error(f"⚠️ 保存task-set检查点失败: {e}")
    
    def load_taskset_checkpoint(self, task_set_name: str) -> Optional[Dict[str, Any]]:
        """
        加载task-set检查点
        
        Args:
            task_set_name: 任务集名称
            
        Returns:
            检查点数据，如果不存在则返回None
        """
        checkpoint_path = self.get_taskset_checkpoint_path(task_set_name)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"📥 Task-set检查点已加载: {checkpoint_path}")
            logger.info(f"   进度: {checkpoint_data['completed_tasks_count']}/{checkpoint_data['total_tasks']} tasks")
            logger.info(f"   时间: {checkpoint_data['timestamp']}")
            
            return checkpoint_data
        except Exception as e:
            logger.error(f"⚠️ 加载task-set检查点失败: {e}")
            return None
    
    def delete_taskset_checkpoint(self, task_set_name: str):
        """
        删除task-set检查点文件
        
        Args:
            task_set_name: 任务集名称
        """
        checkpoint_path = self.get_taskset_checkpoint_path(task_set_name)
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"🗑️ Task-set检查点已删除: {checkpoint_path}")
            except Exception as e:
                logger.error(f"⚠️ 删除task-set检查点失败: {e}")


class CheckpointConfig:
    """检查点配置"""
    
    def __init__(
        self,
        enabled: bool = True,
        save_interval: int = 5,  # 每N个trial保存一次
        auto_resume: bool = True,  # 自动恢复
        cleanup_on_complete: bool = True,  # 完成后清理检查点
    ):
        """
        初始化检查点配置
        
        Args:
            enabled: 是否启用检查点
            save_interval: 保存间隔（每N个trial）
            auto_resume: 是否自动恢复
            cleanup_on_complete: 完成后是否自动清理检查点
        """
        self.enabled = enabled
        self.save_interval = save_interval
        self.auto_resume = auto_resume
        self.cleanup_on_complete = cleanup_on_complete

