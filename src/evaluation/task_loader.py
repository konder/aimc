"""
任务加载器
Task Loader - Loads and parses evaluation task configurations
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class TaskLoader:
    """加载和管理评估任务配置"""
    
    def __init__(self, config_path: str = "config/eval_tasks.yaml"):
        """
        初始化任务加载器
        
        Args:
            config_path: 任务配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.tasks = self._parse_tasks()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _parse_tasks(self) -> Dict[str, Dict[str, Any]]:
        """解析所有任务配置"""
        tasks = {}
        
        # 新格式：tasks 字典（eval_tasks.yaml）
        if 'tasks' in self.config and isinstance(self.config['tasks'], dict):
            for task_id, task_config in self.config['tasks'].items():
                task_config['task_id'] = task_id  # 确保 task_id 存在
                tasks[task_id] = task_config
        
        # 旧格式：解析不同类别的任务（支持任意以 _tasks 结尾的键）
        for key in self.config.keys():
            if key.endswith('_tasks') and isinstance(self.config[key], list):
                for task in self.config[key]:
                    if 'task_id' in task:
                        task_id = task['task_id']
                        tasks[task_id] = task
        
        return tasks
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定任务的配置
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务配置字典，如果不存在返回None
        """
        return self.tasks.get(task_id)
    
    def get_task_set(self, set_name: str) -> List[str]:
        """
        获取任务集
        
        Args:
            set_name: 任务集名称 (quick_test, baseline_test, harvest_tasks, harvest, 或 all)
            
        Returns:
            任务ID列表
        """
        if set_name == 'all':
            return list(self.tasks.keys())
        
        # 新格式：从 task_sets 字典获取（eval_tasks.yaml）
        if 'task_sets' in self.config and isinstance(self.config['task_sets'], dict):
            task_sets = self.config['task_sets']
            if set_name in task_sets:
                set_config = task_sets[set_name]
                # task_sets 中每个任务集有 'tasks' 列表
                if isinstance(set_config, dict) and 'tasks' in set_config:
                    return set_config['tasks']
                elif isinstance(set_config, list):
                    return set_config
        
        # 旧格式：以 _tasks 结尾的键（如 harvest_tasks）
        if set_name.endswith('_tasks'):
            if set_name in self.config and isinstance(self.config[set_name], list):
                task_list = self.config[set_name]
                # 检查列表中的元素类型
                if task_list and isinstance(task_list[0], dict):
                    # 如果是字典列表（完整任务定义），提取 task_id
                    return [task['task_id'] for task in task_list if 'task_id' in task]
                else:
                    # 如果是字符串列表（任务ID列表），直接返回
                    return task_list
            else:
                return []  # 返回空列表而不是抛出异常
        
        # 旧格式：预定义的任务集（如 quick_test, baseline_test）
        if set_name in self.config:
            task_set_config = self.config[set_name]
            if isinstance(task_set_config, list):
                # 检查列表中的元素类型
                if task_set_config and isinstance(task_set_config[0], dict):
                    return [task.get('task_id', task) for task in task_set_config]
                else:
                    return task_set_config
        
        raise ValueError(f"未知的任务集: {set_name}")
    
    def list_task_sets(self) -> List[str]:
        """
        列出所有可用的任务集
        
        Returns:
            任务集名称列表
        """
        task_sets = []
        
        # 新格式：从 task_sets 字典获取
        if 'task_sets' in self.config and isinstance(self.config['task_sets'], dict):
            task_sets.extend(list(self.config['task_sets'].keys()))
        
        # 旧格式：以 _tasks 结尾的键
        for key in self.config.keys():
            if key.endswith('_tasks') and isinstance(self.config[key], list):
                if key not in task_sets:  # 避免重复
                    task_sets.append(key)
        
        return task_sets
    
    def get_tasks_by_category(self, category: str) -> List[str]:
        """
        获取指定类别的所有任务
        
        Args:
            category: 任务类别 (harvest, combat, techtree)
            
        Returns:
            任务ID列表
        """
        return [
            task_id for task_id, task in self.tasks.items()
            if task.get('category') == category
        ]
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[str]:
        """
        获取指定难度的所有任务
        
        Args:
            difficulty: 难度级别 (easy, medium, hard)
            
        Returns:
            任务ID列表
        """
        return [
            task_id for task_id, task in self.tasks.items()
            if task.get('difficulty') == difficulty
        ]
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估参数配置"""
        return self.config.get('evaluation', {})
    
    def list_all_tasks(self) -> List[str]:
        """列出所有任务ID"""
        return list(self.tasks.keys())
    
    def print_task_summary(self):
        """打印任务集摘要"""
        print("\n" + "="*70)
        print("任务集摘要 (Task Summary)")
        print("="*70)
        
        # 按难度统计
        by_difficulty = {}
        for task_id, task in self.tasks.items():
            difficulty = task.get('difficulty', 'unknown')
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = []
            by_difficulty[difficulty].append(task_id)
        
        print("\n按难度分类:")
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in by_difficulty:
                count = len(by_difficulty[difficulty])
                print(f"  {difficulty:8s}: {count} 个任务")
        
        # 按类别统计
        by_category = {}
        for task_id, task in self.tasks.items():
            category = task.get('category', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(task_id)
        
        print("\n按类别分类:")
        for category, task_list in sorted(by_category.items()):
            count = len(task_list)
            print(f"  {category:8s}: {count} 个任务")
        
        print(f"\n总任务数: {len(self.tasks)}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # 测试代码
    loader = TaskLoader()
    
    # 打印摘要
    loader.print_task_summary()
    
    # 测试获取任务
    task = loader.get_task("harvest_1_log")
    if task:
        print(f"任务详情: {task['task_id']}")
        print(f"  英文指令: {task['en_instruction']}")
        print(f"  中文指令: {task['zh_instruction']}")
        print(f"  语义变体: {task['zh_variants']}")
    
    # 测试获取任务集
    quick_test = loader.get_task_set('quick_test')
    print(f"\n快速测试集: {quick_test}")

