"""
三维能力矩阵分析器
Matrix Analyzer - Analyze evaluation results using three-dimensional capability matrix

基于设计文档中的三维能力矩阵：
- Harvest维度: Level 1-4 (基础采集 → 动物互动 → 工具使用 → 植物采集)
- Combat维度: Level 1-4 (被动生物 → 装备战斗 → 敌对生物 → 高级战斗)
- TechTree维度: Level 1-4 (基础合成 → 木制工具 → 石制工具 → 铁制/钻石工具)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class MatrixAnalyzer:
    """三维能力矩阵分析器"""
    
    # 维度定义
    DIMENSIONS = {
        'harvest': {
            'name': 'Harvest (采集)',
            'levels': {
                1: {'name': '基础采集', 'weight': 1.0, 'keywords': ['basic_resources', 'dirt', 'log', 'sand']},
                2: {'name': '动物互动', 'weight': 1.5, 'keywords': ['animal_products', 'animal_drops', 'milk', 'wool', 'beef']},
                3: {'name': '工具使用', 'weight': 2.0, 'keywords': ['mining', 'coal', 'iron_ore', 'cobblestone']},
                4: {'name': '植物采集', 'weight': 2.5, 'keywords': ['plants', 'food', 'flower', 'sapling', 'apple']},
            }
        },
        'combat': {
            'name': 'Combat (战斗)',
            'levels': {
                1: {'name': '被动生物', 'weight': 1.0, 'keywords': ['passive_mobs', 'chicken', 'pig', 'cow']},
                2: {'name': '装备战斗', 'weight': 1.5, 'keywords': ['hostile_mobs_equipped', 'leather_armor', 'shield']},
                3: {'name': '敌对生物', 'weight': 2.0, 'keywords': ['hostile_mobs', 'zombie', 'spider']},
                4: {'name': '高级战斗', 'weight': 2.5, 'keywords': ['hostile_mobs_advanced', 'skeleton', 'creeper']},
            }
        },
        'techtree': {
            'name': 'TechTree (科技树)',
            'levels': {
                1: {'name': '基础合成', 'weight': 1.0, 'keywords': ['basic_crafting', 'planks', 'crafting_table', 'sticks']},
                2: {'name': '木制工具', 'weight': 1.5, 'keywords': ['wooden_tools', 'wooden_pickaxe', 'wooden_sword']},
                3: {'name': '石制工具', 'weight': 2.0, 'keywords': ['stone_tools', 'stone_pickaxe', 'furnace']},
                4: {'name': '铁制/钻石', 'weight': 2.5, 'keywords': ['iron_tools', 'diamond_tools', 'iron_pickaxe', 'diamond']},
            }
        }
    }
    
    def __init__(self):
        """初始化矩阵分析器"""
        pass
    
    def classify_task(self, task_config: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
        """
        分类任务到维度和等级
        
        Args:
            task_config: 任务配置字典
            
        Returns:
            (dimension, level) 元组，如果无法分类则返回 (None, None)
        """
        task_id = task_config.get('task_id', '')
        category = task_config.get('category', '')
        description = task_config.get('description', '').lower()
        
        # 首先根据任务集分类（harvest_tasks, combat_tasks, techtree_tasks）
        if 'harvest' in task_id.lower() or 'harvest' in category.lower():
            dimension = 'harvest'
        elif 'combat' in task_id.lower() or 'combat' in category.lower():
            dimension = 'combat'
        elif 'techtree' in task_id.lower() or 'techtree' in category.lower() or 'craft' in task_id.lower():
            dimension = 'techtree'
        else:
            # 尝试从category字段推断
            for dim_key, dim_info in self.DIMENSIONS.items():
                for level, level_info in dim_info['levels'].items():
                    if category.lower() in level_info['keywords']:
                        dimension = dim_key
                        break
            else:
                return (None, None)
        
        # 根据关键词确定等级
        for level, level_info in self.DIMENSIONS[dimension]['levels'].items():
            # 检查category和task_id中的关键词
            combined_text = f"{task_id} {category} {description}".lower()
            if any(keyword in combined_text for keyword in level_info['keywords']):
                return (dimension, level)
        
        # 如果没有匹配关键词，默认返回Level 1
        return (dimension, 1)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析评估结果
        
        Args:
            results: 任务结果列表
            
        Returns:
            分析报告字典
        """
        # 按维度和等级分组
        matrix_results = {
            dim: {level: [] for level in range(1, 5)} 
            for dim in self.DIMENSIONS.keys()
        }
        unclassified_tasks = []
        
        # 分类任务结果
        for result in results:
            task_config = result.get('task_config', {})
            dimension, level = self.classify_task(task_config)
            
            if dimension and level:
                matrix_results[dimension][level].append(result)
            else:
                unclassified_tasks.append(result)
        
        # 计算维度得分
        dimension_scores = {}
        for dim_key, dim_info in self.DIMENSIONS.items():
            level_scores = []
            total_weight = 0
            
            for level, level_info in dim_info['levels'].items():
                level_tasks = matrix_results[dim_key][level]
                
                if level_tasks:
                    # 计算该等级的平均成功率
                    success_rates = [
                        task.get('success_rate', 0.0) for task in level_tasks
                    ]
                    avg_success_rate = np.mean(success_rates) if success_rates else 0.0
                    
                    # 加权得分
                    weight = level_info['weight']
                    level_scores.append(avg_success_rate * weight)
                    total_weight += weight
                else:
                    level_scores.append(0.0)
            
            # 维度总分 = 加权平均
            if total_weight > 0:
                dimension_score = sum(level_scores) / total_weight
            else:
                dimension_score = 0.0
            
            dimension_scores[dim_key] = dimension_score
        
        # 计算综合得分
        # Harvest权重40%, Combat权重30%, TechTree权重30%
        overall_score = (
            dimension_scores.get('harvest', 0) * 0.40 +
            dimension_scores.get('combat', 0) * 0.30 +
            dimension_scores.get('techtree', 0) * 0.30
        )
        
        # 构建分析报告
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'matrix_results': self._format_matrix_results(matrix_results),
            'summary': self._generate_summary(matrix_results, dimension_scores),
            'unclassified_tasks': [
                {
                    'task_id': t.get('task_config', {}).get('task_id'),
                    'category': t.get('task_config', {}).get('category'),
                }
                for t in unclassified_tasks
            ]
        }
        
        return analysis
    
    def _format_matrix_results(self, matrix_results: Dict) -> Dict:
        """格式化矩阵结果"""
        formatted = {}
        
        for dim_key, levels in matrix_results.items():
            dim_name = self.DIMENSIONS[dim_key]['name']
            formatted[dim_key] = {
                'name': dim_name,
                'levels': {}
            }
            
            for level, tasks in levels.items():
                level_name = self.DIMENSIONS[dim_key]['levels'][level]['name']
                
                if tasks:
                    success_rates = [t.get('success_rate', 0.0) for t in tasks]
                    avg_steps = [
                        t.get('avg_steps', 0) for t in tasks 
                        if t.get('avg_steps') is not None
                    ]
                    
                    formatted[dim_key]['levels'][level] = {
                        'name': level_name,
                        'task_count': len(tasks),
                        'avg_success_rate': float(np.mean(success_rates)),
                        'avg_steps': float(np.mean(avg_steps)) if avg_steps else None,
                        'tasks': [
                            {
                                'task_id': t.get('task_config', {}).get('task_id'),
                                'success_rate': t.get('success_rate', 0.0),
                                'avg_steps': t.get('avg_steps'),
                            }
                            for t in tasks
                        ]
                    }
                else:
                    formatted[dim_key]['levels'][level] = {
                        'name': level_name,
                        'task_count': 0,
                        'avg_success_rate': 0.0,
                        'avg_steps': None,
                        'tasks': []
                    }
        
        return formatted
    
    def _generate_summary(
        self, 
        matrix_results: Dict, 
        dimension_scores: Dict
    ) -> Dict[str, Any]:
        """生成分析摘要"""
        summary = {
            'total_tasks': 0,
            'dimensions': {},
            'recommendations': []
        }
        
        # 统计各维度
        for dim_key, levels in matrix_results.items():
            total_count = sum(len(tasks) for tasks in levels.values())
            summary['total_tasks'] += total_count
            
            summary['dimensions'][dim_key] = {
                'name': self.DIMENSIONS[dim_key]['name'],
                'task_count': total_count,
                'score': dimension_scores.get(dim_key, 0.0),
                'level_distribution': {
                    level: len(tasks) for level, tasks in levels.items()
                }
            }
        
        # 生成建议
        for dim_key, score in dimension_scores.items():
            dim_name = self.DIMENSIONS[dim_key]['name']
            
            if score < 0.3:
                summary['recommendations'].append(
                    f"⚠️ {dim_name}得分较低（{score:.1%}），建议优化指令或增加训练"
                )
            elif score < 0.5:
                summary['recommendations'].append(
                    f"📝 {dim_name}有提升空间（{score:.1%}），可尝试调整任务难度或指令"
                )
            elif score >= 0.7:
                summary['recommendations'].append(
                    f"✅ {dim_name}表现优秀（{score:.1%}），继续保持"
                )
        
        return summary
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: Path):
        """
        保存分析结果
        
        Args:
            analysis: 分析结果
            output_path: 输出路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, analysis: Dict[str, Any]):
        """
        打印分析摘要
        
        Args:
            analysis: 分析结果
        """
        print(f"\n{'='*80}")
        print(f"三维能力矩阵分析报告")
        print(f"{'='*80}\n")
        
        # 综合得分
        overall_score = analysis['overall_score']
        print(f"📊 综合得分: {overall_score:.1%}\n")
        
        # 维度得分
        print("维度得分:")
        for dim_key, score in analysis['dimension_scores'].items():
            dim_name = self.DIMENSIONS[dim_key]['name']
            bar_length = int(score * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {dim_name:<20} {bar} {score:.1%}")
        
        print()
        
        # 各维度详情
        for dim_key, dim_data in analysis['matrix_results'].items():
            print(f"\n## {dim_data['name']}")
            print("-" * 80)
            
            for level_key, level_data in dim_data['levels'].items():
                level_name = level_data['name']
                task_count = level_data['task_count']
                avg_success = level_data['avg_success_rate']
                
                if task_count > 0:
                    print(f"  Level {level_key} - {level_name:<15} "
                          f"({task_count}个任务) 平均成功率: {avg_success:.1%}")
                else:
                    print(f"  Level {level_key} - {level_name:<15} "
                          f"(无任务)")
        
        # 建议
        summary = analysis['summary']
        if summary['recommendations']:
            print(f"\n## 建议")
            print("-" * 80)
            for rec in summary['recommendations']:
                print(f"  {rec}")
        
        print(f"\n{'='*80}\n")

