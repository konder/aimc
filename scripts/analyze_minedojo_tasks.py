#!/usr/bin/env python3
"""
MineDojo任务分析工具
用于从官方任务清单中提取、分析和推荐适合Steve1评估的任务

Usage:
    python scripts/analyze_minedojo_tasks.py --action list --category harvest
    python scripts/analyze_minedojo_tasks.py --action search --keyword milk
    python scripts/analyze_minedojo_tasks.py --action recommend
"""

import yaml
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import requests


MINEDOJO_TASKS_URL = "https://raw.githubusercontent.com/MineDojo/MineDojo/main/minedojo/tasks/description_files/programmatic_tasks.yaml"


def download_tasks(url: str = MINEDOJO_TASKS_URL) -> Dict:
    """下载MineDojo任务清单"""
    print(f"下载任务清单: {url}")
    response = requests.get(url)
    response.raise_for_status()
    tasks = yaml.safe_load(response.text)
    print(f"✓ 成功加载 {len(tasks)} 个任务\n")
    return tasks


def analyze_tasks(tasks: Dict) -> Dict[str, List[str]]:
    """按类别分析任务"""
    categories = defaultdict(list)
    
    for task_id, task_info in tasks.items():
        category = task_info.get('category', 'unknown')
        categories[category].append(task_id)
    
    return dict(categories)


def list_category_tasks(tasks: Dict, category: str, limit: int = 20):
    """列出指定类别的任务"""
    filtered = {k: v for k, v in tasks.items() if v.get('category') == category}
    
    print(f"\n{'='*80}")
    print(f"类别: {category.upper()} (共 {len(filtered)} 个任务)")
    print(f"{'='*80}\n")
    
    for i, (task_id, task_info) in enumerate(filtered.items(), 1):
        if i > limit:
            print(f"\n... 还有 {len(filtered) - limit} 个任务未显示")
            break
        
        prompt = task_info.get('prompt', 'N/A')
        print(f"{i}. {task_id}")
        print(f"   指令: {prompt}")
        print()


def search_tasks(tasks: Dict, keyword: str):
    """搜索包含关键词的任务"""
    results = []
    
    for task_id, task_info in tasks.items():
        if keyword.lower() in task_id.lower() or keyword.lower() in task_info.get('prompt', '').lower():
            results.append((task_id, task_info))
    
    print(f"\n{'='*80}")
    print(f"搜索关键词: '{keyword}' (找到 {len(results)} 个任务)")
    print(f"{'='*80}\n")
    
    for i, (task_id, task_info) in enumerate(results, 1):
        prompt = task_info.get('prompt', 'N/A')
        category = task_info.get('category', 'N/A')
        print(f"{i}. [{category}] {task_id}")
        print(f"   指令: {prompt}")
        print()


def extract_simple_instruction(prompt: str) -> str:
    """从MineDojo的prompt中提取简洁的Steve1指令"""
    # 移除常见的前缀
    prompt = prompt.lower()
    
    # 提取核心动作
    if 'harvest' in prompt or 'obtain' in prompt or 'get' in prompt:
        # harvest 1 milk -> milk cow
        match = re.search(r'(?:harvest|obtain|get)\s+(\d+\s+)?(\w+)', prompt)
        if match:
            item = match.group(2)
            
            # 特殊映射
            mapping = {
                'milk': 'milk cow',
                'wool': 'shear sheep',
                'beef': 'kill cow',
                'porkchop': 'kill pig',
                'chicken': 'kill chicken',
                'leather': 'get leather from cow',
                'log': 'chop tree',
                'dirt': 'dig dirt',
                'sand': 'dig sand',
                'cobblestone': 'mine stone',
                'coal': 'mine coal',
                'iron_ore': 'mine iron ore',
                'gold_ore': 'mine gold ore',
                'diamond': 'mine diamond',
            }
            
            if item in mapping:
                return mapping[item]
            
            # 默认模式
            if 'ore' in item:
                return f"mine {item.replace('_ore', '')}"
            return f"get {item}"
    
    elif 'combat' in prompt:
        # combat a zombie -> kill zombie
        match = re.search(r'combat\s+a?\s*(\w+)', prompt)
        if match:
            mob = match.group(1)
            return f"kill {mob}"
    
    elif 'craft' in prompt or 'starting from' in prompt:
        # craft a wooden pickaxe -> craft wooden pickaxe
        match = re.search(r'craft(?:\s+a(?:nd use)?)?\s+([\w\s]+?)(?:\s+and\s+use)?$', prompt)
        if match:
            item = match.group(1).strip()
            return f"craft {item}"
        
        # starting from X, craft and use Y -> make Y
        match = re.search(r'craft and use a?\s*([\w\s]+)', prompt)
        if match:
            item = match.group(1).strip()
            return f"make {item}"
    
    # 默认返回简化版
    return prompt[:50]


def recommend_tasks_for_steve1(tasks: Dict) -> Dict[str, List[Tuple[str, str, str]]]:
    """推荐适合Steve1评估的任务"""
    
    recommendations = {
        'Harvest - Level 1 (基础采集)': [],
        'Harvest - Level 2 (动物互动)': [],
        'Harvest - Level 3 (工具使用)': [],
        'Harvest - Level 4 (植物采集)': [],
        'Combat - Level 1 (被动生物)': [],
        'Combat - Level 2 (装备战斗)': [],
        'Combat - Level 3 (敌对生物)': [],
        'TechTree - Level 1 (基础合成)': [],
        'TechTree - Level 2 (木制工具)': [],
        'TechTree - Level 3 (石制工具)': [],
        'TechTree - Level 4 (铁制工具)': [],
    }
    
    # Harvest Level 1 - 基础采集
    basic_harvest = [
        'harvest_1_log', 'harvest_1_dirt', 'harvest_1_sand', 
        'harvest_1_cobblestone', 'harvest_1_gravel'
    ]
    for task_id in basic_harvest:
        if task_id in tasks:
            prompt = tasks[task_id]['prompt']
            instruction = extract_simple_instruction(prompt)
            recommendations['Harvest - Level 1 (基础采集)'].append((task_id, prompt, instruction))
    
    # Harvest Level 2 - 动物互动
    animal_tasks = [
        'harvest_1_milk', 'harvest_1_wool', 'harvest_1_beef', 
        'harvest_1_porkchop', 'harvest_1_leather'
    ]
    for task_id in animal_tasks:
        if task_id in tasks:
            prompt = tasks[task_id]['prompt']
            instruction = extract_simple_instruction(prompt)
            recommendations['Harvest - Level 2 (动物互动)'].append((task_id, prompt, instruction))
    
    # Harvest Level 3 - 工具使用（带工具版本）
    tool_tasks = []
    for task_id in tasks:
        if ('harvest_1_coal' in task_id or 'harvest_1_iron_ore' in task_id or 
            'harvest_1_gold_ore' in task_id or 'harvest_1_diamond' in task_id):
            if 'with' in task_id:  # 带工具版本
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                tool_tasks.append((task_id, prompt, instruction))
    
    # 如果没有带工具版本，使用基础版本
    if not tool_tasks:
        for task_id in ['harvest_1_coal', 'harvest_1_iron_ore', 'harvest_1_gold_ore']:
            if task_id in tasks:
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                recommendations['Harvest - Level 3 (工具使用)'].append((task_id, prompt, instruction))
    else:
        recommendations['Harvest - Level 3 (工具使用)'] = tool_tasks[:4]
    
    # Harvest Level 4 - 植物采集
    plant_tasks = [
        'harvest_1_red_flower', 'harvest_1_yellow_flower', 
        'harvest_1_apple', 'harvest_1_sapling'
    ]
    for task_id in plant_tasks:
        if task_id in tasks:
            prompt = tasks[task_id]['prompt']
            instruction = extract_simple_instruction(prompt)
            recommendations['Harvest - Level 4 (植物采集)'].append((task_id, prompt, instruction))
    
    # Combat Level 1 - 被动生物
    passive_mobs = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'combat':
            if any(mob in task_id for mob in ['chicken', 'pig', 'cow', 'sheep']):
                if 'wooden_sword' in task_id and 'plains' in task_id:
                    prompt = tasks[task_id]['prompt']
                    instruction = extract_simple_instruction(prompt)
                    passive_mobs.append((task_id, prompt, instruction))
    recommendations['Combat - Level 1 (被动生物)'] = passive_mobs[:3]
    
    # Combat Level 2 - 装备战斗
    equipped_combat = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'combat':
            if 'zombie' in task_id and 'leather_armors' in task_id and 'shield' in task_id:
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                equipped_combat.append((task_id, prompt, instruction))
                if len(equipped_combat) >= 2:
                    break
    recommendations['Combat - Level 2 (装备战斗)'] = equipped_combat
    
    # Combat Level 3 - 敌对生物
    hostile_mobs = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'combat':
            if any(mob in task_id for mob in ['zombie', 'spider', 'skeleton']):
                if 'iron_armors' in task_id and 'iron_sword' in task_id:
                    prompt = tasks[task_id]['prompt']
                    instruction = extract_simple_instruction(prompt)
                    hostile_mobs.append((task_id, prompt, instruction))
                    if len(hostile_mobs) >= 3:
                        break
    recommendations['Combat - Level 3 (敌对生物)'] = hostile_mobs
    
    # TechTree Level 1 - 基础合成
    basic_craft = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'techtree':
            if any(item in task_id for item in ['planks', 'crafting_table', 'sticks', 'torch']):
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                basic_craft.append((task_id, prompt, instruction))
    recommendations['TechTree - Level 1 (基础合成)'] = basic_craft[:4]
    
    # TechTree Level 2 - 木制工具
    wooden_tools = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'techtree':
            if 'wooden' in task_id and any(tool in task_id for tool in ['pickaxe', 'sword', 'axe']):
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                wooden_tools.append((task_id, prompt, instruction))
    recommendations['TechTree - Level 2 (木制工具)'] = wooden_tools[:4]
    
    # TechTree Level 3 - 石制工具
    stone_tools = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'techtree':
            if 'stone' in task_id and ('pickaxe' in task_id or 'sword' in task_id or 'furnace' in task_id):
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                stone_tools.append((task_id, prompt, instruction))
    recommendations['TechTree - Level 3 (石制工具)'] = stone_tools[:4]
    
    # TechTree Level 4 - 铁制工具
    iron_tools = []
    for task_id in tasks:
        if tasks[task_id].get('category') == 'techtree':
            if 'iron' in task_id and any(tool in task_id for tool in ['pickaxe', 'sword', 'ingot']):
                prompt = tasks[task_id]['prompt']
                instruction = extract_simple_instruction(prompt)
                iron_tools.append((task_id, prompt, instruction))
    recommendations['TechTree - Level 4 (铁制工具)'] = iron_tools[:4]
    
    return recommendations


def print_recommendations(recommendations: Dict[str, List[Tuple[str, str, str]]]):
    """打印推荐任务"""
    print(f"\n{'='*80}")
    print("Steve1评估任务推荐")
    print(f"{'='*80}\n")
    
    total_tasks = 0
    for level, tasks in recommendations.items():
        if tasks:
            print(f"\n## {level} ({len(tasks)}个任务)\n")
            print(f"{'Task ID':<50} {'推荐指令':<30}")
            print("-" * 80)
            
            for task_id, prompt, instruction in tasks:
                print(f"{task_id:<50} {instruction:<30}")
                total_tasks += 1
            
            print()
    
    print(f"\n总计推荐: {total_tasks} 个任务\n")


def export_to_yaml(recommendations: Dict[str, List[Tuple[str, str, str]]], output_file: str):
    """导出推荐任务到YAML配置文件"""
    print(f"\n导出任务配置到: {output_file}")
    
    config = {
        'evaluation': {
            'n_trials': 10,
            'max_steps': 6000,
            'image_size': [320, 640],
            'output': {
                'results_dir': 'results/evaluation',
                'enable_video': True,
                'enable_report': True,
            }
        }
    }
    
    # 添加任务
    for level, tasks in recommendations.items():
        if not tasks:
            continue
        
        category_key = level.split(' - ')[0].lower() + '_tasks'
        if category_key not in config:
            config[category_key] = []
        
        for task_id, prompt, instruction in tasks:
            task_config = {
                'task_id': task_id,
                'en_instruction': instruction,
                'description': prompt,
                'max_steps': 2000,
                'n_trials': 10,
            }
            config[category_key].append(task_config)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ 成功导出配置文件")


def main():
    parser = argparse.ArgumentParser(description='MineDojo任务分析工具')
    parser.add_argument('--action', choices=['list', 'search', 'recommend', 'export'], 
                        default='recommend', help='操作类型')
    parser.add_argument('--category', help='任务类别 (harvest/combat/techtree/survival)')
    parser.add_argument('--keyword', help='搜索关键词')
    parser.add_argument('--limit', type=int, default=20, help='显示任务数量限制')
    parser.add_argument('--output', default='config/eval_tasks_minedojo.yaml', 
                        help='导出文件路径')
    parser.add_argument('--offline', action='store_true', help='离线模式（需要先下载任务文件）')
    
    args = parser.parse_args()
    
    # 加载任务
    if args.offline:
        print("离线模式：请确保已下载任务文件")
        # TODO: 从本地文件加载
        return
    else:
        try:
            tasks = download_tasks()
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("\n提示: 可以手动下载任务文件:")
            print(f"  wget {MINEDOJO_TASKS_URL} -O minedojo_tasks.yaml")
            return
    
    # 执行操作
    if args.action == 'list':
        if not args.category:
            # 显示统计
            categories = analyze_tasks(tasks)
            print("\nMineDojo任务统计:")
            print("-" * 40)
            for cat, task_list in sorted(categories.items()):
                print(f"{cat:<15} {len(task_list):>5} 个任务")
            print("-" * 40)
            print(f"{'总计':<15} {len(tasks):>5} 个任务\n")
        else:
            list_category_tasks(tasks, args.category, args.limit)
    
    elif args.action == 'search':
        if not args.keyword:
            print("错误: 搜索需要提供 --keyword 参数")
            return
        search_tasks(tasks, args.keyword)
    
    elif args.action == 'recommend':
        recommendations = recommend_tasks_for_steve1(tasks)
        print_recommendations(recommendations)
    
    elif args.action == 'export':
        recommendations = recommend_tasks_for_steve1(tasks)
        export_to_yaml(recommendations, args.output)


if __name__ == '__main__':
    main()

