#!/usr/bin/env python3
"""
DAgger Web 控制台 - 多任务管理
提供友好的 Web 界面来管理 DAgger 训练流程
直接从文件系统读取任务信息，无需数据库
"""

import os
import sys
import json
import yaml
import subprocess
import threading
import time
import numpy as np
import pickle
import base64
import logging
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# 添加项目根目录到Python路径
# web/app.py 在 src/web/ 下，所以项目根目录是 parent.parent.parent
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置模板
from src.web.task_config_template import DEFAULT_TASK_CONFIG, CONFIG_FIELDS, CONFIG_CATEGORIES

app = Flask(__name__)
CORS(app)

# 配置日志：屏蔽频繁的状态检查请求
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # 只显示错误日志

# 自定义日志过滤器
class RequestFilter(logging.Filter):
    def filter(self, record):
        # 屏蔽 /api/status 和 /api/tasks/*/directory 的日志
        if '/api/status' in record.getMessage():
            return False
        if '/api/tasks/' in record.getMessage() and '/directory' in record.getMessage():
            return False
        return True

# 应用过滤器
for handler in log.handlers:
    handler.addFilter(RequestFilter())

# 任务根目录（在 data 目录下）
TASKS_ROOT = project_root / 'data' / 'tasks'
TASKS_ROOT.mkdir(parents=True, exist_ok=True)

# 全局状态
current_task_id = None
current_process = None  # 当前运行的子进程
task_status = {
    'running': False,
    'stage': '',
    'progress': 0,
    'message': '',
    'logs': []
}


# ============================================================================
# 任务管理 - 基于文件系统
# ============================================================================

def get_task_path(task_id):
    """获取任务根目录"""
    return TASKS_ROOT / task_id


def get_task_dirs(task_id):
    """获取任务的各个子目录"""
    task_root = get_task_path(task_id)
    return {
        'root': task_root,
        'baseline_model': task_root / 'baseline_model',    # BC基线模型
        'dagger_model': task_root / 'dagger_model',        # DAgger迭代模型
        'expert_demos': task_root / 'expert_demos',
        'expert_labels': task_root / 'expert_labels',
        'policy_states': task_root / 'policy_states',
        'dagger': task_root / 'dagger',
        # 向后兼容：checkpoints 指向 baseline_model（用于现有代码）
        'checkpoints': task_root / 'baseline_model',
    }


def task_exists(task_id):
    """检查任务是否存在"""
    return get_task_path(task_id).exists()


def list_all_tasks():
    """列出所有任务（扫描 tasks 目录）"""
    if not TASKS_ROOT.exists():
        return []
    
    tasks = []
    for item in TASKS_ROOT.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            tasks.append(item.name)
    
    return sorted(tasks)


def get_task_config(task_id):
    """获取任务配置（优先 YAML，兼容旧 JSON，否则使用默认值）"""
    task_path = get_task_path(task_id)
    config_yaml = task_path / 'config.yaml'
    config_json = task_path / 'config.json'
    
    # 默认配置（从模板复制）
    default_config = DEFAULT_TASK_CONFIG.copy()
    default_config['task_id'] = task_id
    
    # 优先读取 YAML 配置
    if config_yaml.exists():
        try:
            with open(config_yaml, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    default_config.update(user_config)
        except Exception as e:
            print(f"读取 YAML 配置文件失败: {e}")
    # 兼容旧的 JSON 配置
    elif config_json.exists():
        try:
            with open(config_json, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                if user_config:
                    default_config.update(user_config)
        except Exception as e:
            print(f"读取 JSON 配置文件失败: {e}")
    
    return default_config


def save_task_config(task_id, config):
    """保存任务配置（YAML 格式）"""
    task_path = get_task_path(task_id)
    config_file = task_path / 'config.yaml'
    task_path.mkdir(parents=True, exist_ok=True)
    
    # 添加元数据
    config_with_meta = {
        '_metadata': {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        },
        **config
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_task_stats(task_id):
    """获取任务统计信息（从文件系统推断）"""
    dirs = get_task_dirs(task_id)
    
    # 统计专家演示数量
    expert_count = 0
    if dirs['expert_demos'].exists():
        expert_count = len(list(dirs['expert_demos'].glob('episode_*')))
    
    # 检查BC基线（新路径）
    bc_model = dirs['baseline_model'] / 'bc_baseline.zip'
    has_bc = bc_model.exists()
    
    # 统计DAgger迭代（新路径）
    dagger_iterations = 0
    latest_iteration = 0
    dagger_models = []
    if dirs['dagger_model'].exists():
        model_files = sorted(dirs['dagger_model'].glob('dagger_iter_*.zip'))
        for f in model_files:
            try:
                iter_num = int(f.stem.split('_')[-1])
                dagger_iterations += 1
                latest_iteration = max(latest_iteration, iter_num)
                dagger_models.append(f.name)
            except:
                pass
    
    return {
        'expert_demos': expert_count,
        'has_bc_baseline': has_bc,
        'bc_model_path': str(bc_model) if has_bc else None,
        'dagger_iterations': dagger_iterations,
        'latest_iteration': latest_iteration,
        'latest_model_path': str(dirs['dagger_model'] / f'dagger_iter_{latest_iteration}.zip') if latest_iteration > 0 else None,
        'dagger_models': dagger_models  # 添加模型文件名列表
    }


def get_task_directory_info(task_id):
    """获取任务目录详细信息"""
    dirs = get_task_dirs(task_id)
    info = {}
    
    # BC基线模型目录
    baseline_info = {
        'path': str(dirs['baseline_model'].relative_to(project_root)),
        'models': []
    }
    if dirs['baseline_model'].exists():
        for model_file in sorted(dirs['baseline_model'].glob('*.zip')):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            
            # 检查是否有评估结果
            eval_file = model_file.parent / f"{model_file.stem}_eval_results.npy"
            eval_result = None
            if eval_file.exists():
                try:
                    results = np.load(eval_file, allow_pickle=True).item()
                    eval_result = {
                        'success_rate': f"{results.get('success_rate', 0) * 100:.1f}%",
                        'avg_steps': f"{results.get('avg_steps', 0):.0f}",
                        'episodes': results.get('total_episodes', 0)
                    }
                except:
                    pass
            
            baseline_info['models'].append({
                'name': model_file.name,
                'size': f"{size_mb:.1f} MB",
                'modified': modified,
                'path': str(model_file),
                'eval_result': eval_result
            })
    info['baseline_model'] = baseline_info
    
    # DAgger迭代模型目录
    dagger_model_info = {
        'path': str(dirs['dagger_model'].relative_to(project_root)),
        'models': []
    }
    if dirs['dagger_model'].exists():
        for model_file in sorted(dirs['dagger_model'].glob('*.zip')):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            
            # 检查是否有评估结果
            eval_file = model_file.parent / f"{model_file.stem}_eval_results.npy"
            eval_result = None
            if eval_file.exists():
                try:
                    results = np.load(eval_file, allow_pickle=True).item()
                    eval_result = {
                        'success_rate': f"{results.get('success_rate', 0) * 100:.1f}%",
                        'avg_steps': f"{results.get('avg_steps', 0):.0f}",
                        'episodes': results.get('total_episodes', 0)
                    }
                except:
                    pass
            
            dagger_model_info['models'].append({
                'name': model_file.name,
                'size': f"{size_mb:.1f} MB",
                'modified': modified,
                'path': str(model_file),
                'eval_result': eval_result
            })
    info['dagger_model'] = dagger_model_info
    
    # Expert demos目录
    expert_info = {
        'path': str(dirs['expert_demos'].relative_to(project_root)),
        'episodes': []
    }
    if dirs['expert_demos'].exists():
        episodes = sorted(dirs['expert_demos'].glob('episode_*'))
        for ep in episodes[:10]:  # 只显示前10个
            frame_count = len(list(ep.glob('frame_*.npy')))
            expert_info['episodes'].append({
                'name': ep.name,
                'frames': frame_count
            })
        if len(episodes) > 10:
            expert_info['episodes'].append({
                'name': f'... 还有 {len(episodes) - 10} 个episodes',
                'frames': '—'
            })
    info['expert_demos'] = expert_info
    
    # Expert labels目录
    labels_info = {
        'path': str(dirs['expert_labels'].relative_to(project_root)),
        'files': []
    }
    if dirs['expert_labels'].exists():
        for label_file in sorted(dirs['expert_labels'].glob('iter_*.pkl')):
            size_kb = label_file.stat().st_size / 1024
            labels_info['files'].append({
                'name': label_file.name,
                'size': f"{size_kb:.1f} KB"
            })
    info['expert_labels'] = labels_info
    
    # Policy states目录
    states_info = {
        'path': str(dirs['policy_states'].relative_to(project_root)),
        'iterations': []
    }
    if dirs['policy_states'].exists():
        for iter_dir in sorted(dirs['policy_states'].glob('iter_*')):
            state_count = len(list(iter_dir.glob('episode_*.npy')))
            states_info['iterations'].append({
                'name': iter_dir.name,
                'states': state_count
            })
    info['policy_states'] = states_info
    
    # Dagger目录
    dagger_info = {
        'path': str(dirs['dagger'].relative_to(project_root)),
        'files': []
    }
    if dirs['dagger'].exists():
        for dagger_file in sorted(dirs['dagger'].glob('combined_iter_*.pkl')):
            size_mb = dagger_file.stat().st_size / (1024 * 1024)
            dagger_info['files'].append({
                'name': dagger_file.name,
                'size': f"{size_mb:.1f} MB"
            })
    info['dagger'] = dagger_info
    
    return info


def create_task(task_id, config):
    """创建新任务"""
    if task_exists(task_id):
        return False, "任务已存在"
    
    # 创建任务目录结构
    dirs = get_task_dirs(task_id)
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config['task_id'] = task_id
    config['created_at'] = datetime.now().isoformat()
    save_task_config(task_id, config)
    
    return True, "任务创建成功"


def delete_task(task_id):
    """删除任务（仅删除配置，保留数据）"""
    config_file = get_task_path(task_id) / 'config.json'
    if config_file.exists():
        config_file.unlink()
    return True, "任务已删除（数据文件保留）"


# ============================================================================
# 辅助函数
# ============================================================================

def log_message(message):
    """添加日志消息"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    task_status['logs'].append(log_entry)
    task_status['message'] = message
    print(log_entry)


def run_command(cmd, cwd=None):
    """运行命令并实时获取输出"""
    global current_process
    
    log_message(f"执行命令: {cmd}")
    
    import os
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd or str(project_root),
        preexec_fn=os.setsid  # 创建新的进程组，以便可以杀掉所有子进程
    )
    
    current_process = process  # 保存进程引用
    
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                task_status['logs'].append(line.rstrip())
        
        process.wait()
        return process.returncode
    except Exception as e:
        log_message(f"命令执行异常: {str(e)}")
        return -1
    finally:
        current_process = None


# ============================================================================
# 路由：任务管理
# ============================================================================

@app.route('/')
def index():
    """主页 - 任务列表"""
    return render_template('tasks.html')


@app.route('/task/<task_id>')
def task_page(task_id):
    """任务训练页面"""
    if not task_exists(task_id):
        return "任务不存在", 404
    
    config = get_task_config(task_id)
    return render_template('training.html', task_id=task_id, config=config)


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """获取所有任务列表"""
    task_ids = list_all_tasks()
    
    tasks = {}
    for task_id in task_ids:
        config = get_task_config(task_id)
        stats = get_task_stats(task_id)
        
        tasks[task_id] = {
            **config,
            'stats': stats
        }
    
    return jsonify(tasks)


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task_api(task_id):
    """获取单个任务的详细信息"""
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    config = get_task_config(task_id)
    stats = get_task_stats(task_id)
    
    return jsonify({
        **config,
        'stats': stats
    })


@app.route('/api/tasks/<task_id>/config', methods=['PUT'])
def update_task_config_api(task_id):
    """更新任务配置"""
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    try:
        new_config = request.json
        if not new_config:
            return jsonify({'error': '配置数据为空'}), 400
        
        # 读取现有配置
        current_config = get_task_config(task_id)
        
        # 更新配置（合并）
        current_config.update(new_config)
        
        # 保存配置
        save_task_config(task_id, current_config)
        
        return jsonify({
            'success': True,
            'message': '配置已更新',
            'config': current_config
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config_template', methods=['GET'])
def get_config_template():
    """获取任务配置模板"""
    return jsonify({
        'default_config': DEFAULT_TASK_CONFIG,
        'fields': CONFIG_FIELDS,
        'categories': CONFIG_CATEGORIES
    })


@app.route('/api/tasks', methods=['POST'])
def create_task_api():
    """创建新任务（使用完整配置）"""
    data = request.json
    task_id = data.get('task_id')
    
    if not task_id:
        return jsonify({'error': '任务ID不能为空'}), 400
    
    # 验证任务ID格式（只允许字母、数字、下划线）
    if not task_id.replace('_', '').replace('-', '').isalnum():
        return jsonify({'error': '任务ID只能包含字母、数字、下划线和连字符'}), 400
    
    # 从请求中获取完整配置，使用默认值填充缺失字段
    config = DEFAULT_TASK_CONFIG.copy()
    config['task_id'] = task_id
    
    # 更新用户提供的配置
    for key in DEFAULT_TASK_CONFIG.keys():
        if key in data:
            config[key] = data[key]
    
    success, message = create_task(task_id, config)
    
    if success:
        return jsonify({'success': True, 'task': get_task_config(task_id)})
    else:
        return jsonify({'error': message}), 400


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task_api(task_id):
    """删除任务"""
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    success, message = delete_task(task_id)
    return jsonify({'success': True, 'message': message})


@app.route('/api/tasks/<task_id>', methods=['PUT'])
def update_task_api(task_id):
    """更新任务配置"""
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    config = get_task_config(task_id)
    config.update(request.json)
    save_task_config(task_id, config)
    
    return jsonify({'success': True, 'task': config})


@app.route('/api/current_task', methods=['GET', 'POST'])
def current_task():
    """获取或设置当前任务"""
    global current_task_id
    
    if request.method == 'GET':
        return jsonify({'task_id': current_task_id})
    else:
        task_id = request.json.get('task_id')
        
        if not task_exists(task_id):
            return jsonify({'error': '任务不存在'}), 404
        
        current_task_id = task_id
        return jsonify({'success': True, 'task_id': task_id})


@app.route('/api/status')
def status():
    """获取当前任务状态"""
    return jsonify(task_status)


@app.route('/api/stop', methods=['POST'])
def stop_task():
    """停止当前运行的任务"""
    global current_process
    
    if not task_status['running']:
        return jsonify({'error': '没有正在运行的任务'}), 400
    
    if current_process is None:
        return jsonify({'error': '无法找到运行中的进程'}), 400
    
    try:
        import signal
        import os
        
        # 终止进程组（包括所有子进程）
        if current_process.poll() is None:  # 进程还在运行
            pgid = os.getpgid(current_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            
            # 等待一下，如果还没结束就强制kill
            try:
                current_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
            
            log_message("")
            log_message("=" * 60)
            log_message("⚠️  任务已被用户停止")
            log_message("=" * 60)
            
            task_status['running'] = False
            task_status['stage'] = 'stopped'
            task_status['message'] = '任务已停止'
            
            return jsonify({'success': True, 'message': '任务已停止'})
        else:
            return jsonify({'error': '进程已结束'}), 400
            
    except Exception as e:
        log_message(f"停止任务失败: {str(e)}")
        return jsonify({'error': f'停止失败: {str(e)}'}), 500


@app.route('/api/tasks/<task_id>/directory')
def get_task_directory(task_id):
    """获取任务目录详细信息"""
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    info = get_task_directory_info(task_id)
    return jsonify(info)


@app.route('/api/tasks/<task_id>/evaluate', methods=['POST'])
def evaluate_model(task_id):
    """评估模型"""
    if task_status['running']:
        return jsonify({'error': '任务正在运行中'}), 400
    
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    data = request.json
    model_name = data.get('model_name')
    episodes = data.get('episodes')  # 获取自定义评估次数
    
    if not model_name:
        return jsonify({'error': '缺少model_name参数'}), 400
    
    # 在后台线程中运行
    thread = threading.Thread(target=_evaluate_model_task, args=(task_id, model_name, episodes))
    thread.start()
    
    return jsonify({'success': True, 'message': f'开始评估 {model_name}'})


@app.route('/api/tasks/<task_id>/relabel_iteration', methods=['POST'])
def relabel_iteration(task_id):
    """清除指定迭代的专家标注数据，以便重新标注"""
    if not task_exists(task_id):
        return jsonify({'error': '任务不存在'}), 404
    
    data = request.json
    iteration = data.get('iteration')
    
    if not iteration:
        return jsonify({'error': '缺少iteration参数'}), 400
    
    try:
        dirs = get_task_dirs(task_id)
        label_file = dirs['expert_labels'] / f'iter_{iteration}.pkl'
        
        if label_file.exists():
            label_file.unlink()
            return jsonify({
                'success': True, 
                'message': f'已清除迭代 {iteration} 的专家标注数据，可以重新标注'
            })
        else:
            return jsonify({'error': f'迭代 {iteration} 的标注数据不存在'}), 404
            
    except Exception as e:
        return jsonify({'error': f'清除失败: {str(e)}'}), 500


def _evaluate_model_task(task_id, model_name, episodes=None):
    """评估模型任务（后台线程）- 调用 run_evaluation.sh"""
    task_status['running'] = True
    task_status['stage'] = 'evaluating'
    task_status['progress'] = 0
    task_status['logs'] = []
    
    try:
        config = get_task_config(task_id)
        dirs = get_task_dirs(task_id)
        
        # 使用自定义episodes或配置中的默认值
        eval_episodes = episodes if episodes is not None else config.get('eval_episodes', 20)
        
        log_message("=" * 70)
        log_message(f"评估模型: {model_name}")
        log_message("=" * 70)
        
        # 根据模型名称判断在哪个目录
        if 'bc_baseline' in model_name:
            model_path = dirs['baseline_model'] / model_name
        elif 'dagger_iter' in model_name:
            model_path = dirs['dagger_model'] / model_name
        else:
            # 默认先找baseline，再找dagger
            model_path = dirs['baseline_model'] / model_name
            if not model_path.exists():
                model_path = dirs['dagger_model'] / model_name
        
        if not model_path.exists():
            raise Exception(f"模型不存在: {model_path}")
        
        # 构建命令参数
        cmd_parts = [
            "bash scripts/run_evaluation.sh",
            f"--task {task_id}",
            f"--model {model_path}",
            f"--episodes {eval_episodes}",
            f"--max-steps {config.get('max_steps', 1000)}",
        ]
        
        cmd = " ".join(cmd_parts)
        log_message("执行命令:")
        log_message(cmd)
        log_message("=" * 70)
        log_message("")
        
        task_status['progress'] = 10
        returncode = run_command(cmd)
        
        if returncode != 0:
            raise Exception(f"评估失败，退出码: {returncode}")
        
        task_status['progress'] = 100
        log_message("")
        log_message("=" * 60)
        log_message(f"✅ {model_name} 评估完成！")
        log_message("=" * 60)
        
    except Exception as e:
        log_message("")
        log_message("=" * 60)
        log_message(f"❌ 错误: {str(e)}")
        log_message("=" * 60)
        import traceback
        log_message(traceback.format_exc())
    finally:
        task_status['running'] = False
        task_status['stage'] = 'completed'


# ============================================================================
# 功能1：录制和评估基线
# ============================================================================

@app.route('/api/record_and_train', methods=['POST'])
def record_and_train():
    """录制专家演示并训练BC基线"""
    if task_status['running']:
        return jsonify({'error': '任务正在运行中'}), 400
    
    if not current_task_id:
        return jsonify({'error': '请先选择任务'}), 400
    
    # 在后台线程中运行
    thread = threading.Thread(target=_record_and_train_task, args=(current_task_id,))
    thread.start()
    
    return jsonify({'success': True, 'message': '任务已启动'})


def _record_and_train_task(task_id):
    """录制和训练任务（后台线程）- 调用 run_recording_and_baseline.sh"""
    task_status['running'] = True
    task_status['stage'] = 'recording'
    task_status['progress'] = 0
    task_status['logs'] = []
    
    try:
        config = get_task_config(task_id)
        
        log_message("=" * 70)
        log_message(f"录制专家演示和训练BC基线 - 任务: {task_id}")
        log_message("=" * 70)
        
        # 构建命令参数
        cmd_parts = [
            "bash scripts/run_recording_and_baseline.sh",
            f"--task {task_id}",
            f"--num-episodes {config.get('num_expert_episodes', 10)}",
            f"--mouse-sensitivity {config.get('mouse_sensitivity', 0.15)}",
            f"--max-frames {config.get('max_frames', 6000)}",
            f"--bc-epochs {config.get('bc_epochs', 50)}",
            f"--bc-learning-rate {config.get('bc_learning_rate', 0.0003)}",
            f"--bc-batch-size {config.get('bc_batch_size', 64)}",
            f"--device {config.get('device', 'mps')}",
            f"--eval-episodes {config.get('eval_episodes', 20)}",
            f"--max-steps {config.get('max_steps', 1000)}",
        ]
        
        # 可选参数
        if config.get('fullscreen'):
            cmd_parts.append("--fullscreen")
        if not config.get('skip_idle_frames', True):
            cmd_parts.append("--no-skip-idle")
        
        cmd = " ".join(cmd_parts)
        log_message("执行命令:")
        log_message(cmd)
        log_message("=" * 70)
        log_message("")
        
        task_status['progress'] = 10
        returncode = run_command(cmd)
        
        if returncode != 0:
            raise Exception("录制和训练失败")
        
        task_status['progress'] = 100
        log_message("")
        log_message("=" * 70)
        log_message("✅ 录制和BC训练完成！")
        log_message("=" * 70)
        
    except Exception as e:
        log_message(f"\n❌ 错误: {str(e)}")
    finally:
        task_status['running'] = False
        task_status['stage'] = 'completed'


# ============================================================================
# 功能2：DAgger 迭代
# ============================================================================

@app.route('/api/dagger_iteration', methods=['POST'])
def dagger_iteration():
    """执行DAgger迭代（调用 run_dagger_workflow.sh）"""
    if task_status['running']:
        return jsonify({'error': '任务正在运行中'}), 400
    
    if not current_task_id:
        return jsonify({'error': '请先选择任务'}), 400
    
    data = request.json
    mode = data.get('mode', 'continue')  # 'continue' 或 'restart'
    clean_data = data.get('clean_data', False)  # 是否清理历史数据
    
    # 在后台线程中运行
    thread = threading.Thread(target=_dagger_iteration_task, args=(current_task_id, mode, clean_data))
    thread.start()
    
    return jsonify({'success': True, 'message': 'DAgger 迭代已启动'})


def _dagger_iteration_task(task_id, mode='continue', clean_data=False):
    """DAgger迭代任务（后台线程）- 调用 run_dagger_iteration.sh"""
    task_status['running'] = True
    task_status['stage'] = 'dagger_iteration'
    task_status['progress'] = 0
    task_status['logs'] = []
    
    try:
        config = get_task_config(task_id)
        dirs = get_task_dirs(task_id)
        stats = get_task_stats(task_id)
        
        log_message("=" * 70)
        log_message(f"DAgger 迭代训练 - 任务: {task_id}")
        log_message(f"模式: {'继续迭代' if mode == 'continue' else '重新开始'}")
        if clean_data:
            log_message("清理模式: 将删除历史DAgger数据")
        log_message("=" * 70)
        
        # 构建命令参数
        cmd_parts = [
            "bash scripts/run_dagger_iteration.sh",
            f"--task {task_id}",
            f"--iterations 1",  # 每次只执行一轮
            f"--collect-episodes {config.get('collect_episodes', 20)}",
            f"--dagger-epochs {config.get('dagger_epochs', 30)}",
            f"--device {config.get('device', 'mps')}",
            f"--failure-window {config.get('failure_window', 10)}",
            f"--random-sample-rate {config.get('random_sample_rate', 0.1)}",
            f"--eval-episodes {config.get('eval_episodes', 20)}",
            f"--max-steps {config.get('max_steps', 1000)}",
            "--skip-eval",  # 跳过迭代后的自动评估（在Web UI中手动评估）
        ]
        
        # 清理历史数据（如果需要）
        if clean_data:
            cmd_parts.append("--clean-restart")
            log_message("将清理历史DAgger数据，从BC基线重新开始")
        # 根据mode决定是否继续现有模型
        elif mode == 'continue' and stats['latest_iteration'] > 0:
            # 继续迭代：从最后一个模型开始
            last_model_name = stats['dagger_models'][-1] if stats['dagger_models'] else 'bc_baseline.zip'
            if 'dagger_iter' in last_model_name:
                last_model_path = dirs['dagger_model'] / last_model_name
            else:
                last_model_path = dirs['baseline_model'] / last_model_name
            
            cmd_parts.append(f"--continue-from {last_model_path}")
            cmd_parts.append(f"--start-iteration {stats['latest_iteration'] + 1}")
            log_message(f"继续迭代: 第 {stats['latest_iteration'] + 1} 轮")
            log_message(f"基于模型: {last_model_name}")
        else:
            # 重新开始：从BC基线开始
            bc_model = dirs['baseline_model'] / 'bc_baseline.zip'
            cmd_parts.append(f"--continue-from {bc_model}")
            cmd_parts.append("--start-iteration 1")
            log_message("重新开始: 从BC基线开始第 1 轮迭代")
        
        cmd = " ".join(cmd_parts)
        log_message("")
        log_message("执行命令:")
        log_message(cmd)
        log_message("=" * 70)
        log_message("")
        
        task_status['progress'] = 10
        returncode = run_command(cmd)
        
        if returncode != 0:
            raise Exception("DAgger 迭代失败")
        
        task_status['progress'] = 100
        log_message("")
        log_message("=" * 70)
        log_message("✅ DAgger 迭代完成！")
        log_message("=" * 70)
        
    except Exception as e:
        log_message(f"\n❌ 错误: {str(e)}")
    finally:
        task_status['running'] = False
        task_status['stage'] = 'completed'


# ============================================================================
# 注意：标注功能通过调用 label_states.py 脚本实现，不在Web层独立实现
# ============================================================================


# ============================================================================
# 启动服务器
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DAgger Web 控制台 - 多任务管理")
    print("=" * 70)
    print(f"任务根目录: {TASKS_ROOT}")
    print(f"访问: http://localhost:5000")
    print("=" * 70 + "\n")
    
    # 配置 Werkzeug 日志（屏蔽频繁请求）
    werkzeug_log = logging.getLogger('werkzeug')
    werkzeug_log.setLevel(logging.WARNING)  # 只显示警告和错误
    
    # 应用过滤器
    for handler in werkzeug_log.handlers:
        handler.addFilter(RequestFilter())
    
    # 关闭自动重载，避免修改代码时中断正在运行的任务
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
