"""
任务配置模板
基于 run_dagger_workflow.sh 的所有参数
"""

DEFAULT_TASK_CONFIG = {
    # 任务配置
    'task_id': '',
    'max_steps': 1000,
    
    # BC训练配置
    'bc_epochs': 50,
    'bc_learning_rate': 0.0003,
    'bc_batch_size': 64,
    'device': 'mps',  # auto/cpu/cuda/mps
    
    # DAgger配置
    'dagger_iterations': 3,
    'collect_episodes': 20,
    'dagger_epochs': 30,
    
    # 评估配置
    'eval_episodes': 20,
    
    # 录制配置
    'num_expert_episodes': 10,
    'mouse_sensitivity': 0.15,
    'max_frames': 6000,
    'skip_idle_frames': True,
    'fullscreen': False,
    
    # 标注配置
    'smart_sampling': True,
    'failure_window': 10,
    'random_sample_rate': 0.1,
}

# 配置字段描述（用于UI显示）
CONFIG_FIELDS = {
    # 基础配置
    'task_id': {
        'label': '任务ID',
        'type': 'text',
        'required': True,
        'description': '任务唯一标识符，如：harvest_1_log',
        'category': 'basic'
    },
    'max_steps': {
        'label': '最大步数',
        'type': 'number',
        'min': 100,
        'max': 10000,
        'description': '每个episode的最大步数',
        'category': 'basic'
    },
    'device': {
        'label': '训练设备',
        'type': 'select',
        'options': ['auto', 'cpu', 'cuda', 'mps'],
        'description': '训练使用的设备（MPS for Mac, CUDA for NVIDIA GPU）',
        'category': 'basic'
    },
    
    # 录制配置
    'num_expert_episodes': {
        'label': '录制Episodes数',
        'type': 'number',
        'min': 1,
        'max': 100,
        'description': '需要录制的专家演示数量',
        'category': 'recording'
    },
    'mouse_sensitivity': {
        'label': '鼠标灵敏度',
        'type': 'number',
        'min': 0.01,
        'max': 1.0,
        'step': 0.01,
        'description': '控制视角转动的灵敏度',
        'category': 'recording'
    },
    'max_frames': {
        'label': '最大帧数',
        'type': 'number',
        'min': 1000,
        'max': 20000,
        'description': '每个episode最大录制帧数',
        'category': 'recording'
    },
    'skip_idle_frames': {
        'label': '跳过静止帧',
        'type': 'checkbox',
        'description': '录制时自动跳过无操作的帧（节省空间）',
        'category': 'recording'
    },
    'fullscreen': {
        'label': '全屏模式',
        'type': 'checkbox',
        'description': '全屏显示游戏窗口（推荐，防止鼠标移出）',
        'category': 'recording'
    },
    
    # BC训练配置
    'bc_epochs': {
        'label': 'BC训练轮数',
        'type': 'number',
        'min': 10,
        'max': 200,
        'description': 'Behavioral Cloning 训练的epoch数',
        'category': 'training'
    },
    'bc_learning_rate': {
        'label': 'BC学习率',
        'type': 'number',
        'min': 0.00001,
        'max': 0.01,
        'step': 0.00001,
        'description': 'BC训练的学习率',
        'category': 'training'
    },
    'bc_batch_size': {
        'label': 'BC批次大小',
        'type': 'number',
        'min': 16,
        'max': 256,
        'description': 'BC训练的batch size',
        'category': 'training'
    },
    
    # DAgger配置
    'dagger_iterations': {
        'label': 'DAgger迭代次数',
        'type': 'number',
        'min': 1,
        'max': 10,
        'description': 'DAgger算法的迭代轮数',
        'category': 'dagger'
    },
    'collect_episodes': {
        'label': '收集Episodes数',
        'type': 'number',
        'min': 5,
        'max': 100,
        'description': '每轮DAgger收集的episode数量',
        'category': 'dagger'
    },
    'dagger_epochs': {
        'label': 'DAgger训练轮数',
        'type': 'number',
        'min': 10,
        'max': 100,
        'description': '每轮DAgger的训练epoch数',
        'category': 'dagger'
    },
    'failure_window': {
        'label': '失败窗口',
        'type': 'number',
        'min': 5,
        'max': 50,
        'description': '失败前N步需要标注',
        'category': 'dagger'
    },
    'random_sample_rate': {
        'label': '成功采样率',
        'type': 'number',
        'min': 0.0,
        'max': 1.0,
        'step': 0.01,
        'description': '成功episode的随机采样率（0.1 = 10%）',
        'category': 'dagger'
    },
    'smart_sampling': {
        'label': '智能采样',
        'type': 'checkbox',
        'description': '启用智能采样（只标注关键状态）',
        'category': 'dagger'
    },
    
    # 评估配置
    'eval_episodes': {
        'label': '评估Episodes数',
        'type': 'number',
        'min': 5,
        'max': 100,
        'description': '模型评估时运行的episode数量',
        'category': 'evaluation'
    },
}

# 配置分类（用于UI分组显示）
CONFIG_CATEGORIES = [
    {
        'id': 'basic',
        'name': '通用配置',
        'icon': '⚙️',
        'description': '任务的基本设置',
        'fields': ['task_id', 'max_steps', 'device']
    },
    {
        'id': 'recording',
        'name': '录制配置',
        'icon': '📹',
        'description': '专家演示录制相关设置',
        'fields': ['num_expert_episodes', 'mouse_sensitivity', 'max_frames', 'skip_idle_frames', 'fullscreen']
    },
    {
        'id': 'training',
        'name': 'BC训练配置',
        'icon': '🎓',
        'description': '行为克隆（BC）基线训练设置',
        'fields': ['bc_epochs', 'bc_learning_rate', 'bc_batch_size']
    },
    {
        'id': 'dagger',
        'name': 'DAgger迭代配置',
        'icon': '🔄',
        'description': 'DAgger算法迭代训练设置',
        'fields': ['dagger_iterations', 'collect_episodes', 'dagger_epochs']
    },
    {
        'id': 'labeling',
        'name': '标注配置',
        'icon': '🏷️',
        'description': '失败状态标注相关设置',
        'fields': ['smart_sampling', 'failure_window', 'random_sample_rate']
    },
    {
        'id': 'evaluation',
        'name': '评估配置',
        'icon': '📊',
        'description': '模型性能评估设置',
        'fields': ['eval_episodes']
    },
]

