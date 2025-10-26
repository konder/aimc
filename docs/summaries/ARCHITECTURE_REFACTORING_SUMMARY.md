# 项目架构重构总结

> **综合记录项目结构优化、脚本重构、Web架构改进**

---

## 📋 目录

1. [目录结构重构](#-目录结构重构)
2. [脚本分拆与统一](#-脚本分拆与统一)
3. [Web架构重构](#-web架构重构)
4. [总体效果](#-总体效果)

---

## 📁 目录结构重构

### 改进动机

**旧结构问题**：
- ❌ 任务数据分散在多个目录
- ❌ 检查点目录与数据目录分离
- ❌ 不易于备份和迁移
- ❌ Web控制台和src不统一

### 新目录结构

```
aimc/
├── src/
│   ├── web/                      # ✅ Web作为src的模块
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── task_config_template.py
│   │   └── templates/
│   ├── training/
│   │   ├── train_bc.py
│   │   └── train_dagger.py
│   └── utils/
│
├── data/
│   ├── tasks/                    # ✅ 所有任务集中管理
│   │   └── harvest_1_log/
│   │       ├── config.yaml       # ✅ 任务配置
│   │       ├── baseline_model/   # ✅ BC基线模型
│   │       │   ├── bc_baseline.zip
│   │       │   └── bc_baseline_eval_results.npy
│   │       ├── dagger_model/     # ✅ DAgger迭代模型
│   │       │   ├── dagger_iter_1.zip
│   │       │   └── dagger_iter_N.zip
│   │       ├── expert_demos/
│   │       ├── expert_labels/
│   │       ├── policy_states/
│   │       └── dagger/
│   ├── clip_tokenizer/           # 共享资源
│   └── mineclip/                 # 共享资源
│
├── scripts/                      # ✅ 所有脚本统一管理
│   ├── start_web.sh
│   ├── stop_web.sh
│   ├── run_dagger_workflow.sh
│   ├── run_recording_and_baseline.sh
│   └── run_dagger_iteration.sh
│
└── requirements.txt              # ✅ 统一依赖管理
```

### 关键改进

#### 1. 任务目录统一

**旧路径** → **新路径**

| 数据类型 | 旧路径 | 新路径 |
|---------|--------|--------|
| 专家演示 | `data/expert_demos/harvest_1_log/` | `data/tasks/harvest_1_log/expert_demos/` |
| 检查点 | `checkpoints/dagger/harvest_1_log/` | `data/tasks/harvest_1_log/baseline_model/`<br>`data/tasks/harvest_1_log/dagger_model/` |
| 专家标注 | `data/expert_labels/harvest_1_log/` | `data/tasks/harvest_1_log/expert_labels/` |
| 策略状态 | `data/policy_states/harvest_1_log/` | `data/tasks/harvest_1_log/policy_states/` |

#### 2. 模型分类存放

**问题**：BC基线和DAgger迭代模型混在一起

**解决**：
```
baseline_model/          # BC基线专用
  └── bc_baseline.zip

dagger_model/           # DAgger迭代专用
  ├── dagger_iter_1.zip
  ├── dagger_iter_2.zip
  └── dagger_iter_N.zip
```

**优势**：
- ✅ 职责清晰，一目了然
- ✅ 易于管理和备份
- ✅ 将来可扩展（ppo_model/等）

#### 3. Web模块化

**旧结构**：
```
web/                    # 独立目录
├── requirements.txt    # ❌ 重复依赖管理
├── install_deps.sh
├── start_web.sh
└── app.py
```

**新结构**：
```
src/web/               # ✅ 作为src的标准模块
├── __init__.py       # ✅ Python包
├── app.py
├── task_config_template.py
└── templates/

scripts/              # ✅ 脚本统一位置
├── start_web.sh
└── stop_web.sh

requirements.txt      # ✅ 统一依赖
```

### 迁移影响

#### 更新的文件

- ✅ `src/web/app.py` - 更新 `TASKS_ROOT` 路径
- ✅ `scripts/run_dagger_workflow.sh` - 更新所有路径变量
- ✅ `scripts/start_web.sh` - 修改启动方式
- ✅ `scripts/stop_web.sh` - 修改进程匹配
- ✅ `README.md` - 所有示例路径
- ✅ `FAQ.md` - 所有问题解答中的路径
- ✅ 所有文档 - 路径引用

#### 无需修改

- ✅ Python训练脚本 - 使用参数传递路径
- ✅ 工具脚本 - 接受路径参数

---

## 🔧 脚本分拆与统一

### 设计原则

**Web层作为控制台，不实现业务逻辑**

- Web层：读取配置 → 组装参数 → 调用脚本
- 脚本层：执行业务逻辑 → 调用工具
- 工具层：具体功能实现

### 脚本架构

#### 旧架构 ❌

```
Web Backend (Python) → 直接调用 Python 脚本
                    ├→ record_manual_chopping.py
                    ├→ train_bc.py
                    └→ evaluate_policy.py
                    
                    → run_dagger_workflow.sh (部分参数)
```

**问题**：
- Web层实现业务逻辑
- 调用方式不统一
- 参数传递不完整

#### 新架构 ✅

```
Web Backend → 读取 config.yaml
           → 组装参数
           ├→ run_recording_and_baseline.sh
           └→ run_dagger_iteration.sh
```

**优势**：
- 架构统一清晰
- Web层只负责调用
- 参数完整传递
- 易于维护和扩展

### 新增脚本

#### 1. `scripts/run_recording_and_baseline.sh`

**功能**：录制 → BC训练 → 评估

**参数**：
```bash
--task TASK_ID              # 任务ID
--num-episodes N            # 录制数量
--mouse-sensitivity N       # 鼠标灵敏度
--max-frames N              # 最大帧数
--bc-epochs N               # BC训练轮数
--bc-learning-rate N        # BC学习率
--bc-batch-size N           # BC批次大小
--device DEVICE             # 训练设备
--eval-episodes N           # 评估episodes
--skip-recording            # 跳过录制
--skip-bc                   # 跳过BC训练
--skip-eval                 # 跳过评估
```

**示例**：
```bash
bash scripts/run_recording_and_baseline.sh \
    --task harvest_1_log \
    --num-episodes 10 \
    --bc-epochs 50 \
    --device mps
```

#### 2. `scripts/run_dagger_iteration.sh`

**功能**：收集 → 标注 → 训练 → 评估

**参数**：
```bash
--task TASK_ID              # 任务ID
--iterations N              # 迭代次数
--collect-episodes N        # 收集episodes
--dagger-epochs N           # 训练轮数
--device DEVICE             # 训练设备
--failure-window N          # 失败前N步标注
--random-sample-rate N      # 成功采样率
--eval-episodes N           # 评估episodes
--skip-eval                 # 跳过评估
--continue-from MODEL       # 从指定模型继续
--start-iteration N         # 起始迭代编号
```

**示例**：
```bash
# 执行第一轮DAgger迭代
bash scripts/run_dagger_iteration.sh \
    --task harvest_1_log \
    --iterations 1 \
    --collect-episodes 20

# 继续迭代
bash scripts/run_dagger_iteration.sh \
    --task harvest_1_log \
    --continue-from data/tasks/harvest_1_log/dagger_model/dagger_iter_2.zip \
    --start-iteration 3 \
    --iterations 1
```

### Web层调用

#### 录制和训练

```python
def _record_and_train_task(task_id):
    """录制和训练任务（后台线程）"""
    config = get_task_config(task_id)  # 读取YAML配置
    
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
    
    cmd = " ".join(cmd_parts)
    run_command(cmd)
```

#### DAgger迭代

```python
def _dagger_iteration_task(task_id, mode='continue'):
    """DAgger迭代任务（后台线程）"""
    config = get_task_config(task_id)
    
    cmd_parts = [
        "bash scripts/run_dagger_iteration.sh",
        f"--task {task_id}",
        f"--iterations 1",
        f"--collect-episodes {config.get('collect_episodes', 20)}",
        f"--dagger-epochs {config.get('dagger_epochs', 30)}",
        f"--device {config.get('device', 'mps')}",
        f"--failure-window {config.get('failure_window', 10)}",
        f"--random-sample-rate {config.get('random_sample_rate', 0.1)}",
        f"--eval-episodes {config.get('eval_episodes', 20)}",
        "--skip-eval",
    ]
    
    # 处理 continue/restart 模式
    # ...
    
    cmd = " ".join(cmd_parts)
    run_command(cmd)
```

### 参数流转

```
1. 用户在 Web UI 创建/修改任务配置
   ↓
2. 保存到 data/tasks/{task_id}/config.yaml
   ↓
3. Web 后端读取 config.yaml
   ↓
4. 组装命令行参数
   ↓
5. 调用 Shell 脚本
   ↓
6. 脚本调用 Python 工具
   ↓
7. 实际执行
```

**示例**：`failure_window` 参数

```yaml
# config.yaml
failure_window: 15
```

```python
# Web 后端
config = get_task_config('harvest_1_log')
failure_window = config.get('failure_window', 10)  # 15
```

```bash
# 命令行
bash scripts/run_dagger_iteration.sh --failure-window 15
```

```bash
# 脚本内
FAILURE_WINDOW=15
python src/training/dagger/label_states.py --failure-window $FAILURE_WINDOW
```

---

## 🌐 Web架构重构

### 完成的改进

#### 1. 模块化结构

- ✅ Web作为 `src/web/` 模块
- ✅ 创建 `__init__.py` 使其成为Python包
- ✅ 统一依赖管理（`requirements.txt`）
- ✅ 脚本集中到 `scripts/`

#### 2. 功能增强

**自定义评估次数**：
```javascript
// 点击评估时弹出对话框
const episodes = await showDialog('评估模型', '请输入评估次数', {
    type: 'input',
    defaultValue: CONFIG.eval_episodes || 20
});

fetch('/api/tasks/task_id/evaluate', {
    method: 'POST',
    body: JSON.stringify({ model_name, episodes })
});
```

**停止任务功能**：
```python
# 进程组管理
process = subprocess.Popen(
    cmd,
    shell=True,
    preexec_fn=os.setsid  # 创建新进程组
)

# 终止整个进程组
pgid = os.getpgid(process.pid)
os.killpg(pgid, signal.SIGTERM)
```

**美化交互弹框**：
- 自定义对话框替代原生 `alert()`, `confirm()`, `prompt()`
- 支持输入框、选择框、提示框、确认框
- 紫色渐变头部，现代化设计

#### 3. DAgger迭代简化

**旧流程**（复杂）：
```
录制 → 训练BC → 评估BC → 开始迭代 → 收集 → 标注 → 训练 → 评估
```

**新流程**（简化）：
```
点击"开始 DAgger 迭代" → 选择模式 → 一键完成
```

**支持两种模式**：
- **继续迭代**：从最后一个模型继续
- **重新开始**：从BC基线重新开始

#### 4. 配置管理系统

**配置模板** (`src/web/task_config_template.py`)：
```python
DEFAULT_TASK_CONFIG = {
    'task_id': '',
    'max_steps': 1000,
    'bc_epochs': 50,
    'bc_learning_rate': 0.0003,
    'bc_batch_size': 64,
    'device': 'mps',
    'dagger_iterations': 3,
    'collect_episodes': 20,
    'dagger_epochs': 30,
    'eval_episodes': 20,
    'num_expert_episodes': 10,
    'mouse_sensitivity': 0.15,
    'max_frames': 6000,
    'smart_sampling': True,
    'failure_window': 10,
    'random_sample_rate': 0.1,
}
```

**YAML配置文件**：
```yaml
_metadata:
  created_at: '2025-10-25T16:00:00'
  updated_at: '2025-10-25T16:00:00'

# 所有参数集中管理
task_id: harvest_1_log
max_steps: 1000
# ... 更多配置
```

**配置读取**：
```python
def get_task_config(task_id):
    """优先读取 YAML，兼容 JSON"""
    config_yaml = task_path / 'config.yaml'
    config_json = task_path / 'config.json'
    
    if config_yaml.exists():
        return yaml.safe_load(f)
    elif config_json.exists():
        return json.load(f)
    else:
        return DEFAULT_TASK_CONFIG
```

#### 5. 其他改进

- ✅ 关闭Flask自动重载（避免中断任务）
- ✅ 实时日志显示
- ✅ 进度条和状态指示
- ✅ 任务卡片和详情页美化

---

## 🎯 总体效果

### 架构清晰度

**改进前**：
```
❌ 任务数据分散
❌ Web独立存在
❌ 脚本调用不统一
❌ 参数硬编码
```

**改进后**：
```
✅ 任务数据集中在 data/tasks/
✅ Web是 src/ 的标准模块
✅ 统一通过脚本调用
✅ 配置驱动，参数集中管理
```

### 易用性提升

| 操作 | 改进前 | 改进后 | 提升 |
|-----|-------|-------|------|
| 创建任务 | 手动创建多个目录 | Web界面一键创建 | **80%简化** |
| 修改配置 | 修改多个脚本 | 编辑单个YAML文件 | **90%简化** |
| DAgger迭代 | 5步操作 | 1步操作 | **80%简化** |
| 任务备份 | 复制多个目录 | `tar -czf backup.tar.gz data/tasks/task_id/` | **简单直观** |

### 可维护性

- ✅ **单一数据源**：所有配置从YAML读取
- ✅ **职责分离**：Web层、脚本层、工具层各司其职
- ✅ **易于扩展**：添加新任务只需创建目录
- ✅ **文档完善**：详细的使用指南和技术文档

### 开发体验

- ✅ **模块化代码**：src/web/作为标准Python包
- ✅ **统一依赖**：requirements.txt集中管理
- ✅ **脚本集中**：所有脚本在scripts/目录
- ✅ **配置模板**：新任务使用默认配置

---

## 📚 相关文档

### 用户指南
- [Web完整指南](../guides/WEB_COMPREHENSIVE_GUIDE.md) - Web控制台使用
- [DAgger完整指南](../guides/DAGGER_COMPREHENSIVE_GUIDE.md) - DAgger算法
- [配置文件指南](../guides/CONFIG_YAML_SUPPORT.md) - YAML配置

### 技术文档
- [目录结构设计](../design/NEW_DIRECTORY_STRUCTURE.md) - 目录结构详解
- [CNN架构](../technical/DAGGER_CNN_ARCHITECTURE.md) - 模型架构

### 历史记录
- [Web改进总结](WEB_IMPROVEMENTS_SUMMARY.md) - Web功能迭代
- [Web重构完成](WEB_RESTRUCTURE.md) - Web模块化重构

---

## 🎓 最佳实践

### 1. 目录管理

- ✅ 每个任务独立目录
- ✅ 定期备份重要任务
- ✅ 使用语义化的task_id命名

### 2. 配置管理

- ✅ 创建任务时设置完整配置
- ✅ 不同任务使用不同配置
- ✅ 重要配置加注释说明

### 3. 脚本使用

- ✅ 优先使用Web控制台
- ✅ 命令行调试使用脚本
- ✅ 自动化训练使用脚本

### 4. 版本控制

- ✅ 提交代码更改
- ✅ 不提交data/目录（.gitignore）
- ✅ 重要模型单独备份

---

## 🎉 总结

这次架构重构实现了：

1. **目录结构优化**：任务数据集中，模型分类存放
2. **脚本架构统一**：Web层→脚本层→工具层
3. **配置驱动开发**：所有参数从YAML读取
4. **Web功能增强**：评估、停止、美化弹框
5. **开发体验提升**：模块化、统一依赖、文档完善

**结果**：
- ✅ 代码更清晰
- ✅ 操作更简单
- ✅ 维护更容易
- ✅ 扩展更灵活

符合**关注点分离**和**配置驱动**的架构原则！🚀

---

**版本**: 2.0  
**更新日期**: 2025-10-25  
**维护者**: AIMC项目组

