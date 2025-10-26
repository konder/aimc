# Web 配置管理改进方案

## 🎯 改进目标

1. ✅ **新增评估脚本** `scripts/run_evaluation.sh`
2. ✅ **Web 评估功能调用脚本** 
3. 🔄 **配置文件按模块分组显示**（创建任务页面）
4. 🔄 **任务详情页支持编辑配置**

---

## ✅ 已完成的工作

### 1. 创建评估脚本 `scripts/run_evaluation.sh`

**功能**: 评估指定模型的性能
**支持参数**:
- `--task`: 任务ID
- `--model`: 模型路径（必需）
- `--episodes`: 评估episode数
- `--max-steps`: 最大步数

**特性**:
- 自动从 `config.yaml` 读取默认参数
- 支持命令行覆盖配置值
- 完整的错误处理和帮助信息

**使用示例**:
```bash
# 评估BC基线
bash scripts/run_evaluation.sh \
    --task harvest_1_log \
    --model data/tasks/harvest_1_log/baseline_model/bc_baseline.zip \
    --episodes 20

# 评估DAgger迭代模型
bash scripts/run_evaluation.sh \
    --task harvest_1_log \
    --model data/tasks/harvest_1_log/dagger_model/dagger_iter_2.zip \
    --episodes 50
```

### 2. Web 后端评估功能改为调用脚本

**修改文件**: `src/web/app.py`
**修改函数**: `_evaluate_model_task()`

**改进前**:
```python
cmd = f"""bash scripts/run_minedojo_x86.sh python src/training/dagger/evaluate_policy.py \
    --model {model_path} \
    --episodes {eval_episodes} \
    --task-id {task_id} \
    --max-steps {config['max_steps']}"""
```

**改进后**:
```python
cmd_parts = [
    "bash scripts/run_evaluation.sh",
    f"--task {task_id}",
    f"--model {model_path}",
    f"--episodes {eval_episodes}",
    f"--max-steps {config.get('max_steps', 1000)}",
]
cmd = " ".join(cmd_parts)
```

### 3. 更新配置模板分类

**修改文件**: `src/web/task_config_template.py`
**变量**: `CONFIG_CATEGORIES`

**改进前**: 简单的字典映射
```python
CONFIG_CATEGORIES = {
    'basic': '基础配置',
    'recording': '录制配置',
    ...
}
```

**改进后**: 结构化分类，包含图标和描述
```python
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
    # ... 更多分类
]
```

**分类列表**:
1. ⚙️ **通用配置**: task_id, max_steps, device
2. 📹 **录制配置**: num_expert_episodes, mouse_sensitivity, max_frames, skip_idle_frames, fullscreen
3. 🎓 **BC训练配置**: bc_epochs, bc_learning_rate, bc_batch_size
4. 🔄 **DAgger迭代配置**: dagger_iterations, collect_episodes, dagger_epochs
5. 🏷️ **标注配置**: smart_sampling, failure_window, random_sample_rate
6. 📊 **评估配置**: eval_episodes

### 4. 新增配置更新 API

**修改文件**: `src/web/app.py`
**新增端点**: `PUT /api/tasks/<task_id>/config`

```python
@app.route('/api/tasks/<task_id>/config', methods=['PUT'])
def update_task_config_api(task_id):
    """更新任务配置"""
    # ... 实现配置更新逻辑
```

**功能**:
- 接收 JSON 格式的配置更新
- 合并到现有配置
- 保存为 YAML 格式
- 返回更新后的配置

---

## 🔄 待完成的工作

### 1. 优化创建任务页面 - 分组显示配置

**文件**: `src/web/templates/tasks.html`

**目标**: 
- 从 API 获取配置模板
- 按分类动态渲染表单
- 使用折叠面板（accordion）展示各个配置模块

**UI 设计**:
```
创建新任务
━━━━━━━━━━━━━━━━━━━━━━━
📋 基本信息
  任务ID: ____________
  任务名称: __________
  任务描述: __________

⚙️ 通用配置 [展开/折叠]
  最大步数: [1000]
  训练设备: [MPS ▼]

📹 录制配置 [展开/折叠]
  录制Episodes: [10]
  鼠标灵敏度: [0.15]
  最大帧数: [6000]
  □ 跳过静止帧
  □ 全屏模式

🎓 BC训练配置 [展开/折叠]
  ...

🔄 DAgger迭代配置 [展开/折叠]
  ...

🏷️ 标注配置 [展开/折叠]
  ...

📊 评估配置 [展开/折叠]
  ...

[🚀 创建任务]
```

**实现步骤**:
1. 修改模态框 HTML 结构
2. 添加 accordion CSS 样式
3. 实现动态加载配置模板的 JavaScript
4. 根据 `CONFIG_CATEGORIES` 渲染分组表单
5. 更新提交逻辑，收集所有配置字段

### 2. 任务详情页添加配置编辑

**文件**: `src/web/templates/training.html`

**目标**:
- 在任务详情页显示当前配置
- 支持在线编辑配置
- 点击"保存配置"按钮更新配置

**UI 设计**:
```
任务详情页
━━━━━━━━━━━━━━━━━━━━━━━
[返回任务列表]  harvest_1_log

📊 任务数据
  专家演示: 10 episodes
  ...

⚙️ 任务配置                [编辑]
  ━━━━━━━━━━━━━━━━━━━━━━━━
  ⚙️ 通用配置
    最大步数: 1000
    训练设备: mps
  
  📹 录制配置
    录制Episodes: 10
    鼠标灵敏度: 0.15
    ...
  
  [编辑模式下显示输入框]
  [💾 保存配置] [❌ 取消]
```

**实现步骤**:
1. 添加配置显示卡片
2. 按分类分组显示配置
3. 添加"编辑"按钮
4. 编辑模式：所有值变为可编辑输入框
5. 调用 `PUT /api/tasks/<task_id>/config` 保存
6. 保存成功后刷新页面

---

## 📊 架构总览

### 评估流程

```
Web UI (点击评估)
  ↓
Web Backend (读取 config.yaml)
  ↓
组装参数
  ↓
调用 run_evaluation.sh
  ↓
run_evaluation.sh (读取 config.yaml 补充参数)
  ↓
调用 evaluate_policy.py
  ↓
执行评估
```

### 配置管理流程

```
1. 创建任务
   Web UI (填写分组表单)
     ↓
   POST /api/tasks
     ↓
   保存到 config.yaml (按分类组织)

2. 查看配置
   Web UI (任务详情页)
     ↓
   GET /api/tasks/<task_id>
     ↓
   读取 config.yaml
     ↓
   分组显示配置

3. 编辑配置
   Web UI (编辑模式)
     ↓
   PUT /api/tasks/<task_id>/config
     ↓
   更新 config.yaml
     ↓
   刷新显示
```

---

## 🔧 使用场景

### 场景1: 创建任务时配置所有参数

1. 用户点击"创建新任务"
2. 填写基本信息（任务ID、名称、描述）
3. 展开"录制配置"，调整录制参数
4. 展开"BC训练配置"，调整训练参数
5. 展开"DAgger迭代配置"，调整迭代参数
6. 展开"标注配置"，调整标注策略
7. 点击"创建任务"
8. 所有配置保存到 `config.yaml`

### 场景2: 调整现有任务配置

1. 进入任务详情页
2. 点击"编辑配置"按钮
3. 修改需要调整的参数（如增加BC训练轮数）
4. 点击"保存配置"
5. 配置更新，下次训练使用新配置

### 场景3: 评估模型

1. 在任务详情页点击"评估"按钮
2. 输入评估episodes数量（或使用配置默认值）
3. Web 后端读取 `config.yaml`
4. 调用 `run_evaluation.sh` 并传递参数
5. 脚本从 `config.yaml` 补充未指定的参数
6. 执行评估，返回结果

---

## ✅ 测试清单

### 已测试 ✅
- [x] `run_evaluation.sh --help` 显示帮助信息
- [x] Web 后端无 lint 错误
- [x] `/api/config_template` 返回正确的分类结构

### 待测试 ⏳
- [ ] 通过 Web UI 创建任务（分组表单）
- [ ] 在任务详情页查看配置（分组显示）
- [ ] 在任务详情页编辑并保存配置
- [ ] 通过 Web UI 评估模型
- [ ] 修改配置后训练使用新配置

---

## 📁 修改的文件

### 新增文件
1. `scripts/run_evaluation.sh` (新建，185行)
2. `docs/WEB_CONFIG_MANAGEMENT_IMPROVEMENTS.md` (本文档)

### 修改文件
1. `src/web/app.py`
   - `_evaluate_model_task()` 改为调用 `run_evaluation.sh`
   - 新增 `PUT /api/tasks/<task_id>/config` 端点

2. `src/web/task_config_template.py`
   - `CONFIG_CATEGORIES` 改为结构化列表

3. `src/web/templates/tasks.html` (待修改)
   - 创建任务表单改为分组显示

4. `src/web/templates/training.html` (待修改)
   - 新增配置显示和编辑功能

---

## 🎯 下一步行动

### 立即实施
1. ✅ 创建 `run_evaluation.sh`
2. ✅ 修改 Web 后端评估功能
3. ✅ 更新配置模板分类
4. ✅ 新增配置更新 API

### 待用户确认后实施
5. ⏳ 修改 `tasks.html` - 分组显示配置（需要大幅度修改HTML和JS）
6. ⏳ 修改 `training.html` - 配置显示和编辑（新增功能模块）

**建议**: 
- 由于前端改动较大，建议先测试已完成的后端功能
- 确认评估脚本和 API 正常工作后
- 再进行前端的配置管理UI改进

---

## 💡 总结

本次改进主要完成了后端架构的统一：

1. **评估功能统一**: Web 评估调用 `run_evaluation.sh`，与录制、迭代保持一致
2. **配置管理增强**: 新增配置更新 API，支持动态修改任务配置
3. **配置结构化**: 配置按模块分类，为前端分组显示做好准备

**架构特点**:
- ✅ Web 层只负责调用脚本
- ✅ 所有参数从 `config.yaml` 读取
- ✅ 脚本层处理业务逻辑
- ✅ 配置文件结构清晰

符合架构原则：**Web 作为控制台，不实现业务逻辑**！🚀

