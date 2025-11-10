# DAgger Web 控制台完整指南

> **一站式Web界面，简化DAgger训练全流程**

---

## 📋 目录

1. [快速开始](#-快速开始)
2. [任务管理](#-任务管理)
3. [录制与训练](#-录制与训练)
4. [DAgger迭代](#-dagger迭代)
5. [配置管理](#-配置管理)
6. [高级功能](#-高级功能)
7. [故障排查](#-故障排查)

---

## 🚀 快速开始

### 启动服务器

```bash
# 方式1：使用脚本（推荐）
bash scripts/run_web.sh start

# 方式2：直接运行
cd /Users/nanzhang/aimc
conda activate minedojo
python -m src.web.app
```

### 访问界面

打开浏览器访问：**http://localhost:5000**

### 检查状态

```bash
# 查看Web服务器运行状态
bash scripts/run_web.sh        # 默认为 status
bash scripts/run_web.sh status
```

### 停止服务器

```bash
bash scripts/run_web.sh stop
```

---

## 📁 任务管理

### 目录结构

```
data/tasks/
  └── harvest_1_log/          # 任务目录
      ├── config.yaml         # 任务配置
      ├── baseline_model/     # BC基线模型
      ├── dagger_model/       # DAgger迭代模型
      ├── expert_demos/       # 专家演示
      ├── expert_labels/      # 专家标注
      ├── policy_states/      # 策略收集状态
      └── dagger/             # 聚合数据
```

### 创建新任务

#### 方式1：Web界面

1. 访问主页 http://localhost:5000
2. 点击 **➕ 创建新任务**
3. 填写任务配置：

```yaml
任务ID: harvest_1_log         # 必填，唯一标识
最大步数: 1000
训练设备: mps                 # mps/cuda/cpu/auto

# BC训练配置
BC训练轮数: 50
BC学习率: 0.0003
BC批次大小: 64

# DAgger配置
DAgger迭代次数: 3
收集Episodes: 20
DAgger训练轮数: 30

# 录制配置
专家演示数量: 10
鼠标灵敏度: 0.15
最大帧数: 6000

# 评估配置
评估Episodes: 20
```

4. 点击 **创建任务**

#### 方式2：命令行

```bash
# 创建目录结构
mkdir -p data/tasks/new_task/{baseline_model,dagger_model,expert_demos,expert_labels,policy_states,dagger}

# 复制配置模板
cp data/tasks/harvest_1_log/config.yaml data/tasks/new_task/config.yaml

# 编辑配置
vim data/tasks/new_task/config.yaml
```

刷新Web页面，新任务自动显示！

### 查看任务列表

主页显示所有任务卡片：
- 任务ID和描述
- BC基线模型状态
- DAgger迭代进度
- 快速操作按钮

---

## 📹 录制与训练

### 完整流程

1. **进入任务页面**
   - 点击任务卡片进入详情

2. **开始录制**
   - 点击 **📹 录制专家演示**
   - Pygame窗口自动打开

3. **录制控制**

```
🖱️  鼠标移动   - 转动视角
🖱️  鼠标左键   - 攻击/挖掘
⌨️  WASD       - 移动
⌨️  Space      - 跳跃
⌨️  Shift      - 潜行
⌨️  方向键     - 精确调整视角
⌨️  Q          - 重录当前episode
⌨️  ESC        - 完成录制
```

4. **自动处理**
   - ✅ 保存专家演示
   - ✅ 训练BC基线（读取config.yaml参数）
   - ✅ 评估BC基线
   - ✅ 显示成功率

5. **查看结果**
   - 实时日志显示在控制台
   - 进度条显示训练进度
   - TensorBoard可视化（单独启动）

### 使用配置参数

所有参数从 `data/tasks/{task_id}/config.yaml` 读取：

```yaml
# 录制阶段使用
num_expert_episodes: 10      # 录制数量
mouse_sensitivity: 0.15      # 鼠标灵敏度
max_frames: 6000            # 最大帧数

# BC训练使用
bc_epochs: 50               # 训练轮数
bc_learning_rate: 0.0003    # 学习率
bc_batch_size: 64           # 批次大小
device: mps                 # 训练设备

# 评估使用
eval_episodes: 20           # 评估次数
max_steps: 1000            # 最大步数
```

---

## 🔄 DAgger迭代

### 启动迭代

1. **点击"开始DAgger迭代"**
2. **选择模式**：
   - **🔄 继续迭代**：从最后一个模型继续
   - **🔃 重新开始**：从BC基线重新开始

3. **自动执行**：
   ```
   收集失败状态 (20 episodes)
      ↓
   Web界面标注动作
      ↓
   聚合数据并训练
      ↓
   保存新模型
   ```

### 交互式标注

当收集完成后，Web界面显示：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖼️  状态帧图像
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Episode: 5/20  |  Frame: 150/1000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

动作选择：
  [W] Forward    [S] Backward    [A] Left      [D] Right
  [Space] Jump   [Shift] Sneak   [Click] Attack

  [↑] Pitch Up   [↓] Pitch Down
  [←] Yaw Left   [→] Yaw Right

  [Enter] 确认   [Skip] 跳过   [ESC] 退出
```

### 配置参数

```yaml
# DAgger配置
collect_episodes: 20         # 每轮收集episodes数
dagger_epochs: 30           # 训练轮数
failure_window: 10          # 失败前N步需要标注
random_sample_rate: 0.1     # 成功episodes采样率
smart_sampling: true        # 智能采样
```

### 迭代进度

任务详情页显示：
- 当前迭代轮数
- 每轮成功率变化
- 模型文件列表
- 快速评估按钮

---

## ⚙️ 配置管理

### 配置文件结构

`data/tasks/{task_id}/config.yaml`：

```yaml
_metadata:
  created_at: '2025-10-25T16:00:00'
  updated_at: '2025-10-25T16:00:00'
  description: '任务描述'

# 基础配置
task_id: harvest_1_log
max_steps: 1000

# BC训练配置
bc_epochs: 50
bc_learning_rate: 0.0003
bc_batch_size: 64
device: mps

# DAgger配置
dagger_iterations: 3
collect_episodes: 20
dagger_epochs: 30

# 评估配置
eval_episodes: 20

# 录制配置
num_expert_episodes: 10
mouse_sensitivity: 0.15
max_frames: 6000
skip_idle_frames: true
fullscreen: false

# 标注配置
smart_sampling: true
failure_window: 10
random_sample_rate: 0.1
```

### 修改配置

#### 方式1：Web界面（未来功能）

任务详情页 → ⚙️ 配置 → 编辑 → 保存

#### 方式2：直接编辑

```bash
vim data/tasks/harvest_1_log/config.yaml
```

修改后自动生效（Web后端会重新读取）。

### 配置模板

创建新任务时，使用 `src/web/task_config_template.py` 中的默认值：

```python
DEFAULT_TASK_CONFIG = {
    'task_id': '',
    'max_steps': 1000,
    'bc_epochs': 50,
    'bc_learning_rate': 0.0003,
    'bc_batch_size': 64,
    'device': 'mps',
    # ... 更多配置
}
```

---

## 🎯 高级功能

### 1. 自定义评估次数

评估模型时：
1. 点击 **📊 评估** 按钮
2. 弹出对话框，输入评估次数（默认20）
3. 点击 **开始评估**
4. 查看日志输出和成功率

### 2. 停止运行任务

任务运行时：
- 显示红色 **⏹️ 停止任务** 按钮
- 点击确认后终止进程
- 日志显示 "⚠️ 任务已被用户停止"

### 3. 继续中断的训练

DAgger训练中断后：
1. 点击 **🔄 开始DAgger迭代**
2. 选择 **继续迭代**
3. 自动从最后保存的模型继续

### 4. 实时日志查看

控制台显示：
- 实时命令输出
- 训练进度（epoch、loss等）
- 评估结果
- 错误信息

支持：
- 自动滚动
- 筛选（警告、错误）
- 导出日志

### 5. TensorBoard集成

```bash
# 启动TensorBoard
tensorboard --logdir logs/tensorboard

# 访问
http://localhost:6006
```

查看：
- 训练曲线（loss、explained_variance）
- MineCLIP相似度变化
- 成功率趋势

---

## 🔍 故障排查

### 问题1：Web服务无法启动

**症状**：`Address already in use`

**解决**：
```bash
# 检查端口占用和服务状态
bash scripts/run_web.sh status

# 停止旧进程
bash scripts/run_web.sh stop

# 或手动kill
pkill -f "python.*src.web.app"
```

### 问题2：录制窗口白屏/崩溃

**症状**：Pygame窗口白屏或立即崩溃

**解决**：
```bash
# 方式1：使用headless模式
export JAVA_OPTS="-Djava.awt.headless=true"

# 方式2：重启MineDojo
conda deactivate
conda activate minedojo

# 方式3：检查x86环境
arch  # 应该显示 i386
```

### 问题3：任务不显示

**症状**：创建的任务在Web界面看不到

**解决**：
```bash
# 检查目录结构
ls -la data/tasks/your_task/

# 确保必需目录存在
mkdir -p data/tasks/your_task/{baseline_model,dagger_model}

# 刷新浏览器
```

### 问题4：配置不生效

**症状**：修改config.yaml后没有变化

**解决**：
1. 检查YAML语法（使用空格缩进，不是Tab）
2. 确认文件路径正确
3. 重启Web服务
4. 查看Web日志是否有配置读取错误

### 问题5：标注界面不显示

**症状**：DAgger收集完成后没有标注界面

**解决**：
```bash
# 检查状态文件
ls data/tasks/harvest_1_log/policy_states/iter_1/

# 检查Web日志
tail -f logs/web.log

# 手动标注（如果Web标注不可用）
bash scripts/run_minedojo_x86.sh python src/training/dagger/label_states.py \
    --task harvest_1_log \
    --iteration 1
```

### 问题6：训练速度慢

**优化**：
```yaml
# 减少训练轮数
bc_epochs: 30          # 默认50
dagger_epochs: 20      # 默认30

# 减少收集episodes
collect_episodes: 10   # 默认20

# 使用更强设备
device: cuda           # 如果有NVIDIA GPU
device: mps            # Mac M1/M2
```

---

## 📚 相关文档

### 配置与使用
- [配置文件支持](CONFIG_YAML_SUPPORT.md) - YAML配置详解
- [DAgger完整指南](DAGGER_COMPREHENSIVE_GUIDE.md) - DAgger算法详解
- [DAgger快捷操作](DAGGER_WORKFLOW_SKIP_GUIDE.md) - 跳过步骤指南

### 技术细节
- [CNN架构详解](../technical/DAGGER_CNN_ARCHITECTURE.md) - 模型架构
- [MineCLIP指南](MINECLIP_COMPREHENSIVE_GUIDE.md) - 奖励函数

### 历史记录
- [Web改进总结](../summaries/WEB_IMPROVEMENTS_SUMMARY.md) - 功能迭代历史
- [Web重构记录](../summaries/WEB_RESTRUCTURE.md) - 架构重构

---

## 🎓 最佳实践

### 1. 高质量录制

- ✅ 使用鼠标控制（比键盘更自然）
- ✅ 动作流畅（避免卡顿）
- ✅ 行为一致（同样的情况做同样的动作）
- ✅ 录制多样场景（不同位置、不同树）

### 2. 高效标注

- ✅ 使用智能采样（`smart_sampling: true`）
- ✅ 只标注失败前N步（`failure_window: 10`）
- ✅ 成功episodes低采样率（`random_sample_rate: 0.1`）
- ✅ 标注时专注（质量>数量）

### 3. 合理配置

- ✅ 初期：多录制（10+ episodes），少迭代（1-2轮）
- ✅ 后期：少收集（5 episodes），多迭代（3-5轮）
- ✅ BC成功率 < 60%：增加专家数据
- ✅ BC成功率 > 70%：可以开始DAgger

### 4. 调试技巧

- ✅ 使用test预设快速验证
- ✅ 先跑1轮DAgger确认流程
- ✅ 查看TensorBoard确认学习
- ✅ 定期备份模型和数据

---

## 🎉 总结

**DAgger Web 控制台** 将复杂的命令行操作简化为友好的Web界面：

- ✅ **一键操作**：录制、训练、评估全流程自动化
- ✅ **配置驱动**：所有参数从YAML文件读取
- ✅ **实时反馈**：日志、进度条、状态显示
- ✅ **灵活控制**：继续、重启、停止随心所欲
- ✅ **易于扩展**：基于文件系统，支持任意任务

现在开始你的第一个DAgger训练吧！🚀

---

**版本**: 2.0  
**更新日期**: 2025-10-25  
**维护者**: AIMC项目组

