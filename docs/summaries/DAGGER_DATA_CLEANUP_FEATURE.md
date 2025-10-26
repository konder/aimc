# DAgger 数据清理功能实现总结

**日期**: 2025-10-25  
**更新**: 2025-10-25（修正逻辑）  
**功能**: 为 DAgger 训练流程添加历史数据清理功能

## 问题背景

在 DAgger 训练过程中，如果需要"重头开始"，应该清理相关的历史数据以避免数据混乱。

### 重要原则

⚠️ **专家演示数据非常珍贵（需要人工录制），任何情况下都不应该删除！**

### 需要清理的场景

当需要从BC基线重新开始DAgger训练时，应该删除：
- `expert_labels/` - 专家标注数据
- `policy_states/` - 策略收集的状态
- `dagger/` - 聚合数据
- `dagger_model/` - DAgger迭代模型

### 永远保留的数据

✅ **以下数据永远不删除**：
- `expert_demos/` - 专家演示（人工录制，非常珍贵）
- `baseline_model/` - BC基线模型（基于专家演示训练）

## 实现方案

### 1. `run_dagger_iteration.sh` 修改

**新增参数**:
```bash
--clean-restart    # 清理历史DAgger数据，从BC基线重新开始
```

**清理内容**:
- `policy_states/` 目录下所有内容
- `expert_labels/` 目录下所有内容
- `dagger/` 目录下所有内容
- `dagger_model/` 目录下所有内容

**行为**:
- 强制从 BC 基线模型开始
- 重置迭代编号为 1

**使用示例**:
```bash
# 清理DAgger数据，从BC基线重新开始
bash scripts/run_dagger_iteration.sh --task harvest_1_log --clean-restart
```

### 2. `run_dagger_workflow.sh` 修改

**新增参数**:
```bash
--clean-restart       # 清理DAgger数据，从BC基线重新开始（保留专家演示和BC基线）
--clean-dagger-only   # 同 --clean-restart（兼容旧参数）
```

#### `--clean-restart` 和 `--clean-dagger-only` (清理DAgger数据)

**清理内容**:
- `policy_states/` - 策略状态
- `expert_labels/` - 专家标注
- `dagger/` - 聚合数据
- `dagger_model/` - DAgger模型

**保留内容**:
- `expert_demos/` - 专家演示（保留）
- `baseline_model/` - BC基线模型（保留）

**行为**:
- 跳过录制 (`SKIP_RECORDING=true`)
- 跳过 BC 训练 (`SKIP_BC=true`)
- 从 BC 基线开始 DAgger 迭代

**使用示例**:
```bash
# 清理DAgger数据，从BC基线重新开始
bash scripts/run_dagger_workflow.sh --task harvest_1_log --clean-restart

# 或使用兼容参数
bash scripts/run_dagger_workflow.sh --task harvest_1_log --clean-dagger-only
```

**注意**: 两个参数效果相同，都是清理DAgger数据并保留专家演示和BC基线。

### 3. Web UI 修改

**后端 (`app.py`)**:
- 修改 `/api/dagger_iteration` 接口，新增 `clean_data` 参数
- 修改 `_dagger_iteration_task()` 函数，支持清理数据
- 当 `clean_data=True` 时，向脚本传递 `--clean-restart` 参数

**前端 (`training.html`)**:
- 在 DAgger 迭代对话框中提供两个清晰的选项：
- 选项说明：
  - **🔄 继续迭代**: 使用最新的DAgger模型继续训练（如从第3轮继续第4轮）
  - **🗑️ 清理重启**: 删除所有DAgger数据，从BC基线重新开始第1轮（保留专家演示和BC基线）

**逻辑改进**:
- ❌ 移除了"重新开始（保留历史数据）"选项，因为它会导致文件覆盖混乱
- ✅ 只保留两个清晰的选项：继续或清理重启

**用户体验**:
1. 点击"开始 DAgger 迭代"按钮
2. 如果已有迭代模型，弹出选择对话框
3. 用户可选择清理重启选项
4. 系统自动调用脚本清理数据并重新开始

## 安全机制

所有清理操作都使用了安全的 bash 语法：

```bash
# 检查目录存在且非空
if [ -d "$POLICY_STATES_DIR" ] && [ "$(ls -A $POLICY_STATES_DIR 2>/dev/null)" ]; then
    # 使用 :? 确保变量非空
    rm -rf "${POLICY_STATES_DIR:?}/"*
fi
```

这样可以：
- 避免误删除根目录
- 检查目录是否存在
- 检查目录是否非空
- 防止变量未设置时执行危险操作

## 目录结构

```
data/tasks/harvest_1_log/
├── expert_demos/          # 专家演示（永远保留，非常珍贵！）
├── baseline_model/        # BC基线模型（永远保留）
├── policy_states/         # 策略状态（--clean-restart 会删除）
├── expert_labels/         # 专家标注（--clean-restart 会删除）
├── dagger/                # 聚合数据（--clean-restart 会删除）
└── dagger_model/          # DAgger模型（--clean-restart 会删除）
```

## 使用场景

### 场景1: 继续迭代训练（正常流程）

**问题**: 继续迭代时，收集错误状态用的是哪个模型？

**答案**: 使用**最新的DAgger模型**（如果存在），否则使用BC基线。

```bash
# 假设当前已有 dagger_iter_1.zip 和 dagger_iter_2.zip
# 继续迭代时，会使用 dagger_iter_2.zip 来收集失败状态
bash scripts/run_dagger_iteration.sh \
  --task harvest_1_log \
  --iterations 1

# 这会：
# 1. 使用 dagger_iter_2.zip 运行并收集失败状态
# 2. 标注这些失败状态
# 3. 聚合所有历史数据（BC基线 + iter1 + iter2 + 新标注）
# 4. 训练新模型 dagger_iter_3.zip
```

### 场景2: BC基线效果好，但DAgger迭代走偏了

```bash
# 方式1：使用workflow脚本
bash scripts/run_dagger_workflow.sh \
  --task harvest_1_log \
  --clean-restart \
  --iterations 3

# 方式2：使用iteration脚本
bash scripts/run_dagger_iteration.sh \
  --task harvest_1_log \
  --clean-restart \
  --iterations 1
```

### 场景3: 通过Web UI操作

1. 打开 `http://localhost:5000`
2. 进入任务详情页
3. 点击"开始 DAgger 迭代"
4. 选择"🗑️ 清理重启"选项
5. 确认开始

## 测试建议

1. **测试清理功能**:
   ```bash
   # 先运行一轮训练
   bash scripts/run_dagger_iteration.sh --task test_task --iterations 1
   
   # 检查生成的文件
   ls data/tasks/test_task/policy_states/
   ls data/tasks/test_task/expert_labels/
   ls data/tasks/test_task/dagger_model/
   
   # 清理重启
   bash scripts/run_dagger_iteration.sh --task test_task --clean-restart
   
   # 验证文件已删除
   ls data/tasks/test_task/policy_states/  # 应该为空
   ```

2. **测试Web UI**:
   - 创建测试任务
   - 运行几轮迭代
   - 测试"清理重启"选项
   - 验证数据已清理

## 重要说明

### 继续迭代的工作流程

当选择"继续迭代"时：
1. 系统自动查找最新的DAgger模型（如 `dagger_iter_2.zip`）
2. 使用这个模型运行游戏，收集失败状态
3. 对失败状态进行标注
4. 基于所有历史数据训练新模型（如 `dagger_iter_3.zip`）
5. 重复这个过程

**关键点**: 每次迭代都在改进**最新的模型**，而不是从BC基线重新开始。

### 清理重启 vs 继续迭代

| 操作 | 使用模型 | 保留数据 | 适用场景 |
|-----|---------|---------|---------|
| 继续迭代 | 最新DAgger模型 | 保留所有数据 | 正常迭代优化 |
| 清理重启 | BC基线 | 删除DAgger数据 | 迭代走偏，重新开始 |

## 注意事项

1. **数据不可恢复**: 清理操作会永久删除数据，使用前请确认
2. ✅ **专家演示永远保留**: `--clean-restart` 不会删除专家演示（非常珍贵）
3. ✅ **BC基线永远保留**: `--clean-restart` 不会删除BC基线
4. **安全检查**: 所有脚本都有环境检查，确保在正确的conda环境中运行

## 相关文件

- `scripts/run_dagger_iteration.sh`
- `scripts/run_dagger_workflow.sh`
- `src/web/app.py`
- `src/web/templates/training.html`

## 后续改进

可能的改进方向：

1. **备份功能**: 清理前自动备份数据到 `.backup/` 目录
2. **选择性清理**: 允许用户选择具体要清理的迭代轮次
3. **磁盘空间提示**: 清理前显示将释放的磁盘空间
4. **确认对话框**: 添加二次确认，防止误操作
5. **清理日志**: 记录清理操作的详细日志

## 总结

通过本次改进，DAgger 训练流程现在支持灵活的数据清理功能，用户可以根据实际情况选择：
- 完全重新开始（包括录制）
- 仅重新开始 DAgger（保留BC基线）
- 继续现有训练

这大大提高了训练流程的灵活性和可维护性。

