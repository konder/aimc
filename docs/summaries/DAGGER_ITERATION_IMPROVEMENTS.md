# DAgger 迭代流程改进总结

> **更新时间**: 2025-10-25  
> **改进内容**: 跳过收集、重新标注、视角复位功能

---

## 📋 概述

本次改进为 DAgger 迭代训练流程增加了三项关键功能，提升了工作流的灵活性和用户体验：

1. **跳过收集状态** - 支持在已有收集数据时直接进入标注流程
2. **重新标注迭代** - 在 Web UI 中清除指定迭代的标注数据，方便重新标注
3. **视角复位快捷键** - 在标注工具中增加 `P` 键快速复位视角

---

## 🎯 功能1: 跳过收集状态

### 问题背景

在 DAgger 迭代过程中，如果已经收集了策略状态数据，但标注过程中途退出或标注结果不满意，需要重新标注时，必须重新运行收集步骤，浪费时间。

### 解决方案

在 `run_dagger_iteration.sh` 中增加 `--skip-collect` 参数，允许跳过收集状态步骤，直接使用已有的状态数据进入标注流程。

### 使用方法

```bash
# 跳过收集，直接使用已有状态数据进行标注
bash scripts/run_dagger_iteration.sh \
    --task harvest_1_log \
    --skip-collect \
    --start-iteration 3
```

### 实现细节

```bash
if [ "$SKIP_COLLECT" = true ]; then
    print_warning "[$iter] 步骤 1/4: 跳过收集状态 (--skip-collect)"
    
    # 检查状态目录是否存在
    if [ ! -d "$STATES_DIR" ] || [ -z "$(ls -A $STATES_DIR 2>/dev/null)" ]; then
        print_error "状态目录不存在或为空: $STATES_DIR"
        echo "提示: 使用 --skip-collect 时，必须已有收集好的状态数据"
        exit 1
    fi
    
    print_info "  使用已有状态: $STATES_DIR"
    print_success "[$iter] 跳过收集，使用现有状态"
else
    # 正常收集流程
    ...
fi
```

### 使用场景

- 标注过程中误操作需要重新标注
- 想要调整智能采样参数重新标注
- 标注质量不满意需要重新标注
- 调试标注工具时多次测试

---

## 🔄 功能2: Web UI 重新标注

### 问题背景

在 Web UI 中没有便捷的方式清除已有的标注数据，用户需要手动删除文件或使用命令行操作，不够友好。

### 解决方案

在 Web UI 的"专家标注"区域为每个迭代的标注文件增加"🔄 重新标注"按钮，点击后清除该迭代的标注数据。

### UI 展示

在任务数据面板中，专家标注文件列表：

```
🏷️ 专家标注
data/tasks/harvest_1_log/expert_labels

iter_1.pkl          5.2 KB       [🔄 重新标注]
iter_2.pkl          6.8 KB       [🔄 重新标注]
iter_3.pkl          7.1 KB       [🔄 重新标注]
```

### API 实现

**后端 API** (`src/web/app.py`):

```python
@app.route('/api/tasks/<task_id>/relabel_iteration', methods=['POST'])
def relabel_iteration(task_id):
    """清除指定迭代的专家标注数据，以便重新标注"""
    data = request.json
    iteration = data.get('iteration')
    
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
```

**前端实现** (`src/web/templates/training.html`):

```javascript
async function relabelIteration(iterNum) {
    const confirmed = await showConfirm(
        '🔄 重新标注迭代',
        `确定要删除迭代 ${iterNum} 的专家标注数据吗？\n\n删除后可以使用 label_states.py 重新标注该迭代的状态。`
    );
    
    if (!confirmed) return;
    
    const response = await fetch(`/api/tasks/${TASK_ID}/relabel_iteration`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ iteration: iterNum })
    });
    
    const data = await response.json();
    if (data.success) {
        await showAlert('✅ 标注已清除', data.message);
        await loadDirectoryInfo();  // 刷新目录信息
    }
}
```

### 操作流程

1. 打开任务训练页面
2. 在"任务数据"区域找到"🏷️ 专家标注"
3. 点击目标迭代的"🔄 重新标注"按钮
4. 确认删除
5. 使用命令行工具重新标注：
   ```bash
   python src/training/dagger/label_states.py \
       --states data/tasks/harvest_1_log/policy_states/iter_3 \
       --output data/tasks/harvest_1_log/expert_labels/iter_3.pkl \
       --smart-sampling
   ```

### 结合使用

**完整的重新标注工作流**：

```bash
# 1. 在 Web UI 中点击"🔄 重新标注"按钮清除标注数据

# 2. 使用跳过收集参数重新运行迭代（直接进入标注）
bash scripts/run_dagger_iteration.sh \
    --task harvest_1_log \
    --skip-collect \
    --start-iteration 3
```

---

## 🎮 功能3: 视角复位快捷键

### 问题背景

在标注过程中，经常需要调整视角（pitch 和 yaw）。当视角被调整后，想要快速恢复到中心位置（12, 12）时，需要连续按多次方向键，不够高效。

### 解决方案

增加快捷键 `P`，一键将视角置为中心位置 (pitch=12, yaw=12)。

### 动作映射

在 `src/training/dagger/label_states.py` 中增加：

```python
self.action_mapping = {
    # ... 其他映射 ...
    
    # 视角复位
    'p': [0, 0, 0, 12, 12, 0, 0, 0],  # 视角置为中心 (12, 12) ⭐
    
    # ... 特殊操作 ...
}
```

### MineDojo 动作编码

MineDojo 使用 8 维 MultiDiscrete 动作空间：
```
[forward/back, left/right, jump, pitch, yaw, functional, sprint, sneak]
                                   ↑      ↑
                                   12     12  ← 中心位置
```

- **pitch** (维度3): 0-24，中心为 12（水平）
  - < 12: 向上看
  - > 12: 向下看
  - = 12: 水平
  
- **yaw** (维度4): 0-24，中心为 12（正前方）
  - < 12: 向左看
  - > 12: 向右看
  - = 12: 正前方

### 使用场景

- 视角被方向键调整偏了，快速复位
- 在标注新状态前，统一视角为中心
- 对比不同状态时，保持一致的视角

### 界面更新

标注工具窗口底部显示：

```
WASD: Move  |  Arrows: Look  |  F: Attack  |  Space: Jump
Q: Fwd+Jump  |  E: Fwd+Attack  |  R: Fwd+Jump+Attack  |  P: Reset View
X: Keep Policy  |  C: Skip  |  Z: Undo  |  ESC: Finish
```

命令行提示：

```
视角控制:
  P            - 视角置为中心 (12, 12) ⭐
```

---

## 📖 理论解析：跳过 vs 保持策略

这三个功能的实现过程中，用户提出了一个重要的理论问题：

### 专家标注中"跳过"和"保持策略"的区别

| 操作 | 按键 | 训练数据 | 模型影响 | 适用场景 |
|------|------|---------|---------|----------|
| **跳过 (Skip)** | `C` | ❌ 不加入 | 该状态不参与学习 | 不确定、无意义的状态 |
| **保持策略 (Keep)** | `X` | ✅ 加入 `(obs, policy_action)` | 强化策略当前行为 | 策略动作已正确 |
| **修正动作** | 其他键 | ✅ 加入 `(obs, expert_action)` | 纠正策略错误 | 策略动作错误需纠正 |

### 代码实现对比

**跳过 (Skip)**:
```python
elif action is None:
    # 跳过此状态 - 不保存任何数据
    print(f"  ⊘ [{current_idx+1}/{len(states_to_label)}] SKIP (跳过)")
    current_idx += 1
```

**保持策略 (Keep)**:
```python
if action == 'pass':
    labeled_item = {
        'observation': state_info['state']['observation'],
        'expert_action': state_info['policy_action'],  # 使用策略的原动作
        'policy_action': state_info['policy_action'],
    }
    self.labeled_data.append(labeled_item)  # 保存到训练数据
```

### 实际使用示例

假设在失败前的某一步，策略选择了"前进"：

- **策略是对的** → 按 **X** (保持策略)，快速标注，强化正确行为
- **应该"前进+攻击"** → 按 **E**，纠正动作
- **这一帧完全没价值** → 按 **C** (跳过)，不浪费时间

---

## 🚀 完整工作流示例

### 场景：迭代3标注不满意，需要重新标注

```bash
# 步骤1: 在 Web UI 中点击"🔄 重新标注"清除 iter_3.pkl

# 步骤2: 使用已有状态数据直接进入标注
bash scripts/run_dagger_iteration.sh \
    --task harvest_1_log \
    --skip-collect \
    --start-iteration 3 \
    --smart-sampling \
    --failure-window 15 \
    --random-sample-rate 0.15

# 步骤3: 在标注过程中
# - 策略动作正确 → 按 X (保持策略)
# - 需要纠正动作 → 按相应动作键 (W/A/S/D/Q/E/F等)
# - 视角偏了 → 按 P (复位视角) ⭐
# - 这一帧无价值 → 按 C (跳过)
# - 标错了 → 按 Z (撤销)
# - 完成 → 按 ESC

# 步骤4: 标注完成后，继续训练
# (脚本会自动进行数据聚合和模型训练)
```

---

## 📊 改进效果

### 时间节省

| 场景 | 改进前 | 改进后 | 节省 |
|------|--------|--------|------|
| 重新标注一次 | 重新收集 (~5分钟) + 标注 | 直接标注 | ~5分钟 |
| 清除标注数据 | 命令行操作 (~30秒) | Web UI 点击 | ~25秒 |
| 调整视角 | 连按方向键多次 | 按一次 P | 明显提升 |

### 用户体验提升

- ✅ **操作更直观**: Web UI 一键清除标注数据
- ✅ **流程更灵活**: 可以只重新标注而不重新收集
- ✅ **标注更高效**: 视角复位快捷键减少重复操作
- ✅ **调试更方便**: 快速迭代优化标注策略

---

## 🔧 技术实现要点

### 1. Bash 脚本参数处理

```bash
# 新增参数
--skip-collect)
    SKIP_COLLECT=true
    shift
    ;;

# 条件执行
if [ "$SKIP_COLLECT" = true ]; then
    # 检查必要条件
    if [ ! -d "$STATES_DIR" ] || [ -z "$(ls -A $STATES_DIR 2>/dev/null)" ]; then
        print_error "状态目录不存在或为空"
        exit 1
    fi
else
    # 正常执行收集
fi
```

### 2. Flask API 实现

```python
@app.route('/api/tasks/<task_id>/relabel_iteration', methods=['POST'])
def relabel_iteration(task_id):
    """RESTful API - 清除标注数据"""
    # 参数验证
    # 文件操作
    # 错误处理
    # 返回结果
```

### 3. JavaScript 前端交互

```javascript
// 确认对话框
async function relabelIteration(iterNum) {
    const confirmed = await showConfirm(...);
    if (!confirmed) return;
    
    // 发送请求
    const response = await fetch(...);
    
    // 刷新UI
    await loadDirectoryInfo();
}
```

### 4. OpenCV 键盘映射

```python
self.action_mapping = {
    'p': [0, 0, 0, 12, 12, 0, 0, 0],  # MineDojo action array
}
```

---

## 📚 相关文档

- [Label States 快捷键指南](../guides/LABEL_STATES_SHORTCUTS_GUIDE.md) - 完整快捷键列表
- [DAgger 工作流跳过指南](../guides/DAGGER_WORKFLOW_SKIP_GUIDE.md) - 跳过步骤说明
- [DAgger 完整指南](../guides/DAGGER_COMPREHENSIVE_GUIDE.md) - DAgger 训练全流程
- [Web UI 使用指南](../guides/WEB_COMPREHENSIVE_GUIDE.md) - Web 控制台说明

---

## 💡 最佳实践

### 何时使用 --skip-collect

**✅ 推荐使用**:
- 标注过程中误操作需要重来
- 想要调整智能采样参数
- 标注质量不满意需要重新标注
- 调试标注工具时多次测试

**❌ 不推荐使用**:
- 模型已经更新，需要用新模型收集状态
- 状态数据不存在或不完整
- 想要增加收集的 episodes 数量

### 重新标注的最佳时机

- 发现标注质量不够好时
- 策略动作解读错误时
- 想要调整采样策略时
- 想要增加或减少标注密度时

### 视角复位的使用技巧

- 标注前先按 `P` 统一视角
- 视角被方向键调整偏后立即按 `P`
- 对比不同状态时保持视角一致

---

## 🎓 总结

本次改进通过三个小功能的增加，显著提升了 DAgger 迭代训练的灵活性和效率：

1. **跳过收集** - 允许在不重新收集的情况下重新标注
2. **Web UI 重新标注** - 提供友好的界面清除标注数据
3. **视角复位** - 提高标注过程中的操作效率

这些改进使得 DAgger 训练流程更加**灵活、高效、易用**，特别适合在标注质量优化和模型调试阶段使用。

---

**最后更新**: 2025-10-25  
**作者**: AIMC Team  
**版本**: v1.0

