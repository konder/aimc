# MineDojo Inventory 功能回滚

**日期**: 2025-11-06  
**状态**: ✅ 已回滚  
**原因**: 发现意料之外的行为，决定变更其他方向

---

## 回滚原因

用户在测试 `test_inventory_keep_open.py` 时发现：

> "在第一个等待5秒的时候，物品栏在打开关闭，和代码不一致，这个时候，我们什么都没做。"

**问题**：即使代码中只是等待（不发送任何动作），inventory GUI 也在自动打开/关闭。

**决定**：不再走这条路，恢复所有修改，变更其他方向。

---

## 回滚内容

### 已恢复的文件

1. **sim/sim.py**
   - ✅ 从 `common_actions` 中移除 `"inventory"`
   - 恢复为原始的 10 个动作

2. **sim/wrappers/ar_nn/nn_action_space_wrapper.py**
   - ✅ 动作空间从 9 恢复为 8
   - ✅ 移除 `elif fn_action == 8:` 处理逻辑

3. **sim/handlers/agent/actions/inventory.py**
   - ✅ 删除自定义的 `InventoryAction` 类

### 验证结果

```
✅ sim.py: common_actions 不包含 'inventory'
✅ nn_action_space_wrapper.py: 动作空间为 8
✅ nn_action_space_wrapper.py: fn_action == 8 逻辑已移除
✅ inventory.py: 已删除
```

**MineDojo 已完全回到原始状态。**

---

## 保留的内容（作为历史记录）

### Patch 文件

- `docker/minedojo_inventory_final.patch` - 最终版本（基于 MineRL 标准方式）
- `docker/minedojo_inventory_full.patch` - 完整版本
- `docker/minedojo_inventory_complete.patch` - 早期版本

### 文档

- `docs/summaries/MINEDOJO_INVENTORY_FINAL_IMPLEMENTATION.md` - 完整实施总结
- `docs/issues/MINEDOJO_INVENTORY_GUI_TOGGLE_BEHAVIOR.md` - Toggle 行为分析
- `docs/technical/MINEDOJO_INVENTORY_ACTION_FEASIBILITY.md` - 可行性分析
- `docs/technical/MINERL_GUI_MOUSE_OPERATION_ANALYSIS.md` - MineRL GUI 操作分析
- `docs/technical/MINERL_MINEDOJO_ACTION_MAPPING_ANALYSIS.md` (已删除)

### 测试脚本

- `scripts/test_inventory_final.py`
- `scripts/test_inventory_keep_open.py`
- `scripts/test_minedojo_inventory.py`

### 工具代码

- `src/utils/action_converter.py` - MineRL ↔ MineDojo 动作转换器（仍可用）

---

## 经验教训

### 发现的问题

1. **意料之外的行为**: Inventory GUI 在没有动作时自动打开/关闭
2. **潜在原因**:
   - Malmo 的 GUI 状态管理可能与预期不符
   - MineDojo 的 step() 循环可能有额外的状态处理
   - Toggle 语义在实际环境中比预期更复杂

### 技术收获

1. ✅ **深入理解了 MineDojo/MineRL 架构**
   - `common_actions` 机制
   - `KeybasedCommandAction` 实现
   - Malmo XML 命令生成流程

2. ✅ **掌握了动作空间扩展方法**
   - MultiDiscrete 空间定义
   - Action handler 创建和注册
   - 动作到 XML 命令的转换

3. ✅ **学习了 MineRL 的标准实现**
   - 通过查看 MineRL 源码找到最优解
   - 理解了框架的设计哲学

### 未来方向

**当前策略**: 不直接修改 MineDojo 的 inventory 功能

**替代方案**:
1. 在 MineRL 环境中运行 STEVE-1/VPT（原生支持 inventory）
2. 使用 MineDojo 的 `craft` 动作作为高层抽象
3. 专注于不需要 GUI 操作的任务类别

---

## 对项目的影响

### ✅ 无影响

- 评估框架已实现（`src/evaluation/`）
- STEVE-1 集成已完成（基于 MineRL）
- 任务配置和翻译功能正常
- 文档体系完整

### 📋 后续工作

用户决定"变更其他方向"，可能的方向：
1. 专注于 MineRL 环境的 STEVE-1 评估
2. 实现不需要 GUI 的 MineDojo 任务评估
3. 探索其他模型训练方案

---

## 总结

虽然 inventory 功能的技术实现是成功的（GUI 确实打开了），但在实际使用中发现了意料之外的行为。

**用户决策明智**: 
- ✅ 避免在不稳定的功能上浪费时间
- ✅ 保留了所有技术成果（patch、文档、代码）
- ✅ 可以随时重新启用或参考

**感谢用户的精准判断和及时反馈！**

---

**文档版本**: 1.0  
**最后更新**: 2025-11-06  
**结论**: 已完全回滚，MineDojo 恢复原始状态，技术成果已保留

