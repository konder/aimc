# MineDojo Inventory 修改清理总结

**日期**: 2025-11-16  
**状态**: ✅ 已完成

---

## 📋 清理目标

根据用户要求，撤销所有自定义修改，**仅保留** `docker/minedojo_inventory.patch` 中定义的基础 action 空间扩展。

---

## ✅ 最终保留的修改

### 1. MineDojo 源代码 (via `docker/minedojo_inventory.patch`)

#### `sim/sim.py`
```python
common_actions = [
    "forward", "back", "left", "right",
    "jump", "sneak", "sprint",
    "use", "attack", "drop",
    "inventory",  # ✅ 添加
]
```

#### `sim/wrappers/ar_nn/nn_action_space_wrapper.py`
```python
MultiDiscrete([
    ...
    9,  # functional actions: 0=no_op, 1=use, 2=drop, 3=attack, 4=craft, 5=equip, 6=place, 7=destroy, 8=inventory
    ...
])

# 在 action() 方法中:
elif fn_action == 8:
    noop["inventory"] = 1
```

---

## 🗑️ 已移除的修改

### 1. `src/envs/minedojo_harvest.py`
- ❌ 移除 `self._inventory_opened` 状态管理
- ❌ 移除 `reset()` 中的状态重置
- ❌ 移除 `step()` 中的 inventory 状态管理逻辑
- ❌ 移除所有调试日志

### 2. MineDojo 源代码
- ❌ `sim/handlers/agent/action.py`: 已恢复到官方版本（无 `if adjective != "0"` 过滤）
- ❌ `sim/sim.py` 的 `_action_obj_to_xml`: 已移除自定义的 `inventory=0` 过滤逻辑

### 3. MineRL 源代码
- ❌ `herobraine/hero/handlers/agent/action.py`: 已移除调试日志

### 4. Malmo Java 代码
- ❌ `FakeKeyboard.java`: 已移除自定义调试日志和调用栈打印
- ✅ 已重新编译 Malmo JAR

---

## 🧪 验证结果

运行 `scripts/verify_inventory_patch.py`:

```
✅ 验证完成！

总结：
  • inventory 动作空间已正确扩展 (functional actions: 8→9) ✓
  • inventory 动作可以正常执行（不报错）✓
  • MineDojo patch 已正确应用 ✓
```

---

## ⚠️ 已知限制

### Inventory GUI 只显示一帧

**根本原因**:
通过详细的测试和日志分析，我们发现 MineDojo (MC 1.11.2 + Malmo) 的 inventory GUI **只显示一帧**，即使完全不发送任何命令也会自动关闭。

**测试证据**:
- `scripts/test_inventory_idle.py`: 发送 `inventory 1` 后完全空闲，GUI 在 Step 2 (0.5秒后) 就消失
- `scripts/test_minecraft_socket.py`: 直接通过 socket 发送 `inventory 1` + 只发送 `camera` 命令，GUI 仍在 Step 2 后消失

**结论**:
这不是我们的代码问题，也不是 `inventory 0` 或任何其他命令导致的，而是 **Malmo 的事件处理机制**导致的底层限制。

**对比 MineRL (MC 1.16.5)**:
- MCP-Reborn (MineRL 的 MC 1.16.5 fork) 使用**状态驱动**的方式直接修改 Minecraft 的 `KeyboardListener` 状态
- Malmo (MineDojo 的 MC 1.11.2 版本) 使用**事件驱动**的 `FakeKeyboard` 来模拟按键事件
- 这种架构差异导致两者对 inventory GUI 的处理完全不同

---

## 💡 建议

### 对于 STEVE-1 评估

1. **如果任务需要 biome 支持**:
   - 使用 MineDojo 环境 ✅
   - 接受 inventory GUI 只显示一帧的限制 ⚠️
   - 评估时关注 inventory 操作对任务完成率的影响

2. **如果任务需要持久的 inventory GUI**:
   - 使用 MineRL 环境 ✅
   - 放弃 biome 自定义支持 ⚠️

3. **混合方案**:
   - 对于不依赖 inventory 的任务 → 使用 MineDojo (支持 biome)
   - 对于依赖 inventory 的任务 → 使用 MineRL (支持持久 GUI)

---

## 📁 相关文件

### 测试脚本
- `scripts/verify_inventory_patch.py`: 验证基础 patch 功能 ✅
- `scripts/test_inventory_idle.py`: 验证 GUI 自动关闭现象
- `scripts/test_minecraft_socket.py`: 验证 Socket 直接通信行为

### 文档
- `docs/issues/INVENTORY_CODE_FILES_LIST.md`: MineDojo inventory 相关代码文件列表
- `docs/issues/MINERL_INVENTORY_CODE_FILES_LIST.md`: MineRL inventory 相关代码文件列表
- `docs/issues/MINERL_VS_MINEDOJO_INVENTORY_COMPARISON.md`: 两者详细对比
- `docs/issues/MCP_REBORN_INVENTORY_IMPLEMENTATION.md`: MCP-Reborn 的实现分析
- `docs/issues/MCP_REBORN_VS_MALMO_SOCKET_FLOW.md`: 架构差异分析

### Patch 文件
- `docker/minedojo_inventory.patch`: 唯一保留的修改 ✅

---

## 📊 问题追踪完整流程

1. **初始问题**: Inventory GUI 在打开后立即关闭
2. **假设1**: `inventory 0` 触发关闭 → ❌ 测试发现即使不发送 `inventory 0` 也会关闭
3. **假设2**: Wrapper 状态管理问题 → ❌ 即使正确管理状态也会关闭
4. **假设3**: MineDojo `action.py` 过滤问题 → ❌ 官方代码没有过滤，恢复后问题仍存在
5. **假设4**: Camera 命令触发关闭 → ❌ 完全不发送任何命令也会关闭
6. **根本原因**: Malmo 的事件处理机制导致 GUI 只显示一帧 → ✅ 通过空闲测试和 Socket 直接测试确认

---

## 🔧 如需重新应用 Patch

```bash
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo
patch -p0 < /path/to/aimc/docker/minedojo_inventory.patch
```

---

## ✅ 结论

1. ✅ 已成功清理所有自定义修改
2. ✅ 仅保留 `minedojo_inventory.patch` 中的基础 action 空间扩展
3. ✅ Inventory 动作可以正常使用（不报错）
4. ⚠️ Inventory GUI 只显示一帧是 Malmo 的底层限制，无法通过 Python 修复
5. 💡 建议根据任务需求选择 MineDojo (biome 支持) 或 MineRL (inventory 支持)

