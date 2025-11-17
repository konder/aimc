# MineDojo Inventory GUI 保持打开的最终解决方案

## 问题分析

### 根本原因

MineDojo (MC 1.11.2) 和 MineRL (MC 1.16.5) 对 inventory 键的处理方式不同：

1. **MineRL (MC 1.16.5 / MCP-Reborn)**
   - 使用**状态驱动**架构
   - 直接修改 Minecraft 的 `KeyboardListener` 和 `KeyBinding` 类
   - `inventory=1` → 设置 `keysPressed.add("key.keyboard.e")`
   - `inventory=0` → 不影响 `keysPressed`（状态保持）
   - **结果**：inventory GUI 保持打开 ✅

2. **MineDojo (MC 1.11.2 / Malmo)**
   - 使用**事件驱动**架构
   - 通过 `FakeKeyboard.press()` / `FakeKeyboard.release()` 模拟按键事件
   - `inventory=1` → `FakeKeyboard.press(18)`  → `keysDown.add(18)` → GUI 打开
   - `inventory=0` → `FakeKeyboard.release(18)` → `keysDown.remove(18)` → **GUI 关闭 ❌**

### 问题深入分析

当我们尝试在 `MineDojoBiomeWrapper` 中删除 `inventory` 键时，发现：

1. **Wrapper 层删除成功** (`minedojo_harvest.py`)
   ```python
   if 'inventory' in minerl_action:
       del minerl_action['inventory']  # ✅ 键被删除
   ```

2. **Action 转换仍然生成 `inventory=0`** (`nn_action_space_wrapper.py`)
   ```python
   noop = self.env.action_space.no_op()  # ← 包含 inventory=0
   # ... 根据 MultiDiscrete array 更新 noop ...
   # 如果 action[5] == 0，noop["inventory"] 保持为 0
   ```

3. **MineDojo 将所有 action 转换为 XML** (`sim.py`)
   ```python
   def _action_obj_to_xml(self, action):
       parsed_action.extend(
           [h.to_hero(action[h.to_string()]) for h in self._sim_spec.actionables]
       )
       # ← 会为所有 actionables 生成命令，包括 "inventory 0"
   ```

4. **Malmo 执行 `inventory 0`** (`CommandForKey.java`)
   ```java
   if (parameter.equalsIgnoreCase(UP_COMMAND_STRING)) {  // "0"
       FakeKeyboard.release(keyHook.getKeyCode());  // release(18) ← GUI 关闭！
   }
   ```

## 最终解决方案

### 1. MineDojo 源代码修改

**文件**: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/sim.py`

**修改位置**: `_action_obj_to_xml` 方法（约第 659 行）

```python
def _action_obj_to_xml(self, action):
    parsed_action = [f'chat {action["chat"]}'] if "chat" in action else []
    
    # ✅ Filter out inventory=0 to prevent closing GUI (MC 1.11.2 compatibility)
    if action.get('inventory') == 0:
        action = {k: v for k, v in action.items() if k != 'inventory'}
    
    parsed_action.extend(
        [h.to_hero(action[h.to_string()]) for h in self._sim_spec.actionables]
        )
    return "\n".join(parsed_action)
```

**原理**：
- 在将 action 字典转换为 XML 之前，检查 `inventory` 是否为 0
- 如果是 0，从字典中移除 `inventory` 键
- 这样 `h.to_hero()` 就不会为 inventory 生成任何命令
- 结果：Malmo 不会收到 `"inventory 0"`，`FakeKeyboard.release(18)` 不会被调用

### 2. Wrapper 层优化

**文件**: `src/envs/minedojo_harvest.py`

**关键修改**：

1. **删除 `inventory` 键的逻辑**（第 245-260 行）
   ```python
   current_inventory = minerl_action.get('inventory', 0)
   
   if current_inventory == 1 and not self._inventory_opened:
       # 第一次打开 inventory - 保持 inventory=1
       self._inventory_opened = True
       # inventory=1 会被正常转换并发送
   else:
       # 其他情况：从 action 中移除 inventory 键
       if 'inventory' in minerl_action:
           minerl_action = minerl_action.copy()
           del minerl_action['inventory']
   ```

2. **Action 转换中的键检查**（第 407 行）
   ```python
   # ⚠️ 必须用 'in' 检查键是否存在，而不是 get() 的默认值
   if 'inventory' in minerl_action and minerl_action['inventory']:
       minedojo_action[5] = 8  # inventory
   # 如果键不存在，minedojo_action[5] 保持为 0（初始值）
   ```

### 3. 完整流程

```
Step 1: STEVE-1 输出 inventory=1
  ↓
MineDojoBiomeWrapper.step()
  → _inventory_opened = False
  → 保持 inventory=1
  ↓
_convert_action_to_minedojo()
  → minedojo_action[5] = 8
  ↓
NNActionSpaceWrapper.action()
  → noop["inventory"] = 1
  ↓
MineDojoSim.step()
  → _action_obj_to_xml() → "inventory 1"
  ↓
Malmo CommandForKey.onExecute()
  → FakeKeyboard.press(18) ✅
  → GUI 打开 ✅

Step 2+: STEVE-1 输出 inventory=0 或 inventory=1
  ↓
MineDojoBiomeWrapper.step()
  → _inventory_opened = True
  → 删除 inventory 键
  ↓
_convert_action_to_minedojo()
  → 'inventory' not in minerl_action
  → minedojo_action[5] = 0（初始值，未被修改）
  ↓
NNActionSpaceWrapper.action()
  → noop = self.env.action_space.no_op()  # 包含 inventory=0
  → fn_action = 0（no-op）
  → noop["inventory"] 保持为 0
  ↓
MineDojoSim.step()
  → _action_obj_to_xml()
    → action.get('inventory') == 0  ✅
    → 删除 inventory 键
    → **不生成 "inventory 0" 命令** ✅
  ↓
Malmo
  → 不收到 inventory 命令
  → FakeKeyboard.release(18) 不被调用 ✅
  → **GUI 保持打开** ✅✅✅
```

## 测试验证

### 预期行为

```
Step 1: inventory=1
  → Pressed 18 ✅
  → keysDown.size = 1 ✅
  → GUI 打开 ✅

Step 2-7: inventory=0
  → 无 Released 18 ✅
  → keysDown.size = 1 ✅
  → **GUI 保持打开** ✅
```

### 测试命令

```bash
cd /Users/nanzhang/aimc
bash scripts/run_minedojo_x86.sh python scripts/test_wrapper_inventory.py
```

### 关键日志

```
[14:XX:XX] [Client thread/INFO]: [STDOUT]: Pressed 18          ← Step 1
[14:XX:XX] [Client thread/INFO]: [STDOUT]: keysDown.size=1     ← Step 1 后
# Step 2-7: 无 "Released 18" 日志 ✅
```

## 文件清单

### 修改的文件

1. **MineDojo 源代码**
   - `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/sim.py`
     - 修改：`_action_obj_to_xml` 方法，过滤 `inventory=0`

2. **项目文件**
   - `src/envs/minedojo_harvest.py`
     - 修改：`MineDojoBiomeWrapper.step()` - 删除 inventory 键
     - 修改：`_convert_action_to_minedojo()` - 键存在性检查

3. **Patch 文件**
   - `docker/minedojo_inventory.patch` - 原始 patch（sim.py, nn_action_space_wrapper.py）
   - `docker/minedojo_inventory_filter.patch` - 新的 patch（包含 `_action_obj_to_xml` 修改）

### 测试文件

- `scripts/test_wrapper_inventory.py` - Wrapper inventory 状态管理测试
- `scripts/add_malmo_command_logs.sh` - Malmo Java 日志注入脚本

### 文档

- `docs/issues/INVENTORY_CODE_FILES_LIST.md` - MineDojo inventory 相关代码文件列表
- `docs/issues/MINERL_INVENTORY_CODE_FILES_LIST.md` - MineRL inventory 相关代码文件列表
- `docs/issues/MCP_REBORN_VS_MALMO_SOCKET_FLOW.md` - MCP-Reborn 与 Malmo 消息处理流程对比
- `docs/issues/MCP_REBORN_INVENTORY_IMPLEMENTATION.md` - MCP-Reborn 的 inventory 实现分析
- `docs/issues/MINEDOJO_INVENTORY_ROOT_CAUSE_CONFIRMED.md` - 根本原因确认文档
- `docs/issues/MINEDOJO_INVENTORY_SOLUTION_FINAL.md` - **本文档**

## 技术要点总结

### 关键发现

1. **MC 1.11.2 vs MC 1.16.5 的键盘处理差异**
   - MC 1.11.2 (Malmo): 事件驱动 (`press` / `release`)
   - MC 1.16.5 (MCP-Reborn): 状态驱动 (`keysPressed` set)

2. **MineDojo 的 action 处理链路**
   ```
   MultiDiscrete array
     → NNActionSpaceWrapper.action()
       → Dict action
         → MineDojoSim._action_obj_to_xml()
           → XML string
             → Malmo CommandForKey
               → FakeKeyboard
   ```

3. **`no_op()` 字典包含所有 action 的默认值**
   - 即使删除了某个 action 的键，`no_op()` 仍然包含它
   - 这导致 `inventory=0` 被隐式包含

### 解决方案核心

**在最接近 Malmo 的地方过滤 `inventory=0`**：
- ✅ `MineDojoSim._action_obj_to_xml()` - 直接在转换为 XML 之前过滤
- ❌ Wrapper 层删除键 - 不够，因为 `no_op()` 会恢复默认值
- ❌ Malmo Java 层过滤 - 过于底层，需要重新编译

## 后续工作

1. **清理调试日志**
   - 移除 `src/envs/minedojo_harvest.py` 中的 `print()` 语句
   - 移除 Malmo Java 中的调试日志

2. **更新 patch 文件**
   - 创建标准的统一 patch 文件，包含所有必要的修改
   - 更新 Dockerfile 以自动应用 patch

3. **性能测试**
   - 使用 STEVE-1 运行完整的 evaluation
   - 确认 inventory 功能在实际任务中正常工作

4. **文档完善**
   - 更新 `docs/guides/MULTILINGUAL_MINECLIP_QUICK_START.md`
   - 添加 inventory 修复说明到用户文档

---

**状态**: ✅ 解决方案已实现  
**最后更新**: 2025-11-16  
**相关 Issue**: MineDojo inventory GUI 自动关闭问题

