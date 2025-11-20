# MineDojo 动作空间扩展 Patch 总结

**版本**: v1.0  
**生成日期**: 2025-11-20  
**Patch 文件**: `docker/minedojo_action_extension.patch`  
**应用脚本**: `docker/apply_minedojo_patch.sh`

---

## 📊 修改统计

| 类别 | 数量 | 说明 |
|------|------|------|
| Python 文件修改 | 4 | 动作空间定义和映射 |
| Java 文件修改 | 4 | 核心逻辑实现 |
| Java 新增文件 | 2 | Mixin 扩展 |
| 配置文件修改 | 2 | Mixin 注册和 GUI 设置 |
| **总计** | **12** | **所有修改文件** |

**Patch 文件大小**: 587 行

---

## 📝 完整文件清单

### 1. Python 文件 (4个)

#### 1.1 sim/sim.py
- **修改内容**: 添加 `inventory`, `swapHands`, `pickItem` 到 `common_actions` 列表
- **行数变化**: +3 行
- **关键代码**:
  ```python
  common_actions = [
      # ... 原有动作 ...
      "inventory",    # 新增
      "swapHands",    # 新增
      "pickItem",     # 新增
  ]
  ```

#### 1.2 sim/handlers/agent/action.py
- **修改内容**: 添加键位映射定义
- **行数变化**: +3 行
- **关键代码**:
  ```python
  KEYMAP_KEYBOARD_MOUSE = {
      # ... 原有映射 ...
      "inventory": "key.keyboard.e",
      "swapHands": "key.keyboard.f",
      "pickItem": "key.mouse.middle",
  }
  ```

#### 1.3 sim/mc_meta/mc.py
- **修改内容**: 添加键位码映射
- **行数变化**: +3 行
- **关键代码**:
  ```python
  ALL_KEYS = {
      # ... 原有键位 ...
      "key.keyboard.e": 18,
      "key.keyboard.f": 33,
      "key.mouse.middle": -98,
  }
  ```

#### 1.4 sim/wrappers/ar_nn/nn_action_space_wrapper.py
- **修改内容**: 扩展 action[5] 支持新动作
- **行数变化**: +9 行
- **关键代码**:
  ```python
  # action[5]: 8=inventory, 9=swapHands, 10=pickItem
  if action[5] == 8:
      noop["inventory"] = 1
  elif action[5] == 9:
      noop["swapHands"] = 1
  elif action[5] == 10:
      noop["pickItem"] = 1
  ```

---

### 2. Java 文件 (4个)

#### 2.1 CommandForKey.java
- **修改内容**: 
  - 实现 inventory toggle 逻辑
  - 添加 swapHands 和 pickItem 支持
- **行数变化**: +40 行
- **关键功能**:
  - `toggleInventoryKey()` 方法
  - 状态变量 `inventoryInKeysList`
  - GUI 打开/关闭控制

#### 2.2 CameraCommandsImplementation.java
- **修改内容**: 启用 FakeMouse 用于 GUI 鼠标控制
- **行数变化**: +5 行
- **关键代码**:
  ```java
  if (MalmoMod.isLowLevelInput()) {
      FakeMouse.addMovement(dx, dy);
  }
  ```

#### 2.3 FakeMouse.java
- **修改内容**: 启用虚拟鼠标光标
- **行数变化**: +1 行
- **关键代码**:
  ```java
  private static FakeMouseCursor cursor = new FakeMouseCursor();
  ```

#### 2.4 ClientStateMachine.java
- **修改内容**: 设置 GUI 缩放比例
- **行数变化**: +1 行
- **关键代码**:
  ```java
  Minecraft.getMinecraft().gameSettings.guiScale = 1;  // 从 2 改为 1
  ```

---

### 3. 新增文件 (2个)

#### 3.1 MixinMinecraftGuiIdempotent.java
- **文件大小**: ~150 行
- **功能**: 防止 GUI 重复打开/关闭，解决鼠标重置问题
- **核心逻辑**:
  - 幂等性检查 1: 阻止在 GuiInventory 打开时关闭它
  - 幂等性检查 2: 阻止重复打开相同的 GuiInventory
- **使用 Mixin**: `@Overwrite displayGuiScreen()`

#### 3.2 MixinGuiAchievementDisable.java
- **文件大小**: ~20 行
- **功能**: 禁用成就通知弹窗
- **核心逻辑**: `@Overwrite updateAchievementWindow()` 为空方法

---

### 4. 配置文件 (2个)

#### 4.1 mixins.overclocking.malmomod.json
- **修改内容**: 注册新增的 Mixin
- **行数变化**: +2 行
- **关键代码**:
  ```json
  {
    "mixins": [
      // ... 原有 Mixins ...
      "MixinMinecraftGuiIdempotent",
      "MixinGuiAchievementDisable"
    ]
  }
  ```

#### 4.2 sim/Malmo/Minecraft/run/options.txt
- **修改内容**: 设置 GUI 缩放比例
- **行数变化**: ±1 行
- **关键代码**:
  ```
  guiScale:0  →  guiScale:1
  ```
- **说明**: 
  - `guiScale:0` = 自动（通常是 2 或 3）
  - `guiScale:1` = 小（Small），匹配 MineRL

---

## 🎯 功能实现对照表

| 功能 | Python 层 | Java 层 | Mixin 层 | 配置文件 | 状态 |
|------|----------|---------|----------|----------|------|
| inventory 动作 | ✅ sim.py<br>✅ action.py<br>✅ mc.py<br>✅ wrapper.py | ✅ CommandForKey | ✅ GuiIdempotent | ✅ mixins.json | ✅ 完成 |
| swapHands 动作 | ✅ sim.py<br>✅ action.py<br>✅ mc.py<br>✅ wrapper.py | ✅ CommandForKey | - | - | ✅ 完成 |
| pickItem 动作 | ✅ sim.py<br>✅ action.py<br>✅ mc.py<br>✅ wrapper.py | ✅ CommandForKey | - | - | ✅ 完成 |
| GUI 鼠标控制 | - | ✅ CameraCommands<br>✅ FakeMouse | ✅ GuiIdempotent | - | ✅ 完成 |
| 虚拟鼠标光标 | - | ✅ FakeMouse | - | - | ✅ 完成 |
| 禁用成就通知 | - | - | ✅ AchievementDisable | ✅ mixins.json | ✅ 完成 |
| GUI 缩放匹配 | - | ✅ ClientStateMachine | - | ✅ options.txt | ✅ 完成 |

---

## 🔧 技术细节

### inventory Toggle 机制

```
MineRL 输入 → MineDojo Wrapper → Java CommandForKey → Minecraft
   ↓                ↓                    ↓                ↓
inventory=1    保持状态判断        toggleInventoryKey()    GUI 打开/关闭
inventory=0    不改变状态          保持当前状态            GUI 保持
```

**状态管理**:
- `inventoryInKeysList = false` → inventory=1 → 打开 GUI → `inventoryInKeysList = true`
- `inventoryInKeysList = true` → inventory=1 → 关闭 GUI → `inventoryInKeysList = false`
- `inventoryInKeysList = true` → inventory=0 → 保持打开 → `inventoryInKeysList = true`

### GUI 鼠标控制流程

```
Python: action['camera'] = [dx, dy]
   ↓
MineDojo Wrapper: 转换为 XML 命令
   ↓
Java: CameraCommandsImplementation.onExecute()
   ↓
if (MalmoMod.isLowLevelInput()) {
    FakeMouse.addMovement(dx, dy);  ← 关键！
}
   ↓
FakeMouse: 更新内部坐标，添加事件到队列
   ↓
MixinMouse: 拦截 LWJGL Mouse 调用，返回 FakeMouse 坐标
   ↓
Minecraft: 读取鼠标坐标，更新 GUI 光标位置
```

### Mixin 幂等性检查

```java
@Overwrite
public void displayGuiScreen(GuiScreen guiScreenIn) {
    // 检查 1: 阻止在 GUI 打开时关闭
    if (guiScreenIn == null && this.currentScreen instanceof GuiInventory) {
        return;  // 阻止 while(isPressed()) 循环中的关闭
    }
    
    // 检查 2: 阻止重复打开相同 GUI
    if (guiScreenIn != null && this.currentScreen != null &&
        guiScreenIn.getClass() == this.currentScreen.getClass()) {
        return;  // 阻止重复打开
    }
    
    // 原始逻辑...
}
```

---

## 📦 Patch 应用指南

### 快速应用

```bash
# 1. 应用 patch
bash docker/apply_minedojo_patch.sh install

# 2. 验证
bash docker/apply_minedojo_patch.sh verify
```

### 手动应用

```bash
# 1. 进入 MineDojo 目录
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo

# 2. 应用 patch
patch -p1 < /path/to/minedojo_action_extension.patch

# 3. 编译 Java
cd sim/Malmo/Minecraft
./gradlew shadowJar

# 4. 验证
grep -q "inventory" ../../../sim.py && echo "✓ Python 修改成功"
grep -q "toggleInventoryKey" src/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java && echo "✓ Java 修改成功"
```

---

## ✅ 验证清单

应用 patch 后，请验证以下内容：

- [ ] **Python 文件**
  - [ ] `sim.py` 包含 `inventory`, `swapHands`, `pickItem`
  - [ ] `action.py` 包含键位映射
  - [ ] `mc.py` 包含键位码
  - [ ] `wrapper.py` 支持 action[5]=8/9/10

- [ ] **Java 文件**
  - [ ] `CommandForKey.java` 包含 `toggleInventoryKey()`
  - [ ] `CameraCommandsImplementation.java` 包含 `FakeMouse.addMovement`
  - [ ] `FakeMouse.java` 启用了 `FakeMouseCursor`
  - [ ] `ClientStateMachine.java` 设置 `guiScale=1`

- [ ] **新增文件**
  - [ ] `MixinMinecraftGuiIdempotent.java` 存在
  - [ ] `MixinGuiAchievementDisable.java` 存在

- [ ] **配置文件**
  - [ ] `mixins.json` 注册了新 Mixin
  - [ ] `options.txt` 设置 `guiScale:1`

- [ ] **编译**
  - [ ] `./gradlew shadowJar` 成功
  - [ ] `MalmoMod-0.37.0-fat.jar` 已更新

---

## 🧪 功能测试

### 测试 1: inventory 动作

```python
env = gym.make('MineDojoHarvestEnv-v0')
obs = env.reset()

# 打开 GUI
action = {'inventory': 1, ...}
obs, _, _, _ = env.step(action)
# 预期: GUI 打开

# 保持 GUI 打开
action = {'inventory': 0, ...}
for _ in range(10):
    obs, _, _, _ = env.step(action)
# 预期: GUI 保持打开

# 关闭 GUI
action = {'inventory': 1, ...}
obs, _, _, _ = env.step(action)
# 预期: GUI 关闭
```

### 测试 2: GUI 鼠标控制

```python
# 打开 GUI
action = {'inventory': 1, 'camera': [0, 0], ...}
obs, _, _, _ = env.step(action)

# 移动鼠标
action = {'inventory': 0, 'camera': [10, 10], ...}
obs, _, _, _ = env.step(action)
# 预期: 鼠标在 GUI 中移动

# 点击
action = {'inventory': 0, 'attack': 1, ...}
obs, _, _, _ = env.step(action)
# 预期: 点击 GUI 中的物品
```

---

## 📚 相关文档

- **应用脚本**: `docker/apply_minedojo_patch.sh`
- **使用指南**: `docker/README_MINEDOJO_PATCH.md`
- **修改清单**: `docs/reference/MINEDOJO_MODIFICATIONS_CHECKLIST.md`
- **技术文档**: 
  - `docs/technical/MINEDOJO_GUI_MOUSE_CONTROL_MISSING.md`
  - `docs/technical/INVENTORY_ISPRESSED_MECHANISM.md`
  - `docs/technical/GUI_ALREADY_OPEN_CHECK_SOLUTION.md`

---

## 🎉 总结

**Patch 生成完成！**

- ✅ **12 个文件修改**（4 Python + 4 Java + 2 新增 + 2 配置）
- ✅ **587 行 patch 代码**
- ✅ **完整的应用和验证脚本**
- ✅ **详细的文档和测试指南**

**下一步**:

1. 应用 patch: `bash docker/apply_minedojo_patch.sh install`
2. 验证安装: `bash docker/apply_minedojo_patch.sh verify`
3. 运行测试: 使用 STEVE-1 评估器测试新动作

**Happy Patching!** 🚀
