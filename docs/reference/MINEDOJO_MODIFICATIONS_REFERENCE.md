# MineDojo 源代码修改清单

**生成时间**: 2025-11-20  
**修改时间范围**: 最近10天  
**目标**: 支持 MineRL 动作空间（inventory, pickItem, swapHands, GUI 鼠标控制）

---

## 📋 修改文件清单

### 1. Python 文件修改 (4个)

| 文件 | 修改时间 | 主要修改内容 |
|------|---------|-------------|
| `sim/sim.py` | Nov 19 18:45 | 添加 `inventory`, `swapHands`, `pickItem` 到 `common_actions` |
| `sim/handlers/agent/action.py` | Nov 16 01:36 | 添加 `inventory`, `swapHands`, `pickItem` 动作定义 |
| `sim/mc_meta/mc.py` | Nov 18 11:00 | 添加 `inventory`, `swapHands`, `pickItem` 键位映射 |
| `sim/wrappers/ar_nn/nn_action_space_wrapper.py` | Nov 18 11:19 | 扩展 action[5] 支持 inventory(8), swapHands(9), pickItem(10) |

### 2. Java 文件修改 (5个)

| 文件 | 修改时间 | 主要修改内容 |
|------|---------|-------------|
| `CommandForKey.java` | Nov 19 20:55 | 实现 inventory toggle 逻辑，支持 swapHands, pickItem |
| `CameraCommandsImplementation.java` | Nov 19 17:37 | 启用 `FakeMouse.addMovement` 用于 GUI 鼠标控制 |
| `FakeMouse.java` | Nov 19 17:37 | 启用虚拟鼠标光标渲染 |
| `FakeKeyboard.java` | Nov 17 14:09 | 添加调试日志 |
| `ClientStateMachine.java` | Nov 20 10:19 | 设置 `guiScale=1` 匹配 MineRL |

### 3. 新增 Java 文件 (2个)

| 文件 | 创建时间 | 功能 |
|------|---------|------|
| `Mixins/MixinMinecraftGuiIdempotent.java` | Nov 19 20:55 | 防止 GUI 重复打开/关闭，解决鼠标重置问题 |
| `Mixins/MixinGuiAchievementDisable.java` | Nov 20 10:41 | 禁用成就通知 |

### 4. 配置文件修改 (1个)

| 文件 | 修改时间 | 主要修改内容 |
|------|---------|-------------|
| `mixins.overclocking.malmomod.json` | Nov 19 20:55 | 注册新增的两个 Mixin |

---

## 📝 详细修改内容

### 1.1 sim/sim.py

**修改位置**: 第 48 行附近

```python
# 原始代码
common_actions = [
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
]

# 修改后
common_actions = [
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
    "inventory",    # 新增：打开/关闭物品栏 (E键)
    "swapHands",    # 新增：交换主手和副手物品 (F键)
    "pickItem",     # 新增：中键点击复制方块 (鼠标中键)
]
```

**影响**: 这些动作会自动在 `_action_obj_to_xml` 中处理

---

### 1.2 sim/handlers/agent/action.py

**修改位置**: 添加新的动作定义

```python
# 在 KEYMAP_KEYBOARD_MOUSE 中添加
KEYMAP_KEYBOARD_MOUSE = {
    # ... 原有动作 ...
    "inventory": "key.keyboard.e",      # 新增
    "swapHands": "key.keyboard.f",      # 新增
    "pickItem": "key.mouse.middle",     # 新增
}
```

**影响**: 定义了动作名称到 Minecraft 键位的映射

---

### 1.3 sim/mc_meta/mc.py

**修改位置**: 添加键位码映射

```python
# 在 ALL_KEYS 中添加
ALL_KEYS = {
    # ... 原有键位 ...
    "key.keyboard.e": 18,        # 新增：E键（物品栏）
    "key.keyboard.f": 33,        # 新增：F键（交换手持物品）
    "key.mouse.middle": -98,     # 新增：鼠标中键（复制方块）
}
```

**影响**: 定义了键位名称到键位码的映射

---

### 1.4 sim/wrappers/ar_nn/nn_action_space_wrapper.py

**修改位置**: 第 277 行附近，扩展 `action[5]` 的处理

```python
# 原始代码
# action[5]: Functional
# 0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy

# 修改后
# action[5]: Functional
# 0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy,
# 8: inventory, 9: swapHands, 10: pickItem

# 在 noop 字典中添加
if action[5] == 8:
    noop["inventory"] = 1
elif action[5] == 9:
    noop["swapHands"] = 1
elif action[5] == 10:
    noop["pickItem"] = 1
```

**影响**: 将离散动作值映射到具体的动作名称

---

### 2.1 CommandForKey.java

**主要修改**:

1. **添加状态变量**:
```java
private static boolean inventoryInKeysList = false;
```

2. **实现 inventory toggle 逻辑**:
```java
private static void toggleInventoryKey(KeyBinding keyBinding, boolean pressed){
    int keyCode = keyBinding.getKeyCode();
    
    if (pressed){
        KeyBinding.setKeyBindState(keyCode, pressed);
        if (!inventoryInKeysList){
            // 第一次 inventory=1: 打开 GUI
            KeyBinding.onTick(keyCode);
            inventoryInKeysList = true;
        }else{
            // 第二次 inventory=1: 关闭 GUI
            Minecraft mc = Minecraft.getMinecraft();
            if (mc.currentScreen != null) {
                mc.currentScreen.onGuiClosed();
                mc.currentScreen = null;
            }
            KeyBinding.setKeyBindState(keyCode, false);
            inventoryInKeysList = false;
        }
    }else{
        // inventory=0: 保持当前状态
        if (inventoryInKeysList){
            KeyBinding.onTick(keyCode);
        }
    }
}
```

3. **在 `onExecute` 中处理 inventory**:
```java
if (verb.equalsIgnoreCase("inventory")) {
    KeyBinding keyBinding = Minecraft.getMinecraft().gameSettings.keyBindInventory;
    toggleInventoryKey(keyBinding, pressed);
    return;
}
```

4. **添加 swapHands 和 pickItem 支持**:
```java
if (verb.equalsIgnoreCase("swapHands")) {
    setKeyBindingStateDirect(Minecraft.getMinecraft().gameSettings.keyBindSwapHands, pressed);
}
if (verb.equalsIgnoreCase("pickItem")) {
    setKeyBindingStateDirect(Minecraft.getMinecraft().gameSettings.keyBindPickBlock, pressed);
}
```

**影响**: 核心逻辑，实现了 inventory 的 toggle 机制和新动作支持

---

### 2.2 CameraCommandsImplementation.java

**主要修改**: 启用 `FakeMouse` 用于 GUI 鼠标控制

```java
// 在 onExecute 方法中
if (MalmoMod.isLowLevelInput()) {
    // 使用 FakeMouse 进行鼠标移动（支持 GUI 内鼠标控制）
    FakeMouse.addMovement(dx, dy);
}
```

**影响**: 允许在 GUI 打开时通过 camera 命令控制鼠标移动

---

### 2.3 FakeMouse.java

**主要修改**: 启用虚拟鼠标光标

```java
// 取消注释
private static FakeMouseCursor cursor = new FakeMouseCursor();
```

**影响**: 在 GUI 中显示虚拟鼠标光标

---

### 2.4 FakeKeyboard.java

**主要修改**: 添加调试日志

```java
System.out.println("[FakeKeyboard.press] key=" + key + ", keysDown.contains=" + keysDown.contains(key) + ", keysDown.size=" + keysDown.size());
System.out.println("[FakeKeyboard.release] key=" + key + ", keysDown.contains=" + keysDown.contains(key) + ", keysDown.size=" + keysDown.size());
```

**影响**: 方便调试键盘事件

---

### 2.5 ClientStateMachine.java

**主要修改**: 设置 GUI 缩放比例

```java
// 原始代码
Minecraft.getMinecraft().gameSettings.guiScale = 2;

// 修改后
Minecraft.getMinecraft().gameSettings.guiScale = 1;  // 匹配 MineRL
```

**影响**: 使 MineDojo 的 GUI 大小与 MineRL 一致

---

### 3.1 MixinMinecraftGuiIdempotent.java

**功能**: 防止 GUI 重复打开和鼠标重置

**核心逻辑**:

```java
@Overwrite
public void displayGuiScreen(GuiScreen guiScreenIn) {
    // 幂等性检查 1: 阻止在 GuiInventory 打开时关闭它
    if (guiScreenIn == null && this.currentScreen instanceof GuiInventory) {
        return;  // 阻止循环中的关闭
    }
    
    // 幂等性检查 2: 阻止重复打开相同的 GuiInventory
    if (guiScreenIn != null && this.currentScreen != null &&
        guiScreenIn.getClass() == this.currentScreen.getClass()) {
        return;  // 直接返回
    }
    
    // 原始的 displayGuiScreen 逻辑
    // ...
}
```

**影响**: 解决了 GUI 闪烁和鼠标位置重置的问题

---

### 3.2 MixinGuiAchievementDisable.java

**功能**: 禁用成就通知

```java
@Mixin(GuiAchievement.class)
public class MixinGuiAchievementDisable {
    @Overwrite
    public void updateAchievementWindow() {
        // 不做任何事，阻止成就通知显示
    }
}
```

**影响**: 移除了屏幕上的成就通知弹窗

---

### 4.1 mixins.overclocking.malmomod.json

**修改**: 注册新增的 Mixin

```json
{
  "mixins": [
    // ... 原有 Mixins ...
    "MixinMinecraftGuiIdempotent",    // 新增
    "MixinGuiAchievementDisable"      // 新增
  ]
}
```

**影响**: 使新增的 Mixin 生效

---

## 🎯 修改目标和效果

### 目标 1: 支持 inventory 动作 ✅

- **Python 层**: 添加 `inventory` 到动作空间
- **Java 层**: 实现 toggle 逻辑
- **Mixin 层**: 防止 GUI 重复打开/关闭
- **效果**: inventory 可以正常打开和关闭，GUI 保持稳定

### 目标 2: 支持 GUI 鼠标控制 ✅

- **Java 层**: 启用 `FakeMouse` 和虚拟光标
- **Mixin 层**: 防止鼠标位置重置
- **效果**: 可以在 GUI 中移动鼠标和点击物品

### 目标 3: 支持 swapHands 和 pickItem ✅

- **Python 层**: 添加动作定义和键位映射
- **Java 层**: 添加命令处理
- **效果**: 可以交换手持物品和复制方块

### 目标 4: 匹配 MineRL 环境 ✅

- **GUI 缩放**: `guiScale=1`
- **屏幕分辨率**: `640x320`
- **禁用通知**: 成就通知已禁用
- **效果**: MineDojo 环境与 MineRL 行为一致

---

## 📦 生成 Patch 文件

所有修改将合并到一个 patch 文件：

```bash
docker/minedojo_action_extension.patch
```

**包含内容**:
- ✅ 所有 Python 文件修改
- ✅ 所有 Java 文件修改
- ✅ 新增的 Mixin 文件
- ✅ Mixin 配置文件修改

**应用方法**:

```bash
# 进入 MineDojo 安装目录
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo

# 应用 patch
patch -p1 < /path/to/minedojo_action_extension.patch

# 重新编译 Java 代码
cd sim/Malmo/Minecraft
./gradlew shadowJar
```

---

## ✅ 验证清单

在应用 patch 后，请验证以下功能：

- [ ] `inventory` 动作可以打开/关闭 GUI
- [ ] GUI 打开时不会闪烁或重置鼠标
- [ ] 可以在 GUI 中移动鼠标（通过 camera 命令）
- [ ] 可以在 GUI 中点击物品（通过 attack/use 命令）
- [ ] `swapHands` 可以交换主手和副手物品
- [ ] `pickItem` 可以复制方块
- [ ] GUI 大小与 MineRL 一致
- [ ] 没有成就通知弹窗
- [ ] 虚拟鼠标光标在 GUI 中可见

---

## 📚 相关文档

- **动作回放指南**: `docs/guides/STEVE1_ACTION_REPLAY_GUIDE.md`
- **GUI 鼠标控制**: `docs/technical/MINEDOJO_GUI_MOUSE_CONTROL_MISSING.md`
- **Inventory Toggle**: `docs/technical/INVENTORY_ISPRESSED_MECHANISM.md`
- **Mixin 实现**: `docs/technical/GUI_ALREADY_OPEN_CHECK_SOLUTION.md`

---

**生成完成！请确认以上修改清单，然后我将生成完整的 patch 文件。** ✨

