# macOS Minecraft白屏问题分析与解决方案

## 问题描述

MineDojo启动后，Minecraft窗口显示为白屏：
- ✅ 窗口能正常打开
- ❌ 窗口内容全白，没有游戏画面
- ❌ 可能看到窗口标题但内容不渲染

**截图特征**：
```
┌─────────────────────────────┐
│     Minecraft 1.11.2        │
├─────────────────────────────┤
│                             │
│                             │
│      (白屏/灰屏)            │
│                             │
│                             │
└─────────────────────────────┘
```

## 根本原因

### 1. **headless模式冲突** ⭐⭐⭐ 最常见

如果JVM参数中包含 `-Djava.awt.headless=true`，会导致：
- ❌ AWT/Swing组件无法渲染
- ❌ OpenGL上下文无法正确绑定到窗口
- ❌ 窗口创建成功但内容为空

**问题代码**：
```bash
java -Djava.awt.headless=true ...  # ❌ 导致白屏
```

### 2. **OpenGL软件渲染未启用**

在某些macOS配置下，硬件OpenGL可能不可用，需要明确启用软件渲染：

```bash
-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true  # ✓ 必需
```

### 3. **LWJGL与macOS显示器配置冲突**

多显示器或Retina显示器可能导致渲染问题。

### 4. **窗口刷新问题**

窗口创建了但渲染循环没有启动，或者被阻塞。

## 解决方案

### 方案A：有GUI模式（推荐用于调试）

**适用场景**：
- 你想看到Minecraft窗口
- 正在调试游戏行为
- 需要人工介入操作

**修复方法**：
移除或注释掉 `headless=true` 参数，确保包含必要的OpenGL参数：

```bash
# 有GUI版本的JVM参数
JVM_ARGS="-Xmx2G -Xms512M"
JVM_ARGS="$JVM_ARGS -XX:+UseG1GC -XX:MaxGCPauseMillis=50"
JVM_ARGS="$JVM_ARGS -XX:+DisableExplicitGC -XX:+ParallelRefProcEnabled"
JVM_ARGS="$JVM_ARGS -XX:+UseBiasedLocking -XX:+UseStringDeduplication"

# OpenGL设置（不要headless）
JVM_ARGS="$JVM_ARGS -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
JVM_ARGS="$JVM_ARGS -XstartOnFirstThread"
# 注意：不要添加 -Djava.awt.headless=true

# Malmo设置
JVM_ARGS="$JVM_ARGS -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin"
JVM_ARGS="$JVM_ARGS -Dfile.encoding=UTF-8 -Duser.country=US -Duser.language=en -Duser.variant"
```

### 方案B：无GUI模式（推荐用于训练）

**适用场景**：
- 后台运行训练
- 不需要看窗口
- 服务器环境

**修复方法**：
使用真正的无头模式，但需要额外配置：

```bash
# 无GUI版本的JVM参数
JVM_ARGS="-Xmx2G -Xms512M"
JVM_ARGS="$JVM_ARGS -XX:+UseG1GC -XX:MaxGCPauseMillis=50"
JVM_ARGS="$JVM_ARGS -XX:+DisableExplicitGC -XX:+ParallelRefProcEnabled"

# 真正的无头模式
JVM_ARGS="$JVM_ARGS -Djava.awt.headless=true"
JVM_ARGS="$JVM_ARGS -Dorg.lwjgl.util.NoChecks=true"

# OpenGL offscreen渲染
JVM_ARGS="$JVM_ARGS -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
JVM_ARGS="$JVM_ARGS -XstartOnFirstThread"

# Malmo设置
JVM_ARGS="$JVM_ARGS -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin"
JVM_ARGS="$JVM_ARGS -Dfile.encoding=UTF-8"
```

**并且在环境变量中设置**：
```bash
export MINEDOJO_HEADLESS=1
export DISPLAY=  # 清空DISPLAY变量
```

## 快速修复步骤

### 步骤1：确定你的使用场景

**我想看到窗口（调试）**：
```bash
# 使用有GUI版本
bash scripts/apply_macos_stability_fix.sh --gui
```

**我不需要窗口（训练）**：
```bash
# 使用无GUI版本
bash scripts/apply_macos_stability_fix.sh --headless
```

### 步骤2：手动修改（如果需要）

如果自动脚本不工作，手动编辑launchClient.sh：

```bash
# 1. 找到MineDojo安装路径
MINEDOJO_PATH=$(python3 -c "import minedojo, os; print(os.path.dirname(minedojo.__file__))")

# 2. 编辑启动脚本
nano $MINEDOJO_PATH/sim/Malmo/Minecraft/launchClient.sh

# 3. 找到包含 -Djava.awt.headless=true 的行
# 4. 删除或注释掉这个参数（如果你想看窗口）
# 5. 保存并退出
```

### 步骤3：验证修复

```bash
# 测试启动
python scripts/test_macos_stability.py --check-gui
```

## 诊断工具

### 检查当前配置

```bash
# 运行诊断脚本
python scripts/diagnose_white_screen.py
```

这个脚本会检查：
- ✓ 是否设置了headless模式
- ✓ OpenGL参数是否正确
- ✓ 显示器配置
- ✓ Java AWT可用性

### 查看详细启动日志

```bash
# 启用详细日志
export MINEDOJO_VERBOSE=1
python your_script.py
```

## 常见组合问题

### 问题1：白屏 + 崩溃

**症状**：有时白屏，有时崩溃

**原因**：同时存在OpenGL兼容性和渲染问题

**解决**：
1. 先应用崩溃修复
2. 然后修复白屏问题
3. 使用有GUI模式测试

```bash
bash scripts/apply_macos_stability_fix.sh --gui --with-crash-fix
```

### 问题2：窗口闪现后消失

**症状**：窗口出现一瞬间就关闭或白屏

**原因**：渲染初始化失败

**解决**：
```bash
# 添加调试参数
-Dorg.lwjgl.util.Debug=true
-Dorg.lwjgl.util.DebugLoader=true
```

### 问题3：多显示器环境白屏

**症状**：特定显示器上白屏

**解决**：
```bash
# 强制使用主显示器
-Dorg.lwjgl.opengl.Window.undecorated=false
```

## 针对不同macOS版本的建议

### macOS Sonoma (14.x)

```bash
# 需要额外的兼容性参数
-Dapple.awt.application.appearance=system
```

### macOS Ventura (13.x)

```bash
# 通常默认参数即可
# 移除headless即可解决
```

### Apple Silicon (M1/M2/M3)

```bash
# Rosetta 2环境，需要所有稳定性参数
# 推荐使用有GUI模式，因为headless在Rosetta下更不稳定
```

## 创建自适应启动配置

创建一个智能启动脚本，根据环境自动选择：

```bash
#!/bin/bash
# scripts/launch_minecraft_adaptive.sh

# 检测是否有显示器
if [ -n "$DISPLAY" ] || [ "$(uname)" = "Darwin" ]; then
    echo "检测到GUI环境，使用有GUI模式"
    USE_GUI=true
else
    echo "无GUI环境，使用headless模式"
    USE_GUI=false
fi

# 设置JVM参数
if [ "$USE_GUI" = true ]; then
    # 有GUI版本
    JVM_ARGS="-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
    JVM_ARGS="$JVM_ARGS -XstartOnFirstThread"
else
    # 无GUI版本
    JVM_ARGS="-Djava.awt.headless=true"
    JVM_ARGS="$JVM_ARGS -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
    JVM_ARGS="$JVM_ARGS -XstartOnFirstThread"
fi

# 公共参数
JVM_ARGS="$JVM_ARGS -Xmx2G -Xms512M"
JVM_ARGS="$JVM_ARGS -XX:+UseG1GC"

echo "使用参数: $JVM_ARGS"
```

## Python代码中的解决方案

如果你想在Python代码中控制：

```python
import os
import minedojo

# 方案1：强制有GUI模式
os.environ['MINEDOJO_HEADLESS'] = '0'
os.environ.pop('DISPLAY', None)  # 不要移除DISPLAY

# 方案2：强制无GUI模式（更复杂，需要offscreen渲染支持）
os.environ['MINEDOJO_HEADLESS'] = '1'
os.environ['DISPLAY'] = ''

# 创建环境
env = minedojo.make(
    task_id="harvest_log",
    image_size=(160, 256)
)
```

## 性能对比

| 模式 | 启动速度 | 稳定性 | 适用场景 |
|------|----------|--------|----------|
| 有GUI | 慢 (~20秒) | ⭐⭐⭐ 中等 | 调试、演示 |
| 无GUI | 快 (~15秒) | ⭐⭐ 较低 | 批量训练 |
| 混合 | 中 (~18秒) | ⭐⭐⭐⭐ 好 | 推荐 |

**推荐**：使用有GUI模式，因为：
1. 更稳定（白屏概率低）
2. 可以看到实际行为
3. 容易调试
4. 只是启动慢一点

如果一定要无GUI，建议使用Docker或远程服务器。

## 验证修复成功

修复后应该看到：
- ✅ 窗口正常显示游戏画面
- ✅ 看到Minecraft加载进度
- ✅ 能看到方块、天空、地面
- ✅ 观察窗口显示正常渲染

**正常窗口应该显示**：
```
┌─────────────────────────────┐
│     Minecraft 1.11.2        │
├─────────────────────────────┤
│   [天空]                     │
│   [地平线]                   │
│   [草地/方块]                │
│   [手/物品栏]                │
└─────────────────────────────┘
```

## 相关问题

### OpenGL版本不支持

**错误**：`OpenGL 2.0 required`

**解决**：确保启用软件渲染
```bash
-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true
```

### 窗口尺寸异常

**症状**：窗口太小或太大

**解决**：在Python中指定尺寸
```python
env = minedojo.make(
    task_id="harvest_log",
    image_size=(640, 360)  # 16:9 比例
)
```

### 渲染卡顿

**症状**：窗口显示但很卡

**解决**：降低分辨率或关闭垂直同步
```bash
-Dorg.lwjgl.opengl.Display.enableHighDPI=false
```

## 文件清单

创建以下文件来实现完整的修复：

```
scripts/
├── apply_macos_stability_fix.sh        # 主修复脚本（需要更新）
├── diagnose_white_screen.py            # 白屏诊断工具（新建）
└── launch_minecraft_adaptive.sh        # 自适应启动（新建）

docker/
├── minedojo_macos_stability.patch           # 崩溃修复
└── minedojo_macos_white_screen_fix.patch    # 白屏修复（新建）
```

## 总结

**白屏问题的核心**：
- ❌ 不要在想看窗口时使用 `-Djava.awt.headless=true`
- ✅ 始终包含 `-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true`
- ✅ 使用 `-XstartOnFirstThread` (macOS必需)

**推荐配置**（调试/开发）：
```bash
java -Xmx2G -Xms512M \
  -XX:+UseG1GC \
  -XstartOnFirstThread \
  -Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true \
  -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin \
  -jar MalmoMod.jar
```

**注意**：移除了 `-Djava.awt.headless=true`！

## 更新记录

- **2025-11-18**: 初始版本
  - 识别headless参数导致的白屏问题
  - 提供有GUI和无GUI两种解决方案
  - 创建诊断和修复工具

