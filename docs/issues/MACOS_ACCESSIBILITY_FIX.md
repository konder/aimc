# macOS辅助功能权限修复指南

## ❌ **问题现象**

运行pynput时出现：
```
This process is not trusted! Input event monitoring will not be possible 
until it is added to accessibility clients.
```

所有按键都检测不到（100% IDLE帧）。

## 🔧 **解决方案**

### **方法1: 授予Terminal辅助功能权限** ⭐ 推荐

#### **步骤1: 打开系统设置**

1. 点击左上角  菜单
2. 选择 **System Settings**（系统设置）
3. 点击 **Privacy & Security**（隐私与安全性）
4. 在左侧列表中找到 **Accessibility**（辅助功能）

#### **步骤2: 添加Terminal**

1. 点击 **Accessibility** 进入
2. 点击右下角的 **+** 按钮（可能需要先点击左下角🔒解锁）
3. 找到并选择：
   - `/Applications/Utilities/Terminal.app`
   - 或者 `/Applications/iTerm.app`（如果你使用iTerm）
4. 勾选该应用旁边的复选框✅

#### **步骤3: 验证**

```bash
# 关闭当前Terminal窗口，重新打开
conda activate minedojo-x86
pythonf test_pynput_realtime.py
```

如果正常，应该看到：
```
✅ 实时按键监听器已启动
测试开始！（20秒）

[按住W键]
[  2.1s] 帧 42: Forward | IDLE: 2/42 (4.8%)
[  2.2s] 帧 44: Forward | IDLE: 2/44 (4.5%)
```

---

### **方法2: 使用sudo运行** ⚠️ 不推荐

```bash
sudo python test_pynput_realtime.py
```

**缺点**: 
- 需要输入密码
- 安全风险
- conda环境可能失效

---

### **方法3: 使用pygame替代方案** 🎮

如果无法授予辅助功能权限，可以使用pygame：

```bash
conda activate minedojo-x86
conda install -y pygame

# 使用pygame版本
python demo_pygame_multikey.py
```

**优点**:
- 无需特殊权限
- pygame自己管理输入

**缺点**:
- 需要pygame窗口（不能与OpenCV窗口共存）
- 实现稍复杂

---

## 🔍 **验证权限**

### **检查当前权限状态**

```bash
# 方法1: 通过系统设置
System Settings → Privacy & Security → Accessibility
# 查看Terminal是否在列表中且已勾选

# 方法2: 运行测试
python -c "from pynput import keyboard; l = keyboard.Listener(on_press=lambda k: None); l.start(); import time; time.sleep(0.5); print('✅ 权限OK' if l.running else '❌ 无权限'); l.stop()"
```

---

## 📸 **截图指南**

### **macOS Ventura/Sonoma (13.0+)**

1. **系统设置界面**:
   ```
   System Settings (系统设置)
   └── Privacy & Security (隐私与安全性)
       └── Accessibility (辅助功能)
           ├── Terminal.app ✅
           └── [+ 按钮添加更多应用]
   ```

2. **旧版macOS (Big Sur/Monterey)**:
   ```
   System Preferences (系统偏好设置)
   └── Security & Privacy (安全性与隐私)
       └── Privacy (隐私)
           └── Accessibility (辅助功能)
               ├── Terminal ✅
               └── [+ 添加]
   ```

---

## 🚨 **常见问题**

### Q1: 已经添加Terminal但还是不行？

**解决**:
1. 确保Terminal旁边的复选框已勾选✅
2. **完全退出**Terminal（Command+Q），重新打开
3. 重启Mac

### Q2: 找不到"辅助功能"选项？

**解决**:
- macOS版本可能不同，搜索"Accessibility"
- 或者在系统设置中搜索"accessibility"

### Q3: 锁定图标🔒无法点击+按钮？

**解决**:
1. 点击左下角的🔒图标
2. 输入管理员密码
3. 解锁后再点击+按钮

### Q4: 不想授予辅助功能权限怎么办？

**解决**:
- 使用**方案B: pygame**（见下方）
- 或者使用**方案C: 混合模式**（每帧等待 + 优化）

---

## 🎯 **替代方案**

### **方案B: pygame实时录制**

如果不想授予辅助功能权限，可以创建一个pygame版本：

```bash
# 安装pygame
conda install -y pygame

# 使用pygame录制（待实现）
python tools/record_manual_chopping_pygame.py
```

**优势**:
- 无需特殊权限
- pygame窗口直接显示游戏画面
- 同样支持多键检测

**劣势**:
- 需要修改显示逻辑（用pygame窗口代替OpenCV）

### **方案C: 优化每帧等待模式**

保留原有`cv2.waitKey(0)`，但优化按键映射：

```python
# 添加更多组合键
key_map = {
    ord('u'): 'forward+jump',
    ord('y'): 'forward+attack',
    ord('h'): 'forward+attack+jump',
    # ... 更多组合
}
```

**优势**: 无需权限，立即可用
**劣势**: 静态帧仍然较多（但可控制在30-40%）

---

## ✅ **推荐流程**

### **优先级1: 授予辅助功能权限** ⭐

这是最佳方案，完全发挥pynput优势：

1. 系统设置 → 隐私与安全性 → 辅助功能
2. 添加Terminal并勾选
3. 重启Terminal
4. 测试: `python test_pynput_realtime.py`

### **优先级2: 使用pygame**

如果无法授予权限（公司电脑、受限环境）：

1. 安装pygame
2. 等待pygame版本录制工具（或使用现有demo）

### **优先级3: 优化现有方案**

继续使用`cv2.waitKey(0)`，但：
- 增加组合键映射
- 录制时更注意按键节奏
- 接受30-40%的静态帧占比

---

## 📝 **验证成功标志**

运行测试后，应该看到：

```bash
python test_pynput_realtime.py

# ✅ 成功
✅ 实时按键监听器已启动
测试开始！（20秒）

[按住W键5秒]
[  1.0s] 帧 20: Forward | IDLE: 0/20 (0.0%)
[  2.0s] 帧 40: Forward | IDLE: 0/40 (0.0%)
[  3.0s] 帧 60: Forward | IDLE: 0/60 (0.0%)

最长连续Forward: 100 帧
✅ 验证通过！按住W键时能持续检测到Forward动作
✅ 静态帧占比 < 10%: 符合预期！
```

---

**状态**: 等待用户授予Terminal辅助功能权限后重新测试

