# MineRL 1.0.0 快速参考 🚀

## ⚡ 一键运行

```bash
./scripts/run_minedojo_x86.sh python your_script.py
```

---

## 📦 快速安装

```bash
# 1. 从 GitHub 安装
cd /tmp && git clone https://github.com/minerllabs/minerl.git
cd minerl && git checkout v1.0.0
git submodule update --init --recursive

# 2. 修改 MCP-Reborn/launchClient.sh（添加 -XstartOnFirstThread）

# 3. 安装
pip install -e .

# 4. 安装正确的 OpenCV ⭐⭐⭐
pip install opencv-python==4.8.1.78 --force-reinstall
```

---

## 🎮 基本使用

```python
import gym
import minerl

# 创建环境
env = gym.make("MineRLBasaltBuildVillageHouse-v0")

# 重置
obs = env.reset()

# 运行
for _ in range(100):
    action = env.action_space.noop()
    action['forward'] = 1
    obs, reward, done, info = env.step(action)
    
    # ✅ 显示窗口（已修复）
    env.render()
    
    if done:
        obs = env.reset()

env.close()
```

---

## ⚙️ 关键配置

### 必需版本

```
OpenCV: 4.8.1.78  ← ⭐⭐⭐ 关键！
Java: 1.8 + -XstartOnFirstThread
gym: 0.19.0
numpy: 1.24.3
```

### 环境变量

```bash
export JAVA_OPTS="-XstartOnFirstThread -Xmx4G"
unset DISPLAY  # 如果安装了 XQuartz
```

---

## 🔍 验证安装

```bash
./scripts/run_minedojo_x86.sh python -c "
import gym, minerl, cv2
print('✓ MineRL installed')
print('✓ OpenCV:', cv2.__version__)
env = gym.make('MineRLBasaltFindCave-v0')
env.close()
print('✓ All OK')
"
```

---

## 📚 BASALT 任务

```python
'MineRLBasaltFindCave-v0'              # 找洞穴
'MineRLBasaltMakeWaterfall-v0'         # 造瀑布
'MineRLBasaltCreateVillageAnimalPen-v0'  # 建动物圈
'MineRLBasaltBuildVillageHouse-v0'     # 建村庄房屋
```

---

## 🐛 故障排除

### env.render() 失败？

```bash
# 检查 OpenCV 版本
pip list | grep opencv
# 必须是: opencv-python 4.8.1.78

# 重新安装
pip install opencv-python==4.8.1.78 --force-reinstall
```

### 窗口不显示？

```bash
# 检查 launchClient.sh
grep "XstartOnFirstThread" \
  $(python -c "import minerl; print(minerl.__path__[0])")/MCP-Reborn/launchClient.sh
```

### GLFW 错误？

```bash
# 确保在 x86 模式
uname -m  # 应该输出: x86_64

# 确保 JAVA_OPTS 正确
echo $JAVA_OPTS  # 应该包含: -XstartOnFirstThread
```

---

## 📖 详细文档

完整指南: `docs/guides/MINERL_GUIDE.md`

---

## 🎯 核心要点

1. ⭐ **OpenCV 4.8.1.78 是关键**
2. ⭐ **launchClient.sh 需要添加 -XstartOnFirstThread**
3. ⭐ **清除 DISPLAY 变量（如果有 XQuartz）**

---

**最后更新**: 2025-10-28  
**状态**: ✅ 完全可用

