# 录制脚本：每帧等待输入模式

## 📌 问题背景

之前的录制脚本存在严重问题：
- `cv2.waitKey(1)` 无法可靠检测"按住"状态
- 导致大量意外的静态帧（63.4%）
- 例如：按住W键前进，只有第1帧记录了Forward，其余帧都是IDLE
- BC训练时模型学到"站着不动"的错误行为

## ✅ 解决方案

**新录制模式：每帧等待输入**
- 每帧暂停，等待用户按键
- 按下按键后，执行该动作并自动进入下一帧
- 用户完全控制每一帧的动作
- 不会产生意外的静态帧

## 🎮 使用方法

### 启动录制

```bash
conda activate minedojo-x86
bash scripts/run_minedojo_x86.sh python tools/record_manual_chopping.py \
    --base-dir data/expert_demos/harvest_1_log_v2 \
    --max-frames 500 \
    --camera-delta 1
```

### 录制流程

1. **环境启动**: MineDojo环境加载完成
2. **等待输入**: 画面暂停，显示"Waiting for key"
3. **按下按键**: 执行该动作（例如：W=前进）
4. **自动进帧**: 环境step，显示新画面，再次等待输入
5. **重复**: 持续按键控制直到任务完成

### 按键说明

#### 移动控制
- `W` - 前进（1帧）
- `S` - 后退（1帧）
- `A` - 左移（1帧）
- `D` - 右移（1帧）
- `Space` - 跳跃（1帧）

#### 相机控制
- `I` - 向上看
- `K` - 向下看
- `J` - 向左看
- `L` - 向右看

#### 动作
- `F` - 攻击/挖掘 ⭐（砍树）

#### 组合动作
- `U` - 前进+跳跃（同时执行）

#### 特殊
- `.` (句号) - IDLE（站立不动，无动作帧）
  - 适用于需要观察环境的时候

#### 系统
- `Q` - 重新录制当前episode（不保存）
- `ESC` - 退出程序（不保存当前episode）

## 📊 统计信息

录制完成后，脚本会自动分析并显示：

```
✓ episode_000 已保存: 150 帧
  - 150 PNG图片
  - 150 NPY文件（BC训练）
  - 静态帧: 5/150 (3.3%)      ← 现在应该很低！
  - 攻击帧: 80/150 (53.3%)
```

同时保存到 `episode_XXX/metadata.txt`:

```
Action Statistics:
  IDLE frames: 5/150 (3.3%)
  Forward: 30 frames
  Back: 0 frames
  Left: 5 frames
  Right: 3 frames
  Jump: 8 frames
  Attack: 80 frames
  Camera: 25 frames
```

## 💡 录制技巧

### 砍树任务示例

1. **寻找树木**: 按 `J`/`L` 转动视角
2. **接近树木**: 连续按 `W` 前进（每次1帧）
3. **调整视角**: 按 `I`/`K` 调整俯仰角
4. **开始砍树**: 连续按 `F` 攻击（每次1帧）
5. **任务完成**: 环境自动检测到完成，保存数据

### 连续动作示例

如果需要连续前进10帧：
```
按W → 看画面 → 按W → 看画面 → 按W → ... （重复10次）
```

如果需要边前进边攻击：
```
按W → 按F → 按W → 按F → ... （交替）
```

或使用组合键（未来可扩展）。

## ⚠️ 注意事项

1. **每帧都需要按键**
   - 不按键，画面不会前进
   - 想站立不动，按 `.` 键

2. **点击窗口获得焦点**
   - OpenCV窗口需要处于激活状态
   - 否则按键无法被捕获

3. **可以随时重录**
   - 按 `Q` 重新录制当前episode
   - 不会影响已保存的其他episode

4. **静态帧应该很少**
   - 只有主动按 `.` 键才会产生静态帧
   - 正常录制应该 < 5% 静态帧

## 🔄 与旧版本对比

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 按键检测 | `cv2.waitKey(1)` | `cv2.waitKey(0)` |
| 连续按键 | 不可靠 | 精确控制 |
| 静态帧 | 63.4% | < 5% |
| 控制方式 | 按住按键 | 每帧按键 |
| 适用场景 | 实时控制 | 精确录制 |

## 🚀 下一步

录制完新数据后：

1. **检查质量**
   ```bash
   cat data/expert_demos/harvest_1_log_v2/episode_000/metadata.txt
   ```

2. **训练BC模型**
   ```bash
   python src/training/train_bc.py \
       --data data/expert_demos/harvest_1_log_v2 \
       --output checkpoints/dagger/harvest_1_log/bc_baseline_v2.zip \
       --epochs 50 \
       --device cpu
   ```

3. **评估模型**
   ```bash
   python tools/evaluate_policy.py \
       --model checkpoints/dagger/harvest_1_log/bc_baseline_v2.zip \
       --episodes 10
   ```

## 🐛 故障排除

**Q: 按键没有反应**
- A: 点击OpenCV窗口获得焦点

**Q: 想要连续执行多帧相同动作，很累**
- A: 可以使用宏或脚本辅助，或者考虑减少帧数

**Q: 静态帧占比还是很高**
- A: 检查是否误按了 `.` 键

**Q: 想要更快的录制速度**
- A: 这个模式牺牲速度换取精确度，适合高质量数据收集

