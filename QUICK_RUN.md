# 快速运行指南 - 获得木头训练

## 🚀 一键启动

### 在你的终端中运行：

```bash
# 1. 进入项目目录
cd /Users/nanzhang/aimc

# 2. 初始化 conda（只需第一次运行）
source /usr/local/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate minedojo

# 3. 快速测试（10K步，约5-10分钟，推荐第一次运行）
./scripts/train_get_wood.sh test --mineclip

# 或者：标准训练（200K步，约2-3小时）
./scripts/train_get_wood.sh standard --mineclip
```

## 📊 监控训练

训练开始后，脚本会自动启动 TensorBoard：

```bash
# 浏览器打开（脚本会显示这个地址）
http://localhost:6006
```

在 TensorBoard 中查看：
- **SCALARS** 标签页：
  - `rollout/ep_rew_mean`: 平均奖励（应该逐渐上升）
  - `info/mineclip_similarity`: MineCLIP相似度（0-1之间）
  - `info/mineclip_reward`: MineCLIP密集奖励

## 📁 训练结果

- **检查点**：`checkpoints/get_wood/`
- **训练日志**：`logs/training/`
- **TensorBoard日志**：`logs/tensorboard/`

## 🔍 测试模式 vs 标准模式

| 模式 | 总步数 | 训练时间 | 用途 |
|------|--------|----------|------|
| test | 10K | 5-10分钟 | 快速验证环境是否正常 |
| quick | 50K | 30-60分钟 | 初步验证训练效果 |
| standard | 200K | 2-3小时 | 正式训练（推荐） |
| long | 500K | 6-10小时 | 追求更高性能 |

## ⚡ 性能优化

脚本已启用：
- ✅ Java无头模式（不显示游戏窗口）
- ✅ 自动设备检测（优先使用MPS/CUDA）
- ✅ 并行环境（standard模式使用2个环境）

## 🐛 常见问题

### 1. conda activate 报错

**错误**：`CondaError: Run 'conda init' before 'conda activate'`

**解决**：
```bash
# 运行这行命令初始化
source /usr/local/Caskroom/miniforge/base/etc/profile.d/conda.sh
```

### 2. 找不到 minedojo 环境

**错误**：`EnvironmentNotFoundError: Could not find conda environment: minedojo`

**解决**：
```bash
# 创建环境
conda create -n minedojo python=3.9 -y
conda activate minedojo
pip install -r requirements.txt
```

### 3. TensorBoard 无法访问

**检查**：
```bash
# 查看 TensorBoard 是否运行
lsof -i :6006

# 如果没有运行，手动启动
tensorboard --logdir logs/tensorboard --port 6006
```

## 📝 下一步

训练完成后，你可以：

1. **查看训练效果**：在 TensorBoard 中分析曲线
2. **评估模型**：运行评估脚本（即将添加）
3. **调整超参数**：修改 `train_get_wood.py` 中的参数
4. **扩展到其他技能**：参考 `docs/guides/TASKS_QUICK_START.md`

---

## 🎯 预期结果

使用 MineCLIP（`--mineclip`）：
- **首次成功**：约20K-50K步
- **稳定成功**：约100K-200K步
- **成功率**：85%+（200K步后）

不使用 MineCLIP：
- **首次成功**：约100K-200K步
- **稳定成功**：约500K步
- **成功率**：70%（500K步后）

**加速效果**：3-5倍 🚀

---

祝训练顺利！如有问题，查看 `docs/FAQ.md`

