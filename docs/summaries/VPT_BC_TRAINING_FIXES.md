# VPT BC训练问题修复总结

## 🐛 遇到的问题列表

### 问题1: load_vpt_policy() 参数错误
```
错误: load_vpt_policy() got an unexpected keyword argument 'policy'
```
**修复**: 函数会自动创建policy，移除policy参数

### 问题2: 模块导入错误
```
错误: ModuleNotFoundError: No module named 'src'
```
**修复**: 修正PROJECT_ROOT路径（`../..` → `../../..`）

### 问题3: 不支持的参数
```
错误: unrecognized arguments: --log-interval 10
```
**修复**: 从测试脚本中移除 `--log-interval` 参数

### 问题4: 数据加载失败
```
错误: 加载 0 个episodes，IndexError: list index out of range
```
**修复**: ExpertDataset适配Web录制系统的数据格式（frame_xxxx.npy包含dict）

### 问题5: 图像通道顺序错误 ⭐
```
错误: expected input to have 3 channels, but got 256 channels instead
原因: 数据是CHW格式(3, 160, 256)，VPT期待HWC格式
```
**修复**: 在`__getitem__`中自动检测并转换CHW→HWC

### 问题6: 评估脚本找不到模型
```
错误: ✗ 未找到模型文件
原因: 脚本查找latest.pth或epoch_X.pth，但训练保存的是best_model.pth
```
**修复**: 优先查找best_model.pth，其次final_model.pth

### 问题7: 评估时图像通道顺序错误 ⭐⭐
```
错误: expected input to have 3 channels, but got 128 channels instead
原因: 评估脚本中也需要CHW→HWC转换
```
**修复**: 在evaluate_bc_vpt.py的predict方法中添加相同的转换逻辑

### 问题8: 评估环境尺寸与录制不一致 ⭐⭐⭐
```
错误: 评估环境使用128x128，但录制时使用160x256
影响: 
  - 视野不一致（160x256更宽）
  - 纵横比不同（变形）
  - 无法公平比较效果
```
**修复**: 评估环境改用(160, 256)与录制一致，predict中自动resize到128x128给VPT

---

## ✅ 最终修复

### 修复的文件

1. **src/training/vpt/train_bc_vpt.py**
   - 修复PROJECT_ROOT路径
   - 重写ExpertDataset数据加载
   - 添加CHW→HWC自动转换
   - 添加dtype检查和转换

2. **src/training/vpt/evaluate_bc_vpt.py**
   - 完全重写，使用新API
   - 移除旧的依赖
   - 添加CHW→HWC自动转换（predict方法）
   - 优化dtype处理逻辑
   - 修复环境尺寸：128x128 → 160x256（与录制一致）

3. **scripts/vpt_quick_test.sh**
   - 修复load_vpt_policy调用
   - 移除--log-interval参数
   - 修复模型查找逻辑（best_model.pth优先）

4. **scripts/vpt_full_training.sh**
   - 移除--log-interval参数
   - 修复模型查找逻辑（best_model.pth优先）
   - 同步所有修复

---

## 🎯 关键修复1：训练时的图像格式转换

```python
# train_bc_vpt.py 的 ExpertDataset.__getitem__
def __getitem__(self, idx):
    obs = self.all_obs[idx]  # 可能是 (C, H, W) 或 (H, W, C)
    
    # 检查并转换为HWC格式
    if obs.shape[0] == 3:  # CHW格式
        if obs.shape[0] < obs.shape[1] and obs.shape[0] < obs.shape[2]:
            obs = np.transpose(obs, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    
    # 确保uint8类型
    if obs.dtype != np.uint8:
        if obs.max() <= 1.0:
            obs = (obs * 255).astype(np.uint8)
        else:
            obs = obs.astype(np.uint8)
    
    # Resize到128x128
    if obs.shape[:2] != (128, 128):
        obs = cv2.resize(obs, (128, 128))
    
    # 转换为tensor [0,1]
    obs = torch.from_numpy(obs).float() / 255.0
    
    return obs, action
```

## 🎯 关键修复2：评估时的图像格式转换

```python
# evaluate_bc_vpt.py 的 MinedojoActionAdapter.predict
def predict(self, obs, deterministic=True):
    # 检查并转换为HWC格式
    if obs.shape[0] == 3:  # CHW格式
        if obs.shape[0] < obs.shape[1] and obs.shape[0] < obs.shape[2]:
            obs = np.transpose(obs, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    
    # 处理数据类型和范围
    is_normalized = obs.dtype in [np.float32, np.float64] and obs.max() <= 1.0
    
    if not is_normalized:
        obs = obs.astype(np.float32) / 255.0  # uint8 -> float
    else:
        obs = obs.astype(np.float32)  # 确保是float32
    
    # Resize到128x128
    if obs.shape[:2] != (128, 128):
        obs = cv2.resize(obs, (128, 128))
    
    # 添加batch维度并前向传播
    obs = obs[np.newaxis, ...]  # (1, H, W, C)
    action = self.forward(obs)
    
    return action
```

---

## 📊 验证结果

### 训练阶段
```
✓ 找到 101 个episode目录
✓ 加载完成: 总样本数 22,437
✓ 原始图像shape: (3, 160, 256) - CHW格式
✓ 自动转换为: (160, 256, 3) - HWC格式
✓ Resize到: (128, 128, 3) - VPT需要的尺寸
✓ 训练集: 20,194 样本
✓ 验证集: 2,243 样本
✓ VPT权重加载: Missing=0, Unexpected=0
✓ 训练完成: 最佳loss 0.8770
✓ 模型已保存: best_model.pth (1.1GB)
```

### 评估阶段
```
✓ 加载模型: best_model.pth
✓ 创建环境: harvest_1_log (160x256) - 与录制一致
✓ 环境返回: (3, 160, 256) float32 [0,1] - CHW格式
✓ 自动转换为: (160, 256, 3) - HWC格式
✓ Resize到: (128, 128, 3) - VPT需要的尺寸
✓ 前向传播成功
✓ 开始评估...
```

---

## 🚀 现在可以训练了

```bash
# 快速测试（2 epochs）
bash scripts/vpt_quick_test.sh

# 完整训练（20 epochs）
bash scripts/vpt_full_training.sh
```

---

## 📝 经验教训

1. **图像格式很关键** ⭐⭐⭐
   - MineDojo/Gym通常使用CHW格式输出
   - VPT期待HWC格式输入
   - 训练和评估都需要进行转换
   - 需要在Dataset和Adapter两处都添加转换逻辑

2. **数据格式要匹配**
   - Web录制系统保存为逐帧的dict
   - 需要正确解析{'observation': ..., 'action': ...}

3. **路径计算要准确**
   - `src/training/vpt/` 到项目根是 `../../..` 不是 `../..`

4. **dtype处理要优化**
   - 避免不必要的uint8↔float转换
   - 如果已经是[0,1]范围的float，直接使用
   - 减少精度损失和计算开销

5. **环境一致性很关键** ⭐⭐⭐
   - 评估环境必须与录制/训练环境一致
   - 图像尺寸、视野、纵横比都要匹配
   - 模型可以在内部resize，但输入环境要一致
   - 否则无法公平比较性能

6. **逐步验证**
   - ✅ 环境验证
   - ✅ 数据加载验证  
   - ✅ 模型创建验证
   - ✅ 前向传播验证（训练）
   - ✅ 训练验证
   - ✅ 前向传播验证（评估）
   - ✅ 环境一致性验证
   - ⏳ 评估验证

---

## 相关文档

- `docs/reference/VPT_MODELS_REFERENCE.md` - VPT模型选择
- `tools/test_vpt_env.py` - VPT环境验证脚本

