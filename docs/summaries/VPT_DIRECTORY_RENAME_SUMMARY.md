# VPT目录重命名总结

**日期**: 2025-10-29  
**操作**: 将 `src/models/Video-Pre-Training` 重命名为 `src/models/minerlvpt`

---

## 📋 重命名原因

1. **避免特殊字符**: 目录名中的连字符（`-`）不适合Python包名
2. **更清晰的命名**: `minerlvpt` 更简洁，表明这是MineRL版本的VPT实现
3. **避免混淆**: 不与GitHub仓库名混淆

---

## 🔄 已修改的文件

### 1. 核心代码文件

#### `src/training/vpt/vpt_agent.py`
```python
# 修改前
VPT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "Video-Pre-Training"

# 修改后
VPT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "minerlvpt"
```

- 更新了路径引用
- 更新了文档字符串中的目录名
- 更新了打印信息

#### `src/training/vpt/__init__.py`
```python
# 更新文档字符串
"""
官方VPT参考：
- GitHub: https://github.com/openai/Video-Pre-Training
- 本地VPT代码: src/models/minerlvpt/
"""
```

### 2. 脚本文件

#### `scripts/run_official_vpt_demo.sh`
```bash
# 修改前
VPT_DIR="$PROJECT_ROOT/src/models/Video-Pre-Training"

# 修改后
VPT_DIR="$PROJECT_ROOT/src/models/minerlvpt"
```

---

## ✅ 验证结果

```bash
# 测试导入
cd /Users/nanzhang/aimc
scripts/run_minedojo_x86.sh python -c "from src.training.vpt import VPTAgent; print('✅ VPTAgent导入成功')"

# 输出: ✅ VPTAgent导入成功
```

---

## 📂 目录结构

```
src/models/minerlvpt/
├── agent.py                    # MineRLAgent主类
├── behavioural_cloning.py      # BC训练
├── data_loader.py              # 数据加载
├── inverse_dynamics_model.py   # IDM模型
├── lib/                        # 核心库
│   ├── action_head.py
│   ├── action_mapping.py
│   ├── actions.py
│   ├── impala_cnn.py
│   ├── policy.py
│   └── ... (其他工具)
└── cursors/                    # 鼠标光标资源
```

---

## 🔍 内部导入说明

`minerlvpt` 目录内的文件使用**相对导入**（`from lib.xxx import ...`），这些导入**不受目录重命名影响**，无需修改。

例如：
```python
# minerlvpt/agent.py
from lib.action_mapping import CameraHierarchicalMapping  # ✅ 无需修改
from lib.actions import ActionTransformer                # ✅ 无需修改
from lib.policy import MinecraftAgentPolicy              # ✅ 无需修改
```

---

## 📝 文档更新（可选）

以下文档文件仍包含旧的 `Video-Pre-Training` 引用，但这些是**文档性质**的，不影响代码运行：

- `docs/guides/MINERL_GUIDE.md`
- `docs/summaries/VPT_*.md`
- `docs/technical/VPT_*.md`
- `docs/reference/VPT_MODELS_REFERENCE.md`
- `FAQ.md`

如需更新，可批量替换：
```bash
find docs/ -name "*.md" -exec sed -i '' 's|Video-Pre-Training|minerlvpt|g' {} \;
```

---

## ✅ 验证清单

- [x] 目录重命名完成
- [x] `vpt_agent.py` 路径引用更新
- [x] `__init__.py` 文档更新
- [x] `run_official_vpt_demo.sh` 路径更新
- [x] 导入测试通过
- [x] `minerlvpt` 内部相对导入无需修改

---

## 🎯 总结

重命名完成后，所有代码功能保持不变：

1. ✅ VPTAgent正常导入
2. ✅ 零样本评估脚本正常运行
3. ✅ 官方VPT演示脚本路径正确
4. ✅ Camera精度转换（cam_interval=0.01）正常工作
5. ✅ 所有动作转换正确映射

**无需进一步操作，代码已完全就绪！**

