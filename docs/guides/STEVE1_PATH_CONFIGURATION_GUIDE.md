# STEVE-1 路径配置指南

> **目的**: 将 STEVE-1 集成到 AIMC 项目中，修正所有路径配置  
> **位置**: `src/training/steve1/`  
> **权重目录**: `/Users/nanzhang/aimc/data/weights/`

---

## ✅ 已完成的修改

### **1. 配置文件路径修正** ✅

修改了 `src/training/steve1/config.py`：

```python
import os

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# MineCLIP 配置
MINECLIP_CONFIG = {
    'ckpt': {
        'path': os.path.join(DATA_DIR, "weights/mineclip/attn.pth"),
        ...
    }
}

# Prior 配置
PRIOR_INFO = {
    'model_path': os.path.join(DATA_DIR, 'weights/steve1/steve1_prior.pt'),
    ...
}
```

### **2. Shell 脚本路径修正** ✅

修改了以下脚本：
- `1_gen_paper_videos.sh`
- `2_gen_vid_for_text_prompt.sh`
- `3_run_interactive_session.sh`
- `1_generate_dataset.sh`
- `3_train.sh`

所有脚本现在使用动态路径：

```bash
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

python "$SCRIPT_DIR/run_agent/run_agent.py" \
--in_model "$PROJECT_ROOT/data/weights/vpt/2x.model" \
--in_weights "$PROJECT_ROOT/data/weights/steve1/steve1.weights" \
...
```

### **3. 核心 Python 文件修正** ✅

修改了以下文件添加路径初始化：
- `run_agent/run_agent.py`
- `run_agent/paper_prompts.py`
- `utils/mineclip_agent_env_utils.py`

每个文件开头添加：

```python
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### **4. 包初始化文件** ✅

创建了 `src/training/steve1/__init__.py`：

```python
import os
import sys

# Get the absolute path of the parent directory (src/training)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# Add to sys.path if not already there
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Project root and data directory
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '../../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
```

---

## ⚠️ 需要手动修改的文件

由于有 24 个 Python 文件包含 `from steve1.` 导入，建议使用以下脚本批量修改：

### **批量修改脚本**

创建文件 `src/training/steve1/fix_imports.py`：

```python
#!/usr/bin/env python3
"""
批量为 STEVE-1 的 Python 文件添加路径初始化代码
"""
import os
import re

STEVE1_DIR = os.path.dirname(os.path.abspath(__file__))

# 需要添加的路径初始化代码
PATH_INIT_CODE = """import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

"""

def should_skip_file(filepath):
    """检查是否应该跳过该文件"""
    skip_patterns = [
        '__init__.py',
        'config.py',
        'fix_imports.py'
    ]
    return any(pattern in filepath for pattern in skip_patterns)

def already_has_path_init(content):
    """检查文件是否已经有路径初始化代码"""
    return 'sys.path.insert' in content

def add_path_init(filepath):
    """为 Python 文件添加路径初始化代码"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 如果已经有路径初始化，跳过
    if already_has_path_init(content):
        return False
    
    # 查找第一个 from steve1. 导入
    if 'from steve1.' not in content:
        return False
    
    # 找到文件开头的 import 部分
    lines = content.split('\n')
    insert_pos = 0
    
    # 找到第一个非注释、非空行的导入语句之前
    in_docstring = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 处理文档字符串
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        
        if in_docstring:
            continue
        
        # 找到第一个 import 或 from 语句
        if stripped.startswith('import ') or stripped.startswith('from '):
            insert_pos = i
            break
    
    # 插入路径初始化代码
    lines.insert(insert_pos, PATH_INIT_CODE)
    new_content = '\n'.join(lines)
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    """主函数"""
    modified_files = []
    skipped_files = []
    
    # 遍历所有 .py 文件
    for root, dirs, files in os.walk(STEVE1_DIR):
        for filename in files:
            if not filename.endswith('.py'):
                continue
            
            filepath = os.path.join(root, filename)
            
            if should_skip_file(filepath):
                skipped_files.append(filepath)
                continue
            
            if add_path_init(filepath):
                modified_files.append(filepath)
                print(f"✅ Modified: {filepath}")
            else:
                print(f"⏭️  Skipped: {filepath}")
    
    print(f"\n{'='*60}")
    print(f"✅ Modified {len(modified_files)} files")
    print(f"⏭️  Skipped {len(skipped_files)} files")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
```

### **运行修改脚本**

```bash
cd /Users/nanzhang/aimc/src/training/steve1
python fix_imports.py
```

---

## 📂 权重文件组织

确保权重文件在正确的位置：

```
/Users/nanzhang/aimc/data/
├── weights/
│   ├── mineclip/
│   │   └── attn.pth                    ✅ 已存在
│   ├── steve1/
│   │   ├── steve1.weights              ❓ 需要下载
│   │   └── steve1_prior.pt             ✅ 已存在
│   └── vpt/
│       ├── 2x.model                    ✅ 已存在
│       └── rl-from-foundation-2x.weights ✅ 已存在
├── visual_prompt_embeds/               ✅ 已存在
│   ├── dig.pkl
│   ├── dirt.pkl
│   └── ...
└── generated_videos/                   ⏱️ 运行时生成
    └── ...
```

### **缺失的权重文件**

如果缺少 `steve1.weights`，您需要：

1. **从 STEVE-1 官方下载**：
```bash
cd /Users/nanzhang/aimc/data/weights/steve1
# 使用官方的 download_weights.sh 脚本中的链接
# 或从项目主页下载
```

2. **或使用训练脚本训练**：
```bash
cd /Users/nanzhang/aimc/src/training/steve1
./3_train.sh
```

---

## 🚀 使用方法

### **1. 生成论文演示视频**

```bash
cd /Users/nanzhang/aimc/src/training/steve1
chmod +x 1_gen_paper_videos.sh
./1_gen_paper_videos.sh
```

输出位置：`/Users/nanzhang/aimc/data/generated_videos/paper_prompts/`

### **2. 生成自定义文本提示视频**

```bash
cd /Users/nanzhang/aimc/src/training/steve1
chmod +x 2_gen_vid_for_text_prompt.sh

# 编辑脚本修改文本提示
vim 2_gen_vid_for_text_prompt.sh
# 修改最后一行的 --custom_text_prompt "look at the sky"

./2_gen_vid_for_text_prompt.sh
```

### **3. 交互式会话**

```bash
cd /Users/nanzhang/aimc/src/training/steve1
chmod +x 3_run_interactive_session.sh
./3_run_interactive_session.sh
```

⚠️ **注意**：需要图形界面，不支持 headless 模式。

### **4. 直接使用 Python**

```python
import os
import sys

# 添加路径
sys.path.insert(0, '/Users/nanzhang/aimc/src/training')

from steve1.run_agent.run_agent import run_agent
from steve1.config import DATA_DIR
import os

# 运行 Agent
prompt_embed = ...  # 从 MineCLIP 获取
run_agent(
    prompt_embed=prompt_embed,
    gameplay_length=1000,
    save_video_filepath=os.path.join(DATA_DIR, 'generated_videos/test.mp4'),
    in_model=os.path.join(DATA_DIR, 'weights/vpt/2x.model'),
    in_weights=os.path.join(DATA_DIR, 'weights/steve1/steve1.weights'),
    seed=None,
    cond_scale=6.0
)
```

---

## 🔧 故障排查

### **问题 1: ModuleNotFoundError: No module named 'steve1'**

**原因**：Python 无法找到 steve1 模块。

**解决方案**：

方案 A - 设置 PYTHONPATH（推荐）：
```bash
export PYTHONPATH="/Users/nanzhang/aimc/src/training:$PYTHONPATH"
```

将上述命令添加到 `~/.bashrc` 或 `~/.zshrc`：
```bash
echo 'export PYTHONPATH="/Users/nanzhang/aimc/src/training:$PYTHONPATH"' >> ~/.zshrc
source ~/.zshrc
```

方案 B - 在脚本中添加路径：
```bash
# 在每个脚本开头添加
export PYTHONPATH="/Users/nanzhang/aimc/src/training:$PYTHONPATH"
```

方案 C - 使用 Python -m 运行：
```bash
cd /Users/nanzhang/aimc/src/training
python -m steve1.run_agent.run_agent --help
```

### **问题 2: FileNotFoundError: 权重文件不存在**

**检查文件是否存在**：
```bash
ls -lh /Users/nanzhang/aimc/data/weights/steve1/
ls -lh /Users/nanzhang/aimc/data/weights/vpt/
ls -lh /Users/nanzhang/aimc/data/weights/mineclip/
```

**缺失文件**：参考上面的"缺失的权重文件"部分。

### **问题 3: 路径仍然不正确**

**手动检查并修复**：
```bash
# 检查 config.py 的路径
python -c "from steve1.config import DATA_DIR; print(DATA_DIR)"
# 应该输出：/Users/nanzhang/aimc/data

# 检查文件存在性
python -c "from steve1.config import MINECLIP_CONFIG; print(MINECLIP_CONFIG['ckpt']['path'])"
```

---

## 📊 完整的文件修改清单

### **已修改**：
- ✅ `config.py` - 路径配置
- ✅ `__init__.py` - 包初始化
- ✅ `1_gen_paper_videos.sh` - Shell 脚本
- ✅ `2_gen_vid_for_text_prompt.sh` - Shell 脚本
- ✅ `3_run_interactive_session.sh` - Shell 脚本
- ✅ `1_generate_dataset.sh` - Shell 脚本
- ✅ `3_train.sh` - Shell 脚本
- ✅ `run_agent/run_agent.py` - Python 导入
- ✅ `run_agent/paper_prompts.py` - Python 导入
- ✅ `utils/mineclip_agent_env_utils.py` - Python 导入

### **需手动修改** (使用 `fix_imports.py`):
- ⏳ `utils/text_overlay_utils.py`
- ⏳ `utils/embed_utils.py`
- ⏳ `training/train.py`
- ⏳ `run_agent/run_interactive.py`
- ⏳ `helpers.py`
- ⏳ `embed_conditioned_policy.py`
- ⏳ `MineRLConditionalAgent.py`
- ⏳ VPT 目录下的所有文件 (17个)

---

## 🎯 快速启动检查清单

### **步骤 1: 验证路径设置** ✅
```bash
cd /Users/nanzhang/aimc/src/training/steve1
python -c "from steve1.config import DATA_DIR, PROJECT_ROOT; print(f'PROJECT_ROOT: {PROJECT_ROOT}'); print(f'DATA_DIR: {DATA_DIR}')"
```

预期输出：
```
PROJECT_ROOT: /Users/nanzhang/aimc
DATA_DIR: /Users/nanzhang/aimc/data
```

### **步骤 2: 验证权重文件** ✅
```bash
ls -lh /Users/nanzhang/aimc/data/weights/mineclip/attn.pth
ls -lh /Users/nanzhang/aimc/data/weights/steve1/steve1_prior.pt
ls -lh /Users/nanzhang/aimc/data/weights/vpt/2x.model
```

### **步骤 3: 运行批量修改脚本** ⏳
```bash
cd /Users/nanzhang/aimc/src/training/steve1
# 将上面的 fix_imports.py 保存到这个目录
python fix_imports.py
```

### **步骤 4: 测试运行** ⏳
```bash
cd /Users/nanzhang/aimc/src/training/steve1
./2_gen_vid_for_text_prompt.sh
```

---

## 💡 与 AIMC 项目集成

### **方式 1: 作为独立模块使用**

```python
# 在您的 AIMC 代码中
import sys
sys.path.insert(0, '/Users/nanzhang/aimc/src/training')

from steve1.run_agent.run_agent import run_agent
# 使用 STEVE-1
```

### **方式 2: 比较评估**

```python
# src/training/compare_steve1_aimc.py
from steve1.run_agent.run_agent import run_agent
from dagger.evaluate_policy import evaluate_policy

# 对比 STEVE-1 和您的方法
steve1_results = test_steve1(task="harvest_1_log")
aimc_results = test_aimc(task="harvest_1_log")

print(f"STEVE-1: {steve1_results}")
print(f"AIMC:    {aimc_results}")
```

### **方式 3: 学习其方法**

参考 STEVE-1 的：
- 事后重标记（Hindsight Relabeling）
- CVAE Prior 训练
- Classifier-Free Guidance

---

## 📚 相关文档

- `docs/reference/STEVE1_MODEL_DOWNLOAD_GUIDE.md` - STEVE-1 模型下载指南
- `docs/guides/STEVE1_EVALUATION_GUIDE.md` - STEVE-1 评估使用指南
- `docs/technical/MINECLIP_INSTRUCTION_DRIVEN_AGENT.md` - MineCLIP 指令驱动原理

---

## ✅ 总结

### **已完成**：
1. ✅ 修改 `config.py` 指向项目 data 目录
2. ✅ 修改所有 Shell 脚本使用动态路径
3. ✅ 为核心 Python 文件添加路径初始化
4. ✅ 创建包初始化文件
5. ✅ 创建批量修改脚本

### **下一步**：
1. ⏳ 运行 `fix_imports.py` 修改剩余文件
2. ⏳ 验证权重文件完整性
3. ⏳ 测试运行脚本
4. ⏳ 集成到 AIMC 工作流程

---

**所有修改的目标**：让 STEVE-1 能在 `/Users/nanzhang/aimc/src/training/steve1/` 位置正常运行，并使用 `/Users/nanzhang/aimc/data/` 目录存放所有数据和权重。

