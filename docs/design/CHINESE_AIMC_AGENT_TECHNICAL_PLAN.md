# 中文AIMC Agent技术方案

> **目标**: 训练支持中文指令的Minecraft Agent  
> **关键需求**: 
> 1. 支持中文语义理解（"砍树" = "chop tree"）
> 2. MineDojo任务评估基线
> 3. 中英文等价指令成功率对比
>
> **设计日期**: 2025-11-05

---

## 🎯 核心问题澄清

### 你的理解 vs 实际情况

```
你的理解 ❓:
  "需要重复STEVE-1训练过程，增加中文CLIP"
  → 理解为：完全重新训练STEVE-1

实际情况 ✅:
  STEVE-1模型本身不需要重新训练！
  只需要解决"中文→MineCLIP嵌入"的问题

关键洞察:
  STEVE-1的核心是 Goal-Conditioned策略
  它接收的是MineCLIP嵌入（512维向量）
  不关心嵌入来自英文还是中文
  
  问题只在于: 如何让中文指令也能产生合适的嵌入
```

### 为什么不需要重新训练STEVE-1？

```
STEVE-1的工作流程:

训练时:
  1. 人类录像 → MineCLIP编码 → 视觉嵌入
  2. 训练: (画面, 视觉嵌入, 动作)
  3. 输出: Goal-Conditioned策略

推理时（英文）:
  文本"chop tree" → MineCLIP.encode_text() → 嵌入(512维)
                                                ↓
  画面 + 嵌入 → STEVE-1策略 → 动作

推理时（中文）- 我们要做的:
  文本"砍树" → [某种方法] → 嵌入(512维)  ← 关键！
                                     ↓
  画面 + 嵌入 → STEVE-1策略 → 动作  ← 已有，不用改！
              (相同的模型)

结论:
  ✅ STEVE-1模型可以直接使用
  ✅ 只需要解决: "砍树" → 512维嵌入
  ❌ 不需要重新训练整个STEVE-1
```

---

## 📋 技术方案对比

### 方案A: 翻译桥接 ⚡ (推荐第一阶段)

**原理**: 中文 → 英文 → MineCLIP

```python
# 最简单、最快的方案
chinese_text = "砍树"
english_text = translate_zh_to_en(chinese_text)  # "chop tree"
embed = mineclip.encode_text(english_text)
action = steve1(obs, embed)
```

**优点**:
- ✅ 无需训练，立即可用
- ✅ 0额外成本
- ✅ 可以快速验证整体流程
- ✅ 适合快速原型和基线评估

**缺点**:
- ❌ 依赖翻译质量
- ❌ Minecraft术语可能翻译不准
- ❌ 增加推理延迟（~100ms）

**实现难度**: ⭐ (1-2天)

**适用场景**: 
- 快速验证idea
- 建立评估基线
- 第一阶段原型

---

### 方案B: 多语言MineCLIP适配 ⭐ (推荐生产环境)

**原理**: 训练对齐层，映射中文嵌入到MineCLIP空间

```
架构:
  Chinese-CLIP (预训练) → 对齐层(需训练) → MineCLIP空间
                                              ↓
                            画面 + 嵌入 → STEVE-1策略

训练对齐层:
  数据: 中英文对照pairs ("砍树", "chop tree")
  目标: 让中英文嵌入在MineCLIP空间接近
  
  loss = ||Align(Chinese-CLIP("砍树")) - MineCLIP("chop tree")||²
```

**优点**:
- ✅ 不依赖翻译
- ✅ 直接中文理解
- ✅ 性能接近原生MineCLIP
- ✅ STEVE-1完全不用改

**缺点**:
- ❌ 需要收集中英文对照数据（1000-5000对）
- ❌ 需要训练对齐层（1-3天GPU时间）
- ❌ 需要Chinese-CLIP模型

**实现难度**: ⭐⭐⭐ (1-2周)

**适用场景**:
- 生产环境部署
- 追求最佳性能
- 有一定资源投入

---

### 方案C: 从头训练Chinese-MineCLIP ❌ (不推荐)

**原理**: 在中文Minecraft视频上训练新的MineCLIP

```
需要:
  1. 收集10万+中文Minecraft视频（B站、抖音）
  2. 人工标注视频-文本对
  3. 训练Chinese-MineCLIP（多GPU，数周）
  4. 重新训练STEVE-1（用新的Chinese-MineCLIP）
```

**优点**:
- ✅ 原生中文支持，理论最优

**缺点**:
- ❌ 成本极高（数十万元）
- ❌ 时间极长（数月）
- ❌ 需要大量数据标注
- ❌ 需要重新训练STEVE-1

**实现难度**: ⭐⭐⭐⭐⭐ (数月)

**适用场景**:
- 大型研究项目
- 有充足预算和时间
- 追求最优理论性能

---

## 🎯 推荐实施路径

### 阶段1: 快速验证 (1-2周)

**目标**: 验证中文指令可行性，建立评估基线

```
步骤1: 实现翻译桥接 (2-3天)
  ├─ 集成翻译API（百度/腾讯/OpenAI）
  ├─ 实现中文→MineCLIP pipeline
  └─ 测试基本功能

步骤2: 建立评估基线 (3-5天)
  ├─ 设计MineDojo任务集
  ├─ 实现评估框架
  ├─ 收集中英文等价指令对
  └─ 运行基线评估

步骤3: 分析和优化翻译 (2-3天)
  ├─ 分析翻译错误
  ├─ 建立Minecraft术语词典
  └─ 优化翻译质量

输出:
  ✅ 可工作的中文AIMC原型
  ✅ 评估代码和基线数据
  ✅ 性能报告
```

### 阶段2: 性能优化 (2-4周)

**目标**: 提升中文理解质量

```
步骤1: 数据准备 (1周)
  ├─ 收集中英文指令对（1000-3000对）
  ├─ 覆盖所有MineDojo任务类型
  └─ 人工校验质量

步骤2: 训练对齐层 (1-2周)
  ├─ 实现多语言MineCLIP适配
  ├─ 训练对齐层
  ├─ 验证和调优
  └─ 评估性能提升

步骤3: 系统集成 (3-5天)
  ├─ 集成到AIMC系统
  ├─ 性能测试
  └─ 文档和部署

输出:
  ✅ 高质量中文AIMC Agent
  ✅ 性能对比报告
  ✅ 部署文档
```

### 阶段3: 持续改进 (持续)

**目标**: 根据使用反馈持续优化

```
数据积累:
  ├─ 收集用户中文指令
  ├─ 分析失败case
  └─ 扩充训练数据

模型优化:
  ├─ 微调对齐层
  ├─ 优化翻译规则
  └─ 性能监控
```

---

## 🔧 详细技术方案

### 1. 翻译桥接实现方案（阶段1）

#### 1.1 架构设计

```python
# 系统架构
class ChineseAIMCAgent:
    def __init__(self):
        self.translator = ChineseTranslator()      # 翻译模块
        self.mineclip = load_mineclip()           # MineCLIP
        self.steve1 = load_steve1()               # STEVE-1策略
        self.term_dict = load_mc_dictionary()     # Minecraft术语词典
    
    def execute_chinese_command(self, chinese_text):
        """执行中文指令"""
        # 1. 翻译 + 术语修正
        english_text = self.translate_with_terms(chinese_text)
        
        # 2. 编码
        text_embed = self.mineclip.encode_text(english_text)
        
        # 3. 执行
        while not task_done:
            obs = env.get_obs()
            action = self.steve1(obs, text_embed)
            env.step(action)
```

#### 1.2 翻译模块设计

```python
class ChineseTranslator:
    """中文翻译器（带Minecraft术语优化）"""
    
    def __init__(self):
        # 选择翻译后端
        self.backend = "openai"  # 或 "baidu", "tencent"
        
        # Minecraft术语词典
        self.mc_terms = {
            "砍树": "chop tree",
            "挖矿": "mine",
            "建造": "build",
            "合成": "craft",
            "红石": "redstone",
            "钻石镐": "diamond pickaxe",
            # ... 更多术语
        }
        
        # 缓存（相同指令不重复翻译）
        self.cache = {}
    
    def translate(self, chinese_text):
        """翻译中文到英文"""
        
        # 1. 检查缓存
        if chinese_text in self.cache:
            return self.cache[chinese_text]
        
        # 2. 检查精确匹配术语
        if chinese_text in self.mc_terms:
            return self.mc_terms[chinese_text]
        
        # 3. 检查部分匹配术语
        for zh, en in self.mc_terms.items():
            if zh in chinese_text:
                chinese_text = chinese_text.replace(zh, en)
        
        # 4. 调用翻译API
        english_text = self._call_translate_api(chinese_text)
        
        # 5. 缓存
        self.cache[chinese_text] = english_text
        
        return english_text
    
    def _call_translate_api(self, text):
        """调用翻译API"""
        if self.backend == "openai":
            return self._translate_openai(text)
        elif self.backend == "baidu":
            return self._translate_baidu(text)
        # ... 其他后端
```

#### 1.3 术语词典构建

```python
# data/chinese_mc_terms.json
{
  "basic_actions": {
    "砍树": "chop tree",
    "挖掘": "dig",
    "建造": "build",
    "攻击": "attack",
    "跳跃": "jump",
    "游泳": "swim"
  },
  "materials": {
    "木头": "wood",
    "石头": "stone",
    "铁": "iron",
    "钻石": "diamond",
    "红石": "redstone"
  },
  "tools": {
    "斧头": "axe",
    "镐": "pickaxe",
    "铲": "shovel",
    "剑": "sword"
  },
  "tasks": {
    "找到洞穴": "find cave",
    "猎杀牛": "hunt cow",
    "建造房屋": "build house"
  }
}
```

### 2. 多语言MineCLIP适配方案（阶段2）

#### 2.1 架构设计

```python
class MultilingualMineCLIP:
    """多语言MineCLIP适配器"""
    
    def __init__(self):
        # 原始MineCLIP（英文）
        self.mineclip_en = load_mineclip()
        
        # Chinese-CLIP（中文）
        self.chinese_clip = ChineseCLIP.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16"
        )
        
        # 对齐层（需要训练）
        self.alignment_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
    
    def encode_text(self, text, language='auto'):
        """编码文本（自动检测语言）"""
        
        # 1. 检测语言
        if language == 'auto':
            language = detect_language(text)
        
        # 2. 编码
        if language == 'en':
            # 英文：直接用MineCLIP
            return self.mineclip_en.encode_text(text)
        
        elif language == 'zh':
            # 中文：Chinese-CLIP + 对齐层
            zh_embed = self.chinese_clip.encode_text(text)
            aligned_embed = self.alignment_layer(zh_embed)
            return aligned_embed
        
    def encode_image(self, image):
        """编码图像（保持不变）"""
        return self.mineclip_en.encode_image(image)
```

#### 2.2 对齐层训练

```python
# 训练对齐层
def train_alignment_layer():
    """
    训练目标: 让中文嵌入映射到MineCLIP空间
    """
    
    # 1. 准备数据
    pairs = load_chinese_english_pairs()  # 中英文对照
    # [("砍树", "chop tree"), ("挖矿", "mine"), ...]
    
    # 2. 训练循环
    for epoch in range(epochs):
        for zh_text, en_text in pairs:
            # 编码
            zh_embed = chinese_clip.encode_text(zh_text)
            en_embed = mineclip.encode_text(en_text)  # 目标
            
            # 对齐
            aligned_embed = alignment_layer(zh_embed)
            
            # 损失：L2距离
            loss = F.mse_loss(aligned_embed, en_embed)
            
            # 优化
            loss.backward()
            optimizer.step()
    
    # 3. 验证
    validate_alignment(alignment_layer)
```

#### 2.3 数据收集策略

```python
# 中英文对照数据收集
def collect_chinese_english_pairs():
    """
    策略1: MineDojo任务集翻译
      - 所有MineDojo任务的中文翻译
      - 约200个基础对
    
    策略2: 动作扩展
      - 每个动作的多种中文表述
      - 例如: "砍树" / "伐木" / "获取木头" → "chop tree"
      - 约500-1000对
    
    策略3: 组合指令
      - 复杂任务的中文描述
      - 例如: "找到一棵橡树并砍下它" → "find oak tree and chop it"
      - 约500-1000对
    
    策略4: 社区收集
      - 中文Minecraft论坛/视频的常用表述
      - 约1000-2000对
    
    总计: 2000-3000对（足够训练对齐层）
    ```

---

## 📊 评估框架设计

### 1. 评估任务集设计

#### 1.1 MineDojo基线任务

```python
# 评估任务集配置
EVAL_TASKS = {
    "basic": [
        {
            "id": "chop_tree",
            "en": "chop tree",
            "zh": ["砍树", "伐木", "获取木头"],
            "difficulty": "easy",
            "success_metric": "has_log_in_inventory",
            "time_limit": 300  # 秒
        },
        {
            "id": "hunt_cow",
            "en": "hunt cow",
            "zh": ["猎牛", "杀牛", "获得牛肉"],
            "difficulty": "easy",
            "success_metric": "has_beef_in_inventory",
            "time_limit": 300
        },
        # ... 更多基础任务
    ],
    
    "medium": [
        {
            "id": "find_cave",
            "en": "find cave",
            "zh": ["找到洞穴", "寻找洞穴", "进入洞穴"],
            "difficulty": "medium",
            "success_metric": "in_cave_biome",
            "time_limit": 600
        },
        # ... 更多中等任务
    ],
    
    "hard": [
        {
            "id": "build_house",
            "en": "build a house",
            "zh": ["建造房屋", "盖房子", "搭建小屋"],
            "difficulty": "hard",
            "success_metric": "placed_blocks_count > 50",
            "time_limit": 1200
        },
        # ... 更多困难任务
    ]
}
```

#### 1.2 评估维度

```python
class EvaluationMetrics:
    """评估指标"""
    
    @staticmethod
    def task_success(trajectory, task):
        """任务成功率（主要指标）"""
        return check_task_completion(trajectory, task)
    
    @staticmethod
    def language_equivalence(zh_success_rate, en_success_rate):
        """语言等价性（关键指标）"""
        # 中英文成功率应该接近
        return abs(zh_success_rate - en_success_rate)
    
    @staticmethod
    def efficiency(steps, expert_steps):
        """效率"""
        return expert_steps / steps
    
    @staticmethod
    def semantic_variations(results_per_variation):
        """语义变体鲁棒性"""
        # 不同中文表述的成功率方差
        return np.std(results_per_variation)
```

### 2. 评估代码框架

```python
class ChineseAIMCEvaluator:
    """中文AIMC评估器"""
    
    def __init__(self, agent, tasks):
        self.agent = agent
        self.tasks = tasks
    
    def evaluate_all(self):
        """完整评估"""
        results = {
            "basic": self.evaluate_category("basic"),
            "medium": self.evaluate_category("medium"),
            "hard": self.evaluate_category("hard")
        }
        
        # 生成报告
        self.generate_report(results)
        
        return results
    
    def evaluate_category(self, category):
        """评估单个类别"""
        tasks = self.tasks[category]
        results = []
        
        for task in tasks:
            # 英文baseline
            en_result = self.evaluate_task(task, language='en')
            
            # 中文变体
            zh_results = []
            for zh_text in task['zh']:
                result = self.evaluate_task(task, language='zh', text=zh_text)
                zh_results.append(result)
            
            # 汇总
            results.append({
                'task_id': task['id'],
                'en_success_rate': en_result['success_rate'],
                'zh_success_rates': [r['success_rate'] for r in zh_results],
                'zh_avg_success_rate': np.mean([r['success_rate'] for r in zh_results]),
                'equivalence_gap': abs(en_result['success_rate'] - 
                                      np.mean([r['success_rate'] for r in zh_results])),
                'semantic_variance': np.std([r['success_rate'] for r in zh_results])
            })
        
        return results
    
    def evaluate_task(self, task, language, text=None, n_trials=10):
        """评估单个任务"""
        
        if text is None:
            text = task['en'] if language == 'en' else task['zh'][0]
        
        successes = 0
        steps_list = []
        
        for trial in range(n_trials):
            # 运行
            trajectory = self.run_episode(text, task['time_limit'])
            
            # 检查成功
            success = self.check_success(trajectory, task)
            if success:
                successes += 1
            
            steps_list.append(len(trajectory))
        
        return {
            'success_rate': successes / n_trials,
            'avg_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list)
        }
    
    def generate_report(self, results):
        """生成评估报告"""
        
        report = {
            "summary": {
                "overall_en_success": np.mean([...]),
                "overall_zh_success": np.mean([...]),
                "equivalence_gap": np.mean([...]),
                "semantic_robustness": np.mean([...])
            },
            "by_category": results,
            "detailed_analysis": self.analyze_results(results)
        }
        
        # 保存
        save_json(report, "evaluation_results.json")
        
        # 可视化
        self.plot_results(report)
```

### 3. 评估报告格式

```python
# 评估报告示例
{
  "summary": {
    "overall_en_success_rate": 0.82,
    "overall_zh_success_rate": 0.78,
    "equivalence_gap": 0.04,        # 目标: <0.10
    "semantic_robustness": 0.03     # 目标: <0.05
  },
  
  "by_task": [
    {
      "task_id": "chop_tree",
      "en_success_rate": 0.92,
      "zh_variants": [
        {"text": "砍树", "success_rate": 0.90},
        {"text": "伐木", "success_rate": 0.88},
        {"text": "获取木头", "success_rate": 0.85}
      ],
      "zh_avg": 0.88,
      "gap": 0.04,
      "analysis": "中文性能略低4%，可能是'获取木头'翻译不够精确"
    },
    // ... 更多任务
  ],
  
  "failure_analysis": {
    "translation_errors": [
      {"zh": "红石电路", "en_translated": "red stone circuit", 
       "expected": "redstone circuit", "impact": "high"}
    ],
    "semantic_mismatches": [...]
  }
}
```

---

## 📅 实施时间表

### 第1周：环境准备和翻译实现
```
Day 1-2: 环境配置
  - 安装依赖
  - 配置STEVE-1
  - 测试英文baseline

Day 3-4: 翻译模块
  - 实现翻译API集成
  - 构建术语词典
  - 测试基本翻译

Day 5-7: 初步集成
  - 中文AIMC Agent实现
  - 简单任务测试
  - 问题排查
```

### 第2周：评估框架开发
```
Day 8-10: 评估代码
  - 实现评估框架
  - 配置MineDojo任务
  - 实现评估指标

Day 11-12: 数据收集
  - 收集中英文指令对
  - 构建测试集
  - 人工验证

Day 13-14: 基线评估
  - 运行完整评估
  - 生成报告
  - 分析结果
```

### 第3-4周：优化和完善（可选）
```
Day 15-18: 翻译优化
  - 分析翻译错误
  - 优化术语词典
  - 重新评估

Day 19-21: 数据准备（如果进入阶段2）
  - 收集更多中英文对
  - 准备对齐层训练

Day 22-28: 模型训练（如果进入阶段2）
  - 训练对齐层
  - 验证和调优
  - 对比评估
```

---

## 💡 关键决策点

### 决策1: 方案选择

**推荐**: 
- 第一阶段用翻译桥接
- 根据评估结果决定是否进入阶段2

**判断标准**:
```
翻译方案足够好的标准:
  ✅ 中英文成功率gap < 10%
  ✅ 语义变体方差 < 5%
  ✅ 关键任务成功率 > 70%

如果满足 → 继续优化翻译即可
如果不满足 → 进入阶段2（训练对齐层）
```

### 决策2: 数据量

```
阶段1（翻译）:
  术语词典: 200-500个术语 ✅
  测试指令: 20-50个任务 × 3个变体 ✅
  
阶段2（对齐层）:
  训练数据: 2000-3000个中英文对 ✅
  验证数据: 500个对 ✅
```

### 决策3: 评估频率

```
开发阶段: 每次修改后快速测试（5个任务）
基线评估: 完整评估（所有任务，每个10次trial）
优化评估: 每次优化后完整评估
最终评估: 正式评估（每个任务30次trial）
```

---

## 🎯 成功标准

### 阶段1目标（翻译方案）

```
必达指标:
  ✅ 系统可运行（中文指令→执行）
  ✅ 基线评估完成
  ✅ 中英文gap < 15%

期望指标:
  ⭐ 中英文gap < 10%
  ⭐ 基础任务成功率 > 75%
  ⭐ 语义变体方差 < 5%
```

### 阶段2目标（对齐层方案）

```
必达指标:
  ✅ 中英文gap < 10%
  ✅ 基础任务成功率 > 80%

期望指标:
  ⭐ 中英文gap < 5%
  ⭐ 全部任务平均成功率 > 75%
  ⭐ 推理速度 < 100ms
```

---

## 📚 参考资源

**代码参考**:
- STEVE-1代码: `src/training/steve1/`
- MineCLIP: `src/training/steve1/mineclip_code/`
- 评估代码: `docs/guides/STEVE1_EVALUATION_GUIDE.md`

**文档参考**:
- 中文支持方案: `docs/guides/STEVE1_ADVANCED_SOLUTIONS.md`
- MineCLIP原理: `docs/guides/STEVE1_TRAINING_EXPLAINED.md`
- 评估方法: `docs/technical/SEQUENTIAL_POLICY_EVALUATION.md`

**外部资源**:
- Chinese-CLIP: https://github.com/OFA-Sys/Chinese-CLIP
- 翻译API: 百度翻译/腾讯翻译/OpenAI

---

## ❓ 常见问题

**Q: 一定要重新训练STEVE-1吗？**
A: 不需要！STEVE-1只看512维嵌入，不关心来源。只需解决中文→嵌入问题。

**Q: 翻译方案够好吗？**
A: 对大部分任务够用。如果gap>10%，再考虑训练对齐层。

**Q: 需要多少中文数据？**
A: 阶段1: 200-500术语即可；阶段2: 2000-3000对。

**Q: 能不能跳过阶段1？**
A: 不建议。翻译方案可能够用，先验证再决定是否投入更多资源。

**Q: 多久能看到效果？**
A: 翻译方案1-2周可以看到初步结果。

---

**方案版本**: v1.0  
**设计日期**: 2025-11-05  
**下一步**: 等待确认后开始实施

