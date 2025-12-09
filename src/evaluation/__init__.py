"""
AIMC Agent 评估框架
AIMC Agent Evaluation Framework
"""

# 只导入存在的模块，避免导入错误
try:
    from .task_loader import TaskLoader
except ImportError:
    TaskLoader = None

try:
    from .metrics import EvaluationMetrics
except ImportError:
    EvaluationMetrics = None

try:
    from .eval_framework import EvaluationFramework
except ImportError:
    EvaluationFramework = None

try:
    from .policy_evaluator import STEVE1Evaluator
    PolicyEvaluator = STEVE1Evaluator  # 别名
except ImportError:
    STEVE1Evaluator = None
    PolicyEvaluator = None

try:
    from .prior_evaluator import Steve1PriorEvaluator
    PriorEvaluator = Steve1PriorEvaluator  # 别名
except ImportError:
    Steve1PriorEvaluator = None
    PriorEvaluator = None

try:
    from .vpt_evaluator import VPTEvaluator
except ImportError:
    VPTEvaluator = None

__all__ = [
    'TaskLoader',
    'EvaluationMetrics',
    'EvaluationFramework',
    'STEVE1Evaluator',
    'PolicyEvaluator',  # 别名
    'Steve1PriorEvaluator',
    'PriorEvaluator',  # 别名
    'VPTEvaluator',
]

