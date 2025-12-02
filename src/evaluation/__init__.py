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
    from .steve1_evaluator import STEVE1Evaluator
except ImportError:
    STEVE1Evaluator = None

try:
    from .vpt_evaluator import VPTEvaluator
except ImportError:
    VPTEvaluator = None

__all__ = [
    'TaskLoader',
    'EvaluationMetrics',
    'EvaluationFramework',
    'STEVE1Evaluator',
    'VPTEvaluator',
]

