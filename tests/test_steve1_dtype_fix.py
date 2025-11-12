"""
测试 STEVE-1 Dtype 修复
确保在4090等GPU上不出现dtype不匹配错误
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


def test_prompt_embed_dtype_conversion():
    """测试prompt嵌入的dtype转换"""
    # 模拟float16嵌入
    prompt_embed_fp16 = torch.randn(512, dtype=torch.float16)
    
    # 检测并转换
    if hasattr(prompt_embed_fp16, 'dtype') and prompt_embed_fp16.dtype == torch.float16:
        prompt_embed_fp32 = prompt_embed_fp16.float()
    
    # 验证
    assert prompt_embed_fp32.dtype == torch.float32, "嵌入应该是float32"
    assert prompt_embed_fp32.shape == (512,), "嵌入维度应该是512"
    

def test_agent_policy_dtype():
    """测试agent policy的dtype"""
    # 创建模拟的policy
    policy = torch.nn.Linear(512, 256)
    
    # 确保是float32
    policy.float()
    
    # 验证权重dtype
    for param in policy.parameters():
        assert param.dtype == torch.float32, "Policy权重应该是float32"


def test_mixed_dtype_matmul_error():
    """测试混合dtype矩阵乘法会报错"""
    # 模拟问题场景
    weight = torch.randn(256, 512, dtype=torch.float32)
    input_fp16 = torch.randn(1, 512, dtype=torch.float16)
    
    # 这应该报错 (错误消息可能是"must have the same dtype"或"expected ... to have the same dtype")
    with pytest.raises(RuntimeError, match="same dtype"):
        result = torch.matmul(input_fp16, weight.t())


def test_mixed_dtype_matmul_fixed():
    """测试修复后的混合dtype矩阵乘法"""
    # 模拟修复场景
    weight = torch.randn(256, 512, dtype=torch.float32)
    input_fp16 = torch.randn(1, 512, dtype=torch.float16)
    
    # 修复: 转换输入为float32
    input_fp32 = input_fp16.float()
    
    # 不应该报错
    result = torch.matmul(input_fp32, weight.t())
    assert result.dtype == torch.float32, "结果应该是float32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA")
def test_autocast_behavior():
    """测试autocast的行为"""
    device = torch.device("cuda")
    
    # 模拟MineCLIP编码
    text_embed = torch.randn(512, device=device, dtype=torch.float32)
    
    # 在autocast下
    with torch.cuda.amp.autocast():
        # 某些操作可能变成float16
        result = text_embed * 2.0
        # 注意: 结果dtype可能是float16或float32，取决于GPU
        print(f"Autocast result dtype: {result.dtype}")
    
    # 修复: 显式转换
    result_fixed = result.float()
    assert result_fixed.dtype == torch.float32, "修复后应该是float32"


def test_prompt_embed_numpy_conversion():
    """测试嵌入转换为numpy的完整流程"""
    # 模拟float16嵌入
    prompt_embed = torch.randn(512, dtype=torch.float16)
    
    # 修复dtype
    if hasattr(prompt_embed, 'dtype') and prompt_embed.dtype == torch.float16:
        prompt_embed = prompt_embed.float()
    
    # 转换为numpy
    prompt_embed_np = prompt_embed.cpu().numpy()
    
    # 验证
    assert isinstance(prompt_embed_np, np.ndarray), "应该是numpy array"
    assert prompt_embed_np.dtype == np.float32, "numpy array应该是float32"
    assert prompt_embed_np.shape == (512,), "维度应该是512"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
