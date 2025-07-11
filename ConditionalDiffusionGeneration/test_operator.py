#!/usr/bin/env python3
"""
测试脚本：验证operator注册和错误处理
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from guided_diffusion.measurements import get_operator, __OPERATOR__

def test_operator_registration():
    """测试operator注册情况"""
    print("=== 测试Operator注册 ===")
    print(f"已注册的operators: {list(__OPERATOR__.keys())}")
    print()

def test_get_operator_with_invalid_name():
    """测试使用无效名称调用get_operator"""
    print("=== 测试无效Operator名称 ===")
    try:
        operator = get_operator(name='invalid_operator_name')
    except NameError as e:
        print(f"预期的错误: {e}")
    print()

def test_get_operator_with_wrong_params():
    """测试使用错误参数调用get_operator"""
    print("=== 测试错误参数 ===")
    try:
        # 尝试使用错误的参数调用inflow_prediction
        operator = get_operator(name='inflow_prediction', wrong_param='value')
    except TypeError as e:
        print(f"预期的错误: {e}")
    print()

def test_get_operator_correct_usage():
    """测试正确的get_operator使用方式"""
    print("=== 测试正确使用方式 ===")
    print("正确的调用方式应该是:")
    print("operator = get_operator(name='inflow_prediction', device=device, ...)")
    print("而不是:")
    print("operator = get_operator(device=device, name='inflow_prediction', ...)")
    print()

if __name__ == "__main__":
    test_operator_registration()
    test_get_operator_with_invalid_name()
    test_get_operator_with_wrong_params()
    test_get_operator_correct_usage() 