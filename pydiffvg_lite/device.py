"""
设备和配置管理 - 简化版本
"""
import torch

# 全局设置
_use_gpu = False
_device = torch.device('cpu')
_print_timing = False

def set_use_gpu(use_gpu):
    """设置是否使用GPU - VAE训练通常在CPU上进行"""
    global _use_gpu
    _use_gpu = use_gpu

def get_use_gpu():
    """获取GPU使用状态"""
    return _use_gpu

def set_device(device):
    """设置设备"""
    global _device
    if isinstance(device, str):
        _device = torch.device(device)
    else:
        _device = device

def get_device():
    """获取当前设备"""
    return _device

def set_print_timing(print_timing):
    """设置是否打印时间信息"""
    global _print_timing
    _print_timing = print_timing

def get_print_timing():
    """获取时间打印设置"""
    return _print_timing
