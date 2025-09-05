"""
PyDiffVG Lite - 单文件版本
直接替代 import pydiffvg，保持API完全兼容
"""

# 从子模块导入所有功能
from pydiffvg_lite.parse_svg import svg_to_scene
from pydiffvg_lite.shape import Path, Circle, Ellipse, Polygon, Rect, ShapeGroup, from_svg_path
from pydiffvg_lite.save_svg import save_svg
from pydiffvg_lite.color import LinearGradient, RadialGradient
from pydiffvg_lite.device import set_print_timing, get_device, get_use_gpu, set_use_gpu

# 可微分渲染的fallback
class RenderFunction:
    @staticmethod
    def apply(width, height, *args):
        """Fallback渲染函数，返回dummy tensor用于VAE训练"""
        import torch
        # 返回白色背景的dummy图像 (RGBA格式)
        return torch.ones(height, width, 4)
    
    @staticmethod
    def serialize_scene(width, height, shapes, shape_groups):
        """Fallback序列化函数，返回dummy scene args"""
        return []

# 确保所有API都可用
__all__ = [
    'svg_to_scene', 'Path', 'Circle', 'Ellipse', 'Polygon', 'Rect', 'ShapeGroup',
    'from_svg_path', 'save_svg', 'LinearGradient', 'RadialGradient',
    'set_print_timing', 'get_device', 'get_use_gpu', 'set_use_gpu', 'RenderFunction'
]
