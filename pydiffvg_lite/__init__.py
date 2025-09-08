"""
PyDiffVG Lite - Pure Python subset for VAE training
只包含SVG解析和数据结构，不包含可微分渲染
"""

from .parse_svg import svg_to_scene
from .shape import Path, Circle, Ellipse, Polygon, Rect, ShapeGroup, from_svg_path
from .save_svg import save_svg
from .color import LinearGradient, RadialGradient
from .device import set_print_timing, get_device, get_use_gpu

# 简单的兼容性函数
def set_use_gpu(use_gpu):
    """兼容性函数，VAE训练不需要GPU渲染"""
    pass

# 导入渲染功能
from .render import RenderFunction, svg_to_png, png_to_tensor, svg_to_tensor

__all__ = [
    'svg_to_scene', 'Path', 'Circle', 'Ellipse', 'Polygon', 'Rect', 'ShapeGroup',
    'from_svg_path', 'save_svg', 'LinearGradient', 'RadialGradient',
    'set_print_timing', 'get_device', 'get_use_gpu', 'set_use_gpu', 'RenderFunction',
    'svg_to_png', 'png_to_tensor', 'svg_to_tensor'
]
