# PyDiffVG Lite

**纯Python版本的PyDiffVG子集，专为VAE训练设计**

## 🎯 设计目标

- ✅ **纯Python** - 无需C++编译，可直接在Google Colab运行
- ✅ **轻量级** - 只包含VAE训练需要的核心功能
- ✅ **兼容性** - 与原版PyDiffVG API保持兼容

## 📦 包含的模块

### 核心功能
- **`svg_to_scene()`** - SVG文件解析
- **`Path`, `Circle`, `Polygon`等** - 几何形状数据结构
- **`save_svg()`** - SVG文件保存
- **`LinearGradient`, `RadialGradient`** - 颜色渐变

### Fallback功能
- **`RenderFunction`** - 返回dummy tensor，VAE训练时不需要实际渲染
- **设备管理** - 简化的CPU/GPU设置

## 🚀 使用方法

```python
# 自动fallback到pydiffvg_lite
import pydiffvg_lite as pydiffvg

# 解析SVG文件
canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene("file.svg")

# 获取路径点数据（用于VAE训练）
for path in shapes:
    points = path.points
    num_control_points = path.num_control_points
    break
```

## 🔍 与原版的区别

| 功能 | 原版PyDiffVG | PyDiffVG Lite |
|------|-------------|---------------|
| SVG解析 | ✅ C++实现 | ✅ 纯Python |
| 几何数据结构 | ✅ C++类 | ✅ Python类 |
| 可微分渲染 | ✅ CUDA/CPU | ❌ Fallback |
| 编译要求 | ❌ 需要C++/CUDA | ✅ 无需编译 |
| Colab兼容性 | ❌ 编译困难 | ✅ 直接使用 |

## 📚 依赖项

```bash
pip install torch svgpathtools cssutils matplotlib
```

## 💡 适用场景

- ✅ **VAE训练** - 只需要SVG几何数据
- ✅ **数据预处理** - SVG文件解析和点提取
- ✅ **Colab环境** - 无编译限制
- ❌ **VSD训练** - 需要可微分渲染，请使用原版PyDiffVG
