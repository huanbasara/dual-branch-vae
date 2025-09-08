"""
简化的SVG渲染功能 - 专注VAE训练需求
"""
import torch
import numpy as np

def svg_to_png(svg_content, width=224, height=224, output_path=None):
    """
    将SVG内容渲染为PNG图像
    
    Args:
        svg_content: SVG文件内容字符串
        width: 输出宽度
        height: 输出高度
        output_path: 保存路径，如果为None则不保存
        
    Returns:
        PIL.Image: PNG图像对象
    """
    try:
        import cairosvg
        from PIL import Image
        import io
        
        # SVG -> PNG bytes
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            output_width=width,
            output_height=height
        )
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
        
        # 保存文件
        if output_path:
            image.save(output_path)
            print(f"✅ PNG saved to: {output_path}")
        
        return image
        
    except ImportError:
        raise ImportError("CairoSVG not installed. Run: pip install cairosvg")
    except Exception as e:
        raise RuntimeError(f"SVG to PNG conversion failed: {e}")

def png_to_tensor(image, device="cuda"):
    """
    将PNG图像转换为torch.Tensor
    
    Args:
        image: PIL.Image对象
        device: 目标设备
        
    Returns:
        torch.Tensor: [height, width, 4] RGBA格式
    """
    # PIL -> numpy -> torch
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)
    
    # 移动到指定设备
    if device and torch.cuda.is_available() and device.startswith('cuda'):
        img_tensor = img_tensor.cuda()
    
    return img_tensor

def svg_to_tensor(svg_content, width=224, height=224, device="cuda"):
    """
    SVG直接转换为tensor（组合上面两个函数）
    """
    image = svg_to_png(svg_content, width, height)
    return png_to_tensor(image, device)

# 简化的RenderFunction（仅保持兼容性）
class RenderFunction:
    @staticmethod
    def apply(width, height, num_samples_x, num_samples_y, seed, background_image, *scene_args):
        """简化的渲染函数，返回白色背景"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_tensor = torch.ones(height, width, 4, dtype=torch.float32)
        if device.startswith('cuda'):
            img_tensor = img_tensor.cuda()
        return img_tensor
    
    @staticmethod
    def serialize_scene(canvas_width, canvas_height, shapes, shape_groups):
        """简化的序列化函数"""
        return [shapes, shape_groups]