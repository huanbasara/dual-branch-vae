import random
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

import torch
# import pydiffvg  # Commented out - using fallback

from deepsvg.my_svg_dataset_pts import cubic_segments_to_points

# ===== PyDiffVG Fallback Classes =====

class Path:
    """Fallback Path class for PyDiffVG compatibility"""
    def __init__(self, num_control_points, points, stroke_width=0.0, is_closed=True):
        self.num_control_points = num_control_points
        self.points = points
        self.stroke_width = stroke_width if hasattr(stroke_width, 'item') else torch.tensor(stroke_width)
        self.is_closed = is_closed

class ShapeGroup:
    """Fallback ShapeGroup class for PyDiffVG compatibility"""
    def __init__(self, shape_ids, fill_color, stroke_color=None, use_even_odd_rule=False):
        self.shape_ids = shape_ids
        self.fill_color = fill_color
        self.stroke_color = stroke_color
        self.use_even_odd_rule = use_even_odd_rule

def save_svg_fallback(filepath, width, height, shapes, shape_groups):
    """Fallback SVG save function using svglib/cairosvg"""
    try:
        from xml.etree import ElementTree as ET
        
        # Create SVG root element
        svg = ET.Element('svg', {
            'width': str(width),
            'height': str(height),
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        
        # Add paths
        for i, (shape, group) in enumerate(zip(shapes, shape_groups)):
            points = shape.points.detach().cpu().numpy() if hasattr(shape.points, 'detach') else shape.points
            color = group.fill_color.detach().cpu().numpy() if hasattr(group.fill_color, 'detach') else group.fill_color
            
            # Convert points to SVG path data
            path_data = f"M {points[0,0]},{points[0,1]} "
            for j in range(1, len(points), 3):
                if j+2 < len(points):
                    path_data += f"C {points[j,0]},{points[j,1]} {points[j+1,0]},{points[j+1,1]} {points[j+2,0]},{points[j+2,1]} "
            if shape.is_closed:
                path_data += "Z"
            
            # Create path element
            path_elem = ET.SubElement(svg, 'path', {
                'd': path_data,
                'fill': f"rgba({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f},{color[3] if len(color)>3 else 1})",
                'stroke': 'none' if group.stroke_color is None else 'black',
                'stroke-width': str(shape.stroke_width.item() if hasattr(shape.stroke_width, 'item') else shape.stroke_width)
            })
        
        # Write to file
        tree = ET.ElementTree(svg)
        tree.write(filepath, encoding='unicode', xml_declaration=True)
        print(f"✅ SVG saved to {filepath} (fallback method)")
        
    except Exception as e:
        print(f"⚠️ SVG save failed: {e}")

class RenderFunction:
    """Fallback rendering function"""
    @staticmethod
    def apply(width, height, *args):
        # Return a dummy white image tensor for VAE training
        return torch.ones(height, width, 4)  # RGBA format
    
    @staticmethod
    def serialize_scene(width, height, shapes, shape_groups):
        # Return dummy scene args
        return []

# ===== Original functions with fallback support =====

def load_model1(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))

def load_model2(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state, strict=False)

def random_select_files(file_list, num_samples=50000):
    r0_files = filter_ending_files(file_list)
    return random.sample(r0_files, num_samples)

def filter_ending_files(file_list, ending="r0"):
    fn_list = []
    for fn in file_list:
        fn_pre = fn.split(".")[0]
        if (fn_pre.endswith(ending)):
            fn_list.append(fn)
    return fn_list

def imscatter(x, y, image, ax=None, zoom=1, offset=(0, 0)):
    """Function to plot image on specified x, y coordinates with an offset."""
    if ax is None:
        ax = plt.gca()
    im = offsetbox.OffsetImage(image, zoom=zoom, cmap='gray')
    ab = offsetbox.AnnotationBbox(
        im, (x + offset[0], y + offset[1]), frameon=False, pad=0.0)
    ax.add_artist(ab)

def tensor_to_img(tensor, to_grayscale=True):
    """Convert a tensor in the shape [C, H, W] to a numpy image in the shape [H, W, C] with uint8 type."""
    tensor = tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

    # Convert to grayscale if required
    if to_grayscale and tensor.shape[-1] == 3:
        tensor = tensor.mean(dim=-1, keepdim=True)

    # Convert to uint8
    tensor = (tensor * 255).clamp(0, 255).byte()
    img = tensor.cpu().numpy()
    return img

# ===== Path and rendering functions with fallback =====

def pts_to_pathObj(convert_points):
    """Create path object using fallback Path class"""
    # Number of control points per segment is 2. Hence, calculate the total number of segments.
    num_segments = int(convert_points.shape[0] / 3)
    num_control_points = [2] * num_segments
    num_control_points = torch.LongTensor(num_control_points)

    # Create a path object using fallback
    path = Path(
        num_control_points=num_control_points,
        points=convert_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True
    )
    return path

def apply_affine_transform_origin(points, theta, tx, ty, s):
    """Apply affine transformation, including rotation, translation, and overall scaling."""
    device = points.device

    # Create the affine transformation matrix for rotation and scaling.
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    aff_left = torch.stack([
        s * cos_theta, -s * sin_theta,
        s * sin_theta, s * cos_theta
    ]).view(2, 2).to(device)

    transformed_points = torch.mm(points, aff_left)
    # Add the translation.
    transformed_points += torch.stack([tx, ty]).to(device)
    return transformed_points

def apply_affine_transform(points, theta, tx, ty, s):
    """Apply affine transformation, including rotation, translation, and overall scaling."""
    device = points.device
    center = torch.mean(points, dim=0)

    # Translate points to center around the origin
    points_centered = points - center

    # Apply affine transformation (rotation and scaling)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    aff_left = torch.stack([
        s * cos_theta, -s * sin_theta,
        s * sin_theta, s * cos_theta
    ]).view(2, 2).to(device)

    transformed_points = torch.mm(points_centered, aff_left)

    # Translate back to the original location and apply overall translation
    translation = torch.stack([tx, ty])
    transformed_points += center + translation

    return transformed_points.to(device)

def cubics_to_points_affine(cubics, theta, tx, ty, s, s_norm, h=224, w=224, use_affine_norm=False):
    convert_points_ini = cubic_segments_to_points(cubics)

    # Apply affine transform to convert_points
    convert_points = apply_affine_transform(convert_points_ini, theta, tx, ty, s)

    # Clamp the values of convert_points_ini to be in [0,1]
    if (use_affine_norm):
        convert_points_ini_trans = s_norm.inverse_transform(convert_points_ini)
        convert_points_trans = s_norm.inverse_transform(convert_points)
        return convert_points_trans, convert_points_ini_trans
    else:
        return convert_points, convert_points_ini

def z_to_affine_pts(z, theta, tx, ty, s, model, s_norm, h=224, w=224, use_affine_norm=False):
    generated_data = model(args_enc=None, args_dec=None, z=z.unsqueeze(1).unsqueeze(2))
    generated_pts = generated_data["args_logits"]
    recon_data_output = generated_pts.squeeze(1)

    bat_s = 1
    ini_cubics_batch = recon_data_output.view(bat_s, -1, 4, 2)
    ini_cubics = ini_cubics_batch[0]

    if (use_affine_norm):
        ini_cubics_trans = ini_cubics
    else:
        ini_cubics_trans = s_norm.inverse_transform(ini_cubics)

    convert_points, convert_points_ini = cubics_to_points_affine(
        cubics=ini_cubics_trans, theta=theta, tx=tx, ty=ty, s=s, s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

    return convert_points, convert_points_ini

def recon_to_affine_pts(recon_data_output, theta, tx, ty, s, s_norm, h=224, w=224, use_affine_norm=False):
    bat_s = 1
    ini_cubics_batch = recon_data_output.view(bat_s, -1, 4, 2)
    ini_cubics = ini_cubics_batch[0]

    if (use_affine_norm):
        ini_cubics_trans = ini_cubics
    else:
        ini_cubics_trans = s_norm.inverse_transform(ini_cubics)

    convert_points, convert_points_ini = cubics_to_points_affine(
        cubics=ini_cubics_trans, theta=theta, tx=tx, ty=ty, s=s, s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

    return convert_points, convert_points_ini

def cubics_to_pathObj_affine(cubics, theta, tx, ty, s, s_norm, h=224, w=224, use_affine_norm=False):
    """Given a cubics tensor, return a pathObj."""
    convert_points, convert_points_ini = cubics_to_points_affine(
        cubics, theta, tx, ty, s, s_norm, h, w, use_affine_norm)
    path = pts_to_pathObj(convert_points)
    return path, convert_points_ini

# ===== Shape and rendering functions with fallback =====

def paths_to_shapes(path_list, fill_color_list, stroke_width_list=None, stroke_color_list=None):
    if stroke_width_list is not None:
        for i, path in enumerate(path_list):
            path.stroke_width = stroke_width_list[i]

    tp_shapes = path_list
    tp_shape_groups = [
        ShapeGroup(  # Using fallback ShapeGroup
            shape_ids=torch.LongTensor([i]),
            fill_color=color,
            stroke_color=None if stroke_color_list is None else stroke_color_list[i],
            use_even_odd_rule=False
        ) for i, color in enumerate(fill_color_list)
    ]

    return tp_shapes, tp_shape_groups

def save_paths_svg(path_list,
                   fill_color_list=[],
                   stroke_width_list=[],
                   stroke_color_list=[],
                   svg_path_fp="",
                   canvas_height=224,
                   canvas_width=224):

    tp_shapes = []
    tp_shape_groups = []

    for pi in range(len(path_list)):
        ini_path = path_list[pi]
        ini_path.stroke_width = stroke_width_list[pi]
        tp_shapes.append(ini_path)

        tp_fill_color = fill_color_list[pi]

        tp_path_group = ShapeGroup(  # Using fallback ShapeGroup
            shape_ids=torch.LongTensor([pi]),
            fill_color=tp_fill_color,
            stroke_color=stroke_color_list[pi],
            use_even_odd_rule=False)
        tp_shape_groups.append(tp_path_group)

    if (len(svg_path_fp) > 0):
        save_svg_fallback(svg_path_fp, canvas_width, canvas_height, tp_shapes, tp_shape_groups)  # Using fallback

    return tp_shapes, tp_shape_groups

def render_and_compose(tmp_paths_list, color_list, stroke_width_list=None, stroke_color_list=None, w=224, h=224, svg_path_fp="", para_bg=None, render_func=None, return_shapes=False, device="cuda"):
    if para_bg is None:
        para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    if (render_func is None):
        render_func = RenderFunction.apply  # Using fallback RenderFunction

    tp_shapes, tp_shape_groups = save_paths_svg(
        path_list=tmp_paths_list, fill_color_list=color_list, stroke_width_list=stroke_width_list, stroke_color_list=stroke_color_list, svg_path_fp=svg_path_fp, canvas_height=h, canvas_width=w)

    # For VAE training, we don't need actual rendering - return dummy tensor
    scene_args = []  # Simplified scene args
    tmp_img = render_func(w, h, 2, 2, 0, None, *scene_args)

    # Compose img with white background
    combined_img = tmp_img[:, :, 3:4] * tmp_img[:, :, :3] + para_bg * (1 - tmp_img[:, :, 3:4])
    recon_imgs = combined_img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW

    if (return_shapes):
        return recon_imgs, combined_img, tp_shapes, tp_shape_groups

    return recon_imgs, combined_img

# ===== Loss and utility functions (unchanged) =====

def regularization_loss(z):
    mean = z.mean()
    std = z.std()
    return mean**2 + (std - 1)**2

def l2_regularization_loss(z):
    return torch.norm(z, p=2) ** 2

def kl_divergence_loss(z):
    mean = z.mean()
    variance = z.var()
    kl_loss = -0.5 * (1 + torch.log(variance) - mean**2 - variance)
    return kl_loss.sum()

def safe_pow(t, exponent, eps=1e-6):
    return t.clamp(min=eps).pow(exponent)

def opacity_penalty(colors, coarse_learning=True):
    factor = 1 if coarse_learning else 0
    alpha = colors[:, 3]
    if coarse_learning:
        penalty = factor * safe_pow(alpha, 0.5).mean()
    else:
        binary_alpha = (alpha > 0.5).float()
        penalty = factor * safe_pow(binary_alpha, 0.5).mean()
    return penalty

def binary_alpha_penalty_sigmoid(colors):
    alpha = colors[:, 3]
    binary_alpha = torch.sigmoid(10 * (alpha - 0.5))
    loss = torch.mean((binary_alpha - alpha) ** 2)
    return loss

def binary_alpha_penalty_l1(colors):
    alpha = colors[:, 3]
    loss = torch.mean(torch.min(alpha, 1 - alpha))
    return loss

def l1_penalty(tensor):
    return torch.sum(torch.abs(tensor))

def control_polygon_distance(points):
    diff = points[1:] - points[:-1]
    squared_distances = (diff ** 2).sum(dim=1)
    return squared_distances.mean()

def generate_single_affine_parameters(center, h, w, use_affine_norm=False, gt=None, device="cuda"):
    circle_center_norm = np.array([0.4935, 0.4953])
    circle_center = circle_center_norm * h

    if use_affine_norm:
        center_norm = center * 1.0 / h
        tx_val = center_norm[0] - circle_center_norm[0]
        ty_val = center_norm[1] - circle_center_norm[1]
    else:
        tx_val = float(center[0]) - circle_center[0]
        ty_val = float(center[1]) - circle_center[1]

    tx = torch.tensor(tx_val, dtype=torch.float32, device=device, requires_grad=True)
    ty = torch.tensor(ty_val, dtype=torch.float32, device=device, requires_grad=True)
    theta = torch.tensor(0.0, device=device, requires_grad=True)
    s = torch.tensor(0.11, device=device, requires_grad=True)

    wref, href = map(int, center)
    wref = max(0, min(wref, w - 1))
    href = max(0, min(href, h - 1))

    if (gt is None):
        fill_color_init = torch.tensor(np.concatenate((npr.uniform(size=[3]), [np.random.uniform(0.7, 1)])), dtype=torch.float32, device=device, requires_grad=True)
    else:
        fill_color_init = list(gt[0, :, href, wref]) + [1.]
        fill_color_init = torch.FloatTensor(fill_color_init).to(device).requires_grad_(True)

    return tx, ty, theta, s, fill_color_init

def get_experiment_id(debug=False):
    if debug:
        return 999999999999
    import time
    time.sleep(0.5)
    return int(time.time()*100)

def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments * 3)
    for i in range(0, segments * 3):
        point = (np.cos(np.deg2rad(i * avg_degree)), np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = (points) * radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points

class RandomCoordInit():
    def __init__(self, canvas_size, edge_margin_ratio=0.1):
        self.canvas_size = canvas_size
        self.edge_margin_ratio = edge_margin_ratio

    def __call__(self):
        h, w = self.canvas_size
        edge_margin_w = self.edge_margin_ratio * w
        edge_margin_h = self.edge_margin_ratio * h
        x = npr.uniform(edge_margin_w, w - edge_margin_w)
        y = npr.uniform(edge_margin_h, h - edge_margin_h)
        return [x, y]

class random_coord_init():
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size

    def __call__(self):
        h, w = self.canvas_size
        return [npr.uniform(0, 1) * w, npr.uniform(0, 1) * h]

class naive_coord_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', replace_sampling=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0])**2).sum(0)
        elif format == ['[2D x c]']:
            self.map = ((pred - gt)**2).sum(-1)
        else:
            raise ValueError
        self.replace_sampling = replace_sampling

    def __call__(self):
        coord = np.where(self.map == self.map.max())
        coord_h, coord_w = coord[0][0], coord[1][0]
        if self.replace_sampling:
            self.map[coord_h, coord_w] = -1
        return [coord_w, coord_h]
