"""
Path utilities for converting different path types to cubic bezier curves
"""
import torch
from .shape import Path


def convert_path_cubic(cur_path):
    """
    Convert all path segments to cubic bezier curves (standardization)
    
    Args:
        cur_path: Path object with mixed segment types
        
    Returns:
        tuple: (num_points, standardized_path) where all segments are cubic bezier
    """
    idx_pts = 0
    pre_points = cur_path.points[0]
    new_points = [[pre_points[0], pre_points[1]]]
    new_num_control_points = []

    for num_i in cur_path.num_control_points:
        if (num_i == 0):
            # Line segment -> convert to degenerate cubic bezier
            idx_pts += 1
            idx_pts = idx_pts % len(cur_path.points)

            # Create degenerate cubic: P0, P0, P1, P1 (control points repeated)
            new_points.extend([
                [pre_points[0], pre_points[1]], 
                [cur_path.points[idx_pts][0], cur_path.points[idx_pts][1]], 
                [cur_path.points[idx_pts][0], cur_path.points[idx_pts][1]]
            ])

        elif (num_i == 1):
            # Quadratic bezier -> convert to cubic bezier
            idx_pts += 2
            idx_pts = idx_pts % len(cur_path.points)
            
            # Convert quadratic to cubic using standard formula
            p0 = pre_points
            p1 = cur_path.points[idx_pts-1]  # control point
            p2 = cur_path.points[idx_pts]    # end point
            
            # Cubic control points: C1 = P0 + 2/3*(P1-P0), C2 = P2 + 2/3*(P1-P2)
            c1 = [p0[0] + 2/3*(p1[0]-p0[0]), p0[1] + 2/3*(p1[1]-p0[1])]
            c2 = [p2[0] + 2/3*(p1[0]-p2[0]), p2[1] + 2/3*(p1[1]-p2[1])]
            
            new_points.extend([c1, c2, [p2[0], p2[1]]])

        else:
            # Already cubic bezier -> keep as is
            idx_pts += 3
            idx_pts = idx_pts % len(cur_path.points)

            new_points.extend([
                [cur_path.points[idx_pts-2][0], cur_path.points[idx_pts-2][1]], 
                [cur_path.points[idx_pts-1][0], cur_path.points[idx_pts-1][1]], 
                [cur_path.points[idx_pts][0], cur_path.points[idx_pts][1]]
            ])

        pre_points = cur_path.points[idx_pts]
        new_num_control_points.append(2)  # All segments now have 2 control points

    # Convert to tensors
    new_points = torch.tensor(new_points, dtype=torch.float32)
    new_num_control_points = torch.LongTensor(new_num_control_points)

    # Create standardized path using pydiffvg_lite
    tmp_path = Path(
        num_control_points=new_num_control_points,
        points=new_points,
        stroke_width=getattr(cur_path, 'stroke_width', torch.tensor(1.0)),
        is_closed=getattr(cur_path, 'is_closed', True)
    )

    num_pts = tmp_path.points.shape[0]
    assert ((num_pts-1) % 3 == 0), f"Invalid point count: {num_pts}, should be 3k+1"

    return num_pts, tmp_path


def standardize_svg_paths(shapes):
    """
    Standardize all paths in a list of shapes to use only cubic bezier curves
    
    Args:
        shapes: List of shape objects from svg_to_scene
        
    Returns:
        List of standardized shapes with only cubic bezier segments
    """
    standardized_shapes = []
    
    for shape in shapes:
        if hasattr(shape, 'points') and hasattr(shape, 'num_control_points'):
            # This is a Path object, standardize it
            _, standardized_path = convert_path_cubic(shape)
            standardized_shapes.append(standardized_path)
        else:
            # Other shape types (Circle, Ellipse, etc.), keep as is
            standardized_shapes.append(shape)
    
    return standardized_shapes
