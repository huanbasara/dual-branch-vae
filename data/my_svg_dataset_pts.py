from sklearn.preprocessing import MinMaxScaler
import os
import random
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import PIL
import PIL.Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 使用本地pydiffvg_lite替代pydiffvg
import pydiffvg_lite as pydiffvg

pydiffvg.set_print_timing(False)
gamma = 1.0


class Normalize(object):
    def __init__(self, w, h):
        self.w = w * 1.0
        self.h = h * 1.0
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def __call__(self, points):
        points = points / \
            torch.tensor([self.w, self.h], dtype=torch.float32).to(
                points.device)

        return points

    def inverse_transform(self, points):

        points = points * \
            (torch.tensor([self.w, self.h],
             dtype=torch.float32).to(points.device))

        return points


def load_target_new(fp, img_size=64):
    target = PIL.Image.open(fp).resize((img_size, img_size))
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # target = np.array(target)

    transforms_ = []
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    target = data_transforms(target)
    target = target[0]
    target = target.unsqueeze(0)
    return target


def load_target(fp):
    target = PIL.Image.open(fp)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # target = np.array(target)

    transforms_ = []
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    target = data_transforms(target)
    return target


class SVGDataset_nopadding(Dataset):
    def __init__(self, directory, h, w, fixed_length=60, file_list=None, img_dir=None, transform=None, use_model_fusion=False):
        super(SVGDataset_nopadding, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.transform = transform
        self.h = h
        self.w = w
        self.fixed_length = fixed_length
        self.img_dir = img_dir
        self.use_model_fusion = use_model_fusion

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        try:
            assert os.path.exists(filepath)

            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
                filepath)

            for path in shapes:
                points = path.points
                num_control_points = path.num_control_points
                break

            # Transform points if applicable
            if self.transform:
                points = self.transform(points)

            # Truncate if sequence is too long
            if points.shape[0] > self.fixed_length:
                points = points[:self.fixed_length]

            # Compute the cubics segments
            cubics = get_cubic_segments_from_points(
                points=points, num_control_points=num_control_points)

            desired_cubics_length = self.fixed_length // 3

            assert cubics.shape[0] == desired_cubics_length

            path_img = []
            if self.img_dir:
                im_pre = self.file_list[idx].split(".")[0]
                im_path = os.path.join(self.img_dir, im_pre + ".png")
                if (os.path.exists(im_path)):
                    if (self.use_model_fusion):
                        path_img = load_target_new(im_path)
                    else:
                        path_img = load_target(im_path)

            res_data = {
                # control points
                "points": points,
                # cubics segments
                "cubics": cubics,
                "lengths": self.fixed_length,
                "filepaths": filepath,
                "path_img": path_img
            }

        except Exception as e:
            print(f"Error processing index: {idx}, Filepath: {filepath}")
            print(f"Error message: {str(e)}")
            raise e

        return res_data


class SVGDataset(Dataset):
    def __init__(self, directory, h, w, fixed_length=60, file_list=None, img_dir=None, transform=None, use_model_fusion=False):
        super(SVGDataset, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.transform = transform
        self.h = h
        self.w = w
        self.fixed_length = fixed_length
        self.img_dir = img_dir
        self.use_model_fusion = use_model_fusion

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
            filepath)

        for path in shapes:
            points = path.points
            num_control_points = path.num_control_points
            break

        # Transform points if applicable
        if self.transform:
            points = self.transform(points)

        # Truncate if sequence is too long
        len_points = points.shape[0]
        if len_points > self.fixed_length - 1:
            points = points[:self.fixed_length - 1]

        # Compute the cubics segments
        cubics = get_cubic_segments_from_points(
            points=points, num_control_points=num_control_points)

        # Determine the desired number of cubics based on fixed_length and existing points
        desired_cubics_length = (self.fixed_length - 1) // 3

        # Pad or truncate cubics based on desired length
        if cubics.shape[0] < desired_cubics_length:
            padding_needed = desired_cubics_length - cubics.shape[0]
            # Using zero-padding
            padding_tensor = torch.full((padding_needed, 4, 2), 0.0)
            cubics = torch.cat([cubics, padding_tensor], dim=0)
        elif cubics.shape[0] > desired_cubics_length:
            cubics = cubics[:desired_cubics_length]

        # Append end token (0.0, 0.0)
        points = torch.cat((points, torch.tensor(
            [[0.0, 0.0]], dtype=torch.float32)))

        len_points = points.shape[0]
        # Pad to fixed_length
        if (len_points < self.fixed_length):
            points = torch.cat((points, torch.tensor(
                [[0.0, 0.0]] * (self.fixed_length - len_points), dtype=torch.float32)))

        assert points.shape[0] == self.fixed_length

        path_img = []
        if self.img_dir:
            im_pre = self.file_list[idx].split(".")[0]
            im_path = os.path.join(self.img_dir, im_pre + ".png")
            if (os.path.exists(im_path)):
                if (self.use_model_fusion):
                    path_img = load_target_new(im_path)
                else:
                    path_img = load_target(im_path)

        res_data = {
            # control points
            "points": points,
            # cubics segments
            "cubics": cubics,
            "lengths": len_points,
            "filepaths": filepath,
            "path_img": path_img
        }

        return res_data


class SVGDataset_GoogleDrive(Dataset):
    """
    Dataset for Google Drive SVG files with full preprocessing in __init__
    """
    def __init__(self, data_path=None, h=224, w=224, fixed_length=30, category=None, file_list=None, svg_files=None, transform=None, use_model_fusion=False):
        super(SVGDataset_GoogleDrive, self).__init__()
        self.h = h
        self.w = w
        self.fixed_length = fixed_length
        self.transform = transform
        self.use_model_fusion = use_model_fusion
        
        # Collect SVG files to process
        svg_file_paths = []
        
        if svg_files:
            # Direct file list input (for batch processing)
            svg_file_paths = svg_files
        elif data_path and category and file_list:
            # Process specific files in a category
            category_dir = os.path.join(data_path, category)
            for filename in file_list:
                svg_path = os.path.join(category_dir, filename)
                if os.path.exists(svg_path):
                    svg_file_paths.append(svg_path)
        elif data_path:
            # Process all categories and files
            for category_name in os.listdir(data_path):
                category_dir = os.path.join(data_path, category_name)
                if os.path.isdir(category_dir):
                    for filename in os.listdir(category_dir):
                        if filename.endswith('.svg'):
                            svg_path = os.path.join(category_dir, filename)
                            svg_file_paths.append(svg_path)
        
        
        # Process all SVG files and store ready-to-use samples
        self.samples = []
        
        for i, svg_path in enumerate(svg_file_paths):
            
            try:
                samples = self.process_svg_file(svg_path)
                self.samples.extend(samples)
            except Exception as e:
                pass  # Skip processing warnings
        

    def process_svg_file(self, svg_path):
        """Process a single SVG file and return list of valid samples"""
        # Parse SVG file
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
        
        # Standardize all paths to cubic bezier curves
        standardized_shapes = pydiffvg.standardize_svg_paths(shapes)
        
        valid_samples = []
        
        for path_idx, shape in enumerate(standardized_shapes):
            if not hasattr(shape, 'points'):
                continue
            
            # Filter out border/frame paths
            if self._is_border_frame(shape, path_idx, canvas_width, canvas_height):
                print(f"🚫 Filtered border frame: path {path_idx} ({shape.points.shape[0]} points)")
                continue
                
            try:
                sample = self.process_single_path(svg_path, shape, path_idx)
                if sample is not None:  # Only add valid samples
                    valid_samples.append(sample)
            except Exception as e:
                pass  # Skip invalid paths silently
        
        return valid_samples
    
    def _is_border_frame(self, shape, path_idx, canvas_width=192, canvas_height=192):
        """
        Detect if a shape is likely a border/frame that should be filtered out
        
        Args:
            shape: The parsed shape object
            path_idx: Index of the path (borders are usually path_idx=0)
            canvas_width, canvas_height: SVG canvas dimensions
            
        Returns:
            bool: True if this shape appears to be a border/frame
        """
        if not hasattr(shape, 'points') or shape.points.shape[0] == 0:
            return False
            
        points = shape.points
        
        # Heuristic 1: Border frames are usually the first sub-path (path_idx=0)
        # and have relatively few points (< 30 points typically)
        if path_idx == 0 and points.shape[0] < 30:
            
            # Heuristic 2: Check if points span nearly the entire canvas
            x_min, x_max = points[:, 0].min().item(), points[:, 0].max().item()
            y_min, y_max = points[:, 1].min().item(), points[:, 1].max().item()
            
            # Convert to original coordinate system (considering SVG transforms)
            # Most SVGs use viewBox="0 0 192 192" and transform="scale(0.1, -0.1)"
            # So coordinates are typically in range [0, 1920] before transform
            
            width_coverage = (x_max - x_min) / canvas_width
            height_coverage = (y_max - y_min) / canvas_height
            
            # Heuristic 3: Border covers > 70% of canvas in both dimensions
            if width_coverage > 0.7 and height_coverage > 0.7:
                
                # Heuristic 4: Border has simple geometry (rectangular-like)
                # Check if most points are on the perimeter
                corner_tolerance = 0.1 * min(canvas_width, canvas_height)
                
                edge_points = 0
                total_points = points.shape[0]
                
                for point in points:
                    x, y = point[0].item(), point[1].item()
                    # Check if point is near any edge
                    near_left = abs(x - x_min) < corner_tolerance
                    near_right = abs(x - x_max) < corner_tolerance  
                    near_top = abs(y - y_min) < corner_tolerance
                    near_bottom = abs(y - y_max) < corner_tolerance
                    
                    if near_left or near_right or near_top or near_bottom:
                        edge_points += 1
                
                # If >70% of points are on edges, likely a border
                edge_ratio = edge_points / total_points
                if edge_ratio > 0.7:
                    return True
        
        return False
    
    def calculate_max_segments(self, num_points, num_control_points):
        """Calculate maximum number of segments that can fit in truncated points"""
        # For cubic bezier: each segment needs 3 points (start point shared with previous)
        # First segment: 1 start + 3 = 4 points
        # Each additional segment: +3 points
        # Total points = 1 + 3*num_segments
        # So: num_segments = (num_points - 1) / 3
        
        max_segments = (num_points - 1) // 3
        return max(0, min(max_segments, len(num_control_points)))
    
    def _create_svg_from_points(self, points):
        """Create SVG content from processed points"""
        # Apply inverse transform to get pixel coordinates for SVG
        normalizer = Normalize(self.w, self.h)
        pixel_points = normalizer.inverse_transform(points)
        
        # Start SVG content
        svg_content = f'''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}">
    <path d="'''
        
        # Create path data from points
        path_data = f"M {pixel_points[0][0]} {pixel_points[0][1]}"
        
        # Add cubic bezier curves (3 points per curve)
        # We have 30 points: point 0 is start, then 9 groups of 3 points each = 27 points
        # So we need: points[1,2,3], [4,5,6], ..., [25,26,27], [28,29,0] for 10 curves
        for i in range(1, len(pixel_points), 3):
            if i + 2 <= len(pixel_points) - 1:  # Normal case: all 3 points exist
                p1 = pixel_points[i]
                p2 = pixel_points[i + 1] 
                p3 = pixel_points[i + 2]
                path_data += f" C {p1[0]} {p1[1]} {p2[0]} {p2[1]} {p3[0]} {p3[1]}"
            elif i + 1 <= len(pixel_points) - 1:  # Last curve: use start point to close
                p1 = pixel_points[i]
                p2 = pixel_points[i + 1] 
                p3 = pixel_points[0]  # Close back to start point
                path_data += f" C {p1[0]} {p1[1]} {p2[0]} {p2[1]} {p3[0]} {p3[1]}"
        
        svg_content += path_data
        svg_content += '''" fill="none" stroke="black" stroke-width="2"/>
</svg>'''
        
        return svg_content
    
    def process_single_path(self, svg_path, shape, path_idx):
        """Process a single path and return ready-to-use sample"""
        points = shape.points
        num_control_points = shape.num_control_points
        
        # Transform points if applicable
        if self.transform:
            points = self.transform(points)
        
        # First convert to cubic bezier curves (from original path)
        original_cubics = get_cubic_segments_from_points(
            points=points, num_control_points=num_control_points)
        
        # Check if we have enough curves to sample from
        if original_cubics.shape[0] == 0:
            return None
        
        # Sample fixed number of points from the path
        # Target: 10 cubics = 31 points (including start point), then remove last point = 30 points
        target_points = 31  # 1 start + 10*3 = 31 points
        desired_cubics_length = 10
        
        # Sample points uniformly along the path
        if original_cubics.shape[0] == 1:
            # Single curve: sample more points from this curve
            sampled_points = sample_bezier(original_cubics, k=target_points)
        else:
            # Multiple curves: distribute sampling across curves
            points_per_curve = max(3, target_points // original_cubics.shape[0])
            sampled_points = sample_bezier(original_cubics, k=points_per_curve)
            
            # If we have too many points, subsample uniformly
            if sampled_points.shape[0] > target_points:
                indices = torch.linspace(0, sampled_points.shape[0] - 1, target_points, dtype=torch.long)
                sampled_points = sampled_points[indices]
            # If we have too few points, repeat the last segment sampling
            elif sampled_points.shape[0] < target_points:
                # Add more samples from the last curve
                needed = target_points - sampled_points.shape[0]
                last_curve = original_cubics[-1:, :, :]
                extra_points = sample_bezier(last_curve, k=needed)
                sampled_points = torch.cat([sampled_points, extra_points], dim=0)
        
        # Ensure we have exactly target_points
        if sampled_points.shape[0] > target_points:
            sampled_points = sampled_points[:target_points]
        elif sampled_points.shape[0] < target_points:
            # Pad by repeating the last point
            last_point = sampled_points[-1:, :]
            needed = target_points - sampled_points.shape[0]
            padding = last_point.repeat(needed, 1)
            sampled_points = torch.cat([sampled_points, padding], dim=0)
        
        # Remove the last point to get 30 points (following circle_10.svg logic)
        points = sampled_points[:-1]  # 30 points
        
        # Create num_control_points for 10 cubic curves (each curve has 2 control points)
        num_control_points = torch.tensor([2] * desired_cubics_length)
        
        # Compute the final cubics from sampled points
        cubics = get_cubic_segments_from_points(
            points=points, num_control_points=num_control_points)
        
        # Render image if needed
        path_img = []
        if self.use_model_fusion:
            try:
                # Create SVG from processed points and render it
                processed_svg_content = self._create_svg_from_points(points)
                img_tensor = pydiffvg.svg_to_tensor(processed_svg_content, width=self.w, height=self.h)
                # img_tensor should be (H, W, 3) format now
                if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:  # HWC format
                    path_img = img_tensor
                else:
                    print(f"Warning: Unexpected tensor shape {img_tensor.shape}")
                    path_img = torch.zeros((self.h, self.w, 3))
            except Exception as e:
                print(f"Warning: Failed to render processed path: {e}")
                path_img = torch.zeros((self.h, self.w, 3))
        
        return {
            "points": points,
            "cubics": cubics,
            "lengths": self.fixed_length,
            "filepaths": svg_path,
            "path_img": path_img,
            "path_index": path_idx
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Pure data access - all processing done in __init__"""
        return self.samples[idx]


def collate_fn(batch):
    # Fixed length to which sequences should be padded
    fixed_length = 30

    # Truncate too long sequences, append end token and pad each sequence in the batch to fixed_length
    padded_batch = []
    for seq in batch:
        # Truncate if sequence is too long
        if len(seq) > fixed_length - 1:
            seq = seq[:fixed_length - 1]

        # Append end token
        seq = torch.cat((seq, torch.tensor([[0.0, 0.0]], dtype=torch.float32)))

        # Pad to fixed_length
        seq = torch.cat((seq, torch.tensor(
            [[0.0, 0.0]] * (fixed_length - len(seq)), dtype=torch.float32)))

        padded_batch.append(seq)

    # Convert list of tensors to a single tensor
    padded_batch = torch.stack(padded_batch)

    # Compute sequence lengths
    lengths = [len(seq) for seq in padded_batch]

    return padded_batch, lengths


# ----------------------------------------
def get_segments(pathObj):
    segments = []
    lines = []
    quadrics = []
    cubics = []
    # segList = (lines, quadrics, cubics)
    idx = 0
    total_points = pathObj.points.shape[0]

    for ncp in pathObj.num_control_points.numpy():
        pt1 = pathObj.points[idx]

        if ncp == 0:
            segments.append((0, len(lines)))
            pt2 = pathObj.points[(idx + 1) % total_points]
            lines.append(pt1)
            lines.append(pt2)
            idx += 1
        elif ncp == 1:
            segments.append((1, len(quadrics)))
            pt2 = pathObj.points[idx + 1]
            pt3 = pathObj.points[(idx + 2) % total_points]
            quadrics.append(pt1)
            quadrics.append(pt2)
            quadrics.append(pt3)
            idx += ncp+1
        elif ncp == 2:
            segments.append((2, len(cubics)))
            pt2 = pathObj.points[idx + 1]
            pt3 = pathObj.points[idx + 2]
            pt4 = pathObj.points[(idx + 3) % total_points]
            cubics.append(pt1)
            cubics.append(pt2)
            cubics.append(pt3)
            cubics.append(pt4)

            idx += ncp + 1

    # total_points/3*4
    cubics = torch.stack(cubics).view(-1, 4, 2)
    return cubics


def get_cubic_segments_mask(lengths, max_pts_len_thresh, device="cuda"):
    cubic_lengths = (lengths - 1) // 3
    max_cubics_length = (max_pts_len_thresh - 1) // 3

    # Create the mask tensor
    cubics_mask = torch.arange(max_cubics_length).unsqueeze(
        0) < cubic_lengths.unsqueeze(1)

    # Expand dimensions to match cubics tensor shape
    cubics_mask = cubics_mask.unsqueeze(
        -1).unsqueeze(-1).expand(-1, -1, 4, 2).float()

    cubics_mask = cubics_mask.to(device)

    return cubics_mask, cubic_lengths


def get_cubic_segments_from_points(points, num_control_points):
    cubics = []
    idx = 0
    total_points = points.shape[0]

    for ncp in num_control_points.numpy():
        assert ncp == 2

        pt1 = points[idx]
        pt2 = points[idx + 1]
        pt3 = points[idx + 2]
        pt4 = points[(idx + 3) % total_points]

        cubics.append(pt1)
        cubics.append(pt2)
        cubics.append(pt3)
        cubics.append(pt4)

        idx += 3

    # total_points/3*4
    cubics = torch.stack(cubics).view(-1, 4, 2)
    return cubics


def get_cubic_segments(pathObj):
    cubics = get_cubic_segments_from_points(
        points=pathObj.points, num_control_points=pathObj.num_control_points)

    return cubics


def cubic_segments_to_points(cubics):
    num_segments = cubics.shape[0]
    points_list = []

    first_segment = cubics[0]
    points_list.extend([pt for pt in first_segment[:-1]])

    for idx in range(1, num_segments):
        prev_end = cubics[idx-1][3]
        current_start = cubics[idx][0]

        shared_point = (prev_end + current_start) / 2.0
        points_list.append(shared_point)

        points_list.extend([pt for pt in cubics[idx][1:3]])

    convert_points = torch.stack(points_list)

    return convert_points


def cubics_to_pathObj(cubics):
    """Given a cubics tensor, return a pathObj."""
    convert_points = cubic_segments_to_points(cubics)

    # Number of control points per segment is 2. Hence, calculate the total number of segments.
    num_segments = int(convert_points.shape[0] / 3)
    num_control_points = [2] * num_segments
    num_control_points = torch.LongTensor(num_control_points)

    # Create a path object
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=convert_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True
    )

    return path


def cubic_bezier(t, P0, P1, P2, P3):
    """
    Compute point on a cubic Bezier curve.
    :param t: torch.Tensor, Parameter t in range [0,1].
    :param P0: torch.Tensor, Start Point.
    :param P1: torch.Tensor, Control Point 1.
    :param P2: torch.Tensor, Control Point 2.
    :param P3: torch.Tensor, End Point.
    :return: torch.Tensor, the corresponding point on the cubic Bezier curve.
    """
    t_complement = 1 - t
    B = (
        t_complement ** 3 * P0
        + 3 * t_complement ** 2 * t * P1
        + 3 * t_complement * t ** 2 * P2
        + t ** 3 * P3
    )
    return B


def sample_bezier(cubics, k=5):
    """
    Sample points on cubic Bezier curves.
    :param cubics: torch.Tensor, shape [num_curves, 4, 2], representing cubic Bezier curves.
    :param k: int, number of sample points per curve.
    :return: torch.Tensor, shape [num_curves * k, 2], representing the sampled points on the Bezier curves.
    """
    num_curves = cubics.shape[0]
    # shape [1, k, 1]
    ts = torch.linspace(0, 1, k).view(1, k, 1).to(cubics.device)

    P0, P1, P2, P3 = cubics[:, 0], cubics[:, 1], cubics[:, 2], cubics[:, 3]

    # Calculate cubic Bezier for all curves and all t values at once
    point = (1-ts)**3 * P0.unsqueeze(1) + 3*(1-ts)**2*ts * P1.unsqueeze(1) + \
        3*(1-ts)*ts**2 * P2.unsqueeze(1) + ts**3 * P3.unsqueeze(1)

    # Reshape the tensor to get points in [num_curves * k, 2] format
    point = point.reshape(-1, 2)

    # shape [num_curves * k, 2]
    return point


def sample_bezier_batch(cubics, k=5):
    """
    Sample points on cubic Bezier curves.
    :param cubics: torch.Tensor, shape [batch_size, num_curves, 4, 2], representing cubic Bezier curves.
    :param k: int, number of sample points per curve.
    :return: torch.Tensor, shape [batch_size, num_curves * k, 2], representing the sampled points on the Bezier curves.
    """
    batch_size, num_curves = cubics.shape[0], cubics.shape[1]

    # shape [1, 1, k, 1]
    ts = torch.linspace(0, 1, k).view(1, 1, k, 1).to(cubics.device)
    t_inv = 1 - ts

    # Break down the cubics tensor for cubic bezier formula
    P0, P1, P2, P3 = cubics[:, :, 0], cubics[:,
                                             :, 1], cubics[:, :, 2], cubics[:, :, 3]

    # Expand dimensions of P0, P1, P2, P3 for broadcasting with ts
    # shape [batch_size, num_curves, 1, 2]
    P0, P1, P2, P3 = P0[:, :, None, :], P1[:, :,
                                           None, :], P2[:, :, None, :], P3[:, :, None, :]

    # Using the cubic bezier formula
    sampled_points = t_inv**3 * P0 + 3 * t_inv**2 * \
        ts * P1 + 3 * t_inv * ts**2 * P2 + ts**3 * P3

    # Reshape to [batch_size, num_curves * k, 2]
    sampled_points = sampled_points.reshape(batch_size, num_curves * k, 2)

    return sampled_points


def load_init_circle_cubics(circle_svg_fp="../vae_dataset/circle_10.svg", transform=None):

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        circle_svg_fp)

    for path in shapes:
        points = path.points
        num_control_points = path.num_control_points
        break

    # Transform points if applicable
    if transform:
        points = transform(points)

    # Truncate if sequence is too long
    len_points = points.shape[0]

    # Compute the cubics segments
    cubics = get_cubic_segments_from_points(
        points=points, num_control_points=num_control_points)

    desired_cubics_length = len_points // 3

    assert cubics.shape[0] == desired_cubics_length

    return cubics


# ----------------------------------------
def split_train_test(file_list, train_ratio=0.9):
    # split train and test
    random.shuffle(file_list)
    len_train = int(len(file_list)*train_ratio)
    file_list_train = file_list[:len_train]
    file_list_test = file_list[len_train:]

    return {
        "file_list_train": file_list_train,
        "file_list_test": file_list_test
    }


def save_filelist_csv(file_list, file_list_fp):
    file_list_df = pd.DataFrame(file_list, columns=["file_name"])
    file_list_df.to_csv(file_list_fp, index=False)


def split_trte_cmd_pts(svg_meta_fp, max_cmd_len_thresh=16, max_pts_len_thresh=53):
    # read meta file and calculate average length
    meta_data = pd.read_csv(svg_meta_fp)

    max_len_group_np = meta_data["max_len_group"]
    num_pts_np = meta_data["num_pts"]
    avg_len = max_len_group_np.mean()
    max_len = max_len_group_np.max()
    min_len = max_len_group_np.min()

    cond_max_len = (max_len_group_np < max_cmd_len_thresh) & (
        num_pts_np < max_pts_len_thresh + 10)
    filtered_data = meta_data[cond_max_len]

    # If you want to extract the "id" column from the filtered data and add ".svg" extension:
    remove_long_file_list = (filtered_data["id"] + ".svg").tolist()

    # split train and test
    random.shuffle(remove_long_file_list)
    len_train = int(len(remove_long_file_list)*0.9)
    remove_long_file_list_train = remove_long_file_list[:len_train]
    remove_long_file_list_test = remove_long_file_list[len_train:]

    # save remove_long_file_list_train to csv
    remove_long_file_list_train_df = pd.DataFrame(
        remove_long_file_list_train, columns=["file_name"])
    remove_long_file_list_train_df.to_csv(
        "./dataset/file_list_train.csv", index=False)

    remove_long_file_list_test_df = pd.DataFrame(
        remove_long_file_list_test, columns=["file_name"])
    remove_long_file_list_test_df.to_csv(
        "./dataset/file_list_test.csv", index=False)


def split_trte_pts(svg_meta_fp, svg_data_dir, file_list_train_fp, file_list_test_fp, svg_data_img_dir="", max_pts_len_thresh=53, min_area_thresh=3000):
    # read meta file and calculate average length
    meta_data = pd.read_csv(svg_meta_fp)

    max_len_group_np = meta_data["max_len_group"]
    num_pts_np = meta_data["num_pts"]
    area_np = meta_data["area"]

    avg_len = num_pts_np.mean()
    max_len = num_pts_np.max()
    min_len = num_pts_np.min()

    avg_area = area_np.mean()
    max_area = area_np.max()
    min_area = area_np.min()

    # ----------------------------------------
    bins = np.arange(min_len, min_len + 400, 15)
    counts, edges = np.histogram(num_pts_np, bins=bins)
    sum_counts = np.sum(counts)

    small_svg_list_trte = []
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1] - 1
        print(
            f"Processing Lengths between {bin_min} and {bin_max}: {counts[i]}")

        if counts[i] == 0:  # Skip empty bins
            continue

        # Create a new directory for the bin
        new_dir = svg_data_dir[:-1] + "_" + str(int(bin_max)) + "/"
        if (os.path.exists(new_dir)):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir, exist_ok=True)

        new_img_dir = svg_data_dir[:-1] + "_" + str(int(bin_max)) + "_img/"
        if (os.path.exists(new_img_dir)):
            shutil.rmtree(new_img_dir)
        os.makedirs(new_img_dir, exist_ok=True)

        # delete new_dir
        shutil.rmtree(new_dir)
        shutil.rmtree(new_img_dir)
        # continue

        # Filter the data frame to contain only rows within the bin range
        filtered_data = meta_data[(num_pts_np >= bin_min) & (
            num_pts_np < bins[i + 1])]

        # Randomly sample up to 100 rows from the filtered data frame
        # sample_size = min(200, len(filtered_data))
        sample_size = len(filtered_data)
        st = 0 * sample_size
        ed = st + sample_size
        sampled_data = filtered_data[st:ed]
        # sampled_data = filtered_data.sample(n=sample_size, random_state=1)

        # Extract the "id" column from the sampled data and append ".svg" extension
        sampled_files = (sampled_data["id"] + ".svg").tolist()

        # Copy sampled files to the new directory
        for file in sampled_files:
            source_file = os.path.join(svg_data_dir, file)
            file_pre = file.split(".")[0]
            tmp_img_fp = os.path.join(svg_data_img_dir, file_pre + ".png")

            if os.path.exists(source_file) and os.path.exists(tmp_img_fp):
                destination_file = os.path.join(new_dir, file)
                # shutil.copy(source_file, destination_file)
                destination_file = os.path.join(new_img_dir, file_pre + ".png")
                # shutil.copy(tmp_img_fp, destination_file)
                small_svg_list_trte.append(file)

    # split train and test
    sep_res = split_train_test(small_svg_list_trte)
    file_list_train = sep_res["file_list_train"]
    file_list_test = sep_res["file_list_test"]

    # save remove_long_file_list_train to csv
    bin_ind = str(int(bins[1] - 1))
    file_list_train_img_fp = file_list_train_fp.replace(
        ".csv", "_" + bin_ind + ".csv")
    # save_filelist_csv(file_list_train, file_list_train_img_fp)
    file_list_test_img_fp = file_list_test_fp.replace(
        ".csv", "_" + bin_ind + ".csv")
    # save_filelist_csv(file_list_test, file_list_test_img_fp)

    log_data = np.log1p(num_pts_np)
    counts, bins, patches = plt.hist(
        log_data, bins=30, edgecolor='k', alpha=0.65)

    # 标题和标签
    plt.title('Histogram of log1p(Lengths)')
    plt.xlabel('log1p(Length)')
    plt.ylabel('Frequency')

    plt.grid(axis='y', alpha=0.75)

    plt.savefig("./num_pts.png")

    # -----------------------------------------------

    cond_max_len = ((num_pts_np < max_pts_len_thresh)
                    & (area_np > min_area_thresh))
    filtered_data = meta_data[cond_max_len]

    # extract the "id" column from the filtered data and add ".svg" extension:
    remove_long_file_list = (filtered_data["id"] + ".svg").tolist()

    existing_remove_long_file_list = []
    for file in remove_long_file_list:
        if os.path.exists(os.path.join(svg_data_dir, file)):
            append_flg = True
            if (len(svg_data_img_dir) > 0):
                file_pre = file.split(".")[0]
                if (not os.path.exists(os.path.join(svg_data_img_dir, file_pre + ".png"))):
                    append_flg = False

            if (append_flg):
                existing_remove_long_file_list.append(file)

    remove_long_file_list = existing_remove_long_file_list

    # split train and test
    sep_res = split_train_test(remove_long_file_list)
    remove_long_file_list_train = sep_res["file_list_train"]
    remove_long_file_list_test = sep_res["file_list_test"]

    # save remove_long_file_list_train to csv
    save_filelist_csv(remove_long_file_list_train, file_list_train_fp)
    save_filelist_csv(remove_long_file_list_test, file_list_test_fp)
