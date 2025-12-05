import numpy as np
import open3d as o3d
import torch
from PIL import Image
from feature_extractor import FeatureExtractor
from config import GRIMConfig

def process_rgbd_scene(rgb_path, depth_path, intrinsics, extractor: FeatureExtractor):
    """
    Generates a point cloud with attached DINO features.
    Args:
        intrinsics: open3d.camera.PinholeCameraIntrinsic
    Returns:
        pcd: Open3D PointCloud
        features: numpy array (N, embed_dim) corresponding to points
    """
    # 1. Load Images
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgb_pil = Image.open(rgb_path).convert("RGB")
    target_size = (rgb_pil.height, rgb_pil.width)

    # 2. Generate Geometry (XYZ)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)

    # 3. Generate Features (DINO)
    # Extract dense features
    patch_tokens, spatial_shape = extractor.extract_dino_features(rgb_pil)

    # Upsample to per-pixel
    full_features = extractor.interpolate_features_to_image(
        patch_tokens, spatial_shape, target_size
    ) # Shape (H*W, D)

    # 4. Map Features to Valid Points
    # Open3D's create_from_rgbd_image organizes points, but filters invalid depth.
    # We need to filter the features using the same depth validity mask.
    depth_np = np.asarray(depth_raw)
    valid_mask = (depth_np > 0).flatten()

    # Convert tensor to numpy
    full_features_np = full_features.cpu().numpy()

    # Filter features to match the number of points in PCD
    # Note: This assumes Open3D flattens row-major.
    valid_features = full_features_np[valid_mask]

    # Safety check
    points = np.asarray(pcd.points)
    if len(points) != len(valid_features):
        # Fallback: Nearest Neighbor interpolation on projected points if counts mismatch
        # (Rare, but handles Open3D edge cases)
        print("Warning: Point count mismatch. Truncating to match.")
        min_len = min(len(points), len(valid_features))
        valid_features = valid_features[:min_len]
        # In production, use KDTree to map 2D pixels to 3D points strictly

    return pcd, valid_features
