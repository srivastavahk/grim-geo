import clip
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation


class GRIMPerception:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[Perception] Loading models on {device}...")

        # 1. Monocular Depth (Replaces Depth Sensor / MCC-HO)
        # Using Depth Anything or DPT for high quality depth from single RGB
        self.depth_processor = AutoImageProcessor.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        ).to(device)

        # [cite_start]2. DINOv2 (Visual Features) [cite: 125, 439]
        self.dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small-patch14-224"
        )
        self.dino_model = AutoModel.from_pretrained(
            "facebook/dinov2-small-patch14-224"
        ).to(device)

        # [cite_start]3. CLIP (Task Semantics) [cite: 23, 143]
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def rgb_to_pointcloud(self, rgb_image_np):
        """
        Converts single RGB image to 3D Point Cloud using Monocular Depth.
        Returns: o3d.geometry.PointCloud
        """
        image_pil = Image.fromarray(rgb_image_np)

        # Estimate Depth
        inputs = self.depth_processor(images=image_pil, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        depth_map = prediction.squeeze().cpu().numpy()

        # Normalize depth for visualization/processing (Scale is arbitrary in MonoDepth)
        # We assume a working range of 0.3m to 1.0m for manipulation
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)  # 0 to 1
        depth_map = depth_map * 0.7 + 0.3  # Scale to 0.3m - 1.0m metric approx

        # Reproject to 3D
        h, w = depth_map.shape
        # Intrinsic Matrix (Approximate for Stretch Camera if not calibrated)
        fx, fy = 600, 600
        cx, cy = w / 2, h / 2

        # Vectorized projection
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xv, yv = np.meshgrid(x, y)

        z_c = depth_map
        x_c = (xv - cx) * z_c / fx
        y_c = (yv - cy) * z_c / fy

        points = np.stack((x_c, y_c, z_c), axis=-1).reshape(-1, 3)
        colors = rgb_image_np.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Downsample for performance
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        return pcd

    def extract_dino_features(self, rgb_image_np, pcd):
        """
        Projects 2D DINO features onto the 3D Point Cloud.
        """
        image_pil = Image.fromarray(rgb_image_np)
        inputs = self.dino_processor(images=image_pil, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # Extract features (exclude CLS token)
        # Shape: (1, Num_Patches, Dim) -> (Dim, H_grid, W_grid)
        last_hidden = outputs.last_hidden_state
        patch_tokens = last_hidden[0, 1:, :]

        h_in, w_in = inputs["pixel_values"].shape[2:]  # Usually 224x224
        grid_size = 14  # Patch size
        h_grid, w_grid = h_in // grid_size, w_in // grid_size

        feat_map = patch_tokens.transpose(0, 1).view(1, -1, h_grid, w_grid)

        # Project to Points (UV Mapping)
        # For simplicity in this script, we assume the PCD was generated
        # from the SAME image view. We can map directly via image coordinates.
        pts = np.asarray(pcd.points)

        # Need to project points back to UV to sample features
        # Since we generated points from pixels (x,y), we can just carry that mapping index
        # But `voxel_down_sample` breaks the order.
        # Re-projection logic:
        fx, fy = 600, 600
        cx, cy = rgb_image_np.shape[1] / 2, rgb_image_np.shape[0] / 2

        u = (pts[:, 0] * fx / pts[:, 2]) + cx
        v = (pts[:, 1] * fy / pts[:, 2]) + cy

        # Normalize u,v to [-1, 1] for grid_sample
        u_norm = 2 * (u / (rgb_image_np.shape[1] - 1)) - 1
        v_norm = 2 * (v / (rgb_image_np.shape[0] - 1)) - 1

        grid = torch.tensor(np.stack((u_norm, v_norm), axis=1), dtype=torch.float32).to(
            self.device
        )
        grid = grid.view(1, 1, -1, 2)

        # Sample features
        sampled_feats = F.grid_sample(
            feat_map, grid, align_corners=False, mode="bilinear"
        )
        sampled_feats = sampled_feats.squeeze().permute(1, 0).cpu().numpy()

        # [cite_start]L2 Normalize [cite: 424]
        norm = np.linalg.norm(sampled_feats, axis=1, keepdims=True)
        return sampled_feats / (norm + 1e-6)

    def encode_task(self, text):
        tokenized = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokenized)
        return emb.cpu().numpy().flatten()
