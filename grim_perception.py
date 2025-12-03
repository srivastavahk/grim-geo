import clip
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class GRIMPerception:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"[Perception] Loading models on {device}...")

        # 1. DINOv2 (Visual Features)
        # We use the 'small' model for speed, but paper suggests 'vitl14' for best results.
        self.dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small-patch14-224"
        )
        self.dino_model = AutoModel.from_pretrained(
            "facebook/dinov2-small-patch14-224"
        ).to(device)

        # 2. CLIP (Task Semantics)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def rgbd_to_pointcloud(self, rgb_image, depth_image, intrinsic_matrix):
        """
        Converts RGB-D images to a colored Point Cloud.
        """
        # Convert to Open3D images
        o3d_rgb = o3d.geometry.Image(rgb_image)
        o3d_depth = o3d.geometry.Image(depth_image)

        # Create RGBD Image
        # Stretch RealSense usually aligns depth to color.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb,
            o3d_depth,
            depth_scale=1000.0,  # RealSense uses mm usually
            depth_trunc=2.0,  # Clip at 2 meters
            convert_rgb_to_intensity=False,
        )

        # Intrinsic Object
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        # Generate Cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

        # Downsample for performance (GRIM relies on feature alignment, not super dense geo)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        # Remove outliers (noise removal)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        return pcd

    def extract_dino_features(self, rgb_image_np, pcd, intrinsic_matrix):
        """
        Projects 2D DINO features onto the 3D Point Cloud (Descriptor Fields).
        """
        image_pil = Image.fromarray(rgb_image_np)
        inputs = self.dino_processor(images=image_pil, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # Extract patch features
        last_hidden = outputs.last_hidden_state
        patch_tokens = last_hidden[0, 1:, :]  # Remove CLS

        # Reshape to feature map
        h_in, w_in = inputs["pixel_values"].shape[2:]
        grid_size = 14
        h_grid, w_grid = h_in // grid_size, w_in // grid_size
        feat_map = patch_tokens.transpose(0, 1).view(
            1, -1, h_grid, w_grid
        )  # (1, Dim, H, W)

        # Project 3D points to 2D UV coordinates to sample features
        pts = np.asarray(pcd.points)
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        # P_uv = K * P_cam
        # Note: Assuming P_cam is already in camera frame (Z forward)
        z = pts[:, 2] + 1e-8
        u = (pts[:, 0] * fx / z) + cx
        v = (pts[:, 1] * fy / z) + cy

        # Normalize u,v to [-1, 1]
        h_img, w_img = rgb_image_np.shape[:2]
        u_norm = 2 * (u / (w_img - 1)) - 1
        v_norm = 2 * (v / (h_img - 1)) - 1

        grid = torch.tensor(np.stack((u_norm, v_norm), axis=1), dtype=torch.float32).to(
            self.device
        )
        grid = grid.view(1, 1, -1, 2)  # (N, 1, Points, 2)

        # Sample
        sampled_feats = F.grid_sample(
            feat_map, grid, align_corners=False, mode="bilinear"
        )
        sampled_feats = sampled_feats.squeeze().permute(1, 0).cpu().numpy()

        # Normalize features
        norm = np.linalg.norm(sampled_feats, axis=1, keepdims=True)
        return sampled_feats / (norm + 1e-6)

    def encode_task(self, text):
        """Encodes text T -> E_T"""
        tokenized = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokenized)
        return emb.cpu().numpy().flatten()
