import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from transformers import AutoImageProcessor, AutoModel

# --- Configuration Constants based on Paper ---
WEIGHT_TASK = 0.95
WEIGHT_GEO = 0.05
SIGMA_POS = 0.1  #  Scaling factor for Gaussian decay
K_EVAL = 5  #  Nearest neighbors for alignment scoring
DINO_DIM = 384  #  Standard DINOv2-small dimension
PCA_DIM = 3  #  Reduced dimension for coarse alignment


# ==============================================================================
# 0. DINO Feature Extractor
# ==============================================================================
class DINOFeatureExtractor:
    def __init__(self, model_name="facebook/dinov2-small-patch14-224", device="cuda"):
        """
        Initializes the DINOv2 model.
        The paper mentions using 'dinov2-small' for alignment loss
        and 'dinov2-vitl14' for feature-rich point clouds.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[DINOv2] Loading {model_name} on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Standard DINOv2 patch size is usually 14
        self.patch_size = 14

    def extract_image_features(
        self, image_path: str
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Passes the image through the ViT and returns the spatial feature map.
        """
        image = Image.open(image_path).convert("RGB")

        # Preprocess image (Resize, Normalize)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # The output 'last_hidden_state' has shape (Batch, Seq_Len, Dim)
        # Seq_Len includes the CLS token (index 0) + Patch tokens
        last_hidden_states = outputs.last_hidden_state

        # Remove CLS token (index 0) to keep only spatial patches
        patch_embeddings = last_hidden_states[0, 1:, :]  # Shape: (Num_Patches, Dim)

        # Calculate the grid dimensions (H_grid, W_grid)
        # The processor resizes images to a fixed size (e.g., 224x224 or similar)
        # We need to know the actual input size passed to the model to reshape correctly
        input_h, input_w = (
            inputs["pixel_values"].shape[2],
            inputs["pixel_values"].shape[3],
        )
        grid_h = input_h // self.patch_size
        grid_w = input_w // self.patch_size

        # Reshape to (Dim, H_grid, W_grid) for easier sampling
        feature_map = patch_embeddings.transpose(0, 1).view(-1, grid_h, grid_w)

        return feature_map, (input_h, input_w)

    def project_features_to_points(
        self,
        points_3d: np.ndarray,
        feature_map: torch.Tensor,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        img_dim: Tuple[int, int],
    ) -> np.ndarray:
        """
        Maps 2D features to 3D points.
        GRIM creates a "feature mesh" by associating embeddings with vertices.

        Args:
            points_3d: (N, 3) numpy array of point cloud coordinates.
            feature_map: (Dim, H_grid, W_grid) tensor from DINO.
            camera_intrinsics: (3, 3) K matrix.
            camera_extrinsics: (4, 4) World-to-Camera matrix (RT).
            img_dim: (H, W) of the image input to DINO.

        Returns:
            (N, Dim) numpy array of features for each 3D point.
        """
        N = len(points_3d)

        # 1. Convert 3D points to Camera Coordinate System
        # Append 1 for homogeneous coords: (N, 4)
        ones = np.ones((N, 1))
        points_hom = np.hstack((points_3d, ones))

        # Transform: P_cam = T * P_world
        points_cam = (camera_extrinsics @ points_hom.T).T  # (N, 4)
        points_cam = points_cam[:, :3]  # Drop 4th dim

        # 2. Project to Image Plane (Pixels)
        # P_uv = K * P_cam
        points_uv = (camera_intrinsics @ points_cam.T).T  # (N, 3)

        # Normalize by Z (depth) to get u, v
        u = points_uv[:, 0] / (points_uv[:, 2] + 1e-8)
        v = points_uv[:, 1] / (points_uv[:, 2] + 1e-8)

        # 3. Normalize u, v to [-1, 1] for torch.grid_sample
        # This aligns with the feature map grid
        H, W = img_dim
        u_norm = 2 * (u / (W - 1)) - 1
        v_norm = 2 * (v / (H - 1)) - 1

        # Stack into grid shape (1, 1, N, 2) for sampling
        grid = torch.tensor(np.stack((u_norm, v_norm), axis=1), dtype=torch.float32).to(
            self.device
        )
        grid = grid.view(1, 1, N, 2)

        # 4. Bilinear Interpolation (Sampling)
        # Expand feature_map to (1, Dim, H_grid, W_grid)
        feat_batch = feature_map.unsqueeze(0)

        # Sample: Output shape (1, Dim, 1, N)
        sampled_feats = F.grid_sample(
            feat_batch, grid, align_corners=False, mode="bilinear"
        )

        # Reshape to (N, Dim)
        sampled_feats = sampled_feats.squeeze().permute(1, 0).cpu().numpy()

        # Normalize features (Cosine Similarity requires normalized vectors)
        norms = np.linalg.norm(sampled_feats, axis=1, keepdims=True)
        sampled_feats = sampled_feats / (norms + 1e-8)

        return sampled_feats


# ==============================================================================
# Usage Example (Replacing the Mock in Previous Code)
# ==============================================================================
if __name__ == "__main__":
    # Initialize Extractor
    extractor = DINOFeatureExtractor(model_name="facebook/dinov2-small-patch14-224")

    # 1. Load Image and Get Features
    # Make sure you have a dummy image or change path
    # feature_map, dims = extractor.extract_image_features("test_mug.jpg")

    # --- Simulation for code runnable without an image file ---
    print("Simulating feature extraction output...")
    dim = 384  # DINOv2-small dimension
    h_grid, w_grid = 16, 16  # 224 / 14 = 16
    simulated_map = torch.randn(dim, h_grid, w_grid).to(extractor.device)
    simulated_dims = (224, 224)
    # --------------------------------------------------------

    # 2. Define Mock Camera Matrices (Identity for example)
    K = np.array([[500, 0, 112], [0, 500, 112], [0, 0, 1]])  # Simple Intrinsic
    RT = np.eye(4)  # Simple Extrinsic

    # 3. Define Mock 3D Points (The 'Scene Object')
    # In the full pipeline, this comes from 'scene_pcd'
    points = np.random.rand(100, 3)

    # 4. Project Features onto Points
    point_features = extractor.project_features_to_points(
        points, simulated_map, K, RT, simulated_dims
    )

    print(f"Successfully extracted DINO features for {len(points)} points.")
    print(f"Feature Shape: {point_features.shape}")  # Should be (100, 384)

    # 5. Calculate Global Descriptor (for Retrieval Step)
    global_descriptor = np.mean(point_features, axis=0)
    print(f"Global Descriptor Shape: {global_descriptor.shape}")  # Should be (384,)


# ==============================================================================
# 1. Memory Data Structure & Creation Pipeline
# ==============================================================================


@dataclass
class MemoryInstance:
    """
    Stores a single memory experience as defined in Eq. 1.
    M = {(F_M, G, T, O)}
    """

    object_name: str
    task_name: str
    point_cloud: o3d.geometry.PointCloud
    dino_features: np.ndarray  # Shape (N, D)
    global_descriptor: np.ndarray  # Shape (D,) - Mean of dino_features
    grasp_pose: np.ndarray  # Shape (4, 4) Homogeneous Matrix
    task_embedding: np.ndarray  # Shape (E,) - CLIP embedding


class MemoryCreator:
    """
    Pipeline for creating memory instances.
    Since we cannot run Veo2/Gemini/MCC-HO here, these are interfaces.
    """

    def __init__(self):
        pass

    def create_mock_instance(self, obj_name: str, task_name: str) -> MemoryInstance:
        """
        Generates synthetic data for testing the pipeline.
        """
        # 1. Generate random point cloud (Sphere-like)
        pcd = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.1
        ).sample_points_uniformly(number_of_points=500)
        points = np.asarray(pcd.points)

        # 2. Generate synthetic DINO features (Random vectors normalized)
        extractor = DINOFeatureExtractor()
        feat_map, dims = extractor.extract_image_features("path_to_object_image.jpg")
        features = extractor.project_features_to_points(points, feat_map, K, RT, dims)
        # features = np.random.rand(len(points), DINO_DIM)
        # features = features / np.linalg.norm(features, axis=1, keepdims=True)
        global_desc = np.mean(features, axis=0)

        # 3. Generate synthetic Task Embedding (CLIP style)
        task_emb = np.random.rand(512)
        task_emb = task_emb / np.linalg.norm(task_emb)

        # 4. specific Grasp Pose (Identity for simplicity, offset in Z)
        grasp = np.eye(4)
        grasp[2, 3] = 0.15  # 15cm offset in Z

        return MemoryInstance(
            object_name=obj_name,
            task_name=task_name,
            point_cloud=pcd,
            dino_features=features,
            global_descriptor=global_descriptor_norm(global_desc),
            grasp_pose=grasp,
            task_embedding=task_emb,
        )


def global_descriptor_norm(desc):
    return desc / np.linalg.norm(desc)


# ==============================================================================
# 2. Memory Retrieval Module
# ==============================================================================


class MemoryBank:
    def __init__(self):
        self.memory: List[MemoryInstance] = []

    def add(self, instance: MemoryInstance):
        self.memory.append(instance)

    def retrieve(
        self, scene_global_desc: np.ndarray, scene_task_emb: np.ndarray
    ) -> Tuple[MemoryInstance, float]:
        """
        Implements Eq. 2: Joint Similarity.
        S_joint(i,j) = sim_cos(F_SO, F_MO) * sim_cos(E_TS, E_TM)
        """
        best_score = -1.0
        best_instance = None

        # Normalize inputs
        scene_desc_norm = scene_global_desc / np.linalg.norm(scene_global_desc)
        scene_task_norm = scene_task_emb / np.linalg.norm(scene_task_emb)

        for instance in self.memory:
            # Visual Similarity
            vis_sim = np.dot(scene_desc_norm, instance.global_descriptor)

            # Semantic (Task) Similarity
            task_sim = np.dot(scene_task_norm, instance.task_embedding)

            # Joint Score
            joint_score = vis_sim * task_sim

            if joint_score > best_score:
                best_score = joint_score
                best_instance = instance

        print(
            f"[Retrieval] Best Joint Score: {best_score:.4f} (Obj: {best_instance.object_name}, Task: {best_instance.task_name})"
        )
        return best_instance, best_score


# ==============================================================================
# 3. Alignment Module (The Core Logic)
# ==============================================================================


class GRIMAligner:
    def __init__(self, w_g: float = 1.0, w_f: float = 1.0):
        self.w_g = w_g  # Geometric weight
        self.w_f = w_f  # Feature weight

    def reduce_features(
        self, source_feats: np.ndarray, target_feats: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects DINO features to lower dimensions using PCA for alignment.
        """
        # Combine to learn a shared latent space
        combined = np.vstack((source_feats, target_feats))
        pca = PCA(n_components=PCA_DIM)
        transformed = pca.fit_transform(combined)

        return transformed[: len(source_feats)], transformed[len(source_feats) :]

    def compute_hybrid_cost(
        self,
        src_points: np.ndarray,
        src_feats_pca: np.ndarray,
        tgt_tree: NearestNeighbors,
        tgt_points: np.ndarray,
        tgt_feats_pca: np.ndarray,
    ) -> float:
        """
        Implements Eq. 4:
        C_pair = w_g * ||dist||^2 + w_f * (1 - cos_sim)
        """
        # Find K nearest geometric neighbors
        distances, indices = tgt_tree.kneighbors(src_points)

        # Geometric Cost (Mean Squared Distance)
        cost_geo = np.mean(distances**2)

        # Feature Cost (Cosine Distance)
        # Get features of the neighbors
        neighbor_feats = tgt_feats_pca[indices]  # (N, K, Dim)

        # Expand source feats for broadcasting: (N, 1, Dim)
        src_feats_expanded = np.expand_dims(src_feats_pca, axis=1)

        # Compute Cosine Similarity manually for broadcasting support
        # Assuming PCA features are not pre-normalized
        dot_prod = np.sum(src_feats_expanded * neighbor_feats, axis=2)
        norm_src = np.linalg.norm(src_feats_expanded, axis=2)
        norm_neighbors = np.linalg.norm(neighbor_feats, axis=2)

        cosine_sim = dot_prod / (norm_src * norm_neighbors + 1e-8)
        feature_dist = 1.0 - cosine_sim
        cost_feat = np.mean(feature_dist)

        total_cost = (self.w_g * cost_geo) + (self.w_f * cost_feat)
        return total_cost

    def align(
        self,
        mem_pcd: o3d.geometry.PointCloud,
        mem_feats: np.ndarray,
        scene_pcd: o3d.geometry.PointCloud,
        scene_feats: np.ndarray,
    ) -> np.ndarray:
        """
        Full Alignment Pipeline: PCA -> Coarse (Grid) -> Fine (ICP).
        """
        # 1. PCA Reduction
        mem_feats_pca, scene_feats_pca = self.reduce_features(mem_feats, scene_feats)

        # Prepare Data
        src_pts = np.asarray(mem_pcd.points)
        tgt_pts = np.asarray(scene_pcd.points)

        # 2. Pre-computation: Centroids and Scale
        c_mem = np.mean(src_pts, axis=0)
        c_scene = np.mean(tgt_pts, axis=0)

        # Simplified Scale estimation (Ratio of average distance to centroid)
        d_mem = np.mean(np.linalg.norm(src_pts - c_mem, axis=1))
        d_scene = np.mean(np.linalg.norm(tgt_pts - c_scene, axis=1))
        scale_factor = d_scene / d_mem

        # Build KDTree for Scene (Target) for fast neighbor lookup
        nbrs = NearestNeighbors(n_neighbors=K_EVAL, algorithm="auto").fit(tgt_pts)

        # 3. Coarse Alignment: Grid Search over Rotations
        best_coarse_transform = np.eye(4)
        min_cost = float("inf")

        # Define a simple grid of Euler angles (0, 90, 180, 270) for X, Y, Z
        # In a real scenario, use a Fibonacci sphere or finer grid.
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        print("[Alignment] Starting Coarse Grid Search...")
        for rx in angles:
            for ry in angles:
                for rz in angles:
                    R = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))

                    # Construct Transform T_init
                    # T(p) = s * R * (p - c_mem) + c_scene
                    # We apply this to points manually to score
                    transformed_pts = scale_factor * (src_pts - c_mem) @ R.T + c_scene

                    cost = self.compute_hybrid_cost(
                        transformed_pts, mem_feats_pca, nbrs, tgt_pts, scene_feats_pca
                    )

                    if cost < min_cost:
                        min_cost = cost
                        # Build 4x4 Matrix
                        T = np.eye(4)
                        T[:3, :3] = R * scale_factor
                        # Translation component derived from centroid alignment
                        T[:3, 3] = c_scene - (scale_factor * R @ c_mem)
                        best_coarse_transform = T

        print(f"[Alignment] Coarse Best Cost: {min_cost:.4f}")

        # 4. Fine Refinement (ICP)
        # We use Open3D's point-to-point ICP, initialized with coarse result
        print("[Alignment] Running Fine ICP Refinement...")

        # Note: Standard ICP is purely geometric. The paper mentions "Refined poses... re-evaluated using combined score".
        # Implementation: We use geometric ICP to tighten the fit, then return that transform.

        result_icp = o3d.pipelines.registration.registration_icp(
            mem_pcd,
            scene_pcd,
            0.02,
            best_coarse_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        print(
            f"[Alignment] Final Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.in_rmse:.4f}"
        )
        return result_icp.transformation


# ==============================================================================
# 4. Grasp Transfer & Evaluation
# ==============================================================================


class GraspManager:
    def transfer_grasp(
        self, memory_grasp: np.ndarray, alignment_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Transforms the memory grasp to the scene frame.
        G_S = T_final * G_M
        """
        return alignment_matrix @ memory_grasp

    def score_candidates(
        self, target_grasp: np.ndarray, candidate_grasps: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluates candidate grasps against the transferred memory grasp intent.
        Implements Eq. 5 and Eq. 6.

        Args:
            target_grasp: The transferred G_S (4x4).
            candidate_grasps: List of (G_A (4x4), geometric_score).

        Returns:
            Best grasp pose (4x4) and its score.
        """
        # Extract Target Attributes
        # Assume Z-axis (col 2) is approach direction
        v_target = target_grasp[:3, 2]
        p_target = target_grasp[:3, 3]

        best_score = -float("inf")
        best_grasp = None

        for G_A, s_geo in candidate_grasps:
            # Extract Candidate Attributes
            o_z = G_A[:3, 2]  # Approach direction
            t_A = G_A[:3, 3]  # Position

            # Eq. 5: Task Compatibility Score
            # Cosine similarity of approach vectors
            dir_score = np.dot(v_target, o_z) / (
                np.linalg.norm(v_target) * np.linalg.norm(o_z)
            )

            # Gaussian position decay
            pos_dist_sq = np.sum((t_A - p_target) ** 2)
            pos_score = np.exp(-pos_dist_sq / (2 * (SIGMA_POS**2)))

            s_task = dir_score + pos_score

            # Eq. 6: Final Combined Score
            final_score = (WEIGHT_TASK * s_task) + (WEIGHT_GEO * s_geo)

            if final_score > best_score:
                best_score = final_score
                best_grasp = G_A

        return best_grasp, best_score


def mock_anygrasp_sampler(
    scene_pcd_center, num_grasps=10
) -> List[Tuple[np.ndarray, float]]:
    """
    Placeholder for AnyGrasp.
    Generates random valid rotation matrices around the object center.
    """
    candidates = []
    for _ in range(num_grasps):
        # Random rotation
        R = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * np.pi)
        # Position near center with slight noise
        t = scene_pcd_center + (np.random.rand(3) - 0.5) * 0.05

        G = np.eye(4)
        G[:3, :3] = R
        G[:3, 3] = t

        geo_score = np.random.rand()  # Random stability score 0-1
        candidates.append((G, geo_score))
    return candidates


# ==============================================================================
# 5. Visualization Tools
# ==============================================================================


class GRIMVisualizer:
    @staticmethod
    def draw_registration(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow (Memory)
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue (Scene)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries(
            [source_temp, target_temp], window_name="GRIM: Alignment Result"
        )

    @staticmethod
    def draw_grasp(pcd, grasp_pose, title="GRIM: Final Grasp"):
        """
        Draws the object and the gripper coordinate frame.
        """
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        mesh_frame.transform(grasp_pose)
        o3d.visualization.draw_geometries([pcd, mesh_frame], window_name=title)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    print("=== initializing GRIM Framework ===")

    # 1. Setup Mock Memory
    creator = MemoryCreator()
    bank = MemoryBank()

    # Create a memory: A "Mug" used for "Drinking"
    mem_mug = creator.create_mock_instance("Mug", "Drink")
    bank.add(mem_mug)
    print(f"Memory populated with: {mem_mug.object_name} for task {mem_mug.task_name}")

    # 2. Create a Mock Scene (Novel Object)
    # Let's say the scene has a "Pitcher" (similar to mug) and we want to "Pour"
    # For simulation, we create a slightly deformed sphere
    scene_pcd = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.12
    ).sample_points_uniformly(number_of_points=500)
    # Apply some transformation to make it 'novel' (rotate and translate)
    T_scene_actual = np.eye(4)
    T_scene_actual[:3, 3] = [0.2, 0.0, 0.0]
    scene_pcd.transform(T_scene_actual)

    scene_points = np.asarray(scene_pcd.points)
    scene_dino = np.random.rand(len(scene_points), DINO_DIM)  # Mock features
    scene_global = np.mean(scene_dino, axis=0)
    scene_task_emb = np.random.rand(512)  # Mock "Pour" embedding

    # 3. Retrieval Step
    print("\n--- Step 1: Retrieval ---")
    retrieved_mem, score = bank.retrieve(scene_global, scene_task_emb)

    # 4. Alignment Step
    print("\n--- Step 2: Alignment ---")
    aligner = GRIMAligner(w_g=1.0, w_f=1.0)
    T_final = aligner.align(
        retrieved_mem.point_cloud, retrieved_mem.dino_features, scene_pcd, scene_dino
    )
    print("Alignment Transformation Matrix:\n", T_final)

    # Visualize Alignment
    # Note: In a non-interactive environment, this might block execution until window closed
    # GRIMVisualizer.draw_registration(retrieved_mem.point_cloud, scene_pcd, T_final)

    # 5. Grasp Transfer & Selection
    print("\n--- Step 3: Grasp Transfer & Scoring ---")
    manager = GraspManager()

    # Transfer memory grasp to scene
    G_S = manager.transfer_grasp(retrieved_mem.grasp_pose, T_final)

    # Generate candidates (AnyGrasp placeholder)
    scene_center = np.mean(scene_points, axis=0)
    candidates = mock_anygrasp_sampler(scene_center)

    # Select best grasp
    best_grasp, best_score = manager.score_candidates(G_S, candidates)

    print(f"Best Grasp Score: {best_score:.4f}")
    print("Best Grasp Pose:\n", best_grasp)

    # Visualize Final Grasp
    # GRIMVisualizer.draw_grasp(scene_pcd, best_grasp)
    print("\n=== GRIM Pipeline Completed Successfully ===")
