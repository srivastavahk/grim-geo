import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from config import GRIMConfig
import copy

class AlignmentModule:
    """
    Implements:
    1. PCA Reduction of Features [cite: 150]
    2. Coarse Alignment (Grid Search) [cite: 153]
    3. Feature-Guided Scoring (Eq 4/8) [cite: 450]
    4. Fine Refinement (ICP) [cite: 25, 171]
    """

    def __init__(self):
        pass

    def reduce_features(self, source_feats, target_feats):
        """
        Projects DINO features to lower dimension (e.g. 3) using PCA.
        [cite: 150]
        """
        pca = PCA(n_components=GRIMConfig.PCA_COMPONENTS)
        # Fit on concatenation to share subspace
        combined = np.vstack((source_feats, target_feats))
        pca.fit(combined)

        source_pca = pca.transform(source_feats)
        target_pca = pca.transform(target_feats)

        # Normalize for cosine similarity calculation later
        source_pca = source_pca / np.linalg.norm(source_pca, axis=1, keepdims=True)
        target_pca = target_pca / np.linalg.norm(target_pca, axis=1, keepdims=True)

        return source_pca, target_pca

    def compute_hybrid_cost(self, source_pts, target_pts, source_feats, target_feats, transform):
        """
        Calculates C_pair = w_g * geom_dist + w_f * feat_dist
        Eq 4 / Eq 8 [cite: 168, 450]
        """
        # Apply transform to source points
        # Transform is 4x4
        R = transform[:3, :3]
        t = transform[:3, 3]
        transformed_source = (source_pts @ R.T) + t

        # Find nearest neighbors in Target
        nbrs = NearestNeighbors(n_neighbors=GRIMConfig.K_EVAL_NEIGHBORS, algorithm='auto').fit(target_pts)
        distances, indices = nbrs.kneighbors(transformed_source)

        # We average over the K neighbors for each source point
        # Geometric cost: L2 squared distance
        cost_geo = np.mean(distances ** 2)

        # Feature cost: 1 - Cosine Similarity
        # Gather target features of neighbors
        # indices shape: (N_source, K)
        # target_feats shape: (N_target, D)
        neighbor_feats = target_feats[indices] # (N_source, K, D)

        # Expand source feats for broadcasting
        # source_feats shape: (N_source, D) -> (N_source, 1, D)
        source_feats_expanded = source_feats[:, np.newaxis, :]

        # Cosine similarity (already normalized in reduce_features)
        # Dot product over last axis
        cos_sim = np.sum(source_feats_expanded * neighbor_feats, axis=2) # (N_source, K)
        cost_feat = np.mean(1.0 - cos_sim)

        # Total Weighted Cost
        total_cost = (GRIMConfig.W_G * cost_geo) + (GRIMConfig.W_F * cost_feat)
        return total_cost

    def align(self, source_pcd, target_pcd, source_feats, target_feats):
        # 0. Data Prep
        source_pts = np.asarray(source_pcd.points)
        target_pts = np.asarray(target_pcd.points)

        # 1. PCA Reduction [cite: 150]
        src_feats_pca, tgt_feats_pca = self.reduce_features(source_feats, target_feats)

        # 2. Coarse Alignment: Centroid + Grid Search [cite: 153]
        c_source = np.mean(source_pts, axis=0)
        c_target = np.mean(target_pts, axis=0)

        # Scale factor (simplified based on bounding box diagonal ratio)
        # Paper mentions eigenvalues, diagonal is a robust approximation
        s_g = np.linalg.norm(np.max(target_pts, 0) - np.min(target_pts, 0)) / \
              np.linalg.norm(np.max(source_pts, 0) - np.min(source_pts, 0))

        candidates = []

        # Grid Search Euler Angles
        step = GRIMConfig.GRID_SEARCH_ANGLE_STEP
        angles = np.arange(0, 360, step)

        print(f"Running Coarse Grid Search ({len(angles)**3} orientations)...")

        # Optimized loop
        # Pre-center points
        src_centered = source_pts - c_source

        # Generate Rotations
        # In production, use scipy's Rotation.from_euler for batch processing if needed
        # but nested loops are acceptable for CPU given the sparsity of angles
        for ax in angles:
            for ay in angles:
                for az in angles:
                    r = Rotation.from_euler('xyz', [ax, ay, az], degrees=True)
                    R_mat = r.as_matrix()

                    # Construct Transform T_init (Eq 3)
                    # p' = s * R * (p - c_m) + c_s
                    # T = [sR | c_s - sR*c_m]
                    t_vec = c_target - s_g * (R_mat @ c_source)

                    T_candidate = np.eye(4)
                    T_candidate[:3, :3] = s_g * R_mat
                    T_candidate[:3, 3] = t_vec

                    # Calculate Score
                    score = self.compute_hybrid_cost(
                        source_pts, target_pts, src_feats_pca, tgt_feats_pca, T_candidate
                    )
                    candidates.append((score, T_candidate))

        # Sort by score (ascending, lower cost is better)
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:GRIMConfig.K_ORIENT_CANDIDATES] # [cite: 171]

        # 3. Fine Refinement (ICP)
        # "followed by fine-grained refinement using the classical ICP" [cite: 25]
        # We run Standard ICP on the top geometric candidates, then re-rank by Hybrid Score.

        best_final_score = float('inf')
        best_final_transform = np.eye(4)

        print(f"Refining Top {len(top_candidates)} candidates with ICP...")

        for _, T_init in top_candidates:
            # Prepare Open3D ICP
            curr_source = copy.deepcopy(source_pcd)
            curr_source.transform(T_init)

            # Standard Geometric ICP (Point-to-Point)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                curr_source, target_pcd, max_correspondence_distance=0.05,
                init=np.eye(4), # Already transformed
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # Combine transformations: T_final = T_icp * T_init
            T_refined = reg_p2p.transformation @ T_init

            # Re-Evaluate using Hybrid Cost (Eq 8) [cite: 174]
            final_score = self.compute_hybrid_cost(
                source_pts, target_pts, src_feats_pca, tgt_feats_pca, T_refined
            )

            if final_score < best_final_score:
                best_final_score = final_score
                best_final_transform = T_refined

        return best_final_transform
