from dataclasses import dataclass

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Tunable Weights from Paper
WEIGHT_G_ALIGN = 1.0
WEIGHT_F_ALIGN = 1.0
WEIGHT_TASK = 0.95
WEIGHT_GEO = 0.05
SIGMA_POS = 0.1


@dataclass
class MemoryInstance:
    obj_name: str
    task_name: str
    pcd: o3d.geometry.PointCloud
    feats: np.ndarray  # DINO features
    global_desc: np.ndarray  # Mean feature
    task_emb: np.ndarray  # CLIP embedding
    grasp_pose: np.ndarray  # 4x4 Matrix


class GRIMCore:
    def __init__(self):
        self.memory_bank = []
        self.pca = PCA(n_components=3)

    def add_memory(self, instance: MemoryInstance):
        self.memory_bank.append(instance)

    def retrieve(self, scene_global_desc, scene_task_emb):
        """Equation 2: Joint Similarity Score"""
        best_score = -np.inf
        best_mem = None

        s_vis = scene_global_desc / np.linalg.norm(scene_global_desc)
        s_task = scene_task_emb / np.linalg.norm(scene_task_emb)

        for mem in self.memory_bank:
            v_score = np.dot(s_vis, mem.global_desc)
            t_score = np.dot(s_task, mem.task_emb)
            joint_score = v_score * t_score

            if joint_score > best_score:
                best_score = joint_score
                best_mem = mem
        return best_mem, best_score

    def align(self, mem_inst, scene_pcd, scene_feats):
        """Feature-Guided Alignment (Coarse + Fine)"""
        mem_pcd = mem_inst.pcd
        mem_feats = mem_inst.feats

        # 1. PCA Reduction (Fast Matching)
        combined = np.vstack((mem_feats, scene_feats))
        combined_pca = self.pca.fit_transform(combined)
        mem_pca = combined_pca[: len(mem_feats)]
        scene_pca = combined_pca[len(mem_feats) :]

        mem_pts = np.asarray(mem_pcd.points)
        scene_pts = np.asarray(scene_pcd.points)

        # 2. Pre-Alignment (Centroid & Scale)
        c_mem = np.mean(mem_pts, axis=0)
        c_scn = np.mean(scene_pts, axis=0)
        s_g = np.mean(np.linalg.norm(scene_pts - c_scn, axis=1)) / np.mean(
            np.linalg.norm(mem_pts - c_mem, axis=1)
        )

        # 3. Coarse Grid Search
        nbrs = NearestNeighbors(n_neighbors=5).fit(scene_pts)
        best_T = np.eye(4)
        min_cost = np.inf

        # Define rotations (Standard 90 deg steps)
        # For production, use a finer grid or Fibonacci sphere
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        # Downsample for speed
        idx = np.random.choice(len(mem_pts), min(500, len(mem_pts)), replace=False)
        pts_sub = mem_pts[idx]
        feats_sub = mem_pca[idx]

        for rx in angles:
            for ry in angles:
                for rz in angles:
                    R = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))

                    # Transform Memory to Scene
                    p_trans = s_g * (pts_sub - c_mem) @ R.T + c_scn

                    # Compute Cost (Eq 4 / 8)
                    dists, n_idx = nbrs.kneighbors(p_trans)
                    geo_cost = np.mean(dists[:, 0] ** 2)

                    # Feature Cost
                    target_feats = scene_pca[n_idx[:, 0]]
                    # Normalize
                    f1 = feats_sub / (
                        np.linalg.norm(feats_sub, axis=1, keepdims=True) + 1e-6
                    )
                    f2 = target_feats / (
                        np.linalg.norm(target_feats, axis=1, keepdims=True) + 1e-6
                    )
                    feat_cost = np.mean(1.0 - np.sum(f1 * f2, axis=1))

                    total = WEIGHT_G_ALIGN * geo_cost + WEIGHT_F_ALIGN * feat_cost

                    if total < min_cost:
                        min_cost = total
                        T = np.eye(4)
                        T[:3, :3] = R * s_g
                        T[:3, 3] = c_scn - (s_g * R @ c_mem)
                        best_T = T

        # 4. Fine Refinement (ICP)
        # Using Point-to-Point ICP initiated with best Feature Match
        reg = o3d.pipelines.registration.registration_icp(
            mem_pcd,
            scene_pcd,
            0.03,
            best_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return reg.transformation

    def sample_grasps_heuristic(self, pcd, n_samples=30):
        """
        Replaces AnyGrasp. Finds surface normals and generates grasp candidates.
        """
        pcd.estimate_normals()
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        grasps = []
        indices = np.random.choice(len(points), n_samples)

        for i in indices:
            pt = points[i]
            # Approach vector (z) is opposite to surface normal
            z = -normals[i]
            z /= np.linalg.norm(z)

            # Y vector (orthogonal)
            y = np.cross(z, np.array([0, 0, 1]))
            if np.linalg.norm(y) < 0.1:
                y = np.cross(z, np.array([1, 0, 0]))
            y /= np.linalg.norm(y)
            x = np.cross(y, z)

            rot = np.column_stack((x, y, z))
            G = np.eye(4)
            G[:3, :3] = rot
            G[:3, 3] = pt  # Grasp AT surface point

            grasps.append((G, 1.0))  # Score 1.0 (Geometric stability placeholder)
        return grasps

    def score_grasps(self, transferred_grasp, candidates):
        """Evaluates candidates based on task intent"""
        v_target = transferred_grasp[:3, 2]
        p_target = transferred_grasp[:3, 3]

        best_g = None
        max_score = -np.inf

        for G_cand, geo_score in candidates:
            v_cand = G_cand[:3, 2]
            p_cand = G_cand[:3, 3]

            # 1. Orientation Consistency (Cosine Sim)
            cos_sim = np.dot(v_target, v_cand)

            # 2. Position Consistency (Gaussian)
            dist_sq = np.sum((p_target - p_cand) ** 2)
            gauss = np.exp(-dist_sq / (2 * SIGMA_POS**2))

            # Weighted Sum (Eq 6)
            s_task = cos_sim + gauss
            final_score = WEIGHT_TASK * s_task + WEIGHT_GEO * geo_score

            if final_score > max_score:
                max_score = final_score
                best_g = G_cand

        return best_g
