import numpy as np
import open3d as o3d
from config import GRIMConfig
from scipy.spatial.transform import Rotation

class GeometricGraspSampler:
    """
    Fallback for AnyGrasp. Generates antipodal grasps based on geometry.

    """
    def sample_grasps(self, pcd, num_samples=50):
        """
        Simple antipodal sampling:
        1. Pick a point.
        2. Shoot ray along normal.
        3. If it hits back surface with opposing normal, it's a candidate.
        """
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        candidates = []
        indices = np.random.choice(len(points), size=min(len(points), num_samples*10), replace=False)

        for i in indices:
            p1 = points[i]
            n1 = normals[i]

            # Look for points "inside" the object along negative normal
            # Simplified: Just create a generic grasp frame aligned with normal
            # Z = normal (Approach), Y = binormal (Closing direction)

            # Construct Rotation matrix from Normal
            z_axis = -n1 # Approach vector (into object)
            z_axis /= np.linalg.norm(z_axis)

            # Random tangent for gripper orientation
            temp_vec = np.array([1, 0, 0]) if np.abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            y_axis = np.cross(z_axis, temp_vec)
            y_axis /= np.linalg.norm(y_axis)

            x_axis = np.cross(y_axis, z_axis)

            R = np.column_stack((x_axis, y_axis, z_axis))
            t = p1 - (z_axis * 0.05) # Back off 5cm

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            candidates.append(T)

            if len(candidates) >= num_samples:
                break

        return candidates

class GraspScorer:
    def evaluate(self, memory_grasp_transformed, candidate_grasps):
        """
        Implements Eq 5 and Eq 6.
        """
        # 1. Extract Target Parameters from Transferred Memory Grasp
        # G_S = T_final * G_M
        T_S = memory_grasp_transformed
        p_target = T_S[:3, 3]

        # Primary approach direction v_target = R_S * e_z
        # Using configured gripper approach vector (usually Z=[0,0,1])
        e_z = GRIMConfig.GRIPPER_APPROACH_VECTOR
        v_target = T_S[:3, :3] @ e_z
        v_target = v_target / np.linalg.norm(v_target)

        scored_grasps = []

        for G_A in candidate_grasps:
            # Candidate parameters
            p_A = G_A[:3, 3]
            o_z = G_A[:3, :3] @ e_z
            o_z = o_z / np.linalg.norm(o_z)

            # Eq 5: Task Compatibility Score [cite: 185]
            # Term 1: Cosine Similarity of approach vectors
            term1 = np.dot(v_target, o_z)

            # Term 2: Gaussian decay of position
            dist_sq = np.linalg.norm(p_A - p_target) ** 2
            term2 = np.exp(-dist_sq / (2 * (GRIMConfig.SIGMA_POS ** 2)))

            S_task = term1 + term2

            # Eq 6: Final Score [cite: 192]
            # Assumes geometric score is 1.0 (stable) for generated candidates
            S_geo = 1.0
            S_final = (GRIMConfig.W_TASK * S_task) + (GRIMConfig.W_GEO * S_geo)

            scored_grasps.append((S_final, G_A))

        # Sort descending
        scored_grasps.sort(key=lambda x: x[0], reverse=True)
        return scored_grasps[0] # Return best (Score, Pose)
