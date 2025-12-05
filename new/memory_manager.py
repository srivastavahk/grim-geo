import numpy as np
import torch
from scipy.spatial.distance import cosine
from config import GRIMConfig

class MemoryItem:
    def __init__(self, obj_id, pcd, dino_features, task_embedding, grasp_pose, task_name):
        self.obj_id = obj_id
        self.pcd = pcd                     # Open3D PointCloud (P_MO)
        self.dino_features = dino_features # Numpy (N, D) (F_MO)
        self.task_embedding = task_embedding # Tensor (1, D)
        self.grasp_pose = grasp_pose       # 4x4 Matrix (G_M)
        self.task_name = task_name

    @property
    def global_descriptor(self):
        # Average pooling of DINO features [cite: 166]
        return np.mean(self.dino_features, axis=0)

class MemoryManager:
    def __init__(self):
        self.memory = []

    def add_item(self, item: MemoryItem):
        self.memory.append(item)

    def retrieve(self, scene_pcd_features, scene_task_embedding):
        """
        Implements Joint Similarity Score (Eq 2).
        S_joint(i,j) = sim_cos(F_SO, F_MO) * sim_cos(E_TS, E_TM)
        [cite: 145]
        """
        best_score = -1.0
        best_item = None

        # 1. Compute Global Scene Descriptor
        scene_global = np.mean(scene_pcd_features, axis=0)

        # 2. Normalize inputs for Cosine Similarity
        scene_global_norm = scene_global / np.linalg.norm(scene_global)
        scene_task_norm = scene_task_embedding / scene_task_embedding.norm()

        for item in self.memory:
            # Object Visual Similarity
            mem_global_norm = item.global_descriptor / np.linalg.norm(item.global_descriptor)
            sim_vis = np.dot(scene_global_norm, mem_global_norm)

            # Task Semantic Similarity
            # Ensure tensors are on same device/type
            mem_task_norm = item.task_embedding.to(GRIMConfig.DEVICE)
            sim_task = torch.matmul(scene_task_norm, mem_task_norm.T).item()

            # Joint Score
            joint_score = sim_vis * sim_task

            if joint_score > best_score:
                best_score = joint_score
                best_item = item

        print(f"Retrieved Memory: {best_item.obj_id} (Task: {best_item.task_name}) | Score: {best_score:.4f}")
        return best_item
