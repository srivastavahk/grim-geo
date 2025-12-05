import numpy as np
import open3d as o3d
from feature_extractor import FeatureExtractor
from preprocessing import process_rgbd_scene
from memory_manager import MemoryManager, MemoryItem
from alignment import AlignmentModule
from grasp_scorer import GeometricGraspSampler, GraspScorer

def main():
    # --- SETUP ---
    # 1. Initialize Modules
    extractor = FeatureExtractor()
    memory = MemoryManager()
    aligner = AlignmentModule()
    sampler = GeometricGraspSampler()
    scorer = GraspScorer()

    # 2. Populate Memory (Simulation)
    # In real usage, you load these from disk
    print("\n--- Populating Memory ---")
    # Mock Memory Item: A "Mug" with a specific handle grasp
    mock_pcd = o3d.geometry.TriangleMesh.create_box().sample_points_poisson_disk(500)
    mock_feats = np.random.rand(500, 384) # DINOv2-small dim
    mock_task_emb = extractor.extract_task_embedding("pour water")
    mock_pose = np.eye(4)

    memory.add_item(MemoryItem("mug_01", mock_pcd, mock_feats, mock_task_emb, mock_pose, "pour water"))

    # --- INFERENCE ---
    print("\n--- Starting Inference ---")

    # 3. Load Inputs
    # Replace these with your actual file paths
    rgb_path = "data/scene_rgb.png"
    depth_path = "data/scene_depth.png"
    # Example Intrinsic (Stretch 3 Realsense D435i usually)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 600, 600, 320, 240)

    task_description = "pour water"

    # 4. Preprocess Scene
    print("Processing Scene RGB-D...")
    try:
        scene_pcd, scene_feats = process_rgbd_scene(rgb_path, depth_path, intrinsics, extractor)
    except Exception as e:
        print(f"Error loading images: {e}. creating dummy scene for demonstration.")
        scene_pcd = o3d.geometry.TriangleMesh.create_sphere().sample_points_poisson_disk(500)
        scene_feats = np.random.rand(500, 384)

    # 5. Retrieve [cite: 145]
    print(f"Retrieving memory for task: '{task_description}'...")
    scene_task_emb = extractor.extract_task_embedding(task_description)
    memory_item = memory.retrieve(scene_feats, scene_task_emb)

    if memory_item is None:
        print("No memory item found.")
        return

    # 6. Align [cite: 153, 450]
    print("Aligning Memory Object to Scene Object...")
    T_align = aligner.align(memory_item.pcd, scene_pcd, memory_item.dino_features, scene_feats)

    # 7. Transfer Grasp
    # G_S = T_final * G_M [cite: 176]
    transferred_grasp = T_align @ memory_item.grasp_pose

    # 8. Sample Candidates [cite: 179]
    print("Sampling Geometric Candidates...")
    candidate_grasps = sampler.sample_grasps(scene_pcd)

    # 9. Score & Select [cite: 184]
    print("Scoring Candidates...")
    best_score, best_grasp = scorer.evaluate(transferred_grasp, candidate_grasps)

    print(f"\nOptimization Complete.")
    print(f"Best Grasp Score: {best_score:.4f}")
    print("Best Grasp Matrix:\n", best_grasp)

    # Visualization
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    mesh_frame.transform(best_grasp)
    o3d.visualization.draw_geometries([scene_pcd, mesh_frame])

if __name__ == "__main__":
    main()
