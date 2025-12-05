import numpy as np
import open3d as o3d

from alignment import AlignmentModule
from feature_extractor import FeatureExtractor
from grasp_scorer import GeometricGraspSampler, GraspScorer
from memory_manager import MemoryItem, MemoryManager
from preprocessing import process_rgbd_scene
from visualizer import visualize_grim_result

# import rclpy
# import threading
# from ros_integration import StretchGraspTransformer


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
    mock_feats = np.random.rand(500, 384)  # DINOv2-small dim
    mock_task_emb = extractor.extract_task_embedding("pour water")
    mock_pose = np.eye(4)

    memory.add_item(
        MemoryItem(
            "mug_01", mock_pcd, mock_feats, mock_task_emb, mock_pose, "pour water"
        )
    )

    # NEW: Load from Disk
    # memory.load_memory_bank("memory_bank")

    # # Check if empty
    # if len(memory.memory) == 0:
    #     print("Memory bank is empty! Run 'memory_tool.py' first.")
    #     return

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
        scene_pcd, scene_feats = process_rgbd_scene(
            rgb_path, depth_path, intrinsics, extractor
        )
    except Exception as e:
        print(f"Error loading images: {e}. creating dummy scene for demonstration.")
        scene_pcd = (
            o3d.geometry.TriangleMesh.create_sphere().sample_points_poisson_disk(500)
        )
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
    T_align = aligner.align(
        memory_item.pcd, scene_pcd, memory_item.dino_features, scene_feats
    )

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

    # --- VISUALIZATION START ---
    # Verify the alignment visually before sending to robot
    print("Visualizing Alignment... (Close window to continue to execution)")
    visualize_grim_result(
        scene_pcd=scene_pcd,
        memory_pcd=memory_item.pcd,
        T_align=T_align,
        best_grasp=best_grasp,
    )
    # --- VISUALIZATION END ---

    # # --- ROS 2 INTEGRATION START ---
    #     print("\n--- Initializing ROS 2 Transformation ---")

    #     # Initialize ROS context
    #     if not rclpy.ok():
    #         rclpy.init()

    #     node = StretchGraspTransformer()

    #     # Spin ROS in a background thread to handle TF callbacks
    #     spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    #     spin_thread.start()

    #     # Define your specific camera frame
    #     # For D435i on Stretch 3, this is standard
    #     CAMERA_FRAME = "camera_color_optical_frame"

    #     # Perform Transformation
    #     target_pose_base = node.transform_grasp_to_base(best_grasp, CAMERA_FRAME)

    #     if target_pose_base:
    #         print(f"\nSUCCESS: Grasp Transformed to 'base_link'")
    #         print(f"Position: x={target_pose_base.pose.position.x:.3f}, y={target_pose_base.pose.position.y:.3f}, z={target_pose_base.pose.position.z:.3f}")

    #         # Optional: Save to file for a separate execution script
    #         # np.save("target_grasp.npy", target_pose_base)
    #     else:
    #         print("\nFAILURE: Could not transform grasp. Check TF tree.")

    #     # Clean up
    #     node.destroy_node()
    #     rclpy.shutdown()
    #     # --- ROS 2 INTEGRATION END ---

    # Visualization
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    mesh_frame.transform(best_grasp)
    o3d.visualization.draw_geometries([scene_pcd, mesh_frame])


if __name__ == "__main__":
    main()
