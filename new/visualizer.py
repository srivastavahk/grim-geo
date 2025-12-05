import copy

import numpy as np
import open3d as o3d


def visualize_grim_result(
    scene_pcd, memory_pcd, T_align, best_grasp, window_name="GRIM Alignment"
):
    """
    Visualizes the result of the GRIM pipeline.

    Colors:
    - Grey:  Scene Point Cloud (What the robot sees)
    - Red:   Memory Object (The retrieved knowledge, transformed to match the scene)
    - Axis:  The final Grasp Pose (RGB = XYZ)

    Args:
        scene_pcd: Open3D PointCloud of the scene
        memory_pcd: Open3D PointCloud of the memory item
        T_align: 4x4 Transformation Matrix (Memory -> Scene)
        best_grasp: 4x4 Transformation Matrix (Grasp Pose in Scene Frame)
    """
    print(f"\n[Visualizer] Launching window: {window_name}")
    print("[Visualizer] Red Object = Aligned Memory Model")
    print("[Visualizer] Grey Object = Your Scene Data")

    # 1. Prepare Scene (Grey)
    scene_viz = copy.deepcopy(scene_pcd)
    scene_viz.paint_uniform_color([0.7, 0.7, 0.7])

    # 2. Prepare Memory Object (Red)
    # Apply the calculated alignment transform so it overlaps the scene
    mem_viz = copy.deepcopy(memory_pcd)
    mem_viz.transform(T_align)
    mem_viz.paint_uniform_color([1.0, 0.0, 0.0])

    # 3. Prepare Grasp Frame
    # Create a coordinate axis to represent the gripper
    # Red=X, Green=Y, Blue=Z (Approach)
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[0, 0, 0]
    )
    grasp_frame.transform(best_grasp)

    # 4. Draw
    o3d.visualization.draw_geometries(
        [scene_viz, mem_viz, grasp_frame],
        window_name=window_name,
        width=1024,
        height=768,
        left=50,
        top=50,
    )
