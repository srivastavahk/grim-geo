import argparse
import os
import pickle

import numpy as np
import open3d as o3d

from config import GRIMConfig
from feature_extractor import FeatureExtractor
from grasp_scorer import GeometricGraspSampler
from memory_manager import MemoryItem
from preprocessing import process_rgbd_scene


def create_memory_entry(args):
    # 1. Setup
    print(f"--- GRIM Memory Creation Tool ---")
    print(f"Task: {args.task}")

    extractor = FeatureExtractor()
    sampler = GeometricGraspSampler()

    # Intrinsic (Update with your camera's actual values if different)
    # Defaulting to Stretch 3 D435i Color Optical Frame values roughly
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 607.5, 607.5, 320, 240)

    # 2. Process Image -> Feature Cloud
    print("Processing RGB-D data...")
    pcd, features = process_rgbd_scene(args.rgb, args.depth, intrinsics, extractor)

    # 3. Generate Task Embedding
    print("Encoding Task...")
    task_emb = extractor.extract_task_embedding(args.task)

    # 4. Grasp Selection Interface
    print("Sampling candidate grasps for annotation...")
    # Sample more grasps than usual to ensure a good one exists
    candidates = sampler.sample_grasps(pcd, num_samples=30)

    if not candidates:
        print("Error: Could not find any stable geometric grasps on this object.")
        return

    # Visualizer for Selection
    print("\n--- INSTRUCTIONS ---")
    print("1. A window will open showing the object and numbered grasp arrows.")
    print("2. RED arrow = X axis, GREEN = Y, BLUE = Z (Approach).")
    print("3. Inspect the grasps. Find the ID of the one that best fits the task.")
    print("4. Close the window and enter the ID in the terminal.")

    geometries = [pcd]

    # Draw arrows for candidates
    # We color code them or just rely on index?
    # To keep it simple, we draw coordinate frames.
    for i, grasp in enumerate(candidates):
        # Create a small frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(grasp)

        # Add a text label (3D text is hard in Open3D standard, so we print location)
        # We will visualize them in batches or all at once? All at once might be cluttered.
        # Let's visualize just the frames.
        geometries.append(frame)

    # Note: Identifying specific grasp indices in a dense cloud is hard.
    # A better UX for CLI: Iterate through them one by one?
    # Or just save the point cloud and define grasp manually?
    # Let's try "Iterative Review": Show 1 grasp at a time.

    selected_grasp = None

    # Simple Review Loop
    print("\nStarting Review Mode. Close window to vote (y/n).")
    for i, grasp in enumerate(candidates):
        print(f"Showing Grasp Candidate #{i}...")

        # Visualize specific grasp
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(grasp)

        # Draw object + this grasp
        o3d.visualization.draw_geometries([pcd, frame], window_name=f"Candidate {i}")

        choice = input(
            f"Is Grasp #{i} good for task '{args.task}'? (y/n/q to quit): "
        ).lower()
        if choice == "y":
            selected_grasp = grasp
            print(f"Selected Grasp #{i}.")
            break
        elif choice == "q":
            return

    if selected_grasp is None:
        print("No grasp selected. Exiting.")
        return

    # 5. Save Memory Item
    item = MemoryItem(
        obj_id=args.object_name,
        pcd=pcd,
        dino_features=features,
        task_embedding=task_emb,
        grasp_pose=selected_grasp,
        task_name=args.task,
    )

    # Ensure directory exists
    os.makedirs("memory_bank", exist_ok=True)
    filename = f"memory_bank/{args.object_name}_{args.task.replace(' ', '_')}.grim"

    # We use pickle for serialization (numpy/torch/open3d compatible)
    # Open3D objects are not natively pickleable in older versions.
    # We convert PCD to dict for storage.
    serialized_data = {
        "obj_id": item.obj_id,
        "points": np.asarray(item.pcd.points),
        "colors": np.asarray(item.pcd.colors),
        "dino_features": item.dino_features,
        "task_embedding": item.task_embedding,
        "grasp_pose": item.grasp_pose,
        "task_name": item.task_name,
    }

    with open(filename, "wb") as f:
        pickle.dump(serialized_data, f)

    print(f"✅ Memory saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRIM Memory Creation Tool")
    parser.add_argument("--rgb", required=True, help="Path to reference RGB image")
    parser.add_argument("--depth", required=True, help="Path to reference Depth image")
    parser.add_argument(
        "--object_name", required=True, help="Name of the object (e.g., 'mug')"
    )
    parser.add_argument(
        "--task", required=True, help="Task description (e.g., 'drink')"
    )

    args = parser.parse_args()
    create_memory_entry(args)
