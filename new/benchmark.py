import glob
import json
import os
import pickle

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from alignment import AlignmentModule
from config import GRIMConfig
from feature_extractor import FeatureExtractor
from grasp_scorer import GeometricGraspSampler, GraspScorer
from memory_manager import MemoryManager
from preprocessing import process_rgbd_scene


class GrimEvaluator:
    def __init__(self, memory_dir="memory_bank", test_dir="test_data"):
        self.test_dir = test_dir

        # Initialize Pipeline Components
        self.extractor = FeatureExtractor()
        self.memory = MemoryManager()
        self.memory.load_memory_bank(memory_dir)
        self.aligner = AlignmentModule()
        self.sampler = GeometricGraspSampler()
        self.scorer = GraspScorer()

        # Intrinsic (Update with your specific camera values)
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 607.5, 607.5, 320, 240
        )

    def calculate_pose_error(self, pred_pose, gt_pose):
        """
        Calculates translation (meters) and rotation (degrees) error.
        """
        # Translation Error (Euclidean)
        t_pred = pred_pose[:3, 3]
        t_gt = gt_pose[:3, 3]
        t_err = np.linalg.norm(t_pred - t_gt)

        # Rotation Error (Geodesic distance in SO3)
        R_pred = pred_pose[:3, :3]
        R_gt = gt_pose[:3, :3]

        # R_diff = R_gt * R_pred^T
        R_diff = np.dot(R_gt, R_pred.T)
        trace = np.trace(R_diff)

        # theta = arccos((trace - 1) / 2)
        # Clamp for numerical stability
        trace = np.clip(trace, -1.0, 3.0)
        angle = np.arccos((trace - 1) / 2.0)
        r_err_deg = np.degrees(angle)

        return t_err, r_err_deg

    def evaluate(self):
        results = []
        test_cases = glob.glob(os.path.join(self.test_dir, "*"))

        print(f"\n--- Starting Evaluation on {len(test_cases)} cases ---")

        for case_path in test_cases:
            if not os.path.isdir(case_path):
                continue

            case_id = os.path.basename(case_path)
            print(f"\nProcessing Case: {case_id}")

            # 1. Load Data
            try:
                rgb_path = os.path.join(case_path, "rgb.png")
                depth_path = os.path.join(case_path, "depth.png")
                meta_path = os.path.join(case_path, "meta.json")

                with open(meta_path, "r") as f:
                    meta = json.load(f)

                task_desc = meta["task"]
                gt_pose = np.array(meta["ground_truth_pose"])  # Expecting 4x4 list

            except Exception as e:
                print(f"Skipping {case_id}: Missing files ({e})")
                continue

            # 2. Run Pipeline
            # A. Process Scene
            scene_pcd, scene_feats = process_rgbd_scene(
                rgb_path, depth_path, self.intrinsics, self.extractor
            )

            # B. Retrieve
            scene_task_emb = self.extractor.extract_task_embedding(task_desc)
            memory_item = self.memory.retrieve(scene_feats, scene_task_emb)

            if memory_item is None:
                print("Retrieval Failed.")
                results.append(
                    {"id": case_id, "success": False, "note": "Retrieval Fail"}
                )
                continue

            # C. Align
            T_align = self.aligner.align(
                memory_item.pcd, scene_pcd, memory_item.dino_features, scene_feats
            )
            transferred_grasp = T_align @ memory_item.grasp_pose

            # D. Score (Refine)
            candidates = self.sampler.sample_grasps(scene_pcd)
            if not candidates:
                print("Sampling Failed.")
                results.append(
                    {"id": case_id, "success": False, "note": "Sampling Fail"}
                )
                continue

            best_score, best_grasp = self.scorer.evaluate(transferred_grasp, candidates)

            # 3. Compute Metrics
            t_err, r_err = self.calculate_pose_error(best_grasp, gt_pose)

            # Definition of Success: < 5cm translation error AND < 30 degrees rotation error
            # (Standard grasping thresholds)
            is_success = (t_err < 0.05) and (r_err < 30.0)

            print(f"  -> Errors: Trans={t_err * 100:.1f}cm, Rot={r_err:.1f}deg")
            print(f"  -> Result: {'SUCCESS' if is_success else 'FAILURE'}")

            results.append(
                {
                    "id": case_id,
                    "task": task_desc,
                    "t_err": t_err,
                    "r_err": r_err,
                    "success": is_success,
                }
            )

        self.print_summary(results)

    def print_summary(self, results):
        if not results:
            print("No results to summarize.")
            return

        total = len(results)
        successes = sum(1 for r in results if r["success"])
        avg_t_err = np.mean([r["t_err"] for r in results if "t_err" in r])
        avg_r_err = np.mean([r["r_err"] for r in results if "r_err" in r])

        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total Cases: {total}")
        print(f"Success Rate: {successes / total * 100:.1f}%")
        print(f"Avg Trans Error: {avg_t_err * 100:.2f} cm")
        print(f"Avg Rot Error:   {avg_r_err:.2f} deg")
        print("==========================")


if __name__ == "__main__":
    # Ensure test directories exist
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
        print("Created 'test_data' folder. Add test cases there.")
        print("Structure: test_data/case_01/ {rgb.png, depth.png, meta.json}")
    else:
        evaluator = GrimEvaluator()
        evaluator.evaluate()
