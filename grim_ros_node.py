import os
import pickle

import numpy as np
import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from grim_core import GRIMCore, MemoryInstance
from grim_perception import GRIMPerception


class GRIMNode(Node):
    def __init__(self):
        super().__init__("grim_node")

        # 1. Config
        self.declare_parameter("mode", "inference")  # 'record' or 'inference'
        self.mode = self.get_parameter("mode").get_parameter_value().string_value
        self.memory_file = "grim_memory_bank.pkl"

        self.perception = GRIMPerception()
        self.core = GRIMCore()
        self.bridge = CvBridge()

        # 2. ROS Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._action_client = ActionClient(self, MoveGroup, "move_action")

        # Synchronized RGB-D Subs (Stretch specific topics)
        # Note: Ensure these match your stretch launch file
        self.rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(
            self, Image, "/camera/aligned_depth_to_color/image_raw"
        )
        self.info_sub = Subscriber(self, CameraInfo, "/camera/color/camera_info")

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], 10, 0.1
        )
        self.ts.registerCallback(self.camera_callback)

        self.processing = False
        self.camera_intrinsics = None

        # Load Memory
        if os.path.exists(self.memory_file):
            self.get_logger().info(f"Loading memory from {self.memory_file}")
            with open(self.memory_file, "rb") as f:
                self.core.memory_bank = pickle.load(f)

    def camera_callback(self, rgb_msg, depth_msg, info_msg):
        if self.processing:
            return

        # Store intrinsics once
        if self.camera_intrinsics is None:
            k = np.array(info_msg.k).reshape(3, 3)
            self.camera_intrinsics = k

        if self.mode == "record":
            self.processing = True
            self.record_memory(rgb_msg, depth_msg)
            rclpy.shutdown()  # Exit after recording one instance

        elif self.mode == "inference":
            self.processing = True
            self.run_inference(rgb_msg, depth_msg)
            self.processing = False  # Allow next frame (or keep True to stop)

    def record_memory(self, rgb_msg, depth_msg):
        self.get_logger().info("--- RECORDING MEMORY INSTANCE ---")

        # 1. Process Data
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        pcd = self.perception.rgbd_to_pointcloud(
            rgb_img, depth_img, self.camera_intrinsics
        )
        feats = self.perception.extract_dino_features(
            rgb_img, pcd, self.camera_intrinsics
        )
        global_desc = np.mean(feats, axis=0)
        global_desc /= np.linalg.norm(global_desc)

        # 2. User Input for Metadata
        print("Enter Object Name (e.g., mug):")
        obj_name = input()
        print("Enter Task Name (e.g., pour):")
        task_name = input()

        task_emb = self.perception.encode_task(task_name)
        task_emb /= np.linalg.norm(task_emb)

        # 3. Capture Grasp Pose
        # Ideally, you jog the robot to the grasp pose and read TF.
        # Here, we assume the camera is LOOKING at a human demonstration or pre-positioned robot.
        # For simplicity in this script, we record an Identity grasp at the cloud centroid.
        # **In real deployment**: Replace this with `lookup_transform('base_link', 'link_grasp_center')`
        grasp = np.eye(4)
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        grasp[:3, 3] = centroid

        mem = MemoryInstance(
            obj_name, task_name, pcd, feats, global_desc, task_emb, grasp
        )

        # Save
        self.core.add_memory(mem)
        with open(self.memory_file, "wb") as f:
            pickle.dump(self.core.memory_bank, f)
        self.get_logger().info("Memory Saved!")

    def run_inference(self, rgb_msg, depth_msg):
        self.get_logger().info("--- RUNNING GRIM INFERENCE ---")

        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        # 1. Perception
        scene_pcd = self.perception.rgbd_to_pointcloud(
            rgb_img, depth_img, self.camera_intrinsics
        )
        scene_feats = self.perception.extract_dino_features(
            rgb_img, scene_pcd, self.camera_intrinsics
        )
        scene_desc = np.mean(scene_feats, axis=0)
        scene_desc /= np.linalg.norm(scene_desc) + 1e-6

        # 2. Retrieve
        target_task = "pour"  # Hardcoded for demo, or read from param
        task_emb = self.perception.encode_task(target_task)
        mem_inst, score = self.core.retrieve(scene_desc, task_emb)

        if not mem_inst:
            self.get_logger().warn("Memory Bank Empty!")
            return

        self.get_logger().info(f"Retrieved: {mem_inst.obj_name} ({score:.3f})")

        # 3. Align
        T_align = self.core.align(mem_inst, scene_pcd, scene_feats)

        # 4. Transfer & Score
        transferred_grasp = T_align @ mem_inst.grasp_pose
        candidates = self.core.sample_grasps_heuristic(scene_pcd)
        best_grasp = self.core.score_grasps(transferred_grasp, candidates)

        # 5. MoveIt Execution
        if best_grasp is not None:
            self.execute_moveit(best_grasp)
        else:
            self.get_logger().error("No valid grasp found.")

    def execute_moveit(self, grasp_matrix):
        # ... (Same MoveIt logic as previous response) ...
        # Ensure you include the TF transform logic here!
        pass


def main(args=None):
    rclpy.init(args=args)
    node = GRIMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
