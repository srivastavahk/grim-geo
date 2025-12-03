import cv2

# Math
import numpy as np
import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion

# MoveIt 2 Messages
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive

# TF2 for Coordinate Transformation
from tf2_ros import Buffer, TransformListener

# Import GRIM Modules
from  import GRIMCore, MemoryInstance
from  import GRIMPerception


class GRIMMoveItNode(Node):
    def __init__(self):
        super().__init__("grim_moveit_node")

        # 1. Init GRIM Engines
        self.perception = GRIMPerception()
        self.core = GRIMCore()
        self.bridge = CvBridge()

        # 2. Setup TF2 (Critical for Camera -> Base transform)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Setup MoveIt Action Client
        self._action_client = ActionClient(self, MoveGroup, "move_action")
        self.planning_group = "stretch_arm"  # Check your SRDF group name (often 'stretch_arm' or 'mobile_manipulator')

        # 4. Populate Memory
        self.get_logger().info("Populating Memory...")
        self._populate_dummy_memory()

        # 5. Subscribers
        self.sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, 10
        )  # Adjust topic if needed (e.g. /camera/aligned_depth_to_color/image_raw)

        self.target_task = "Lift the bottle"
        self.processing = False
        self.camera_frame = (
            "camera_color_optical_frame"  # Standard RealSense frame name on Stretch
        )
        self.base_frame = "base_link"

    def _populate_dummy_memory(self):
        """Creates synthetic memory for testing"""
        import open3d as o3d

        # Mock Bottle
        pcd = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.04, height=0.2
        ).sample_points_uniformly(1000)
        feats = np.random.rand(1000, 384)
        desc = np.mean(feats, axis=0)
        desc /= np.linalg.norm(desc)
        task_emb = self.perception.encode_task("Lift the bottle")
        grasp = np.eye(4)
        # Offset grasp 10cm up in Z
        grasp[2, 3] = 0.10

        mem = MemoryInstance("Bottle", "Lift", pcd, feats, desc, task_emb, grasp)
        self.core.add_memory(mem)

    def image_callback(self, msg):
        if self.processing:
            return
        self.processing = True
        self.get_logger().info("--- Processing Frame ---")

        try:
            # 1. Perception & Core Logic
            rgb_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")

            # RGB -> PCD
            scene_pcd = self.perception.rgb_to_pointcloud(rgb_img)

            # Features
            scene_feats = self.perception.extract_dino_features(rgb_img, scene_pcd)
            scene_desc = np.mean(scene_feats, axis=0)
            scene_desc /= np.linalg.norm(scene_desc) + 1e-6

            # Retrieve & Align
            task_emb = self.perception.encode_task(self.target_task)
            mem_inst, score = self.core.retrieve(scene_desc, task_emb)

            if score < 0.3:  # Threshold
                self.get_logger().warn(f"Low retrieval score ({score:.3f}). Skipping.")
                self.processing = False
                return

            self.get_logger().info(f"Retrieved: {mem_inst.obj_name}")
            T_align = self.core.align(mem_inst, scene_pcd, scene_feats)

            transferred_grasp = T_align @ mem_inst.grasp_pose
            candidates = self.core.sample_grasps_heuristic(scene_pcd)
            best_grasp = self.core.score_grasps(transferred_grasp, candidates)

            # 2. Execute with MoveIt
            self.execute_grasp_moveit(best_grasp)

        except Exception as e:
            self.get_logger().error(f"Error in pipeline: {e}")
            import traceback

            traceback.print_exc()

        self.processing = False

    def execute_grasp_moveit(self, grasp_matrix):
        """Transforms grasp to base frame and sends to MoveIt"""
        self.get_logger().info("Preparing MoveIt Plan...")

        # A. Matrix -> PoseStamped (Camera Frame)
        pose_cam = PoseStamped()
        pose_cam.header.stamp = self.get_clock().now().to_msg()
        pose_cam.header.frame_id = self.camera_frame

        # Translation
        pose_cam.pose.position.x = grasp_matrix[0, 3]
        pose_cam.pose.position.y = grasp_matrix[1, 3]
        pose_cam.pose.position.z = grasp_matrix[2, 3]

        # Rotation (Matrix -> Quaternion)
        r = R.from_matrix(grasp_matrix[:3, :3])
        q = r.as_quat()  # x, y, z, w
        pose_cam.pose.orientation.x = q[0]
        pose_cam.pose.orientation.y = q[1]
        pose_cam.pose.orientation.z = q[2]
        pose_cam.pose.orientation.w = q[3]

        # B. Transform to Base Frame (TF2)
        try:
            # Wait for transform availability
            if not self.tf_buffer.can_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time()
            ):
                self.get_logger().error("TF Transform not available!")
                return

            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time()
            )

            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)

            self.get_logger().info(
                f"Target Pose (Base Frame): {pose_base.pose.position}"
            )

            # C. Construct MoveGroup Goal
            self.send_moveit_goal(pose_base)

        except Exception as e:
            self.get_logger().error(f"Transform Error: {e}")

    def send_moveit_goal(self, target_pose_stamped):
        # Check server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveIt Action Server not available!")
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.workspace_parameters.header.frame_id = self.base_frame
        goal_msg.request.workspace_parameters.min_corner.x = -1.0
        goal_msg.request.workspace_parameters.min_corner.y = -1.0
        goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0
        goal_msg.request.workspace_parameters.max_corner.y = 1.0
        goal_msg.request.workspace_parameters.max_corner.z = 1.0

        goal_msg.request.group_name = self.planning_group
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.num_planning_attempts = 10

        # Define Constraints
        # Position Constraint
        pcm = PositionConstraint()
        pcm.header = target_pose_stamped.header
        pcm.link_name = (
            "link_grasp_center"  # Check your Stretch URDF for the gripper link name!
        )
        pcm.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
        )
        pcm.constraint_region.primitive_poses.append(target_pose_stamped.pose)
        pcm.weight = 1.0

        # Orientation Constraint
        ocm = OrientationConstraint()
        ocm.header = target_pose_stamped.header
        ocm.link_name = "link_grasp_center"
        ocm.orientation = target_pose_stamped.pose.orientation
        ocm.absolute_x_axis_tolerance = 0.1
        ocm.absolute_y_axis_tolerance = 0.1
        ocm.absolute_z_axis_tolerance = 0.1
        ocm.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pcm)
        constraints.orientation_constraints.append(ocm)
        goal_msg.request.goal_constraints.append(constraints)

        # Send Goal
        self.get_logger().info("Sending Goal to MoveIt...")
        self._future = self._action_client.send_goal_async(goal_msg)
        self._future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("MoveIt Goal rejected :(")
            return

        self.get_logger().info("MoveIt Goal accepted! Executing...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:  # SUCCESS
            self.get_logger().info("MoveIt Execution Successful!")
        else:
            self.get_logger().error(
                f"MoveIt Execution Failed. Error Code: {result.error_code.val}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GRIMMoveItNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
