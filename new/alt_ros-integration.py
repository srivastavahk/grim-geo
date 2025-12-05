import time

import numpy as np
import rclpy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation


class StretchGraspTransformer(Node):
    def __init__(self):
        super().__init__("grim_grasp_transformer")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.grasp_pub = self.create_publisher(PoseStamped, "/grim/target_grasp", 10)

        self.get_logger().info("Stretch Grasp Transformer Initialized.")

    def numpy_to_pose_stamped(self, matrix_4x4, frame_id):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id

        pose_msg.pose.position.x = matrix_4x4[0, 3]
        pose_msg.pose.position.y = matrix_4x4[1, 3]
        pose_msg.pose.position.z = matrix_4x4[2, 3]

        r = Rotation.from_matrix(matrix_4x4[:3, :3])
        quat = r.as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        return pose_msg

    def wait_for_transform_availability(
        self, target_frame, source_frame, timeout_sec=10.0
    ):
        """
        Blocks until the transform is available or timeout is reached.
        """
        self.get_logger().info(
            f"Waiting for transform: {source_frame} -> {target_frame}..."
        )
        start_time = time.time()
        while (time.time() - start_time) < timeout_sec:
            # Check if frames exist in buffer
            if self.tf_buffer.can_transform(
                target_frame, source_frame, rclpy.time.Time()
            ):
                self.get_logger().info("Transform found!")
                return True
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks to update buffer

        self.get_logger().error(
            f"Timeout waiting for transform {source_frame} -> {target_frame}"
        )
        return False

    def transform_grasp_to_base(
        self, grasp_matrix, camera_frame="camera_color_optical_frame"
    ):
        # 1. Ensure transform exists before calculating
        if not self.wait_for_transform_availability("base_link", camera_frame):
            return None

        # 2. Convert Numpy -> PoseStamped
        pose_cam = self.numpy_to_pose_stamped(grasp_matrix, camera_frame)

        try:
            # 3. Look up Transform
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )

            # 4. Apply Transform
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam.pose, transform)

            final_pose = PoseStamped()
            final_pose.header.stamp = self.get_clock().now().to_msg()
            final_pose.header.frame_id = "base_link"
            final_pose.pose = pose_base

            self.grasp_pub.publish(final_pose)
            return final_pose

        except Exception as e:
            self.get_logger().error(f"Transformation Failed: {e}")
            return None
