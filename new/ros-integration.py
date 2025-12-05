import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
import tf2_geometry_msgs  # Essential for do_transform_pose

class StretchGraspTransformer(Node):
    """
    ROS 2 Node to handle coordinate transformations for Hello Robot Stretch 3.
    Converts GRIM inference results (Camera Frame) -> Robot Base Frame.
    """
    def __init__(self):
        super().__init__('grim_grasp_transformer')

        # 1. TF Buffer & Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 2. Visualizer Publisher (for RViz)
        self.grasp_pub = self.create_publisher(PoseStamped, '/grim/target_grasp', 10)

        self.get_logger().info("Stretch Grasp Transformer Initialized. Waiting for TF...")

    def numpy_to_pose_stamped(self, matrix_4x4, frame_id):
        """
        Converts a 4x4 numpy homogeneous matrix to a ROS 2 PoseStamped message.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = frame_id

        # Translation
        pose_msg.pose.position.x = matrix_4x4[0, 3]
        pose_msg.pose.position.y = matrix_4x4[1, 3]
        pose_msg.pose.position.z = matrix_4x4[2, 3]

        # Rotation (Matrix -> Quaternion)
        r = Rotation.from_matrix(matrix_4x4[:3, :3])
        quat = r.as_quat() # [x, y, z, w]

        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        return pose_msg

    def transform_grasp_to_base(self, grasp_matrix, camera_frame="camera_color_optical_frame"):
        """
        Transforms the grasp from Camera Optical Frame to Base Link.

        Args:
            grasp_matrix (np.array): 4x4 transform of the grasp in camera frame.
            camera_frame (str): The frame ID of the camera (check your URDF, usually 'camera_color_optical_frame' or 'camera_depth_optical_frame').

        Returns:
            geometry_msgs/PoseStamped: The grasp pose in 'base_link'.
        """
        # 1. Convert Numpy -> PoseStamped (in Camera Frame)
        pose_cam = self.numpy_to_pose_stamped(grasp_matrix, camera_frame)

        try:
            # 2. Look up Transform (Base -> Camera)
            # timeout of 1.0 seconds
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # 3. Apply Transform
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam.pose, transform)

            # Repackage into PoseStamped with correct header
            final_pose = PoseStamped()
            final_pose.header.stamp = self.get_clock().now().to_msg()
            final_pose.header.frame_id = 'base_link'
            final_pose.pose = pose_base

            # 4. Publish for Visualization
            self.grasp_pub.publish(final_pose)
            self.get_logger().info(f"Transformed Grasp: \n{final_pose.pose.position}")

            return final_pose

        except tf2_ros.LookupException as e:
            self.get_logger().error(f"TF Lookup Failed: {e}")
            return None
        except Exception as e:
            self.get_logger().error(f"Transformation Failed: {e}")
            return None

def execute_on_stretch(pose_stamped):
    """
    Placeholder for Hello Robot execution API (stretch_body).
    """
    if pose_stamped is None:
        print("Cannot execute: Invalid Pose.")
        return

    print("Sending Grasp Pose to Stretch Controller...")
    # Example logic using stretch_body primitives would go here:
    # robot.arm.move_to(...)
    # robot.lift.move_to(...)
    # robot.end_of_arm.move_to(...)
    pass
