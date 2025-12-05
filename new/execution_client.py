import math
import sys

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectoryPoint


class StretchGraspExecutor(Node):
    def __init__(self):
        super().__init__("stretch_grasp_executor")

        # 1. Subscriber to GRIM output
        self.subscription = self.create_subscription(
            PoseStamped, "/grim/target_grasp", self.grasp_callback, 10
        )

        # 2. Action Client for Robot Control
        self._action_client = ActionClient(
            self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        )

        # Robot Physical Constraints (Stretch 3 defaults)
        self.WRIST_EXTENSION_OFFSET = (
            0.35  # Approx distance from mast to gripper center when retracted
        )
        self.Z_LIFT_OFFSET = 0.0  # Calibration offset if needed

        self.get_logger().info(
            "Grasp Executor Waiting for Pose on /grim/target_grasp ..."
        )

    def grasp_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Received Target Grasp: {msg.pose.position}")

        # 1. Calculate Joint Goals (Analytical IK)
        joint_goals = self.compute_stretch_ik(msg.pose)

        if joint_goals:
            # 2. Send Command
            self.send_trajectory(joint_goals)
        else:
            self.get_logger().error("IK Failed: Target out of workspace?")

    def compute_stretch_ik(self, pose):
        """
        Simplified Analytical IK for Hello Robot Stretch.
        Decomposes Cartesian Pose (x,y,z) into (lift_height, arm_extension, base_theta).

        Note: This ignores full 6D orientation for the mobile base, focusing on
        positional reachability for the grasp center.
        """
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z

        # 1. Lift Height (Z)
        # Stretch lift moves linearly in Z
        lift_height = z + self.Z_LIFT_OFFSET

        # Limit check (Standard Stretch 3: 0.2m to 1.1m)
        if not (0.2 < lift_height < 1.1):
            self.get_logger().warn(f"Target Z {lift_height:.2f} out of Lift range.")
            # return None # Uncomment to enforce strict safety

        # 2. Arm Extension & Base Rotation
        # The arm extends from the mast. We calculate the distance from base (0,0) to target (x,y).
        # We assume the base rotates to face the target so the arm can extend straight to it.

        dist_to_target = math.sqrt(x**2 + y**2)

        # Arm extension needed = Total Distance - Offset of arm/wrist
        arm_extension = dist_to_target - self.WRIST_EXTENSION_OFFSET

        # Limit check (Standard Stretch 3: 0.0m to 0.5m)
        if not (0.0 <= arm_extension <= 0.52):
            self.get_logger().warn(
                f"Target Radius {dist_to_target:.2f} out of Arm range."
            )
            # return None

        # Base Yaw: Point the arm towards the object
        # Note: Stretch arm extends to the RIGHT of the base link usually,
        # but usually the driver compensates so 'translate_mobile_base' works in X.
        # Here we assume we want to orient the base such that the arm aligns.
        yaw = math.atan2(y, x)

        # 3. Wrist Orientation (Simplified)
        # Align gripper to object using standard yaw
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rot = Rotation.from_quat(q)
        # Extract yaw for the wrist
        wrist_yaw = rot.as_euler("xyz")[2]

        self.get_logger().info(
            f"IK Solution: Lift={lift_height:.2f}, Ext={arm_extension:.2f}, Yaw={yaw:.2f}"
        )

        return {
            "joint_lift": lift_height,
            "joint_arm_l0": arm_extension / 4,  # Telescoping joints split load
            "joint_arm_l1": arm_extension / 4,
            "joint_arm_l2": arm_extension / 4,
            "joint_arm_l3": arm_extension / 4,
            "joint_wrist_yaw": wrist_yaw,
            # 'joint_head_pan': 0.0,
            # 'joint_head_tilt': -0.5
        }

    def send_trajectory(self, joint_goals):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Stretch Controller Action Server not available!")
            return

        goal_msg = FollowJointTrajectory.Goal()

        # Define joints to control
        # Note: Mobile base is usually controlled via /cmd_vel or separate action
        # This trajectory controls the ARM and LIFT.
        joint_names = list(joint_goals.keys())
        goal_msg.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(joint_goals[name]) for name in joint_names]
        point.time_from_start.sec = 5  # Take 5 seconds to reach goal

        goal_msg.trajectory.points = [point]

        self.get_logger().info("Sending Trajectory...")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected :(")
            return
        self.get_logger().info("Goal accepted :)")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result}")


def main(args=None):
    rclpy.init(args=args)
    executor = StretchGraspExecutor()
    rclpy.spin(executor)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
