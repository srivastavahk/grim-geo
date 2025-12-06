import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import sys

class HeadMover(Node):
    def __init__(self):
        super().__init__('head_mover')
        self._action_client = ActionClient(
            self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')

    def move(self, tilt, pan):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_head_tilt', 'joint_head_pan']

        point = JointTrajectoryPoint()
        point.positions = [float(tilt), float(pan)]
        point.time_from_start.sec = 2 # Move in 2 seconds

        goal_msg.trajectory.points = [point]

        self.get_logger().info(f'Moving Head: Tilt={tilt}, Pan={pan}...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, self._send_goal_future)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 move_head.py [tilt] [pan]")
        print("Example: python3 move_head.py -0.5 0.0")
        return

    rclpy.init()
    mover = HeadMover()

    tilt = sys.argv[1]
    pan = sys.argv[2]

    mover.move(tilt, pan)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
