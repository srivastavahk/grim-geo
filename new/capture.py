import os

import cv2
import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DataCaptureNode(Node):
    def __init__(self):
        super().__init__("data_capture_node")
        self.bridge = CvBridge()

        # 1. Define Topics
        # We use the ALIGNED depth topic. This is critical.
        # It ensures pixel (100,100) in color is the same physical point as (100,100) in depth.
        rgb_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"

        self.get_logger().info(f"Subscribing to {rgb_topic} and {depth_topic}...")

        # 2. Setup Synchronizer
        # We need to grab color and depth from the exact same moment.
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)

        # ApproximateTimeSynchronizer allows slight timestamp jitter (common in USB cameras)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.latest_rgb = None
        self.latest_depth = None

    def callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS messages to OpenCV images
            # RGB: "bgr8" for OpenCV display
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

            # Depth: "16UC1" (16-bit Unsigned Single Channel) -> mm
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

            # Show Preview
            cv2.imshow("Preview (Press 's' to save, 'q' to quit)", self.latest_rgb)

            key = cv2.waitKey(1)
            if key == ord("s"):
                self.save_data()
            elif key == ord("q"):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")

    def save_data(self):
        if self.latest_rgb is None:
            return

        print("Saving images...")
        # Save RGB
        cv2.imwrite("capture_rgb.png", self.latest_rgb)

        # Save Depth (PNG format preserves 16-bit values correctly)
        cv2.imwrite("capture_depth.png", self.latest_depth)

        print(f"Saved 'capture_rgb.png' and 'capture_depth.png' in {os.getcwd()}")
        print("You can now close the script.")


def main(args=None):
    rclpy.init(args=args)
    node = DataCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
