#!/usr/bin/env python3
"""
Capture ONE synchronized RGB + depth frame from ROS2 topics and save them
in HGGD's expected images folder:

  /ros2_ws/src/hggd_humble/HGGD/images/demo_rgb.png
  /ros2_ws/src/hggd_humble/HGGD/images/demo_depth.png   (uint16, millimeters)

Topics (defaults):
  /wrist_mounted_camera/image        encoding: rgb8
  /wrist_mounted_camera/depth_image  encoding: 32FC1 (meters)
"""

import argparse
from pathlib import Path

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


def find_hggd_images_dir() -> Path:
    """
    Try to locate /ros2_ws/src/hggd_humble/HGGD/images by walking upward
    from this file location. Fallback to the known absolute path if not found.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        candidate = p / "HGGD" / "images"
        if candidate.is_dir():
            return candidate

        # common layout: .../hggd_humble/grasp_pose_detection/grasp_pose_detection/test.py
        # so parents may contain "hggd_humble"
        if p.name == "hggd_humble":
            candidate2 = p / "HGGD" / "images"
            if candidate2.is_dir():
                return candidate2

    # fallback (your container paths)
    fallback = Path("/ros2_ws/src/hggd_humble/HGGD/images")
    return fallback


def depth_m_to_u16mm(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """
    Convert float depth in meters (32FC1) -> uint16 depth in millimeters.
    Invalid (nan/inf/out-of-range) becomes 0.
    """
    d = depth_m.astype(np.float32, copy=False)

    # squeeze HxWx1 -> HxW if needed
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[:, :, 0]

    # clean invalids
    finite = np.isfinite(d)
    d_clean = np.zeros_like(d, dtype=np.float32)
    d_clean[finite] = d[finite]

    # clip / invalidate outside range
    valid = (d_clean >= float(min_m)) & (d_clean <= float(max_m))
    d_clean = np.where(valid, d_clean, 0.0).astype(np.float32)

    # meters -> mm
    mm = np.round(d_clean * 1000.0).astype(np.uint16)
    return mm


class HGGDCaptureImages(Node):
    def __init__(self, args):
        super().__init__("hggd_capture_images")
        self.args = args
        self.bridge = CvBridge()
        self.done = False

        # Output directory
        if args.out_dir:
            self.out_dir = Path(args.out_dir).expanduser().resolve()
        else:
            self.out_dir = find_hggd_images_dir()

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rgb_path = self.out_dir / "ros2_rgb.png"
        self.depth_path = self.out_dir / "ros2_depth.png"

        self.get_logger().info(f"Saving RGB  to: {self.rgb_path}")
        self.get_logger().info(f"Saving Depth to: {self.depth_path}")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.color_sub = Subscriber(self, RosImage, args.color_topic, qos_profile=qos)
        self.depth_sub = Subscriber(self, RosImage, args.depth_topic, qos_profile=qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=args.sync_slop
        )
        self.sync.registerCallback(self.cb)

        self.get_logger().info("Waiting for one synchronized RGB+Depth pair...")

    def cb(self, color_msg: RosImage, depth_msg: RosImage):
        if self.done:
            return
        self.done = True

        self.get_logger().info(
            f"Got pair. enc_color={color_msg.encoding} enc_depth={depth_msg.encoding} "
            f"size_color=({color_msg.height}x{color_msg.width}) size_depth=({depth_msg.height}x{depth_msg.width})"
        )

        # RGB: force rgb8
        try:
            rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            self.done = False
            return

        # Depth: passthrough (your sim publishes 32FC1 meters)
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")
            self.done = False
            return

        # Ensure depth is float32 meters
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        # If sizes mismatch, resize depth to RGB (nearest to preserve values)
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            self.get_logger().warn(
                f"RGB size {rgb.shape[:2]} != Depth size {depth.shape[:2]}. Resizing depth to RGB."
            )
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert depth to uint16 mm (HGGD sample depth is 16-bit)
        depth_u16 = depth_m_to_u16mm(depth, self.args.depth_min_m, self.args.depth_max_m)

        # Save: RGB as PNG (write via OpenCV expects BGR)
        ok_rgb = cv2.imwrite(str(self.rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        ok_d = cv2.imwrite(str(self.depth_path), depth_u16)

        if not ok_rgb:
            self.get_logger().error(f"Failed to write {self.rgb_path}")
            self.done = False
            return
        if not ok_d:
            self.get_logger().error(f"Failed to write {self.depth_path}")
            self.done = False
            return

        self.get_logger().info(" Saved one RGB+Depth pair. Exiting.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_topic", default="/wrist_mounted_camera/image")
    parser.add_argument("--depth_topic", default="/wrist_mounted_camera/depth_image")
    parser.add_argument("--sync_slop", type=float, default=0.1)

    # Output folder (default auto-finds /hggd_humble/HGGD/images)
    parser.add_argument("--out_dir", default="", help="Override output dir (default: HGGD/images)")

    # Depth range to keep (meters). Outside becomes 0 (invalid).
    parser.add_argument("--depth_min_m", type=float, default=0.05)
    parser.add_argument("--depth_max_m", type=float, default=2.0)

    args = parser.parse_args()

    rclpy.init()
    node = HGGDCaptureImages(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
