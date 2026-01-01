#!/usr/bin/env python3
import argparse
import logging
import os
import random
import time
from time import time as wall_time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

from dataset.config import get_camera_intrinsic
from dataset.evaluation import (
    anchor_output_process,
    collision_detect,
    detect_2d_grasp,
    detect_6d_grasp_multi,
)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from train_utils import *

# ROS2 (publishing)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", default=None)

# image input
parser.add_argument("--rgb-path", required=True)
parser.add_argument("--depth-path", required=True)

# 2d
parser.add_argument("--input-h", type=int, required=True)
parser.add_argument("--input-w", type=int, required=True)
parser.add_argument("--sigma", type=int, default=10)
parser.add_argument("--use-depth", type=int, default=1)
parser.add_argument("--use-rgb", type=int, default=1)
parser.add_argument("--ratio", type=int, default=8)
parser.add_argument("--anchor-k", type=int, default=6)
parser.add_argument("--anchor-w", type=float, default=50.0)
parser.add_argument("--anchor-z", type=float, default=20.0)
parser.add_argument("--grid-size", type=int, default=8)

# pc
parser.add_argument("--anchor-num", type=int, required=True)
parser.add_argument("--all-points-num", type=int, required=True)
parser.add_argument("--center-num", type=int, required=True)
parser.add_argument("--group-num", type=int, required=True)

# grasp detection
parser.add_argument("--heatmap-thres", type=float, default=0.01)
parser.add_argument("--local-k", type=int, default=10)
parser.add_argument("--local-thres", type=float, default=0.01)
parser.add_argument("--rotation-num", type=int, default=1)

# others
parser.add_argument("--random-seed", type=int, default=123, help="Random seed")

# Publishing options
parser.add_argument("--publish", type=int, default=1, help="1=publish, 0=don’t publish")
parser.add_argument("--publish-hz", type=float, default=10.0, help="Publish rate (Hz)")
parser.add_argument("--publish-seconds", type=float, default=5.0,
                    help="How long to publish then exit (seconds). Use 0 to publish forever.")
parser.add_argument("--frame-id", default="camera_color_frame", help="PoseStamped frame_id")
parser.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"],
                    help="Order stored in pred_gg.rotations. geometry_msgs expects x,y,z,w.")
parser.add_argument("--pose-topic", default="/hggd_grasp_pose", help="PoseStamped topic")
parser.add_argument("--width-topic", default="/hggd_grasp_width_m", help="Float32 width topic (meters)")
parser.add_argument("--score-topic", default="/hggd_grasp_score", help="Float32 score topic")

# IMPORTANT: allow running with --ros-args etc.
args, _unknown = parser.parse_known_args()


def _to_flat_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def print_top_grasps(pred_gg, k: int = 3):
    if pred_gg is None or len(pred_gg) == 0:
        print("No grasp candidates to display.")
        return

    scores = [float(_to_flat_np(pred_gg.scores[i])[0]) for i in range(len(pred_gg))]
    order = np.argsort(scores)[::-1]
    top = min(k, len(pred_gg))

    print(f"\nTop-{top} grasp poses (camera frame):")
    for rank in range(top):
        i = int(order[rank])
        pos = _to_flat_np(pred_gg.translations[i])[:3]
        quat = _to_flat_np(pred_gg.rotations[i])[:4]
        w = float(_to_flat_np(pred_gg.widths[i])[0])
        s = float(_to_flat_np(pred_gg.scores[i])[0])

        print(
            f" #{rank+1}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m | "
            f"quat(raw)=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}) | "
            f"width={w:.3f} m | score={s:.3f}"
        )
    print("-" * 72)


def extract_best_grasp(pred_gg):
    """
    Return best grasp (pos_xyz, quat_xyzw, width, score).
    """
    if pred_gg is None or len(pred_gg) == 0:
        return None

    scores = [float(_to_flat_np(pred_gg.scores[i])[0]) for i in range(len(pred_gg))]
    best_i = int(np.argmax(scores))

    pos = _to_flat_np(pred_gg.translations[best_i])[:3].astype(float)
    quat = _to_flat_np(pred_gg.rotations[best_i])[:4].astype(float)
    width = float(_to_flat_np(pred_gg.widths[best_i])[0])
    score = float(_to_flat_np(pred_gg.scores[best_i])[0])

    # convert to ROS xyzw
    if args.quat_order == "wxyz":
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=float)
    else:  # xyzw
        qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=float)

    return pos, quat_xyzw, width, score


class GraspPublisher(Node):
    """
    Publishes:
      - PoseStamped grasp pose
      - Float32 grasp width (meters)
      - Float32 grasp score

    Uses TRANSIENT_LOCAL so a late subscriber still gets the last message (latch-like).
    """
    def __init__(self,
                 pose_xyz,
                 quat_xyzw,
                 width_m: float,
                 score: float,
                 frame_id: str,
                 publish_hz: float,
                 pose_topic: str,
                 width_topic: str,
                 score_topic: str):
        super().__init__("hggd_grasp_publisher")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latch-like
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, qos)
        self.width_pub = self.create_publisher(Float32, width_topic, qos)
        self.score_pub = self.create_publisher(Float32, score_topic, qos)

        self.pose_xyz = pose_xyz
        self.quat_xyzw = quat_xyzw
        self.width_m = float(width_m)
        self.score = float(score)
        self.frame_id = frame_id

        period = 1.0 / max(0.1, float(publish_hz))
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"Publishing:\n"
            f"  pose : {pose_topic}\n"
            f"  width: {width_topic}\n"
            f"  score: {score_topic}\n"
            f"at {publish_hz:.1f} Hz, frame_id='{frame_id}'"
        )

    def _tick(self):
        now = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(self.pose_xyz[0])
        pose_msg.pose.position.y = float(self.pose_xyz[1])
        pose_msg.pose.position.z = float(self.pose_xyz[2])
        pose_msg.pose.orientation.x = float(self.quat_xyzw[0])
        pose_msg.pose.orientation.y = float(self.quat_xyzw[1])
        pose_msg.pose.orientation.z = float(self.quat_xyzw[2])
        pose_msg.pose.orientation.w = float(self.quat_xyzw[3])

        width_msg = Float32()
        width_msg.data = float(self.width_m)

        score_msg = Float32()
        score_msg.data = float(self.score)

        self.pose_pub.publish(pose_msg)
        self.width_pub.publish(width_msg)
        self.score_pub.publish(score_msg)


class PointCloudHelper:
    def __init__(self, all_points_num) -> None:
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)

        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()

        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]),
                                 np.arange(self.output_shape[0]))
        factor = 1280 / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    def to_scene_points(self, rgbs: torch.Tensor, depths: torch.Tensor, include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones((batch_size, self.all_points_num, feature_len),
                                 dtype=torch.float32).cuda()

        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs

        for i in range(batch_size):
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T

            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)), self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                idxs.append(cur_idxs)

            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points

        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode="nearest").squeeze(1).cuda()
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def inference(view_points, xyzs, x, ori_depth, vis_heatmap=False, vis_grasp=True):
    with torch.no_grad():
        pred_2d, perpoint_features = anchornet(x)

        loc_map, cls_mask, theta_offset, height_offset, width_offset = \
            anchor_output_process(*pred_2d, sigma=args.sigma)

        rect_gg = detect_2d_grasp(
            loc_map,
            cls_mask,
            theta_offset,
            height_offset,
            width_offset,
            ratio=args.ratio,
            anchor_k=args.anchor_k,
            anchor_w=args.anchor_w,
            anchor_z=args.anchor_z,
            mask_thre=args.heatmap_thres,
            center_num=args.center_num,
            grid_size=args.grid_size,
            grasp_nms=args.grid_size,
            reduce="max",
        )

        if rect_gg.size == 0:
            print("No 2d grasp found")
            return None

        if vis_heatmap:
            rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
            resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
            resized_rgb = np.array(resized_rgb.resize((args.input_w, args.input_h))) / 255.0
            depth_t = ori_depth.cpu().numpy().squeeze().T
            plt.subplot(221); plt.imshow(rgb_t)
            plt.subplot(222); plt.imshow(depth_t)
            plt.subplot(223); plt.imshow(loc_map.squeeze().T, cmap="jet")
            plt.subplot(224)
            rect_rgb = rect_gg.plot_rect_grasp_group(resized_rgb, 0)
            plt.imshow(rect_rgb)
            plt.tight_layout()
            plt.show()

        points_all = feature_fusion(view_points[..., :3], perpoint_features, xyzs)

        rect_ggs = [rect_gg]
        pc_group, valid_local_centers = data_process(
            points_all,
            ori_depth,
            rect_ggs,
            args.center_num,
            args.group_num,
            (args.input_w, args.input_h),
            min_points=32,
            is_training=False,
        )

        rect_gg = rect_ggs[0]
        points_all = points_all.squeeze()

        grasp_info = np.zeros((0, 3), dtype=np.float32)
        g_thetas = rect_gg.thetas[None]
        g_ws = rect_gg.widths[None]
        g_ds = rect_gg.depths[None]
        cur_info = np.vstack([g_thetas, g_ws, g_ds])
        grasp_info = np.vstack([grasp_info, cur_info.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device="cuda")

        _, pred, offset = localnet(pc_group, grasp_info)

        _, pred_rect_gg = detect_6d_grasp_multi(
            rect_gg,
            pred,
            offset,
            valid_local_centers,
            (args.input_w, args.input_h),
            anchors,
            k=args.local_k,
        )

        pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        pred_gg, _ = collision_detect(points_all, pred_grasp_from_rect, mode="graspnet")

        pred_gg = pred_gg.nms()

        if vis_grasp:
            print("pred grasp num ==", len(pred_gg))
            grasp_geo = pred_gg.to_open3d_geometry_list()
            points = view_points[..., :3].cpu().numpy().squeeze()
            colors = view_points[..., 3:6].cpu().numpy().squeeze()
            vispc = o3d.geometry.PointCloud()
            vispc.points = o3d.utility.Vector3dVector(points)
            vispc.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([vispc] + grasp_geo)

        return pred_gg


if __name__ == "__main__":
    pc_helper = PointCloudHelper(all_points_num=args.all_points_num)

    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError("CUDA not available")

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    anchornet = AnchorGraspNet(in_dim=4, ratio=args.ratio, anchor_k=args.anchor_k).cuda()
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2).cuda()

    if args.checkpoint_path is None:
        raise RuntimeError("--checkpoint-path is required")
    check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point["anchor"])
    localnet.load_state_dict(check_point["local"])

    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {"gamma": basic_anchors, "beta": basic_anchors}
    anchors["gamma"] = check_point["gamma"]
    anchors["beta"] = check_point["beta"]
    logging.info("Using saved anchors")
    print("-> loaded checkpoint %s " % (args.checkpoint_path))

    anchornet.eval()
    localnet.eval()

    # read image and convert to tensor
    ori_depth_np = np.array(Image.open(args.depth_path))
    ori_rgb_np = np.array(Image.open(args.rgb_path)) / 255.0

    # depth in [0..1000] mm-ish in this codepath
    ori_depth_np = np.clip(ori_depth_np, 0, 1000).astype(np.float32)

    ori_rgb = torch.from_numpy(ori_rgb_np).permute(2, 1, 0)[None].to(device="cuda", dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth_np).T[None].to(device="cuda", dtype=torch.float32)

    view_points, _, _ = pc_helper.to_scene_points(ori_rgb, ori_depth, include_rgb=True)
    xyzs = pc_helper.to_xyz_maps(ori_depth)

    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)

    x = torch.concat([depth[None], rgb], 1).to(device="cuda", dtype=torch.float32)

    pred_gg = inference(view_points, xyzs, x, ori_depth, vis_heatmap=True, vis_grasp=True)

    print_top_grasps(pred_gg, k=3)

    best = extract_best_grasp(pred_gg)
    if best is None:
        print("No valid grasp to publish.")
    else:
        best_pos, best_quat_xyzw, best_width, best_score = best
        print(f"[BEST] pos={best_pos} quat_xyzw={best_quat_xyzw} width={best_width:.4f} score={best_score:.4f}")

        if args.publish == 1:
            if not ROS_AVAILABLE:
                print("ROS2 not available (rclpy import failed). Source ROS2 or install rclpy.")
            else:
                rclpy.init()
                node = GraspPublisher(
                    best_pos,
                    best_quat_xyzw,
                    best_width,
                    best_score,
                    args.frame_id,
                    args.publish_hz,
                    args.pose_topic,
                    args.width_topic,
                    args.score_topic,
                )

                print(
                    f"Publishing best grasp:\n"
                    f"  Pose : {args.pose_topic}\n"
                    f"  Width: {args.width_topic}\n"
                    f"  Score: {args.score_topic}\n"
                    f"frame_id='{args.frame_id}'."
                )

                try:
                    if args.publish_seconds and args.publish_seconds > 0.0:
                        t_end = wall_time() + float(args.publish_seconds)
                        while rclpy.ok() and wall_time() < t_end:
                            rclpy.spin_once(node, timeout_sec=0.1)
                        print(f"Done publishing for {args.publish_seconds:.1f}s, exiting.")
                    else:
                        print("Publishing forever. Ctrl+C to stop.")
                        rclpy.spin(node)
                except KeyboardInterrupt:
                    pass
                node.destroy_node()
                rclpy.shutdown()

    # time test
    start = wall_time()
    T = 100
    for _ in range(T):
        _ = inference(view_points, xyzs, x, ori_depth, vis_heatmap=False, vis_grasp=False)
        torch.cuda.synchronize()
    print("avg time ==", (wall_time() - start) / T * 1e3, "ms")
