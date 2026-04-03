#!/usr/bin/env python3


import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField


# ── 图像处理参数（与 deploy_mujoco/configs/g1_16dof_resi_moe.yaml 一致） ──────
DEPTH_NEAR_CLIP        = 0.0
DEPTH_FAR_CLIP         = 2.0
DEPTH_OUTPUT_SIZE      = (64, 64)          # (height, width) after crop+resize
CROP_SIZE              = [10, 20, 10, 2]   # [left, top, right, bottom]
DEPTH_INPUT_WIDTH      = 64               # rot90 后图像的宽度（用于右裁剪边界）
GAUSSIAN_FILTER        = False
GAUSSIAN_FILTER_KERNEL = 5
GAUSSIAN_FILTER_SIGMA  = 1.5
GAUSSIAN_NOISE         = False
GAUSSIAN_NOISE_STD     = 0.05
DEPTH_DIS_NOISE        = 0.0


def img_process(depth_image: np.ndarray) -> np.ndarray:
    """
    对已旋转后的 float32 深度图执行处理。
    流程与 deploy_mujoco/deploy_mujoco_with_resi.py 中一致：
      1. 位移噪声（depth_dis_noise=0，无效果）
      2. 高斯噪声（gaussian_noise=False，跳过）
      3. Clip 到 [depth_near_clip, depth_far_clip]
      4. 归一化到 [-0.5, 0.5]
      5. 裁剪 + bilinear resize 到 (64, 64)
      6. 高斯模糊（gaussian_filter=False，跳过）
    """
    img = depth_image.copy().astype(np.float32)

    # ── 1. 位移噪声 ───────────────────────────────────────────────────────────
    img += DEPTH_DIS_NOISE * 2 * (np.random.rand(1) - 0.5)

    # ── 2. 高斯像素噪声 ───────────────────────────────────────────────────────
    if GAUSSIAN_NOISE:
        img += GAUSSIAN_NOISE_STD * np.random.randn(*img.shape).astype(np.float32)

    # ── 3. Clip ───────────────────────────────────────────────────────────────
    img = np.clip(img, DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP)

    # ── 4. 归一化到 [-0.5, 0.5] ──────────────────────────────────────────────
    img = (img - DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP) - 0.5

    # ── 5. 裁剪 + bilinear resize 到 DEPTH_OUTPUT_SIZE ───────────────────────
    clip_left, clip_top, clip_right, clip_bottom = CROP_SIZE
    h = img.shape[0]
    cropped = img[clip_top:h - clip_bottom, clip_left:DEPTH_INPUT_WIDTH - clip_right]
    img = cv2.resize(cropped, (DEPTH_OUTPUT_SIZE[1], DEPTH_OUTPUT_SIZE[0]),
                     interpolation=cv2.INTER_LINEAR)

    # ── 6. 高斯模糊 ───────────────────────────────────────────────────────────
    if GAUSSIAN_FILTER:
        img = cv2.GaussianBlur(img,
                               (GAUSSIAN_FILTER_KERNEL, GAUSSIAN_FILTER_KERNEL),
                               GAUSSIAN_FILTER_SIGMA)

    return img.astype(np.float32)



def depth_to_pointcloud2(depth: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """
    将 2D 归一化深度图打包为 PointCloud2（单字段 z，float32）。
    height=1, width=H*W, point_step=4
    """
    total = depth.size
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height          = 1
    msg.width           = total
    msg.is_bigendian    = False
    msg.point_step      = 4
    msg.row_step        = 4 * total
    msg.is_dense        = True
    msg.fields          = [
        PointField(name="z", offset=0, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = depth.flatten().astype(np.float32).tobytes()
    return msg



class SimRealsenseNode(Node):

    def __init__(self):
        super().__init__("sim_realsense_node")

        self.img_width  = 64   
        self.img_height = 64   

        self.subscription = self.create_subscription(PointCloud2, '/camera/depth', self.callback, 10)
        self.point_pub = self.create_publisher(PointCloud2, "/camera/processed_depth_cloud", 10)
        self.img_pub = self.create_publisher(Image, '/camera/processed_image', 10)

        self.get_logger().info(f"SimRealsenseNode started. ")

    # ── ROS callback ─────────────────────────────────────────────────────────

    def callback(self, msg: PointCloud2) -> None:
        raw_bytes  = bytes(msg.data)
        depth_flat = np.frombuffer(raw_bytes, dtype=np.float32)

        # ── 数据长度检查 ──────────────────────────────────────────────────────
        expected = self.img_width * self.img_height
        if len(depth_flat) != expected:
            self.get_logger().warn(
                f"Unexpected data length: {len(depth_flat)}, expected {expected}",
                throttle_duration_sec=5.0,
            )
            return

        # ── reshape: (height=64, width=64) ───────────────────────────────────
        depth_image = depth_flat.reshape(self.img_height, self.img_width).copy()
        # ── 处理无效值（rangefinder 无效时返回 -1） ───────────────────────────
        invalid_mask_orig = depth_image < 0
        depth_image[invalid_mask_orig] = 0.0

        # ── 限制最大深度 ──────────────────────────────────────────────────────
        # depth_image[depth_image > 10.0] = 10.0

        # ── 逆时针旋转 90° ────────────────────────────────────────
        depth_image = np.rot90(depth_image, k=1)


        # ── 调试打印（旋转后、处理前） ────────────────────────────────────────
        # cy, cx = depth_image.shape[0] // 2, depth_image.shape[1] // 2
        # valid_pixels = depth_image[depth_image > 0]
        # print(f"[raw rot90] center={depth_image[cy, cx]:.4f} m  "
        #       f"valid={len(valid_pixels)}/{expected}  "
        #       f"range=[{valid_pixels.min() if len(valid_pixels) else 0:.3f}, "
        #       f"{depth_image.max():.3f}]")

        processed = img_process(depth_image)

        # ── 发布处理后点云（PointCloud2） ─────────────────────────────────────
        pc_msg = depth_to_pointcloud2(processed, msg.header.stamp, msg.header.frame_id)
        self.point_pub.publish(pc_msg)

        # ── 发布处理后深度图 ───────────────────────────────────
        # img_msg = Image()
        # img_msg.header.stamp    = msg.header.stamp
        # img_msg.header.frame_id = msg.header.frame_id
        # img_msg.height          = processed.shape[0]   # 18
        # img_msg.width           = processed.shape[1]   # 32
        # img_msg.encoding        = "32FC1"
        # img_msg.is_bigendian    = False
        # img_msg.step            = processed.shape[1] * 4
        # img_msg.data            = processed.astype(np.float32).tobytes()
        # self.img_pub.publish(img_msg)
    
        img_msg = Image()
        img_msg.header.stamp    = msg.header.stamp
        img_msg.header.frame_id = msg.header.frame_id
        img_msg.height          = depth_image.shape[0]   # 36
        img_msg.width           = depth_image.shape[1]   # 64
        img_msg.encoding        = "32FC1"
        img_msg.is_bigendian    = False
        img_msg.step            = depth_image.shape[1] * 4
        img_msg.data            = depth_image.astype(np.float32).tobytes()
        self.img_pub.publish(img_msg)

        # ── cv2 可视化────────────────────────
        self._visualize(depth_image, processed)


    def _visualize(self, raw_rot: np.ndarray, processed: np.ndarray) -> None:
        scale = 8
        # --- 旋转后原始深度（处理前） ---
        valid = raw_rot > 0
        if valid.any():
            d_min, d_max = raw_rot[valid].min(), raw_rot[valid].max()
            norm_raw = np.zeros_like(raw_rot, dtype=np.float32)
            norm_raw[valid] = (raw_rot[valid] - d_min) / (d_max - d_min + 1e-6)
        else:
            norm_raw = np.zeros_like(raw_rot, dtype=np.float32)

        vis_raw  = (norm_raw * 255).astype(np.uint8)
        color_raw = cv2.applyColorMap(vis_raw, cv2.COLORMAP_JET)
        color_raw[~valid] = 0
        rh, rw = raw_rot.shape
        cv2.imshow("raw image (colormap)",cv2.resize(color_raw, (rw * scale, rh * scale), interpolation=cv2.INTER_NEAREST),)

        # --- 处理后深度（已归一化 -0.5~0.5，平移到 0~1 显示） ---
        vis_proc  = ((processed + 0.5) * 255).astype(np.uint8)
        color_proc = cv2.applyColorMap(vis_proc, cv2.COLORMAP_JET)
        ph, pw = processed.shape
        cv2.imshow("processed (colormap)",cv2.resize(color_proc, (pw * scale, ph * scale), interpolation=cv2.INTER_NEAREST),)
        cv2.imshow("processed (gray)",cv2.resize(vis_proc, (pw * scale, ph * scale), interpolation=cv2.INTER_NEAREST),)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = SimRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down.")
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
