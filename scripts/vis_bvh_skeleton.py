#!/usr/bin/env python3
"""
BVH 人体骨架运动可视化脚本

加载 BVH 动作数据并将人体骨架运动以 3D 骨骼+关节的形式可视化。
支持 lafan1 和 nokov 两种 BVH 格式。

用法:
    python scripts/vis_bvh_skeleton.py --bvh_file <path/to/motion.bvh>
    python scripts/vis_bvh_skeleton.py --bvh_file <path/to/motion.bvh> --record_video --video_path output.mp4
"""

import argparse
import json
import os
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor import utils as lafan_utils


# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------
JOINT_COLOR = np.array([0.2, 0.6, 0.9, 0.9])       # 关节球颜色 (蓝)
BONE_COLOR = np.array([0.3, 0.3, 0.3, 0.7])         # 骨骼连线颜色 (灰)
ROOT_COLOR = np.array([0.9, 0.3, 0.2, 0.95])        # 根关节颜色 (红)
LEFT_LIMB_COLOR = np.array([0.2, 0.7, 0.3, 0.8])    # 左侧肢体颜色 (绿)
RIGHT_LIMB_COLOR = np.array([0.8, 0.4, 0.1, 0.8])   # 右侧肢体颜色 (橙)

# 左侧骨骼名称关键字
LEFT_KEYWORDS = ["Left", "left", "L_", "l_"]
# 右侧骨骼名称关键字
RIGHT_KEYWORDS = ["Right", "right", "R_", "r_"]

# 最小 MuJoCo 场景 XML
MINIMAL_SCENE_XML = """<mujoco>
  <worldbody>
    <light name="light1" pos="3 3 5" dir="-0.5 -0.5 -1"/>
    <light name="light2" pos="-3 -3 3" dir="0.3 0.3 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
  </worldbody>
</mujoco>"""


def get_bone_color(bone_name: str) -> np.ndarray:
    """根据骨骼名称返回对应颜色：左侧绿、右侧橙、躯干灰"""
    for kw in LEFT_KEYWORDS:
        if kw in bone_name:
            return LEFT_LIMB_COLOR
    for kw in RIGHT_KEYWORDS:
        if kw in bone_name:
            return RIGHT_LIMB_COLOR
    return BONE_COLOR


def load_bvh_global_positions(bvh_file: str) -> tuple:
    """
    加载 BVH 文件并返回逐帧全局关节位置与旋转（四元数）。

    Returns:
        frames_global_pos: list of dict, {bone_name: np.array([x,y,z])}
        frames_global_quat: list of dict, {bone_name: np.array([x,y,z,w])} (scipy xyzw)
        bones: list of str
        parents: np.array
        fps: float
    """
    anim = read_bvh(bvh_file)
    global_quats, global_positions = lafan_utils.quat_fk(
        anim.quats, anim.pos, anim.parents
    )

    # BVH Y-up -> MuJoCo Z-up
    rot_conv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rot_correction = R.from_matrix(rot_conv)

    num_frames = anim.pos.shape[0]
    frames_global_pos = []
    frames_global_quat = []
    for f in range(num_frames):
        pos_data = {}
        quat_data = {}
        for i, bone in enumerate(anim.bones):
            pos = global_positions[f, i] @ rot_conv.T / 100.0
            pos_data[bone] = pos
            # global_quats: scalar-first (w,x,y,z) -> scipy xyzw
            q = global_quats[f, i]
            rot = rot_correction * R.from_quat([q[1], q[2], q[3], q[0]])
            quat_data[bone] = rot.as_quat()  # xyzw
        frames_global_pos.append(pos_data)
        frames_global_quat.append(quat_data)

    fps = _parse_bvh_frame_time(bvh_file)
    if fps is None:
        fps = 30.0

    return frames_global_pos, frames_global_quat, anim.bones, anim.parents, fps


def _parse_bvh_frame_time(bvh_file: str) -> float:
    """从 BVH 文件中解析 Frame Time，失败返回 None"""
    import re
    try:
        with open(bvh_file, "r") as f:
            content = f.read()
        match = re.search(r"Frame Time:\s*([\d\.]+)", content)
        if match:
            return 1.0 / float(match.group(1))
    except Exception:
        pass
    return None


# 脚掌平面默认参数
FOOT_PLANE_WIDTH = 0.08
FOOT_PLANE_ALPHA = 0.55


def draw_foot_planes(viewer, global_positions, global_quats, foot_calib,
                      alpha=FOOT_PLANE_ALPHA):
    """
    根据标定数据渲染脚掌矩形面。

    对每只脚，用踝关节全局旋转将零位标定的四角相对坐标变换到世界系，
    然后用四条边连线绘制矩形。

    公式: corner_world = ankle_pos + ankle_rot @ corner_rel
    """
    for pair in foot_calib["foot_pairs"]:
        ankle_name = pair["ankle"]
        ankle_idx = pair["ankle_idx"]
        corners_rel = [np.array(c) for c in pair["corners_rel"]]

        ankle_pos = global_positions.get(ankle_name)
        ankle_quat = global_quats.get(ankle_name)
        if ankle_pos is None or ankle_quat is None:
            continue

        # 踝全局旋转矩阵
        ankle_rot = R.from_quat(ankle_quat)

        # 将四个角变换到世界系
        corners_world = [ankle_pos + ankle_rot.apply(c) for c in corners_rel]

        # 确定颜色
        is_left = any(kw in ankle_name for kw in LEFT_KEYWORDS)
        if is_left:
            color = LEFT_LIMB_COLOR
        elif any(kw in ankle_name for kw in RIGHT_KEYWORDS):
            color = RIGHT_LIMB_COLOR
        else:
            color = BONE_COLOR
        rgba = np.array([color[0], color[1], color[2], alpha])

        # 四条边连线
        for k in range(4):
            p1 = corners_world[k]
            p2 = corners_world[(k + 1) % 4]
            mj.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mj.mjtGeom.mjGEOM_CAPSULE,
                width=0.012,
                from_=p1,
                to=p2,
            )
            for ch in range(4):
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[ch] = rgba[ch]
            viewer.user_scn.ngeom += 1


def draw_skeleton_frame(viewer, bones, parents, global_positions, joint_radius=0.03):
    """
    在 MuJoCo viewer 中绘制一帧骨架。

    Args:
        viewer: mjv 句柄
        bones: 骨骼名称列表
        parents: 父骨骼索引数组
        global_positions: {bone_name: np.array([x,y,z])}
        joint_radius: 关节球半径
    """
    # 清除上一帧的自定义几何体
    viewer.user_scn.ngeom = 0

    # 1. 绘制骨骼连线 (从子关节连向父关节)
    for i, (bone, parent_idx) in enumerate(zip(bones, parents)):
        if parent_idx < 0:
            continue  # 根骨骼没有父骨骼
        parent_bone = bones[parent_idx]
        child_pos = global_positions[bone]
        parent_pos = global_positions[parent_bone]

        color = get_bone_color(bone)
        mj.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            width=0.025,
            from_=parent_pos,
            to=child_pos,
        )
        # 设置颜色
        for k in range(3):
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[k] = color[k]
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[3] = color[3]
        viewer.user_scn.ngeom += 1

    # 2. 绘制关节球
    for bone in bones:
        pos = global_positions[bone]
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        if bone == bones[0]:
            color = ROOT_COLOR      # 根关节
            radius = joint_radius * 1.5
        else:
            color = get_bone_color(bone)
            radius = joint_radius
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(
        description="BVH 人体骨架运动可视化"
    )
    parser.add_argument(
        "--bvh_file", type=str, required=True,
        help="BVH 动作文件路径",
    )
    parser.add_argument(
        "--format", type=str, choices=["lafan1", "nokov"], default="lafan1",
        help="BVH 格式 (默认 lafan1)",
    )
    parser.add_argument(
        "--motion_fps", type=int, default=30,
        help="播放帧率",
    )
    parser.add_argument(
        "--joint_radius", type=float, default=0.03,
        help="关节球半径 (m)",
    )
    parser.add_argument(
        "--record_video", action="store_true",
        help="录制视频",
    )
    parser.add_argument(
        "--video_path", type=str, default="videos/bvh_skeleton.mp4",
        help="视频输出路径",
    )
    parser.add_argument(
        "--video_width", type=int, default=1280,
    )
    parser.add_argument(
        "--video_height", type=int, default=720,
    )
    parser.add_argument(
        "--no_loop", action="store_true",
        help="不循环播放",
    )
    parser.add_argument(
        "--camera_distance", type=float, default=3.5,
        help="相机距离 (m)",
    )
    parser.add_argument(
        "--camera_elevation", type=float, default=-15,
        help="相机俯仰角 (度)",
    )
    parser.add_argument(
        "--camera_azimuth", type=float, default=90,
        help="相机方位角 (度)",
    )
    parser.add_argument(
        "--foot_calib", type=str, default=None,
        help="脚掌标定 JSON 路径 (由 vis_bvh_zero_pose.py --save_calib 生成)",
    )

    args = parser.parse_args()

    # --- 加载 BVH 数据 ---
    print(f"加载 BVH 文件: {args.bvh_file}")
    frames_global_pos, frames_global_quat, bones, parents, bvh_fps = \
        load_bvh_global_positions(args.bvh_file)
    num_frames = len(frames_global_pos)
    print(f"  骨骼数: {len(bones)}")
    print(f"  帧数: {num_frames}")
    print(f"  BVH 帧率: {bvh_fps:.1f}")

    # --- 加载脚掌标定 (可选) ---
    foot_calib = None
    if args.foot_calib:
        print(f"加载脚掌标定: {args.foot_calib}")
        with open(args.foot_calib, "r") as f:
            foot_calib = json.load(f)
        for pair in foot_calib.get("foot_pairs", []):
            print(f"  {pair['ankle']} -> {pair['toe']} "
                  f"(chain: {len(pair['chain_indices'])} bones, "
                  f"corners: {len(pair['corners_rel'])} pts)")

    # --- 创建 MuJoCo 场景 ---
    model = mj.MjModel.from_xml_string(MINIMAL_SCENE_XML)
    data = mj.MjData(model)

    # --- 启动交互式 viewer ---
    viewer = mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    )

    # 设置相机
    viewer.cam.distance = args.camera_distance
    viewer.cam.elevation = args.camera_elevation
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.lookat[:] = [0, 0, 0.9]  # 看向人体中心高度

    rate_limiter = RateLimiter(frequency=args.motion_fps, warn=False)

    # --- 视频录制 ---
    mp4_writer = None
    renderer = None
    if args.record_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        mp4_writer = __import__("imageio").get_writer(
            args.video_path, fps=args.motion_fps
        )
        renderer = mj.Renderer(model, height=args.video_height, width=args.video_width)
        print(f"录制视频到: {args.video_path}")

    # --- 主循环 ---
    print("\n播放中... 按 ESC 或关闭窗口退出。\n")
    frame_idx = 0
    fps_counter = 0
    fps_start = time.time()

    while viewer.is_running():
        # FPS 统计
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 2.0:
            print(f"  实际 FPS: {fps_counter / elapsed:.1f}")
            fps_counter = 0
            fps_start = time.time()

        # 获取当前帧骨骼数据
        global_positions = frames_global_pos[frame_idx]
        global_quats = frames_global_quat[frame_idx]

        # 绘制骨架
        draw_skeleton_frame(viewer, bones, parents, global_positions, args.joint_radius)

        # 绘制脚掌面
        if foot_calib:
            draw_foot_planes(viewer, global_positions, global_quats, foot_calib,
                             alpha=foot_calib.get("plane_alpha", FOOT_PLANE_ALPHA))

        # 同步
        viewer.sync()
        rate_limiter.sleep()

        # 录制
        if args.record_video and renderer is not None and mp4_writer is not None:
            renderer.update_scene(data, camera=viewer.cam)
            img = renderer.render()
            mp4_writer.append_data(img)

        # 推进帧
        frame_idx += 1
        if frame_idx >= num_frames:
            if args.no_loop:
                break
            frame_idx = 0

    # --- 清理 ---
    viewer.close()
    time.sleep(0.3)
    if mp4_writer is not None:
        mp4_writer.close()
        print(f"视频已保存: {args.video_path}")
    print("Done.")


if __name__ == "__main__":
    main()
