#!/usr/bin/env python3
"""
BVH 骨架零位（T-Pose）静态可视化脚本。

加载 BVH 文件，根据骨骼的 offset 数据计算零位（T-Pose）下各关节的全局位置，
并在 MuJoCo viewer 中静态渲染骨架。

用法:
    python scripts/vis_bvh_zero_pose.py --bvh_file <path/to/motion.bvh>
    python scripts/vis_bvh_zero_pose.py --bvh_file <path/to/motion.bvh> --record_video --video_path output.mp4
"""

import argparse
import os
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------
BONE_COLOR = np.array([0.3, 0.3, 0.3, 0.7])
JOINT_COLOR = np.array([0.2, 0.6, 0.9, 0.9])
ROOT_COLOR = np.array([0.9, 0.3, 0.2, 0.95])
LEFT_COLOR = np.array([0.2, 0.7, 0.3, 0.8])
RIGHT_COLOR = np.array([0.8, 0.4, 0.1, 0.8])

LEFT_KEYWORDS = ["Left", "left", "L_", "l_"]
RIGHT_KEYWORDS = ["Right", "right", "R_", "r_"]
FOOT_KEYWORDS = ["Toe", "toe", "Foot", "foot", "Ankle", "ankle"]

MINIMAL_SCENE_XML = """<mujoco>
  <worldbody>
    <light name="light1" pos="3 3 5" dir="-0.5 -0.5 -1"/>
    <light name="light2" pos="-3 -3 3" dir="0.3 0.3 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
  </worldbody>
</mujoco>"""


def get_bone_color(bone_name: str) -> np.ndarray:
    for kw in LEFT_KEYWORDS:
        if kw in bone_name:
            return LEFT_COLOR
    for kw in RIGHT_KEYWORDS:
        if kw in bone_name:
            return RIGHT_COLOR
    return BONE_COLOR


def align_to_ground(positions: dict, bones: list) -> dict:
    """
    将骨架平移到最低处脚趾刚好触地（z=0）。

    找到所有脚部骨骼（Foot/Toe/Ankle）中 Z 值最小的点，
    将所有关节整体上移使该点 Z=0。
    """
    foot_z_vals = []
    for bone in bones:
        if any(kw in bone for kw in FOOT_KEYWORDS):
            foot_z_vals.append(positions[bone][2])
    if not foot_z_vals:
        return positions

    min_z = min(foot_z_vals)
    offset = np.array([0.0, 0.0, -min_z])
    aligned = {b: pos + offset for b, pos in positions.items()}
    return aligned


def compute_zero_pose(bvh_file: str) -> tuple:
    """
    从 BVH 文件的 offset 数据计算零位姿态（所有旋转为 0）的全局关节位置。

    对每个骨骼，按层级累加 offset（子关节 = 父关节位置 + 子关节的 offset），
    等同于所有关节旋转为 identity 时的前向运动学结果。

    Returns:
        zero_positions: {bone_name: np.array([x,y,z])}
        bones: list
        parents: np.array
    """
    anim = read_bvh(bvh_file)
    bones = anim.bones
    parents = anim.parents
    offsets = anim.offsets  # (num_bones, 3) 单位 cm

    # BVH Y-up → MuJoCo Z-up
    rot_conv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # 前向累加 offset（零旋转下，全局位置 = 父位置 + offset）
    global_pos = np.zeros((len(bones), 3))
    for i in range(len(bones)):
        p = parents[i]
        if p < 0:
            global_pos[i] = offsets[i] @ rot_conv.T / 100.0
        else:
            global_pos[i] = global_pos[p] + offsets[i] @ rot_conv.T / 100.0

    zero_positions = {bones[i]: global_pos[i] for i in range(len(bones))}
    return zero_positions, bones, parents


def draw_skeleton(viewer, bones, parents, positions, joint_radius=0.03):
    """绘制一帧静态骨架"""
    # 骨骼连线
    for i, (bone, p) in enumerate(zip(bones, parents)):
        if p < 0:
            continue
        color = get_bone_color(bone)
        mj.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            width=0.025,
            from_=positions[bones[p]],
            to=positions[bone],
        )
        for k in range(4):
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[k] = color[k]
        viewer.user_scn.ngeom += 1

    # 关节球
    for bone in bones:
        pos = positions[bone]
        is_root = (bone == bones[0])
        color = ROOT_COLOR if is_root else get_bone_color(bone)
        radius = joint_radius * 1.5 if is_root else joint_radius
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom, type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0], pos=pos,
            mat=np.eye(3).flatten(), rgba=color,
        )
        viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="BVH 骨架零位姿态可视化")
    parser.add_argument("--bvh_file", type=str, required=True, help="BVH 文件路径")
    parser.add_argument("--joint_radius", type=float, default=0.03, help="关节球半径")
    parser.add_argument("--camera_distance", type=float, default=3.5, help="相机距离")
    parser.add_argument("--camera_elevation", type=float, default=-10, help="相机俯仰角")
    parser.add_argument("--camera_azimuth", type=float, default=90, help="相机方位角")
    parser.add_argument("--record_video", action="store_true", help="录制视频")
    parser.add_argument("--video_path", type=str, default="videos/bvh_zero_pose.mp4")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--hold_seconds", type=float, default=0,
                        help="自动退出秒数 (0=手动关闭)")
    args = parser.parse_args()

    # 加载 & 计算零位
    print(f"BVH 文件: {args.bvh_file}")
    zero_positions, bones, parents = compute_zero_pose(args.bvh_file)
    print(f"骨骼数: {len(bones)}, 层级深度: {max(parents)+1}")

    # 平移至脚趾触地
    zero_positions = align_to_ground(zero_positions, bones)

    # MuJoCo
    model = mj.MjModel.from_xml_string(MINIMAL_SCENE_XML)
    data = mj.MjData(model)
    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False)
    viewer.cam.distance = args.camera_distance
    viewer.cam.elevation = args.camera_elevation
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.lookat[:] = [0, 0, 0.9]

    # 绘制
    viewer.user_scn.ngeom = 0
    draw_skeleton(viewer, bones, parents, zero_positions, args.joint_radius)
    viewer.sync()

    # 视频
    mp4_writer = None
    renderer = None
    if args.record_video:
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
        mp4_writer = __import__("imageio").get_writer(args.video_path, fps=30)
        renderer = mj.Renderer(model, height=args.video_height, width=args.video_width)

    print("显示中... 按 ESC 或关闭窗口退出。")
    start = time.time()

    while viewer.is_running():
        viewer.sync()
        if args.record_video and renderer and mp4_writer:
            renderer.update_scene(data, camera=viewer.cam)
            mp4_writer.append_data(renderer.render())
        if args.hold_seconds > 0 and (time.time() - start) >= args.hold_seconds:
            break
        time.sleep(0.03)

    viewer.close()
    time.sleep(0.3)
    if mp4_writer:
        mp4_writer.close()
        print(f"视频已保存: {args.video_path}")
    print("Done.")


if __name__ == "__main__":
    main()
