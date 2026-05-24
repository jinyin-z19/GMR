#!/usr/bin/env python3
"""
AMASS / SMPL-X 骨架零位（T-Pose）静态可视化脚本。

加载 AMASS NPZ 格式的 SMPL-X T-Pose 数据，通过 SMPL-X 前向运动学
计算各关节的全局位置，在 MuJoCo viewer 中静态渲染人体骨架。

用法:
    python scripts/vis_amass_zero_pose.py
    python scripts/vis_amass_zero_pose.py --tpose_npz <path/to/tpose.npz>
    python scripts/vis_amass_zero_pose.py --save_calib outputs/amass_foot_calib.json
    python scripts/vis_amass_zero_pose.py --record_video --video_path videos/amass_tpose.mp4
"""

import argparse
import json
import os
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R

from smplx.joint_names import JOINT_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SMPLX_PATH = os.path.join(HERE, "..", "assets", "body_models")
DEFAULT_TPOSE_PATH = os.path.join(HERE, "..", "ik_config_manager", "SMPLX_TPOSE_UNIFIED_AMASS.npz")

# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------
BONE_COLOR = np.array([0.3, 0.3, 0.3, 0.7])        # 骨骼连线
JOINT_COLOR = np.array([0.2, 0.6, 0.9, 0.9])        # 关节球
ROOT_COLOR = np.array([0.9, 0.3, 0.2, 0.95])        # 根关节
LEFT_COLOR = np.array([0.2, 0.7, 0.3, 0.8])         # 左侧肢体
RIGHT_COLOR = np.array([0.8, 0.4, 0.1, 0.8])        # 右侧肢体
HAND_COLOR = np.array([0.6, 0.5, 0.3, 0.6])         # 手部关节

LEFT_KEYWORDS = ["left"]
RIGHT_KEYWORDS = ["right"]

# 脚掌矩形参数
FOOT_PLANE_WIDTH = 0.08
FOOT_PLANE_ALPHA = 0.55
HEEL_EXTENSION = 0.04

# 手指关节名称关键字
FINGER_KEYWORDS = ["index", "middle", "pinky", "ring", "thumb"]

# 最小 MuJoCo 场景 XML
MINIMAL_SCENE_XML = """<mujoco>
  <worldbody>
    <light name="light1" pos="3 3 5" dir="-0.5 -0.5 -1"/>
    <light name="light2" pos="-3 -3 3" dir="0.3 0.3 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
  </worldbody>
</mujoco>"""


def get_bone_color(bone_name: str) -> np.ndarray:
    """根据骨骼名称返回对应颜色"""
    if any(kw in bone_name for kw in FINGER_KEYWORDS):
        return HAND_COLOR
    for kw in LEFT_KEYWORDS:
        if kw in bone_name:
            return LEFT_COLOR
    for kw in RIGHT_KEYWORDS:
        if kw in bone_name:
            return RIGHT_COLOR
    return BONE_COLOR


def smplx_fk(smplx_data: dict, body_model, num_frames: int = None) -> dict:
    """
    使用 SMPL-X body model 计算前向运动学，返回关节全局位置。

    Returns:
        dict with keys:
            joints: np.array (N, 127, 3)  全部 127 个关节位置 (SMPL-X Y-up)
            joint_names: list of str       关节名称 (55 个骨架关节)
            parents: list of int           父关节索引
    """
    if num_frames is None:
        num_frames = smplx_data["pose_body"].shape[0]

    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    global_orient = torch.tensor(smplx_data["root_orient"][:num_frames]).float()
    body_pose = torch.tensor(smplx_data["pose_body"][:num_frames]).float()
    transl = torch.tensor(smplx_data["trans"][:num_frames]).float()

    smplx_output = body_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )

    joints = smplx_output.joints.detach().numpy()  # (N, 127, 3)
    joint_names = JOINT_NAMES[: len(body_model.parents)]  # 55 skeleton joints
    parents = body_model.parents

    return {
        "joints": joints,
        "joint_names": joint_names,
        "parents": parents,
        "smplx_output": smplx_output,
    }


def convert_to_mujoco(positions_smplx: np.ndarray) -> np.ndarray:
    """
    将 SMPL-X 坐标系 (Y-up) 转换为 MuJoCo 坐标系 (Z-up)。

    SMPL-X: X=右, Y=上, Z=前
    MuJoCo: X=右, Y=前, Z=上

    使用右手旋转矩阵: [[1,0,0],[0,0,-1],[0,1,0]]
    X_mj = X_s, Y_mj = -Z_s, Z_mj = Y_s
    """
    rot_conv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    return positions_smplx @ rot_conv.T


def align_to_ground(positions: dict, joint_names: list) -> dict:
    """
    将骨架整体平移，使最低处脚关节刚好触地（Z=0）。
    """
    foot_keywords = ["ankle", "foot"]
    foot_z_vals = []
    for name in joint_names:
        if any(kw in name for kw in foot_keywords):
            foot_z_vals.append(positions[name][2])
    if not foot_z_vals:
        return positions

    min_z = min(foot_z_vals)
    offset = np.array([0.0, 0.0, -min_z])
    aligned = {b: pos + offset for b, pos in positions.items()}
    return aligned


def detect_foot_pairs(joint_names: list) -> list:
    """
    自动检测左右脚的 (踝关节, 足关节) 骨骼对。

    Returns:
        list of (ankle_name, foot_name, side_color)
    """
    pairs = []
    for side_kw, color in [("left", LEFT_COLOR), ("right", RIGHT_COLOR)]:
        ankles = [b for b in joint_names
                  if side_kw in b and "ankle" in b]
        feet = [b for b in joint_names
                if side_kw in b and "foot" in b and "ankle" not in b]
        if ankles and feet:
            pairs.append((ankles[0], feet[0], color))
    return pairs


def draw_foot_planes(viewer, positions: dict, joint_names: list):
    """
    用薄长方体绘制脚掌矩形面。
    """
    foot_pairs = detect_foot_pairs(joint_names)
    if not foot_pairs:
        return

    for ankle_name, foot_name, color in foot_pairs:
        ankle = positions.get(ankle_name)
        foot = positions.get(foot_name)
        if ankle is None or foot is None:
            continue

        plane_z = foot[2]
        dir_xy = np.array([foot[0] - ankle[0], foot[1] - ankle[1]])
        foot_len = np.linalg.norm(dir_xy)
        if foot_len < 0.01:
            foot_len = 0.12

        center = np.array([
            (ankle[0] + foot[0]) / 2.0,
            (ankle[1] + foot[1]) / 2.0,
            plane_z,
        ])

        if foot_len > 1e-6:
            forward = np.array([dir_xy[0] / foot_len, dir_xy[1] / foot_len, 0.0])
        else:
            forward = np.array([1.0, 0.0, 0.0])
        sideways = np.array([-forward[1], forward[0], 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        rot_matrix = np.column_stack([forward, sideways, normal])

        half_extents = [foot_len / 2.0, FOOT_PLANE_WIDTH / 2.0, 0.005]
        rgba = np.array([color[0], color[1], color[2], FOOT_PLANE_ALPHA])

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_BOX,
            size=half_extents,
            pos=center,
            mat=rot_matrix.T.flatten(),
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def export_foot_calib(positions: dict, joint_names: list,
                      parents: np.ndarray, calib_path: str):
    """
    从 T-Pose 姿态导出脚掌标定数据（JSON 格式）。
    """
    foot_pairs = detect_foot_pairs(joint_names)
    calib = {
        "foot_pairs": [],
        "plane_width": FOOT_PLANE_WIDTH,
        "plane_alpha": FOOT_PLANE_ALPHA,
        "heel_extension": HEEL_EXTENSION,
    }

    bone_idx = {b: i for i, b in enumerate(joint_names)}

    for ankle_name, foot_name, _color in foot_pairs:
        ankle = positions[ankle_name]
        foot = positions[foot_name]

        dir_xy = np.array([foot[0] - ankle[0], foot[1] - ankle[1]])
        foot_len_xy = np.linalg.norm(dir_xy)
        if foot_len_xy < 0.01:
            foot_len_xy = 0.12
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = np.array([dir_xy[0] / foot_len_xy,
                                dir_xy[1] / foot_len_xy, 0.0])

        sideways = np.array([-forward[1], forward[0], 0.0])

        heel_xy = np.array([ankle[0] - forward[0] * HEEL_EXTENSION,
                            ankle[1] - forward[1] * HEEL_EXTENSION])
        total_len = foot_len_xy + HEEL_EXTENSION

        center = np.array([(heel_xy[0] + foot[0]) / 2.0,
                           (heel_xy[1] + foot[1]) / 2.0,
                           foot[2]])

        hl = total_len / 2.0
        hw = FOOT_PLANE_WIDTH / 2.0

        corners_world = [
            center + forward * hl + sideways * hw,
            center + forward * hl - sideways * hw,
            center - forward * hl - sideways * hw,
            center - forward * hl + sideways * hw,
        ]

        corners_rel = [(c - ankle).tolist() for c in corners_world]

        ankle_idx = bone_idx[ankle_name]
        chain = _get_bone_chain(ankle_idx, parents)

        calib["foot_pairs"].append({
            "ankle": ankle_name,
            "toe": foot_name,
            "ankle_idx": ankle_idx,
            "chain_indices": chain,
            "corners_rel": corners_rel,
        })

    os.makedirs(os.path.dirname(calib_path) or ".", exist_ok=True)
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"脚掌标定已保存: {calib_path}")
    return calib


def _get_bone_chain(idx: int, parents: np.ndarray) -> list:
    """从骨骼索引出发，向上追溯到根。"""
    chain = []
    while idx >= 0:
        chain.append(int(idx))
        idx = parents[idx]
    chain.reverse()
    return chain


def draw_skeleton(viewer, joint_names, parents, positions,
                  joint_radius=0.03):
    """绘制一帧静态骨架"""
    # 骨骼连线
    for i, (bone, p) in enumerate(zip(joint_names, parents)):
        if p < 0:
            continue
        color = get_bone_color(bone)
        mj.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            width=0.025,
            from_=positions[joint_names[p]],
            to=positions[bone],
        )
        for k in range(4):
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[k] = color[k]
        viewer.user_scn.ngeom += 1

    # 关节球
    for bone in joint_names:
        pos = positions[bone]
        is_root = (bone == joint_names[0])
        is_finger = any(kw in bone for kw in FINGER_KEYWORDS)

        if is_root:
            color = ROOT_COLOR
            radius = joint_radius * 1.5
        elif is_finger:
            color = HAND_COLOR
            radius = joint_radius * 0.4
        else:
            color = get_bone_color(bone)
            radius = joint_radius

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom, type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0], pos=pos,
            mat=np.eye(3).flatten(), rgba=color,
        )
        viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser(
        description="AMASS / SMPL-X 骨架零位（T-Pose）静态可视化"
    )
    parser.add_argument(
        "--tpose_npz", type=str, default=DEFAULT_TPOSE_PATH,
        help=f"SMPL-X T-Pose NPZ 文件路径 (默认: {DEFAULT_TPOSE_PATH})",
    )
    parser.add_argument(
        "--smplx_model_path", type=str, default=DEFAULT_SMPLX_PATH,
        help=f"SMPL-X 身体模型目录 (默认: {DEFAULT_SMPLX_PATH})",
    )
    parser.add_argument(
        "--joint_radius", type=float, default=0.03,
        help="关节球半径 (m)",
    )
    parser.add_argument(
        "--camera_distance", type=float, default=3.5,
        help="相机距离 (m)",
    )
    parser.add_argument(
        "--camera_elevation", type=float, default=-10,
        help="相机俯仰角 (度)",
    )
    parser.add_argument(
        "--camera_azimuth", type=float, default=90,
        help="相机方位角 (度)",
    )
    parser.add_argument(
        "--record_video", action="store_true",
        help="录制视频",
    )
    parser.add_argument(
        "--video_path", type=str, default="videos/amass_zero_pose.mp4",
        help="视频输出路径",
    )
    parser.add_argument(
        "--video_width", type=int, default=1280,
    )
    parser.add_argument(
        "--video_height", type=int, default=720,
    )
    parser.add_argument(
        "--hold_seconds", type=float, default=0,
        help="自动退出秒数 (0=手动关闭)",
    )
    parser.add_argument(
        "--save_calib", type=str, default=None,
        help="导出脚掌标定 JSON 路径 (如 outputs/amass_foot_calib.json)",
    )

    args = parser.parse_args()

    # --- 加载 SMPL-X 身体模型 ---
    print(f"加载 SMPL-X 身体模型: {args.smplx_model_path}")
    tpose_npz = np.load(args.tpose_npz, allow_pickle=True)
    gender = str(tpose_npz["gender"])
    print(f"  性别: {gender}")

    body_model = smplx.create(
        args.smplx_model_path, "smplx",
        gender=gender, use_pca=False,
    )

    # --- 计算 T-Pose 关节位置 ---
    print(f"计算 T-Pose 前向运动学 (来源: {args.tpose_npz})")
    fk_result = smplx_fk(tpose_npz, body_model, num_frames=1)
    joints_smplx = fk_result["joints"][0]  # (127, 3) in Y-up
    joint_names = fk_result["joint_names"]
    parents = fk_result["parents"]

    # 转为 MuJoCo 坐标系 (Z-up)
    joints_mj = convert_to_mujoco(joints_smplx)
    positions = {joint_names[i]: joints_mj[i] for i in range(len(joint_names))}

    print(f"  骨架关节数: {len(joint_names)}")
    print(f"  全部顶点数: {joints_smplx.shape[0]}")

    # --- 贴地 ---
    positions = align_to_ground(positions, joint_names)

    # --- 导出脚掌标定 ---
    if args.save_calib:
        export_foot_calib(positions, joint_names, parents, args.save_calib)

    # --- MuJoCo 场景 ---
    model = mj.MjModel.from_xml_string(MINIMAL_SCENE_XML)
    data = mj.MjData(model)
    viewer = mjv.launch_passive(model=model, data=data,
                                show_left_ui=False, show_right_ui=False)
    viewer.cam.distance = args.camera_distance
    viewer.cam.elevation = args.camera_elevation
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.lookat[:] = [0, 0, 0.9]

    # --- 绘制 ---
    viewer.user_scn.ngeom = 0
    draw_skeleton(viewer, joint_names, parents, positions, args.joint_radius)
    draw_foot_planes(viewer, positions, joint_names)
    viewer.sync()

    # --- 视频 ---
    mp4_writer = None
    renderer = None
    if args.record_video:
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
        mp4_writer = __import__("imageio").get_writer(args.video_path, fps=30)
        renderer = mj.Renderer(model, height=args.video_height,
                               width=args.video_width)

    print("\n显示中... 按 ESC 或关闭窗口退出。")
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
