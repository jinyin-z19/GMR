#!/usr/bin/env python3
"""
AMASS / SMPL-X 人体骨架运动可视化脚本。

加载 AMASS NPZ 格式的 SMPL-X 动作数据，通过 SMPL-X 前向运动学
计算逐帧关节全局位置，在 MuJoCo viewer 中实时渲染人体骨架动画。

用法:
    python scripts/vis_amass_skeleton.py --amass_npz <path/to/motion.npz>
    python scripts/vis_amass_skeleton.py --amass_npz assets/amass_npz_test/01_01_stageii.npz
    python scripts/vis_amass_skeleton.py --amass_npz <path> --record_video --video_path output.mp4
    python scripts/vis_amass_skeleton.py --amass_npz <path> --foot_calib outputs/amass_foot_calib.json
    python scripts/vis_amass_skeleton.py --amass_npz <path> --motion_fps 30 --start_frame 100
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
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from smplx.joint_names import JOINT_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SMPLX_PATH = os.path.join(HERE, "..", "assets", "body_models")

# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------
JOINT_COLOR = np.array([0.2, 0.6, 0.9, 0.9])       # 关节球颜色 (蓝)
BONE_COLOR = np.array([0.3, 0.3, 0.3, 0.7])         # 骨骼连线颜色 (灰)
ROOT_COLOR = np.array([0.9, 0.3, 0.2, 0.95])        # 根关节颜色 (红)
LEFT_LIMB_COLOR = np.array([0.2, 0.7, 0.3, 0.8])    # 左侧肢体颜色 (绿)
RIGHT_LIMB_COLOR = np.array([0.8, 0.4, 0.1, 0.8])   # 右侧肢体颜色 (橙)
HAND_COLOR = np.array([0.6, 0.5, 0.3, 0.6])         # 手部关节颜色

# 左侧骨骼名称关键字
LEFT_KEYWORDS = ["left"]
# 右侧骨骼名称关键字
RIGHT_KEYWORDS = ["right"]
# 手指关键字
FINGER_KEYWORDS = ["index", "middle", "pinky", "ring", "thumb"]

# 最小 MuJoCo 场景 XML
MINIMAL_SCENE_XML = """<mujoco>
  <worldbody>
    <light name="light1" pos="3 3 5" dir="-0.5 -0.5 -1"/>
    <light name="light2" pos="-3 -3 3" dir="0.3 0.3 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
  </worldbody>
</mujoco>"""

# 脚掌平面默认参数
FOOT_PLANE_WIDTH = 0.08
FOOT_PLANE_ALPHA = 0.55


def get_bone_color(bone_name: str) -> np.ndarray:
    """根据骨骼名称返回对应颜色：左侧绿、右侧橙、手部棕、躯干灰"""
    if any(kw in bone_name for kw in FINGER_KEYWORDS):
        return HAND_COLOR
    for kw in LEFT_KEYWORDS:
        if kw in bone_name:
            return LEFT_LIMB_COLOR
    for kw in RIGHT_KEYWORDS:
        if kw in bone_name:
            return RIGHT_LIMB_COLOR
    return BONE_COLOR


def load_smplx_body_model(smplx_model_path: str, gender: str):
    """加载 SMPL-X 身体模型"""
    return smplx.create(smplx_model_path, "smplx", gender=gender, use_pca=False)


def smplx_fk_batch(smplx_data: dict, body_model,
                   start_frame: int = 0, num_frames: int = None) -> dict:
    """
    批量计算 SMPL-X 前向运动学。

    Returns:
        dict with:
            all_joints: np.array (N, 127, 3)  关节位置 (SMPL-X Y-up)
            joint_names: list of str
            parents: list of int
    """
    total_frames = smplx_data["pose_body"].shape[0]
    if num_frames is None:
        end_frame = total_frames
    else:
        end_frame = min(start_frame + num_frames, total_frames)

    frame_indices = slice(start_frame, end_frame)
    n_frames = end_frame - start_frame

    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    global_orient = torch.tensor(smplx_data["root_orient"][frame_indices]).float()
    body_pose = torch.tensor(smplx_data["pose_body"][frame_indices]).float()
    transl = torch.tensor(smplx_data["trans"][frame_indices]).float()

    smplx_output = body_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=torch.zeros(n_frames, 45).float(),
        right_hand_pose=torch.zeros(n_frames, 45).float(),
        jaw_pose=torch.zeros(n_frames, 3).float(),
        leye_pose=torch.zeros(n_frames, 3).float(),
        reye_pose=torch.zeros(n_frames, 3).float(),
        return_full_pose=True,
    )

    all_joints = smplx_output.joints.detach().numpy()  # (N, 127, 3)
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    return {
        "all_joints": all_joints,
        "joint_names": joint_names,
        "parents": parents,
    }


def convert_to_mujoco(positions_smplx: np.ndarray) -> np.ndarray:
    """
    SMPL-X FK 输出已是 Z-up 坐标系，与 MuJoCo 一致，无需旋转。

    SMPL-X FK 输出 (已含 global_orient): X=右, Y=后, Z=上
    MuJoCo: X=右, Y=前, Z=上

    恒等映射即可，人的朝向由 global_orient 决定。
    """
    return positions_smplx.copy()


def get_global_quats(smplx_data: dict, body_model,
                     start_frame: int = 0, num_frames: int = None) -> list:
    """
    计算每帧每个骨架关节的全局旋转 (四元数, xyzw, MuJoCo Z-up 坐标系)。

    通过 SMPL-X body model 运行 FK，使用 smplx_output.full_pose
    (shape: N,55,3) 逐关节计算全局旋转。

    Returns:
        list of dict: [{bone_name: np.array([x,y,z,w]), ...}, ...]
    """
    total_frames = smplx_data["pose_body"].shape[0]
    if num_frames is None:
        end_frame = total_frames
    else:
        end_frame = min(start_frame + num_frames, total_frames)
    frame_indices = slice(start_frame, end_frame)
    n_frames = end_frame - start_frame

    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    global_orient = torch.tensor(smplx_data["root_orient"][frame_indices]).float()
    body_pose = torch.tensor(smplx_data["pose_body"][frame_indices]).float()
    transl = torch.tensor(smplx_data["trans"][frame_indices]).float()

    smplx_output = body_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=torch.zeros(n_frames, 45).float(),
        right_hand_pose=torch.zeros(n_frames, 45).float(),
        jaw_pose=torch.zeros(n_frames, 3).float(),
        leye_pose=torch.zeros(n_frames, 3).float(),
        reye_pose=torch.zeros(n_frames, 3).float(),
        return_full_pose=True,
    )

    # full_pose: (N, 165) — 扁平化, 需 reshape 为 (N, 55, 3)
    full_pose = smplx_output.full_pose.detach().numpy().reshape(n_frames, -1, 3)
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    # SMPL-X FK 输出已是 Z-up，与 MuJoCo 一致，使用恒等旋转变换
    rot_correction = R.from_matrix(np.eye(3))

    frames_quat = []
    for f in range(n_frames):
        joint_orientations = []
        quat_data = {}
        for i, name in enumerate(joint_names):
            rotvec = full_pose[f, i]  # (3,) axis-angle
            if i == 0:
                rot = R.from_rotvec(rotvec)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(rotvec)
            joint_orientations.append(rot)
            # 变换到 MuJoCo 坐标系
            rot_mj = rot_correction * rot * rot_correction.inv()
            quat_data[name] = rot_mj.as_quat()  # xyzw
        frames_quat.append(quat_data)

    return frames_quat


def draw_foot_planes(viewer, global_positions, global_quats, foot_calib,
                      alpha=FOOT_PLANE_ALPHA):
    """
    根据标定数据渲染脚掌矩形面。
    """
    for pair in foot_calib["foot_pairs"]:
        ankle_name = pair["ankle"]
        corners_rel = [np.array(c) for c in pair["corners_rel"]]

        ankle_pos = global_positions.get(ankle_name)
        ankle_quat = global_quats.get(ankle_name)
        if ankle_pos is None or ankle_quat is None:
            continue

        ankle_rot = R.from_quat(ankle_quat)
        corners_world = [ankle_pos + ankle_rot.apply(c) for c in corners_rel]

        is_left = any(kw in ankle_name for kw in LEFT_KEYWORDS)
        if is_left:
            color = LEFT_LIMB_COLOR
        elif any(kw in ankle_name for kw in RIGHT_KEYWORDS):
            color = RIGHT_LIMB_COLOR
        else:
            color = BONE_COLOR
        rgba = np.array([color[0], color[1], color[2], alpha])

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


def draw_skeleton_frame(viewer, joint_names, parents,
                        global_positions, joint_radius=0.03):
    """
    在 MuJoCo viewer 中绘制一帧骨架。

    Args:
        viewer: mjv 句柄
        joint_names: 关节名称列表
        parents: 父关节索引数组
        global_positions: {joint_name: np.array([x,y,z])}
        joint_radius: 关节球半径
    """
    viewer.user_scn.ngeom = 0

    # 1. 骨骼连线
    for i, (bone, parent_idx) in enumerate(zip(joint_names, parents)):
        if parent_idx < 0:
            continue
        parent_bone = joint_names[parent_idx]
        child_pos = global_positions.get(bone)
        parent_pos = global_positions.get(parent_bone)
        if child_pos is None or parent_pos is None:
            continue

        color = get_bone_color(bone)
        is_finger = any(kw in bone for kw in FINGER_KEYWORDS)
        bone_width = 0.012 if is_finger else 0.025

        mj.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            width=bone_width,
            from_=parent_pos,
            to=child_pos,
        )
        for k in range(3):
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[k] = color[k]
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[3] = color[3]
        viewer.user_scn.ngeom += 1

    # 2. 关节球
    for bone in joint_names:
        pos = global_positions.get(bone)
        if pos is None:
            continue

        is_root = (bone == joint_names[0])
        is_finger = any(kw in bone for kw in FINGER_KEYWORDS)

        if is_root:
            color = ROOT_COLOR
            radius = joint_radius * 1.5
        elif is_finger:
            color = HAND_COLOR
            radius = joint_radius * 0.35
        else:
            color = get_bone_color(bone)
            radius = joint_radius

        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
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
        description="AMASS / SMPL-X 人体骨架运动可视化"
    )
    parser.add_argument(
        "--amass_npz", type=str, required=True,
        help="AMASS / SMPL-X NPZ 动作文件路径",
    )
    parser.add_argument(
        "--smplx_model_path", type=str, default=DEFAULT_SMPLX_PATH,
        help=f"SMPL-X 身体模型目录 (默认: {DEFAULT_SMPLX_PATH})",
    )
    parser.add_argument(
        "--motion_fps", type=int, default=30,
        help="播放帧率 (默认: 30)",
    )
    parser.add_argument(
        "--start_frame", type=int, default=0,
        help="起始帧 (默认: 0)",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="加载帧数 (默认全部)",
    )
    parser.add_argument(
        "--frame_skip", type=int, default=1,
        help="跳帧步长 (默认 1, 即不跳帧)",
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
        "--video_path", type=str, default="videos/amass_skeleton.mp4",
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
        help="脚掌标定 JSON 路径 (由 vis_amass_zero_pose.py --save_calib 生成)",
    )

    args = parser.parse_args()

    # --- 加载 NPZ 数据 ---
    print(f"加载 AMASS 文件: {args.amass_npz}")
    smplx_data = np.load(args.amass_npz, allow_pickle=True)
    gender = str(smplx_data["gender"])
    total_frames = smplx_data["pose_body"].shape[0]
    src_fps = smplx_data["mocap_frame_rate"].item()
    print(f"  性别: {gender}")
    print(f"  总帧数: {total_frames}")
    print(f"  原始帧率: {src_fps:.1f}")

    # --- 加载 SMPL-X 身体模型 ---
    print(f"加载 SMPL-X 身体模型: {args.smplx_model_path}")
    body_model = smplx.create(
        args.smplx_model_path, "smplx",
        gender=gender, use_pca=False,
    )

    # --- 批量计算 FK ---
    print("计算前向运动学...")
    fk_result = smplx_fk_batch(
        smplx_data, body_model,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
    )
    all_joints_smplx = fk_result["all_joints"]  # (N, 127, 3) Y-up
    joint_names = fk_result["joint_names"]
    parents = fk_result["parents"]
    num_loaded = all_joints_smplx.shape[0]
    print(f"  骨架关节数: {len(joint_names)}")
    print(f"  加载帧数: {num_loaded}")

    # --- 转换为 MuJoCo 坐标系 & 构建逐帧位置字典 ---
    print("坐标系: SMPL-X FK (Z-up) → MuJoCo (Z-up), 恒等映射")
    all_joints_mj = convert_to_mujoco(all_joints_smplx)  # (N, 127, 3)

    frames_global_pos = []
    for f in range(num_loaded):
        pos_data = {}
        for i, name in enumerate(joint_names):
            pos_data[name] = all_joints_mj[f, i].copy()
        frames_global_pos.append(pos_data)

    # --- 计算每帧旋转 (用于脚掌平面渲染) ---
    frames_global_quat = None
    if args.foot_calib:
        print("计算全局旋转...")
        frames_global_quat = get_global_quats(
            smplx_data, body_model,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
        )

    # --- 加载脚掌标定 ---
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

    viewer = mjv.launch_passive(
        model=model, data=data,
        show_left_ui=False, show_right_ui=False,
    )

    viewer.cam.distance = args.camera_distance
    viewer.cam.elevation = args.camera_elevation
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.lookat[:] = [0, 0, 0.9]

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
        renderer = mj.Renderer(model, height=args.video_height,
                               width=args.video_width)
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

        # 当前帧
        actual_idx = frame_idx * args.frame_skip
        if actual_idx >= num_loaded:
            actual_idx = actual_idx % num_loaded

        global_positions = frames_global_pos[actual_idx]
        global_quats = frames_global_quat[actual_idx] if frames_global_quat else None

        # 绘制骨架
        draw_skeleton_frame(viewer, joint_names, parents,
                            global_positions, args.joint_radius)

        # 绘制脚掌面
        if foot_calib and global_quats:
            draw_foot_planes(viewer, global_positions, global_quats, foot_calib)

        viewer.sync()
        rate_limiter.sleep()

        # 录制
        if args.record_video and renderer is not None and mp4_writer is not None:
            renderer.update_scene(data, camera=viewer.cam)
            img = renderer.render()
            mp4_writer.append_data(img)

        # 推进帧
        frame_idx += 1
        if frame_idx * args.frame_skip >= num_loaded:
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
