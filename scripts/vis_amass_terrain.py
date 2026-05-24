#!/usr/bin/env python3
"""
AMASS 动作 → 地形生成脚本。

基于 AMASS 动作数据中的支撑相信息，自动生成与脚掌贴合的地形方块。
每段连续支撑相生成一个方块，上表面与脚掌平面平行贴合。

用法:
    python scripts/vis_amass_terrain.py --amass_npz assets/amass_npz_test/01_01_stageii.npz
    python scripts/vis_amass_terrain.py --amass_npz <path> --block_length 0.25 --block_width 0.12
    python scripts/vis_amass_terrain.py --amass_npz <path> --save_terrain outputs/terrain.npz
    python scripts/vis_amass_terrain.py --amass_npz <path> --no_skeleton
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
JOINT_COLOR = np.array([0.2, 0.6, 0.9, 0.9])
BONE_COLOR = np.array([0.3, 0.3, 0.3, 0.7])
ROOT_COLOR = np.array([0.9, 0.3, 0.2, 0.95])
LEFT_LIMB_COLOR = np.array([0.2, 0.7, 0.3, 0.8])
RIGHT_LIMB_COLOR = np.array([0.8, 0.4, 0.1, 0.8])
HAND_COLOR = np.array([0.6, 0.5, 0.3, 0.6])

# 地形方块颜色
TERRAIN_LEFT_COLOR = np.array([0.3, 0.7, 0.4, 0.6])   # 左脚地形
TERRAIN_RIGHT_COLOR = np.array([0.8, 0.55, 0.3, 0.6])  # 右脚地形

LEFT_KEYWORDS = ["left"]
RIGHT_KEYWORDS = ["right"]
FINGER_KEYWORDS = ["index", "middle", "pinky", "ring", "thumb"]

# 最小 MuJoCo 场景 XML (不含 floor 避免视觉干扰地形块)
MINIMAL_SCENE_XML = """<mujoco>
  <worldbody>
    <light name="light1" pos="3 3 5" dir="-0.5 -0.5 -1"/>
    <light name="light2" pos="-3 -3 3" dir="0.3 0.3 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>
  </worldbody>
</mujoco>"""

# ---------------------------------------------------------------------------
# 复用 SMPL-X FK 函数 (与 vis_amass_skeleton 一致)
# ---------------------------------------------------------------------------

def get_bone_color(bone_name: str) -> np.ndarray:
    if any(kw in bone_name for kw in FINGER_KEYWORDS):
        return HAND_COLOR
    for kw in LEFT_KEYWORDS:
        if kw in bone_name:
            return LEFT_LIMB_COLOR
    for kw in RIGHT_KEYWORDS:
        if kw in bone_name:
            return RIGHT_LIMB_COLOR
    return BONE_COLOR


def smplx_fk_batch(smplx_data: dict, body_model,
                   start_frame: int = 0, num_frames: int = None) -> dict:
    """批量计算 SMPL-X 前向运动学。"""
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

    all_joints = smplx_output.joints.detach().numpy()
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    return {
        "all_joints": all_joints,
        "joint_names": joint_names,
        "parents": parents,
    }


def convert_to_mujoco(positions_smplx: np.ndarray) -> np.ndarray:
    """SMPL-X FK 输出已是 Z-up，恒等映射。"""
    return positions_smplx.copy()


def get_global_quats(smplx_data: dict, body_model,
                     start_frame: int = 0, num_frames: int = None) -> list:
    """计算每帧每个骨架关节的全局旋转 (四元数 xyzw)。"""
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
        betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl,
        left_hand_pose=torch.zeros(n_frames, 45).float(),
        right_hand_pose=torch.zeros(n_frames, 45).float(),
        jaw_pose=torch.zeros(n_frames, 3).float(),
        leye_pose=torch.zeros(n_frames, 3).float(),
        reye_pose=torch.zeros(n_frames, 3).float(),
        return_full_pose=True,
    )

    full_pose = smplx_output.full_pose.detach().numpy().reshape(n_frames, -1, 3)
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents
    rot_correction = R.from_matrix(np.eye(3))

    frames_quat = []
    for f in range(n_frames):
        joint_orientations = []
        quat_data = {}
        for i, name in enumerate(joint_names):
            rotvec = full_pose[f, i]
            if i == 0:
                rot = R.from_rotvec(rotvec)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(rotvec)
            joint_orientations.append(rot)
            rot_mj = rot_correction * rot * rot_correction.inv()
            quat_data[name] = rot_mj.as_quat()
        frames_quat.append(quat_data)

    return frames_quat


# ---------------------------------------------------------------------------
# 地形生成核心
# ---------------------------------------------------------------------------

def find_stance_segments(stance_per_frame: list, ankle_name: str) -> list:
    """
    寻找某只脚的连续支撑相片段。

    Returns:
        list of (start_frame, end_frame), 每段连续帧范围 [start, end] 闭区间
    """
    segments = []
    in_segment = False
    start = 0
    for f, stance in enumerate(stance_per_frame):
        is_stance = stance.get(ankle_name, False)
        if is_stance and not in_segment:
            start = f
            in_segment = True
        elif not is_stance and in_segment:
            segments.append((start, f - 1))
            in_segment = False
    if in_segment:
        segments.append((start, len(stance_per_frame) - 1))
    return segments


def compute_foot_plane_normal(ankle_quat, corners_rel):
    """从踝旋转和标定角点计算脚掌平面法向量 (世界系, 单位)。"""
    ankle_rot = R.from_quat(ankle_quat)
    corners_world = [ankle_rot.apply(np.array(c)) for c in corners_rel]
    edge1 = corners_world[1] - corners_world[0]
    edge2 = corners_world[3] - corners_world[0]
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal = normal / norm
    if normal[2] < 0:
        normal = -normal
    return normal


def compute_terrain_blocks(frames_global_pos, frames_global_quat,
                           foot_calib, stance_per_frame,
                           block_length=0.22, block_width=0.10, block_height=0.08):
    """
    根据支撑相片段生成地形方块。

    对每段连续支撑相，取该段内脚掌位置与朝向的均值，生成方块。

    Args:
        block_length: 方块长度 (沿脚掌前后方向, m)
        block_width:  方块宽度 (沿脚掌侧向, m)
        block_height: 方块厚度 (向下延伸, m)

    Returns:
        list of dict:
            pos: np.array(3)  方块中心世界坐标
            mat: np.array(9)  MuJoCo 3x3 旋转矩阵 (row-major)
            size: [hl, hw, hh]  半长宽高
            rgba: np.array(4)
            label: str  左脚/右脚 标签
            segment: (start, end)  帧范围
    """
    blocks = []

    for pair in foot_calib["foot_pairs"]:
        ankle_name = pair["ankle"]
        toe_name = pair["toe"]
        corners_rel = pair["corners_rel"]
        is_left = any(kw in ankle_name for kw in LEFT_KEYWORDS)

        segments = find_stance_segments(stance_per_frame, ankle_name)
        if not segments:
            continue

        for seg_start, seg_end in segments:
            # 收集该片段内所有帧的 toe 位置和脚掌朝向
            seg_toes = []
            seg_ankles = []
            seg_normals = []
            seg_forwards = []

            for f in range(seg_start, seg_end + 1):
                toe_pos = frames_global_pos[f].get(toe_name)
                ankle_pos = frames_global_pos[f].get(ankle_name)
                ankle_quat = frames_global_quat[f].get(ankle_name)
                if toe_pos is None or ankle_pos is None or ankle_quat is None:
                    continue

                # 脚掌法向量
                normal = compute_foot_plane_normal(ankle_quat, corners_rel)

                # 脚掌前向: 踝→趾在水平面的投影
                dir_xy = np.array([toe_pos[0] - ankle_pos[0],
                                    toe_pos[1] - ankle_pos[1]])
                fwd_len = np.linalg.norm(dir_xy)
                if fwd_len > 1e-6:
                    forward = np.array([dir_xy[0] / fwd_len,
                                         dir_xy[1] / fwd_len, 0.0])
                else:
                    forward = np.array([1.0, 0.0, 0.0])

                seg_toes.append(toe_pos)
                seg_ankles.append(ankle_pos)
                seg_normals.append(normal)
                seg_forwards.append(forward)

            if not seg_toes:
                continue

            # 均值
            avg_toe = np.mean(seg_toes, axis=0)
            avg_normal = np.mean(seg_normals, axis=0)
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-12)
            avg_forward = np.mean(seg_forwards, axis=0)
            fwd_norm = np.linalg.norm(avg_forward)
            if fwd_norm > 1e-6:
                avg_forward = avg_forward / fwd_norm
            else:
                avg_forward = np.array([1.0, 0.0, 0.0])
            # 确保 forward 垂直于 normal
            avg_forward = avg_forward - np.dot(avg_forward, avg_normal) * avg_normal
            avg_forward = avg_forward / (np.linalg.norm(avg_forward) + 1e-12)

            # 侧向 = normal × forward
            sideways = np.cross(avg_normal, avg_forward)
            sideways = sideways / (np.linalg.norm(sideways) + 1e-12)

            # 方块中心: toe 位置向下偏移半个方块高度
            center = avg_toe - avg_normal * (block_height / 2.0)

            # 旋转矩阵 (列向量: forward, sideways, normal)
            rot_mat = np.column_stack([avg_forward, sideways, avg_normal])

            color = TERRAIN_LEFT_COLOR if is_left else TERRAIN_RIGHT_COLOR
            blocks.append({
                "pos": center,
                "mat": rot_mat.T.flatten(),
                "size": [block_length / 2.0, block_width / 2.0, block_height / 2.0],
                "rgba": color.copy(),
                "label": "left" if is_left else "right",
                "segment": (seg_start, seg_end),
            })

    return blocks


def deduplicate_blocks(blocks: list, distance_threshold: float = 0.05) -> list:
    """
    去重：中心距小于阈值的方块合并为一组。
    每组保留中心 Z 最低的位置，法向量和朝向取组内均值。

    Args:
        blocks: 方块列表
        distance_threshold: 中心距离阈值 (m)

    Returns:
        去重后的方块列表
    """
    if len(blocks) <= 1:
        return blocks

    n = len(blocks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    positions = np.array([b["pos"] for b in blocks])
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < distance_threshold:
                union(i, j)

    # 按组收集
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    kept = []
    removed_count = 0
    for indices in groups.values():
        if len(indices) == 1:
            kept.append(blocks[indices[0]])
        else:
            # 取中心 Z 最低的位置
            best_idx = min(indices, key=lambda i: blocks[i]["pos"][2])
            best = blocks[best_idx].copy()

            # --- 组内法向量取均值 ---
            # mat 为 row-major 9 元素: [forward(3), sideways(3), normal(3)]
            normals = np.array([blocks[i]["mat"][6:9] for i in indices])
            forwards = np.array([blocks[i]["mat"][0:3] for i in indices])

            avg_normal = np.mean(normals, axis=0)
            nrm = np.linalg.norm(avg_normal)
            if nrm > 1e-12:
                avg_normal = avg_normal / nrm

            avg_forward = np.mean(forwards, axis=0)
            # 投影到垂直于 avg_normal 的平面
            avg_forward = avg_forward - np.dot(avg_forward, avg_normal) * avg_normal
            fwd_nrm = np.linalg.norm(avg_forward)
            if fwd_nrm > 1e-6:
                avg_forward = avg_forward / fwd_nrm
            else:
                # 退化情况：任选一个垂直于 normal 的方向
                if abs(avg_normal[0]) < 0.9:
                    avg_forward = np.cross(avg_normal, [1, 0, 0])
                else:
                    avg_forward = np.cross(avg_normal, [0, 1, 0])
                avg_forward = avg_forward / np.linalg.norm(avg_forward)

            sideways = np.cross(avg_normal, avg_forward)
            sideways = sideways / np.linalg.norm(sideways)

            # 重建旋转矩阵 (row-major: forward, sideways, normal)
            rot_mat = np.column_stack([avg_forward, sideways, avg_normal])
            best["mat"] = rot_mat.T.flatten()

            kept.append(best)
            removed_count += len(indices) - 1

    if removed_count > 0:
        print(f"  去重: 移除 {removed_count} 个重叠方块 "
              f"(阈值 {distance_threshold*100:.0f}cm, 法向量取均值)")
    return kept


# ---------------------------------------------------------------------------
# 骨架绘制 (与 vis_amass_skeleton 一致)
# ---------------------------------------------------------------------------

def draw_skeleton_frame(viewer, joint_names, parents,
                        global_positions, joint_radius=0.03):
    """绘制一帧骨架（不重置 ngeom，由调用方负责清除）。"""

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
            geom, type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0], pos=pos,
            mat=np.eye(3).flatten(), rgba=color,
        )
        viewer.user_scn.ngeom += 1


def draw_terrain_blocks(viewer, blocks: list):
    """
    将地形方块绘制到 MuJoCo user_scn 中。
    方块只绘制一次 (静态)，后续每帧骨架会覆盖 user_scn.ngeom=0 重新开始。
    因此需在每帧骨架绘制后调用此函数。
    """
    base_geom = viewer.user_scn.ngeom
    for blk in blocks:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_BOX,
            size=blk["size"],
            pos=blk["pos"],
            mat=blk["mat"],
            rgba=blk["rgba"],
        )
        viewer.user_scn.ngeom += 1
    return base_geom


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AMASS 动作 → 地形生成 (基于支撑相脚掌)"
    )
    parser.add_argument("--amass_npz", type=str, required=True,
                        help="AMASS / SMPL-X NPZ 动作文件路径")
    parser.add_argument("--smplx_model_path", type=str, default=DEFAULT_SMPLX_PATH,
                        help="SMPL-X 身体模型目录")
    parser.add_argument("--foot_calib", type=str,
                        default="outputs/amass_foot_calib.json",
                        help="脚掌标定 JSON 路径")
    parser.add_argument("--motion_fps", type=int, default=30,
                        help="播放帧率 (默认 30)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="起始帧")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="加载帧数")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="跳帧步长")
    parser.add_argument("--stance_threshold", type=float, default=0.03,
                        help="支撑相速度阈值 (m/s, 默认 0.03)")
    parser.add_argument("--block_length", type=float, default=0.22,
                        help="地形方块长 (前后方向, m)")
    parser.add_argument("--block_width", type=float, default=0.10,
                        help="地形方块宽 (侧向, m)")
    parser.add_argument("--block_height", type=float, default=0.08,
                        help="地形方块厚 (向下, m)")
    parser.add_argument("--dedup_threshold", type=float, default=0.05,
                        help="去重距离阈值 (m, 默认 0.05 = 5cm)。"
                             "中心距小于此值的方块只保留 Z 最低的")
    parser.add_argument("--joint_radius", type=float, default=0.03,
                        help="关节球半径")
    parser.add_argument("--no_skeleton", action="store_true",
                        help="不显示骨架动画 (仅地形)")
    parser.add_argument("--no_loop", action="store_true",
                        help="不循环播放")
    parser.add_argument("--record_video", action="store_true",
                        help="录制视频")
    parser.add_argument("--video_path", type=str,
                        default="videos/amass_terrain.mp4")
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--camera_distance", type=float, default=5.0,
                        help="相机距离")
    parser.add_argument("--camera_elevation", type=float, default=-25,
                        help="相机俯仰角")
    parser.add_argument("--camera_azimuth", type=float, default=90,
                        help="相机方位角")
    parser.add_argument("--save_terrain", type=str, default=None,
                        help="导出地形数据 (.npz)")

    args = parser.parse_args()

    # --- 加载数据 ---
    print(f"加载 AMASS 文件: {args.amass_npz}")
    smplx_data = np.load(args.amass_npz, allow_pickle=True)
    gender = str(smplx_data["gender"])
    total_frames = smplx_data["pose_body"].shape[0]
    src_fps = smplx_data["mocap_frame_rate"].item()
    print(f"  性别: {gender}, 总帧数: {total_frames}, 原始帧率: {src_fps:.1f}")

    body_model = smplx.create(
        args.smplx_model_path, "smplx", gender=gender, use_pca=False,
    )

    # --- FK ---
    print("计算前向运动学...")
    fk_result = smplx_fk_batch(
        smplx_data, body_model,
        start_frame=args.start_frame, num_frames=args.num_frames,
    )
    all_joints_mj = convert_to_mujoco(fk_result["all_joints"])
    joint_names = fk_result["joint_names"]
    parents = fk_result["parents"]
    num_loaded = all_joints_mj.shape[0]
    print(f"  骨架关节数: {len(joint_names)}, 加载帧数: {num_loaded}")

    frames_global_pos = []
    for f in range(num_loaded):
        pos_data = {joint_names[i]: all_joints_mj[f, i].copy()
                     for i in range(len(joint_names))}
        frames_global_pos.append(pos_data)

    # --- 全局旋转 ---
    print("计算全局旋转...")
    frames_global_quat = get_global_quats(
        smplx_data, body_model,
        start_frame=args.start_frame, num_frames=args.num_frames,
    )

    # --- 脚掌标定 ---
    print(f"加载脚掌标定: {args.foot_calib}")
    with open(args.foot_calib, "r") as f:
        foot_calib = json.load(f)

    # --- 支撑相检测 ---
    print(f"计算支撑相 (阈值: {args.stance_threshold*100:.1f} cm/s)...")
    frame_time = args.frame_skip / args.motion_fps
    stance_per_frame = []
    for f in range(num_loaded):
        frame_stance = {}
        for pair in foot_calib["foot_pairs"]:
            ankle_name = pair["ankle"]
            toe_name = pair["toe"]
            curr_pos = frames_global_pos[f].get(toe_name)
            if f == 0:
                frame_stance[ankle_name] = True
            else:
                prev_pos = frames_global_pos[f - 1].get(toe_name)
                if curr_pos is not None and prev_pos is not None:
                    vel = np.linalg.norm(curr_pos - prev_pos) / frame_time
                    frame_stance[ankle_name] = vel < args.stance_threshold
                else:
                    frame_stance[ankle_name] = False
        stance_per_frame.append(frame_stance)

    stance_count = sum(1 for fs in stance_per_frame for v in fs.values() if v)
    total_checks = len(stance_per_frame) * len(foot_calib["foot_pairs"])
    print(f"  支撑相占比: {stance_count}/{total_checks} "
          f"({100*stance_count/max(1,total_checks):.1f}%)")

    # --- 生成地形方块 ---
    print(f"生成地形方块 (长={args.block_length}m, 宽={args.block_width}m, "
          f"厚={args.block_height}m)...")
    terrain_blocks = compute_terrain_blocks(
        frames_global_pos, frames_global_quat,
        foot_calib, stance_per_frame,
        block_length=args.block_length,
        block_width=args.block_width,
        block_height=args.block_height,
    )
    n_left = sum(1 for b in terrain_blocks if b["label"] == "left")
    n_right = sum(1 for b in terrain_blocks if b["label"] == "right")
    print(f"  地形方块总数: {len(terrain_blocks)} (左脚: {n_left}, 右脚: {n_right})")

    # --- 去重 ---
    if args.dedup_threshold > 0:
        terrain_blocks = deduplicate_blocks(terrain_blocks, args.dedup_threshold)
        n_left = sum(1 for b in terrain_blocks if b["label"] == "left")
        n_right = sum(1 for b in terrain_blocks if b["label"] == "right")
        print(f"  去重后方块总数: {len(terrain_blocks)} "
              f"(左脚: {n_left}, 右脚: {n_right})")

    # --- 导出地形 ---
    if args.save_terrain:
        save_dir = os.path.dirname(args.save_terrain)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez(
            args.save_terrain,
            block_pos=np.array([b["pos"] for b in terrain_blocks]),
            block_mat=np.array([b["mat"] for b in terrain_blocks]),
            block_size=np.array([b["size"] for b in terrain_blocks]),
            block_label=np.array([b["label"] for b in terrain_blocks]),
            block_segment=np.array([b["segment"] for b in terrain_blocks]),
            block_length=args.block_length,
            block_width=args.block_width,
            block_height=args.block_height,
            stance_threshold=args.stance_threshold,
        )
        print(f"地形数据已保存: {args.save_terrain}")

    # --- MuJoCo 场景 ---
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

    # --- 视频 ---
    mp4_writer = None
    renderer = None
    if args.record_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        mp4_writer = __import__("imageio").get_writer(
            args.video_path, fps=args.motion_fps,
        )
        renderer = mj.Renderer(model, height=args.video_height,
                               width=args.video_width)
        print(f"录制视频到: {args.video_path}")

    # --- 主循环 ---
    print("\n播放中... 按 ESC 或关闭窗口退出。\n")
    frame_idx = 0
    while viewer.is_running():
        actual_idx = frame_idx * args.frame_skip
        if actual_idx >= num_loaded:
            actual_idx = actual_idx % num_loaded

        # 清除后绘制地形 + 骨架
        viewer.user_scn.ngeom = 0
        draw_terrain_blocks(viewer, terrain_blocks)

        if not args.no_skeleton:
            global_positions = frames_global_pos[actual_idx]
            draw_skeleton_frame(viewer, joint_names, parents,
                                global_positions, args.joint_radius)

        viewer.sync()
        rate_limiter.sleep()

        if args.record_video and renderer is not None and mp4_writer is not None:
            renderer.update_scene(data, camera=viewer.cam)
            mp4_writer.append_data(renderer.render())

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
