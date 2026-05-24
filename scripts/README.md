# BVH 脚掌标定与可视化

## 1. 生成 LAFAN T-pose（脚掌着地修正）

```bash
# 让脚踝Z转-66.245度
# 输出: ik_config_manager/TPOSE_footfix.bvh
```

## 2. Xsens 格式（零位即站立）

```bash
# 标定
python scripts/vis_bvh_zero_pose.py \
    --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
    --save_calib assets/xsens_bvh_test/foot_calib.json

# 动画
python scripts/vis_bvh_skeleton.py \
    --bvh_file assets/xsens_bvh_test/251021_04_boxing_120Hz_cm_3DsMax.bvh \
    --motion_fps 60 \
    --foot_calib assets/xsens_bvh_test/foot_calib.json
```

## 3. LAFAN1 格式（需 T-pose 辅助标定）

```bash
# 标定
python scripts/vis_bvh_zero_pose.py \
    --bvh_file assets/lafan_bvh_test/dance1_subject1_lafan.bvh \
    --tpose_bvh ik_config_manager/TPOSE_footfix.bvh \
    --save_calib assets/lafan_bvh_test/foot_calib.json

# 动画
python scripts/vis_bvh_skeleton.py \
    --bvh_file assets/lafan_bvh_test/dance1_subject1_lafan.bvh \
    --motion_fps 60 \
    --foot_calib assets/lafan_bvh_test/foot_calib.json
```

## 4. 录制视频

```bash
# 标定 + 录制
python scripts/vis_bvh_zero_pose.py \
    --bvh_file ... --tpose_bvh ... --save_calib ... \
    --record_video --video_path videos/calib.mp4

# 动画 + 录制
python scripts/vis_bvh_skeleton.py \
    --bvh_file ... --motion_fps 60 --foot_calib ... \
    --record_video --video_path videos/output.mp4
```

## 5. 脚本说明


| 脚本                     | 用途                                          |
| ------------------------ | --------------------------------------------- |
| `create_lafan_tpose.py`  | 基于 Xsens 零位修正 LAFAN T-pose 的踝旋转偏置 |
| `vis_bvh_zero_pose.py`   | BVH 零位/T-pose 静态可视化 + 脚掌标定导出     |
| `vis_bvh_skeleton.py`    | BVH 骨架运动动画 + 脚掌动态渲染               |
| `vis_amass_zero_pose.py` | AMASS/SMPL-X T-pose 静态可视化 + 脚掌标定导出 |
| `vis_amass_skeleton.py`  | AMASS/SMPL-X 骨架运动动画 + 脚掌动态渲染      |

## 6. AMASS / SMPL-X 格式

AMASS 数据集使用 SMPL-X 身体模型，NPZ 文件包含 `root_orient`、`pose_body`、`trans`、`betas` 等参数。
T-pose 数据位于 `ik_config_manager/SMPLX_TPOSE_UNIFIED_AMASS.npz`。

### 6.1 标定（零位可视化）

```bash
# 默认使用内置 T-pose，显示并导出脚掌标定
# python scripts/vis_amass_zero_pose.py \
    #--save_calib outputs/amass_foot_calib.json

# 使用自定义 T-pose 文件
python scripts/vis_amass_zero_pose.py \
    --tpose_npz ik_config_manager/SMPLX_TPOSE_UNIFIED_AMASS.npz \
    --save_calib outputs/amass_foot_calib.json

# 录制标定视频
python scripts/vis_amass_zero_pose.py \
    --save_calib outputs/amass_foot_calib.json \
    --record_video --video_path videos/amass_tpose.mp4
```

### 6.2 动画（骨架运动演示）

```bash
# 基础播放
python scripts/vis_amass_skeleton.py \
    --amass_npz assets/amass_npz_test/01_01_stageii.npz

# 带脚掌标定
python scripts/vis_amass_skeleton.py \
    --amass_npz assets/amass_npz_test/01_01_stageii.npz \
    --foot_calib outputs/amass_foot_calib.json

# 控制播放参数
python scripts/vis_amass_skeleton.py \
    --amass_npz assets/amass_npz_test/01_01_stageii.npz \
    --motion_fps 30 \
    --start_frame 100 \
    --num_frames 300 \
    --no_loop \
    --foot_calib outputs/amass_foot_calib.json

# 录制视频
python scripts/vis_amass_skeleton.py \
    --amass_npz assets/amass_npz_test/01_01_stageii.npz \
    --motion_fps 30 \
    --foot_calib outputs/amass_foot_calib.json \
    --record_video --video_path videos/amass_walk.mp4
```

### 6.3 完整工作流

```bash
# 第一步：生成脚掌标定
python scripts/vis_amass_zero_pose.py \
    --save_calib outputs/amass_foot_calib.json

# 第二步：播放动画（带脚掌）
python scripts/vis_amass_skeleton.py \
    --amass_npz assets/amass_npz_test/01_01_stageii.npz \
    --foot_calib outputs/amass_foot_calib.json
```

### 6.4 参数说明


| 参数                 | 适用脚本  | 说明                        |
| -------------------- | --------- | --------------------------- |
| `--tpose_npz`        | zero_pose | T-pose NPZ 路径（默认内置） |
| `--amass_npz`        | skeleton  | AMASS 动作 NPZ 路径         |
| `--smplx_model_path` | 两者      | SMPL-X 身体模型目录         |
| `--save_calib`       | zero_pose | 导出脚掌标定 JSON           |
| `--foot_calib`       | skeleton  | 加载脚掌标定 JSON           |
| `--motion_fps`       | skeleton  | 播放帧率（默认 30）         |
| `--start_frame`      | skeleton  | 起始帧号                    |
| `--num_frames`       | skeleton  | 加载帧数                    |
| `--frame_skip`       | skeleton  | 跳帧步长（默认 1）          |
| `--no_loop`          | skeleton  | 不循环播放                  |
| `--record_video`     | 两者      | 录制视频                    |
| `--video_path`       | 两者      | 视频输出路径                |
| `--hold_seconds`     | zero_pose | 自动退出秒数                |
| `--joint_radius`     | 两者      | 关节球半径                  |
| `--camera_distance`  | 两者      | 相机距离                    |
| `--camera_elevation` | 两者      | 相机俯仰角                  |
| `--camera_azimuth`   | 两者      | 相机方位角                  |
