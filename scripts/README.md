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


| 脚本                    | 用途                                          |
| ----------------------- | --------------------------------------------- |
| `create_lafan_tpose.py` | 基于 Xsens 零位修正 LAFAN T-pose 的踝旋转偏置 |
| `vis_bvh_zero_pose.py`  | BVH 零位/T-pose 静态可视化 + 脚掌标定导出     |
| `vis_bvh_skeleton.py`   | BVH 骨架运动动画 + 脚掌动态渲染               |
