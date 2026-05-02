# Test Motions

We list some test motions here that are not easy for retargeting (i.e., sometimes giving bad results).
The [x] means the motion is retargeted successfully.


# AMASS

- [x] BMLrub_rub081_0031_rom_stageii.npz
- [x] KIT_3_walk_6m_straight_line04_stageii.npz
- [ ] DanceDB/20120807_CliodelaVara/Clio_Maleviziotikos_stageii.npz
- [ ] DanceDB/20130216_AnnaCharalambous/Anna_Curiosity_C3D_stageii.npz (Body jittering)
- [ ] CMU/111/111_21_stageii.npz (Should be laying on the ground facing right. The right arm is wristed on G1)

python scripts/smplx_to_robot.py --smplx_file datas/Walk_B10_smplx.npz --robot azureloong_v9 --save_path datas/retarget_g1.pkl --rate_limit --cam_distance 4.0 --follow_camera

# LAFAN1

- [x] dance1_subject2.bvh

python scripts/bvh_to_robot.py --bvh_file datas/dance1_subject1_lafan.bvh --robot azureloong_v9 --save_path datas/retarget_g1.pkl --rate_limit --format lafan1 --cam_distance 4.0



