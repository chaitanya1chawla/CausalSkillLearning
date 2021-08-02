
bs = 32
np.set_printoptions(precision=2)

array_errors = np.zeros(bs)
array_dataset_vs_recon_ee_pose_error = np.zeros(bs)
array_dataset_vs_IK_ee_pose_error = np.zeros(bs)

for k in range(bs):
    
    js = input_dictionary['joint_angle_traj'][:,k,:14]
    ee = input_dictionary['end_effector_traj'][:,k,:14]
    pjs = update_dictionary['predicted_joint_states'].view(js.shape[0],bs,-1)
    vpjs = pjs[:,k]

    errors = np.zeros(js.shape[0])
    dataset_vs_recon_ee_pose_error = np.zeros(js.shape[0])
    dataset_vs_IK_ee_pose_error = np.zeros(js.shape[0])

    for t in range(js.shape[0]):
        js1 = js[t]
        ee1 = ee[t]
        pjs1 = vpjs[t]

        ee_pose = ee1
        seed = pjs1 
        self.visualizer.baxter_IK_object.controller.sync_ik_robot(seed)

        peep = np.concatenate(self.visualizer.baxter_IK_object.controller.ik_robot_eef_joint_cartesian_pose())

        dataset_vs_recon_ee_pose_error[t] = (abs(peep-ee1)).mean()

        joint_positions = np.array(self.visualizer.baxter_IK_object.controller.inverse_kinematics(
                    target_position_right=ee_pose[:3],
                    target_orientation_right=ee_pose[3:7],
                    target_position_left=ee_pose[7:10],
                    target_orientation_left=ee_pose[10:14],
                    rest_poses=seed
                ))

        errors[t] = (abs(joint_positions-js1)).mean()

        self.visualizer.baxter_IK_object.controller.sync_ik_robot(joint_positions)
        peep2 = np.concatenate(self.visualizer.baxter_IK_object.controller.ik_robot_eef_joint_cartesian_pose())
        
        dataset_vs_IK_ee_pose_error[t] = (abs(peep2-ee1)).mean()

    # print("BI:",k,errors.max(),dataset_vs_recon_ee_pose_error.max(),dataset_vs_IK_ee_pose_error.max())
    array_errors[k] = errors.max()
    array_dataset_vs_recon_ee_pose_error[k] = dataset_vs_recon_ee_pose_error.max()
    array_dataset_vs_IK_ee_pose_error[k] = dataset_vs_IK_ee_pose_error.max()

    print("BI:", k, array_errors[k], array_dataset_vs_recon_ee_pose_error[k], array_dataset_vs_IK_ee_pose_error[k], js.shape[0])


    