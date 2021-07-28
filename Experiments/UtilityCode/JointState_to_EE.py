# Map joint states to end effector code

from Visualizers import BaxterVisualizer
import numpy as np, robosuite, copy


x = np.load("MIMEDataArray.npy",allow_pickle=True)
np.set_printoptions(suppress=True,precision=2)
# Create visualizer object that we are going to use to interact with the environment.. 
visualizer = BaxterVisualizer()

# # For every trajectory in dataset
for k, v in enumerate(x):
	
#     # For every timestep in joint trjaecotry, just set the robot state to th joint state position, and then parse into EE pose
#     # Store the EE pose in the same dictionary..

	if k%500==0:
		print(k)
	ee_traj = None

	for t in range(v['demo'].shape[0]):

		# Get joint state. 
		joint_state = v['demo'][t]

		# # Set joint state. 
		# visualizer.set_joint_pose(joint_state)
		# visualizer.update_state()

		# Get EE position.
		# obs = copy.deepcopy(visualizer.full_state)
		# Assembly.
		# COncatenating in order: right ee pos, right ee q, left ee pos, left ee q.
		# ee_state = np.concatenate([obs['right_eef_pos'], obs['right_eef_quat'], obs['left_eef_pos'], obs['left_eef_quat'], joint_state[-2:]])

		# Instead of using env... 
		# Use...
		visualizer.baxter_IK_object.controller.sync_ik_robot(joint_state[:-2])
		ee_state = np.concatenate(visualizer.baxter_IK_object.controller.ik_robot_eef_joint_cartesian_pose())

		# # Can verify this EE state is okay by running it through IK. 
		ee_pose = ee_state
		# seed = np.random.random(14)

		# Seed at mean
		seed = np.array([ 0.42632668,  0.48221445, -1.87011259,  0.94430763, -2.01179089,
       		-1.43602596,  1.53576452, -0.41248854,  0.40729346,  1.56634601,
        	1.29183579, -1.15044519,  1.07504698,  1.68546272])


		ik_joint_state = np.array(visualizer.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=ee_pose[:3],
				target_orientation_right=ee_pose[3:7],
				target_position_left=ee_pose[7:10],
				target_orientation_left=ee_pose[10:],
				rest_poses=seed))
					
		norm = np.linalg.norm(joint_state[:-2] - ik_joint_state)
		if norm > 0.01:
			# print("Embedding in IK")
			print(k, t, norm)
			# embed()
		
		# Assembly.	
		# COncatenating in order: right ee pos, right ee q, left ee pos, left ee q, gripper.. 
		ee_state = np.concatenate([ee_state, joint_state[-2:]])
		
		if ee_traj is None:
			ee_traj = ee_state.reshape(1,-1)
		else: 
			ee_traj = np.concatenate([ee_traj, ee_state.reshape(1,-1)],axis=0)

	v['endeffector_trajectory'] = ee_traj

np.save("MIMEDataArray_EEAugmented.npy",x)

# js = x[0]['demo'][20]

# visualizer.update_state()
# orig_obs = copy.deepcopy(visualizer.full_state)

# visualizer.set_joint_pose(js)
# visualizer.update_state()
# next_obs = copy.deepcopy(visualizer.full_state)
