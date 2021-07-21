# Map joint states to end effector code

from Visualizers import BaxterVisualizer
import numpy as np, robosuite, copy

x = np.load("MIMEDataArray.npy",allow_pickle=True)

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

		# Set joint state. 
		visualizer.set_joint_pose(joint_state)
		visualizer.update_state()
		
		# Get EE position.
		obs = copy.deepcopy(visualizer.full_state)

		# Assembly.
		# COncatenating in order: right ee pos, right ee q, left ee pos, left ee q.
		ee_state = np.concatenate([obs['right_eef_pos'], obs['right_eef_quat'], obs['left_eef_pos'], obs['left_eef_quat'], joint_state[-2:]])
		
		if ee_traj is None:
			ee_traj = ee_state.reshape(1,-1)
		else: 
			ee_traj = np.concatenate([ee_traj, ee_state.reshape(1,-1)],axis=0)

	v['endeffector_trajectory'] = ee_traj

# js = x[0]['demo'][20]

# visualizer.update_state()
# orig_obs = copy.deepcopy(visualizer.full_state)

# visualizer.set_joint_pose(js)
# visualizer.update_state()
# next_obs = copy.deepcopy(visualizer.full_state)
