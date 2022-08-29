# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cgitb import handler

from absl import flags, app
import copy, os, imageio, scipy.misc, pdb, math, time, numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import embed
from memory_profiler import profile
from PolicyNetworks import *
import torch
from moviepy.video.io.bindings import mplfig_to_npimage

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_printoptions(sci_mode=False, precision=2)

# # Mocap viz.
# import MocapVisualizationUtils
# from mocap_processing.motion.pfnn import Animation, BVH

class SawyerVisualizer(object):

	def __init__(self, has_display=False):
	
		# Create environment.
		print("Do I have a display?", has_display)
		
		import robosuite, threading
		# Create kinematics object. 
		if float(robosuite.__version__[:3])<1.:
			self.new_robosuite = 0
			self.base_env = robosuite.make("SawyerViz",has_renderer=has_display)
			from robosuite.wrappers import IKWrapper					
			self.sawyer_IK_object = IKWrapper(self.base_env)
			self.environment = self.sawyer_IK_object.env
		else:
			self.new_robosuite = 1
			self.base_env = robosuite.make("Viz",robots=['Sawyer'],has_renderer=has_display)
			self.sawyer_IK_object = None
			self.environment = self.base_env
		
	def update_state(self):
		# Updates all joint states
		self.full_state = self.environment._get_observation()

	def set_joint_pose_return_image(self, joint_angles, arm='both', gripper=False):

		# In the roboturk dataset, we've the following joint angles: 
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint')

		# Set usual joint angles through set joint positions API.
		self.environment.reset()
		if self.new_robosuite==0:
			self.environment.set_robot_joint_positions(joint_angles[:7])
		else:
			self.environment.robots[0].set_robot_joint_positions(joint_angles[:7])

		# For gripper, use "step". 
		# Mujoco requires actions that are -1 for Open and 1 for Close.

		# [l,r]
		# gripper_open = [0.0115, -0.0115]
		# gripper_closed = [-0.020833, 0.020833]
		# In mujoco, -1 is open, and 1 is closed.
		
		actions = np.zeros((8))
		actions[-1] = joint_angles[-1]

		# Move gripper positions.
		self.environment.step(actions)

		# Should set positions correctly.. Only really relevant for OBJECTS
		self.environment.sim.forward()

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False, task_id=None):

		image_list = []
		for t in range(trajectory.shape[0]):
			new_image = self.set_joint_pose_return_image(trajectory[t])
			image_list.append(new_image)

			# Insert white 
			if segmentations is not None:
				if t>0 and segmentations[t]==1:
					image_list.append(255*np.ones_like(new_image)+new_image)

		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)            

class FrankaVisualizer(SawyerVisualizer):

	def __init__(self, has_display=False):

		super(FrankaVisualizer, self).__init__(has_display=has_display)

		import robosuite, threading

		# Create kinematics object. 
		self.base_env = robosuite.make("Viz",robots=['Panda'],has_renderer=has_display)
		self.sawyer_IK_object = None
		self.environment = self.base_env

	def set_joint_pose_return_image(self, joint_angles, arm='both', gripper=False):

		# Set usual joint angles through set joint positions API.
		self.environment.reset()
		if self.new_robosuite==0:
			self.environment.set_robot_joint_positions(joint_angles[:7])
		else:
			self.environment.robots[0].set_robot_joint_positions(joint_angles[:7])
		actions = np.zeros((8))
		actions[-1] = joint_angles[-1]

		# Move gripper positions.
		self.environment.step(actions)

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview2'))
		return image

class BaxterVisualizer(object):

	# def __init__(self, has_display=False, args=None, IK_network_path="ExpWandbLogs/IK_010/saved_models/Model_epoch500"):
	# def __init__(self, has_display=False, args=None, IK_network_path="ExpWandbLogs/IK_050/saved_models/Model_epoch2000"):
	def __init__(self, has_display=False, args=None, IK_network_path=None):		


		# Create environment.
		print("Do I have a display?", has_display)
		
		import robosuite, threading
		# from robosuite.wrappers import IKWrapper		

		if float(robosuite.__version__[:3])<1.:
			self.new_robosuite = 0
			self.base_env = robosuite.make("BaxterViz",has_renderer=has_display)
			from robosuite.wrappers import IKWrapper					
			self.baxter_IK_object = IKWrapper(self.base_env)
			self.environment = self.baxter_IK_object.env

			if IK_network_path is not None:
				self.load_IK_network(IK_network_path)
			else:
				self.IK_network = None

		else:
			self.new_robosuite = 1
			self.base_env = robosuite.make("TwoArmViz",robots=['Baxter'],has_renderer=has_display)
			self.baxter_IK_object = None
			self.environment = self.base_env

		# # self.base_env = robosuite.make('BaxterLift', has_renderer=has_display)
		# self.base_env = robosuite.make("BaxterViz",has_renderer=has_display)

		# # Create kinematics object. 
		# self.baxter_IK_object = IKWrapper(self.base_env)
		# self.environment = self.baxter_IK_object.env  
		# self.args = args 

	def load_IK_network(self, path):
		
		# Now load the IK network! 
		self.IK_state_size = 14
		self.hidden_size = 48
		self.number_layers = 4
		self.IK_network = ContinuousMLP(self.IK_state_size, self.hidden_size, self.IK_state_size, args=self.args, number_layers=self.number_layers).to(device)

		load_object = torch.load(path)
		self.IK_network.load_state_dict(load_object['IK_Network'])

		print("Loaded IK Network from: ", path)
	
	def update_state(self):
		# Updates all joint states
		if self.new_robosuite:
			self.full_state = self.environment._get_observations()
		else:
			self.full_state = self.environment._get_observation()

	def set_ee_pose(self, ee_pose, arm='both', seed=None):

		# Assumes EE pose is Position in the first three elements, and quaternion in last 4 elements. 

		self.environment.reset()		
		self.update_state()

		#################################################
		# Normalize EE pose Quaternions
		#################################################

		if arm=='both':
			ee_pose[3:7] = ee_pose[3:7]/np.linalg.norm(ee_pose[3:7])
			ee_pose[10:14] = ee_pose[10:14]/np.linalg.norm(ee_pose[10:14])
		else:
			ee_pose[3:] = ee_pose[3:]/np.linalg.norm(ee_pose[3:])

		if seed is None:
			if self.IK_network is None:
				# Set seed to current state.
				seed = self.full_state['joint_pos']
			else:
				# Feed to IK network			
				# Nice thing about doing this inside the visualizer is that the trajectories will always be correctly unnormalized w.r.t mean / variance / min max. 
				# HEre, just normalize the L and R ee quaternions.. important when feeding in ee poses that are predicted, because otherwise domain shift. 			

				# Should do this before feeding to IK Network.
			
				# print("Embed in IK Viz")
				# embed()

				seed = self.IK_network.forward(torch.tensor(ee_pose[:14]).to(device).float()).detach().cpu().numpy()
				# ditch network and see what happens...
				# seed = self.full_state['joint_pos']
				# seed = np.zeros(14)
				# seed = np.random.random(14)
				# seed = np.ones(14)*0.5

				# Mean position
				# mean_position = np.array([ 0.43,  0.48, -1.87,  0.94, -2.01, -1.44,  1.54, -0.41,  0.41, 1.57,  1.29, -1.15,  1.08,  1.69])
				# mean_position = np.array([ 0.21,  0.2 , -1.26,  1.28, -0.96,  0.13,  0.  , -0.3 ,  0.06, 1.33,  1.29,  0.01,  0.26, -0.01])
				# seed = mean_position

			# The rest poses / seed only makes a difference when you make the IK_object's controller state get set to this seed....

			# Maybe try not syncing? 
			self.baxter_IK_object.controller.sync_ik_robot(seed, simulate=False, sync_last=True)

		if arm == 'right':
			joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=ee_pose[:3],
				target_orientation_right=ee_pose[3:],
				target_position_left=self.full_state['left_eef_pos'],
				target_orientation_left=self.full_state['left_eef_quat'],
				rest_poses=seed
			)

		elif arm == 'left':
			joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=self.full_state['right_eef_pos'],
				target_orientation_right=self.full_state['right_eef_quat'],
				target_position_left=ee_pose[:3],
				target_orientation_left=ee_pose[3:],
				rest_poses=seed
			)

		elif arm == 'both':
			joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=ee_pose[:3],
				target_orientation_right=ee_pose[3:7],
				target_position_left=ee_pose[7:10],
				target_orientation_left=ee_pose[10:14],
				rest_poses=seed
			)

		# self.set_joint_pose(joint_positions, arm=arm, gripper=False)

		return joint_positions

	def set_ee_pose_return_image(self, ee_pose, arm='both', seed=None):
		
		joint_positions = self.set_ee_pose(ee_pose, arm=arm, seed=seed)

		image = self.set_joint_pose_return_image(joint_positions, arm=arm, gripper=False)

		return image, joint_positions

	def set_joint_pose(self, joint_pose, arm='both', gripper=False):

		# FOR FULL 16 DOF STATE: ASSUMES JOINT_POSE IS <LEFT_JA, RIGHT_JA, LEFT_GRIPPER, RIGHT_GRIPPER>.

		self.update_state()

		if self.new_robosuite:
			# self.state = copy.deepcopy(self.full_state['robot0_joint_pos'])

			# Since the environment observation function doesn't return raw joint poses, we are going to index into the sim state instead.
			indices = [0,1,2,3,4,5,6,9,10,11,12,13,14,15]
			self.state = self.environment.sim.get_state()[1][indices]
		else:
			self.state = copy.deepcopy(self.full_state['joint_pos'])
		# THE FIRST 7 JOINT ANGLES IN MUJOCO ARE THE RIGHT HAND. 
		# THE LAST 7 JOINT ANGLES IN MUJOCO ARE THE LEFT HAND. 
		
		if arm=='right':
			# Assume joint_pose is 8 DoF - 7 for the arm, and 1 for the gripper.
			self.state[:7] = copy.deepcopy(joint_pose[:7])
		elif arm=='left':    
			# Assume joint_pose is 8 DoF - 7 for the arm, and 1 for the gripper.
			self.state[7:] = copy.deepcopy(joint_pose[:7])
		elif arm=='both':
			# The Plans were generated as: Left arm, Right arm, left gripper, right gripper.
			# Assume joint_pose is 16 DoF. 7 DoF for left arm, 7 DoF for right arm. (These need to be flipped)., 1 for left gripper. 1 for right gripper.            
			# First right hand. 
			self.state[:7] = joint_pose[7:14]
			# Now left hand. 
			self.state[7:] = joint_pose[:7]
			
		# Set the joint angles magically. 
		# self.environment.set_robot_joint_positions(self.state)

		if self.new_robosuite==0:
			self.environment.set_robot_joint_positions(self.state)
		else:
			self.environment.robots[0].set_robot_joint_positions(self.state)

		action = np.zeros((16))
		if gripper:
			# Left gripper is 15. Right gripper is 14. 
			# MIME Gripper values are from 0 to 100 (Close to Open), but we treat the inputs to this function as 0 to 1 (Close to Open), and then rescale to (-1 Open to 1 Close) for Mujoco.
			if arm=='right':
				action[14] = -joint_pose[-1]*2+1
			elif arm=='left':                        
				action[15] = -joint_pose[-1]*2+1
			elif arm=='both':
				action[14] = -joint_pose[15]*2+1
				action[15] = -joint_pose[14]*2+1
			
			# Move gripper positions.
			self.environment.step(action)

		# Apparently this needs to be called to set the positions correctly IN GENERAL
		self.environment.sim.forward()

		

	def set_joint_pose_return_image(self, joint_pose, arm='both', gripper=False):

		# Just use the set pose function..
		self.set_joint_pose(joint_pose=joint_pose, arm=arm, gripper=gripper)

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False, task_id=None):

		image_list = []
		previous_joint_positions = None

		for t in range(trajectory.shape[0]):

			# Check whether it's end effector or joint trajectory. 
			if end_effector: 
				new_image, previous_joint_positions = self.set_ee_pose_return_image(trajectory[t], seed=previous_joint_positions)
			else:
				new_image = self.set_joint_pose_return_image(trajectory[t])

			image_list.append(new_image)

			# Insert white 
			if segmentations is not None:
				if t>0 and segmentations[t]==1:
					image_list.append(255*np.ones_like(new_image)+new_image)

		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

class GRABVisualizer(object):

	def __init__(self, has_display=False):

		# THis class implements skeleton based visualization of the joints predicted by our model, rather than trying to visualize meshes. 

		# Remember, the relevant joints - 
		self.arm_joint_names = np.array(['pelvis',
					'left_collar',
					'right_collar',
					'left_shoulder',
					'right_shoulder',
					'left_elbow',
					'right_elbow',
					'left_wrist',
					'right_wrist'])
		
		# Skeleton - Pelvis --> Collar --> Shoulder --> Elbow --> Wrist (for each hand)
		# Add zeros as pelvis pose. 

		# Set colors of joints. 
		self.colors = ['k','b','r','b','r','b','r','b','r']
		
		# Set index pairs for links to be drawn. 
		# 9 links, for 2 x Pelvis --> Collar --> Shoulder --> Elbow --> Wrist
		# Also adding Collar <-> Collar links. 
		self.link_indices = np.zeros((9,2),dtype=int)
		self.link_indices = np.array([[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8]])
		self.link_colors = ['k','k','k','b','r','b','r','b','r']
		
		# Now set pelvis pose.
		self.default_pelvis_pose = np.zeros((3))
				
	def set_joint_pose_return_image(self, joint_angles, additional_info=None):

		# This code just plots skeleton. 			

		# First create figure object. 
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim(-0.5,0.5)
		ax.set_ylim(-0.5,0.5)
		ax.set_zlim(-0.5,0.5)
		
		# Add pelvis joint. 
		# Assumes joint_angles are dimensions N joints x 3 dimensions. 
		joints = copy.deepcopy(joint_angles)
		joints = joints.reshape((8,3))
		joints = np.insert(joints, 0, self.default_pelvis_pose, axis=0)
		# Unnormalization w.r.t pelvis doesn't need to happen, because default pelvis pose 0. 
		
		# Now plot all joints, with left hand blue and right hand red to differentiate, and pelvis in black. 

		# print("Embedding in set joint pose")
		# embed()
		ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=self.colors, s=20, depthshade=False)
		
		# Now plot links. 
		for k, v in enumerate(self.link_indices):
			ax.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]],c=self.link_colors[k])

		if additional_info is not None:
			ax.set_title(additional_info)
		# Now get image from figure object to return .
		# image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = mplfig_to_npimage(fig)
		# image = np.transpose(image, axes=[2,0,1])

		# Clear figure from memory.
		ax.clear()
		fig.clear()
		plt.close(fig)

		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False, task_id=None):

		image_list = []
		for t in range(trajectory.shape[0]):
			new_image = self.set_joint_pose_return_image(trajectory[t], additional_info=additional_info)
			image_list.append(new_image)

			# Insert white 
			if segmentations is not None:
				if t>0 and segmentations[t]==1:
					image_list.append(255*np.ones_like(new_image)+new_image)

		print(gif_path, gif_name)

		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

class GRABHandVisualizer(GRABVisualizer):
	
	def __init__(self, args, has_display=False):

		super(GRABHandVisualizer, self).__init__(has_display=has_display)

		self.side = args.single_hand
		# THis class implements skeleton based visualization of the joints predicted by our model, rather than trying to visualize meshes. 

		# Remember, the relevant joints - 
		self.hand_joint_names = np.array(['left_wrist', 
										  'left_index1',
										  'left_index2',
										  'left_index3',
										  'left_middle1',
										  'left_middle2',
										  'left_middle3',
										  'left_pinky1',
										  'left_pinky2',
										  'left_pinky3',
										  'left_ring1',
										  'left_ring2',
										  'left_ring3',
										  'left_thumb1',
										  'left_thumb2',
										  'left_thumb3',
										  'left_thumb',
										  'left_index', # 17
										  'left_middle',
										  'left_ring',
										  'left_pinky',
										  'right_wrist',  # index 21
										  'right_index1',
										  'right_index2',
										  'right_index3',
										  'right_middle1',
										  'right_middle2',
										  'right_middle3',
										  'right_pinky1',
										  'right_pinky2',
										  'right_pinky3',
										  'right_ring1',
										  'right_ring2',
										  'right_ring3',
										  'right_thumb1',
										  'right_thumb2',
										  'right_thumb3',
										  'right_thumb',
										  'right_index',
										  'right_middle',
										  'right_ring',
										  'right_pinky'])
		
		# Skeleton - Pelvis --> Collar --> Shoulder --> Elbow --> Wrist (for each hand)
		# Add zeros as pelvis pose. 

		# Set colors of joints. 
		self.colors = ['r' for i in range(21)].extend(['b' for j in range(21)])
		
		# Set index pairs for links to be drawn. 

		self.link_indices = np.zeros((46,2),dtype=int)
		self.link_indices = np.array([[0,1],[1,2],[2,3],[3,17], # left index finger
									  [0,4], [4,5], [5,6], [6,18], # left middle finger
									  [0,7], [7,8], [8,9], [9,20], # left pinky
									  [0,10], [10,11], [11,12], [12,19], # left ring finger
									  [0,13], [13,14], [14,15], [15,16], # left thumb
									  [1,4], [4,7], [7,10], # left hand outline
									  [21,22],[22,23],[23,24],[24,38], # right index finger
									  [21,25], [25,26], [26,27], [27,39], # right middle finger
									  [21,28], [28,29], [29,30], [30,41], # right pinky
									  [21,31], [31,32], [32,33], [33,40], # right ring finger
									  [21,34], [34,35], [35,36], [36,37], # right thumb
									  [22,25], [25,28], [28,31]])  # right hand outline
									  
		self.link_colors = ['k' for i in range(46)]
		self.link_colors[16:20] = 'r'
		self.link_colors[39:43] = 'b'

	
	def set_joint_pose_return_image(self, joint_angles, additional_info=None):

		# This code just plots skeleton. 			

		# First create figure object. 
		# One plot for each hand
		fig = plt.figure()

		if self.side not in ['left', 'right']:
			ax_left = fig.add_subplot(121, projection='3d')
			ax_left.set_xlim(-0.15,0.15)
			ax_left.set_ylim(-0.15,0.15)
			ax_left.set_zlim(-0.15,0.15)
			ax_right = fig.add_subplot(122, projection='3d')
			ax_right.set_xlim(-0.15,0.15)
			ax_right.set_ylim(-0.15,0.15)
			ax_right.set_zlim(-0.15,0.15)
		elif self.side == 'left':
			ax_left = fig.add_subplot(111, projection='3d')
			ax_left.set_xlim(-0.15,0.15)
			ax_left.set_ylim(-0.15,0.15)
			ax_left.set_zlim(-0.15,0.15)
		elif self.side == 'right':
			ax_right = fig.add_subplot(111, projection='3d')
			ax_right.set_xlim(-0.15,0.15)
			ax_right.set_ylim(-0.15,0.15)
			ax_right.set_zlim(-0.15,0.15)

		
		# Add pelvis joint. 
		# Assumes joint_angles are dimensions N joints x 3 dimensions. 
		joints = copy.deepcopy(joint_angles)
		if self.side in ['left', 'right']:
			joints = joints.reshape((21,3))
		else:
			joints = joints.reshape((42,3))
		# joints = np.insert(joints, 0, self.default_pelvis_pose, axis=0)
		# Unnormalization w.r.t pelvis doesn't need to happen, because default pelvis pose 0. 

		if self.side == 'left':
			leftjoints = joints[:21]
			leftjoints[0] = [0, 0, 0]
			ax_left.scatter(leftjoints[:, 0], leftjoints[:, 1], leftjoints[:, 2], color=self.colors, s=20, depthshade=False)
			for k, v in enumerate(self.link_indices[:23]):
				ax_left.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]],c=self.link_colors[k])


		elif self.side == 'right':
			rightjoints = joints[:21]
			rightjoints[0] = [0, 0, 0]
			ax_right.scatter(rightjoints[:, 0], rightjoints[:, 1], rightjoints[:, 2], color=self.colors, s=20, depthshade=False)
			for k, v in enumerate(self.link_indices[:23]):
				ax_right.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]],c=self.link_colors[k])

		else:
			leftjoints = joints[:21]
			leftjoints[0] = [0, 0, 0]
			rightjoints = joints[21:]
			rightjoints[0] = [0, 0, 0]
			# Now plot all joints, with left hand blue and right hand red to differentiate, and pelvis in black. 
			ax_left.scatter(leftjoints[:, 0], leftjoints[:, 1], leftjoints[:, 2], color=self.colors, s=20, depthshade=False)
			ax_right.scatter(rightjoints[:, 0], rightjoints[:, 1], rightjoints[:, 2], color=self.colors, s=20, depthshade=False)
			# Now plot links. 

			for k, v in enumerate(self.link_indices[:23]):
				ax_left.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]],c=self.link_colors[k])
			for k, v in enumerate(self.link_indices[23:]):
				ax_right.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]],c=self.link_colors[k])

			


		# print("Embedding in set joint pose")
		# embed()
		

		if additional_info is not None:
			if self.side != 'right':
				ax_left.set_title(additional_info)
			if self.side != 'left':
				ax_right.set_title(additional_info)

		# Now get image from figure object to return .
		# image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = mplfig_to_npimage(fig)
		# image = np.transpose(image, axes=[2,0,1])

		# Clear figure from memory.
		if self.side != 'right':
			ax_left.clear()
		if self.side != 'left':
			ax_right.clear()
		fig.clear()
		plt.close(fig)

		return image

class GRABArmHandVisualizer(GRABVisualizer):
	
	def __init__(self, args, has_display=False):

		# Inherit from super class.
		super(GRABArmHandVisualizer, self).__init__(has_display=has_display)

		# THis class implements skeleton based visualization of the joints predicted by our model, rather than trying to visualize meshes. 

		# Remember, the relevant joints - 
		self.arm_and_hand_joint_names = np.array([ #'pelvis', # not counted
												'left_shoulder', # index 0
												'left_elbow',
												'left_collar',
												'left_wrist', 
												'left_index1',
												'left_index2',
												'left_index3',
												'left_middle1',
												'left_middle2',
												'left_middle3',
												'left_pinky1',
												'left_pinky2',
												'left_pinky3',
												'left_ring1',
												'left_ring2',
												'left_ring3',
												'left_thumb1',
												'left_thumb2',
												'left_thumb3',
												'left_thumb',
												'left_index',
												'left_middle',
												'left_ring',
												'left_pinky',
												'right_shoulder', # 24
												'right_elbow',
												'right_collar',
												'right_wrist',
												'right_index1',
												'right_index2',
												'right_index3',
												'right_middle1',
												'right_middle2',
												'right_middle3',
												'right_pinky1',
												'right_pinky2',
												'right_pinky3',
												'right_ring1',
												'right_ring2',
												'right_ring3',
												'right_thumb1',
												'right_thumb2',
												'right_thumb3',
												'right_thumb',
												'right_index', # index 45
												'right_middle', 
												'right_ring',
												'right_pinky'])
		
		# Skeleton - Pelvis --> Collar --> Shoulder --> Elbow --> Wrist (for each hand)
		# Add zeros as pelvis pose. 

		# Set colors of joints. 
		self.colors = ['r' for i in range(21)].extend(['b' for j in range(21)])
		
		# Set index pairs for links to be drawn. 

		self.hand_link_indices = np.zeros((46,2),dtype=int)
		self.hand_link_indices = np.array([[0,1],[1,2],[2,3],[3,17], # left index finger
									  [0,4], [4,5], [5,6], [6,18], # left middle finger
									  [0,7], [7,8], [8,9], [9,20], # left pinky
									  [0,10], [10,11], [11,12], [12,19], # left ring finger
									  [0,13], [13,14], [14,15], [15,16], # left thumb
									  [1,4], [4,7], [7,10], # left hand outline
									  [21,22],[22,23],[23,24],[24,38], # right index finger
									  [21,25], [25,26], [26,27], [27,39], # right middle finger
									  [21,28], [28,29], [29,30], [30,41], # right pinky
									  [21,31], [31,32], [32,33], [33,40], # right ring finger
									  [21,34], [34,35], [35,36], [36,37], # right thumb
									  [22,25], [25,28], [28,31]])  # right hand outline

		# adjust indices
		for i in range(23):
			self.hand_link_indices[i][0] += 3
			self.hand_link_indices[i][1] += 3
		
		for i in range(23):
			self.hand_link_indices[23+i][0] += 6
			self.hand_link_indices[23+i][1] += 6

		self.arm_colors = ['k','b','r','b','r','b','r','b','r']
		
		# Set index pairs for links to be drawn. 
		# 9 links, for 2 x Pelvis --> Collar --> Shoulder --> Elbow --> Wrist
		# Also adding Collar <-> Collar links. 
		self.arm_link_indices = np.zeros((6,2),dtype=int)
		self.arm_link_indices = np.array([[2,0],[0,1],[1,3], # left collar -> shoulder -> elbow -> wrist
										[26,24],[24,25],[25,27]]) # right collar -> shoulder -> elbow -> wrist
		self.arm_link_colors = ['k','k','k','b','r','b','r','b','r']
		
		# Now set pelvis pose.
		self.default_pelvis_pose = np.zeros((3))
									  
		self.hand_link_colors = ['k' for i in range(46)]
		self.hand_link_colors[16:20] = 'r'
		self.hand_link_colors[39:43] = 'b'

	
	def set_joint_pose_return_image(self, joint_angles, additional_info=None):

		# This code just plots skeleton. 			

		# First create figure object. 
		# One plot for each hand
		fig = plt.figure()
		ax_left = fig.add_subplot(121, projection='3d')
		# ax_left.set_xlim(-0.5,0.5)
		# ax_left.set_ylim(-0.5,0.5)
		# ax_left.set_zlim(-0.5,0.5)
		ax_right = fig.add_subplot(122, projection='3d')
		# ax_right.set_xlim(-0.5,0.5)
		# ax_right.set_ylim(-0.5,0.5)
		# ax_right.set_zlim(-0.5,0.5)
		
		# Add pelvis joint. 
		# Assumes joint_angles are dimensions N joints x 3 dimensions. 
		joints = copy.deepcopy(joint_angles)
		joints = joints.reshape((48,3))
		# joints = np.insert(joints, 0, self.default_pelvis_pose, axis=0)
		# Unnormalization w.r.t pelvis doesn't need to happen, because default pelvis pose 0. 
		leftjoints = joints[:24]
		rightjoints = joints[24:]
		# joints[0] = self.default_pelvis_pose

		# Now plot all joints, with left hand blue and right hand red to differentiate, and pelvis in black. 

		# print("Embedding in set joint pose")
		# embed()
		ax_left.scatter(leftjoints[:, 0], leftjoints[:, 1], leftjoints[:, 2], color=self.colors, s=20, depthshade=False)
		ax_right.scatter(rightjoints[:, 0], rightjoints[:, 1], rightjoints[:, 2], color=self.colors, s=20, depthshade=False)

		# Now plot links. 
		for k, v in enumerate(self.hand_link_indices[:23]):
			ax_left.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]]) #,c=self.hand_link_indices[k])

		for k, v in enumerate(self.hand_link_indices[23:]):
			ax_right.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]]) #,c=self.hand_link_colors[k])

		for k, v in enumerate(self.arm_link_indices[:3]):
			ax_left.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]]) #,c=self.arm_link_colors[k])

		for k, v in enumerate(self.arm_link_indices[3:]):
			ax_right.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]]) #,c=self.arm_link_colors[k])


		if additional_info is not None:
			ax_left.set_title(additional_info)
			ax_right.set_title(additional_info)

		# Now get image from figure object to return .
		# image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = mplfig_to_npimage(fig)
		# image = np.transpose(image, axes=[2,0,1])

		# Clear figure from memory.
		ax_left.clear()
		ax_right.clear()
		fig.clear()
		plt.close(fig)

		return image

class RoboturkObjectVisualizer(object):

	def __init__(self, has_display=False, args=None):
		
		self.args = args
		self.has_display = has_display
		
		# Create environment.
		print("Do I have a display?", has_display)
		
		import robosuite, threading
		# Create kinematics object. 
		if float(robosuite.__version__[:3])<1.:
			self.new_robosuite = 0
			self.base_env = robosuite.make("SawyerViz",has_renderer=self.has_display)
			from robosuite.wrappers import IKWrapper					
			self.sawyer_IK_object = IKWrapper(self.base_env)
			self.environment = self.sawyer_IK_object.env
		else:
			self.new_robosuite = 1
			self.base_env = robosuite.make("Viz",robots=['Sawyer'],has_renderer=self.has_display)
			self.sawyer_IK_object = None
			self.environment = self.base_env
	
	def set_object_pose(self, position, orientation):

		# print("Entering object pose")
		# embed()
		
		# Sets object position for environment with one object. 
		# Indices of object position are 9-12. 
		self.environment.sim.data.qpos[9:12] = position
		# Orientation is indexed from 12-16., but is ordered differently. 
		# Orientation argument is ordered as x,y,z,w / This is what Mujoco observation gives us.
		# This qpos argument is ordered as w,x,y,z. 
		self.environment.sim.data.qpos[13:16] = orientation[:-1]
		self.environment.sim.data.qpos[12] = orientation[-1]

		# Sets posiitons correctly. Quaternions slightly off - trend is sstill correct.
		self.environment.sim.forward()

		# print("Exiting object pose")

	def set_joint_pose(self, pose, arm='both', gripper=False):
		
		# Is wrapper for set object pose.		
		object_position = pose[:3]
		object_orientation = pose[3:7]
		# object_to_eef_position = pose[7:10]
		# object_to_eef_quaternion = pose[10:]

		self.set_object_pose(object_position, object_orientation)

	def set_joint_pose_return_image(self, pose, arm='both', gripper=False):

		self.set_joint_pose(pose)

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def create_environment(self, task_id=None):

		print("Creating environment for task: ",task_id)

		import robosuite, threading
		if float(robosuite.__version__[:3])<1.:
			self.new_robosuite = 0
			self.base_env = robosuite.make(task_id,has_renderer=self.has_display)
			from robosuite.wrappers import IKWrapper					
			self.sawyer_IK_object = IKWrapper(self.base_env)
			self.environment = self.sawyer_IK_object.env
		else:
			self.new_robosuite = 1

			task_id_wo_robot_name = task_id.lstrip("Sawyer")
			self.base_env = robosuite.make(task_id_wo_robot_name,robots=['Sawyer'],has_renderer=self.has_display)
			self.sawyer_IK_object = None
			self.environment = self.base_env		

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False, task_id=None):

		image_list = []
		previous_joint_positions = None

		# Recreate environment with new task ID potentially.
		self.create_environment(task_id=task_id)

		for t in range(trajectory.shape[0]):

			# Check whether it's end effector or joint trajectory. 
			# Calls joint pose function, but is really setting the object position
			new_image = self.set_joint_pose_return_image(trajectory[t])

			image_list.append(new_image)

			# Insert white 
			if segmentations is not None:
				if t>0 and segmentations[t]==1:
					image_list.append(255*np.ones_like(new_image)+new_image)

		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)


class RoboturkRobotObjectVisualizer(RoboturkObjectVisualizer):

	def __init__(self, has_display=False, args=None):

		super(GRABArmHandVisualizer, self).__init__(has_display=has_display, args=args)

	def set_joint_pose(self, pose, arm='both', gripper=False):

		# Set object pose.
		# Assume last seven elements of pose are the actual pose.
		object_position = pose[-7:-4]
		object_orientation = pose[-4:]

		self.set_object_pose(object_position, object_orientation)
		
		# Set robot pose.
		# Assumes the  first seven elements are the robot pose.
		if self.new_robosuite==0:
			self.environment.set_robot_joint_positions(pose[:7])
		else:
			self.environment.robots[0].set_robot_joint_positions(pose[:7])
	

class ToyDataVisualizer():

	def __init__(self):

		pass

	# CREATING DUPLICATE
	# @profile
	# def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False):

	# 	fig = plt.figure()		
	# 	ax = fig.gca()
	# 	ax.scatter(trajectory[:,0],trajectory[:,1],c=range(len(trajectory)),cmap='jet')
	# 	plt.xlim(-10,10)
	# 	plt.ylim(-10,10)

	# 	fig.canvas.draw()

	# 	width, height = fig.get_size_inches() * fig.get_dpi()
	# 	image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
	# 	image = np.transpose(image, axes=[2,0,1])

	# 	ax.clear()
	# 	fig.clear()
	# 	plt.close(fig)

	# 	return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, end_effector=False):

		image_list = []
		image_list.append(255*np.ones((600,600,3)))
		image_list.append(255*np.ones((600,600,3)))
		# imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

		return image_list


if __name__ == '__main__':
	# end_eff_pose = [0.3, -0.3, 0.09798524029948213, 0.38044099037703677, 0.9228975092885654, -0.021717379118030174, 0.05525572942370394]
	# end_eff_pose = [0.53303758, -0.59997265,  0.09359371,  0.77337391,  0.34998901, 0.46797516, -0.24576358]
	# end_eff_pose = np.array([0.64, -0.83, 0.09798524029948213, 0.38044099037703677, 0.9228975092885654, -0.021717379118030174, 0.05525572942370394])
	visualizer = MujocoVisualizer()
	# img = visualizer.set_ee_pose_return_image(end_eff_pose, arm='right')
	# scipy.misc.imsave('mj_vis.png', img)


