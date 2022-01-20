# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import copy, os, imageio, scipy.misc, pdb, math, time, numpy as np

import matplotlib.pyplot as plt
from IPython import embed
from memory_profiler import profile
from PolicyNetworks import *
import torch

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

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False):

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

	def set_joint_pose_return_image(self, joint_pose, arm='both', gripper=False):

		# Just use the set pose function..
		self.set_joint_pose(joint_pose=joint_pose, arm=arm, gripper=gripper)

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False):

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
		self.link_indices = np.zeros(9,2,dtype=int)
		self.link_indices = np.array([[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8]])
		self.link_colors = ['k','k','k','b','r','b','r','b','r']
		
		# Now set pelvis pose.
		self.default_pelvis_pose = np.zeros((3))
				
	def set_joint_pose_return_image(self, joint_angles):

		# This code just plots skeleton. 			

		# First create figure object. 
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		# Add pelvis joint. 
		# Assumes joint_angles are dimensions N joints x 3 dimensions. 
		joints = np.insert(joint_angles, 0, self.default_pelvis_pose, axis=0)
		# Unnormalization w.r.t pelvis doesn't need to happen, because default pelvis pose 0. 
		
		# Now plot all joints, with left hand blue and right hand red to differentiate, and pelvis in black. 
		ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=self.colors, s=20, depthshade=False)
		
		# Now plot links. 
		for k, v in enumerate(self.link_indices):
			ax.plot([joints[v[0],0],joints[v[1],0]],[joints[v[0],1],joints[v[1],1]],[joints[v[0],2],joints[v[1],2]],c=self.link_colors[k])
				
		# Now get image from figure object to return .
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)		
		image = np.transpose(image, axes=[2,0,1])

		# Clear figure from memory.
		ax.clear()
		fig.clear()
		plt.close(fig)

		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None, end_effector=False):

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


# class GRABVisualizer(object):

# 	def __init__(self, has_display=False):

# 		# Import files from GRAB repo here? 
# 		# How do we import it in such a way that it is in scope for other functions of this class? 
# 		# Can we create instances of the modules as class variables?  Yes!
		
# 		# Strategy - 
# 		# 1) Add path to GRAB repository here. 
# 		# 2) Import GRAB repository tools. 
# 		# 3) Create local instances of modules.
# 		# 4) Create persistent variables.
# 		# 5) Create function that gets called by visualize_joint_trajectory function; put per-iteration code here.

# 		import sys
# 		sys.path.insert(0,'../../GRAB')

# 		import tools
# 		import smplx
# 		self.grab_tools = tools
# 		self.smplx = smplx

# 		self.setup()

# 	def setup(self):		

# 		self.model_path = '/home/tshankar/Research/Code/GRAB/smplx-models/models/'
# 		# Don't really need the grab_path, if we aren't going to load the object and table meshes.

# 		self.meshviewer = self.grab_tools.meshviewer.MeshViewer(width=600,height=600,offscreen=True)
# 		self.camera_pose = np.eye(4)
# 		self.camera_pose[:3, :3] = self.grab_tools.utils.euler([80, -15, 0], 'xzx')
# 		self.camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
# 		self.meshviewer.update_camera_pose(self.camera_pose)

# 	def visualize_sequence(self, sequence):
		
# 		# This function mimics the function vis_sequence https://github.com/tanmayshankar/GRAB/blob/master/examples/render_grab.py . 		
# 		# We probably need to add functionality to get mean person pose.. 

# 		seq_data = self.grab_tools.utils.parse_npz(sequence)

# 		T = seq_data.n_frames
		
# 		sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
# 		sbj_vtemp = np.array(self.grab_tools.meshviewer.Mesh(filename=sbj_mesh).vertices)

# 		sbj_m = self.smplx.create(model_path=self.model_path,
# 								model_type='smplx',
# 								gender='neutral',								
# 								# v_template=sbj_vtemp,
# 								batch_size=T)

# 		sbj_parms = self.grab_tools.utils.params2torch(seq_data.body.params)
# 		verts_sbj = self.grab_tools.utils.to_cpu(sbj_m(**sbj_parms).vertices)

# 		obj_mesh = os.path.join(self.grab_path, '..', seq_data.object.object_mesh)
# 		obj_mesh = self.grab_tools.meshviewer.Mesh(filename=obj_mesh)
# 		obj_vtemp = np.array(obj_mesh.vertices)
# 		obj_m = self.grab_tools.objectmodel.ObjectModel(v_template=obj_vtemp, batch_size=T)

# 		obj_parms = self.grab_tools.utils.params2torch(seq_data.object.params)
# 		verts_obj = self.grab_tools.utils.to_cpu(obj_m(**obj_parms).vertices)

# 		table_mesh = os.path.join(grab_path, '..', seq_data.table.table_mesh)
# 		table_mesh = self.grab_tools.meshviewer.Mesh(filename=table_mesh)
# 		table_vtemp = np.array(table_mesh.vertices)
# 		table_m = self.grab_tools.objectmodel.ObjectModel(v_template=table_vtemp, batch_size=T)

# 		table_parms = self.grab_tools.utils.params2torch(seq_data.table.params)
# 		verts_table = self.grab_tools.utils.to_cpu(table_m(**table_parms).vertices)

# 		seq_render_path = self.grab_tools.utils.makepath(sequence.replace('.npz','').replace(cfg.grab_path, cfg.render_path))

# 		skip_frame = 4
# 		for frame in range(0,T, skip_frame):
# 			o_mesh = self.grab_tools.meshviewer.Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
# 			o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

# 			s_mesh = self.grab_tools.meshviewer.Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
# 			s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

# 			s_mesh_wf = self.grab_tools.meshviewer.Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)
# 			t_mesh = self.grab_tools.meshviewer.Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

# 			self.meshviewer.set_static_meshes([o_mesh, s_mesh, s_mesh_wf, t_mesh])
# 			self.meshviewer.save_snapshot(seq_render_path+'/%04d.png'%frame)

# 	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, end_effector=False):

# 		image_list = []
# 		image_list.append(255*np.ones((600,600,3)))
# 		image_list.append(255*np.ones((600,600,3)))
# 		# imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

# 		return image_list		

# class SMPLXVisualizer(object):

# 	def __init__(self, has_display=False):

# 		# This class implements visualizing a SMPLX model, adapted from here https://github.com/vchoutas/smplx/blob/master/examples/demo.py#L85
# 		import smplx
# 		# Now import necessary matplotlib tools. 
# 		# from mpl_toolkits.mplot3d import Axes3D
# 		from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 		self.smplx = smplx
# 		# self.axes3d = Axes3D
# 		self.poly3dcollection = Poly3DCollection

# 		self.model_path = '/home/tshankar/Research/Code/GRAB/smplx-models/models/'

# 	def visualize_smplx_from_body_pose(self, body_pose):
		
# 		# This function takes a complete body pose and returns an image visualization of this body pose as a SMPLX model.

# 		self.model = self.smplx.create(self.model_path, 
# 								model_type = 'smplx',
# 								gender = 'neutral',
# 								body_pose = body_pose)

# 		output = self.model(betas=None, expression=None, return_verts=True)
# 		vertices = output.vertices.detach().cpu().numpy().squeeze()
# 		joints = output.joints.detach().cpu().numpy().squeeze()
		
# 		# Now plot these vertices / joints using matplotlib.
# 		fig = plt.figure()
# 		ax = fig.add_subplot(111, projection='3d')
		
# 		# Create model.
# 		mesh = self.poly3dcollection(vertices[model.faces], alpha=0.1)
# 		face_color = (1.0, 1.0, 0.9)
# 		edge_color = (0, 0, 0)
# 		mesh.set_edgecolor(edge_color)
# 		mesh.set_facecolor(face_color)
# 		ax.add_collection3d(mesh)
# 		ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')		
# 		# Actually plot joints.
# 		ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
		
# 		# Now get image from figure object to return .
# 		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)		
# 		image = np.transpose(image, axes=[2,0,1])

# 		# Clear figure from memory.
# 		ax.clear()
# 		fig.clear()
# 		plt.close(fig)

# 		return image

# 	def construct_body_pose(self, joints):
		
# 		# This function takes a joint pose at a single timestep, and fills it out into a full body pose using the mean poses for other joints. 		
# 		# It also handles things like unnormalizing the pelvis. 

# 		# Things this function needs to do - 
# 		# 1) Get mean body pose.  
# 		# 2) Use mean pelvis pose to unnormalize joint positions. 
# 		# 3) Use unnormalized joint positions to construct full body pose. 
# 		# 4) Return body pose for other functions to use, such as visualize_smplx_from_body_pose.

# 		# 1) First, get mean body pose. 
# 		pass




# class MocapVisualizer():

# 	def __init__(self, has_display=False, args=None):

# 		# Load some things from the MocapVisualizationUtils and set things up so that they're ready to go. 
# 		# self.cam_cur = MocapVisualizationUtils.camera.Camera(pos=np.array([6.0, 0.0, 2.0]),
# 		# 						origin=np.array([0.0, 0.0, 0.0]), 
# 		# 						vup=np.array([0.0, 0.0, 1.0]), 
# 		# 						fov=45.0)

# 		self.args = args

# 		# Default is local data. 
# 		self.global_data = False

# 		self.cam_cur = MocapVisualizationUtils.camera.Camera(pos=np.array([4.5, 0.0, 2.0]),
# 								origin=np.array([0.0, 0.0, 0.0]), 
# 								vup=np.array([0.0, 0.0, 1.0]), 
# 								fov=45.0)

# 		# Path to dummy file that is going to populate joint_parents, initial global positions, etc. 
# 		bvh_filename = "/private/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  

# 		# Run init before loading animation.
# 		MocapVisualizationUtils.init()
# 		MocapVisualizationUtils.global_positions, MocapVisualizationUtils.joint_parents, MocapVisualizationUtils.time_per_frame = MocapVisualizationUtils.load_animation(bvh_filename)

# 		# State sizes. 
# 		self.number_joints = 22
# 		self.number_dimensions = 3
# 		self.total_dimensions = self.number_joints*self.number_dimensions

# 		# Run thread of viewer, so that callbacks start running. 
# 		thread = threading.Thread(target=self.run_thread)
# 		thread.start()

# 		# Also create dummy animation object. 
# 		self.animation_object, _, _ = BVH.load(bvh_filename)

# 	def run_thread(self):
# 		MocapVisualizationUtils.viewer.run(
# 			title='BVH viewer',
# 			cam=self.cam_cur,
# 			size=(1280, 720),
# 			keyboard_callback=None,
# 			render_callback=MocapVisualizationUtils.render_callback_time_independent,
# 			idle_callback=MocapVisualizationUtils.idle_callback_return,
# 		) 

# 	def get_global_positions(self, positions, animation_object=None):
# 		# Function to get global positions corresponding to predicted or actual local positions.

# 		traj_len = positions.shape[0]

# 		def resample(original_trajectory, desired_number_timepoints):
# 			original_traj_len = len(original_trajectory)
# 			new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
# 			return original_trajectory[new_timepoints]

# 		if animation_object is not None:
# 			# Now copy over from animation_object instead of just dummy animation object.
# 			new_animation_object = Animation.Animation(resample(animation_object.rotations, traj_len), positions, animation_object.orients, animation_object.offsets, animation_object.parents)
# 		else:	
# 			# Create a dummy animation object. 
# 			new_animation_object = Animation.Animation(self.animation_object.rotations[:traj_len], positions, self.animation_object.orients, self.animation_object.offsets, self.animation_object.parents)

# 		# Then transform them.
# 		transformed_global_positions = Animation.positions_global(new_animation_object)

# 		# Now return coordinates. 
# 		return transformed_global_positions

# 	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None):

# 		image_list = []

# 		if self.global_data:
# 			# If we predicted in the global setting, just reshape.
# 			predicted_global_positions = np.reshape(trajectory, (-1,self.number_joints,self.number_dimensions)) 
# 		else:
# 			# If it's local data, then transform to global. 
# 			# Assume trajectory is number of timesteps x number_dimensions. 
# 			# Convert to number_of_timesteps x number_of_joints x 3.
# 			predicted_local_positions = np.reshape(trajectory, (-1,self.number_joints,self.number_dimensions))

# 			# Assume trajectory was predicted in local coordinates. Transform to global for visualization.
# 			predicted_global_positions = self.get_global_positions(predicted_local_positions, animation_object=additional_info)

# 		# Copy into the global variable.
# 		MocapVisualizationUtils.global_positions = predicted_global_positions

# 		# Reset Image List. 
# 		MocapVisualizationUtils.image_list = []
# 		# Set save_path and prefix.
# 		MocapVisualizationUtils.save_path = gif_path
# 		MocapVisualizationUtils.name_prefix = gif_name.rstrip('.gif')
# 		# Now set the whether_to_render as true. 
# 		MocapVisualizationUtils.whether_to_render = True

# 		# Wait till rendering is complete. 
# 		x_count = 0
# 		while MocapVisualizationUtils.done_with_render==False and MocapVisualizationUtils.whether_to_render==True:
# 			x_count += 1
# 			time.sleep(1)
			
# 		# Now that rendering is complete, load images.
# 		image_list = MocapVisualizationUtils.image_list

# 		# Now actually save the GIF or return.
# 		if return_and_save:
# 			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
# 			return image_list
# 		elif return_gif:
# 			return image_list
# 		else:
# 			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

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

