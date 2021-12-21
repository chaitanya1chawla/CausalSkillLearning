# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class OrigRobomimic_Dataset(Dataset):

	# LINK TO DATASET and INFO: https://arise-initiative.github.io/robomimic-web/docs/introduction/results.html#downloading-released-datasets

	# Class implementing instance of Robomimic dataset. 
	def __init__(self, args):		
		
		self.args = args
		if self.args.datadir is None:
			self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/RoboMimic/'
		else:
			self.dataset_directory = self.args.datadir			
		
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		
		self.task_list = ["can","lift","square","tool_hang"]
		# Excluding the transport task for now, because this has two Franka's. 
		# self.task_list = ["can","lift","square","tool_hang","transport"]

		# Each task has 200 demos according to RoboMimic.
		self.num_demos = 200*np.ones((5),dtype=int)
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
		
		# Append -1 to the start of cummulative_num_demos. This has two purposes. 
		# The first is that when we are at index 0 of the dataset, if we appended 0, np.searchsorted returns 0, rather than 1. 
		# For index 1, it returns 1. This was becoming inconsistent behavior for demonstrations in the same task. 
		# Now with -1 added to cumm_num_demos, when we are at task index 0, it would add -1 to the demo index. This is necessary for ALL tasks, not just the first...  
		# So that foils our really clever idea. 
		# Well, if the searchsorted returns the index of the equalling element, it probably consistently does this irrespective of vlaue. 
		# This means we can use this...

		# No need for a clever solution, searchsorted has a "side" option that takes care of this. 

		self.total_length = self.num_demos.sum()		

		# Seems to follow joint angles order:
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'Milk0', 'Bread0', 'Cereal0', 'Can0').
		# Extract these into... 

		self.joint_angle_indices = [1,3,4,5,6,7,8]
		self.gripper_indices = [9,10]	
		self.ds_freq = 20

		########################
		# TO CHANGE! 
		# TO CHANGE! 
		########################
		

		# self.r_gripper_r_finger_joint = np.array([-0.0116,   0.020833])
		# self.r_gripper_l_finger_joint = np.array([-0.020833, 0.0135])

		# [l,r]
		# gripper_open = [0.0115, -0.0115]
		# gripper_closed = [-0.020833, 0.020833]

		# Set files. 
		self.setup()

	def setup(self):
		# Load data from all tasks. 			
		self.files = []
		for i in range(len(self.task_list)):
			# Changing file name.
			# self.files.append(h5py.File("{0}/{1}/demo.hdf5".format(self.dataset_directory,self.task_list[i]),'r'))
			self.files.append(h5py.File("{0}/{1}/ph/low_dim.hdf5".format(self.dataset_directory,self.task_list[i]),'r'))

	def __len__(self):
		return self.total_length
	
	def __getitem__(self, index):


		# if index>=self.total_length:
		# 	print("Out of bounds of dataset.")
		# 	return None

		# # Get bucket that index falls into based on num_demos array. 
		# task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
		
		# if index==self.total_length-1:
		# 	task_index-=1

		# # Decide task ID, and new index modulo num_demos.
		# # Subtract number of demonstrations in cumsum until then, and then 				
		# new_index = index-self.cummulative_num_demos[max(task_index,0)]+1
		
		# try:
		# 	# Get raw state sequence. 
		# 	state_sequence = self.files[task_index]['data/demo_{0}/states'.format(new_index)].value

		# 	##############  
		# 	##############
		# 	# TO CHANGE
		# 	##############
		# 	# Use self.files[task_index]['data/demo_{0}/obs/robot0_joint_pos']
		# except:
		# 	# If this failed, return invalid. 
		# 	data_element = {}
		# 	data_element['is_valid'] = False

		# 	return data_element

		# # Performing another check that makes sure data element actually has states.
		# if state_sequence.shape[0]==0:
		# 	data_element = {}
		# 	data_element['is_valid'] = False
		# 	return data_element

		# # If we are here, the data element is presumably valid till now.
		# # Get joint angles from this state sequence.
		# joint_values = state_sequence[:,self.joint_angle_indices]
		# # Get gripper values from state sequence. 
		# gripper_finger_values = state_sequence[:,self.gripper_indices]

		# # Normalize gripper values. 

		# # 1 is right finger. 0 is left finger. 
		# # 1-0 is right-left. 
		
		# gripper_values = gripper_finger_values[:,1]-gripper_finger_values[:,0]
		# gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
		# gripper_values = 2*gripper_values-1

		# concatenated_demonstration = np.concatenate([joint_values,gripper_values.reshape((-1,1))],axis=1)
		# downsampled_demonstration = resample(concatenated_demonstration, concatenated_demonstration.shape[0]//self.ds_freq)

		# # Performing another check that makes sure data element actually has states.
		# if downsampled_demonstration.shape[0]==0:
		# 	data_element = {}
		# 	data_element['is_valid'] = False
		# 	return data_element

		# data_element = {}

		# if self.args.smoothen:
		# 	data_element['demo'] = gaussian_filter1d(downsampled_demonstration,self.args.smoothing_kernel_bandwidth,axis=0,mode='nearest')
		# else:
		# 	data_element['demo'] = downsampled_demonstration
		# # Trivially setting is valid to true until we come up wiuth a better strategy. 
		# data_element['is_valid'] = True

		# return data_element

		return {}
		
	def preprocess_dataset(self):

		for task_index in range(len(self.task_list)):

			print("#######################################")
			print("Preprocessing task index: ", task_index)
			print("#######################################")

			# # Get the name of environment.
			# import json
			# environment_meta_dict = json.loads(self.files[task_index]['data'].attrs['env_args'])
			# environment_name = environment_meta_dict['env_name']

			# # Create an actual robo-suite environment. 
			# self.env = robosuite.make(environment_name)

			# # Get sizes. 
			# obs = self.env._get_observation()
			# robot_state_size = obs['robot-state'].shape[0]
			# object_state_size = obs['object-state'].shape[0]	

			# Just get robot state size and object state size, from the demonstration of this task. 
			object_state_size = self.files[task_index]['data/demo_0/obs/object'].shape[1]
			robot_state_size = self.files[task_index]['data/demo_0/obs/robot0_joint_pos'].shape[1]

			# Create list of files for this task. 
			task_demo_list = []

			# For every element in the filelist of the element,
			# for i in range(1,self.num_demos[task_index]+1):
			for i in range(self.num_demos[task_index]):

				print("Preprocessing task index: ", task_index, " Demo Index: ", i, " of: ", self.num_demos[task_index])
			
				# Create list of datapoints for this demonstrations. 
				datapoint = {}
				
				# Get SEQUENCE of flattened states.
				flattened_state_sequence = np.array(self.files[task_index]['data/demo_{0}/states'.format(i)])
				robot_state_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_joint_pos'.format(i)])
				gripper_state_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_gripper_qpos'.format(i)])
				joint_action_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_joint_vel'.format(i)])
				gripper_action_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_gripper_qvel'.format(i)])
				object_state_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/object'.format(i)]) 

				# Downsample. 
				flattened_state_sequence = resample(flattened_state_sequence, flattened_state_sequence.shape[0]//self.ds_freq)
				robot_state_sequence = resample(robot_state_sequence, robot_state_sequence.shape[0]//self.ds_freq)
				gripper_state_sequence = resample(gripper_state_sequence, gripper_state_sequence.shape[0]//self.ds_freq)
				object_state_sequence = resample(object_state_sequence, object_state_sequence.shape[0]//self.ds_freq)
				joint_action_sequence = resample(joint_action_sequence, joint_action_sequence.shape[0]//self.ds_freq)
				gripper_action_sequence = resample(gripper_action_sequence, gripper_action_sequence.shape[0]//self.ds_freq)

				# Normalize gripper values. 
				# 1 is right finger. 0 is left finger.  # 1-0 is right-left. 						
				gripper_values = gripper_state_sequence[:,1]-gripper_state_sequence[:,0]
				gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
				gripper_values = 2*gripper_values-1

				concatenated_demonstration = np.concatenate([robot_state_sequence,gripper_values.reshape((-1,1))],axis=1)
				concatenated_actions = np.concatenate([joint_action_sequence,gripper_action_sequence.reshape((-1,1))],axis=1)

				# Put both lists in a dictionary.
				datapoint['flat-state'] = flattened_state_sequence
				datapoint['robot-state'] = robot_state_sequence
				datapoint['object-state'] = object_state_sequence
				datapoint['demo'] = concatenated_demonstration			
				datapoint['demonstrated_actions'] = concatenated_actions

				# Add this dictionary to the file_demo_list. 
				task_demo_list.append(datapoint)

			# Create array.
			task_demo_array = np.array(task_demo_list)

			# Now save this file_demo_list. 
			np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array.npy"),task_demo_array)
