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

class OrigMOMART_Dataset(Dataset): 

	# Class implementing instance of Robomimic dataset. 
	def __init__(self, args):		
		
		self.args = args

		if self.args.datadir is None:
			self.dataset_directory = '/data/tanmayshankar/Datasets/MOMART'
		else:
			self.dataset_directory = self.args.datadir

		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 		

		self.task_list = ['table_cleanup_to_dishwasher', "table_cleanup_to_sink", "table_setup_to_dishwasher", "table_setup_from_dresser"]
		# self.task_list = ["can","lift","square","tool_hang"]
		# self.environment_names = ['PickPlaceCan','Lift','NutAssemblySquare', 'ToolHang']
						
		self.num_demos = np.array([111, 110, 111, 111])
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)	
		self.total_length = self.num_demos.sum()		

		# Seems to follow joint angles order:
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'Milk0', 'Bread0', 'Cereal0', 'Can0').
		# Extract these into... 

		self.ds_freq = np.array([ 2.4, 2.4, 2.4, 3.5])
		# self.ds_freq = 20

		# Set files. 
		self.setup()

		self.stat_dir_name='MOMART'

	def setup(self):
		
		# Load data from all tasks. 			
		self.files = []
		self.file_list = sorted(glob.glob(os.path.join(self.dataset_directory, "*/*/*.hdf5")))

		for k, file in enumerate(self.file_list):
			# Changing file name.
			self.files.append(h5py.File(file,'r'))
			# self.files.append(h5py.File("{0}/{1}/ph/low_dim.hdf5".format(self.dataset_directory,	self.task_list[i]),'r'))

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
	
	def create_env(self, task_index):
		import robosuite
		
		self.env = robosuite.make(self.environment_names[task_index], robots=['Panda'], has_renderer=False)
		self.env.reset()

	def compute_relative_object_state_quat(self, robot_state_sequence, object_state_sequence):

		############################################
		# Now add our own computed relative quaterion if the ENV is Lift.
		############################################
		import robosuite.utils.transform_utils as RTU					

		object_quat_traj = np.zeros((object_state_sequence.shape[0],4))

		print("Generating relative state quat.")
		for k in range(robot_state_sequence.shape[0]):				
			# Set pose.
			# env.reset()
			self.env.robots[0].set_robot_joint_positions(robot_state_sequence[k,:7])
			self.env.sim.forward()
			obs = self.env._get_observations()
			
			# Get eef pose. 
			robot_eef_quat = obs['robot0_eef_quat']
			
			# Get obj pose. 
			abs_object_quat = object_state_sequence[k,3:7]
			
			# Get relative quat. 					
			object_quat_traj[k] = RTU.quat_multiply(robot_eef_quat, abs_object_quat)

		return object_quat_traj

	def compute_relative_object_state(self, robot_state_sequence, object_state_sequence):

		############################################
		# Now add our own computed relative quaterion if the ENV is Lift.
		############################################
		import robosuite.utils.transform_utils as RTU					

		object_rel_traj = np.zeros((object_state_sequence.shape[0],7))

		print("Generating relative state quat.")
		for k in range(robot_state_sequence.shape[0]):				
			# Set pose.
			# env.reset()
			self.env.robots[0].set_robot_joint_positions(robot_state_sequence[k,:7])
			self.env.sim.forward()
			obs = self.env._get_observations()
			
			# Get eef pose. 
			robot_eef_quat = obs['robot0_eef_quat']
			robot_eef_pos = obs['robot0_eef_pos']
			robot_eef_pose_mat = RTU.pose2mat((robot_eef_pos, robot_eef_quat))
						
			# Get obj pose. 
			abs_object_quat = object_state_sequence[k,3:7]
			abs_object_pos = object_state_sequence[k,:3]
			abs_object_pose_mat = RTU.pose2mat((abs_object_pos, abs_object_quat))

			# Get relative state.
			robot_pose_inv = RTU.pose_inv(robot_eef_pose_mat)
			relative_object_pose_mat = RTU.pose_in_A_to_pose_in_B(abs_object_pose_mat, robot_pose_inv)
			relative_object_pos, relative_object_quat = RTU.mat2pose(relative_object_pose_mat)
				
			# Get relative quat. 					
			object_rel_traj[k,:3] = relative_object_pos
			object_rel_traj[k,3:] = relative_object_quat

		return object_rel_traj

	def preprocess_dataset(self):

		min_lengths = np.ones((4))*1000
		max_lengths = np.zeros((4))

		for task_index in range(len(self.task_list)):
		# for task_index in [1]:


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
		

			self.create_env(task_index=task_index)

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
				number_timesteps = flattened_state_sequence.shape[0]
				if number_timesteps<min_lengths[task_index]:
					min_lengths[task_index] = number_timesteps
				if number_timesteps>max_lengths[task_index]:
					max_lengths[task_index] = number_timesteps
				
				# Number of timesteps to downsample to. 
				number_timesteps = int(flattened_state_sequence.shape[0]//self.ds_freq[task_index])

				flattened_state_sequence = resample(flattened_state_sequence, number_timesteps)
				robot_state_sequence = resample(robot_state_sequence, number_timesteps)
				gripper_state_sequence = resample(gripper_state_sequence, number_timesteps)
				object_state_sequence = resample(object_state_sequence, number_timesteps)
				joint_action_sequence = resample(joint_action_sequence, number_timesteps)
				gripper_action_sequence = resample(gripper_action_sequence, number_timesteps)

				# Normalize gripper values. 
				# 1 is right finger. 0 is left finger.  # 1-0 is right-left. 						
				gripper_values = gripper_state_sequence[:,1]-gripper_state_sequence[:,0]
				gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
				gripper_values = 2*gripper_values-1

				# print("EDC:")
				# embed()				
				concatenated_demonstration = np.concatenate([robot_state_sequence,gripper_values.reshape((-1,1))],axis=1)
				concatenated_actions = np.concatenate([joint_action_sequence,gripper_action_sequence],axis=1)
				
				object_rel_traj = self.compute_relative_object_state(robot_state_sequence, object_state_sequence)
				intermediate_obj_state_seq = np.concatenate([object_state_sequence[:,:7], object_rel_traj, object_state_sequence[:,7:]],axis=-1)
				object_state_sequence = intermediate_obj_state_seq

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
			# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array.npy"),task_demo_array)
			np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_RelObjState.npy"),task_demo_array)
			# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_LiftRelObjState.npy"),task_demo_array)
			# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_LiftObjectQuat.npy"),task_demo_array)

		for j in range(4):
			print("Lengths:", j, min_lengths[j], max_lengths[j])

class Robomimic_Dataset(OrigRobomimic_Dataset):
	
	def __init__(self, args):
		
		super(Robomimic_Dataset, self).__init__(args)	

		# Now that we've run setup, compute dataset_trajectory_lengths for smart batching.
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		for index in range(self.total_length):
			# Get bucket that index falls into based on num_demos array. 
			task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
			
			# Decide task ID, and new index modulo num_demos.
			# Subtract number of demonstrations in cumsum until then, and then 				
			new_index = index-self.cummulative_num_demos[max(task_index,0)]		
			data_element = self.files[task_index][new_index]

			self.dataset_trajectory_lengths[index] = len(data_element['demo'])

		# Now implementing the dataset trajectory length limits. 
		######################################################
		# Now implementing dataset_trajectory_length_limits. 
		######################################################
		
		if self.args.dataset_traj_length_limit>0:
			# Essentially will need new self.cummulative_num_demos and new .. file index map list things. 
			# Also will need to set total_length. 

			self.full_max_length = self.dataset_trajectory_lengths.max()
			self.full_length = copy.deepcopy(self.total_length)
			self.full_cummulative_num_demos = copy.deepcopy(self.cummulative_num_demos)
			self.full_num_demos = copy.deepcopy(self.num_demos)
			self.full_files = copy.deepcopy(self.files)
			self.files = [[] for i in range(len(self.task_list))]
			self.full_dataset_trajectory_lengths = copy.deepcopy(self.dataset_trajectory_lengths)
			self.dataset_trajectory_lengths = []
			self.num_demos = np.zeros(len(self.task_list),dtype=int)

			for index in range(self.full_length):
				# Get bucket that index falls into based on num_demos array. 
				task_index = np.searchsorted(self.full_cummulative_num_demos, index, side='right')-1
				# Get the demo index in this task list. 
				new_index = index-self.full_cummulative_num_demos[max(task_index,0)]

				# Check the length of this particular trajectory and its validity. 
				if (self.full_dataset_trajectory_lengths[index] < self.args.dataset_traj_length_limit) and (index not in self.bad_original_index_list):
					# Add from old list to new. 
					self.files[task_index].append(self.full_files[task_index][new_index])
					self.dataset_trajectory_lengths.append(self.full_dataset_trajectory_lengths[index])
					self.num_demos[task_index] += 1
				else:
					pass

					# Reduce count. 
					# self.num_demos[task_index] -= 1
					
					# # Pop item from files. It's still saved in full_files. 					
					# # self.files[task_index].pop(new_index)
					# self.files[task_index] = np.delete(self.files[task_index],new_index)
					# Approach with opposite pattern.. instead of deleting invalid files, add valid ones.
					
					# # Pop item from dataset_trajectory_lengths. 
					# self.dataset_trajectory_lengths = np.delete(self.dataset_trajectory_lengths, index)

			# Set new cummulative num demos. 
			self.cummulative_num_demos = self.num_demos.cumsum()
			self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
			# Set new total length.
			self.total_length = self.cummulative_num_demos[-1]
			# Make array.
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)

			for t in range(len(self.task_list)):
				self.files[t] = np.array(self.files[t])

			# By popping element from files / dataset_traj_lengths, we now don't need to change indexing.
		

	def setup(self):
		self.files = []
		for i in range(len(self.task_list)):
			self.files.append(np.load("{0}/{1}/New_Task_Demo_Array.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))

	def __getitem__(self, index):

		if index>=self.total_length:
			print("Out of bounds of dataset.")
			return None

		# Get bucket that index falls into based on num_demos array. 
		task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
		
		# Decide task ID, and new index modulo num_demos.
		# Subtract number of demonstrations in cumsum until then, and then 				
		new_index = index-self.cummulative_num_demos[max(task_index,0)]		
		data_element = self.files[task_index][new_index]

		resample_length = len(data_element['demo'])//self.args.ds_freq
		# print("Orig:", len(data_element['demo']),"New length:",resample_length)

		self.kernel_bandwidth = self.args.smoothing_kernel_bandwidth
		
		# Trivially adding task ID to data element.
		data_element['task-id'] = task_index
		data_element['environment-name'] = self.environment_names[task_index]

		if resample_length<=1 or data_element['robot-state'].shape[0]<=1:
			data_element['is_valid'] = False			
		else:
			data_element['is_valid'] = True

			if self.args.smoothen: 
				data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['robot-state'] = gaussian_filter1d(data_element['robot-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')

			# data_element['environment-name'] = self.environment_names[task_index]

		return data_element
	
	def compute_statistics(self):

		self.state_size = 8
		self.total_length = self.__len__()
		mean = np.zeros((self.state_size))
		variance = np.zeros((self.state_size))
		mins = np.zeros((self.total_length, self.state_size))
		maxs = np.zeros((self.total_length, self.state_size))
		lens = np.zeros((self.total_length))

		# And velocity statistics. 
		vel_mean = np.zeros((self.state_size))
		vel_variance = np.zeros((self.state_size))
		vel_mins = np.zeros((self.total_length, self.state_size))
		vel_maxs = np.zeros((self.total_length, self.state_size))
		
		for i in range(self.total_length):

			print("Phase 1: DP: ",i)
			data_element = self.__getitem__(i)

			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				mins[i] = demo.min(axis=0)
				maxs[i] = demo.max(axis=0)
				mean += demo.sum(axis=0)
				lens[i] = demo.shape[0]

				vel_mins[i] = abs(vel).min(axis=0)
				vel_maxs[i] = abs(vel).max(axis=0)
				vel_mean += vel.sum(axis=0)			

		mean /= lens.sum()
		vel_mean /= lens.sum()

		for i in range(self.total_length):

			print("Phase 2: DP: ",i)
			data_element = self.__getitem__(i)
			
			# Just need to normalize the demonstration. Not the rest. 
			if data_element['is_valid']:
				demo = data_element['demo']
				vel = np.diff(demo,axis=0)
				variance += ((demo-mean)**2).sum(axis=0)
				vel_variance += ((vel-vel_mean)**2).sum(axis=0)

		variance /= lens.sum()
		variance = np.sqrt(variance)

		vel_variance /= lens.sum()
		vel_variance = np.sqrt(vel_variance)

		max_value = maxs.max(axis=0)
		min_value = mins.min(axis=0)

		vel_max_value = vel_maxs.max(axis=0)
		vel_min_value = vel_mins.min(axis=0)

		np.save("Robomimic_Mean.npy", mean)
		np.save("Robomimic_Var.npy", variance)
		np.save("Robomimic_Min.npy", min_value)
		np.save("Robomimic_Max.npy", max_value)
		np.save("Robomimic_Vel_Mean.npy", vel_mean)
		np.save("Robomimic_Vel_Var.npy", vel_variance)
		np.save("Robomimic_Vel_Min.npy", vel_min_value)
		np.save("Robomimic_Vel_Max.npy", vel_max_value)

class Robomimic_ObjectDataset(Robomimic_Dataset):

	def __init__(self, args):

		super(Robomimic_ObjectDataset, self).__init__(args)

	def super_getitem(self, index):

		return super().__getitem__(index)

	def __getitem__(self, index):
		
		data_element = copy.deepcopy(super().__getitem__(index))

		# Copy over the demo to the robot-demo key.
		data_element['robot-demo'] = copy.deepcopy(data_element['demo'])
		# Set demo to object-state trajectory. 

		# Also try ignoring the relative positions for now.
		# print("Embedding in get el")
		# embed()
		if self.args.object_pure_relative_state:
			start_index = 7
		else:
			start_index = 0

		data_element['demo'] = data_element['object-state'][:,start_index:start_index+7]

		return data_element

	def setup(self):
		self.files = []
		for i in range(len(self.task_list)):
			self.files.append(np.load("{0}/{1}/New_Task_Demo_Array_RelObjState.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))


class Robomimic_RobotObjectDataset(Robomimic_Dataset):

	def __init__(self, args):

		super(Robomimic_RobotObjectDataset, self).__init__(args)

	def super_getitem(self, index):

		return super().__getitem__(index)

	def __getitem__(self, index):

		data_element = copy.deepcopy(super().__getitem__(index))

		# print("######################")
		# print("Embed in GI")
		# embed()

		data_element['robot-demo'] = copy.deepcopy(data_element['demo'])
		if self.args.object_pure_relative_state:
			start_index = 7
		else:
			start_index = 0

		object_traj = data_element['object-state'][:,start_index:start_index+7]
		
		demo = np.concatenate([data_element['demo'],object_traj],axis=-1)
		data_element['demo'] = copy.deepcopy(demo)

		# print("SHAPE OF 2nd DEMO",data_element['demo'].shape)

		return data_element

	def setup(self):
		self.files = []
		for i in range(len(self.task_list)):
			self.files.append(np.load("{0}/{1}/New_Task_Demo_Array_RelObjState.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))
