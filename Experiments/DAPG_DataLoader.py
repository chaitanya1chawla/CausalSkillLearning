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

# def wrist_norm(relevant_joints_datapoint):
# 	relevant_joints_datapoint[:, 1:21] -= relevant_joints_datapoint[:, 0].reshape(relevant_joints_datapoint.shape[0], 1, 3)
# 	if len(relevant_joints_datapoint[0]) > 22:
# 		relevant_joints_datapoint[:, 22:] -= relevant_joints_datapoint[:, 21].reshape(relevant_joints_datapoint.shape[0], 1, 3)
# 	return relevant_joints_datapoint


class DAPG_PreDataset(Dataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):

		# Some book-keeping first. 
		self.args = args
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/home/ahassan/CausalSkillLearning/Experiments/dapg/hand_dapg/dapg/demonstrations/'
		else:
			self.dataset_directory = self.args.datadir
		   

		# 1) Keep track of joints: 
		#   a) Full joint name list from https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py. 
		#   b) Relevant joint names. 
		# 2) This lets us subsample relevant joints from full joint name list by indexing...
	
		# Logging all the files we need. 
		self.file_path = os.path.join(self.dataset_directory, '*_demos.pickle')
		self.filelist = sorted(glob.glob(self.file_path))

		# Get number of files. 
		self.total_length = len(self.filelist)

		# Set downsampling frequency.
		self.ds_freq = 5

		# Setup. 
		self.setup()

		self.compute_statistics()

	def set_relevant_joints(self):

		# Create index arrays
		self.hand_joint_indices = {}
		self.hand_joint_indices['door-v0_demos.pickle'] = list(range(4, 28))
		self.hand_joint_indices['pen-v0_demos.pickle'] = list(range(0, 24))
		self.hand_joint_indices['hammer-v0_demos.pickle'] = list(range(2, 26))
		self.hand_joint_indices['relocate-v0_demos.pickle'] = list(range(6, 30))

		

	def subsample_relevant_joints(self, datapoint, dataset_name):

		# Remember, the datapoint is going to be of the form.. 
		# Timesteps x Joints x 3 (dimensions). 
		# Index into it as: 

		# Figure out whether to use full hands, or just use the arm positions. 
		# Consider unsupervised translation to robots without articulated grippers. 
		# For now use arm joint indices. 
		# We can later consider adding other robots / hands.

		# Used a dictionary to hold different indices by file

		sampled_joints = np.zeros((datapoint.shape[0], 30))

		self.relevant_joint_indices = self.hand_joint_indices[dataset_name]
		sampled_joints[:, 6:30] = datapoint[:, self.relevant_joint_indices]

		sampled_joints[:, 0:6] = [0, 0.1, 0.1, 0, 0, 0]

		if dataset_name == 'door-v0_demos.pickle':
			sampled_joints[:, 2:6] = datapoint[:, 0:4]
		if dataset_name == 'relocate-v0_demos.pickle':
			sampled_joints[:, 0:6] = datapoint[:, 0:6]
		if dataset_name == 'hammer-v0_demos.pickle':
			sampled_joints[:, 3:5] = datapoint[:, 0:2]

		return sampled_joints
		
	def setup(self):

		# Load all files.. 
		self.files = []
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		
		# set joints
		self.set_relevant_joints()

		# self.cumulative_num_demos = [0, cumulative_length1, c]
		self.cumulative_num_demos = [0]

		# For all files. 
		for k, v in enumerate(self.filelist):
						
			print("Loading from file: ", v)

			# Now actually load file. 
			set = np.load(v, allow_pickle=True)

			for item in set:

				datapoint = item['observations']

				# Subsample relevant joints. 
				# Modified for different dimensions by file.
				v = v.replace(self.dataset_directory, '')

				relevant_joints_datapoint = self.subsample_relevant_joints(datapoint, v)
				print("Preloading from", v)

				# normalized_relevant_joint_datapoint = self.normalize(relevant_joints_datapoint)

				# Reshape. 
				reshaped_normalized_datapoint = relevant_joints_datapoint.reshape(relevant_joints_datapoint.shape[0],-1)

				self.state_size = reshaped_normalized_datapoint.shape[1]

				# Subsample in time. 
				number_of_timesteps = datapoint.shape[0]//self.ds_freq
				# subsampled_data = resample(relevant_joints_datapoint, number_of_timesteps)
				subsampled_data = resample(reshaped_normalized_datapoint, number_of_timesteps)
				
				# Add subsampled datapoint to file. 
				self.files.append(subsampled_data)      
			self.cumulative_num_demos.append(len(self.files))
			print("Cumulative length:", len(self.files))      

		# Create array. 
		self.file_array = np.array(self.files)

		# Now save this file.
		# np.save(os.path.join(self.dataset_directory,"GRAB_DataFile.npy"), self.file_array)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_DataFile_BaseNormalize.npy"), self.file_array)
		np.save(os.path.join(self.dataset_directory, self.getname() + "_OrderedFileList.npy"), self.filelist)

		# Save cumulative lengths
		np.save(os.path.join(self.dataset_directory, self.getname() + "_Lengths.npy"), self.cumulative_num_demos)

	def normalize(self, relevant_joints_datapoint):
		return relevant_joints_datapoint

	def getname(self):
		return "DAPG"

	def __len__(self):
		return self.total_length

	def __getitem__(self, index):
		
		if isinstance(index, np.ndarray):
			return list(self.file_array[index])
		else:
			return self.file_array[index]

	def compute_statistics(self):

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
			data_element = {}
			data_element['is_valid'] = True
			data_element['demo'] = self.file_array[i]
			data_element['file'] = self.filelist[i]

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
			data_element = {}
			data_element['is_valid'] = True
			data_element['demo'] = self.file_array[i]
			data_element['file'] = self.filelist[i]
			
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

		statdir = "Statistics/" + self.getname()

		np.save(os.path.join(statdir, self.getname() + "_Mean.npy"), mean)
		np.save(os.path.join(statdir, self.getname() + "_Var.npy"), variance)
		np.save(os.path.join(statdir, self.getname() + "_Min.npy"), min_value)
		np.save(os.path.join(statdir, self.getname() + "_Max.npy"), max_value)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Mean.npy"), vel_mean)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Var.npy"), vel_variance)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Min.npy"), vel_min_value)
		np.save(os.path.join(statdir, self.getname() + "_Vel_Max.npy"), vel_max_value)

class DAPG_Dataset(Dataset):

	def __init__(self, args):

		# Some book-keeping first. 
		self.args = args

		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = self.dataset_directory = '/home/ahassan/CausalSkillLearning/Experiments/dapg/hand_dapg/dapg/demonstrations'

		else:
			self.dataset_directory = self.args.datadir
		   
		# Load file.
		self.data_list = np.load(os.path.join(self.dataset_directory, self.getname() + "_DataFile_BaseNormalize.npy"), allow_pickle=True)
		self.filelist = np.load(os.path.join(self.dataset_directory, self.getname() + "_OrderedFileList.npy"), allow_pickle=True)
		self.cumulative_num_demos = np.load(os.path.join(self.dataset_directory, self.getname() + "_Lengths.npy"), allow_pickle=True)

		self.dataset_length = len(self.data_list)
		

		if self.args.dataset_traj_length_limit>0:			
			self.short_data_list = []
			self.short_file_list = []
			self.dataset_trajectory_lengths = []
			for i in range(self.dataset_length):
				if self.data_list[i].shape[0]<self.args.dataset_traj_length_limit:
					self.short_data_list.append(self.data_list[i])
					self.dataset_trajectory_lengths.append(self.data_list[i].shape[0])

			for i in range(len(self.filelist)):
				self.short_file_list.append(self.filelist[i])


			self.data_list = self.short_data_list
			self.filelist = self.short_file_list
			self.dataset_length = len(self.data_list)
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)
				
		self.data_list_array = np.array(self.data_list)		

		self.environment_names = []
		for i in range(len(self.filelist)):
			f = self.filelist[i][81:-7] # remove path and .pickle
			for j in range(self.cumulative_num_demos[i]):
				self.environment_names.append(f)
		print("Env names:\n", np.unique(self.environment_names))


	def getname(self):
		return "DAPG"

	def __len__(self):
		# Return length of file list. 
		return self.dataset_length

	def __getitem__(self, index):
		# Return n'th item of dataset.
		# This has already processed everything.

		# if isinstance(index,np.ndarray):			
		# 	return list(self.data_list_array[index])
		# else:
		# 	return self.data_list[index]

		data_element = {}
		data_element['is_valid'] = True
		data_element['demo'] = self.data_list[index]

		# task_index = np.searchsorted(self.cumulative_num_demos, index, side='right')-1
		# data_element['file'] = self.filelist[task_index][81:-7]
		data_element['file'] = self.environment_names[index]

		return data_element