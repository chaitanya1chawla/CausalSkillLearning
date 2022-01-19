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

class GRAB_PreDataset(Dataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):

		# Some book-keeping first. 
		self.args = args
		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/data/tanmayshankar/Datasets/GRAB/GRAB_Joints/'
		else:
			self.dataset_directory = self.args.datadir
		   

		# 1) Keep track of joints: 
		#   a) Full joint name list from https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py. 
		#   b) Relevant joint names. 
		# 2) This lets us subsample relevant joints from full joint name list by indexing...
	
		# Logging all the files we need. 
		self.file_path = os.path.join(self.dataset_directory, '*/*_body_joints.npz')
		self.filelist = glob.glob(self.file_path)

		# Get number of files. 
		self.total_length = len(self.filelist)

		# Set downsampling frequency.
		self.ds_freq = 16        

		# Setup. 
		self.setup()

	def set_relevant_joints(self):

		self.joint_names = np.array(['pelvis',
							'left_hip',
							'right_hip',
							'spine1',
							'left_knee',
							'right_knee',
							'spine2',
							'left_ankle',
							'right_ankle',
							'spine3',
							'left_foot',
							'right_foot',
							'neck',
							'left_collar',
							'right_collar',
							'head',
							'left_shoulder',
							'right_shoulder',
							'left_elbow',
							'right_elbow',
							'left_wrist',
							'right_wrist',
							'jaw',
							'left_eye_smplhf',
							'right_eye_smplhf',
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
							'nose',
							'right_eye',
							'left_eye',
							'right_ear',
							'left_ear',
							'left_big_toe',
							'left_small_toe',
							'left_heel',
							'right_big_toe',
							'right_small_toe',
							'right_heel',
							'left_thumb',
							'left_index',
							'left_middle',
							'left_ring',
							'left_pinky',
							'right_thumb',
							'right_index',
							'right_middle',
							'right_ring',
							'right_pinky',
							'right_eye_brow1',
							'right_eye_brow2',
							'right_eye_brow3',
							'right_eye_brow4',
							'right_eye_brow5',
							'left_eye_brow5',
							'left_eye_brow4',
							'left_eye_brow3',
							'left_eye_brow2',
							'left_eye_brow1',
							'nose1',
							'nose2',
							'nose3',
							'nose4',
							'right_nose_2',
							'right_nose_1',
							'nose_middle',
							'left_nose_1',
							'left_nose_2',
							'right_eye1',
							'right_eye2',
							'right_eye3',
							'right_eye4',
							'right_eye5',
							'right_eye6',
							'left_eye4',
							'left_eye3',
							'left_eye2',
							'left_eye1',
							'left_eye6',
							'left_eye5',
							'right_mouth_1',
							'right_mouth_2',
							'right_mouth_3',
							'mouth_top',
							'left_mouth_3',
							'left_mouth_2',
							'left_mouth_1',
							'left_mouth_5',  # 59 in OpenPose output
							'left_mouth_4',  # 58 in OpenPose output
							'mouth_bottom',
							'right_mouth_4',
							'right_mouth_5',
							'right_lip_1',
							'right_lip_2',
							'lip_top',
							'left_lip_2',
							'left_lip_1',
							'left_lip_3',
							'lip_bottom',
							'right_lip_3'])

		self.arm_joint_names = np.array(['pelvis',
							'left_collar',
							'right_collar',
							'left_shoulder',
							'right_shoulder',
							'left_elbow',
							'right_elbow',
							'left_wrist',
							'right_wrist'])

		self.arm_and_hand_joint_names = np.array(['pelvis',
							'left_collar',
							'right_collar',
							'left_shoulder',
							'right_shoulder',
							'left_elbow',
							'right_elbow',
							'left_wrist',
							'right_wrist',
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
							'left_thumb',
							'left_index',
							'left_middle',
							'left_ring',
							'left_pinky',
							'right_thumb',
							'right_index',
							'right_middle',
							'right_ring',
							'right_pinky'])

		# Create index arrays
		self.arm_joint_indices = np.zeros(len(self.arm_joint_names))
		self.arm_and_hand_joint_indices = np.zeros(len(self.arm_and_hand_joint_names))

		for k, v in enumerate(self.arm_joint_names):
			self.arm_joint_indices[k] = np.where(self.joint_names==v)[0][0]

		for k, v in enumerate(self.arm_and_hand_joint_indices):
			self.arm_and_hand_joint_indices[k] = np.where(self.joint_names==v)[0][0]
		
	def subsample_relevant_joints(self, datapoint):

		# Remember, the datapoint is going to be of the form.. 
		# Timesteps x Joints x 3 (dimensions). 
		# Index into it as: 

		# Figure out whether to use full hands, or just use the arm positions. 
		# Consider unsupervised translation to robots without articulated grippers. 
		# For now use arm joint indices. 
		# We can later consider adding other robots / hands.
		self.relevant_joint_indices = self.arm_joint_indices

		return datapoint[:, self.relevant_joint_indices]
		
	def setup(self):

		# Load all files.. 
		self.files = []
		self.dataset_trajectory_lengths = np.zeros(self.total_length)
		
		# For all files. 
		for k, v in enumerate(self.filelist):
						
			if k%100==0:
				print("Loading file: ",k)

			# Now actually load file. 
			datapoint = np.load(v, allow_pickle=True)['body_joints']

			# Subsample relevant joints. 
			relevant_joints_datapoint = self.subsample_relevant_joints(datapoint)

			# Subsample in time. 
			number_of_timesteps = datapoint.shape[0]//self.ds_freq
			subsampled_data = resample(relevant_joints_datapoint, number_of_timesteps)            
			
			# Add subsampled datapoint to file. 
			self.files.append(subsampled_data)            

		# Create array. 
		self.file_array = np.array(self.files)

		# Now save this file.
		np.save(os.path.join(self.dataset_directory,"GRAB_DataFile.npy"), self.file_array)                

	def __len__(self):
		return self.total_length

	def __getitem__(self, index):

		if isinstance(index, np.ndarray):
			return list(self.file_array[index])
		else:
			return self.file_array[index]   

class GRAB_Dataset(Dataset):

	def __init__(self, args, split='train', short_traj=False, traj_length_threshold=500):

		# Some book-keeping first. 
		self.args = args

		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			# self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
			self.dataset_directory = '/data/tanmayshankar/Datasets/GRAB/GRAB_Joints/'
		else:
			self.dataset_directory = self.args.datadir
		   
		# Load file.
		self.data_list = np.load(os.path.join(self.dataset_directory,"GRAB_DataFile.npy"))
		self.dataset_length = len(self.data_list)

		if short_traj:
			length_threshold = traj_length_threshold
			self.short_data_list = []
			self.dataset_trajectory_lengths = []
			for i in range(self.dataset_length):
				if self.data_list[i].shape[0]<length_threshold:
					self.short_data_list.append(self.data_list[i])
					self.dataset_trajectory_lengths.append(self.data_list[i].shape[0])

			self.data_list = self.short_data_list
			self.dataset_length = len(self.data_list)
			self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)		
				
		self.data_list_array = np.array(self.data_list)

	def __len__(self):
		# Return length of file list. 
		return self.dataset_length

	def __getitem__(self, index):
		# Return n'th item of dataset.
		# This has already processed everything.

		if isinstance(index,np.ndarray):			
			return list(self.data_list_array[index])
		else:
			return self.data_list[index]
