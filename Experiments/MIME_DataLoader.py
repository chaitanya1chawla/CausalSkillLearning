# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *
import os.path as osp

def select_baxter_angles(trajectory, joint_names, arm='right'):
    # joint names in order as used via mujoco visualizer
    baxter_joint_names = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
    if arm == 'right':
        select_joints = baxter_joint_names[:7]
    elif arm == 'left':
        select_joints = baxter_joint_names[7:]
    elif arm == 'both':
        select_joints = baxter_joint_names
    inds = [joint_names.index(j) for j in select_joints]
    return trajectory[:, inds]

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class MIME_Dataset(Dataset):
	'''
	Class implementing instance of dataset class for MIME data. 
	'''
	def __init__(self, split='all'):
		self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
		self.ds_freq = 20

		# Default: /checkpoint/tanmayshankar/MIME/
		self.fulltext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/joint_angles.txt')
		self.filelist = glob.glob(self.fulltext)

		with open(self.filelist[0], 'r') as file:
			lines = file.readlines()
			self.joint_names = sorted(eval(lines[0].rstrip('\n')).keys())

		if split == 'all':
			self.filelist = self.filelist
		else:
			self.task_lists = np.load(os.path.join(
				self.dataset_directory, 'MIME_jointangles/{}_Lists.npy'.format(split.capitalize())))

			self.filelist = []
			for i in range(20):
				self.filelist.extend(self.task_lists[i])
			self.filelist = [f.replace('/checkpoint/tanmayshankar/MIME/', self.dataset_directory) for f in self.filelist]
		# print(len(self.filelist))

	def __len__(self):
		# Return length of file list. 
		return len(self.filelist)

	def __getitem__(self, index):
		'''
		# Returns Joint Angles as: 
		# List of length Number_Timesteps, with each element of the list a dictionary containing the sequence of joint angles. 
		# Assumes index is within range [0,len(filelist)-1]
		'''
		file = self.filelist[index]

		left_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'left_gripper.txt'))
		right_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'right_gripper.txt'))

		orig_left_traj = np.load(osp.join(osp.split(file)[0], 'Left_EE.npy'))
		orig_right_traj = np.load(osp.join(osp.split(file)[0], 'Right_EE.npy'))        

		joint_angle_trajectory = []
		# Open file. 
		with open(file, 'r') as file:
			lines = file.readlines()
			for line in lines:
				dict_element = eval(line.rstrip('\n'))
				if len(dict_element.keys()) == len(self.joint_names):
					# some files have extra lines with gripper keys e.g. MIME_jointangles/4/12405Nov19/joint_angles.txt
					array_element = np.array([dict_element[joint] for joint in self.joint_names])
					joint_angle_trajectory.append(array_element)

		joint_angle_trajectory = np.array(joint_angle_trajectory)

		n_samples = len(orig_left_traj) // self.ds_freq

		elem = {}
		elem['joint_angle_trajectory'] = resample(joint_angle_trajectory, n_samples)
		elem['left_trajectory'] = resample(orig_left_traj, n_samples)
		elem['right_trajectory'] = resample(orig_right_traj, n_samples)
		elem['left_gripper'] = resample(left_gripper, n_samples)/100
		elem['right_gripper'] = resample(right_gripper, n_samples)/100
		elem['path_prefix'] = os.path.split(self.filelist[index])[0]
		elem['ra_trajectory'] = select_baxter_angles(elem['joint_angle_trajectory'], self.joint_names, arm='right')
		elem['la_trajectory'] = select_baxter_angles(elem['joint_angle_trajectory'], self.joint_names, arm='left')
		# If max norm of differences is <1.0, valid. 

		# if elem['joint_angle_trajectory'].shape[0]>1:
		elem['is_valid'] = int(np.linalg.norm(np.diff(elem['joint_angle_trajectory'],axis=0),axis=1).max() < 1.0)

		return elem

	def recreate_dictionary(self, arm, joint_angles):
		if arm=="left":
			offset = 2
			width = 7 
		elif arm=="right":
			offset = 9
			width = 7
		elif arm=="full":
			offset = 0
			width = len(self.joint_names)
		return dict((self.joint_names[i],joint_angles[i-offset]) for i in range(offset,offset+width))

class MIME_NewDataset(Dataset):

	def __init__(self, args, split='all', short_traj=False, traj_length_threshold=500):

		# 
		self.args = args		

		if self.args.datadir is None:
			# self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
			self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/MIME/'
		else:
			self.dataset_directory = self.args.datadir

		# Load the entire set of trajectories. 
		
		self.file = "Data_List.npy"
		self.key = 'demo'
		
		if isinstance(self, MIME_NewMetaDataset):
			# This is not an elif condition, because the first always evaluates to true...		
			# self.file = "MIMEDataArray.npy"
			# Just load the EE augmented version anyway.. just 80Mb bigger
			self.file = "MIMEDataArray_EEAugmented.npy"
		
		self.original_data_list = np.load(os.path.join(self.dataset_directory, self.file),allow_pickle=True)			
		self.original_dataset_length = len(self.original_data_list)

		# Now only selecting valid datapoints.
		self.data_list = []
		for i in range(self.original_dataset_length):
			if self.original_data_list[i]['is_valid']:
				self.data_list.append(self.original_data_list[i])

		self.dataset_length = len(self.data_list)

		if short_traj:
			
			length_threshold = traj_length_threshold
			self.short_data_list = []
			self.dataset_trajectory_lengths = []
			for i in range(self.dataset_length):
				if self.data_list[i][self.key].shape[0]<length_threshold:
					self.short_data_list.append(self.data_list[i])
					self.dataset_trajectory_lengths.append(self.data_list[i][self.key].shape[0])

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

	def compute_statistics(self):

		self.state_size = 16
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

		np.save("MIME_Orig_Mean.npy", mean)
		np.save("MIME_Orig_Var.npy", variance)
		np.save("MIME_Orig_Min.npy", min_value)
		np.save("MIME_Orig_Max.npy", max_value)
		np.save("MIME_Orig_Vel_Mean.npy", vel_mean)
		np.save("MIME_Orig_Vel_Var.npy", vel_variance)
		np.save("MIME_Orig_Vel_Min.npy", vel_min_value)
		np.save("MIME_Orig_Vel_Max.npy", vel_max_value)

class MIME_NewMetaDataset(MIME_NewDataset):

	def __init__(self, args, split='all', short_traj=False, traj_length_threshold=500):

		super(MIME_NewMetaDataset, self).__init__(args, split=split, short_traj=short_traj, traj_length_threshold=traj_length_threshold)

class MIME_OneHandedDataset(MIME_NewMetaDataset):

	def __init__(self, args, split='all', short_traj=False, traj_length_threshold=500):

		super(MIME_OneHandedDataset, self).__init__(args, split=split, short_traj=short_traj, traj_length_threshold=traj_length_threshold)

		# Things that need to be updated based on the one handed nature of this.. 
		# self.data_list = self.short_data_list
		# self.dataset_length = len(self.data_list)
		# self.dataset_trajectory_lengths = np.array(self.dataset_trajectory_lengths)		
		# self.data_list_array = np.array(self.data_list)

		# Create onehanded lists to add selected trajecotries to.
		self.onehanded_data_list = []		
		self.onehanded_dataset_length = 0
		self.onehanded_dataset_trajectory_lengths = []

		# For every element in the data list, add it to the one handed list if the particular hand is active in this trajcetory? 
		# If the other hand is inactive? Hmm...
		# Difference is about bimanual skills - do we want these? Probably not... 	
		for k, v in enumerate(self.data_list):			

			# Add to left handed data list if the right hand is inactive..
			if self.args.single_hand=='left':						
				# Right hand indexing is [:,:7] for all timesteps..
				joint_traj = v['demo'][:,:7]
			# Add to right handed data list if the left hand is inactive..
			elif self.args.single_hand=='right':
				# Left hand indexing is [:,7:14] for all timesteps..
				joint_traj = v['demo'][:,7:14]

			# Now check if the off hand is active
			# Recycling older version of the validity / movement definition.. 
			offhand_inactive = (np.linalg.norm(np.diff(joint_traj,axis=0),axis=1).max() < 0.02)

			if offhand_inactive:
				self.onehanded_data_list.append(v)
				self.onehanded_dataset_trajectory_lengths.append(v['demo'].shape[0])
				# self.onehanded_dataset_trajectory_lengths.append(self.dataset_trajectory_lengths[k])
				self.onehanded_dataset_length += 1
								
		# Rename these elements, so other classes can access them, and the inherited length and get_item functions work.
		self.dataset_length = self.onehanded_dataset_length
		self.dataset_trajectory_lengths = np.array(self.onehanded_dataset_trajectory_lengths)		
		self.data_list = self.onehanded_data_list
		self.data_list_array = np.array(self.data_list)

class MIME_Dataloader_Tester(unittest.TestCase):
	
	def test_MIMEdataloader(self):

		self.dataset = MIME_NewDataset()

		# Check the first index of the dataset.
		data_element = self.dataset[0]

		validity = data_element['is_valid']==1
		check_demo_data = (data_element['demo']==np.load("Test_Data/MIME_Dataloader_DE.npy")).all()

		self.assertTrue(validity and check_demo_data)

if __name__ == '__main__':
	# Run all tests defined for the dataloader.
    unittest.main()
