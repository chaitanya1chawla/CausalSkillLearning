from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class RealWorldRigid_PreDataset(Dataset): 

	# Class implementing instance of RealWorld Rigid Body Dataset. 
	def __init__(self, args):
		
		self.args = args
		if self.args.datadir is None:
			self.dataset_directory = '/home/tshankar/Research/Code/Data/Datasets/RoboMimic/'
		else:
			self.dataset_directory = self.args.datadir			
		
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		self.task_list = ['Pouring', 'BoxOpening', 'DrawerOpening', 'PickPlace', 'Stirring']
		# self.environment_names = ['']

		# Each task has 200 demos according to RoboMimic.
		self.number_tasks = len(self.task_list)
		# self.num_demos = 10*np.ones((self.number_tasks),dtype=int)
		self.num_demos = np.array([10, 10, 6, 10, 10])
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)		
		self.total_length = self.num_demos.sum()		

		self.ds_freq = 1*np.ones(self.number_tasks)

		# Set files. 
		self.setup()

		self.stat_dir_name='Robomimic'

		# Define quaternion normalization function.
	def normalize_quaternion(self, q):
		return q/np.linalg.norm(q)
	
	def interpolate_position(self, valid=None, position_sequence=None):
		
		from scipy import interpolate

		# Interp1d from Scipy expects the last dimension to be the dimension we are ninterpolating over. 
		valid_positions = np.swapaxes(position_sequence[valid==1], 1, 0)
		valid_times = np.where(valid==1)[0]
		query_times = np.arange(0, len(position_sequence))

		# Create interpolating function. 
		interpolating_function = interpolate.interp1d(valid_times, valid_positions)

		# Query interpolating function. 
		interpolated_positions = interpolating_function(query_times)

		# Swap axes back and return. 
		return np.swapaxes(interpolated_positions, 1, 0)
	
	def interpolate_orientation(self, valid=None, orientation_sequence=None):

		from scipy.spatial.transform import Rotation as R
		from scipy.spatial.transform import Slerp

		valid_orientations = orientation_sequence[valid==1]
		rotation_sequence = R.concatenate(R.from_quat(valid_orientations))
		valid_times = np.where(valid==1)[0]
		query_times = np.arange(0, len(orientation_sequence))

		# Create slerp object. 
		slerp_object = Slerp(valid_times, rotation_sequence)

		# Query the slerp object. 
		interpolated_rotations = slerp_object(query_times)

		# Convert to quaternions.
		interpolated_quaternion_sequence = interpolated_rotations.as_quat(canonical=True)
		
		return interpolated_quaternion_sequence

	def interpolate_pose(self, pose_sequence):

		# Assumes pose_sequence is a dictionary with 3 keys: valid, position, and orientation. 
		# valid is 1 when the pose stored is valid, and 0 otherwise. 

		# Pass all data until last valid pose. 
		traj_length = pose_sequence['valid'].shape[0]
		valid_indices = np.where(pose_sequence['valid'])[0]
		first_valid_index = valid_indices[0]
		last_valid_index = valid_indices[-1]

		# This should really be in the parent function, that only calls this function if some poses are Not valid. 
		if (pose_sequence['valid']==1).all():
			return pose_sequence
			
		# Interpolate positions and orientations. 
		interpolated_positions = self.interpolate_position(valid=pose_sequence['valid'][first_valid_index:last_valid_index+1], \
						     position_sequence=pose_sequence['position'][first_valid_index:last_valid_index+1])
		interpolated_orientations = self.interpolate_orientation(valid=pose_sequence['valid'][first_valid_index:last_valid_index+1], \
							orientation_sequence=pose_sequence['orientation'][first_valid_index:last_valid_index+1])

		# Copy interpolated position until last valid index. 
		pose_sequence['position'][first_valid_index:last_valid_index] = interpolated_positions
		pose_sequence['orientation'][first_valid_index:last_valid_index] = interpolated_orientations

		# Copy over valid poses to start of trajectory and end of trajectory if invalid:
		if first_valid_index>0:
			# Until first valid index, these are all invalid. 
			pose_sequence['position'][:first_valid_index] = pose_sequence['position'][first_valid_index]
			pose_sequence['orientation'][:first_valid_index] = pose_sequence['orientation'][first_valid_index]
		
		# Check if last valid index is before the end of the trajectory, i.e. if there are invalid points at the end. 
		if last_valid_index<(traj_length-1):
			pose_sequence['position'][last_valid_index+1:] = pose_sequence['position'][last_valid_index]
			pose_sequence['orientation'][last_valid_index+1:] = pose_sequence['orientation'][last_valid_index]

		return pose_sequence

	def process_demonstration(self, demonstration):

		##########################################
		# Structure of data. 
		##########################################		
		# Assumes demonstration is a dictionary. 
		# Keys in the dictionary correspond to different data. The keys that we care about are: 
		# rs_angle, rs_pose, gripper_state, js_pos, tag0, tag1, tag2, primary_camera. 
		# Within tag#:
		# 	camera#:
		# 		valid, position, orientation. 
		##########################################

		##########################################
		# Things to do within this function. 
		##########################################

		# 0) First select primary camera. 
		# 1) For primary camerya

		# In this function, we are going to interpolate tag data, compute relative poses of tags, and downsample. 

		# Get data streams we care about. 
		# Joint States, Robot States, Gripper States, Ground State, Object 1 State, Object 2 State. 
		# Basically if we just use robot_states that's enough.. 

		# Interpolate. 

		# Downsample each stream.
		number_timepoints = demonstration.shape[0] // self.ds_freq[i]


	
		
		# Smoothen if necessary. 

		# Compute relative pose..

		
	def setup(self):
		# Load data from all tasks. 			
		# numpy_data_path = os.path.join(self.dataset_directory, "*/Numpy_Files/*.npy")
		# self.file_list = sorted(glob.glob(numpy_data_path))
		# self.files = []

		# # For all demos.
		# for k, v in enumerate(self.file_list):
		# 	# Actually load numpy files. 
		# 	self.files.append(np.load(v, allow_pickle=True))

		# For each task, set file list. 
		# self.files = [[] for k in range(self.number_tasks)]

		#########################
		# For each task
		#########################

		print("###################################")
		print("About to process real world dataset.")

		for i in range(len(self.task_list)):

			self.task_demo_array = []

			print("###################################")
			print("Processing task: ", i, " of ", self.number_tasks)

			# Set file path for this task.
			task_file_path = os.path.join(self.dataset_directory, self.task_list[i], 'Numpy_Demos')

			#########################	
			# For every demo in this task
			#########################

			for j in range(self.num_demos[i]):
				
				print("###################################")
				print("Processing task: ", i, " of ", self.number_tasks)

				file = os.path.join(task_file_path, 'Demo_{0}.npy'.format(j+1))
				demonstration = np.load(file, allow_pickle=True)

				#########################
				# Now process in whatever way necessary. 
				#########################

				processed_demonstration = self.process_demonstration(demonstration)

				self.task_demo_array.append(processed_demonstration)


			# For each task, save task_file_list to One numpy. 
			task_numpy_path = os.path.join(self.dataset_directory, self.task_list[i], "New_Task_Demo_Array.npy")
			np.save(self.task_demo_array, task_numpy_path)

			# # Changing file name.
			# # self.files.append(h5py.File("{0}/{1}/demo.hdf5".format(self.dataset_directory,self.task_list[i]),'r'))
			# self.files.append(h5py.File("{0}/{1}/ph/low_dim.hdf5".format(self.dataset_directory,	self.task_list[i]),'r'))

	def __len__(self):
		return self.total_length
	
	def __getitem__(self, index):

		return {}	

	# def preprocess_dataset(self):

	# 	min_lengths = np.ones((self.number_tasks))*10000
	# 	max_lengths = np.zeros((self.number_tasks))

	# 	for task_index in range(self.number_tasks):
	# 	# for task_index in [1]:


	# 		print("#######################################")
	# 		print("Preprocessing task index: ", task_index)
	# 		print("#######################################")

			
	# 		# Just get robot state size and object state size, from the demonstration of this task. 
	# 		object_state_size = self.files[task_index]['data/demo_0/obs/object'].shape[1]
	# 		robot_state_size = self.files[task_index]['data/demo_0/obs/robot0_joint_pos'].shape[1]
		

	# 		self.create_env(task_index=task_index)

	# 		# Create list of files for this task. 
	# 		task_demo_list = []		

	# 		# For every element in the filelist of the element,
	# 		# for i in range(1,self.num_demos[task_index]+1):
	# 		for i in range(self.num_demos[task_index]):

	# 			print("Preprocessing task index: ", task_index, " Demo Index: ", i, " of: ", self.num_demos[task_index])
			
	# 			# Create list of datapoints for this demonstrations. 
	# 			datapoint = {}
				
	# 			# Get SEQUENCE of flattened states.
	# 			flattened_state_sequence = np.array(self.files[task_index]['data/demo_{0}/states'.format(i)])
	# 			robot_state_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_joint_pos'.format(i)])
	# 			gripper_state_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_gripper_qpos'.format(i)])
	# 			joint_action_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_joint_vel'.format(i)])
	# 			gripper_action_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/robot0_gripper_qvel'.format(i)])
	# 			object_state_sequence = np.array(self.files[task_index]['data/demo_{0}/obs/object'.format(i)]) 

	# 			# Downsample. 
	# 			number_timesteps = flattened_state_sequence.shape[0]
	# 			if number_timesteps<min_lengths[task_index]:
	# 				min_lengths[task_index] = number_timesteps
	# 			if number_timesteps>max_lengths[task_index]:
	# 				max_lengths[task_index] = number_timesteps
				
	# 			# Number of timesteps to downsample to. 
	# 			number_timesteps = int(flattened_state_sequence.shape[0]//self.ds_freq[task_index])

	# 			flattened_state_sequence = resample(flattened_state_sequence, number_timesteps)
	# 			robot_state_sequence = resample(robot_state_sequence, number_timesteps)
	# 			gripper_state_sequence = resample(gripper_state_sequence, number_timesteps)
	# 			object_state_sequence = resample(object_state_sequence, number_timesteps)
	# 			joint_action_sequence = resample(joint_action_sequence, number_timesteps)
	# 			gripper_action_sequence = resample(gripper_action_sequence, number_timesteps)

	# 			# Normalize gripper values. 
	# 			# 1 is right finger. 0 is left finger.  # 1-0 is right-left. 						
	# 			gripper_values = gripper_state_sequence[:,1]-gripper_state_sequence[:,0]
	# 			gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
	# 			gripper_values = 2*gripper_values-1

	# 			# print("EDC:")
	# 			# embed()				
	# 			concatenated_demonstration = np.concatenate([robot_state_sequence,gripper_values.reshape((-1,1))],axis=1)
	# 			concatenated_actions = np.concatenate([joint_action_sequence,gripper_action_sequence],axis=1)
				
	# 			object_rel_traj = self.compute_relative_object_state(robot_state_sequence, object_state_sequence)
	# 			intermediate_obj_state_seq = np.concatenate([object_state_sequence[:,:7], object_rel_traj, object_state_sequence[:,7:]],axis=-1)
	# 			object_state_sequence = intermediate_obj_state_seq

	# 			# Put both lists in a dictionary.
	# 			datapoint['flat-state'] = flattened_state_sequence
	# 			datapoint['robot-state'] = robot_state_sequence
	# 			datapoint['object-state'] = object_state_sequence
	# 			datapoint['demo'] = concatenated_demonstration			
	# 			datapoint['demonstrated_actions'] = concatenated_actions

	# 			# Add this dictionary to the file_demo_list. 
	# 			task_demo_list.append(datapoint)

	# 		# Create array.
	# 		task_demo_array = np.array(task_demo_list)

	# 		# Now save this file_demo_list. 
	# 		# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array.npy"),task_demo_array)
	# 		np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_RelObjState.npy"),task_demo_array)
	# 		# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_LiftRelObjState.npy"),task_demo_array)
	# 		# np.save(os.path.join(self.dataset_directory,self.task_list[task_index],"New_Task_Demo_Array_LiftObjectQuat.npy"),task_demo_array)

	# 	for j in range(4):
	# 		print("Lengths:", j, min_lengths[j], max_lengths[j])

class RealWorldRigid_Dataset(RealWorldRigid_PreDataset):
	
	def __init__(self, args):
		
		super(RealWorldRigid_Dataset, self).__init__(args)	

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

		np.save("RealWorldRigid_Mean.npy", mean)
		np.save("RealWorldRigid_Var.npy", variance)
		np.save("RealWorldRigid_Min.npy", min_value)
		np.save("RealWorldRigid_Max.npy", max_value)
		np.save("RealWorldRigid_Vel_Mean.npy", vel_mean)
		np.save("RealWorldRigid_Vel_Var.npy", vel_variance)
		np.save("RealWorldRigid_Vel_Min.npy", vel_min_value)
		np.save("RealWorldRigid_Vel_Max.npy", vel_max_value)
