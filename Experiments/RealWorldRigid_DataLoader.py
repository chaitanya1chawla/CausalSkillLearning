from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *
from scipy.spatial.transform import Rotation as R

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

def invert(homogenous_matrix):
	
	from scipy.spatial.transform import Rotation as R
	inverse = np.zeros((4,4))
	rotation = R.from_matrix(homogenous_matrix[:3,:3])
	inverse[:3, :3] = rotation.inv().as_matrix()
	inverse[:3, -1] = -rotation.inv().apply(homogenous_matrix[:3,-1])
	inverse[-1, -1] = 1.

	return inverse

def invert_batch_matrix(homogenous_matrix):

	from scipy.spatial.transform import Rotation as R
	inverse = np.zeros((homogenous_matrix.shape[0], 4,4))
	rotation = R.from_matrix(homogenous_matrix[:,:3,:3])
	inverse[:, :3, :3] = rotation.inv().as_matrix()
	inverse[:, :3, -1] = -rotation.inv().apply(homogenous_matrix[:, :3,-1])
	inverse[:, -1, -1] = 1.

	return inverse

def create_matrix(position, orientation):

	homogenous_matrix = np.zeros((4,4))
	homogenous_matrix[:3,-1] = position
	homogenous_matrix[:3,:3] = R.from_quat(orientation).as_matrix()
	homogenous_matrix[-1,-1] = 1.

	return homogenous_matrix

def create_batch_matrix(position, orientation):

	traj_length = position.shape[0]
	homogenous_matrix = np.zeros((traj_length, 4,4))
	homogenous_matrix[:, :3,-1] = position
	homogenous_matrix[:, :3,:3] = R.from_quat(orientation).as_matrix()
	homogenous_matrix[:,-1,-1] = 1.

	return homogenous_matrix

def define_ground_world_transformation():

	# Defines the transformation between the ground frame (located on the table), and the world frame (located at the base of the robot). 
	rotation = R.from_euler('Z', 180, degrees=True)

	ground_in_world_homogenous_matrix = np.zeros((4,4))
	ground_in_world_homogenous_matrix[:3, :3] = rotation.as_matrix()
	ground_in_world_homogenous_matrix[:3, -1] = np.array([0.595, 0.0145, -0.15])
	ground_in_world_homogenous_matrix[-1, -1] = 1.

	return ground_in_world_homogenous_matrix

def transform_pose_from_ground_to_world(pose_array):

	# Takes a pose in the ground frame and transforms it to the world frame. 
	ground_in_world_homogenous_matrix = define_ground_world_transformation()
	
	# Set pose pose in homogenous matrix
	pose_in_ground_hmat = np.zeros((pose_array.shape[0], 4,4))
	pose_in_ground_hmat[:,:3, :3] = R.from_quat(pose_array[:,3:]).as_matrix()
	pose_in_ground_hmat[:,:3, -1] = pose_array[:,:3]
	pose_in_ground_hmat[:, -1, -1] = 1.

	# Transform
	# world_in_ground_hmat = invert(ground_in_world_homogenous_matrix)
	pose_in_world_hmat = np.matmul(ground_in_world_homogenous_matrix, pose_in_ground_hmat)

	# Retrieve pose. 
	pose_in_world_quaternion = R.from_matrix(pose_in_world_hmat[:, :3, :3]).as_quat()
	pose_in_world_position = pose_in_world_hmat[:, :3, -1]

	return pose_in_world_position, pose_in_world_quaternion

def transform_pose_from_world_to_ground(pose_array):

	# Takes a pose in the ground frame and transforms it to the world frame. 
	ground_in_world_homogenous_matrix = define_ground_world_transformation()
	world_in_ground_homogenous_matrix = invert(ground_in_world_homogenous_matrix)
	
	# Set pose in homogenous matrix
	pose_in_world_hmat = np.zeros((pose_array.shape[0], 4,4))
	pose_in_world_hmat[:,:3, :3] = R.from_quat(pose_array[:,3:]).as_matrix()
	pose_in_world_hmat[:,:3, -1] = pose_array[:,:3]
	pose_in_world_hmat[:, -1, -1] = 1.

	# Transform
	# world_in_ground_hmat = invert(ground_in_world_homogenous_matrix)
	pose_in_ground_hmat = np.matmul(world_in_ground_homogenous_matrix, pose_in_world_hmat)

	# Retrieve pose. 
	pose_in_ground_quaternion = R.from_matrix(pose_in_ground_hmat[:, :3, :3]).as_quat()
	pose_in_ground_position = pose_in_ground_hmat[:, :3, -1]

	return pose_in_ground_position, pose_in_ground_quaternion


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
		self.environment_names = ['Pouring', 'BoxOpening', 'DrawerOpening', 'PickPlace', 'Stirring']
		self.num_demos = np.array([10, 10, 6, 10, 10])

		# self.task_list = ['BoxOpening', 'DrawerOpening', 'PickPlace']
		# self.environment_names = ['BoxOpening', 'DrawerOpening', 'PickPlace']
		# self.num_demos = np.array([10, 6, 10])

		# Each task has 200 demos according to RoboMimic.
		self.number_tasks = len(self.task_list)
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)		
		self.total_length = self.num_demos.sum()		

		# self.ds_freq = 1*np.ones(self.number_tasks).astype(int)
		self.ds_freq = np.array([6, 6, 7, 8, 8])

		# Set files. 
		self.set_ground_tag_pose_dict()
		self.setup()
		

		self.stat_dir_name ='RealWorldRigid'
				
	def normalize_quaternion(self, q):
		# Define quaternion normalization function.
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
		interpolated_quaternion_sequence = interpolated_rotations.as_quat()
		
		return interpolated_quaternion_sequence

	def interpolate_pose(self, pose_sequence):

		# Assumes pose_sequence is a dictionary with 3 keys: valid, position, and orientation. 
		# valid is 1 when the pose stored is valid, and 0 otherwise. 

		# Only interpolate if ther are invalid poses. Otherwise jsut return. 
		if (pose_sequence['validity']==1).all():
			return pose_sequence

		# Pass all data until last valid pose. 
		traj_length = pose_sequence['validity'].shape[0]
		valid_indices = np.where(pose_sequence['validity'])[0]
		first_valid_index = valid_indices[0]
		last_valid_index = valid_indices[-1]

		# Interpolate positions and orientations. 
		interpolated_positions = self.interpolate_position(valid=pose_sequence['validity'][first_valid_index:last_valid_index+1], \
							 position_sequence=pose_sequence['position'][first_valid_index:last_valid_index+1])
		interpolated_orientations = self.interpolate_orientation(valid=pose_sequence['validity'][first_valid_index:last_valid_index+1], \
							orientation_sequence=pose_sequence['orientation'][first_valid_index:last_valid_index+1])

		# Copy interpolated position until last valid index. 
		pose_sequence['position'][first_valid_index:last_valid_index+1] = interpolated_positions
		pose_sequence['orientation'][first_valid_index:last_valid_index+1] = interpolated_orientations

		# Copy over valid poses to start of trajectory and end of trajectory if invalid:
		if first_valid_index>0:
			# Until first valid index, these are all invalid. 
			pose_sequence['position'][:first_valid_index] = pose_sequence['position'][first_valid_index]
			pose_sequence['orientation'][:first_valid_index] = pose_sequence['orientation'][first_valid_index]
		
		# Check if last valid index is before the end of the trajectory, i.e. if there are invalid points at the end. 
		if last_valid_index<(traj_length-1):
			pose_sequence['position'][last_valid_index+1:] = pose_sequence['position'][last_valid_index]
			pose_sequence['orientation'][last_valid_index+1:] = pose_sequence['orientation'][last_valid_index]

		# Pop validity. 
		pose_sequence.pop('validity')

		return pose_sequence

	def compute_relative_poses(self, demonstration):

		from scipy.spatial.transform import Rotation as R

		# Get poses of object1 and object2 with respect to ground. 
		demonstration['object1_pose'] = {}
		demonstration['object2_pose'] = {}

		# Construct homogenous transformation matrices. 
		ground_in_camera_homogenous_matrices = create_batch_matrix(demonstration['ground_cam_frame_pose']['position'], demonstration['ground_cam_frame_pose']['orientation'])
		object1_in_camera_homogenous_matrices = create_batch_matrix(demonstration['object1_cam_frame_pose']['position'], demonstration['object1_cam_frame_pose']['orientation'])
		object2_in_camera_homogenous_matrices = create_batch_matrix(demonstration['object2_cam_frame_pose']['position'], demonstration['object2_cam_frame_pose']['orientation'])

		# Inverse of the ground in camera.
		camera_in_ground_homogenous_matrices = invert_batch_matrix(ground_in_camera_homogenous_matrices)

		# Now transform with batch multiplication. 
		object1_in_ground_homogenous_matrices = np.matmul(camera_in_ground_homogenous_matrices, object1_in_camera_homogenous_matrices)
		object2_in_ground_homogenous_matrices = np.matmul(camera_in_ground_homogenous_matrices, object2_in_camera_homogenous_matrices)

		# Now retieve poses.
		demonstration['object1_pose']['position'] = object1_in_ground_homogenous_matrices[:,:3,-1]
		demonstration['object2_pose']['position'] = object2_in_ground_homogenous_matrices[:,:3,-1]
		demonstration['object1_pose']['orientation'] = R.from_matrix(object1_in_ground_homogenous_matrices[:,:3,:3]).as_quat()
		demonstration['object2_pose']['orientation'] = R.from_matrix(object2_in_ground_homogenous_matrices[:,:3,:3]).as_quat()

		return demonstration

	def downsample_data(self, demonstration, task_index, ds_freq=None):

		if ds_freq is None:
			ds_freq = self.ds_freq[task_index]
		# Downsample each stream.
		number_timepoints = int(demonstration['demo'].shape[0] // ds_freq)

		# for k in demonstration.keys():
		key_list = ['robot-state', 'object-state', 'eef-state', 'demo']
		if self.args.images_in_real_world_dataset:
			key_list.append('images')
		for k in key_list:
			demonstration[k] = resample(demonstration[k], number_timepoints)
		
	def collate_states(self, demonstration):

		# We're going to collate states, and remove irrelevant ones.  
		# First collect object states.
		new_demonstration = {}
		demonstration['object1_state'] = np.concatenate([ demonstration['object1_pose']['position'], \
														demonstration['object1_pose']['orientation']], axis=-1)
		demonstration['object2_state'] = np.concatenate([ demonstration['object2_pose']['position'], \
														demonstration['object2_pose']['orientation']], axis=-1)
		new_demonstration['object-state'] = np.concatenate([ demonstration['object1_state'], \
						  								demonstration['object2_state']], axis=-1)

		# First collate robot states. 
		new_demonstration['robot-state'] = np.concatenate( [demonstration['js_pos'], \
						  						demonstration['gripper_data'].reshape(-1,1) ], axis=-1)
		new_demonstration['eef-state'] = demonstration['rs_pose']
		new_demonstration['demo'] = np.concatenate([ new_demonstration['robot-state'], \
					  							new_demonstration['object-state']], axis=-1)

		if self.args.images_in_real_world_dataset:
			# Put images of primary camera into separate topic.. 
			new_demonstration['images'] = demonstration['images']['cam{0}'.format(demonstration['primary_camera'])]

		return new_demonstration

	def set_ground_tag_pose_dict(self):

		self.ground_pose_dict = {}
		self.ground_pose_dict['0'] = {}
		self.ground_pose_dict['1'] = {}
		
		# Manually set pose, so we don't have to deal with arbitrary flips in the data. 
		self.ground_pose_dict['0']['position'] = np.array([0.13983876, 0.09984951, 0.64977687])
		self.ground_pose_dict['0']['orientation'] = np.array([-0.5428623 ,  0.6971846 , -0.42725178, -0.19154654])

		self.ground_pose_dict['1']['position'] = np.array([0.04871597, 0.07525627, 0.66383035])
		self.ground_pose_dict['1']['orientation'] = np.array([ 0.78081735, -0.42594218,  0.19169668,  0.41490953])		

	def set_ground_tag_pose(self, length=None, primary_camera=None):

		pose_dictionary = {}
		pos = self.ground_pose_dict[str(primary_camera)]['position']
		orient = self.ground_pose_dict[str(primary_camera)]['orientation']

		pose_dictionary['position'] = np.repeat(pos[np.newaxis, :], length, axis=0)
		pose_dictionary['orientation'] = np.repeat(orient[np.newaxis, :], length, axis=0)
		
		return pose_dictionary

	def process_demonstration(self, demonstration, task_index):

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
		
		# 0) For primary camera, retrieve and interpolate tag poses for that camera. 
		# 1) Compute relative poses. 
		# 2) Collate all the relevant data streams, remove irrelevant data streams.
		# 3) Downsample all relevant data streams. 		
		# 4) Return new demo file. 

		#############
		# 0) For primary camera, retrieve tag poses. 
		#############
		
		# Now, instead of interpolating the ground tag detection from the camera frame, set it to constant value. 
		# demonstration['ground_cam_frame_pose'] = self.interpolate_pose( demonstration['tag0']['cam{0}'.format(demonstration['primary_camera'])] )				

		demo_length = demonstration['js_pos'].shape[0]
		demonstration['ground_cam_frame_pose'] = self.set_ground_tag_pose( length=demo_length, primary_camera=demonstration['primary_camera'] )
		demonstration['object1_cam_frame_pose'] = self.interpolate_pose( demonstration['tag1']['cam{0}'.format(demonstration['primary_camera'])] )
		demonstration['object2_cam_frame_pose'] = self.interpolate_pose( demonstration['tag2']['cam{0}'.format(demonstration['primary_camera'])] )
		
		#############
		# 1) Compute relative poses.
		#############
		
		demonstration = self.compute_relative_poses(demonstration=demonstration)

		#############
		# 2) Stack relevant data that we care about. 
		#############

		demonstration = self.collate_states(demonstration=demonstration)

		#############
		# 3) Downsample.
		#############
		
		self.downsample_data(demonstration=demonstration, task_index=task_index)
		# Add task ID to demo. 
		demonstration['task_id'] = task_index
		demonstration['task_name'] = self.task_list[task_index]
			
		# Smoothen if necessary. 

		# return demonstration
		return demonstration
		
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

		for task_index in range(len(self.task_list)):
		# for task_index in range(0):
			self.task_demo_array = []

			print("###################################")
			print("Processing task: ", task_index, " of ", self.number_tasks)

			# Set file path for this task.
			alt_task_file_path = os.path.join(self.dataset_directory, self.task_list[task_index], 'NumpyDemos')
			task_file_path = os.path.join(self.dataset_directory, self.task_list[task_index], 'ImageData')			

			#########################	
			# For every demo in this task
			#########################

			for j in range(self.num_demos[task_index]):			
				
				print("####################")
				print("Processing demo: ", j, " of ", self.num_demos[task_index], " from task ", task_index)
				
				# file = os.path.join(task_file_path, 'demo{0}.npy'.format(j))
				# demonstration = np.load(file, allow_pickle=True).item()
				
				file = os.path.join(task_file_path, 'demo{0}.npy'.format(j))
				alt_file = os.path.join(alt_task_file_path, 'demo{0}.npy'.format(j))
				demonstration = np.load(file, allow_pickle=True).item()
				alt_demonstration = np.load(alt_file, allow_pickle=True).item()
				demonstration['primary_camera'] = alt_demonstration['primary_camera']

				#########################
				# Now process in whatever way necessary. 
				#########################

				processed_demonstration = self.process_demonstration(demonstration, task_index)

				self.task_demo_array.append(processed_demonstration)


			# For each task, save task_file_list to One numpy. 
			suffix = ""
			if self.args.images_in_real_world_dataset:
				suffix = "_wSingleImages"
			task_numpy_path = os.path.join(self.dataset_directory, self.task_list[task_index], "New_Task_Demo_Array{0}.npy".format(suffix))
			np.save(task_numpy_path, self.task_demo_array)

	def __len__(self):
		return self.total_length
	
	def __getitem__(self, index):

		return {}	


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
				if (self.full_dataset_trajectory_lengths[index] < self.args.dataset_traj_length_limit):
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
			# self.files.append(np.load("{0}/{1}/New_Task_Demo_Array.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))
			# self.files.append(np.load("{0}/{1}/New_Task_Demo_Array_wImages.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))
			self.files.append(np.load("{0}/{1}/New_Task_Demo_Array_NewRelPose.npy".format(self.dataset_directory, self.task_list[i]), allow_pickle=True))

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

			###############################
			# If we have additional downsampling, do it here. 
			if self.args.ds_freq>1.:				
				self.downsample_data(data_element, data_element['task-id'], self.args.ds_freq)
				# data_element = 
			
			if self.args.smoothen:
				data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['robot-state'] = gaussian_filter1d(data_element['robot-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')

			if self.args.data in ['RealWorldRigidRobot']:
				data_element['demo'] = data_element['robot-state']
			# data_element['environment-name'] = self.environment_names[task_index]

		return data_element
	
	def compute_statistics(self, prefix='RealWorldRigid'):

		if prefix=='RealWorldRigid':
			self.state_size = 21
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

		np.save("{0}_Mean.npy".format(prefix), mean)
		np.save("{0}_Var.npy".format(prefix), variance)
		np.save("{0}_Min.npy".format(prefix), min_value)
		np.save("{0}_Max.npy".format(prefix), max_value)
		np.save("{0}_Vel_Mean.npy".format(prefix), vel_mean)
		np.save("{0}_Vel_Var.npy".format(prefix), vel_variance)
		np.save("{0}_Vel_Min.npy".format(prefix), vel_min_value)
		np.save("{0}_Vel_Max.npy".format(prefix), vel_max_value)

class RealWorldRigid_JointEEFDataset(RealWorldRigid_Dataset):

	def __init__(self, args):
				
		super(RealWorldRigid_JointEEFDataset, self).__init__(args)	
		self.stat_dir_name ='RealWorldRigidJointEEF'

	def process_end_effector_state(self, end_effector_trajectory):

		# Takes in end effector trajectory of the robot, and:
		# 1) Transforms this to a quaternion. 
		# 2) Transforms this from the world frame to the ground frame. 

		end_effectory_pose_in_world = np.zeros((end_effector_trajectory.shape[0], 7))

		# scale factor is just mm to m, because EE trajectory is mm by default. 
		scale = 1./1000.

		# Convert to quaternion and pose array. 
		end_effectory_pose_in_world[:,:3] = end_effector_trajectory[:,:3] * scale
		end_effectory_pose_in_world[:,3:] = R.from_euler('XYZ', end_effector_trajectory[:,3:]).as_quat()

		# Now transform this to ground frame. 
		end_effectory_pos_in_ground, end_effectory_quat_in_ground = transform_pose_from_world_to_ground(end_effectory_pose_in_world)
		
		return end_effectory_pos_in_ground, end_effectory_quat_in_ground

	def compute_object_eef_relative_state(self, eef_traj, object_traj):

		# Parse into individual position and quaternion. 
		eef_pos = eef_traj[:,:3]
		eef_quat = eef_traj[:,3:]

		object_pos = object_traj[:,:3]
		object_quat = object_traj[:,3:]

		# Construct homogenous matrices for each frame. 
		eef_in_ground_hmats = create_batch_matrix(eef_pos, eef_quat)
		object_in_ground_hmats = create_batch_matrix(object_pos, object_quat)

		# We want the Object pose in the frame of the EEF pose, because there's two objects. This makes more sense than transforming EEF twice / everything to one object frame. 
		
		# Pattern. 
		# # Inverse of the ground in camera.
		# camera_in_ground_homogenous_matrices = invert_batch_matrix(ground_in_camera_homogenous_matrices)

		# # Now transform with batch multiplication. 
		# object1_in_ground_homogenous_matrices = np.matmul(camera_in_ground_homogenous_matrices, object1_in_camera_homogenous_matrices)

		# First invert eef_in_ground. 
		ground_in_eef_hmats = invert_batch_matrix(eef_in_ground_hmats)
		object_in_eef_hmats = np.matmul(ground_in_eef_hmats, object_in_ground_hmats)

		# Recover Poses. 
		object_in_eef_pos = object_in_eef_hmats[:,:3,-1]
		object_in_eef_quat = R.from_matrix(object_in_eef_hmats[:,:3,:3]).as_quat()

		object_in_eef_pose = np.concatenate([object_in_eef_pos, object_in_eef_quat],axis=-1)
		return object_in_eef_pose
		
	def __getitem__(self, index):
		
		# Run super getitem.
		data_element = super().__getitem__(index)

		# Now add the EEF state in addition to the robot joint state and object state...

		# Backup original. 
		data_element['old_demo'] = copy.deepcopy(data_element['demo'])
		# Now create new. 
		
		# Take original eef state, convert orientations to quaternions and transform frames. 
		transformed_eef_pos, transformed_eef_quat = self.process_end_effector_state(data_element['eef-state'])
		data_element['transformed_eef_state'] = np.concatenate([transformed_eef_pos, transformed_eef_quat], axis=-1)

		# Now also retrieve the object poses relative to the EEF.
		data_element['relative-object-state'] = np.zeros_like(data_element['object-state'])
		data_element['relative-object-state'][...,:7] = self.compute_object_eef_relative_state(data_element['transformed_eef_state'], data_element['object-state'][...,:7])
		data_element['relative-object-state'][...,7:] = self.compute_object_eef_relative_state(data_element['transformed_eef_state'], data_element['object-state'][...,7:])

		data_element['demo'] = np.concatenate([ data_element['robot-state'], \
										 		data_element['transformed_eef_state'], \
					  							data_element['object-state']], axis=-1)

		return data_element

	def compute_statistics(self, prefix='RealWorldRigidJointEEF'):
		
		if prefix=='RealWorldRigidJointEEF':
			self.state_size = 28
		return super().compute_statistics(prefix=prefix)


class RealWorldRigid_JointEEF_AbsRelObj_Dataset(RealWorldRigid_JointEEFDataset):

	def __init__(self, args):

		##################################################
		# Make dataset that has the following information:	
		# 0) Robot Joint States
		# 1) Robot EEF States
		# 2) Absolute Object States
		# 3) Relative Object States
		##################################################
							
		super(RealWorldRigid_JointEEF_AbsRelObj_Dataset, self).__init__(args)	
		self.stat_dir_name ='RealWorldRigidJointEEFAbsRelObj'

	def __getitem__(self, index):
		
		# Run super getitem.
		data_element = super().__getitem__(index)

		# Set old demo to: 
		data_element['JEEF_AbsObj_demo'] = data_element['demo']

	
		##################################################
		# Make dataset that has the following information:	
		# 0) Robot Joint States (7)
		# 1) Robot EEF States (7)
		# 2) Absolute Object States (14)
		# 3) Relative Object States (14)
		##################################################

		# Set new demo to: 
		data_element['demo'] = np.concatenate([ data_element['JEEF_AbsObj_demo'], \
										 		data_element['relative-object-state']], axis=-1)

		return data_element
	
	def compute_statistics(self, prefix='RealWorldRigidJointEEFAbsRelObj'):

		if prefix=='RealWorldRigidJointEEFAbsRelObj':
			self.state_size = 28+14
		return super().compute_statistics(prefix=prefix)
