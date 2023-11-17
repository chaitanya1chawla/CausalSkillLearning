from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *
from scipy.spatial.transform import Rotation as R

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class RealWorldHumanRigid_PreDataset(Dataset): 

	# Class implementing instance of RealWorld Human Rigid Body Dataset. 
	def __init__(self, args):
		
		self.args = args
		if self.args.datadir is None:
			self.dataset_directory = '/scratch/cchawla/RigidBodyHumanData/'
		else:
			self.dataset_directory = self.args.datadir			
		
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		self.task_list = ['Pouring', 'BoxOpening', 'DrawerOpening', 'PickPlace', 'Stirring']		
		self.environment_names = ['Pouring', 'BoxOpening', 'DrawerOpening', 'PickPlace', 'Stirring']
		self.num_demos = np.array([10, 10, 6, 10, 10])

		# Each task has different number of demos according to our Human Dataset.
		self.number_tasks = len(self.task_list)
		self.cummulative_num_demos = self.num_demos.cumsum()
		# [0, 10, 20, 26, 36, 46]
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)		
		self.total_length = self.num_demos.sum()		

		# self.ds_freq = 1*np.ones(self.number_tasks).astype(int)
		self.ds_freq = np.array([6, 6, 7, 8, 8])

		# Set files. 
		# TODO : check if values from set_ground_tag_pose_dict() are still valid
		self.set_ground_tag_pose_dict()
		self.setup()
		

		self.stat_dir_name ='RealWorldHumanRigid'
				
	def normalize_quaternion(self, q):
		# Define quaternion normalization function.
		return q/np.linalg.norm(q)

	def tag_preprocessing(self, cam_tag_detections=None):
		# 1) Preprocess tag poses to convert the data structure into -
		# data['tag_detections']['cam#'][tag#] = {'position[]', 'orientation[]', 'valid[]'} 
		tag_dict = {
			'tag0': {'position':[], 'orientation':[], 'validity':[]},
			'tag1': {'position':[], 'orientation':[], 'validity':[]},
			'tag2': {'position':[], 'orientation':[], 'validity':[]},
			  }
		
		expected_tags_ids=[0,1,2]
		detected_tag_ids=[]

		for tags_detect in cam_tag_detections:
			# looping over timesteps t
			for tag in tags_detect:
				# looping over tags detected in a single frame:
				id = tag['tag_id']
				tag_dict['tag{}'.format(id)]['position'].append(tag['position'])
				tag_dict['tag{}'.format(id)]['orientation'].append(tag['orientation'])
				tag_dict['tag{}'.format(id)]['validity'].append(1)
				detected_tag_ids.append(id)
			
			# adding Null Data for undetected tags
			not_detected_tags = list( set(expected_tags_ids)-set(detected_tag_ids) )
			
			if len(not_detected_tags) != 0:
				for tag_id in not_detected_tags:
					tag_dict['tag{}'.format(tag_id)]['position'].append(np.array(0,0,0))
					tag_dict['tag{}'.format(tag_id)]['orientation'].append(np.array(0,0,0,0))
					tag_dict['tag{}'.format(tag_id)]['validity'].append(0)

		return tag_dict

	def interpolate_keypoint(self, cam_keypoint_sequence=None):
		# TODO : check if the demonstration['valid_keypoint_frames'] get detected here automatically
		
		# Distance between keypoint [0-1, 1-2, 2-3, 3-4] is >30 and <38  --- class 1 
		# Distance between keypoint [5-6, 6-7, 7-8], [9-10, 10-11, 11-12], [13-14, 14-15, 15-16], 19-20 is >22.5 and <29.5 --- class 2 
		# Distance between keypoint [17-18, 18-19] is >16 and <19 --- class 3
		
		### both_cams_data = {}
		### for idx, cam_keypoint_sequence in enumerate(both_cams_keypoint_sequence.values()):
		### 
		### 	# looping over each camera:
			
		keypoint_sequence = np.array(cam_keypoint_sequence)
		valid_timesteps = np.zeros((len(keypoint_sequence)))
		
		for j in keypoint_sequence:
			# looping over each timestep
			flag = True
			# point pairs defined by seeing mmpose dataset: onehand10k
			# dist limits are defined by approximating distances on own hand 
			point_pairs_1 = [(0,1), (1,2), (2,3), (3,4)]
			for pair in point_pairs_1:
				dist = np.linalg.norm(j[pair[0], :], j[pair[1], :])
				if not (dist > 0.30 and dist < 0.38):
					flag = False

			point_pairs_2 = [(5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (19,20)]
			for pair in point_pairs_2:
				dist = np.linalg.norm(j[pair[0], :], j[pair[1], :])
				if not (dist > 0.225 and dist < 0.295):
					flag = False

			point_pairs_3 = [(17,18), (18,19)]
			for pair in point_pairs_3:
				dist = np.linalg.norm(j[pair[0], :], j[pair[1], :])
				if not (dist > 0.16 and dist < 0.19):
					flag = False

			if flag==True:
				valid_timesteps[j]=1

		valid_indices = np.where(valid_timesteps)[0]
		first_valid_index = valid_indices[0]
		last_valid_index = valid_indices[-1]
		# for single_keypoint in keypoint_sequence:
		# 	# looping over each timestep

		# Interpolate positions 
		both_cams_data = self.interpolate_keypoint_position(valid=valid_timesteps, \
								 position_sequence=keypoint_sequence[first_valid_index:last_valid_index+1])
			
		return both_cams_data

	def interpolate_keypoint_position(self, valid=None, position_sequence=None):
		
		from scipy import interpolate

		# Interp1d from Scipy expects the last dimension to be the dimension we are interpolating over. 
		valid_positions = np.swapaxes(position_sequence[valid==1], 2, 0)
		valid_times = np.where(valid==1)[0]
		query_times = np.arange(0, len(position_sequence))

		# Create interpolating function. 
		interpolating_function = interpolate.interp1d(valid_times, valid_positions)

		# Query interpolating function. 
		interpolated_positions = interpolating_function(query_times)

		# Swap axes back and return. 
		return np.swapaxes(interpolated_positions, 2, 0)

	def interpolate_position(self, valid=None, position_sequence=None):
		
		from scipy import interpolate

		# Interp1d from Scipy expects the last dimension to be the dimension we are interpolating over. 
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

	def fuse_keypoint_data(self, both_cam_keypoint_sequence, valid_keypoint_frames, primary_keypoint_cam):
		
		valid_keypoint_data = np.zeros_like(both_cam_keypoint_sequence['cam0'])

		both_valid = np.where(np.array(valid_keypoint_frames['cam0'])==1) and np.where(np.array(valid_keypoint_frames['cam1'])==1)
		valid_keypoint_data[both_valid] = both_cam_keypoint_sequence['cam{}'.format(primary_keypoint_cam)][both_valid]

		cam0_valid = np.where(np.array(valid_keypoint_frames['cam0'])==1) and np.where(np.array(valid_keypoint_frames['cam1'])==0)
		valid_keypoint_data[cam0_valid] = both_cam_keypoint_sequence['cam{}'.format(0)][cam0_valid]

		cam1_valid = np.where(np.array(valid_keypoint_frames['cam0'])==0) and np.where(np.array(valid_keypoint_frames['cam1'])==1)
		valid_keypoint_data[cam1_valid] = both_cam_keypoint_sequence['cam{}'.format(1)][cam1_valid]

		return valid_keypoint_data 

	def compute_relative_poses(self, demonstration):
    
		from scipy.spatial.transform import Rotation as R
    
		# Get poses of object1 and object2 with respect to ground. 
		demonstration['object1_pose'] = {}
		demonstration['object2_pose'] = {}
    
		##  Construct homogenous transformation matrices. 
		# ground_in_camera_homogenous_matrices = create_batch_matrix(demonstration['ground_cam_frame_pose']['position'], demonstration['ground_cam_frame_pose']['orientation'])
		# object1_in_camera_homogenous_matrices = create_batch_matrix(demonstration['object1_cam_frame_pose']['position'], demonstration['object1_cam_frame_pose']['orientation'])
		# object2_in_camera_homogenous_matrices = create_batch_matrix(demonstration['object2_cam_frame_pose']['position'], demonstration['object2_cam_frame_pose']['orientation'])
    
		# Inverse of the ground in camera.
		# camera_in_ground_homogenous_matrices = invert_batch_matrix(ground_in_camera_homogenous_matrices)
    
		##  Now transform with batch multiplication. 
		# object1_in_ground_homogenous_matrices = np.matmul(camera_in_ground_homogenous_matrices, object1_in_camera_homogenous_matrices)
		# object2_in_ground_homogenous_matrices = np.matmul(camera_in_ground_homogenous_matrices, object2_in_camera_homogenous_matrices)
    
		# # Now retieve poses.
		# demonstration['object1_pose']['position'] = object1_in_ground_homogenous_matrices[:,:3,-1]
		# demonstration['object2_pose']['position'] = object2_in_ground_homogenous_matrices[:,:3,-1]
		# demonstration['object1_pose']['orientation'] = R.from_matrix(object1_in_ground_homogenous_matrices[:,:3,:3]).as_quat()
		# demonstration['object2_pose']['orientation'] = R.from_matrix(object2_in_ground_homogenous_matrices[:,:3,:3]).as_quat()
    
		return demonstration

	def invert(self, homogenous_matrix):
	
		inverse = np.zeros((4,4))
		rotation = R.from_matrix(homogenous_matrix[:3,:3])
		inverse[:3, :3] = rotation.inv().as_matrix()
		inverse[:3, -1] = -rotation.inv().apply(homogenous_matrix[:3,-1])
		inverse[-1, -1] = 1.

		return inverse

	def transform_point_3d_from_cam_to_ground(self, points, gnd_R, gnd_t):

		idx = np.arange(3)
		points_in_cam_hmat = np.zeros((len(points), 4, 4))
		# Saving the rotation matrix as identity
		points_in_cam_hmat[:, idx, idx] = 1.
		points_in_cam_hmat[:, :3, -1] = np.array(points)
		points_in_cam_hmat[:, -1, -1] = 1.

		gnd_in_cam_hmat = np.zeros((4,4))
		gnd_in_cam_hmat[:3, :3] = gnd_R
		gnd_in_cam_hmat[:3, -1] = np.reshape(gnd_t,(3,))
		gnd_in_cam_hmat[-1, -1] = 1.

		cam_in_gnd_hmat = self.invert(gnd_in_cam_hmat)

		points_in_gnd_hmat = np.matmul(cam_in_gnd_hmat, points_in_cam_hmat)

		# Retrieve new points
		points_in_gnd_position = points_in_gnd_hmat[:, :3, -1]

		return points_in_gnd_position

	def transform_pose_from_cam_to_ground(self, pose_R, pose_t, gnd_R, gnd_t):
	
		object_in_cam_hmat = np.zeros((len(pose_R), 4, 4))
		object_in_cam_hmat[:, :3, :3] = pose_R
		object_in_cam_hmat[:, :3, -1] = pose_t
		object_in_cam_hmat[:, -1, -1] = 1.	

		gnd_in_cam_hmat = np.zeros((4,4))
		gnd_in_cam_hmat[:3, :3] = gnd_R
		gnd_in_cam_hmat[:3, -1] = np.reshape(gnd_t,(3,))
		gnd_in_cam_hmat[-1, -1] = 1.	

		cam_in_gnd_hmat = self.invert(gnd_in_cam_hmat)	

		# Transform
		# world_in_ground_hmat = invert(ground_in_world_homogenous_matrix)
		object_in_gnd_hmat = np.matmul(cam_in_gnd_hmat, object_in_cam_hmat)	

		# Retrieve pose. 
		object_in_gnd_R = object_in_gnd_hmat[:, :3, :3]
		object_in_gnd_t = object_in_gnd_hmat[:, :3, -1]	

		return object_in_gnd_t, object_in_gnd_R

	def downsample_data(self, demonstration, task_index, ds_freq=None):

		if ds_freq is None:
			ds_freq = self.ds_freq[task_index]
		# Downsample each stream.
		number_timepoints = int(demonstration['demo'].shape[0] // ds_freq)

		# for k in demonstration.keys():
		key_list = ['hand-state', 'object-state', 'demo']
		if self.args.images_in_real_world_dataset:
			key_list.append('images')
		for k in key_list:
			demonstration[k] = resample(demonstration[k], number_timepoints)
		
	def collate_states(self, demonstration):

		# We're going to collate states, and remove irrelevant ones.  
		# First collect object states.
		new_demonstration = {}
		# Makes it a 7 element long item per array element
		demonstration['object1_state'] = np.concatenate([ demonstration['object1_gnd_frame_pose']['position'], \
														demonstration['object1_gnd_frame_pose']['orientation']], axis=-1)
		demonstration['object2_state'] = np.concatenate([ demonstration['object2_gnd_frame_pose']['position'], \
														demonstration['object2_gnd_frame_pose']['orientation']], axis=-1)

		# Makes it a 14 element long item per array element
		new_demonstration['object-state'] = np.concatenate([ demonstration['object1_state'], \
						  								demonstration['object2_state']], axis=-1)
		
		# Before collating hand states, flatten the 21x3 matrices first.
		demonstration['flat_keypoints'] = demonstration['keypoints'].flatten()
		demonstration['flat_keypoints'].reshape(-1, 63) # where 63 stands for 21x3 values per frame

		# First collate hand states. 
		new_demonstration['hand-state'] = demonstration['flat_keypoints']
		new_demonstration['demo'] = np.concatenate([ new_demonstration['hand-state'], \
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
		# Assumes demonstration is a tuple, with name of task as first element and data as second element. 
		# data is a dictionary, with 4 Keys:
		# {'frame_id', 'raw_keypoints', 'avg_keypoints_score', 'tag_detections'}
		# 		data['frame_id'] is int[], with size=t timesteps 
		# 		data['raw_keypoints'] is dict:
		# 				{'cam0', 'cam1'} #
		# 						data['raw_keypoints']['cam0'] -> 21x3matrices[]  ---> 21 keypoints' positions wrt gnd tag
		# 						data['raw_keypoints']['cam1'] -> 21x3matrices[] 			
		# 		data['avg_keypoint_score'] is dict:
		# 				{'cam0', 'cam1'} -> each element is double[], size=t timesteps
		#		data['valid_keypoint_frames'] is dict:
		#				{'cam0', 'cam1'} -> each element is int[], size=timesteps
		#		data['primary_keypoint_camera'] is int
		# 		data['tag_detections_in_cam'] is dict:
		# 				{'cam0', 'cam1'}:
		# 						data['tag_detections_in_cam']['cam#'] -> [ [], [], [], .... ]:
		# 							each element is list of n dictionaries (n = number of visible tags): [ {}, {}, {} ]
		# 								each dictionary:
		# 									{ 'tag_id', 'position', 'orientation', 'pose_err' } ---> positions and orientations wrt gnd tag
		#		data['primary_camera'] is int
		#		data['images'] is dict:
		#				{'cam0', 'cam1'}:
		#						data['images']['cam#'] is list, with size=t timesteps
		#
		# After processing:
		#		data['keypoints'] has keypoints in gnd frame
		#		data['tag_detections'] has tag poses in gnd frame, with a new structure defined below
		##########################################


		##########################################
		# Things to do within this function. 
		##########################################
		
		# 0) Preprocess tag pose data to convert into a new structure
		# 1) Convert keypoints and tags to ground frame
		# 2) Fuse camera for keypoints   
		# 3) Select primary camera, by given criteria
		# 4) For primary camera, retrieve and interpolate tag poses for that camera. 
		# 5) Collate all the relevant data streams, remove irrelevant data streams.
		# 6) Downsample all relevant data streams. 		
		# 7) Return new demo file. 

		#############
		# 0) Preprocess tag poses to convert the data structure into -
		# data['tag_detections']['cam#']
		# 		{'tag1', 'tag2', 'tag3'}
		# 				'tag#' = {'position[]', 'orientation[]', 'valid[]'} 
		#############
		
		for idx, cam_tag_detections in enumerate(demonstration['tag_detections_in_cam'].values()):
			demonstration['tag_detections_in_cam']['cam{}'.format(idx)] = self.tag_preprocessing(cam_tag_detections)

		#############
		# 1) Convert keypoints and tags to ground frame
		#############

		demo_length = demonstration['frame_id'].shape[0]
		demonstration['ground_cam_frame_pose'] = self.ground_pose_dict   #self.set_ground_tag_pose( length=demo_length, primary_camera=demonstration['primary_camera'] )

		# demonstration['keypoints'] has keypoints in ground frame
		demonstration['keypoints'] = {'cam0':[], 'cam1':[]}
		# demonstration['tag_detections'] has tag_detections in ground frame
		demonstration['tag_detections'] = {
											'cam0':{'tag1':{'position':[], 'orientation':[]},
				   									'tag2':{'position':[], 'orientation':[]}
												}, 
										
											'cam1':{'tag1':{'position':[], 'orientation':[]},
				   									'tag2':{'position':[], 'orientation':[]}
												}
										}

		for cam_num, cam in enumerate( ['cam0', 'cam1'] ):

			gnd_cam_R=demonstration['ground_cam_frame_pose'][str(cam_num)]['orientation']
			gnd_cam_t=demonstration['ground_cam_frame_pose'][str(cam_num)]['position']
			for idx in range(demo_length):

				keypoint_data = demonstration['raw_keypoints'][cam][idx]
				demonstration['keypoints'][cam][idx] = self.transform_point_3d_from_cam_to_ground(keypoint_data, gnd_cam_R, gnd_cam_t)


			for tag in demonstration['tag_detections_in_cam'][cam]:

				tag_data = demonstration['tag_detections_in_cam'][cam][tag]
				pose_R = np.array(tag_data['orientation'])     #(len(tag), 3, 3))
				pose_t = np.zeros(tag_data['position'])  #(len(tag), 3))
				
				new_tag_data = demonstration['tag_detections'][cam][tag]
				new_tag_data['position'], new_tag_data['orientation'] = self.transform_pose_from_cam_to_ground(pose_R, pose_t, gnd_cam_R, gnd_cam_t)

		#############
		# 2) Fuse camera for keypoints    
		#############

		demonstration['keypoints'] = self.fuse_keypoint_data(demonstration['keypoints'], demonstration['valid_keypoint_frames'], demonstration['primary_keypoint_cam'])
		
		# demonstration['keypoints']['cam{}'.format(self.primary_keypoint_cam)]    KEY: PRIMARY_KEYPOINT_CAM
		## and then interpolate keypoints only for that camera.
		# edit collate function accordingly

		#############
		# 3) Interpolate keypoints, when they don't maintain normal distance between each other (ie normal distance between finger joints of a person)
		#############		
		
		demonstration['keypoints'] = self.interpolate_keypoint(demonstration['keypoints'])

		#############
		# 4) For primary camera, retrieve tag poses. 
		#############
		
		# Now, instead of interpolating the ground tag detection from the camera frame, set it to constant value. 
		# demonstration['ground_cam_frame_pose'] = self.interpolate_pose( demonstration['tag0']['cam{0}'.format(demonstration['primary_camera'])] )				

		demo_length = demonstration['frame_id'].shape[0]
		demonstration['ground_cam_frame_pose'] = self.set_ground_tag_pose( length=demo_length, primary_camera=demonstration['primary_camera'] )
		demonstration['object1_gnd_frame_pose'] = self.interpolate_pose( demonstration['tag_detections']['cam{0}'.format(demonstration['primary_camera'])]['tag1'] )
		demonstration['object2_gnd_frame_pose'] = self.interpolate_pose( demonstration['tag_detections']['cam{0}'.format(demonstration['primary_camera'])]['tag2'] )
		
		# #############
		# # NOT COMPUTED -- Compute relative poses.
		# #############
		# 
		# demonstration = self.compute_relative_poses(demonstration=demonstration)

		#############
		# 5) Stack relevant data that we care about. 
		#############

		demonstration = self.collate_states(demonstration=demonstration)

		#############
		# 6) Downsample.
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
		print("About to process real world HUMAN dataset.")

		for task_index in range(len(self.task_list)):
		# for task_index in range(0):
			self.task_demo_array = []

			print("###################################")
			print("Processing task: ", task_index, " of ", self.number_tasks)

			# Set file path for this task.
			#alt_task_file_path = os.path.join(self.dataset_directory, self.task_list[task_index], 'NumpyDemos')
			task_file_path = os.path.join(self.dataset_directory, self.task_list[task_index], 'ImageData')			

			#########################	
			# For every demo in this task
			#########################

			#for j in range(self.num_demos[task_index]):			
			for j, file in enumerate(sorted(glob.glob(os.path.join(task_file_path,'*.npy')))):
			
				print("####################")
				print("Processing demo: ", j, " of ", self.num_demos[task_index], " from task ", task_index)
				
				# file = os.path.join(task_file_path, 'demo{0}.npy'.format(j))
				# demonstration = np.load(file, allow_pickle=True).item()
				# file = os.path.join(task_file_path, 'demo{0}.npy'.format(j))
				#alt_file = os.path.join(alt_task_file_path, 'demo{0}.npy'.format(j))
				task_name, demonstration = np.load(file, allow_pickle=True)
				#alt_demonstration = np.load(alt_file, allow_pickle=True).item()
				#demonstration['primary_camera'] = alt_demonstration['primary_camera']

				#########################
				# Now process in whatever way necessary. 
				#########################

				processed_demonstration = self.process_demonstration(demonstration, task_index)

				self.task_demo_array.append(processed_demonstration)


			# For each task, save task_file_list to One numpy. 
			suffix = ""
			# Simply saving everything with Images by default

			#if self.args.images_in_real_world_dataset:
			#	suffix = "_wSingleImages"
			task_numpy_path = os.path.join(self.dataset_directory, self.task_list[task_index], "New_Task_Demo_Array{0}.npy".format(suffix))
			np.save(task_numpy_path, self.task_demo_array)

	def __len__(self):
		return self.total_length
	
	def __getitem__(self, index):

		return {}	


class RealWorldHumanRigid_Dataset(RealWorldHumanRigid_PreDataset):
	
	def __init__(self, args):
		
		super(RealWorldHumanRigid_Dataset, self).__init__(args)	

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

		if resample_length<=1 or data_element['hand-state'].shape[0]<=1:
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
				data_element['hand-state'] = gaussian_filter1d(data_element['hand-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')

			if self.args.data in ['RealWorldHumanRigidHand']:
				data_element['demo'] = data_element['hand-state']
			# data_element['environment-name'] = self.environment_names[task_index]

		return data_element
	
	####################################
	#### Needs to be edited --- self.state_size
	###################################
	
	def compute_statistics(self, prefix='RealWorldHumanRigid'):

		if prefix=='RealWorldHumanRigid':
			self.state_size = 77
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
