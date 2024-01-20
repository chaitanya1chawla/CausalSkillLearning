from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, sys
import glob
import copy
from IPython import embed
from scipy.spatial.transform import Rotation as R

def normalize_quaternion(self, q):
	# Define quaternion normalization function.
	return q/np.linalg.norm(q)

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

def invert(self, homogenous_matrix):

	inverse = np.zeros((4,4))
	rotation = R.from_matrix(homogenous_matrix[:3,:3])
	inverse[:3, :3] = rotation.inv().as_matrix()
	inverse[:3, -1] = -rotation.inv().apply(homogenous_matrix[:3,-1])
	inverse[-1, -1] = 1.

	return inverse

def transform_pose_from_cam_to_ground(self, pose_R, pose_t, gnd_R, gnd_t):
	
	pose_R = R.from_quat(pose_R).as_matrix()
	gnd_R = R.from_quat(gnd_R).as_matrix()

	object_in_cam_hmat = np.zeros((len(pose_R), 4, 4))
	object_in_cam_hmat[:, :3, :3] = pose_R
	object_in_cam_hmat[:, :3, -1] = pose_t
	object_in_cam_hmat[:, -1, -1] = 1.	

	gnd_in_cam_hmat = np.zeros((4,4))
	gnd_in_cam_hmat[:3, :3] = gnd_R
	gnd_in_cam_hmat[:3, -1] = np.reshape(gnd_t,(3,))
	gnd_in_cam_hmat[-1, -1] = 1.	

	cam_in_gnd_hmat = invert(gnd_in_cam_hmat)	

	# Transform
	# world_in_ground_hmat = invert(ground_in_world_homogenous_matrix)
	object_in_gnd_hmat = np.matmul(cam_in_gnd_hmat, object_in_cam_hmat)	

	# Retrieve pose. 
	object_in_gnd_R = R.from_matrix(object_in_gnd_hmat[:, :3, :3]).as_quat()
	object_in_gnd_t = object_in_gnd_hmat[:, :3, -1]	

	return object_in_gnd_t, object_in_gnd_R

def transform_point_3d_from_cam_to_ground(self, points, gnd_R, gnd_t):
	
	gnd_R = R.from_quat(gnd_R).as_matrix()
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

	cam_in_gnd_hmat = invert(gnd_in_cam_hmat)

	points_in_gnd_hmat = np.matmul(cam_in_gnd_hmat, points_in_cam_hmat)

	# Retrieve new points
	points_in_gnd_position = points_in_gnd_hmat[:, :3, -1]

	return points_in_gnd_position

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

class RealWorldHumanRigid_PreDataset(object): 

	# Class implementing instance of RealWorld Human Rigid Body Dataset. 
	def __init__(self):
		
		self.dataset_directory = '/scratch/cchawla/RigidBodyHumanData/Images/Processed_Demos'
		
		# Require a task list. 
		# The task name is needed for setting the environment, rendering. 
		self.task_list = ['PickPlace', 'BoxOpening', 'Stirring', 'Pouring', 'DrawerOpening']
		self.environment_names = ['PickPlace', 'BoxOpening', 'Stirring', 'Pouring', 'DrawerOpening']    
		self.num_demos = np.array([10, 10, 10, 5, 6])

		# Each task has different number of demos according to our Human Dataset.
		self.number_tasks = len(self.task_list)
		self.cummulative_num_demos = self.num_demos.cumsum()
		# [0, 10, 20, 26, 36, 46]
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)		
		self.total_length = self.num_demos.sum()		

		# self.ds_freq = 1*np.ones(self.number_tasks).astype(int)
		self.ds_freq = np.array([6, 6, 7, 8, 8])

		# Set files. 
		self.set_ground_tag_pose_dict()
		self.setup()
		

		self.stat_dir_name ='RealWorldHumanRigid'

	def tag_preprocessing(self, cam_tag_detections=None, task_name=None):
		
		expected_tags = {'Pouring':[0,2,6], 'Stirring':[0,3,6], 'BoxOpening':[0,1,2], 'DrawerOpening':[0,1,2], 'PickPlace':[0,1,2], 
					     'BoxOpening+Pouring':[0,1,2,6], 'DrawerOpening+PickPlace':[0,1,2,6], 'Pouring+Stirring':[0,2,3,6], 
					     'BoxOpening+Pouring+Stirring':[0,1,2,3,6]}

		# 0) If number of expected_tags is not 5, then add dummy tags (-1, -2, -3 ...)
		for task in expected_tags:
			dummy_id = -1
			while len(expected_tags[task]) < 5:
				expected_tags[task].append(dummy_id)
				dummy_id -= 1

		# 1) Preprocess tag poses to convert the data structure into -
		# data['tag_detections']['cam#'][tag#] = {'position[]', 'orientation[]', 'valid[]'} 
		tag_dict = {}
		for tag_id in expected_tags[task_name]:
			tag_dict['tag{}'.format(tag_id)]= {'position':[], 'orientation':[], 'tag_validity':[]}
		
		detected_tag_ids=[]
		for tags_detect_in_frame in cam_tag_detections:

			# Looping over timesteps t
			for tag in tags_detect_in_frame:
				
				# Looping over tags detected in a single frame:
				id = tag['tag_id']
				# ignore tag if something outside the expected_tags is detected
				if id not in expected_tags[task_name]:
					continue
				tag_dict['tag{}'.format(id)]['position'].append(tag['position'])
				tag_dict['tag{}'.format(id)]['orientation'].append(tag['orientation'])
				
				# nonzero_position or nonzero_orientation, where all coordinates are not 0
				nonzero_orientation = np.count_nonzero(np.array(tag['orientation']))
				nonzero_position = np.count_nonzero(np.array(tag['orientation']))

				if nonzero_orientation ==0 or nonzero_position ==0:
					tag_dict['tag{}'.format(id)]['tag_validity'].append(0)
					continue

				tag_dict['tag{}'.format(id)]['tag_validity'].append(1)
				detected_tag_ids.append(id)
			
			# Add Null Data for undetected tags
			not_detected_tags = list( set(expected_tags[task_name])-set(detected_tag_ids) )
			

			if len(not_detected_tags) != 0:
				for tag_id in not_detected_tags:
					tag_dict['tag{}'.format(tag_id)]['position'].append(np.array([0,0,0]))
					tag_dict['tag{}'.format(tag_id)]['orientation'].append(np.array([0,0,0,0]))
					tag_dict['tag{}'.format(tag_id)]['tag_validity'].append(0)

		return tag_dict

	def pickplace_manual_copying(self, demonstration):
		
		tag1_data = demonstration['tag_detections_in_cam']['cam{}'.format(demonstration['primary_camera'])]['tag1']
		tag2_data = demonstration['tag_detections_in_cam']['cam{}'.format(demonstration['primary_camera'])]['tag2']
		
		last_valid_cup_pose_index = np.where(tag1_data['tag_validity'])[0][-1]
		last_tag1_position = copy.deepcopy(tag1_data['position'][last_valid_cup_pose_index])
		last_tag1_orientation = copy.deepcopy(tag1_data['orientation'][last_valid_cup_pose_index])
		
		tag_data_len = len(demonstration['tag_detections_in_cam']['cam{}'.format(demonstration['primary_camera'])]['tag0']['position'])
		
		tag2_data['position'] = [last_tag1_position] * tag_data_len
		tag2_data['orientation'] = [last_tag1_orientation] * tag_data_len
		tag2_data['tag_validity'] = [1] * tag_data_len

	def no_hand_check(self, demonstration):
		
		for cam in ['cam0', 'cam1']:
			for idx, kpts in enumerate(demonstration['raw_keypoints'][cam]):
				
				# kpts is a 21x3 matrix with 21 points
				if demonstration['valid_keypoint_frames'][cam][idx] == 1:
					# For invalid poses, we do not need to check again
		
					# keypoints is a 21x3 matrix with 21 points in 3dim
					if cam == 'cam0':
						validity=1
						for keypoint in kpts:	
							if np.linalg.norm(self.no_hand_pose[cam] - np.array(keypoint)) > 0.05:
								continue
							else:
								validity = 0
						demonstration['valid_keypoint_frames'][cam][idx] = validity

					elif cam == 'cam1':
						validity=1
						for keypoint in kpts:
							if  (np.linalg.norm(self.no_hand_pose[cam]['pose0.0'] - np.array(keypoint)) > 0.05 and 
					   			 np.linalg.norm(self.no_hand_pose[cam]['pose0.1'] - np.array(keypoint)) > 0.05):
								continue
							else:
								validity = 0
						demonstration['valid_keypoint_frames'][cam][idx] = validity

	def rotate_by_180_along_x(self, demonstration):
		
		for cam in ['cam0', 'cam1']:
			print('CAM- ', cam)
			for tag_key in demonstration['tag_detections_in_cam'][cam]:

				tag = demonstration['tag_detections_in_cam'][cam][tag_key]
				# Rotate all tags except ground tag, as we are transforming wrt ground later
				if tag_key != 'tag0':

					print(tag_key)
					tag['orientation'] = np.array(tag['orientation'])
					valid_positions = np.where( np.array(tag['tag_validity']) == 1)
					print('No. of valid positions:', len(valid_positions[0]))
					print('Percent of valid tags in primary_camera= ', len(valid_positions[0]) / self.demo_length)
					if len(valid_positions[0]) == 0:
						continue
					
					pose_R = np.array(tag['orientation'])[valid_positions[0]]
					x_rot = R.from_euler('x', 180, degrees=True).as_matrix()
					transformed_pose = np.matmul(R.from_quat(pose_R).as_matrix(), x_rot)
					tag['orientation'][valid_positions[0]] = R.from_matrix(transformed_pose).as_quat()
					# assigning to variable 'tag' is same as assigning to 'demonstration['tag_detections_in_cam']['cam0']['tag{}']'

		return

	def convert_data_to_ground_frame(self, demonstration, hand_orientation_validity):

		for cam_num, cam in enumerate( ['cam0', 'cam1'] ):

			gnd_cam_R=demonstration['ground_cam_frame_pose'][str(cam_num)]['orientation']
			gnd_cam_t=demonstration['ground_cam_frame_pose'][str(cam_num)]['position']
			for idx in range(self.demo_length):

				keypoint_data = demonstration['raw_keypoints'][cam][idx]
				demonstration['keypoints'][cam].append(transform_point_3d_from_cam_to_ground(keypoint_data, gnd_cam_R, gnd_cam_t))
				
				# Convert hand_orientation to ground frame
				raw_hand_orientation = demonstration['raw_hand_orientation'][cam][idx]
				if np.linalg.norm(raw_hand_orientation) == 0.0:
					demonstration['hand_orientation'][cam].append(np.array([0., 0., 0., 0.]))
					hand_orientation_validity[cam].append(0)
					continue
				else:
					hand_orientation_validity[cam].append(1)
					raw_hand_orientation = normalize_quaternion(raw_hand_orientation)
					gnd_cam_R_mat = R.from_quat(gnd_cam_R).as_matrix()
					# Rotate to bring it wrt ground frame - 
					matrix_product = np.multiply(R.from_quat(raw_hand_orientation).as_matrix(), gnd_cam_R_mat)
					demonstration['hand_orientation'][cam].append(R.from_matrix(matrix_product).as_quat())


			print('Detected tags in {} - '.format(cam))

			# Convert tags to ground frame
			for tag in demonstration['tag_detections_in_cam'][cam]:
				print(tag)

				# Ignore ground tag
				if tag != 'tag0':

					tag_data = demonstration['tag_detections_in_cam'][cam][tag]
					if np.size(np.where(np.array(tag_data['tag_validity'])==1)[0]) ==0:
						print('{} is empty in {}'.format(tag, cam))
						demonstration['tag_detections'][cam][tag] = {
											'position':np.zeros([self.demo_length, 3]), 
											'orientation':np.zeros([self.demo_length, 4]),
											'tag_validity':np.zeros(self.demo_length)
											}
						continue
						
					tag_data['orientation'] = np.array(tag_data['orientation'])     #(len(tag), 3, 3))
					tag_data['position'] = np.array(tag_data['position']) 		    #(len(tag), 3))
					valid_positions = np.array(np.where(np.array(tag_data['orientation']).any(axis=1))[0])[:self.demo_length]

					new_tag_t, new_tag_R = transform_pose_from_cam_to_ground(tag_data['orientation'][valid_positions], tag_data['position'][valid_positions], gnd_cam_R, gnd_cam_t)
				
					if demonstration['tag_detections'][cam].get(tag) == None:
						demonstration['tag_detections'][cam][tag] = {
																	'position':np.zeros([self.demo_length, 3]), 
																	'orientation':np.zeros([self.demo_length, 4]),
																	'tag_validity':np.zeros(self.demo_length)
																	}

					demonstration['tag_detections'][cam][tag]['position'][valid_positions] = new_tag_t
					demonstration['tag_detections'][cam][tag]['orientation'][valid_positions] = new_tag_R
					demonstration['tag_detections'][cam][tag]['tag_validity'][valid_positions] = 1

	def interpolate_keypoint(self, cam_keypoint_sequence=None, keypoints_validity=None):
		# TODO : check if the demonstration['valid_keypoint_frames'] get detected here automatically
		
		# Distance between keypoint [0-1, 1-2, 2-3, 3-4] is >30 and <38  --- class 1 
		# Distance between keypoint [5-6, 6-7, 7-8], [9-10, 10-11, 11-12], [13-14, 14-15, 15-16], 19-20 is >22.5 and <29.5 --- class 2 
		# Distance between keypoint [17-18, 18-19] is >16 and <19 --- class 3
		# UPDATE: taking only thumb and index finger, without their tips.
		# Keypoints: [0, 1, 2, 3, 5, 6, 7]
		
		cam_keypoint_sequence = np.array(cam_keypoint_sequence)
		# keypoint_sequence = np.array(cam_keypoint_sequence)[np.where(keypoints_validity==1)]
		#keypoint_sequence = np.array(cam_keypoint_sequence)
		# valid_timesteps = np.zeros(len(keypoint_sequence))
		ctr = np.zeros(20)

		for idx, j in enumerate(cam_keypoint_sequence):
			
			# Skip iteration if it's already invalid
			if keypoints_validity[idx] == 0:
				continue

			# looping over each timestep
			flag = True
			# point pairs defined by seeing mmpose dataset: onehand10k
			# dist limits are defined by approximating distances on own hand 
			# point_pairs_1 = [(0,1), (1,2), (2,3), (3,4)]
			point_pairs_1 = [(0,1), (1,2), (2,3)]
			for pair in point_pairs_1:
				pt1 = j[pair[0]]
				pt2 = j[pair[1]]
				
				dist = np.linalg.norm(pt1 - pt2)
				if not (dist > -0.001 and dist < 0.090):
					ctr[pair[0]]+=1
					flag = False

			# point_pairs_2 = [(5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (19,20)]
			point_pairs_2 = [(5,6), (6,7)]
			for pair in point_pairs_2:
				# UPDATE: Subtracting 1 as keypoint no. 4 is not included now
				pt1 = j[pair[0]-1]
				pt2 = j[pair[1]-1]

				dist = np.linalg.norm(pt1 - pt2)
				if not (dist > -0.001 and dist < 0.070):
					ctr[pair[0]]+=1
					flag = False

			if flag==False:
				keypoints_validity[idx]=0
				# valid_timesteps[idx]=1

		valid_indices = np.where(keypoints_validity==1)[0]

		print('Percent of valid frame after keypoint-distance filter = ', len(np.where(keypoints_validity==1)[0]) / len(cam_keypoint_sequence))

		first_valid_index = valid_indices[0]
		last_valid_index = valid_indices[-1]

		# Interpolate positions 
		interpolated_positions = self.interpolate_keypoint_position(valid=keypoints_validity[first_valid_index:last_valid_index+1], \
								 position_sequence=cam_keypoint_sequence[first_valid_index:last_valid_index+1])
		
		# Copy interpolated position until last valid index. 
		cam_keypoint_sequence[first_valid_index:last_valid_index+1] = interpolated_positions

		# Copy over valid poses to start of trajectory and end of trajectory if invalid:
		if first_valid_index>0:
			# Until first valid index, these are all invalid. 
			cam_keypoint_sequence[:first_valid_index] = cam_keypoint_sequence[first_valid_index]
		
		# Check if last valid index is before the end of the trajectory, i.e. if there are invalid points at the end. 
		if last_valid_index<(len(keypoints_validity)-1):
			cam_keypoint_sequence[last_valid_index+1:] = cam_keypoint_sequence[last_valid_index]

		return cam_keypoint_sequence

	def interpolate_hand_orientation(self, orientation_array=None, keypoints_validity=None):

		orientation_array = np.array(orientation_array)
		keypoints_validity = np.array(keypoints_validity)
		valid_indices = np.where(keypoints_validity==1)[0]
		first_valid_index = valid_indices[0]
		last_valid_index = valid_indices[-1]

		# Interpolate orientations
		interpolated_orientations = interpolate_orientation(valid=keypoints_validity[first_valid_index:last_valid_index+1], \
								 orientation_sequence=orientation_array[first_valid_index:last_valid_index+1])
		
		# Copy interpolated position until last valid index. 
		orientation_array[first_valid_index:last_valid_index+1] = interpolated_orientations

		# Copy over valid poses to start of trajectory and end of trajectory if invalid:
		if first_valid_index>0:
			# Until first valid index, these are all invalid. 
			orientation_array[:first_valid_index] = orientation_array[first_valid_index]
		
		# Check if last valid index is before the end of the trajectory, i.e. if there are invalid points at the end. 
		if last_valid_index<(len(keypoints_validity)-1):
			orientation_array[last_valid_index+1:] = orientation_array[last_valid_index]

		return orientation_array

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

	def interpolate_pose(self, pose_sequence):

		# Assumes pose_sequence is a dictionary with 3 keys: tag_validity, position, and orientation. 
		# valid is 1 when the pose stored is valid, and 0 otherwise. 

		# Don't interpolate if there are all or no invalid poses. In that case, just return. 
		if (pose_sequence['tag_validity']==1).all() or (pose_sequence['tag_validity']==0).all():
			return pose_sequence

		# Pass all data until last valid pose. 
		traj_length = pose_sequence['tag_validity'].shape[0]
		valid_indices = np.where(pose_sequence['tag_validity'])[0]
		first_valid_index = valid_indices[0]
		last_valid_index = valid_indices[-1]

		# Interpolate positions and orientations. 
		interpolated_positions = interpolate_position(valid=pose_sequence['tag_validity'][first_valid_index:last_valid_index+1], \
							 position_sequence=pose_sequence['position'][first_valid_index:last_valid_index+1])
		interpolated_orientations = interpolate_orientation(valid=pose_sequence['tag_validity'][first_valid_index:last_valid_index+1], \
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
		pose_sequence.pop('tag_validity')

		return pose_sequence

	def fuse_keypoint_data(self, demonstration):

		both_cam_keypoint_sequence = demonstration['keypoints']
		valid_keypoint_frames = demonstration['valid_keypoint_frames']
		primary_keypoint_cam = demonstration['primary_keypoint_camera']

		valid_keypoint_data = np.zeros((len(both_cam_keypoint_sequence['cam0']), self.number_of_keypoints, 3))
		keypoints_validity = np.zeros((len(both_cam_keypoint_sequence['cam0'])))

		# find indices where both cam0 and cam1 give valid keypoints
		both_valid = np.intersect1d(np.where(np.array(valid_keypoint_frames['cam0'])==1)[0], np.where(np.array(valid_keypoint_frames['cam1'])==1)[0])
		if len(both_valid)!=0:
			keypoints_validity[both_valid] = 1
			valid_keypoint_data[both_valid] = [both_cam_keypoint_sequence['cam{}'.format(primary_keypoint_cam)][idx] for idx in both_valid] 

		# find indices where cam0 gives valid keypoints
		cam0_valid = np.intersect1d(np.where(np.array(valid_keypoint_frames['cam0'])==1)[0], np.where(np.array(valid_keypoint_frames['cam1'])==0)[0])
		if len(cam0_valid)!=0:
			keypoints_validity[cam0_valid] = 1
			valid_keypoint_data[cam0_valid] = [both_cam_keypoint_sequence['cam{}'.format(0)][idx] for idx in cam0_valid] 

		# find indices where cam1 gives valid keypoints
		cam1_valid = np.intersect1d(np.where(np.array(valid_keypoint_frames['cam0'])==0)[0], np.where(np.array(valid_keypoint_frames['cam1'])==1)[0])
		if len(cam1_valid)!=0:
			keypoints_validity[cam1_valid] = 1
			valid_keypoint_data[cam1_valid] = [both_cam_keypoint_sequence['cam{}'.format(1)][idx] for idx in cam1_valid] 

		demonstration['keypoints'] = valid_keypoint_data
		demonstration['keypoints_validity'] = keypoints_validity
		print('Percent of valid keypoint frames after fusing cameras = ', len(np.where(np.array(demonstration['keypoints_validity'])==1)[0])/self.demo_length)
	
	def fuse_tag_data(self, demonstration):

		# Overwrite the primary_camera data
		for tag in demonstration['tag_detections']['cam{}'.format(demonstration['primary_camera'])]:
			print(tag)
			
			# tag data from primary camera 
			prim_cam_tag = demonstration['tag_detections']['cam{}'.format(demonstration['primary_camera'])][tag]
			# tag data from non-primary camera 
			non_prim_cam_tag = demonstration['tag_detections']['cam{}'.format(demonstration['primary_camera'] ^ 1)][tag]

			valid_positions = np.where(demonstration['tag_detections']['cam{}'.format(demonstration['primary_camera'])][tag]['tag_validity']==1)[0]
			print('Percent of valid {} frames after fusing cameras = '.format(tag), len(valid_positions)/self.demo_length)

			demo_length = len(prim_cam_tag['tag_validity'])
			valid_tag = {
						'position':np.zeros([demo_length, 3]), 
						'orientation':np.zeros([demo_length, 4]),
						'tag_validity':np.zeros(demo_length)
						}
			
			# find indices where both cam0 and cam1 give valid tag data
			both_valid = np.intersect1d(np.where(prim_cam_tag['tag_validity']==1)[0], np.where(non_prim_cam_tag['tag_validity']==1)[0])
			if len(both_valid)!=0:
				valid_tag['tag_validity'][both_valid] = 1
				valid_tag['position'][both_valid] = [prim_cam_tag['position'][idx] for idx in both_valid] 
				valid_tag['orientation'][both_valid] = [prim_cam_tag['orientation'][idx] for idx in both_valid] 
	
			# find indices where prim_cam gives valid tag data
			prim_valid = np.intersect1d(np.where(prim_cam_tag['tag_validity']==1)[0], np.where(non_prim_cam_tag['tag_validity']==0)[0])
			if len(prim_valid)!=0:
				valid_tag['tag_validity'][prim_valid] = 1
				valid_tag['position'][prim_valid] = [prim_cam_tag['position'][idx] for idx in prim_valid] 
				valid_tag['orientation'][prim_valid] = [prim_cam_tag['orientation'][idx] for idx in prim_valid] 
	
			# find indices where non_prim gives valid tag data
			non_prim_valid = np.intersect1d(np.where(prim_cam_tag['tag_validity']==0)[0], np.where(non_prim_cam_tag['tag_validity']==1)[0])
			if len(non_prim_valid)!=0:
				valid_tag['tag_validity'][non_prim_valid] = 1
				valid_tag['position'][non_prim_valid] = [non_prim_cam_tag['position'][idx] for idx in non_prim_valid] 
				valid_tag['orientation'][non_prim_valid] = [non_prim_cam_tag['orientation'][idx] for idx in non_prim_valid] 

			demonstration['tag_detections']['cam{}'.format(demonstration['primary_camera'])][tag] = valid_tag

			valid_positions = np.where(demonstration['tag_detections']['cam{}'.format(demonstration['primary_camera'])][tag]['tag_validity']==1)[0]
			print('Percent of valid {} frames after fusing cameras = '.format(tag), len(valid_positions)/demo_length)

	def compute_relative_poses(self, demonstration):
        
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

	def downsample_data(self, demonstration, task_index):

		ds_freq = self.ds_freq[task_index]
		# Downsample each stream.
		number_timepoints = int(demonstration['demo'].shape[0] // ds_freq)

		# for k in demonstration.keys():
		key_list = ['hand-state', 'all-object-state', 'demo']
		#if self.args.images_in_real_world_dataset:
		# key_list.append('images')
		for k in key_list:
			demonstration[k] = resample(demonstration[k], number_timepoints)
		
	def collate_states(self, demonstration):

		# We're going to collate states, and remove irrelevant ones.  
		# First collect object states.
		new_demonstration = {}
		demonstration['object_states'] = {}

		# Makes it a 7 element long item per array element
		for idx, tag_pose in enumerate(demonstration['object_gnd_frame_pose']):
			demonstration['object_states']['object{}_state'.format(idx+1)] = np.concatenate([ demonstration['object_gnd_frame_pose'][tag_pose]['position'], \
																		demonstration['object_gnd_frame_pose'][tag_pose]['orientation']], axis=-1)

		# demonstration['object1_state'] = np.concatenate([ demonstration['object1_gnd_frame_pose']['position'], \
		# 												demonstration['object1_gnd_frame_pose']['orientation']], axis=-1)
		# demonstration['object2_state'] = np.concatenate([ demonstration['object2_gnd_frame_pose']['position'], \
		# 												demonstration['object2_gnd_frame_pose']['orientation']], axis=-1)

		
		# Convert values of the dictionary to a list of arrays
		array_list = list(demonstration['object_states'].values())

		# new_demonstration['all-object-state'] = np.concatenate((demonstration['object_states']['object1_state'], demonstration['object_states']['object2_state']),axis=-1)

		# Makes it a 4*7=28 element long item per array element
		new_demonstration['all-object-state'] = np.concatenate(array_list, axis=-1)

		# print('embed after collate obj')
		# embed()

		# Before collating hand states, flatten the 21x3 matrices first.
		demonstration['flat_keypoints'] = demonstration['keypoints'].flatten()
		demonstration['flat_keypoints'] = demonstration['flat_keypoints'].reshape(-1, self.number_of_keypoints*3) # where 63 stands for 21x3 values per frame (UPDATE 7*3 values per frame)

		# First collate hand states. 
		hand_orientation = demonstration['hand_orientation']
		new_demonstration['hand-state'] = np.concatenate((demonstration['flat_keypoints'], hand_orientation), axis=-1) # number_of_keypoints*3 + 4 = 67 values (UPDATE 7*3 + 4 = 25)
		new_demonstration['demo'] = np.concatenate([ new_demonstration['hand-state'], \
					  							new_demonstration['all-object-state']], axis=-1)  # 67+14 = 81 values  (UPDATE  25 + 28 = 53)

		#if self.args.images_in_real_world_dataset:
		# Put images of primary camera into separate topic.. 
		# new_demonstration['images'] = demonstration['images']['cam{0}'.format(demonstration['primary_camera'])]

		return new_demonstration

	def set_ground_tag_pose_dict(self):
		
		x_rot = R.from_euler('x', 180, degrees=True).as_matrix()

		orientation_1 = np.array([ 0.21265824,  0.48707215,  0.66379473, -0.52622595])  # original --- needs to be rotated
		intermdiary_matrix_1 = np.matmul(R.from_quat(orientation_1).as_matrix(), x_rot)
		orientation_1_rotated = R.from_matrix(intermdiary_matrix_1).as_quat()
	
		orientation_2 = np.array([-0.43014719, -0.27647026, -0.47830252,  0.71397779])	# original --- needs to be rotated
		intermdiary_matrix_2 = np.matmul(R.from_quat(orientation_2).as_matrix(), x_rot)
		orientation_2_rotated = R.from_matrix(intermdiary_matrix_2).as_quat()

		self.ground_pose_dict = {}
		self.ground_pose_dict['0'] = {}
		self.ground_pose_dict['1'] = {}
		
		# Manually set pose, so we don't have to deal with arbitrary flips in the data. 
		self.ground_pose_dict['0']['position'] = np.array([0.13440787, 0.18353586, 0.64308258])
		self.ground_pose_dict['0']['orientation'] = orientation_1_rotated

		self.ground_pose_dict['1']['position'] = np.array([-0.05987254,  0.20209948,  0.66636588])
		self.ground_pose_dict['1']['orientation'] = orientation_2_rotated	

	def set_ground_tag_pose(self, primary_camera):

		pose_dictionary = {}
		pos = self.ground_pose_dict[str(primary_camera)]['position']
		orient = self.ground_pose_dict[str(primary_camera)]['orientation']

		pose_dictionary['position'] = np.repeat(pos[np.newaxis, :], self.demo_length, axis=0)
		pose_dictionary['orientation'] = np.repeat(orient[np.newaxis, :], self.demo_length, axis=0)
		
		return pose_dictionary

	def process_demonstration(self, demonstration, task_index, task_name):

		##########################################
		# Structure of data. 
		##########################################		
		# Assumes demonstration is a tuple, with name of task as first element and data as second element. 
		# data is a dictionary, with 10 Keys:
		# 
		# {'frame_id', 'raw_keypoints', 'avg_keypoints_score', 'tag_detections_in_cam', 'valid_keypoint_frames',
		# 	'raw_hand_orientation', 'primary_keypoint_camera', 'tag_detections_in_cam','primary_camera', 'images'}
		# 	
		# 		data['frame_id'] is int[], with size=t timesteps 
		# 		data['raw_keypoints'] is dict:
		# 				{'cam0', 'cam1'} #
		# 						data['raw_keypoints']['cam0'] -> 21x3matrices[]  ---> 21 keypoints' positions wrt camera
		# 						data['raw_keypoints']['cam1'] -> 21x3matrices[] 			
		# 		data['avg_keypoint_score'] is dict:
		# 				{'cam0', 'cam1'} -> each element is double[], size=t timesteps
		#		data['valid_keypoint_frames'] is dict:
		#				{'cam0', 'cam1'} -> each element is int[] -> (0 or 1), size=timesteps  --- Checked if states were all [0,0,0] or not
		#		data['raw_hand_orientation'] is dict:
		#				{'cam0', 'cam1'} -> each element is quaternion[] -> [x, y, z, w=0], size=timesteps
		#		data['primary_keypoint_camera'] is int
		# 		data['tag_detections_in_cam'] is dict:
		# 				{'cam0', 'cam1'}:
		# 						data['tag_detections_in_cam']['cam#'] -> [ [], [], [], .... ]:
		# 							each element is list of n dictionaries (n = number of visible tags): [ {}, {}, {} ]
		# 								each dictionary:
		# 									{ 'tag_id', 'position', 'orientation', 'pose_err' } ---> positions and orientations wrt camera
		#		data['primary_camera'] is int
		#		data['images'] is dict:
		#				{'cam0', 'cam1'}:
		#						data['images']['cam#'] is list, with size=t timesteps
		#
		# After processing:
		#		data['keypoints'] -> keypoints in gnd frame
		#		data['keypoints_validity'] -> keypoint validity list after fusion of keypoints from the 2 cameras
		#		data['tag_detections'] -> tag poses in gnd frame, with a new structure defined below
		##########################################


        ##################
        # Expected objects and corresponding tag ids-
        # Box - 1,2
        # Drawer - 1,2
        # White Bowl - 2
		# Small yellow cup - 1
		# Box-inside tag - 2
        # Stirrer - 3 
        # Red Cup - 6
        ##################
		# expected_tags = {'Pouring':[0,2,6], 'Stirring':[0,3,6], 'BoxOpening':[0,1,2], 'DrawerOpening':[0,1,2], 'PickPlace':[0,1,2]}


		##########################################
		# Things to do within this function. 
		##########################################
		
		# 0) Preprocess tag pose data to convert into a new structure (and manually editing data from BoxOpening and PickPlace)
		# 1) Detect no_hand poses
		# 2) Rotate tags by 180 degree around x, to get them in same frame as ROS Apriltag
		# 3) Convert keypoints and tags to ground frame
		# 4) Fuse camera for keypoints   
		# 5) Select primary camera, by given criteria
		# 6) For primary camera, retrieve and interpolate tag poses for that camera. 
		# 7) Collate all the relevant data streams, remove irrelevant data streams.
		# 8) Downsample all relevant data streams. 		
		# 9) Return new demo file. 

		self.demo_length = len(demonstration['raw_keypoints']['cam0'])

		###################
		# Specifying keypoints that are considered from each frame
		###################

		take_keypoints = [0, 1, 2, 3, 5, 6, 7]
		demonstration['raw_keypoints']['cam0'] = np.array(demonstration['raw_keypoints']['cam0'])
		demonstration['raw_keypoints']['cam0'] = demonstration['raw_keypoints']['cam0'][:, take_keypoints, :]

		demonstration['raw_keypoints']['cam1'] = np.array(demonstration['raw_keypoints']['cam1'])
		demonstration['raw_keypoints']['cam1'] = demonstration['raw_keypoints']['cam1'][:, take_keypoints, :]
		self.number_of_keypoints = len(take_keypoints)

		#############
		# 0) Preprocess tag poses to convert the data structure into -
		# data['tag_detections_in_cam']['cam#']
		# 		{'tag0', 'tag1', 'tag2' ... }
		# 				'tag#' = {'position[]', 'orientation[]', 'tag_validity'} 
		#
		# Adds key - ['tag_validity']
		#############
		
		print('#################')
		print('Step 0')
		print('#################')
		for idx, cam_tag_detections in enumerate(demonstration['tag_detections_in_cam'].values()):
			demonstration['tag_detections_in_cam']['cam{}'.format(idx)] = self.tag_preprocessing(cam_tag_detections, self.task_list[task_index])

		if task_name == 'PickPlace':
		# Copying last valid pose of transported object as the constant Box Pose.
		# As we failed to capture the box pose during data capture.
			self.pickplace_manual_copying(demonstration)

		print('Preprocessing of tag data completed')
	

		#############
		# 1) Mark pose detections near to "No-Hand Pose" as invalid
		# If no hand is visible, the detector gives the detection of the hand at a specific location 
		#############

		print('#################')
		print('Step 1')
		print('#################')

		self.no_hand_pose = {
					   		'cam0':np.array([0.43, 0.23, 0.66]),
							'cam1':{'pose0.0':np.array([0.98, 0.56, 1.49]),
									'pose0.1':np.array([-0.30, 0.23, 0.66])}
							}
		
		self.no_hand_check(demonstration)
					
		print('No. of valid keypoints after removing no_hand detection - ')
		print('cam0- ', np.count_nonzero(np.array(demonstration['valid_keypoint_frames']['cam0'])==1))		
		print('cam1- ', np.count_nonzero(np.array(demonstration['valid_keypoint_frames']['cam1'])==1))		
		print('Percent of valid frames from cam0 after no_hand = ', len(np.where(np.array(demonstration['valid_keypoint_frames']['cam0'])==1)[0])/self.demo_length)
		print('Percent of valid frames from cam1 after no_hand = ', len(np.where(np.array(demonstration['valid_keypoint_frames']['cam1'])==1)[0])/self.demo_length)

		#############
		# 2) Rotate tags by 180 degrees around x to get them in same frame as ROS Apriltags
		#############

		print('#################')
		print('Step 2')
		print('#################')

		self.rotate_by_180_along_x(demonstration)

		#############
		# 3) Convert keypoints and tags to ground frame
		#############

		print('#################')
		print('Step 3')
		print('#################')

		demonstration['ground_cam_frame_pose'] = self.ground_pose_dict   #self.set_ground_tag_pose( length=demo_length, primary_camera=demonstration['primary_camera'] )

		# 'keypoints', 'hand_orientation', and 'tag_detections' are saved in ground frame
		demonstration['keypoints'] = {'cam0':[], 'cam1':[]}
		demonstration['hand_orientation'] = {'cam0':[], 'cam1':[]}
		demonstration['tag_detections'] = { 'cam0':{}, 'cam1':{} }

		hand_orientation_validity = {'cam0':[], 'cam1':[]}

		self.convert_data_to_ground_frame(demonstration, hand_orientation_validity)


		#############
		# 4) Fuse data from 2 cameras    
		#############

		print('#################')
		print('Step 4')
		print('#################')

		# Keypoint data fusion
		self.fuse_keypoint_data(demonstration)

		# Tag data fusion
		self.fuse_tag_data(demonstration)


		#############
		# 5) Interpolate keypoints, when they don't maintain normal distance between each other (ie normal distance between finger joints of a person)
		#############		
		
		print('#################')
		print('Step 5')
		print('#################')

		demonstration['keypoints'] = self.interpolate_keypoint(demonstration['keypoints'], demonstration['keypoints_validity'])

		demonstration['hand_orientation'] = self.interpolate_hand_orientation(demonstration['hand_orientation']['cam{}'.format(demonstration['primary_keypoint_camera'])],
																		 hand_orientation_validity['cam{}'.format(demonstration['primary_keypoint_camera'])])

		#############
		# 6) For primary camera, retrieve tag poses. 
		#############
		
		print('#################')
		print('Step 6')
		print('#################')

		# Now, instead of interpolating the ground tag detection from the camera frame, set it to constant value. 
		# demonstration['ground_cam_frame_pose'] = self.interpolate_pose( demonstration['tag0']['cam{0}'.format(demonstration['primary_camera'])] )				

		demonstration['ground_cam_frame_pose'] = self.set_ground_tag_pose(demonstration['primary_camera'] )

		demonstration['object_gnd_frame_pose'] = {}

		for tag in demonstration['tag_detections']['cam{0}'.format(demonstration['primary_camera'])]:
			if tag != 'tag0':
				if demonstration['object_gnd_frame_pose'].get('{}_gnd_frame_pose'.format(tag)) == None:
					demonstration['object_gnd_frame_pose']['{}_gnd_frame_pose'.format(tag)] = self.interpolate_pose( demonstration['tag_detections']['cam{0}'.format(demonstration['primary_camera'])][tag] )

		# #############
		# # NOT COMPUTED -- Compute relative poses.
		# #############
		# 
		# demonstration = self.compute_relative_poses(demonstration=demonstration)

		#############
		# 7) Stack relevant data that we care about. 
		#############

		print('#################')
		print('Step 7')
		print('#################')

		demonstration = self.collate_states(demonstration=demonstration)
		
		#############
		# 8) Downsample.
		#############
		
		print('#################')
		print('Step 8')
		print('#################')

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
			print("Processing task: ", task_index+1, " of ", self.number_tasks)

			# Set file path for this task.
			#alt_task_file_path = os.path.join(self.dataset_directory, self.task_list[task_index], 'NumpyDemos')
			task_file_path = os.path.join(self.dataset_directory, self.task_list[task_index])			
			print(task_file_path)
			#########################	
			# For every demo in this task
			#########################

			#for j in range(self.num_demos[task_index]):			
			for j, file in enumerate(sorted(glob.glob(os.path.join(task_file_path,'*.npy')))):
			
				print("####################")
				print("Processing demo: ", j+1, " of ", self.num_demos[task_index], " from task ", task_index+1)
				
				print('file name = ',file)
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

				processed_demonstration = self.process_demonstration(demonstration, task_index, task_name)

				self.task_demo_array.append(processed_demonstration)


			# For each task, save task_file_list to One numpy. 
			suffix = ""
			# Simply saving everything with Images by default

			#if self.args.images_in_real_world_dataset:
			#	suffix = "_wSingleImages"
			task_numpy_path = os.path.join(self.dataset_directory, self.task_list[task_index], "New_Task_Demo_Array{}.npy".format(suffix))
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

		if resample_length<=1 or data_element['hand-state'].shape[0]<=1:
			data_element['is_valid'] = False			
		else:
			data_element['is_valid'] = True


			
			if self.args.smoothen:
				data_element['demo'] = gaussian_filter1d(data_element['demo'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['hand-state'] = gaussian_filter1d(data_element['hand-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['object-state'] = gaussian_filter1d(data_element['object-state'],self.kernel_bandwidth,axis=0,mode='nearest')
				data_element['flat-state'] = gaussian_filter1d(data_element['flat-state'],self.kernel_bandwidth,axis=0,mode='nearest')


		return data_element
	
	####################################
	#### Needs to be edited --- self.state_size
	###################################

	def compute_statistics(self):

		######
		# how to handle this ?
		self.state_size = 53
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

		np.save("RealWorldHumanRigid_Mean.npy", mean)
		np.save("RealWorldHumanRigid_Var.npy", variance)
		np.save("RealWorldHumanRigid_Min.npy", min_value)
		np.save("RealWorldHumanRigid_Max.npy", max_value)
		np.save("RealWorldHumanRigid_Vel_Mean.npy", vel_mean)
		np.save("RealWorldHumanRigid_Vel_Var.npy", vel_variance)
		np.save("RealWorldHumanRigid_Vel_Min.npy", vel_min_value)
		np.save("RealWorldHumanRigid_Vel_Max.npy", vel_max_value)


if __name__=='__main__':

	rwhr_ds = RealWorldHumanRigid_PreDataset()
