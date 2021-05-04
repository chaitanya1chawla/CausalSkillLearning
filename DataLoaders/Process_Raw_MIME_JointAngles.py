import subprocess, os, glob, numpy as np
from IPython import embed
import itertools

basepath = '/data1/tanmayshankar/MIME_FullDataset'
os.chdir(basepath)

# Utility funcs
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

joint_names = None 
downsample_frequency = 20
# Set joint names. 
joint_names = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
	

# # Per task, concatenate trajectories and grippers. 
# for i in range(20):
# # for i in range(2):

# 	print("Processing Task", i)
# 	os.chdir(os.path.join(basepath,"Task_{0}".format(i)))

# 	task_dataelement_list = []
# 	# Get file list. 
# 	filelist = sorted(glob.glob("*/*.txt"))

# 	# Now for every element in the filelist, load, process. 
# 	for j in range(0,len(filelist),3):
# 	# for j in range(0,6,3):

# 		data_element = {}
# 		data_element['demonstration_date'] = os.path.split(filelist[j])[0]
# 		data_element['task_id'] = i 

# 		print("Processing Task:", i,"of:",20,", Trajectory:",j//3,"of:",len(filelist)//3,"Key:",data_element['demonstration_date'])


# 		# j is joint_angles file. 
# 		# j+1 is left_gripper. 
# 		# j+2 is right_gripper. 

# 		# First process joint angle trajectory.
# 		joint_angle_trajectory = []

# 		# Now read the file and iterate through lines. 
# 		with open(filelist[j], 'r') as file:

# 			lines = file.readlines()
# 			# For every timestep.			
# 			for line in lines:
# 				dict_element = eval(line.rstrip('\n'))
# 				if isinstance(dict_element, float):
# 					embed()
# 				# some files have extra lines with gripper keys e.g. MIME_jointangles/4/12405Nov19/joint_angles.txt
# 				if len(dict_element.keys())==17:
					
# 					array_element = np.array([dict_element[joint] for joint in joint_names])
# 					joint_angle_trajectory.append(array_element)					

# 		# Arrayify it.
# 		joint_angle_trajectory = np.array(joint_angle_trajectory)

# 		# Now process grippers. 
# 		gripper_half = 50
# 		gripper_half_max = 50
# 		# Normalize to -1 to 1.
# 		left_gripper = (np.loadtxt(filelist[j+1])-gripper_half)/gripper_half_max
# 		right_gripper = (np.loadtxt(filelist[j+2])-gripper_half)/gripper_half_max

# 		# Now resample everything to common length.
# 		desired_trajectory_length = len(joint_angle_trajectory) // downsample_frequency
			
# 		joint_angle_trajectory = resample(joint_angle_trajectory, desired_trajectory_length)
# 		left_gripper = resample(left_gripper, desired_trajectory_length).reshape(-1,1)
# 		right_gripper = resample(right_gripper, desired_trajectory_length).reshape(-1,1)

# 		# Now create dictionary element. 
# 		data_element['demo'] = np.concatenate([joint_angle_trajectory,left_gripper,right_gripper],axis=1)
# 		data_element['is_valid'] = int(np.linalg.norm(np.diff(joint_angle_trajectory,axis=0),axis=1).max() < 1.0)

# 		# Now append to a task level list.
# 		task_dataelement_list.append(data_element)
		
		
# 	# If we are done with a particular task, save this tasks' dataelement_list as an array. 
# 	np.save("Task{0}_DataList.npy".format(i),np.array(task_dataelement_list))

# 	# Also append this task_dataelement_list to the global_data_element_list
# 	# global_data_element_list.append(task_dataelement_list)

# Save
os.chdir(basepath)
global_data_element_list = []

for i in range(20):
	task_de_list = np.load("Task_{0}/Task{0}_DataList.npy".format(i),allow_pickle=True)
	global_data_element_list.append(task_de_list)

np.save("GlobalDataElementListofLists.npy",np.array(global_data_element_list))

# Merge list of lists
np.save("GlobalDataElementArray.npy",np.array(list(itertools.chain.from_iterable(global_data_element_list))))