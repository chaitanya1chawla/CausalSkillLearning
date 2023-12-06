# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from locale import normalize
from os import environ
from headers import *
from PolicyNetworks import *
from RL_headers import *
from PPO_Utilities import PPOBuffer
from Visualizers import BaxterVisualizer, SawyerVisualizer, FrankaVisualizer, ToyDataVisualizer, \
	GRABVisualizer, GRABHandVisualizer, GRABArmHandVisualizer, DAPGVisualizer, \
	RoboturkObjectVisualizer, RoboturkRobotObjectVisualizer,\
	RoboMimicObjectVisualizer, RoboMimicRobotObjectVisualizer, DexMVVisualizer, \
	FrankaKitchenVisualizer, FetchMOMARTVisualizer, DatasetImageVisualizer
	# MocapVisualizer 

# from Visualizers import *
# import TFLogger, DMP, RLUtils
import DMP, RLUtils

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_printoptions(sci_mode=False, precision=2)

# Global data list
global global_dataset_list 
global_dataset_list = ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk', \
			'Mocap','OrigRoboMimic','RoboMimic','GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', \
	  		'GRABObject', 'DAPG', 'DAPGHand', 'DAPGObject', 'DexMV', 'DexMVHand', 'DexMVObject', \
			'RoboturkObjects','RoboturkRobotObjects','RoboMimicObjects','RoboMimicRobotObjects', \
			'RoboturkMultiObjets', 'RoboturkRobotMultiObjects', \
			'MOMARTPreproc', 'MOMART', 'MOMARTObject', 'MOMARTRobotObject', 'MOMARTRobotObjectFlat', \
			'FrankaKitchenPreproc', 'FrankaKitchen', 'FrankaKitchenObject', 'FrankaKitchenRobotObject', \
			'RealWorldRigid', 'RealWorldRigidRobot', 'RealWorldRigidJEEF', 'NDAX', 'NDAXMotorAngles']

class PolicyManager_BaseClass():

	def __init__(self):
		super(PolicyManager_BaseClass, self).__init__()

	def setup(self):

		print("RUNNING SETUP OF: ", self)

		# Fixing seeds.
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)
		np.set_printoptions(suppress=True,precision=2)

		self.create_networks()
		self.create_training_ops()
		# self.create_util_ops()
		# self.initialize_gt_subpolicies()

		if self.args.setting=='imitation':
			extent = self.dataset.get_number_task_demos(self.demo_task_index)
		if (self.args.setting=='transfer' and isinstance(self, PolicyManager_Transfer)) or \
			(self.args.setting=='cycle_transfer' and isinstance(self, PolicyManager_CycleConsistencyTransfer)) or \
			(self.args.setting=='fixembed' and isinstance(self, PolicyManager_FixEmbedCycleConTransfer)) or \
			(self.args.setting=='jointtransfer' and isinstance(self, PolicyManager_JointTransfer)) or \
			(self.args.setting=='jointfixembed' and isinstance(self, PolicyManager_JointFixEmbedTransfer)) or \
			(self.args.setting=='jointcycletransfer' and isinstance(self, PolicyManager_JointCycleTransfer)) or \
			(self.args.setting=='jointfixcycle' and isinstance(self, PolicyManager_JointFixEmbedCycleTransfer)) or \
			(self.args.setting=='densityjointtransfer' and isinstance(self, PolicyManager_DensityJointTransfer)) or \
			(self.args.setting=='densityjointfixembedtransfer' and isinstance(self, PolicyManager_DensityJointFixEmbedTransfer)) or \
			(self.args.setting=='iktrainer' and isinstance(self, PolicyManager_IKTrainer)) or \
			(self.args.setting=='downstreamtasktransfer' and isinstance(self, PolicyManager_DownstreamTaskTransfer)):
				extent = self.extent
		else:
			extent = len(self.dataset)-self.test_set_size

		self.index_list = np.arange(0,extent)
		self.initialize_plots()

		# if self.args.setting in ['transfer','cycle_transfer','fixembed','jointtransfer','jointcycletransfer']:
		if self.args.setting in ['jointtransfer'] and isinstance(self, PolicyManager_JointTransfer) or \
			self.args.setting in ['jointfixembed'] and isinstance(self, PolicyManager_JointFixEmbedTransfer) or \
			self.args.setting in ['jointcycletransfer'] and isinstance(self, PolicyManager_JointCycleTransfer) or \
			self.args.setting in ['fixembed'] and isinstance(self, PolicyManager_FixEmbedCycleConTransfer) or \
			self.args.setting in ['jointfixcycle'] and isinstance(self, PolicyManager_JointFixEmbedCycleTransfer) or \
			self.args.setting in ['densityjointtransfer'] and isinstance(self, PolicyManager_DensityJointTransfer) or \
			self.args.setting in ['densityjointfixembedtransfer'] and isinstance(self, PolicyManager_DensityJointFixEmbedTransfer) or \
			self.args.setting in ['downstreamtasktransfer'] and isinstance(self, PolicyManager_DownstreamTaskTransfer):
			self.load_domain_models()

	def initialize_plots(self):
		if self.args.name is not None:
			logdir = os.path.join(self.args.logdir, self.args.name)
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			logdir = os.path.join(logdir, "logs")
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)

		if self.args.data in ['MIME','OldMIME'] and not(self.args.no_mujoco):
			self.visualizer = BaxterVisualizer(args=self.args)
			# self.state_dim = 16
		
		elif (self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']) and not(self.args.no_mujoco):			
			self.visualizer = SawyerVisualizer()
		elif (self.args.data in ['OrigRoboMimic','RoboMimic']) and not(self.args.no_mujoco):			
			self.visualizer = FrankaVisualizer()

		elif self.args.data=='Mocap':
			self.visualizer = MocapVisualizer(args=self.args)
		elif self.args.data in ['GRAB']:
			self.visualizer = GRABVisualizer()
		elif self.args.data in ['GRABHand']:
			self.visualizer = GRABHandVisualizer(args=self.args)
		elif self.args.data in ['GRABArmHand', 'GRABArmHandObject', 'GRABObject']:
			self.visualizer = GRABArmHandVisualizer(args=self.args)		
		elif self.args.data in ['DAPG', 'DAPGHand', 'DAPGObject']:
			self.visualizer = DAPGVisualizer(args=self.args)
		elif self.args.data in ['DexMV', 'DexMVHand', 'DexMVObject']:
			self.visualizer = DexMVVisualizer(args=self.args)
		elif self.args.data in ['RoboturkObjects']:		
			self.visualizer = RoboturkObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboturkRobotObjects']:
			self.visualizer = RoboturkRobotObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicObjects']:
			self.visualizer = RoboMimicObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicRobotObjects']:
			self.visualizer = RoboMimicRobotObjectVisualizer(args=self.args)
		elif self.args.data in ['FrankaKitchenRobotObject']:
			self.visualizer = FrankaKitchenVisualizer(args=self.args)
		elif self.args.data in ['MOMARTRobotObject', 'MOMARTRobotObjectFlat']:			
			if not hasattr(self, 'visualizer'):
				self.visualizer = FetchMOMARTVisualizer(args=self.args)
		elif self.args.data in ['RealWorldRigid', 'NDAX', 'NDAXMotorAngles']:
			self.visualizer = DatasetImageVisualizer(args=self.args)
		else:
			self.visualizer = ToyDataVisualizer()
		

		self.rollout_gif_list = []
		self.gt_gif_list = []

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def collect_inputs(self, i, get_latents=False, special_indices=None, called_from_train=False):	

		if self.args.data=='DeterGoal':
			
			if special_indices is not None:
				i = special_indices

			sample_traj, sample_action_seq = self.dataset[i]
			latent_b_seq, latent_z_seq = self.dataset.get_latent_variables(i)

			start = 0

			if self.args.traj_length>0:
				sample_action_seq = sample_action_seq[start:self.args.traj_length-1]
				latent_b_seq = latent_b_seq[start:self.args.traj_length-1]
				latent_z_seq = latent_z_seq[start:self.args.traj_length-1]
				sample_traj = sample_traj[start:self.args.traj_length]	
			else:
				# Traj length is going to be -1 here. 
				# Don't need to modify action sequence because it does have to be one step less than traj_length anyway.
				sample_action_seq = sample_action_seq[start:]
				sample_traj = sample_traj[start:]
				latent_b_seq = latent_b_seq[start:]
				latent_z_seq = latent_z_seq[start:]

			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 		
			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			# concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)
			# old_concatenated_traj = self.old_concat_state_action(sample_traj, sample_action_seq)
			
			# If the collect inputs function is being called from the train function, 
			# Then we should corrupt the inputs based on how much the input_corruption_noise is set to. 
			# If it's 0., then no corruption. 
			corrupted_sample_action_seq = self.corrupt_inputs(sample_action_seq)
			corrupted_sample_traj = self.corrupt_inputs(sample_traj)

			concatenated_traj = self.concat_state_action(corrupted_sample_traj, corrupted_sample_action_seq)		
			old_concatenated_traj = self.old_concat_state_action(corrupted_sample_traj, corrupted_sample_action_seq)
		
			if self.args.data=='DeterGoal':
				self.conditional_information = np.zeros((self.args.condition_size))
				self.conditional_information[self.dataset.get_goal(i)] = 1
				self.conditional_information[4:] = self.dataset.get_goal_position[i]
			else:
				self.conditional_information = np.zeros((self.args.condition_size))

			if get_latents:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj, latent_b_seq, latent_z_seq
			else:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj

		elif self.args.data in global_data_list:

			# If we're imitating... select demonstrations from the particular task.
			if self.args.setting=='imitation' and \
				 (self.args.data in ['Roboturk','RoboMimic','RoboturkObjects','RoboturkRobotObjects',\
					'RoboMimicObjects','RoboMimicRobotObjects']):
				data_element = self.dataset.get_task_demo(self.demo_task_index, i)
			else:
				data_element = self.dataset[i]

			if not(data_element['is_valid']):
				return None, None, None, None							

			trajectory = data_element['demo']

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
				trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

			action_sequence = np.diff(trajectory,axis=0)

			self.current_traj_len = len(trajectory)

			if self.args.data in ['MIME','OldMIME','GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', 'GRABObject', 'DAPG', 'DAPGHand', 'DAPGObject', 'DexMV', 'DexMVHand', 'DexMVObject', 'RealWorldRigid']:
				self.conditional_information = np.zeros((self.conditional_info_size))				
			# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic',\
				'RoboMimic','RoboturkObjects','RoboturkRobotObjects', 'RoboMimicObjects', 'RoboMimicRobotObjects']:
				robot_states = data_element['robot-state']
				object_states = data_element['object-state']
				self.current_task_for_viz = data_element['task-id']

				self.conditional_information = np.zeros((self.conditional_info_size))
				# Don't set this if pretraining / baseline.
				if self.args.setting=='learntsub' or self.args.setting=='imitation':
					self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
					self.conditional_information[:,:self.cond_robot_state_size] = robot_states
					# Doing this instead of self.cond_robot_state_size: because the object_states size varies across demonstrations.
					self.conditional_information[:,self.cond_robot_state_size:self.cond_robot_state_size+object_states.shape[-1]] = object_states	
					# Setting task ID too.		
					self.conditional_information[:,-self.number_tasks+data_element['task-id']] = 1.

			# If the collect inputs function is being called from the train function, 
			# Then we should corrupt the inputs based on how much the input_corruption_noise is set to. 
			# If it's 0., then no corruption. 
			corrupted_action_sequence = self.corrupt_inputs(action_sequence)
			corrupted_trajectory = self.corrupt_inputs(trajectory)

			concatenated_traj = self.concat_state_action(corrupted_trajectory, corrupted_action_sequence)		
			old_concatenated_traj = self.old_concat_state_action(corrupted_trajectory, corrupted_action_sequence)

			# # Concatenate
			# concatenated_traj = self.concat_state_action(trajectory, action_sequence)
			# old_concatenated_traj = self.old_concat_state_action(trajectory, action_sequence)

			if self.args.setting=='imitation':
				action_sequence = RLUtils.resample(data_element['demonstrated_actions'],len(trajectory))
				concatenated_traj = np.concatenate([trajectory, action_sequence],axis=1)

			return trajectory, action_sequence, concatenated_traj, old_concatenated_traj, data_element

	def set_extents(self):

		##########################
		# Set extent.
		##########################

		# Modifying to make training functions handle batches. 
		# For every item in the epoch:
		if self.args.setting=='imitation':
			extent = self.dataset.get_number_task_demos(self.demo_task_index)
		# if self.args.setting=='transfer' or self.args.setting=='cycle_transfer' or self.args.setting=='fixembed' or self.args.setting=='jointtransfer':
		if self.args.setting in ['transfer','cycle_transfer','fixembed','jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer','iktrainer']:
			if self.args.debugging_datapoints>-1:
				extent = self.args.debugging_datapoints
				self.extent = self.args.debugging_datapoints
			else:
				extent = self.extent
		else:
			if self.args.debugging_datapoints>-1:				
				extent = self.args.debugging_datapoints
			else:
				extent = len(self.dataset)-self.test_set_size

		if self.args.task_discriminability or self.args.task_based_supervision:
			extent = self.extent	

		return extent
	
	def train(self, model=None):

		print("Running Main Train Function.")

		########################################
		# (1) Load Model If Necessary
		########################################
		if model:
			print("Loading model in training.")
			self.load_all_models(model)			
		
		########################################
		# (2) Set initial values.
		########################################

		counter = self.args.initial_counter_value
		epoch_time = 0.
		cum_epoch_time = 0.		
		self.epoch_coverage = np.zeros(len(self.dataset))

		########################################
		# (3) Outer loop over epochs. 
		########################################
		
		# For number of training epochs. 
		for e in range(self.number_epochs+1): 
					
			########################################
			# (4a) Bookkeeping
			########################################

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			self.current_epoch_running = e
			print("Starting Epoch: ",e)

			########################################
			# (4b) Set extent of dataset. 
			########################################

			# Modifying to make training functions handle batches. 
			extent = self.set_extents()

			########################################
			# (4c) Shuffle based on extent of dataset. 
			########################################						

			# np.random.shuffle(self.index_list)
			self.shuffle(extent)
			self.batch_indices_sizes = []

			########################################
			# (4d) Inner training loop
			########################################

			t1 = time.time()
			self.coverage = np.zeros(len(self.dataset))

			# For all data points in the dataset. 
			for i in range(0,self.training_extent,self.args.batch_size):				
			# for i in range(0,extent-self.args.batch_size,self.args.batch_size):
				# print("RUN TRAIN", i)
				# Probably need to make run iteration handle batch of current index plus batch size.				
				# with torch.autograd.set_detect_anomaly(True):
				t2 = time.time()

				##############################################
				# (5) Run Iteration
				##############################################

				# print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(self.index_list[i]).zfill(5),"Extent:",extent)
				profile_iteration = 0 				
				if profile_iteration:
					self.lp = LineProfiler()
					self.lp_wrapper = self.lp(self.run_iteration)
					# self.lp_wrapper(counter, self.index_list[i])
					self.lp_wrapper(counter, i)
					self.lp.print_stats()			
				else:													
					# self.run_iteration(counter, self.index_list[i])
					self.run_iteration(counter, i)

				t3 = time.time()
				# print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(self.index_list[i]).zfill(5), "Iter Time:",format(t3-t2,".4f"),"PerET:",format(cum_epoch_time/max(e,1),".4f"),"CumET:",format(cum_epoch_time,".4f"),"Extent:",extent)
				print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(i).zfill(5), "Iter Time:",format(t3-t2,".4f"),"PerET:",format(cum_epoch_time/max(e,1),".4f"),"CumET:",format(cum_epoch_time,".4f"),"Extent:",extent)

				counter = counter+1
				
			##############################################
			# (6) Some more book keeping.
			##############################################
				
			t4 = time.time()
			epoch_time = t4-t1
			cum_epoch_time += epoch_time

			##############################################
			# (7) Automatic evaluation if we need it. 
			##############################################
				
			# if e%self.args.eval_freq==0:
			# 	self.automatic_evaluation(e)

			##############################################
			# (8) Debug
			##############################################
						
			self.epoch_coverage += self.coverage
			# if e%100==0:
			# 	print("Debugging dataset coverage")
			# 	embed()

	def automatic_evaluation(self, e):

		# Writing new automatic evaluation that parses arguments and creates an identical command loading the appropriate model. 
		# Note: If the initial command loads a model, ignore that. 

		command_args = self.args._get_kwargs()			
		base_command = 'python Master.py --train=0 --model={0} --batch_size=1'.format("Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e))

		if self.args.data=='Mocap':
			base_command = './xvfb-run-safe ' + base_command

		# For every argument in the command arguments, add it to the base command with the value used, unless it's train or model. 
		for ar in command_args:
			# Skip model and train, because we need to set these manually.
			if ar[0]=='model' or ar[0]=='train' or ar[0]=='batch_size':
				pass
			# Add the rest
			else:				
				base_command = base_command + ' --{0}={1}'.format(ar[0],ar[1])		
		#  cluster_command = 'python cluster_run.py --partition=learnfair --name={0}_Eval --cmd=\'{1}\''.format(self.args.name, base_command)				

		# NOT RUNNING AUTO EVAL FOR NOW.
		# subprocess.Popen([base_command],shell=True)

	def set_visualizer_object(self):

		#####################################################
		# Set visualizer object. 
		#####################################################
		if self.args.data in ['MIME','OldMIME']:
			self.visualizer = BaxterVisualizer(args=self.args)
			# self.state_dim = 16
		elif (self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']) and not(self.args.no_mujoco):			
			self.visualizer = SawyerVisualizer()
		elif (self.args.data in ['OrigRoboMimic','RoboMimic']) and not(self.args.no_mujoco):			
			self.visualizer = FrankaVisualizer()
		elif self.args.data=='Mocap':
			self.visualizer = MocapVisualizer(args=self.args)
			# Because there are just more invalid DP's in Mocap.
			self.N = 100
		elif self.args.data in ['RoboturkObjects']:
			self.visualizer = RoboturkObjectVisualizer(args=self.args)
		elif self.args.data in ['GRABHand']:
			self.visualizer = GRABHandVisualizer(args=self.args)
			self.N = 200
		elif self.args.data in ['GRABArmHand', 'GRABArmHandObject', 'GRABObject']:
			self.visualizer = GRABArmHandVisualizer(args=self.args)
			self.N = 200
		elif self.args.data in ['DAPG', 'DAPGHand', 'DAPGObject']:
			self.visualizer = DAPGVisualizer(args=self.args)		
		elif self.args.data in ['DexMV', 'DexMVHand', 'DexMVObject']:
			self.visualizer = DexMVVisualizer(args=self.args)		
		elif self.args.data in ['RoboturkRobotObjects']:		
			self.visualizer = RoboturkRobotObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicObjects']:
			self.visualizer = RoboMimicObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicRobotObjects']:
			self.visualizer = RoboMimicRobotObjectVisualizer(args=self.args)			
		elif self.args.data in ['FrankaKitchenRobotObject']:
			self.visualizer = FrankaKitchenVisualizer(args=self.args)
		elif self.args.data in ['MOMARTRobotObject', 'MOMARTRobotObjectFlat']:
			if not hasattr(self, 'visualizer'):
				self.visualizer = FetchMOMARTVisualizer(args=self.args)
		elif self.args.data in ['RealWorldRigid', 'NDAX', 'NDAXMotorAngles']:
			self.visualizer = DatasetImageVisualizer(args=self.args)
		else: 
			self.visualizer = ToyDataVisualizer()

	def per_batch_env_management(self, indexed_data_element):
		
		task_id = indexed_data_element['task-id']
		env_name = self.dataset.environment_names[task_id]
		print("Visualizing a trajectory of task:", env_name)

		self.visualizer.create_environment(task_id=env_name)

	def generate_segment_indices(self, batch_latent_b_torch):
		
		self.batch_segment_index_list = []		

		batch_latent_b = batch_latent_b_torch.detach().cpu().numpy()		

		for b in range(self.args.batch_size):
			segments = np.where(batch_latent_b[:self.batch_trajectory_lengths[b],b])[0]

			# Add last index to segments
			segments = np.concatenate([segments, self.batch_trajectory_lengths[b:b+1]])
			self.batch_segment_index_list.append(segments)
			self.global_segment_index_list.append(segments)
		# Need to perform the same manipulation of segment indices that we did in the forward function call.		

	def visualize_robot_data(self, load_sets=False, number_of_trajectories_to_visualize=None):

		
		if number_of_trajectories_to_visualize is not None:
			self.N = number_of_trajectories_to_visualize
		else:

			####################################
			# TEMPORARILY SET N to 10
			####################################
			# self.N = 33
			self.N = self.args.N_trajectories_to_visualize

		self.rollout_timesteps = self.args.traj_length
	
		self.set_visualizer_object()
		np.random.seed(seed=self.args.seed)

		#####################################################
		# Get latent z sets.
		#####################################################
		
		if not(load_sets):

			#####################################################
			# Select Z indices if necessary.
			#####################################################

			if self.args.split_stream_encoder:
				if self.args.embedding_visualization_stream == 'robot':
					stream_z_indices = np.arange(0,int(self.args.z_dimensions/2))
				elif self.args.embedding_visualization_stream == 'env':
					stream_z_indices = np.arange(int(self.args.z_dimensions/2),self.args.z_dimensions)
				else:
					stream_z_indices = np.arange(0,self.args.z_dimensions)	
			else:
				stream_z_indices = np.arange(0,self.args.z_dimensions)

			#####################################################
			# Initialize variables.
			#####################################################

			# self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
			self.latent_z_set = np.zeros((self.N,len(stream_z_indices)))		
			self.queryjoint_latent_z_set = []
			# These are lists because they're variable length individually.
			self.indices = []
			self.trajectory_set = []
			self.trajectory_rollout_set = []		
			self.rollout_gif_list = []
			self.gt_gif_list = []
			self.task_name_set = []

			#####################################################
			# Create folder for gifs.
			#####################################################

			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
			# Create save directory:
			upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

			if not(os.path.isdir(upper_dir_name)):
				os.mkdir(upper_dir_name)

			self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
			if not(os.path.isdir(self.dir_name)):
				os.mkdir(self.dir_name)
			self.traj_dir_name = os.path.join(self.dir_name, "NumpyTrajs")
			if not(os.path.isdir(self.traj_dir_name)):
				os.mkdir(self.traj_dir_name)
			self.z_dir_name = os.path.join(self.dir_name, "NumpyZs")
			if not(os.path.isdir(self.z_dir_name)):
				os.mkdir(self.z_dir_name)


			self.max_len = 0

			#####################################################
			# Initialize variables.
			#####################################################

			self.shuffle(len(self.dataset)-self.test_set_size, shuffle=True)
			
			self.global_segment_index_list = []

			# print("Embedding before the robot visuals loop.s")
			# embed()

			for j in range(self.N//self.args.batch_size):
				
				number_batches_for_dataset = (len(self.dataset)//self.args.batch_size)+1
				i = j % number_batches_for_dataset

				# (1) Encode trajectory. 
				if self.args.setting in ['learntsub','joint', 'queryjoint']:
					
					
					input_dict, var_dict, eval_dict = self.run_iteration(0, j, return_dicts=True, train=False)
					latent_z = var_dict['latent_z_indices']
					sample_trajs = input_dict['sample_traj']
					data_element = input_dict['data_element']
					latent_b = torch.swapaxes(var_dict['latent_b'], 1,0)

					# Generate segment index list..
					self.generate_segment_indices(latent_b)

					# print("Embed to verify segment indices")
					# embed()

				else:
					print("Running iteration of segment in viz, i: ", i, "j:", j)
					latent_z, sample_trajs, _, data_element = self.run_iteration(0, i, return_z=True, and_train=False)
					# latent_z, sample_trajs, _, data_element = self.run_iteration(0, j*self.args.batch_size, return_z=True, and_train=False)

				if self.args.batch_size>1:

					# Set the max length if it's less than this batch of trajectories. 
					if sample_trajs.shape[0]>self.max_len:
						self.max_len = sample_trajs.shape[0]

					#######################
					# Create env for batch.
					if not(self.args.data in ['RealWorldRigid', 'NDAX', 'NDAXMotorAngles']):
						self.per_batch_env_management(data_element[0])

					for b in range(self.args.batch_size):
						
						self.indices.append(j*self.args.batch_size+b)
						print("#########################################")	
						print("Getting visuals for trajectory: ",j*self.args.batch_size+b)
						# print("Getting visuals for trajectory:")
						# print("j:", j, "b:", b, "j*bs+b:", j*self.args.batch_size+b, "il[j*bs+b]:", self.index_list[j*self.args.batch_size+b] "env:", self.dataset[self.index_list[j*self.args.batch_size+b]]['file'])
						# print("j:", j, "b:", b, "j*bs+b:", j*self.args.batch_size+b, "il[j*bs+b]:", self.index_list[j*self.args.batch_size+b])

						if self.args.setting in ['learntsub','joint','queryjoint']:
							self.latent_z_set[j*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())

							# Rollout each individual trajectory in this batch.
							# trajectory_rollout = self.get_robot_visuals(j*self.args.batch_size+b, latent_z[:,b], sample_trajs[:self.batch_trajectory_lengths[b],b], z_seq=True, indexed_data_element=input_dict['data_element'][b])
							trajectory_rollout = self.get_robot_visuals(j*self.args.batch_size+b, latent_z[:self.batch_trajectory_lengths[b],b], \
												sample_trajs[:self.batch_trajectory_lengths[b],b], z_seq=True, indexed_data_element=input_dict['data_element'][b], \
												segment_indices=self.batch_segment_index_list[b])
							
							self.queryjoint_latent_z_set.append(copy.deepcopy(latent_z[:self.batch_trajectory_lengths[b],b].detach().cpu().numpy()))

							# self.queryjoint_latent_b_set.append(copy.deepcopy(latent_b[:self.batch_trajectory_lengths[b],b].detach().cpu().numpy()))
							
							gt_traj = sample_trajs[:self.batch_trajectory_lengths[b],b]
						else:
							# self.latent_z_set[j*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())
							self.latent_z_set[j*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b,stream_z_indices].detach().cpu().numpy())
			
							# Rollout each individual trajectory in this batch.
							trajectory_rollout = self.get_robot_visuals(j*self.args.batch_size+b, latent_z[0,b], sample_trajs[:,b], indexed_data_element=data_element[b])
							gt_traj = sample_trajs[:,b]
							

						# Now append this particular sample traj and the rollout into trajectroy and rollout sets.
						self.trajectory_set.append(copy.deepcopy(gt_traj))
						self.trajectory_rollout_set.append(copy.deepcopy(trajectory_rollout))
						self.task_name_set.append(data_element[b]['environment-name'])
						#######################
						# Save the GT trajectory, the rollout, and Z into numpy files. 

						#####################################################
						# Save trajectories and Zs
						#####################################################

						k = j*self.args.batch_size+b	
						kstr = str(k).zfill(3)

						# print("Before unnorm")
						# embed()
						if self.args.normalization is not None:
							
							gt_traj = (self.trajectory_set[k] *self.norm_denom_value) + self.norm_sub_value
							gt_traj_tuple = (data_element[b]['environment-name'], gt_traj)
							
							# Don't unnormalize, we already did in get robot visuals. 
							rollout_traj = self.trajectory_rollout_set[k]
							rollout_traj_tuple = (data_element[b]['environment-name'], rollout_traj)
							# rollout_traj = (self.trajectory_rollout_set[k]*self.norm_denom_value) + self.norm_sub_value

						# (trajectory_start * self.norm_denom_value ) + self.norm_sub_value
						# np.save(os.path.join(self.traj_dir_name, "GT_Traj{0}.npy".format(k)), gt_traj)
						# np.save(os.path.join(self.traj_dir_name, "Rollout_Traj{0}.npy".format(k)), rollout_traj)
						# np.save(os.path.join(self.z_dir_name, "Latent_Z{0}.npy".format(k)), self.latent_z_set[k])
												
						np.save(os.path.join(self.traj_dir_name, "Traj{0}_GT.npy".format(kstr)), gt_traj_tuple)
						np.save(os.path.join(self.traj_dir_name, "Traj{0}_Rollout.npy".format(kstr)), rollout_traj_tuple)
						np.save(os.path.join(self.z_dir_name, "Traj{0}_Latent_Z.npy".format(kstr)), self.latent_z_set[k])						

				else:

					print("#########################################")	
					print("Getting visuals for trajectory: ",j,i)

					if latent_z is not None:
						self.indices.append(i)

						if len(sample_trajs)>self.max_len:
							self.max_len = len(sample_trajs)
						# Copy z. 
						self.latent_z_set[j] = copy.deepcopy(latent_z.detach().cpu().numpy())

						trajectory_rollout = self.get_robot_visuals(i, latent_z, sample_trajs)								

						self.trajectory_set.append(copy.deepcopy(sample_trajs))
						self.trajectory_rollout_set.append(copy.deepcopy(trajectory_rollout))	

			# Get MIME embedding for rollout and GT trajectories, with same Z embedding. 
			embedded_z = self.get_robot_embedding()

		#####################################################
		# If precomputed sets.
		#####################################################

		else:

			print("Using Precomputed Latent Set and Embedding.")
			# Instead of computing latent sets, just load them from File path. 
			self.load_latent_sets(self.args.latent_set_file_path)
			
			# print("embedding after load")
			# embed()

			# Get embedded z based on what the perplexity is. 			
			embedded_z = self.embedded_zs.item()["perp{0}".format(int(self.args.perplexity))]
			self.max_len = 0
			for i in range(self.N):
				print("Visualizing Trajectory ", i, " of ",self.N)

				# Set the max length if it's less than this batch of trajectories. 
				if self.gt_trajectory_set[i].shape[0]>self.max_len:
					self.max_len = self.gt_trajectory_set[i].shape[0]			

				dummy_task_id_dict = {}
				dummy_task_id_dict['task-id'] = self.task_id_set[i]
				trajectory_rollout = self.get_robot_visuals(i, self.latent_z_set[i], self.gt_trajectory_set[i], indexed_data_element=dummy_task_id_dict)

			self.indices = range(self.N)


		#####################################################
		# Save the embeddings in HTML files.
		#####################################################

		# print("#################################################")
		# print("Embedding in Visualize robot data in Pretrain PM")
		# print("#################################################")
		# embed()

		gt_animation_object = self.visualize_robot_embedding(embedded_z, gt=True)
		rollout_animation_object = self.visualize_robot_embedding(embedded_z, gt=False)
		
		self.task_name_set_array = np.array(self.task_name_set)

		# Save webpage. 
		self.write_results_HTML()
		
		# Save webpage with plots. 
		self.write_results_HTML(plots_or_gif='Plot')
		
		viz_embeddings = True
		if (self.args.data in ['RealWorldRigid', 'RealWorldRigidRobot']) and (self.args.images_in_real_world_dataset==0):
			viz_embeddings = False

		if viz_embeddings:
			self.write_embedding_HTML(gt_animation_object,prefix="GT")
			self.write_embedding_HTML(rollout_animation_object,prefix="Rollout")

	def preprocess_action(self, action=None):

		########################################
		# Numpy-fy and subsample. (It's 1x|S|, that's why we need to index into first dimension.)

		# It is now 1x|S| or Bx|S|? So squeezing should still be okay... 
		########################################
		
		# action_np = action.detach().cpu().numpy()[0,:8]			
		action_np = action.detach().cpu().squeeze(0).numpy()[...,:8]

		########################################
		# Unnormalize action.
		########################################

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			# Remember. actions are normalized just by multiplying denominator, no addition of mean.
			unnormalized_action = (action_np*self.norm_denom_value)
		else:
			unnormalized_action = action_np
		
		########################################
		# Scale action.
		########################################

		scaled_action = unnormalized_action*self.args.sim_viz_action_scale_factor

		########################################
		# Second unnormalization to undo the visualizer environment normalization.... 
		########################################

		ctrl_range = self.visualizer.environment.sim.model.actuator_ctrlrange
		bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
		weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
		# Modify gripper normalization, so that the env normalization actually happens.
		bias = bias[:-1]
		bias[-1] = 0.
		weight = weight[:-1]
		weight[-1] = 1.
		
		# Unnormalized_scaled_action_for_env_step
		if self.visualizer.new_robosuite:
			unnormalized_scaled_action_for_env_step = scaled_action
		else:
			unnormalized_scaled_action_for_env_step = (scaled_action - bias)/weight

		# print("#####################")
		# print("Vanilla A:", action_np)
		# print("Stat Unnorm A: ", unnormalized_action)
		# print("Scaled A: ", scaled_action)
		# print("Env Unnorm A: ", unnormalized_scaled_action_for_env_step)

		return unnormalized_scaled_action_for_env_step

	def compute_next_state(self, current_state=None, action=None):

		####################################
		# If we're stepping in the environment:
		####################################
		
		if self.args.viz_sim_rollout:
			
			####################################
			# Take environment step.
			####################################

			action_to_execute = self.preprocess_action(action)

			########################################
			# Repeat steps for K times.
			########################################
			
			for k in range(self.args.sim_viz_step_repetition):
				# Use environment to take step.
				env_next_state_dict, _, _, _ = self.visualizer.environment.step(action_to_execute)
				gripper_state = env_next_state_dict[self.visualizer.gripper_key]
				if self.visualizer.new_robosuite:
					joint_state = self.visualizer.environment.sim.get_state()[1][:7]
				else:
					joint_state = env_next_state_dict['joint_pos']

			####################################
			# Assemble robot state.
			####################################
			
			gripper_open = np.array([0.0115, -0.0115])
			gripper_closed = np.array([-0.020833, 0.020833])

			# The state that we want is ... joint state?
			gripper_finger_values = gripper_state
			gripper_values = (gripper_finger_values - gripper_open)/(gripper_closed - gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = 2*finger_diff-1

			########################################
			# Concatenate joint and gripper state. 	
			########################################

			robot_state_np = np.concatenate([joint_state, np.array(gripper_value).reshape((1,))])

			########################################
			# Assemble object state.
			########################################

			# Get just the object pose, object quaternion.
			object_state_np = env_next_state_dict['object-state'][:7]

			########################################
			# Assemble next state.
			########################################

			# Parse next state from dictionary, depending on what dataset we're using.

			# If we're using a dataset with both objects and the robot. 
			if self.args.data in ['RoboturkRobotObjects','RoboMimicRobotObjects']:
				next_state_np = np.concatenate([robot_state_np,object_state_np],axis=0)

			# REMEMBER, We're never actually using an only object dataset here, because we can't actually actuate the objects..
			# # If we're using an object only dataset. 
			# elif self.args.data in ['RoboturkObjects']: 
			# 	next_state_np = object_state_np			

			# If we're using a robot only dataset.
			else:
				next_state_np = robot_state_np

			if self.args.normalization in ['meanvar','minmax']:
				next_state_np = (next_state_np - self.norm_sub_value)/self.norm_denom_value

			# Return torchified version of next_state
			next_state = torch.from_numpy(next_state_np).to(device)	


			# print("embedding at gazoo")
			# embed()
			return next_state, env_next_state_dict[self.visualizer.image_key]

		####################################
		# If not using environment to rollout trajectories.
		####################################

		else:			
			# Simply create next state as addition of current state and action.		
			next_state = current_state+action
			# Return - remember this is already a torch tensor now.
			return next_state, None

	def rollout_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None, z_seq=False, original_trajectory=None):

		rendered_rollout_trajectory = []
		
		if self.args.viz_sim_rollout:

			########################################
			# 0) Reset visualizer environment state. 
			########################################

			self.visualizer.environment.reset()

			# Unnormalize the start state. 
			if self.args.normalization in ['minmax','meanvar']:
				unnormalized_trajectory_start = (trajectory_start * self.norm_denom_value ) + self.norm_sub_value
			else:
				unnormalized_trajectory_start = trajectory_start 
			# Now use unnormalized state to set the trajectory state. 
			self.visualizer.set_joint_pose(unnormalized_trajectory_start)		
			
		########################################
		# 1a) Create placeholder policy input tensor. 
		########################################

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).to(device).float()

		if z_seq:
			subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z[0]).to(device).float()	
		else:
			subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()	

		########################################
		# 1b) Set parameters.
		########################################

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = self.rollout_timesteps-1

		########################################
		# 2) Iterate over rollout length:
		########################################

		for t in range(length):
			
			# print("Pause in rollout")
			# embed()
			current_state = subpolicy_inputs[t,:self.state_dim].clone().detach().cpu().numpy()

			########################################
			# 3) Get action from policy. 
			########################################
			
			# Check if we're visualizing the GT trajectory. 
			if self.args.viz_gt_sim_rollout:
				# If we are, then get the action from the original trajectory, not the policy. 

				# Open loop trajectory execution.
				action_to_execute_ol = torch.from_numpy(original_trajectory[t+1]-original_trajectory[t]).cuda()
				# Closed loop 
				action_to_execute_cl = torch.from_numpy(original_trajectory[t+1]-current_state).cuda()

				action_to_execute = action_to_execute_cl

				print("T:", t, " S:", current_state[:8])
				print("A_ol:", action_to_execute_ol[:8].cpu().numpy())
				print("A_cl:", action_to_execute_cl[:8].cpu().numpy())

				
			else:
				# Assume we always query the policy for actions with batch_size 1 here. 
				actions = self.policy_network.get_actions(subpolicy_inputs, greedy=True, batch_size=1)

				# Select last action to execute. 
				action_to_execute = actions[-1].squeeze(1)

				# Downscale the actions by action_scale_factor.
				action_to_execute = action_to_execute/self.args.action_scale_factor

			########################################
			# 4) Compute next state. 
			########################################
			
			# new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute
			new_state, image = self.compute_next_state(subpolicy_inputs[t,:self.state_dim], action_to_execute)
			rendered_rollout_trajectory.append(image)

			########################################
			# 5) Construct new input row.
			########################################

			# New input row. 
			input_row = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
			input_row[0,:self.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute

			if z_seq:
				input_row[0,2*self.state_dim:] = torch.tensor(latent_z[t+1]).to(device).float()
			else:
				input_row[0,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		# print("Embedding in rollout")
		# embed()
		
		trajectory = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		
		return trajectory, rendered_rollout_trajectory

	def retrieve_unnormalized_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None, z_seq=False, original_trajectory=None):

		trajectory, _ = self.rollout_robot_trajectory(trajectory_start, latent_z, rollout_length=rollout_length, z_seq=z_seq, original_trajectory=original_trajectory)

		return self.unnormalize_trajectory(trajectory)

	def unnormalize_trajectory(self, trajectory):
		# Unnormalize. 
		if self.args.normalization is not None:
			unnormalized_trajectory = (trajectory*self.norm_denom_value) + self.norm_sub_value			
		return unnormalized_trajectory

	def partitioned_rollout_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None, z_seq=False, original_trajectory=None, segment_indices=None):

		# If we're running with a sequential factored encoder network, we have pretrain skill policy.
		# This is only trained to rollout individual skills. 
		# Therefore partition the rollout into components that each only run individual skills. 

		# Set initial start state. Overwrite this later. 
		start_state = copy.deepcopy(trajectory_start)
		rollout_trajectory_segment_list = []

		# For each segment, callout rollout robot trajectory. 
		for k in range(len(segment_indices)-1):

			# Start and end indices are start_index = segment_indices[k], end_index = segment_indices[k+1]
			segment_length = segment_indices[k+1] - segment_indices[k]

			# Technically the latent z should be constant across the segment., so just set it to start value. 
			segment_latent_z = latent_z[segment_indices[k]]

			# Rollout. 
			rollout_trajectory_segment, _ = self.rollout_robot_trajectory(start_state, segment_latent_z, rollout_length=segment_length)
			rollout_trajectory_segment_list.append(copy.deepcopy(rollout_trajectory_segment))

			# Set start state. 
			start_state = copy.deepcopy(rollout_trajectory_segment[-1, :self.state_dim])

		# After having rolled out each component, concatenated the trajectories. 
		rollout_fulltrajectory = np.concatenate(rollout_trajectory_segment_list, axis=0)

		return rollout_fulltrajectory, None

	def get_robot_visuals(self, i, latent_z, trajectory, return_image=False, return_numpy=False, z_seq=False, indexed_data_element=None, segment_indices=None):

		########################################
		# 1) Get task ID. 
		########################################
		# Set task ID if the visualizer needs it. 
		# Set task ID if the visualizer needs it. 
		if indexed_data_element is None or ('task-id' not in indexed_data_element.keys()):
			task_id = None
			env_name = None
		else:			
			if self.args.data in ['NDAX', 'NDAXMotorAngles']:
				task_id = indexed_data_element['task_id']
			else:
				task_id = indexed_data_element['task-id']

			# print("EMBED in grv")
			# embed()
			env_name = self.dataset.environment_names[task_id]
			print("Visualizing a trajectory of task:", env_name)

		########################################
		# 2) Feed Z into policy, rollout trajectory.
		########################################
		
		self.visualizer.create_environment(task_id=env_name)

		if self.args.setting in ['queryjoint']:
			trajectory_rollout, rendered_rollout_trajectory = self.partitioned_rollout_robot_trajectory(trajectory[0], latent_z, rollout_length=max(trajectory.shape[0],0), z_seq=z_seq, original_trajectory=trajectory, segment_indices=segment_indices)
		else:
			trajectory_rollout, rendered_rollout_trajectory = self.rollout_robot_trajectory(trajectory[0], latent_z, rollout_length=max(trajectory.shape[0],0), z_seq=z_seq, original_trajectory=trajectory)

		########################################
		# 3) Unnormalize data. 
		########################################

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			unnorm_gt_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			unnorm_pred_trajectory = (trajectory_rollout*self.norm_denom_value) + self.norm_sub_value
		else:
			unnorm_gt_trajectory = trajectory
			unnorm_pred_trajectory = trajectory_rollout

		if self.args.data == 'Mocap':
			# Get animation object from dataset. 
			animation_object = self.dataset[i]['animation']

		print("We are in the PM visualizer function.")

		# Set task ID if the visualizer needs it. 
		# if indexed_data_element is not None and self.args.data == 'DAPG':
		# 	env_name = indexed_data_element['file']
		# 	print("Visualizing trajectory in task environment:", env_name)
		# elif indexed_data_element is None or ('task_id' not in indexed_data_element.keys()):
		# 	task_id = None
		# 	env_name = None

		if self.args.data=='Mocap':
			# Get animation object from dataset. 
			animation_object = self.dataset[i]['animation']

		# print("We are in the PM visualizer function.")
		# embed()

		########################################
		# 4a) Run unnormalized ground truth trajectory in visualizer. 
		########################################

		##############################
		# ADD CHECK FOR REAL WORLD DATA, and then use dataset image..
		# For now
		##############################

		if self.args.data in ['RealWorldRigid'] and self.args.images_in_real_world_dataset:
			# This should already be segmented to the right start and end point...		
			self.ground_truth_gif = self.visualizer.visualize_prerendered_gif(indexed_data_element['subsampled_images'], gif_path=self.dir_name, gif_name="Traj_{0}_GIF_GT.gif".format(str(i).zfill(3)))
		else:			
			self.ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_GT.gif".format(str(i).zfill(3)), return_and_save=True, end_effector=self.args.ee_trajectories, task_id=env_name)

		# Set plot scaling
		# plot_scale = self.norm_denom_value[6:9].max()
		plot_scale = self.norm_denom_value.max()

		# Also plotting trajectory against time. 
		plt.close()
		# plt.plot(range(unnorm_gt_trajectory.shape[0]),unnorm_gt_trajectory[:,:7])
		# plt.plot(range(unnorm_gt_trajectory.shape[0]),unnorm_gt_trajectory[:,6:9])
		plt.plot(range(unnorm_gt_trajectory.shape[0]),unnorm_gt_trajectory)
		ax = plt.gca()
		ax.set_ylim([-plot_scale, plot_scale])
		plt.savefig(os.path.join(self.dir_name,"Traj_{0}_Plot_GT.png".format(str(i).zfill(3))))
		plt.close()

		########################################
		# 4b) Run unnormalized rollout trajectory in visualizer. 
		########################################

		# Also plotting trajectory against time. 
		plt.close()
		# plt.plot(range(unnorm_pred_trajectory.shape[0]),unnorm_pred_trajectory[:,:7])
		# plt.plot(range(unnorm_pred_trajectory.shape[0]),unnorm_pred_trajectory[:,6:9])
		plt.plot(range(unnorm_pred_trajectory.shape[0]),unnorm_pred_trajectory)
		ax = plt.gca()
		ax.set_ylim([-plot_scale, plot_scale])

		if self.args.viz_sim_rollout:
			# No call to visualizer here means we have to save things on our own. 
			self.rollout_gif = rendered_rollout_trajectory
			
			# Set prefix...
			prefix_list = ['Sim','GTSim']
			gtsim_prefix = prefix_list[self.args.viz_gt_sim_rollout]

			self.visualizer.visualize_prerendered_gif(self.rollout_gif, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_{1}Rollout.gif".format(str(i).zfill(3), gtsim_prefix))
			plt.savefig(os.path.join(self.dir_name,"Traj_{0}_Plot_{1}Rollout.png".format(str(i).zfill(3), gtsim_prefix)))		
		else:
			self.rollout_gif = self.visualizer.visualize_joint_trajectory(unnorm_pred_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_Rollout.gif".format(str(i).zfill(3)), return_and_save=True, end_effector=self.args.ee_trajectories, task_id=env_name)
			
			plt.savefig(os.path.join(self.dir_name,"Traj_{0}_Plot_Rollout.png".format(str(i).zfill(3))))

		plt.close()

		########################################
		# 5) Add to GIF lists. 
		########################################		

		self.gt_gif_list.append(copy.deepcopy(self.ground_truth_gif))
		self.rollout_gif_list.append(copy.deepcopy(self.rollout_gif))
		# print("Embed in get robot viz")
		# embed()
		########################################
		# 6) Return: 
		########################################

		if return_numpy:
			self.ground_truth_gif = np.array(self.ground_truth_gif)
			self.rollout_gif = np.array(self.rollout_gif)

		if return_image:
				return unnorm_pred_trajectory, self.ground_truth_gif, self.rollout_gif
		else:
			return unnorm_pred_trajectory

	def write_results_HTML(self, plots_or_gif='GIF'):
		# Retrieve, append, and print images from datapoints across different models. 

		print("Writing HTML File.")
		# Open Results HTML file. 	    
		with open(os.path.join(self.dir_name,'Results_{0}_{1}.html'.format(self.args.name, plots_or_gif)),'w') as html_file:
			
			# Start HTML doc. 
			html_file.write('<html>')
			html_file.write('<body>')
			html_file.write('<p> Model: {0}</p>'.format(self.args.name))						
			# html_file.write('<p> Average Trajectory Distance: {0}</p>'.format(self.mean_distance))

			extension_dict = {}
			extension_dict['GIF'] = 'gif'
			extension_dict['Plot'] = 'png'

			for i in range(self.N):
				
				if i%100==0:
					print("Datapoint:",i)                        
				html_file.write('<p> <b> Trajectory {}  </b></p>'.format(i))

				file_prefix = self.dir_name

				# Create gif_list by prefixing base_gif_list with file prefix.
				# html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_GT.gif"/>  <img src="Traj_{0}_Rollout.gif"/> </div>'.format(i))
				
				# html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_GIF_GT.gif"/>  <img src="Traj_{0}_GIF_Rollout.gif"/> </div>'.format(i))

				html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_{1}_GT.{2}"/>  <img src="Traj_{0}_{1}_Rollout.{2}"/> </div>'.format(str(i).zfill(3), plots_or_gif, extension_dict[plots_or_gif]))
					
				# Add gap space.
				html_file.write('<p> </p>')

			html_file.write('</body>')
			html_file.write('</html>')

	def write_embedding_HTML(self, animation_object, prefix=""):
		print("Writing Embedding File.")

		t1 = time.time()

		# Adding prefix.
		if self.args.viz_sim_rollout:
			# Modifying prefix based on whether we're visualizing GT Or rollout.
			if self.args.viz_gt_sim_rollout:
				sim_or_not = 'GT_Sim'
			else: 
				sim_or_not = 'Sim'
		else:
			sim_or_not = 'Viz'

		# Open Results HTML file. 	    		
		with open(os.path.join(self.dir_name,'Embedding_{0}_{2}_{1}.html'.format(prefix,self.args.name,sim_or_not)),'w') as html_file:
			
			# Start HTML doc. 
			html_file.write('<html>')
			html_file.write('<body>')
			html_file.write('<p> Model: {0}</p>'.format(self.args.name))

			print("TEMPORARILY EMBEDDING VIA VIDEO RATHER THAN ANIMATION")
			html_file.write(animation_object.to_html5_video())

			###############################
			# Regular embedding as animation
			###############################
			
			# html_file.write(animation_object.to_jshtml())
			# print(animation_object.to_html5_video(), file=html_file)

			html_file.write('</body>')
			html_file.write('</html>')

		t2 = time.time()
		# print("Saving Animation Object.")
		# animation_object.save(os.path.join(self.dir_name,'{0}_Embedding_Video.mp4'.format(self.args.name)))
		# animation_object.save(os.path.join(self.dir_name,'{0}_Embedding_Video.mp4'.format(self.args.name)), writer='imagemagick')
		# t3 = time.time()

		print("Time taken to write this embedding in HTML: ",t2-t1)
		# print("Time taken to save the animation object: ",t3-t2)

	def get_robot_embedding(self, return_tsne_object=False, perplexity=None):

		# # Mean and variance normalize z.
		# mean = self.latent_z_set.mean(axis=0)
		# std = self.latent_z_set.std(axis=0)
		# normed_z = (self.latent_z_set-mean)/std
		normed_z = self.latent_z_set

		if perplexity is None:
			perplexity = self.args.perplexity
		
		print("Perplexity: ", perplexity)

		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=perplexity)
		embedded_zs = tsne.fit_transform(normed_z)

		scale_factor = 1
		scaled_embedded_zs = scale_factor*embedded_zs

		if return_tsne_object:
			return scaled_embedded_zs, tsne
		else:
			return scaled_embedded_zs

	def visualize_robot_embedding(self, scaled_embedded_zs, gt=False):

		# Create figure and axis objects
		# matplotlib.rcParams['figure.figsize'] = [8, 8]
		# zoom_factor = 0.04

		# # Good low res parameters: 
		# matplotlib.rcParams['figure.figsize'] = [8, 8]
		# zoom_factor = 0.04

		# Good spaced out highres parameters: 
		matplotlib.rcParams['figure.figsize'] = [40, 40]			
		# zoom_factor = 0.3
		zoom_factor=0.25

		# Set this parameter to make sure we don't drop frames.
		matplotlib.rcParams['animation.embed_limit'] = 2**128
			
		
		fig, ax = plt.subplots()

		# number_samples = 400
		number_samples = self.N		

		# Create a scatter plot of the embedding itself. The plot does not seem to work without this. 
		ax.scatter(scaled_embedded_zs[:number_samples,0],scaled_embedded_zs[:number_samples,1])
		ax.axis('off')
		ax.set_title("Embedding of Latent Representation of our Model",fontdict={'fontsize':5})
		artists = []
		
		# For number of samples in TSNE / Embedding, create a Image object for each of them. 
		for i in range(len(self.indices)):
			if i%10==0:
				print(i)
			# Create offset image (so that we can place it where we choose), with specific zoom. 

			if gt:
				imagebox = OffsetImage(self.gt_gif_list[i][0],zoom=zoom_factor)
			else:
				imagebox = OffsetImage(self.rollout_gif_list[i][0],zoom=zoom_factor)			

			# Create an annotation box to put the offset image into. specify offset image, position, and disable bounding frame. 
			ab = AnnotationBbox(imagebox, (scaled_embedded_zs[self.indices[i],0], scaled_embedded_zs[self.indices[i],1]), frameon=False)
			# Add the annotation box artist to the list artists. 
			artists.append(ax.add_artist(ab))
			
		def update(t):
			# for i in range(number_samples):
			for i in range(len(self.indices)):
				
				if gt:
					imagebox = OffsetImage(self.gt_gif_list[i][min(t, len(self.gt_gif_list[i])-1)],zoom=zoom_factor)
				else:
					imagebox = OffsetImage(self.rollout_gif_list[i][min(t, len(self.rollout_gif_list[i])-1)],zoom=zoom_factor)			

				ab = AnnotationBbox(imagebox, (scaled_embedded_zs[self.indices[i],0], scaled_embedded_zs[self.indices[i],1]), frameon=False)
				artists.append(ax.add_artist(ab))
			
		# update_len = 20
		print("Maximum length of animation:", self.max_len)
		anim = FuncAnimation(fig, update, frames=np.arange(0, self.max_len), interval=200)

		return anim

	def return_wandb_image(self, image):
		return [wandb.Image(image.transpose(1,2,0))]		

	def return_wandb_gif(self, gif):
		return wandb.Video(gif.transpose((0,3,1,2)), fps=4, format='gif')

	def corrupt_inputs(self, input):
		# 0.1 seems like a good value for the input corruption noise value, that's basically the standard deviation of the Gaussian distribution form which we sample additive noise.
		if isinstance(input, np.ndarray):
			corrupted_input = np.random.normal(loc=0.,scale=self.args.input_corruption_noise,size=input.shape) + input
		else:			
			corrupted_input = torch.randn_like(input)*self.args.input_corruption_noise + input
		return corrupted_input	

	def initialize_training_batches(self):

		print("Initializing batches to manage GPU memory.")
		# Set some parameters that we need for the dry run. 
		extent = len(self.dataset)-self.test_set_size # -self.args.batch_size
		counter = 0
		self.batch_indices_sizes = []
		# self.trajectory_lengths = []
		
		self.current_epoch_running = -1
		print("About to run a dry run. ")
		# Do a dry run of 1 epoch, before we actually start running training. 
		# This is so that we can figure out the batch of 1 epoch.
		 		
		self.shuffle(extent,shuffle=False)		

		# Can now skip this entire block, because we've sorted data according to trajectory length.
		# #########################################################
		# #########################################################
		# for i in range(0,extent,self.args.batch_size):		
		# 	# Dry run iteration. 
		# 	self.run_iteration(counter, self.index_list[i], skip_iteration=True)

		# print("About to find max batch size index.")
		# # Now find maximum batch size iteration. 
		# self.max_batch_size_index = 0
		# self.max_batch_size = 0
		# # traj_lengths = []

		# for x in range(len(self.batch_indices_sizes)):
		# 	if self.batch_indices_sizes[x]['batch_size']>self.max_batch_size:
		# 		self.max_batch_size = self.batch_indices_sizes[x]['batch_size']
		# 		self.max_batch_size_index = self.batch_indices_sizes[x]['i']
		# #########################################################
		# #########################################################

		self.max_batch_size_index = 0
		if self.args.data in ['ToyContext','ContinuousNonZero']:
			self.max_batch_size = 'Full'
		else:
			self.max_batch_size = self.dataset.dataset_trajectory_lengths.max()
								
		print("About to run max batch size iteration.")
		print("This batch size is: ", self.max_batch_size)

		# #########################################################
		# #########################################################
		# # Now run another epoch, where we only skip iteration if it's the max batch size.
		# for i in range(0,extent,self.args.batch_size):
		# 	# Skip unless i is ==max_batch_size_index.
		# 	skip = (i!=self.max_batch_size_index)
		# 	self.run_iteration(counter, self.index_list[i], skip_iteration=skip)
		# #########################################################
		# #########################################################

		# Instead of this clumsy iteration, just run iteration with i=0. 
		self.run_iteration(counter, 0, skip_iteration=0, train=False)

	def task_based_shuffling(self, extent, shuffle=True):
		
		#######################################################################

		# Initialize extent as self.extent
		# extent = self.extent
		index_range = np.arange(0,extent)

		# print("Starting task based shuffling")
		# Implement task ID based shuffling / batching here... 
		self.task_id_map = -np.ones(extent,dtype=int)
		self.task_id_count = np.zeros(self.args.number_of_tasks, dtype=int)		

		for k in range(extent):
			self.task_id_map[k] = self.dataset[k]['task_id']
		for k in range(self.args.number_of_tasks):
			self.task_id_count[k] = (self.task_id_map==k).sum()
		
		# What is this doing?! 
		self.cummulative_count = np.concatenate([np.zeros(1,dtype=int),np.cumsum(self.task_id_count)])

		#######################################################################
		# Now that we have an index map and a count of how many demonstrations there are in each task..
	
		#######################################################################
		# Create blocks. 
		# Best way to perform smart batching is perhaps to sort all indices within a task ID. 
		# Next thing to do is to block up the sorted list.
		# As before, add elements to blocks to ensure it's a full batch.
		#######################################################################
		
		# Get list of indices in each task sorted in decreasing order according to trajectory length for smart batching.
		task_sorted_indices_collection = []			
		for k in range(self.args.number_of_tasks):				
			# task_sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths[self.cummulative_count[k]:self.cummulative_count[k+1]])[::-1]
			task_sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths[self.cummulative_count[k]:self.cummulative_count[k+1]])[::-1]+self.cummulative_count[k]
			task_sorted_indices_collection.append(task_sorted_indices)
		
		# Concatenate this into array. 
		# This allows us to use existing blocking code, and just directly index into this! 
		
		self.concatenated_task_id_sorted_indices = np.concatenate(task_sorted_indices_collection)

		#######################################################################
		# Create blocks..
		#######################################################################

		# Strategy - create blocks from each task ID using task_count, and then just add in more trajectories at random to make it a full batch (if needed).		
		
		self.task_based_shuffling_blocks = []
		self.index_task_id_map = []
		# blocks = []
		task_blocks = []
		counter = 0	

		#######################################################################
		# We're going to create blocks, then pick one of the blocks, maybe based on which bucket the index falls into?
		#######################################################################

		for k in range(self.args.number_of_tasks):
			
			j = 0			 		

			####################################
			# Only try to add an entire batch without resampling if we have more than or exactly enough elements for an entire batch.
			####################################

			while j <= self.task_id_count[k]-self.args.batch_size:
							
				# Add a whole batch.
				block = []

				####################################
				# While we still have items to add to this batch.
				####################################

				while len(block)<self.args.batch_size:				

					# Append index to block.., i.e. TASK SORTED INDEX to block..
					block.append(self.concatenated_task_id_sorted_indices[self.cummulative_count[k]+j])
					j += 1				

				####################################
				# Append this block to the block list. 
				####################################

				if shuffle:
					np.random.shuffle(block)

				self.task_based_shuffling_blocks.append(block)
				self.index_task_id_map.append(k)

			####################################
			# Now that we don't have an entire batch to add. 			
			# Get number of samples we need to add, and check if we need to add at all. 
			####################################

			# If j is ==self.args.batch_size-1, skip this.	
			number_of_samples = self.args.batch_size-(self.task_id_count[k]-j)
			
			# Adding check to ssee if there are actually any elements in this task id... 
			# Otherwise just skip.
			# if number_of_samples>0 and self.task_id_count[k]>0 and number_of_samples<self.args.batch_size:
			if number_of_samples>0 and self.task_id_count[k]>0 and not(j==self.args.batch_size-1):
				# Set pool to sample from. 
				# end_index = -1 if (k+1 >= self.args.number_of_tasks) else k+1
				# random_sample_pool = np.arange(self.cummulative_count[k],self.cummulative_count[end_index])
				random_sample_pool = np.arange(self.cummulative_count[k],self.cummulative_count[k+1])

				samples = np.random.randint(self.cummulative_count[k],high=self.cummulative_count[k+1],size=number_of_samples)
				
				# Create last block. 
				block = []
				# # Add original elements. 
				# [block.append(v) for v in np.arange(self.cummulative_count[k]+j, self.cummulative_count[k+1])]
				# # Now add randomly sampled elements.
				# [block.append(v) for v in samples]

				# Append TASK SORTED INDEX to block..
				# Add original elements. 				
				[block.append(self.concatenated_task_id_sorted_indices[v]) for v in np.arange(self.cummulative_count[k]+j, self.cummulative_count[k+1])]				
				# Now add randomly sampled elements.
				[block.append(self.concatenated_task_id_sorted_indices[v]) for v in samples]

				if shuffle:
					np.random.shuffle(block)

				# Finally append block to block list. 
				self.task_based_shuffling_blocks.append(block)
				self.index_task_id_map.append(k)

		# Also create a block - task ID map.., for easy sampling.. 
		# This is a list of bucket indices for each task that can index into self.task_based_shuffling_blocks...
		self.block_index_list_for_task = []
		
		self.index_task_id_map_array = np.array(self.index_task_id_map)

		for k in range(self.args.number_of_tasks):

			temp_indices = np.where(self.index_task_id_map_array==k)[0]			
			self.block_index_list_for_task.append(temp_indices)	

		# Randomly sample the required number of datapoints. 
		#######################################################################
		# New extent...
		self.extent = len(np.concatenate(self.task_based_shuffling_blocks))
		# Try setting  training extent to same hting...
		self.training_extent = len(np.concatenate(self.task_based_shuffling_blocks))

	def trajectory_length_based_shuffling(self, extent, shuffle=True):
		
		# If we're using full trajectories, do trajectory length based shuffling.
		self.sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths)[::-1]

		# # Bias towards using shorter trajectories if we're debugging.
		# Use dataset_trajectory_length_bias arg isntaed.
		# if self.args.debugging_datapoints > -1: 
		# 	# BIAS SORTED INDICES AWAY FROM SUPER LONG TRAJECTORIES... 
		# 	self.traj_len_bias = 3000
		# 	self.sorted_indices = self.sorted_indices[self.traj_len_bias:]
		
		# Actually just uses sorted_indices...		
		blocks = [self.sorted_indices[i:i+self.args.batch_size] for i in range(0, extent, self.args.batch_size)]
		
		if shuffle:
			np.random.shuffle(blocks)
		# Shuffled index list is just a flattening of blocks.
		self.index_list = [b for bs in blocks for b in bs]

	def randomized_trajectory_length_based_shuffling(self, extent, shuffle=True):
		
		# Pipline.
		# 0) Set block size, and set extents. 
		# 1) Create sample index list. 
		# 2) Fluff indices upto training_extent size. 
		# 3) Sort based on dataset trajectory length. 
		# 4) Set block size. 
		# 5) Block up. 
		# 6) Shuffle blocks. 
		# 7) Divide blocks. 

		# 0) Set block size, and extents. 
		# The higher the batches per block parameter, more randomness, but more suboptimality in terms of runtime. 
		# With dataset trajectory limit, should not be too bad.  
		batches_per_block = 2

		# Now that extent is set, create rounded down extent.
		self.rounded_down_extent = extent//self.args.batch_size*self.args.batch_size

		# Training extent:
		if self.rounded_down_extent==extent:
			self.training_extent = self.rounded_down_extent
		else:
			# This needs to be done such that we have %3==0 batches. 
			batches_to_add = batches_per_block-(self.rounded_down_extent//self.args.batch_size)%batches_per_block
			self.training_extent = self.rounded_down_extent+self.args.batch_size*batches_to_add

		# 1) Create sample index list. 
		original_index_list = np.arange(0,extent)
		
		# 2) Fluff indices upto training_extent size. 		
		if self.rounded_down_extent==extent:
			index_list = original_index_list
		else:
			# additional_index_list = np.random.choice(original_index_list, size=extent-self.rounded_down_extent, replace=False)			
			additional_index_list = np.random.choice(original_index_list, size=self.training_extent - extent, replace=self.args.replace_samples)			
			index_list = np.concatenate([original_index_list, additional_index_list])		
			
		# 3) Sort based on dataset trajectory length. 
		lengths = self.dataset.dataset_trajectory_lengths[index_list]
		sorted_resampled_indices = np.argsort(lengths)[::-1]

		block_size = batches_per_block * self.args.batch_size

		# 5) Block up, now up till training extent.
		blocks = [index_list[sorted_resampled_indices[i:i+block_size]] for i in range(0, self.training_extent, block_size)]
		# blocks = [sorted_resampled_indices[i:i+block_size] for i in range(0, self.training_extent, block_size)]	
		
		# 6) Shuffle blocks. 
		if shuffle:
			for blk in blocks:			
				np.random.shuffle(blk)

		# 7) Divide blocks. 
		# self.index_list = np.concatenate(blocks)
		self.sorted_indices = np.concatenate(blocks)	

	def random_shuffle(self, extent):

		################################
		# Old block based shuffling.		
		################################
	
		# # Replaces np.random.shuffle(self.index_list) with block based shuffling.
		# index_range = np.arange(0,extent)
		# blocks = [index_range[i:i+self.args.batch_size] for i in range(0, extent, self.args.batch_size)]
		# if shuffle:
		# 	np.random.shuffle(blocks)
		# # Shuffled index list is just a flattening of blocks.
		# self.index_list = [b for bs in blocks for b in bs]

		##########################
		# Set training extents.
		##########################

		# Now that extent is set, create rounded down extent.
		self.rounded_down_extent = extent//self.args.batch_size*self.args.batch_size

		# Training extent:
		if self.rounded_down_extent==extent:
			self.training_extent = self.rounded_down_extent
		else:
			self.training_extent = self.rounded_down_extent+self.args.batch_size

		##########################
		# Now shuffle
		##########################

		original_index_list = np.arange(0,extent)
		if self.rounded_down_extent==extent:
			index_list = original_index_list
		else:

			# print("Debug")
			# embed()

			# additional_index_list = np.random.choice(original_index_list, size=extent-self.rounded_down_extent, replace=False)			
			# additional_index_list = np.random.choice(original_index_list, size=self.training_extent-self.rounded_down_extent, replace=False)
			additional_index_list = np.random.choice(original_index_list, size=self.training_extent-extent, replace=self.args.replace_samples)
			index_list = np.concatenate([original_index_list, additional_index_list])
		np.random.shuffle(index_list)
		self.index_list = index_list

	def shuffle(self, extent, shuffle=True):
	
		realdata = (self.args.data in global_dataset_list)

		# Length based shuffling.
		if isinstance(self, PolicyManager_BatchJoint) or isinstance(self, PolicyManager_IKTrainer):

			print("##############################")
			print("##############################")
			print("Necessarily running randomized traj length based shuffling")
			print("##############################")
			print("##############################")
			# print("About to run trajectory length based shuffling.")
			# self.trajectory_length_based_shuffling(extent=extent,shuffle=shuffle)
			self.randomized_trajectory_length_based_shuffling(extent=extent, shuffle=shuffle)

		# # Task based shuffling.
		# elif self.args.task_discriminability or self.args.task_based_supervision or self.args.task_based_shuffling:
		# 	if isinstance(self, PolicyManager_BatchJoint):						
		# 		if not(self.already_shuffled):
		# 			self.task_based_shuffling(extent=extent,shuffle=shuffle)				
		# 			self.already_shuffled = 1				
			
		# 	# if isinstance(self, PolicyManager_Transfer):
		# 	# Also create an index list to shuffle the order of blocks that we observe...

		# 	# 
		# 	# self.index_list = np.arange(0,extent)				
		# 	# np.random.shuffle(self.index_list)
		# 	self.random_shuffle(extent)

		# Task based shuffling.
		elif self.args.task_discriminability or self.args.task_based_supervision or self.args.task_based_shuffling:						
			self.task_based_shuffling(extent=extent,shuffle=shuffle)							
						
		# Random shuffling.
		else:

			################################
			# Single element based shuffling because datasets are ordered
			################################
			self.random_shuffle(extent)

class PolicyManager_Pretrain(PolicyManager_BaseClass):

	def __init__(self, number_policies=4, dataset=None, args=None):

		if args.setting=='imitation':
			super(PolicyManager_Pretrain, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		else:
			super(PolicyManager_Pretrain, self).__init__()

		self.args = args
		# Fixing seeds.
		print("Setting random seeds.")
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)	

		self.data = self.args.data
		# Not used if discrete_z is false.
		self.number_policies = number_policies
		self.dataset = dataset

		# Global input size: trajectory at every step - x,y,action
		# Inputs is now states and actions.

		# Model size parameters
		# if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='DirContNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':
		self.state_size = 2
		self.state_dim = 2
		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = 2		
		self.latent_z_dimensionality = self.args.z_dimensions
		self.number_layers = self.args.number_layers
		self.traj_length = 5
		self.number_epochs = self.args.epochs
		self.test_set_size = 500

		stat_dir_name = self.dataset.stat_dir_name
		if self.args.normalization=='meanvar':
			self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
		elif self.args.normalization=='minmax':
			self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			if self.args.data in ['MOMARTRobotObjectFlat']:
				self.norm_denom_value[self.norm_denom_value==0.]=1.

		if self.args.data in ['MIME','OldMIME']:
			self.state_size = 16			
			self.state_dim = 16
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length
			self.number_epochs = self.args.epochs

			if self.args.ee_trajectories:
				if self.args.normalization=='meanvar':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_EE_Mean.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_EE_Var.npy")
				elif self.args.normalization=='minmax':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_EE_Min.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_EE_Max.npy")
			else:
				if self.args.normalization=='meanvar':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_Orig_Mean.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_Orig_Var.npy")
				elif self.args.normalization=='minmax':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_Orig_Min.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_Orig_Max.npy") - self.norm_sub_value

			# Max of robot_state + object_state sizes across all Baxter environments. 			
			self.cond_robot_state_size = 60
			self.cond_object_state_size = 25
			self.test_set_size = 50
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size

		# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
		elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','RoboMimic','OrigRoboMimic']:
			if self.args.gripper:
				self.state_size = 8
				self.state_dim = 8
			else:
				self.state_size = 7
				self.state_dim = 7		
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length

			if self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']:
				stat_dir_name = "Roboturk"
			elif self.args.data in ['RoboMimic','OrigRoboMimic']:
				stat_dir_name = "Robomimic"
				self.test_set_size = 50

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			# Max of robot_state + object_state sizes across all sawyer environments. 
			# Robot size always 30. Max object state size is... 23. 
			self.cond_robot_state_size = 30			
			self.cond_object_state_size = 23			
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size

		elif self.args.data=='Mocap':
			self.state_size = 22*3
			self.state_dim = 22*3	
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0

		elif self.args.data in ['GRAB']:
			
			self.state_size = 24
			self.state_dim = 24
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['GRABHand']:
			
			self.state_size = 120
			self.state_dim = 120

			if self.args.single_hand in ['left', 'right']:
				self.state_dim //= 2
				self.state_size //= 2

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			# Modify to zero out for now..
			if self.args.skip_wrist:
				self.norm_sub_value[:3] = 0.
				self.norm_denom_value[:3] = 1.
		
		elif self.args.data in ['GRABArmHand']:
			
			if self.args.position_normalization == 'pelvis':
				self.state_size = 144
				self.state_dim = 144

				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
					self.state_size //= 2
			else:
				self.state_size = 147
				self.state_dim = 147
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['GRABArmHandObject']:
			
			self.state_size = 96
			self.state_dim = 96

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['GRABObject']:
			
			self.state_size = 6
			self.state_dim = 6
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['DAPG']:
			
			self.state_size = 51
			self.state_dim = 51
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
		
		elif self.args.data in ['DAPGHand']:
			
			self.state_size = 30
			self.state_dim = 30
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			
		elif self.args.data in ['DAPGObject']:
			
			self.state_size = 21
			self.state_dim = 21
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1

		elif self.args.data in ['DexMV']:
			
			self.state_size = 43
			self.state_dim = 43
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]
		
		elif self.args.data in ['DexMVHand']:
			
			self.state_size = 30
			self.state_dim = 30
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

		elif self.args.data in ['DexMVObject']:
			
			self.state_size = 13
			self.state_dim = 13
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

		elif self.args.data in ['RoboturkObjects','RoboMimicObjects','MOMARTObject']:
			# self.state_size = 14
			# self.state_dim = 14

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 7
			self.state_dim = 7

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50

			# stat_dir_name = "RoboturkObjects"
			# stat_dir_name = self.args.data			

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['RoboturkRobotObjects','RoboMimicRobotObjects']:
			# self.state_size = 14
			# self.state_dim = 14

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 15
			self.state_dim = 15

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50

			# stat_dir_name = "RoboturkRobotObjects"			
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['RoboturkRobotMultiObjects', 'RoboMimiRobotMultiObjects']:

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 22
			self.state_dim = 22

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

		elif self.args.data in ['MOMART']:

			self.state_size = 28
			self.state_dim = 28

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

		elif self.args.data in ['MOMARTRobotObject', 'MOMARTRobotObjectFlat']:

			self.state_size = 28
			self.state_dim = 28

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50	

		elif self.args.data in ['MOMARTRobotObjectFlat']:

			self.state_size = 506
			self.state_dim = 506			

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50	

		elif self.args.data in ['FrankaKitchen']:

			self.state_size = 30
			self.state_dim = 30

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

		elif self.args.data in ['FrankaKitchenRobotObject']:

			self.state_size = 30
			self.state_dim = 30

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

			# print("FK Embed")
			# embed()
			
		elif self.args.data in ['RealWorldRigid', 'RealWorldRigidRobot']:

			self.state_size = 21
			self.state_dim = 21

			if self.args.data in ['RealWorldRobotRobot']:
				
				self.state_size = 7
				self.state_dim = 7

				self.norm_sub_value = self.norm_sub_value[:self.state_size]
				self.norm_denom_value = self.norm_denom_value[:self.state_size]
			else:
				#########################################
				# Manually scale.
				#########################################
				
				if self.args.normalization is not None:
					# self.norm_sub_value will remain unmodified. 
					# self.norm_denom_value will get divided by scale.
					self.norm_denom_value /= self.args.state_scale_factor
					# Manually make sure quaternion dims are unscaled.
					self.norm_denom_value[10:14] = 1.
					self.norm_denom_value[17:] = 1.
					self.norm_sub_value[10:14] = 0.
					self.norm_sub_value[17:] = 0.

		elif self.args.data in ['RealWorldRigidJEEF']:

			self.state_size = 28
			self.state_dim = 28

			# self.norm_sub_value will remain unmodified. 
			# self.norm_denom_value will get divided by scale.
			self.norm_denom_value /= self.args.state_scale_factor
			# Manually make sure quaternion dims are unscaled.
			# Now have to do this for EEF, and two objects. 
			self.norm_denom_value[10:14] = 1.
			self.norm_denom_value[17:20] = 1.
			self.norm_denom_value[24:] = 1.
			self.norm_sub_value[10:14] = 0.
			self.norm_sub_value[17:20] = 0.
			self.norm_sub_value[24:] = 0.

		elif self.args.data in ['NDAX']:

			self.state_size = 13
			self.state_dim = 13
			
			# Set orientation dimensions to be unnormalized.
			self.norm_sub_value[10:] = 0.
			self.norm_denom_value[10:] = 1.

		elif self.args.data in ['NDAXMotorAngles']:

			self.state_size = 6
			self.state_dim = 6


			self.norm_denom_value = self.norm_denom_value[:6]
			self.norm_sub_value = self.norm_sub_value[:6]


		elif self.args.data in ['RealWorldRigidHuman']:

			self.state_size = 77
			self.state_dim = 77
			
			# Set orientation dimensions to be unnormalized.
			

		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		self.output_size = self.state_size
		self.traj_length = self.args.traj_length			
		self.conditional_info_size = 0
		self.test_set_size = 0			


		# Training parameters. 		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self. learning_rate = self.args.learning_rate
		
		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_epochs = self.args.epsilon_over
		self.decay_counter = self.decay_epochs*(len(self.dataset)//self.args.batch_size+1)
		self.variance_decay_counter = self.args.policy_variance_decay_over*(len(self.dataset)//self.args.batch_size+1)
		
		if self.args.kl_schedule:
			self.kl_increment_epochs = self.args.kl_increment_epochs
			self.kl_increment_counter = self.kl_increment_epochs*(len(self.dataset)//self.args.batch_size+1)
			self.kl_begin_increment_epochs = self.args.kl_begin_increment_epochs
			self.kl_begin_increment_counter = self.kl_begin_increment_epochs*(len(self.dataset)//self.args.batch_size+1)
			self.kl_increment_rate = (self.args.final_kl_weight-self.args.initial_kl_weight)/(self.kl_increment_counter)
			self.kl_phase_length_counter = self.args.kl_cyclic_phase_epochs*(len(self.dataset)//self.args.batch_size+1)
		# Log-likelihood penalty.
		self.lambda_likelihood_penalty = self.args.likelihood_penalty
		self.baseline = None

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)	
		self.linear_variance_decay_rate = (self.args.initial_policy_variance - self.args.final_policy_variance)/(self.variance_decay_counter)
		self.quadratic_variance_decay_rate = (self.args.initial_policy_variance - self.args.final_policy_variance)/(self.variance_decay_counter**2)

	def create_networks(self):
		
		# print("Embed in create networks")
		# embed()
		
		# Create K Policy Networks. 
		# This policy network automatically manages input size. 
		if self.args.discrete_z:
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).to(device)
		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.latent_z_dimensionality, self.number_layers).to(device)
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

		# Create encoder.
		if self.args.discrete_z: 
			# The latent space is just one of 4 z's. So make output of encoder a one hot vector.		
			self.encoder_network = EncoderNetwork(self.input_size, self.hidden_size, self.number_policies).to(device)
		else:
			# self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality).to(device)

			# if self.args.transformer:
			# 	self.encoder_network = TransformerEncoder(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).to(device)
			# else:

			if self.args.split_stream_encoder:
				self.encoder_network = ContinuousFactoredEncoderNetwork(self.input_size, self.args.var_hidden_size, int(self.latent_z_dimensionality/2), self.args).to(device)
			else:
				self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)
				# self.encoder_network = OldContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)

		# print("Embed in create networks")
		# embed()

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		self.KLDivergence_loss_function = torch.nn.KLDivLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		if self.args.train_only_policy:
			self.parameter_list = self.policy_network.parameters()
		else:
			self.parameter_list = list(self.policy_network.parameters()) + list(self.encoder_network.parameters())
		
		# Optimize with reguliarzation weight.
		self.optimizer = torch.optim.Adam(self.parameter_list,lr=self.learning_rate,weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False, just_subpolicy=False):
		load_object = torch.load(path)

		if self.args.train_only_policy and self.args.train: 		
			self.encoder_network.load_state_dict(load_object['Encoder_Network'])
		else:
			self.policy_network.load_state_dict(load_object['Policy_Network'])
			if not(only_policy):
				self.encoder_network.load_state_dict(load_object['Encoder_Network'])

	def set_epoch(self, counter):
		if self.args.train:

			# Annealing epsilon and policy variance.
			if counter<self.decay_counter:
				self.epsilon = self.initial_epsilon-self.decay_rate*counter
				
				if self.args.variance_mode in ['Constant']:
					self.policy_variance_value = self.args.variance_value
				elif self.args.variance_mode in ['LinearAnnealed']:
					self.policy_variance_value = self.args.initial_policy_variance - self.linear_variance_decay_rate*counter
				elif self.args.variance_mode in ['QuadraticAnnealed']:
					self.policy_variance_value = self.args.final_policy_variance + self.quadratic_variance_decay_rate*((counter-self.variance_decay_counter)**2)				

			else:
				self.epsilon = self.final_epsilon
				if self.args.variance_mode in ['Constant']:
					self.policy_variance_value = self.args.variance_value
				elif self.args.variance_mode in ['LinearAnnealed', 'QuadraticAnnealed']:
					self.policy_variance_value = self.args.final_policy_variance
		else:
			self.epsilon = self.final_epsilon
			# self.policy_variance_value = self.args.final_policy_variance
			
			# Default variance value, but this shouldn't really matter... because it's in test / eval mode.
			self.policy_variance_value = self.args.variance_value
		
		# print("embed in set epoch")
		# embed()

		# Set KL weight. 
		self.set_kl_weight(counter)		

	def set_kl_weight(self, counter):
		
		# Monotonic KL increase.
		if self.args.kl_schedule=='Monotonic':
			if counter>self.kl_begin_increment_counter:
				if (counter-self.kl_begin_increment_counter)<self.kl_increment_counter:
					self.kl_weight = self.args.initial_kl_weight + self.kl_increment_rate*counter
				else:
					self.kl_weight = self.args.final_kl_weight
			else:
				self.kl_weight = self.args.initial_kl_weight

		# Cyclic KL.
		elif self.args.kl_schedule=='Cyclic':

			# Setup is before X epochs, don't decay / cycle. 
			# After X epochs, cycle. 
			
			if counter<self.kl_begin_increment_counter:
				self.kl_weight = self.args.initial_kl_weight				
			else: 			

				
				# While cycling, self.kl_phase_length_counter is the number of iterations over which we repeat. 
				# self.kl_increment_counter is the iterations (within a cycle) over which we increment KL to maximum.
				# Get where in a single cycle it is. 
				kl_counter = counter % self.kl_phase_length_counter

				# If we're done with incremenet, just set to final weight. 
				if kl_counter>self.kl_increment_counter:
					self.kl_weight = self.args.final_kl_weight
				# Otherwise, do the incremene.t 
				else:
					self.kl_weight = self.args.initial_kl_weight + self.kl_increment_rate*kl_counter		
		
		# No Schedule. 
		else:
			self.kl_weight = self.args.kl_weight

		# Adding branch for cyclic KL weight.		

	def visualize_trajectory(self, traj, no_axes=False):

		fig = plt.figure()		
		ax = fig.gca()
		# ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),cmap='jet')
		plt.plot(traj)
		# plt.xlim(-10,10)
		# plt.ylim(-10,10)

		if no_axes:
			plt.axis('off')
		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	def update_plots(self, counter, loglikelihood, sample_traj, stat_dictionary):
		
		# log_dict['Subpolicy Loglikelihood'] = loglikelihood.mean()
		log_dict = {'Subpolicy Loglikelihood': loglikelihood.mean(), 'Total Loss': self.total_loss.mean(), 'Encoder KL': self.encoder_KL.mean(), 'KL Weight': self.kl_weight}
		if self.args.relative_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Relative State Recon Loss'] = self.unweighted_relative_state_reconstruction_loss
			log_dict['Relative State Recon Loss'] = self.relative_state_reconstruction_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.task_based_aux_loss_weight>0.:
			log_dict['Unweighted Task Based Auxillary Loss'] = self.unweighted_task_based_aux_loss
			log_dict['Task Based Auxillary Loss'] = self.task_based_aux_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.relative_state_phase_aux_loss_weight>0.:
			log_dict['Unweighted Relative Phase Auxillary Loss'] = self.unweighted_relative_state_phase_aux_loss
			log_dict['Relative Phase Auxillary Loss'] = self.relative_state_phase_aux_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Cummmulative Computed State Reconstruction Loss'] = self.unweighted_cummulative_computed_state_reconstruction_loss
			log_dict['Cummulative Computed State Reconstruction Loss'] = self.cummulative_computed_state_reconstruction_loss
		if self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Teacher Forced State Reconstruction Loss'] = self.unweighted_teacher_forced_state_reconstruction_loss
			log_dict['Teacher Forced State Reconstruction Loss'] = self.teacher_forced_state_reconstruction_loss
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0. or self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			log_dict['State Reconstruction Loss'] = self.absolute_state_reconstruction_loss

		if counter%self.args.display_freq==0:
			
			if self.args.batch_size>1:
				# Just select one trajectory from batch.
				sample_traj = sample_traj[:,0]

			############
			# Plotting embedding in tensorboard. 
			############

			# Get latent_z set. 
			self.get_trajectory_and_latent_sets(get_visuals=True)

			log_dict['Average Reconstruction Error:'] = self.avg_reconstruction_error

			# Get embeddings for perplexity=5,10,30, and then plot these.
			# Once we have latent set, get embedding and plot it. 
			self.embedded_z_dict = {}
			self.embedded_z_dict['perp5'] = self.get_robot_embedding(perplexity=5)
			self.embedded_z_dict['perp10'] = self.get_robot_embedding(perplexity=10)
			self.embedded_z_dict['perp30'] = self.get_robot_embedding(perplexity=30)

			# Save embedded z's and trajectory and latent sets.
			self.save_latent_sets(stat_dictionary)

			# Now plot the embedding.
			statistics_line = "Epoch: {0}, Count: {1}, I: {2}, Batch: {3}".format(stat_dictionary['epoch'], stat_dictionary['counter'], stat_dictionary['i'], stat_dictionary['batch_size'])
			image_perp5 = self.plot_embedding(self.embedded_z_dict['perp5'], title="Z Space {0} Perp 5".format(statistics_line))
			image_perp10 = self.plot_embedding(self.embedded_z_dict['perp10'], title="Z Space {0} Perp 10".format(statistics_line))
			image_perp30 = self.plot_embedding(self.embedded_z_dict['perp30'], title="Z Space {0} Perp 30".format(statistics_line))
			
			# Now adding image visuals to the wandb logs.
			# log_dict["GT Trajectory"] = self.return_wandb_image(self.visualize_trajectory(sample_traj))
			log_dict["Embedded Z Space Perplexity 5"] = self.return_wandb_image(image_perp5)
			log_dict["Embedded Z Space Perplexity 10"] =  self.return_wandb_image(image_perp10)
			log_dict["Embedded Z Space Perplexity 30"] =  self.return_wandb_image(image_perp30)

		# if counter%self.args.metric_eval_freq==0:
		# 	self.visualize_robot_data(load_sets=False, number_of_trajectories_to_visualize=10)

		wandb.log(log_dict, step=counter)

	def plot_embedding(self, embedded_zs, title, shared=False, trajectory=False):
	
		fig = plt.figure()
		ax = fig.gca()
		
		if shared:
			colors = 0.2*np.ones((2*self.N))
			colors[self.N:] = 0.8
		else:
			colors = 0.2*np.ones((self.N))

		if trajectory:
			# Create a scatter plot of the embedding.

			self.source_manager.get_trajectory_and_latent_sets()
			self.target_manager.get_trajectory_and_latent_sets()

			ratio = 0.4
			color_scaling = 15

			# Assemble shared trajectory set. 
			traj_length = len(self.source_manager.trajectory_set[0,:,0])
			self.shared_trajectory_set = np.zeros((2*self.N, traj_length, 2))
			
			self.shared_trajectory_set[:self.N] = self.source_manager.trajectory_set
			self.shared_trajectory_set[self.N:] = self.target_manager.trajectory_set
			
			color_range_min = 0.2*color_scaling
			color_range_max = 0.8*color_scaling+traj_length-1

			for i in range(2*self.N):
				ax.scatter(embedded_zs[i,0]+ratio*self.shared_trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.shared_trajectory_set[i,:,1],c=colors[i]*color_scaling+range(traj_length),cmap='jet',vmin=color_range_min,vmax=color_range_max)

		else:
			# Create a scatter plot of the embedding.
			ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet')
		
		# Title. 
		ax.set_title("{0}".format(title),fontdict={'fontsize':15})
		fig.canvas.draw()
		# Grab image.
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	def save_latent_sets(self, stats):

		# Save latent sets, trajectory sets, and finally, the embedded z's for later visualization.

		# Create save directory:
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"LatentSetDirectory")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"LatentSetDirectory","E{0}_C{1}".format(stats['epoch'],stats['counter']))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name, "LatentSet.npy") , self.latent_z_set)
		np.save(os.path.join(self.dir_name, "GT_TrajSet.npy") , self.gt_trajectory_set)
		np.save(os.path.join(self.dir_name, "EmbeddedZSet.npy") , self.embedded_z_dict)
		np.save(os.path.join(self.dir_name, "TaskIDSet.npy"), self.task_id_set)

	def load_latent_sets(self, file_path):
		
		self.latent_z_set = np.load(os.path.join(file_path, "LatentSet.npy"))
		self.gt_trajectory_set = np.load(os.path.join(file_path, "GT_TrajSet.npy"), allow_pickle=True)
		self.embedded_zs = np.load(os.path.join(file_path, "EmbeddedZSet.npy"), allow_pickle=True)
		self.task_id_set = np.load(os.path.join(file_path, "TaskIDSet.npy"), allow_pickle=True)

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		if self.args.discrete_z:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sqeuence for policy network. 
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:
			
		
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1)).to(device)

			# Mask input trajectory according to subpolicy dropout. 
			self.subpolicy_input_dropout_layer = torch.nn.Dropout(self.args.subpolicy_input_dropout)

			torch_input_trajectory = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			masked_input_trajectory = self.subpolicy_input_dropout_layer(torch_input_trajectory)
			assembled_inputs[:,:self.input_size] = masked_input_trajectory

			assembled_inputs[range(1,len(input_trajectory)),self.input_size:-1] = latent_z_indices[:-1]
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size:] = latent_z_indices
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sequence for policy network's forward / logprobabilities function. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def get_trajectory_segment(self, i):

		# if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='DirContNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':
		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext','DeterGoal']:
			# Sample trajectory segment from dataset. 
			sample_traj, sample_action_seq = self.dataset[i]

			# Subsample trajectory segment. 		
			start_timepoint = np.random.randint(0,self.args.traj_length-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[start_timepoint:end_timepoint]	
			sample_action_seq = sample_action_seq[start_timepoint:end_timepoint-1]

			self.current_traj_len = self.traj_length

			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

			return concatenated_traj, sample_action_seq, sample_traj
		
		# elif self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
	
		elif self.args.data in global_dataset_list:
		
			data_element = self.dataset[i]

			####################################			
			# If Invalid.
			####################################
						
			if not(data_element['is_valid']):
				return None, None, None
			
			####################################
			# Check for gripper.
			####################################
				
			if self.args.gripper:
				trajectory = data_element['demo']
			else:
				trajectory = data_element['demo'][:,:-1]

			####################################
			# If allowing variable skill length, set length for this sample.				
			####################################

			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				# self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
				self.current_traj_len = np.random.choice(np.arange(12,17),p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length

			####################################
			# Sample random start point.
			####################################
			
			if trajectory.shape[0]>self.current_traj_len:

				bias_length = int(self.args.pretrain_bias_sampling*trajectory.shape[0])

				# Probability with which to sample biased segment: 
				sample_biased_segment = np.random.binomial(1,p=self.args.pretrain_bias_sampling_prob)

				# If we want to bias sampling of trajectory segments towards the middle of the trajectory, to increase proportion of trajectory segments
				# that are performing motions apart from reaching and returning. 

				# Sample a biased segment if trajectory length is sufficient, and based on probability of sampling.
				if ((trajectory.shape[0]-2*bias_length)>self.current_traj_len) and sample_biased_segment:		
					start_timepoint = np.random.randint(bias_length, trajectory.shape[0] - self.current_traj_len - bias_length)
				else:
					start_timepoint = np.random.randint(0,trajectory.shape[0]-self.current_traj_len)

				end_timepoint = start_timepoint + self.current_traj_len

				# Get trajectory segment and actions. 
				trajectory = trajectory[start_timepoint:end_timepoint]

				# If normalization is set to some value.
				if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
					trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

				# CONDITIONAL INFORMATION for the encoder... 
				if self.args.data in global_dataset_list:


					pass
				# if self.args.data in ['MIME','OldMIME'] or self.args.data=='Mocap':
				# 	pass
				# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
				# 	# robot_states = data_element['robot-state'][start_timepoint:end_timepoint]
				# 	# object_states = data_element['object-state'][start_timepoint:end_timepoint]
				# 	pass

				# 	# self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
				# 	# self.conditional_information[:,:self.cond_robot_state_size] = robot_states
				# 	# self.conditional_information[:,self.cond_robot_state_size:object_states.shape[-1]] = object_states								
				# 	# conditional_info = np.concatenate([robot_states,object_states],axis=1)	
			else:					
				return None, None, None

			action_sequence = np.diff(trajectory,axis=0)
			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)

			# NOW SCALE THIS ACTION SEQUENCE BY SOME FACTOR: 
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			return concatenated_traj, scaled_action_sequence, trajectory

	def construct_dummy_latents(self, latent_z):

		if self.args.discrete_z:
			latent_z_indices = latent_z.float()*torch.ones((self.traj_length)).to(device).float()			
		else:
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z.squeeze(0) for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.current_traj_len)).to(device).float()
		# latent_b[-1] = 1.

		return latent_z_indices, latent_b			

	def initialize_aux_losses(self):
		
		# Initialize losses.
		self.unweighted_relative_state_reconstruction_loss = 0.
		self.relative_state_reconstruction_loss = 0.
		# 
		self.unweighted_relative_state_phase_aux_loss = 0.
		self.relative_state_phase_aux_loss = 0.
		# 
		self.unweighted_task_based_aux_loss = 0.
		self.task_based_aux_loss = 0.

		# 
		self.unweighted_teacher_forced_state_reconstruction_loss = 0.
		self.teacher_forced_state_reconstruction_loss = 0.
		self.unweighted_cummmulative_computed_state_reconstruction_loss = 0.
		self.cummulative_computed_state_reconstruction_loss = 0.

	def compute_auxillary_losses(self, update_dict):

		self.initialize_aux_losses()

		# Set the relative state reconstruction loss.
		if self.args.relative_state_reconstruction_loss_weight>0.:
			self.compute_relative_state_reconstruction_loss()
		if self.args.task_based_aux_loss_weight>0. or self.args.relative_state_phase_aux_loss_weight>0.:
			self.compute_pairwise_z_distance(update_dict['latent_z'][0])
		# Task based aux loss weight. 
		if self.args.task_based_aux_loss_weight>0.:
			self.compute_task_based_aux_loss(update_dict)
		# Relative. 
		if self.args.relative_state_phase_aux_loss_weight>0.:
			self.compute_relative_state_phase_aux_loss(update_dict)
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0. or self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			self.compute_absolute_state_reconstruction_loss()

		# Weighting the auxillary loss...
		self.aux_loss = self.relative_state_reconstruction_loss + self.relative_state_phase_aux_loss + self.task_based_aux_loss + self.absolute_state_reconstruction_loss

	def compute_pairwise_z_distance(self, z_set):

		# Compute pairwise task based weights.
		self.pairwise_z_distance = torch.cdist(z_set, z_set)[0]

		# Clamped z distance loss. 
		# self.clamped_pairwise_z_distance = torch.clamp(self.pairwise_z_distance - self.args.pairwise_z_distance_threshold, min=0.)
		self.clamped_pairwise_z_distance = torch.clamp(self.args.pairwise_z_distance_threshold - self.pairwise_z_distance, min=0.)

	def compute_relative_state_class_vectors(self, update_dict):

		# Compute relative state vectors.

		# Get original states. 
		robot_traj = self.sample_traj_var[...,:3]
		env_traj = self.sample_traj_var[...,self.args.robot_state_size:self.args.robot_state_size+3]
		relative_state_traj = robot_traj - env_traj		

		# Compute relative state. 
		# relative_state_traj = torch.tensor(robot_traj - env_traj).to(device)
		
		# Compute diff. 
		robot_traj_diff = np.diff(robot_traj, axis=0)
		env_traj_diff = np.diff(env_traj, axis=0)
		relative_state_traj_diff = np.diff(relative_state_traj, axis=0)		

		# Compute norm. 
		robot_traj_norm = np.linalg.norm(robot_traj_diff, axis=-1)
		env_traj_norm = np.linalg.norm(env_traj_diff, axis=-1)
		relative_state_traj_norm = np.linalg.norm(relative_state_traj_diff, axis=-1)

		# Compute sum.
		beta_vector = np.stack([robot_traj_norm.sum(axis=0), env_traj_norm.sum(axis=0), relative_state_traj_norm.sum(axis=0)])

		# Threshold this vector. 
		self.beta_threshold_value = 0.5
		self.thresholded_beta_vector = np.swapaxes((beta_vector>self.beta_threshold_value).astype(float), 0, 1)		
		self.torch_thresholded_beta_vector = torch.tensor(self.thresholded_beta_vector).to(device)

	def compute_task_based_aux_loss(self, update_dict):

		# Task list. 
		task_list = []
		for k in range(self.args.batch_size):
			task_list.append(update_dict['data_element'][k]['task-id'])
		# task_array = np.array(task_list).reshape(self.args.batch_size,1)
		torch_task_array = torch.tensor(task_list, dtype=float).reshape(self.args.batch_size,1).to(device)
		
		# Compute pairwise task based weights. 
		# pairwise_task_matrix = (scipy.spatial.distance.cdist(task_array)==0).astype(int).astype(float)
		pairwise_task_matrix = (torch.cdist(torch_task_array, torch_task_array)==0).int().float()

		# Positive weighted task loss. 
		positive_weighted_task_loss = pairwise_task_matrix*self.pairwise_z_distance

		# Negative weighted task loss. 
		# MUST CHECK SIGNAGE OF THIS. 
		negative_weighted_task_loss = (1.-pairwise_task_matrix)*self.clamped_pairwise_z_distance

		# Total task_based_aux_loss.
		self.unweighted_task_based_aux_loss = (positive_weighted_task_loss + self.args.negative_task_based_component_weight*negative_weighted_task_loss).mean()
		self.task_based_aux_loss = self.args.task_based_aux_loss_weight*self.unweighted_task_based_aux_loss

	def compute_relative_state_phase_aux_loss(self, update_dict):

		# Compute vectors first for the batch.
		self.compute_relative_state_class_vectors(update_dict)

		# Compute similarity of rel state vector across batch.
		self.relative_state_vector_distance = torch.cdist(self.torch_thresholded_beta_vector, self.torch_thresholded_beta_vector)
		self.relative_state_vector_similarity_matrix = (self.relative_state_vector_distance==0).float()
	
		# Now set positive loss.
		positive_weighted_rel_state_phase_loss = self.relative_state_vector_similarity_matrix*self.pairwise_z_distance

		# Set negative component
		negative_weighted_rel_state_phase_loss = (1.-self.relative_state_vector_similarity_matrix)*self.clamped_pairwise_z_distance

		# Total rel state phase loss.
		self.unweighted_relative_state_phase_aux_loss = (positive_weighted_rel_state_phase_loss + self.args.negative_task_based_component_weight*negative_weighted_rel_state_phase_loss).mean()
		self.relative_state_phase_aux_loss = self.args.relative_state_phase_aux_loss_weight*self.unweighted_relative_state_phase_aux_loss

	def compute_relative_state_reconstruction_loss(self):
		
		# Get mean of actions from the policy networks.
		mean_policy_actions = self.policy_network.mean_outputs

		# Get translational states. 
		mean_policy_robot_actions = mean_policy_actions[...,:3]
		mean_policy_env_actions = mean_policy_actions[...,self.args.robot_state_size:self.args.robot_state_size+3]
		# Compute relative actions. 
		mean_policy_relative_state_actions = mean_policy_robot_actions - mean_policy_env_actions

		# Rollout states, then compute relative states - although this shouldn't matter because it's linear. 

		# # Compute relative initial state. 		
		# initial_state = self.sample_traj_var[0]
		# initial_robot_state = initial_state[:,:3]
		# initial_env_state = initial_state[:,self.args.robot_state_size:self.args.robot_state_size+3]
		# relative_initial_state = initial_robot_state - initial_env_state

		# Get relative states.
		robot_traj = self.sample_traj_var[...,:3]
		env_traj = self.sample_traj_var[...,self.args.robot_state_size:self.args.robot_state_size+3]
		relative_state_traj = torch.tensor(robot_traj - env_traj).to(device)
		initial_relative_state = relative_state_traj[0]
		# torch_initial_relative_state = torch.tensor(initial_relative_state).cuda()		

		# Differentiable rollouts. 
		policy_predicted_relative_state_traj = initial_relative_state + torch.cumsum(mean_policy_relative_state_actions, axis=0)

		# Set reconsturction loss.
		self.unweighted_relative_state_reconstruction_loss = (policy_predicted_relative_state_traj - relative_state_traj).norm(dim=2).mean()
		self.relative_state_reconstruction_loss = self.args.relative_state_reconstruction_loss_weight*self.unweighted_relative_state_reconstruction_loss

	def relabel_relative_object_state(self, torch_trajectory):

		# Copy over
		relabelled_state_sequence = torch_trajectory

		# Relabel the dims. 

		print("Debug in Relabel")
		embed()

		torchified_object_state = torch.from_numpy(self.normalized_subsampled_relative_object_state).to(device).view(-1, self.args.batch_size, self.args.env_state_size)		
		relabelled_state_sequence[..., -self.args.env_state_size:] = torchified_object_state

		return relabelled_state_sequence	

	def compute_absolute_state_reconstruction_loss(self):

		# Get the mean of the actions from the policy networks until the penultimate action.
		mean_policy_actions = self.policy_network.mean_outputs[:-1]

		# Initial state - remember, states are Time x Batch x State.
		torch_trajectory = torch.from_numpy(self.sample_traj_var).to(device)

		if self.args.data in ['RealWorldRigidJEEF']:
			torch_trajectory = self.relabel_relative_object_state(torch_trajectory)

		initial_state = torch_trajectory[0]

		# Compute reconstructed trajectory differentiably excluding the first timestep. 
		cummulative_computed_reconstructed_trajectory = initial_state + torch.cumsum(mean_policy_actions, axis=0)
		# Teacher forced state.
		teacher_forced_reconstructed_trajectory = torch_trajectory[:-1] + mean_policy_actions

		# Set both of the reconstruction losses of absolute state.
		self.unweighted_cummulative_computed_state_reconstruction_loss = (cummulative_computed_reconstructed_trajectory - torch_trajectory[1:]).norm(dim=2).mean()
		self.unweighted_teacher_forced_state_reconstruction_loss = (teacher_forced_reconstructed_trajectory - torch_trajectory[1:]).norm(dim=2).mean()
		
		# Weighted losses. 
		self.cummulative_computed_state_reconstruction_loss = self.args.cummulative_computed_state_reconstruction_loss_weight * self.unweighted_cummulative_computed_state_reconstruction_loss
		self.teacher_forced_state_reconstruction_loss = self.args.teacher_forced_state_reconstruction_loss_weight*self.unweighted_teacher_forced_state_reconstruction_loss

		# Merge. 
		self.absolute_state_reconstruction_loss = self.cummulative_computed_state_reconstruction_loss + self.teacher_forced_state_reconstruction_loss

	def update_policies_reparam(self, loglikelihood, encoder_KL, update_dict=None):
		
		self.optimizer.zero_grad()

		# Losses computed as sums.
		# self.likelihood_loss = -loglikelihood.sum()
		# self.encoder_KL = encoder_KL.sum()

		# Instead of summing losses, we should try taking the mean of the  losses, so we can avoid running into issues of variable timesteps and stuff like that. 
		# We should also consider training with randomly sampled number of timesteps.
		self.likelihood_loss = -loglikelihood.mean()
		self.encoder_KL = encoder_KL.mean()

		self.compute_auxillary_losses(update_dict)
		# Adding a penalty for link lengths. 
		# self.link_length_loss = ... 

		self.total_loss = (self.likelihood_loss + self.kl_weight*self.encoder_KL + self.aux_loss) 
		# + self.link_length_loss) 

		if self.args.debug:
			print("Embedding in Update subpolicies.")
			embed()

		self.total_loss.backward()
		self.optimizer.step()

	def rollout_visuals(self, i, latent_z=None, return_traj=False, rollout_length=None, traj_start=None):

		# Initialize states and latent_z, etc. 
		# For t in range(number timesteps):
		# 	# Retrieve action by feeding input to policy. 
		# 	# Step in environment with action.
		# 	# Update inputs with new state and previously executed action. 

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.state_dim = 2
			self.rollout_timesteps = 5
		elif self.args.data in ['MIME','OldMIME']:
			self.state_dim = 16
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
			self.state_dim = 8
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRAB']:
			self.state_dim = 24
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABArmHand']:
			if self.args.position_normalization == 'pelvis':
				self.state_dim = 144
				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
			else:
				self.state_dim = 147
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABArmHandObject']:
			self.state_size = 96
			self.state_dim = 96
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABObject']:
			self.state_dim = 6
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABHand']:
			self.state_dim = 120
			if self.args.single_hand in ['left', 'right']:
				self.state_dim //= 2
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPG']:
			self.state_dim = 51
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPGHand']:
			self.state_dim = 30
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPGObject']:
			self.state_dim = 21
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMV']:
			self.state_dim = 43
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMVHand']:
			self.state_dim = 30
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMVObject']:
			self.state_dim = 13
			self.rollout_timesteps = self.traj_length

		if rollout_length is not None:
			self.rollout_timesteps = rollout_length
	
		if traj_start is None:
			start_state = torch.zeros((self.state_dim))
		else:
			start_state = torch.from_numpy(traj_start)
		

		if self.args.discrete_z:
			# Assuming 4 discrete subpolicies, just set subpolicy input to 1 at the latent_z index == i. 
			subpolicy_inputs = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
			subpolicy_inputs[0,self.input_size+i] = 1. 
		else:
			subpolicy_inputs = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[0,self.input_size:] = latent_z

		subpolicy_inputs[0,:self.state_dim] = start_state
		# subpolicy_inputs[0,-1] = 1.		
		
		for t in range(self.rollout_timesteps-1):

			actions = self.policy_network.get_actions(subpolicy_inputs,greedy=True,batch_size=1)
			
			# Select last action to execute.
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor

			# Compute next state. 
			new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute

			# New input row: 
			if self.args.discrete_z:
				input_row = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
				input_row[0,self.input_size+i] = 1. 
			else:
				input_row = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device).float()
				input_row[0,self.input_size:] = latent_z
			input_row[0,:self.state_dim] = new_state
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute	
			# input_row[0,-1] = 1.

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)
		# print("latent_z:",latent_z)
		trajectory_rollout = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		# print("Trajectory:",trajectory_rollout)

		if return_traj:
			return trajectory_rollout		

	def run_iteration(self, counter, i, return_z=False, and_train=True):

		####################################
		####################################
		# Basic Training Algorithm: 
		# For E epochs:
		# 	# For all trajectories:
		#		# Sample trajectory segment from dataset. 
		# 		# Encode trajectory segment into latent z. 
		# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		# 		# Update parameters. 
		####################################
		####################################

		self.set_epoch(counter)

		############# (0) ##################
		# Sample trajectory segment from dataset. 
		####################################

		# Sample trajectory segment from dataset.
		input_dict = {}

		input_dict['state_action_trajectory'], input_dict['sample_action_seq'], input_dict['sample_traj'], input_dict['data_element'] = self.get_trajectory_segment(i)
		# state_action_trajectory, sample_action_seq, sample_traj, data_element  = self.get_trajectory_segment(i)
		# self.sample_traj_var = sample_traj
		self.sample_traj_var = input_dict['sample_traj']
		self.input_dict = input_dict
		####################################
		############# (0a) #############
		####################################

		# Corrupt the inputs according to how much input_corruption_noise is set to.
		state_action_trajectory = self.corrupt_inputs(input_dict['state_action_trajectory'])

		if state_action_trajectory is not None:
			
			####################################
			############# (1) #############
			####################################

			torch_traj_seg = torch.tensor(state_action_trajectory).to(device).float()
			# Encode trajectory segment into latent z. 		
						
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon)
			# latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg)
			
			####################################
			########## (2) & (3) ##########
			####################################

			# print("Embed in rut iter")
			# embed()

			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

			############# (3a) #############
			_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(state_action_trajectory, latent_z_seq, latent_b, input_dict['sample_action_seq'])
			
			############# (3b) #############
			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)

			loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq, self.policy_variance_value)
			loglikelihood = loglikelihoods[:-1].mean()
			 
			if self.args.debug:
				print("Embedding in Train.")
				embed()

			####################################
			# (4) Update parameters. 
			####################################
			
			if self.args.train and and_train:

				####################################
				# (4a) Update parameters based on likelihood, subpolicy inputs, and kl divergence.
				####################################
				
				update_dict = input_dict
				update_dict['latent_z'] = latent_z				

				self.update_policies_reparam(loglikelihood, kl_divergence, update_dict=update_dict)

				####################################
				# (4b) Update Plots. 
				####################################
				
				stats = {}
				stats['counter'] = counter
				stats['i'] = i
				stats['epoch'] = self.current_epoch_running
				stats['batch_size'] = self.args.batch_size			
				self.update_plots(counter, loglikelihood, state_action_trajectory, stats)

				####################################
				# (5) Return.
				####################################

			if return_z:
				return latent_z, input_dict['sample_traj'], sample_action_seq, input_dict['data_element']
									
		else: 
			return None, None, None

	def evaluate_metrics(self):		
		self.distances = -np.ones((self.test_set_size))
		self.robot_z_nn_distances = -np.ones((self.test_set_size))
		self.env_z_nn_distances = -np.ones((self.test_set_size))
		# Get test set elements as last (self.test_set_size) number of elements of dataset.
		for i in range(self.test_set_size):

			index = i + len(self.dataset)-self.test_set_size
			print("Evaluating ", i, " in test set, or ", index, " in dataset.")
			# Get latent z. 					
			latent_z, sample_traj, sample_action_seq, _ = self.run_iteration(0, index, return_z=True)

			if sample_traj is not None:

				# Feed latent z to the rollout.
				# rollout_trajectory = self.rollout_visuals(index, latent_z=latent_z, return_traj=True)
				rollout_trajectory, rendered_rollout_trajectory = self.rollout_robot_trajectory(sample_traj[0], latent_z, rollout_length=len(sample_traj))

				self.distances[i] = ((sample_traj-rollout_trajectory)**2).mean()	

			robot_nn_z_dist, env_nn_z_dist = self.evaluate_z_distances_for_batch(latent_z)

		self.mean_distance = self.distances[self.distances>0].mean()

	def evaluate(self, model=None, suffix=None):

		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.visualize_embedding_space(suffix=suffix)

		if self.args.data in global_dataset_list:

			print("Running Evaluation of State Distances on small test set.")
			# self.evaluate_metrics()		

			# Only running viz if we're actually pretraining.
			if self.args.traj_segments:
				print("Running Visualization on Robot Data.")	

				# self.visualize_robot_data(load_sets=True)
				whether_load_z_set = self.args.latent_set_file_path is not None

				# print("###############################################")
				# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				# print("Temporarily not visualizing.")
				# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				# print("###############################################")
				self.visualize_robot_data(load_sets=whether_load_z_set)

				
				print("###############################################")
				print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				print("Query before we run get trajectory latent sets, so latent_z_set isn't overwritten..")
				print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				print("###############################################")				
				embed()

				# Get reconstruction error... 
				self.get_trajectory_and_latent_sets(get_visuals=True)
				print("The Average Reconstruction Error is: ", self.avg_reconstruction_error)


			else:
				# Create save directory:
				upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

				if not(os.path.isdir(upper_dir_name)):
					os.mkdir(upper_dir_name)

				model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))				
				self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))

				if not(os.path.isdir(self.dir_name)):
					os.mkdir(self.dir_name)

	def get_trajectory_and_latent_sets(self, get_visuals=True):
		# For N number of random trajectories from MIME: 
		#	# Encode trajectory using encoder into latent_z. 
		# 	# Feed latent_z into subpolicy. 
		#	# Rollout subpolicy for t timesteps. 
		#	# Plot rollout.
		# Embed plots. 

		# Set N:
		self.N = 500

		self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))
			
		if self.args.setting=='transfer' or self.args.setting=='cycle_transfer' or self.args.setting=='fixembed':
			if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
				self.state_dim = 2
				self.rollout_timesteps = 5		
			if self.args.data in ['MIME','OldMIME']:
				self.state_dim = 16
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
				self.state_dim = 8
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRAB']:
				self.state_dim = 24
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABArmHand']:
				if self.args.position_normalization == 'pelvis':
					self.state_dim = 144
					if self.args.single_hand in ['left', 'right']:
						self.state_dim //= 2
				else:
					self.state_dim = 147
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABArmHandObject']:
				self.state_dim = 96
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABObject']:
				self.state_dim = 6
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABHand']:
				self.state_dim = 126
				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DAPG']:
				self.state_dim = 51
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DAPGHand']:
				self.state_dim = 30
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DAPGObject']:
				self.state_dim = 21
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DexMV']:
				self.state_dim = 43
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DexMVHand']:
				self.state_dim = 30
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DexMVObject']:
				self.state_dim = 13
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['RoboturkObjects']:
				# Now switching to using 7 dimensions instead of 14, so as to not use relative pose.
				self.state_dim = 7
				# self.state_dim = 14
				self.rollout_timesteps = self.traj_length

			self.trajectory_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))

		else:
			self.trajectory_set = []
		# self.gt_trajectory_set = np.zeros((self.N, self., self.state_dim))
		
		self.gt_trajectory_set = []
		# Save TASK IDs 
		self.task_id_set = []

		# Use the dataset to get reasonable trajectories (because without the information bottleneck / KL between N(0,1), cannot just randomly sample.)
		# # for i in range(self.N//self.args.batch_size+1, 32)
		# for i in range(0, self.N, self.args.batch_size):
		for i in range(self.N//self.args.batch_size+1):

			# Mapped index
			number_batches_for_dataset = (len(self.dataset)//self.args.batch_size)+1
			j = i % number_batches_for_dataset

			########################################
			# (1) Encoder trajectory. 
			########################################

			latent_z, sample_trajs, _, data_element = self.run_iteration(0, j*self.args.batch_size, return_z=True, and_train=False)

			########################################
			# Iterate over items in the batch.
			########################################
			# print("Embed in latent set creation")
			# embed()

			for b in range(self.args.batch_size):

				if self.args.batch_size*i+b>=self.N:
					break

				self.latent_z_set[i*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())
				# self.latent_z_set[i+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())
				self.gt_trajectory_set.append(copy.deepcopy(sample_trajs[:,b]))
				
				self.task_id_set.append(data_element[b]['task-id'])

				if get_visuals:
					# (2) Now rollout policy.	
					if self.args.setting=='transfer' or self.args.setting=='cycle_transfer' or self.args.setting=='fixembed':
						self.trajectory_set[i*self.args.batch_size+b] = self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True, traj_start=sample_trajs[0,b])
						# self.trajectory_set[i+b] = self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True)
					elif self.args.setting=='pretrain_sub':							
						self.trajectory_set.append(self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True, traj_start=sample_trajs[0,b], rollout_length=sample_trajs.shape[0]))
					else:
						self.trajectory_set.append(self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True, traj_start=sample_trajs[0,b]))
			
			if self.args.batch_size*i+b>=self.N:
				break

		# print("Embed in latent set creation before trajectory error evaluation.")
		# embed()

		# Compute average reconstruction error.
		if get_visuals:
			self.gt_traj_set_array = np.array(self.gt_trajectory_set, dtype=object)
			self.trajectory_set = np.array(self.trajectory_set, dtype=object)

			# self.gt_traj_set_array = np.array(self.gt_trajectory_set)
			# self.trajectory_set = np.array(self.trajectory_set)

			# self.avg_reconstruction_error = (self.gt_traj_set_array-self.trajectory_set).mean()
			self.reconstruction_errors = np.zeros(len(self.gt_traj_set_array))
			for k in range(len(self.reconstruction_errors)):
				self.reconstruction_errors[k] = ((self.gt_traj_set_array[k]-self.trajectory_set[k])**2).mean()
			self.avg_reconstruction_error = self.reconstruction_errors.mean()
		else:
			self.avg_reconstruction_error = 0.

	def visualize_embedding_space(self, suffix=None):

		self.get_trajectory_and_latent_sets()

		# TSNE on latentz's.
		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=self.args.perplexity)
		embedded_zs = tsne.fit_transform(self.latent_z_set)

		# ratio = 0.3
		# if self.args.setting in ['transfer','cycletransfer','']
		ratio = (embedded_zs.max()-embedded_zs.min())*0.01
		
		for i in range(self.N):
			plt.scatter(embedded_zs[i,0]+ratio*self.trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.trajectory_set[i,:,1],c=range(self.rollout_timesteps),cmap='jet')

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))		
		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		if suffix is not None:
			self.dir_name = os.path.join(self.dir_name, suffix)

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		# Format with name.
		plt.savefig("{0}/Embedding_Joint_{1}.png".format(self.dir_name,self.args.name))
		plt.close()	

	def create_z_kdtrees(self):
		
		####################################
		# Algorithm to construct models
		####################################

		# 0) Assume that the encoder(s) are trained, and that the latent space is trained.
		# 1) Maintain map of Z_R <--> Z_E. 
		# 	1a) Check that the latent_z_sets are tuples. 
		# 	1b) Seems like we don't actually need the map if the latent z sets are constructed by the same function / indexing.
		# 2) Construct KD Trees. 
		# 	2a) KD_R = KDTREE( {Z_R} )
		# 	2b) KD_E = KDTREE( {Z_E} )
		
		# self.kdtree_robot_z = KDTree(self.robot_latent_z_set)
		# self.kdtree_env_z = KDTree(self.env_latent_z_set)		

		self.kdtree_dict = {}
		self.kdtree_dict['robot'] = KDTree(self.stream_latent_z_dict['robot'])
		self.kdtree_dict['env'] = KDTree(self.stream_latent_z_dict['env'])

	def get_query_trajectory(self, input_state_trajectory, stream=None):
		
		# Assume trajectory is dimensions |T| x |S|. 
		index_dict = {}
		index_dict['robot'] = np.arange(0,8)
		index_dict['env'] = np.arange(8,15)

		indices = index_dict[stream] 
		
		stream_input_state_trajectory = input_state_trajectory[:,indices]
		
		# Get actions. 
		actions = np.diff(stream_input_state_trajectory, axis=0)

		# Pad actions.
		padded_actions = np.concatenate([actions,np.zeros((1,stream_input_state_trajectory.shape[1]))], axis=0)

		# Concatenate state and actions. 
		state_action_traj = np.concatenate([stream_input_state_trajectory, padded_actions], axis=1)

		# Torchify. 
		torch_state_action_traj = torch.from_numpy(state_action_traj).to(device).float()

		return torch_state_action_traj

	def retrieve_nearest_neighbors_from_trajectory(self, trajectory, stream, number_neighbors=1, artificial_batch_size=1):

		# Based on stream, set which KDTree and which latent set to use. 
		kdtree = self.kdtree_dict[stream]
		latent_set = self.stream_latent_z_dict[stream]
		if stream=='robot':
			net_dict = self.encoder_network.robot_network_dict
			size_dict = self.encoder_network.robot_size_dict
		elif stream=='env':
			net_dict = self.encoder_network.env_network_dict
			size_dict = self.encoder_network.env_size_dict

		####################################
		# Query, given a trajectory
		####################################

		# 0) Assumes that we have a concatenation of states and actions.
		# 1) z_r = E_r (Tau_r) || z_e = E_e (Tau_e)
		# 2) z_r^{*NN} = KDT_r.query(z_r) || z_e^{*NN} = KDT_e.query(z_e) 	

		# 1) Query encoder for latent representation of trajectory.
		# Well, we just run super.forward() of the encoder network, which is a continuous factored encoder
		# which inherits its forward function from the continuous encoder. 
		
		# Do not need epsilon or eval. 
		retrieved_z, _, _, _ = self.encoder_network.run_super_forward(trajectory, epsilon=0.0, \
			# network_dict=self.encoder_network.robot_network_dict, size_dict=self.encoder_network.robot_size_dict, artificial_batch_size=1)
			network_dict=net_dict, size_dict=size_dict, artificial_batch_size=artificial_batch_size)
		
		# 2) Query KD Tree with encoding of given trajectory. 
		z_neighbor_distances, z_neighbors_indices = kdtree.query(retrieved_z.detach().cpu().numpy(), k=number_neighbors)

		return z_neighbor_distances, z_neighbor_indices

	def retrieve_cross_indexed_nearest_neighbor_from_trajectory(self, trajectory, stream, number_neighbors=1):

		# Get neighbors. 
		z_neighbor_distances , z_neighbor_indices = self.retrieve_nearest_neighbors_from_trajectory(trajectory, stream, number_neighbors)

		# Cross index. 				
		cross_stream = set(self.stream_latent_z_dict.keys()) - set([stream])
		cross_latent_set = self.stream_latent_z_dict[cross_stream]
		# 	desired_z_e = self.robot_latent_z_set[z_r_nearest_neighbor_index]

		cross_indexed_z = cross_latent_set[z_neighbor_indices]

		return cross_indexed_z
	
	def define_forward_inverse_models(self):

		# 0) Get latent z sets.
		# 1) Create KD trees. 

		# 0) Make sure we've run visualize_robot_data; then split z sets. 
		# Run visualize_robot_data. 

		# # Create z sets. 
		# self.robot_latent_z_set = copy.deepcopy(self.latent_z_set[:,:int(self.latent_z_dimensionality/2)])
		# self.env_latent_z_set = copy.deepcopy(self.latent_z_set[:,int(self.latent_z_dimensionality/2):])

		# Single stream latent z set dict. 
		self.stream_latent_z_dict = {}
		self.stream_latent_z_dict['robot']= copy.deepcopy(self.latent_z_set[:,:int(self.latent_z_dimensionality/2)])
		self.stream_latent_z_dict['env']= copy.deepcopy(self.latent_z_set[:,int(self.latent_z_dimensionality/2):])
		
		# 1) Create KD Trees.
		self.create_z_kdtrees()	

	def evaluate_z_distances_for_batch(self, latent_z):

		latent_z_sets = {}
		latent_z_sets['robot'] = latent_z[:,:,:int(self.latent_z_dimensionality/2)].detach().cpu().numpy()
		latent_z_sets['env'] = latent_z[:,:,int(self.latent_z_dimensionality/2):].detach().cpu().numpy()
		
		# Robot nearest neighbors.. 
		# Number nearest neighbor
		number_nearest_neighbors = 5
		robot_nn_distances, robot_nn_indices = self.stream_latent_z_dict['robot'].query(latent_z_sets['robot'], k=number_nearest_neighbors)
		env_nn_distances, env_nn_indices = self.stream_latent_z_dict['env'].query(latent_z_sets['env'], k=number_nearest_neighbors)

		# print("Robot Latent Space Average Distance: ", robot_nn_distances.mean())
		return robot_nn_distances, env_nn_distances

	def evaluate_forward_inverse_models(self):

		# Paradigm for forward model.
		# (0) For number of evaluate trajectories:
		# (1) 	Get trajectory from dataset.
		# (1a) 	Parse trajectory into robot and env trajectory T_r, T_e. 
		# (2) 	Encode trajectory into z_R, z_E via global / joint encoder. 
		# (3)	Find nearest neighbor of z_R / z_E in {Z}_R / {Z}_E i.e. z_R^* \ z_E^*.
		# (4) 	Find corresponding z_E^* / z_R^*.
		# (5)	Evaluate |z_E^* - z_E|_2 / .
		# (6) 	Decode z_E to T_e^* / .
		# (7) 	Evaluate |T_e^* - T_e|_2 / . 

		# Remember, steps 0-3 are executed in sample_trajectories_for_evaluating spaces. 

		pass

	def create_evaluate_dynamics_models(self):

		# After we've created latent sets.
		self.define_forward_inverse_models()

		# Eval
		self.evaluate_forward_inverse_models()

class PolicyManager_BatchPretrain(PolicyManager_Pretrain):

	def __init__(self, number_policies=4, dataset=None, args=None):
		super(PolicyManager_BatchPretrain, self).__init__(number_policies, dataset, args)
		self.blah = 0

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((self.args.batch_size,1,self.output_size)),sample_action_seq],axis=1)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((self.args.batch_size,1,self.output_size))],axis=1)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)
		
	def get_batch_element(self, i):

		# Make data_element a list of dictionaries. 
		data_element = []
						
		# for b in range(min(self.args.batch_size, len(self.index_list) - i)):
		# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
		for b in range(self.args.batch_size):

			# print("Index that the get_batch_element is using: b:",b," i+b: ",i+b, self.index_list[i+b])
			# Because of the new creation of index_list in random shuffling, this should be safe to index dataset with.

			# print("Getting data element, b: ", b, "i+b ", i+b, "index_list[i+b]: ", self.index_list[i+b])
			index = self.index_list[i+b]

			if self.args.train:
				self.coverage[index] += 1
			data_element.append(self.dataset[index])

		return data_element

	def get_trajectory_segment(self, i):
	
		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext','DeterGoal']:
			
			# Sample trajectory segment from dataset. 
			sample_traj, sample_action_seq = self.dataset[i:i+self.args.batch_size]
			
			# print("Getting data points from: ",i, " to: ", i+self.args.batch_size)			

			# Subsample trajectory segment. 		
			start_timepoint = np.random.randint(0,self.args.traj_length-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[:, start_timepoint:end_timepoint]	
			sample_action_seq = sample_action_seq[:, start_timepoint:end_timepoint-1]

			self.current_traj_len = self.traj_length

			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

			return concatenated_traj.transpose((1,0,2)), sample_action_seq.transpose((1,0,2)), sample_traj.transpose((1,0,2))
				
		elif self.args.data in global_dataset_list:

			if self.args.data in ['MIME','OldMIME'] or self.args.data=='Mocap':
				# data_element = self.dataset[i:i+self.args.batch_size]
				data_element = self.dataset[self.index_list[i:i+self.args.batch_size]]				
			else:
				data_element = self.get_batch_element(i)

			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length            
			
			batch_trajectory = np.zeros((self.args.batch_size, self.current_traj_len, self.state_size))
			self.subsampled_relative_object_state = np.zeros((self.args.batch_size, self.current_traj_len, self.args.env_state_size))

			# POTENTIAL:
			# for x in range(min(self.args.batch_size, len(self.index_list) - 1)):

			# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
			for x in range(self.args.batch_size):
			

				# Select the trajectory for each instance in the batch. 
				if self.args.ee_trajectories:
					traj = data_element[x]['endeffector_trajectory']
				else:
					traj = data_element[x]['demo']

				# Pick start and end.               

				# Sample random start point.
				if traj.shape[0]>self.current_traj_len:

					bias_length = int(self.args.pretrain_bias_sampling*traj.shape[0])

					# Probability with which to sample biased segment: 
					sample_biased_segment = np.random.binomial(1,p=self.args.pretrain_bias_sampling_prob)

					# If we want to bias sampling of trajectory segments towards the middle of the trajectory, to increase proportion of trajectory segments
					# that are performing motions apart from reaching and returning. 

					# Sample a biased segment if trajectory length is sufficient, and based on probability of sampling.
					if ((traj.shape[0]-2*bias_length)>self.current_traj_len) and sample_biased_segment:		
						start_timepoint = np.random.randint(bias_length, traj.shape[0] - self.current_traj_len - bias_length)
					else:
						start_timepoint = np.random.randint(0,traj.shape[0]-self.current_traj_len)

					end_timepoint = start_timepoint + self.current_traj_len


					if self.args.ee_trajectories:
						batch_trajectory[x] = data_element[x]['endeffector_trajectory'][start_timepoint:end_timepoint]
					else:
						batch_trajectory[x] = data_element[x]['demo'][start_timepoint:end_timepoint]
					
					if not(self.args.gripper):
						if self.args.ee_trajectories:
							batch_trajectory[x] = data_element['endeffector_trajectory'][start_timepoint:end_timepoint,:-1]
						else:
							batch_trajectory[x] = data_element['demo'][start_timepoint:end_timepoint,:-1]

					if self.args.data in ['RealWorldRigid', 'RealWorldRigidJEEF']:

						# Truncate the images to start and end timepoint. 
						data_element[x]['subsampled_images'] = data_element[x]['images'][start_timepoint:end_timepoint]

					if self.args.data in ['RealWorldRigidJEEF']:
						self.subsampled_relative_object_state[x] = data_element[x]['relative-object-state'][start_timepoint:end_timepoint]

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value

				if self.args.data not in ['NDAX','NDAXMotorAngles']:
					self.normalized_subsampled_relative_object_state = (self.subsampled_relative_object_state - self.norm_sub_value[-self.args.env_state_size:])/self.norm_denom_value[-self.args.env_state_size:]

			# Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)
			if self.args.data not in ['NDAX','NDAXMotorAngles']:
				self.relative_object_state_actions = np.diff(self.normalized_subsampled_relative_object_state, axis=1)

			# Concatenate
			concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)

			# Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			# return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2))
			return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2)), data_element

	def relabel_relative_object_state_actions(self, padded_action_seq):

		# Here, remove the actions computed from the absolute object states; 
		# Instead relabel the actions in these dimensions into actions computed from the relative state to EEF.. 

		relabelled_action_sequence = padded_action_seq
		# Relabel the action size computes.. 
		# relabelled_action_sequence[..., self.args.robot_state_size:] = self.relative_object_state_actions
		relabelled_action_sequence[..., -self.args.env_state_size:] = self.relative_object_state_actions

		return relabelled_action_sequence

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		# Now assemble inputs for subpolicy.
		
		# Create subpolicy inputs tensor. 			
		subpolicy_inputs = torch.zeros((input_trajectory.shape[0], self.args.batch_size, self.input_size+self.latent_z_dimensionality)).to(device)

		# Mask input trajectory according to subpolicy dropout. 
		self.subpolicy_input_dropout_layer = torch.nn.Dropout(self.args.subpolicy_input_dropout)

		torch_input_trajectory = torch.tensor(input_trajectory).view(input_trajectory.shape[0],self.args.batch_size,self.input_size).to(device).float()
		masked_input_trajectory = self.subpolicy_input_dropout_layer(torch_input_trajectory)

		# Now copy over trajectory. 
		# subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()         
		subpolicy_inputs[:,:,:self.input_size] = masked_input_trajectory

		# Now copy over latent z's. 
		subpolicy_inputs[range(input_trajectory.shape[0]),:,self.input_size:] = latent_z_indices

		# # Concatenated action sequence for policy network's forward / logprobabilities function. 
		# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
		# View time first and batch second for downstream LSTM.
		padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.args.batch_size,self.output_size))],axis=0)

		if self.args.data in ['RealRobotRigidJEEF']:
			padded_action_seq = self.relabel_relative_object_state_actions(padded_action_seq)

		return None, subpolicy_inputs, padded_action_seq

	def construct_dummy_latents(self, latent_z):

		if not(self.args.discrete_z):
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.args.batch_size, self.current_traj_len)).to(device).float()
		latent_b[:,0] = 1.

		return latent_z_indices, latent_b	
		# return latent_z_indices

	def setup_vectorized_environments(self):

		# Don't try to recreate env here, just use the env from the visualizer. 
		self.base_env = self.visualizer.env
		# self.vectorized_environments = gym.vector.SyncVectorEnv([ lambda: GymWrapper(robosuite.make("Door", robots=['Sawyer'], has_renderer=False)) for k in range(self.args.batch_size)])

		if not(isinstance(self.base_env, gym.Env)):
			self.base_env = GymWrapper(self.base_env)
		
		# Vectorized.. 		
		self.vectorized_environment = gym.vector.SyncVectorEnv([ lambda: self.base_env for k in range(self.args.batch_size)])

	def batch_compute_next_state(self, current_state, action):

		# Reset. 
		# Set state.
		# Preprocess action. 
		# Step. 
		# Return state. 

		if self.args.viz_sim_rollout:
			
			####################
			# (0) Reset envs. 
			####################

			self.vectorized_environment.reset()

			####################
			# (1) Set state. 		
			####################

			# Option 1 - do this iteratively - not ideal, but probably fine because this is not the bottleneck. 
			# Option 2 - set using set_attr? - testing this out doesn't seem to work? Anyhow, set_attr iterates internally, so might as well do this ourselves. 
			
			# for k in range(self.args.batch_size):
			for k, environment in enumerate(self.vectorized_environment.envs):
				self.visualizer.set_joint_pose(current_state, env=environment)
			
			####################
			# (2) Preprocess action.
			####################

			action_to_execute = self.preprocess_action(action)

			####################
			# (3) Step
			####################
		
			for k in range(self.args.sim_viz_step_repetition):
				# Use environment to take step.
				env_next_state_dict, _, _, _ = self.visualizer.environment.step(action_to_execute)
				gripper_state = env_next_state_dict[self.visualizer.gripper_key]
				if self.visualizer.new_robosuite:
					joint_state = self.visualizer.environment.sim.get_state()[1][:7]
				else:
					joint_state = env_next_state_dict['joint_pos']

		####################
		# (4) Return State
		####################
			
		else:
			next_state = current_state + action

		return next_state

	def differentiable_rollout(self, trajectory_start, latent_z, rollout_length=None):

		subpolicy_inputs = torch.zeros((self.args.batch_size,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[:,:self.state_dim] = torch.tensor(trajectory_start).to(device).float()
		# subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()
		subpolicy_inputs[:,2*self.state_dim:] = latent_z[0]

		if self.args.batch_size>1:
			subpolicy_inputs = subpolicy_inputs.unsqueeze(0)

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = self.rollout_timesteps-1

		for t in range(length):

			# Get actions from the policy.
			actions = self.policy_network.reparameterized_get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor
			
			# Compute next state. 
			new_state = self.batch_compute_next_state(subpolicy_inputs[t,...,:self.state_dim], action_to_execute)
			# new_state = subpolicy_inputs[t,...,:self.state_dim]+action_to_execute

			# Create new input row. 
			input_row = torch.zeros((self.args.batch_size, 2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
			input_row[:,:self.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[:,self.state_dim:2*self.state_dim] = actions[-1].squeeze(1)
			input_row[:,2*self.state_dim:] = latent_z[t+1]

			# Now that we have assembled the new input row, concatenate it along temporal dimension with previous inputs. 
			if self.args.batch_size>1:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row.unsqueeze(0)],dim=0)
			else:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[...,:self.state_dim].detach().cpu().numpy()
		differentiable_trajectory = subpolicy_inputs[...,:self.state_dim]
		differentiable_action_seq = subpolicy_inputs[...,self.state_dim:2*self.state_dim]
		differentiable_state_action_seq = subpolicy_inputs[...,:2*self.state_dim]

		# For differentiabiity, return tuple of trajectory, actions, state actions, and subpolicy_inputs. 
		return differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs
	
	def batched_visualize_robot_data(self, load_sets=False, number_of_trajectories_to_visualize=None):

		#####################
		# Set number of trajectories to visualize.			
		#####################

		if number_of_trajectories_to_visualize is not None:
			self.N = number_of_trajectories_to_visualize
		else:
			self.N = 400
			# self.N = 100	
			
		#####################
		# Set visualizer based on data / domain. 
		#####################

		self.set_visualizer_object()
		np.random.seed(seed=self.args.seed)

		#####################################################
		# Get latent z sets.
		#####################################################
		
		if not(load_sets):

			#####################################################
			# Select Z indices if necessary.
			#####################################################

			if self.args.split_stream_encoder:
				if self.args.embedding_visualization_stream == 'robot':
					stream_z_indices = np.arange(0,int(self.args.z_dimensions/2))
				elif self.args.embedding_visualization_stream == 'env':
					stream_z_indices = np.arange(int(self.args.z_dimensions/2),self.args.z_dimensions)
				else:
					stream_z_indices = np.arange(0,self.args.z_dimensions)	
			else:
				stream_z_indices = np.arange(0,self.args.z_dimensions)

			#####################################################
			# Initialize variables.
			#####################################################

			# self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
			self.latent_z_set = np.zeros((self.N,len(stream_z_indices)))		
			# These are lists because they're variable length individually.
			self.indices = []
			self.trajectory_set = []
			self.trajectory_rollout_set = []		
			self.rollout_gif_list = []
			self.gt_gif_list = []

			#####################################################
			# Create folder for gifs.
			#####################################################

			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
			# Create save directory:
			upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

			if not(os.path.isdir(upper_dir_name)):
				os.mkdir(upper_dir_name)

			self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
			if not(os.path.isdir(self.dir_name)):
				os.mkdir(self.dir_name)

			self.max_len = 0

			#####################################################
			# Initialize variables.
			#####################################################

			self.shuffle(len(self.dataset)-self.test_set_size, shuffle=True)
		
			#############################
			# For appropriate number of batches: 
			#############################

			for j in range(self.N//self.args.batch_size):
			
				#############################		
				# (1) Encode trajectory. 
				#############################

				if self.args.setting in ['learntsub','joint', 'queryjoint']:
					print("Embed in viz robot data")
					
					input_dict, var_dict, eval_dict = self.run_iteration(0, j, return_dicts=True, train=False)
					latent_z = var_dict['latent_z_indices']
					sample_trajs = input_dict['sample_traj']
				else:
					print("Running iteration of segment in viz")
					latent_z, sample_trajs, _, data_element = self.run_iteration(0, j, return_z=True, and_train=False)

				#############################
				# (2) 
				#############################

				# Create env for batch.
				self.per_batch_env_management(data_element[0])

				#############################
				# (3) Rollout for each trajectory in batch.
				#############################

				trajectory_rollout, _, _, _ = self.differentiable_rollout(sample_trajs[0], latent_z)

				# Need to add some stuff here to mimic get_robot_visuals' list management.

				if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
					unnorm_gt_trajectory = (sample_trajs*self.norm_denom_value)+self.norm_sub_value
					unnorm_pred_trajectory = (trajectory_rollout*self.norm_denom_value) + self.norm_sub_value
				else:
					unnorm_gt_trajectory = sample_trajs
					unnorm_pred_trajectory = trajectory_rollout

				#############################
				# (4) Visualize for every trajectory in batch. 
				#############################

				for b in range(self.args.batch_size):

					# First visualize ground truth gif. 					
					self.ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_GT.gif".format(i), return_and_save=True, end_effector=self.args.ee_trajectories, task_id=env_name)
					
					# Now visualize rollout. 
					

					# Copy it to global lists.
					self.gt_gif_list.append(copy.deepcopy(self.ground_truth_gif))
					self.rollout_gif_list.append(copy.deepcopy(self.rollout_gif))


			# Get MIME embedding for rollout and GT trajectories, with same Z embedding. 
			embedded_z = self.get_robot_embedding()

		# Save animations.
		gt_animation_object = self.visualize_robot_embedding(embedded_z, gt=True)
		rollout_animation_object = self.visualize_robot_embedding(embedded_z, gt=False)

		# Save webpage
		self.write_results_HTML()
		# Save webpage plots
		self.write_results_HTML('Plot')

		self.write_embedding_HTML(gt_animation_object,prefix="GT")
		self.write_embedding_HTML(rollout_animation_object,prefix="Rollout")

class PolicyManager_Joint(PolicyManager_BaseClass):

	# Basic Training Algorithm: 
	# For E epochs:
	# 	# For all trajectories:
	#		# Sample latent variables from conditional. 
	# 			# (Concatenate Latent Variables into Input.)
	# 		# Evaluate log likelihoods of actions and options. 
	# 		# Update parameters. 

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_Joint, self).__init__()

		self.args = args
		self.data = self.args.data
		self.number_policies = 4
		self.latent_z_dimensionality = self.args.z_dimensions
		self.dataset = dataset

		# Global input size: trajectory at every step - x,y,action
		# Inputs is now states and actions.
		# Model size parameters
		self.state_size = 2
		self.state_dim = 2
		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = 2					
		self.number_layers = self.args.number_layers
		self.traj_length = 5
		self.conditional_info_size = 6
		self.test_set_size = 50

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.conditional_info_size = self.args.condition_size
			self.conditional_viz_env = False

		if self.args.data in ['MIME','OldMIME']:
			self.state_size = 16	
			self.state_dim = 16
			self.test_set_size = 50
			self.input_size = 2*self.state_size
			self.output_size = self.state_size			
			self.traj_length = self.args.traj_length

			# Create Baxter visualizer for MIME data

			if not(self.args.no_mujoco):
				# self.visualizer = BaxterVisualizer.MujocoVisualizer()
				self.visualizer = BaxterVisualizer(args=self.args)

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/MIME/MIME_Orig_Mean.npy")
				self.norm_denom_value = np.load("Statistics/MIME/MIME_Orig_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/MIME/MIME_Orig_Min.npy")
				self.norm_denom_value = np.load("Statistics/MIME/MIME_Orig_Max.npy") - self.norm_sub_value

			# Max of robot_state + object_state sizes across all Baxter environments. 			
			self.cond_robot_state_size = 60
			self.cond_object_state_size = 25
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size
			self.conditional_viz_env = False

		elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
			self.state_size = 8	
			self.state_dim = 8
			self.input_size = 2*self.state_size	
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length

			if self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']:
				stat_dir_name = "Roboturk"
				self.conditional_viz_env = True
				self.visualizer = SawyerVisualizer()
			elif self.args.data in ['RoboMimic','OrigRoboMimic']:
				stat_dir_name = "Robomimic"
				self.conditional_viz_env = False
				self.test_set_size = 50
				self.visualizer = FrankaVisualizer()

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			# Max of robot_state + object_state sizes across all sawyer environments. 
			# Robot size always 30. Max object state size is... 23. 
			self.cond_robot_state_size = 30
			self.cond_object_state_size = 23
			self.number_tasks = 8
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size+self.number_tasks
			

		elif self.args.data=='Mocap':
			self.state_size = 22*3
			self.state_dim = 22*3	
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length	
			self.conditional_info_size = 0
			self.conditional_information = None
			self.conditional_viz_env = False

			# Create visualizer object
			self.visualizer = MocapVisualizer(args=self.args)

		elif self.args.data in ['GRAB']:
			
			self.state_size = 24
			self.state_dim = 24
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = GRABVisualizer()		

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['GRABHand']:
			
			self.state_size = 120
			self.state_dim = 120

			if self.args.single_hand in ['left', 'right']:
				self.state_dim //= 2
				self.state_size //= 2
			
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = GRABHandVisualizer(args=self.args)		

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['GRABArmHand']:
			

			self.state_size = 144
			self.state_dim = 144
			if self.args.single_hand in ['left', 'right']:
				self.state_size //= 2
				self.state_dim //= 2

			if self.args.position_normalization != 'pelvis':
				self.state_size += 3
				self.state_dim += 3
			
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = GRABArmHandVisualizer(args=self.args)		

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['GRABArmHandObject']:
			
			self.state_size = 96
			self.state_dim = 96
		
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = GRABArmHandVisualizer(args=self.args)		

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['GRABObject']:
			
			self.state_size = 6
			self.state_dim = 6
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = GRABArmHandVisualizer(args=self.args)		

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
			
		elif self.args.data in ['DAPG']:
			
			self.state_size = 51
			self.state_dim = 51
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = DAPGVisualizer(args=self.args)

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
		
		elif self.args.data in ['DAPGHand']:
			
			self.state_size = 30
			self.state_dim = 30
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = DAPGVisualizer(args=self.args)

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1

		elif self.args.data in ['DAPGObject']:
			
			self.state_size = 21
			self.state_dim = 21
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = DAPGVisualizer(args=self.args)

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1

		elif self.args.data in ['DexMV']:
			
			self.state_size = 43
			self.state_dim = 43
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = DexMVVisualizer(args=self.args)

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

		elif self.args.data in ['DexMVHand']:
			
			self.state_size = 30
			self.state_dim = 30
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = DexMVVisualizer(args=self.args)

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

		elif self.args.data in ['DexMVObject']:
			
			self.state_size = 13
			self.state_dim = 13
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data
			self.conditional_information = None
			self.conditional_viz_env = False	

			self.visualizer = DexMVVisualizer(args=self.args)

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]
			

		elif self.args.data=='RoboturkObjects':
			# self.state_size = 14
			# self.state_dim = 14

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 7
			self.state_dim = 7

			self.input_size = 2*self.state_size	
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length
			
			# self.args.data in ['RoboturkObjects']:
			self.visualizer = RoboturkObjectVisualizer(args=self.args)

		elif self.args.data=='RoboturkRobotObjects':

			# Set state size to 14 for now; because we're not using the relative pose.
			self.state_size = 15
			self.state_dim = 15

			self.input_size = 2*self.state_size	
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length
			
			# self.args.data in ['RoboturkObjects']:
			self.visualizer = RoboturkRobotObjectVisualizer(args=self.args)

		elif self.args.data in ['RealWorldRigid']:

			self.state_size = 21
			self.state_dim = 21
			self.input_size = 2*self.state_size

			#########################################
			# Manually scale.
			#########################################
			
			stat_dir_name = self.dataset.stat_dir_name
			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			if self.args.normalization is not None:
				# self.norm_sub_value will remain unmodified. 
				# self.norm_denom_value will get divided by scale.
				self.norm_denom_value /= self.args.state_scale_factor
				# Manually make sure quaternion dims are unscaled.
				self.norm_denom_value[10:14] = 1.
				self.norm_denom_value[17:] = 1.
				self.norm_sub_value[10:14] = 0.
				self.norm_sub_value[17:] = 0.

			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length
			self.conditional_info_size = 0
			self.test_set_size = 0

		self.training_phase_size = self.args.training_phase_size
		self.number_epochs = self.args.epochs
		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self.max_viz_trajs = self.args.max_viz_trajs

		self. learning_rate = self.args.learning_rate

		self.latent_b_loss_weight = self.args.lat_b_wt
		self.latent_z_loss_weight = self.args.lat_z_wt

		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_epochs = self.args.epsilon_over
		self.decay_counter = self.decay_epochs*len(self.dataset)

		# Log-likelihood penalty.
		self.lambda_likelihood_penalty = self.args.likelihood_penalty
		self.baseline = None

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)
		self.extent = len(self.dataset)
		self.already_shuffled = 0

	def create_networks(self):
		if self.args.discrete_z:
			
			# Create K Policy Networks. 
			# This policy network automatically manages input size. 
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.number_policies, self.number_layers).to(device)	
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)		

			# Create latent policy, whose action space = self.number_policies. 
			# This policy network automatically manages input size. 

			# Also add conditional_info_size to this. 
			self.latent_policy = LatentPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.number_layers, self.args.b_exploration_bias).to(device)

			# Create variational network. 
			# self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, number_layers=self.number_layers, z_exploration_bias=self.args.z_exploration_bias, b_exploration_bias=self.args.b_exploration_bias).to(device)
			self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.args, number_layers=self.number_layers).to(device)

		else:
			
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.latent_z_dimensionality, self.number_layers).to(device)
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

			if self.args.constrained_b_prior:
				self.latent_policy = ContinuousLatentPolicyNetwork_ConstrainedBPrior(self.input_size+self.conditional_info_size, self.hidden_size, self.args, self.number_layers).to(device)
				
				self.variational_policy = ContinuousVariationalPolicyNetwork_ConstrainedBPrior(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.number_layers).to(device)

			else:
				# self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.number_layers, self.args.b_exploration_bias).to(device)
				self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size+self.conditional_info_size, self.hidden_size, self.args, self.number_layers).to(device)

				self.variational_policy = ContinuousVariationalPolicyNetwork_BPrior(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.number_layers).to(device)

	def create_training_ops(self):
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		
		# If we are using reparameterization, use a global optimizer, and a global loss function. 
		# This means gradients are being handled properly. 
		self.parameter_list = list(self.latent_policy.parameters()) + list(self.variational_policy.parameters())
		if not(self.args.fix_subpolicy):
			self.parameter_list = self.parameter_list + list(self.policy_network.parameters())
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)
		self.save_object = {}
		self.save_object['Latent_Policy'] = self.latent_policy.state_dict()
		self.save_object['Policy_Network'] = self.policy_network.state_dict()
		self.save_object['Encoder_Network'] = self.variational_policy.state_dict()
		# self.save_object['Variational_Policy'] = self.variational_policy.state_dict()
		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path, just_subpolicy=False):
		self.load_object = torch.load(path)
		self.policy_network.load_state_dict(self.load_object['Policy_Network'])

		if not(just_subpolicy):
			if self.args.load_latent:
				self.latent_policy.load_state_dict(self.load_object['Latent_Policy'])		
				
			# self.variational_policy.load_state_dict(self.load_object['Variational_Policy'])
			self.variational_policy.load_state_dict(self.load_object['Encoder_Network'])

	def load_model_from_transfer(self):

		if self.args.source_model is not None:
			self.load_object = torch.load(self.args.source_model)
			self.policy_network.load_state_dict(self.load_object['Source_Policy_Network'])
			self.variational_policy.load_state_dict(self.load_object['Source_Encoder_Network'])
		elif self.args.target_model is not None:
			self.load_object = torch.load(self.args.target_model)	
			self.policy_network.load_state_dict(self.load_object['Target_Policy_Network'])
			self.variational_policy.load_state_dict(self.load_object['Target_Encoder_Network'])

	def set_epoch(self, counter):
		if self.args.train:
			if counter<self.decay_counter:
				self.epsilon = self.initial_epsilon-self.decay_rate*counter
			else:
				self.epsilon = self.final_epsilon		
		
			if counter<self.training_phase_size:
				self.training_phase=1

				# Set this variable to 0, and then the first time we encounter training phase 2, we change it to 1.
				self.reset_subpolicy_training = 0

			elif self.training_phase_size<=counter and counter<2*self.training_phase_size:
				self.training_phase=2	
				# print("In Phase 2.")		

				# If we are encountering training phase 2 for the first time.
				# if self.reset_subpolicy_training==0 and self.args.setting=='context':
				if self.reset_subpolicy_training==0 and self.args.reset_training:
					self.reset_subpolicy_training = 1
					
					# Instead of recreating the optimizer, we can also add the policy network parameters to the optimizer's paramters. 
					# self.optimizer.add_param_group({'params': self.policy_network.parameters()})

					# Now set args.fix_subpolicy to False, and then recreate optimizer; this will add the policy network parameters to optimizer.
					self.args.fix_subpolicy = 0

					# Initially we weren't recreating the optimizer, but now we are going to try it, and set the learning rate smaller. 
					self.learning_rate = self.args.transfer_learning_rate
					self.create_training_ops()

					# One issue is this resets variational network optimizer parameters, but that's okay.
					# The add param group option didn't do this, but now we are going to recreate the optimizer with a smaller learning rate.

			else:		
				self.training_phase=3
				self.latent_z_loss_weight = 0.01*self.args.lat_b_wt
				# For training phase = 3, set latent_b_loss weight to 1 and latent_z_loss weight to something like 0.1 or 0.01. 
				# After another double training_phase... (i.e. counter>3*self.training_phase_size), 
				# This should be run when counter > 2*self.training_phase_size, and less than 3*self.training_phase_size.
				if counter>3*self.training_phase_size:
					# Set equal after 3. 
					# print("In Phase 4.")
					self.latent_z_loss_weight = 0.1*self.args.lat_b_wt
				else:
					# print("In Phase 3.")
					pass

		else:
			self.epsilon = 0.
			self.training_phase=1

	def visualize_trajectory(self, trajectory, segmentations=None, i=0, suffix='_Img'):

		if self.args.data in global_dataset_list:

			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				unnorm_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			else:
				unnorm_trajectory = trajectory

			if self.args.data=='Mocap':
				# Create save directory:
				upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

				if not(os.path.isdir(upper_dir_name)):
					os.mkdir(upper_dir_name)

				if self.args.model is not None:
					model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
				else:
					model_epoch = self.current_epoch_running

				self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
				if not(os.path.isdir(self.dir_name)):
					os.mkdir(self.dir_name)

				animation_object = self.dataset[i]['animation']

				return self.visualizer.visualize_joint_trajectory(unnorm_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_{1}.gif".format(i,suffix), return_and_save=True, additional_info=animation_object, end_effector=self.args.ee_trajectories)
			else:
				if not(self.args.no_mujoco):
					return self.visualizer.visualize_joint_trajectory(unnorm_trajectory, return_gif=True, segmentations=segmentations, end_effector=self.args.ee_trajectories)
		else:
			return self.visualize_2D_trajectory(trajectory)

	def visualize_2D_trajectory(self, traj):

		fig = plt.figure()		
		ax = fig.gca()
		ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),cmap='jet')

		scale = 30
		plt.xlim(-scale,scale)
		plt.ylim(-scale,scale)

		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

		# Already got image data. Now close plot so it doesn't cry.
		# fig.gcf()
		plt.close()

		image = np.transpose(image, axes=[2,0,1])

		return image

	def compute_evaluation_metrics(self, sample_traj, counter, i):

		# # Generate trajectory rollouts so we can calculate distance metric. 
		# self.rollout_visuals(counter, i, get_image=False)

		if self.args.batch_size>1:
			reference_traj = sample_traj[:self.batch_trajectory_lengths[self.selected_index], self.selected_index]
		else:
			reference_traj = sample_traj

		# Compute trajectory distance between:		
		var_rollout_distance = ((self.variational_trajectory_rollout-reference_traj)**2).mean()
		latent_rollout_distance = 0.
		if self.args.viz_latent_rollout:
			latent_rollout_distance = ((self.latent_trajectory_rollout-reference_traj)**2).mean()

		return var_rollout_distance, latent_rollout_distance
	
	def update_plots(self, counter, i, input_dictionary, variational_dict, eval_likelihood_dict):
	
		# Parse dictionaries: 
		sample_traj = input_dictionary['sample_traj']
		kl_divergence = variational_dict['kl_divergence']
		prior_loglikelihood = variational_dict['prior_loglikelihood']
		subpolicy_loglikelihood = eval_likelihood_dict['learnt_subpolicy_loglikelihood']
		latent_loglikelihood = eval_likelihood_dict['latent_loglikelihood']
		latent_z_logprobability = eval_likelihood_dict['latent_z_logprobability']
		latent_b_logprobability = eval_likelihood_dict['latent_b_logprobability']
		
		# Create wandb log directory. 
		log_dict = {'Latent Policy Loss': torch.mean(self.total_latent_loss),
					'SubPolicy Log Likelihood': subpolicy_loglikelihood.mean(),
					'Latent Log Likelihood': latent_loglikelihood.mean(),
					'Variational Policy Loss': torch.mean(self.variational_loss),
					'Variational Reinforce Loss': torch.mean(self.reinforce_variational_loss),
					'Total Variational Policy Loss': torch.mean(self.total_variational_loss),
					'Baseline': self.baseline.mean(),
					'Total Likelihood': subpolicy_loglikelihood+latent_loglikelihood,
					'Epsilon': self.epsilon,
					'Latent Z LogProbability': latent_z_logprobability,
					'Latent B LogProbability': latent_b_logprobability,
					'KL Divergence': torch.mean(kl_divergence),
					'Prior LogLikelihood': torch.mean(prior_loglikelihood),
					'Epoch': self.current_epoch_running,
					'Latent Z Mean': torch.mean(variational_dict['latent_z_indices']),
					'Training Phase': self.training_phase} 

		if counter%self.args.display_freq==0:
			# Now adding visuals for MIME, so it doesn't depend what data we use.
			if self.args.batch_size>1:
				variational_rollout_image, latent_rollout_image = self.rollout_visuals(counter, i, variational_dict)
			else: 
				variational_rollout_image, latent_rollout_image = self.rollout_visuals(counter, i)

			# Compute distance metrics. 
			var_dist, latent_dist = self.compute_evaluation_metrics(sample_traj, counter, i)
			log_dict['Variational Trajectory Distance'] = var_dist
			log_dict['Latent Trajectory Distance'] = latent_dist
			
			if self.args.batch_size>1:
				gt_trajectory_image = np.array(self.visualize_trajectory(sample_traj[:,0,:], i=i, suffix='GT'))
			else:
				gt_trajectory_image = np.array(self.visualize_trajectory(sample_traj, i=i, suffix='GT'))

			variational_rollout_image = np.array(variational_rollout_image)
			latent_rollout_image = np.array(latent_rollout_image)
			
			if self.args.data in global_dataset_list:

				# Feeding as list of image because gif_summary.				

				# print("Embedding in joint update plots, L2511 ")
				# embed()

				log_dict['GT Trajectory'] = self.return_wandb_gif(gt_trajectory_image)
				log_dict['Variational Rollout'] = self.return_wandb_gif(variational_rollout_image)
				if self.args.viz_latent_rollout:
					log_dict['Latent Rollout'] = self.return_wandb_gif(latent_rollout_image)				
			else:
				# Feeding as list of image because gif_summary.
				log_dict['GT Trajectory'] = self.return_wandb_image(gt_trajectory_image)
				log_dict['Variational Rollout'] = self.return_wandb_image(variational_rollout_image)
				if self.args.viz_latent_rollout:
					log_dict['Latent Rollout'] = self.return_wandb_image(latent_rollout_image)

		# Now actually log things in wandb.
		wandb.log(log_dict, step=counter)

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq, conditional_information=None):

		if self.args.discrete_z:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # This method of concatenation is wrong, because it evaluates likelihood of action [0,0] as well. 
			# # Concatenated action sqeuence for policy network. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			# This is the right method of concatenation, because it evaluates likelihood 			
			padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:

			################
			# This seems like it could be a major issue in the contextual embedding training. 
			# If you made this a copy, and added parameters of the subpolicy to the optimizer... 
			# Stops gradients going back into the variational network from this. 
			################
			# if self.training_phase>1:
			# 	# Prevents gradients being propagated through this..
			# 	latent_z_copy = torch.tensor(latent_z_indices).to(device)
			# else:
			# 	latent_z_copy = latent_z_indices

			# INSTEAD, just try the latent_z_copy. 
			latent_z_copy = latent_z_indices

			if conditional_information is None:
				conditional_information = torch.zeros((self.conditional_info_size)).to(device).float()

			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 			
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1+self.conditional_info_size)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()			
			assembled_inputs[range(1,len(input_trajectory)),self.input_size:self.input_size+self.latent_z_dimensionality] = latent_z_copy[:-1]
			
			# We were writing the wrong dimension... should we be running again? :/ 			
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+self.latent_z_dimensionality] = latent_b[:-1].float().squeeze(1)
			# assembled_inputs[range(1,len(input_trajectory)),-self.conditional_info_size:] = torch.tensor(conditional_information).to(device).float()

			# Instead of feeding conditional infromation only from 1'st timestep onwards, we are going to st it from the first timestep. 
			if self.conditional_info_size>0:
				assembled_inputs[:,-self.conditional_info_size:] = torch.tensor(conditional_information).to(device).float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size:] = latent_z_indices

			# # This method of concatenation is wrong, because it evaluates likelihood of action [0,0] as well. 
			# # Concatenated action sqeuence for policy network. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			# This is the right method of concatenation, because it evaluates likelihood 			
			padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
	
			return assembled_inputs, subpolicy_inputs, padded_action_seq

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an-2, an-1
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to the END of action sequence and then concatenate.
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def differentiable_old_concat_state_action(self, sample_traj, sample_action_seq):
		# sample_action_seq = torch.cat([sample_action_seq[1:], torch.zeros((1,self.output_size)).to(device).float()],axis=0)
		sample_action_seq = torch.roll(sample_action_seq,-1,dims=0)
		return torch.cat([sample_traj, sample_action_seq],axis=-1)

	def setup_eval_against_encoder(self):
		# Creates a network, loads the network from pretraining model file. 
		self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).to(device)				
		load_object = torch.load(self.args.subpolicy_model)
		self.encoder_network.load_state_dict(load_object['Encoder_Network'])

		# Force encoder to use original variance for eval.
		self.encoder_network.variance_factor = 1.

	def evaluate_loglikelihoods(self, input_dictionary, variational_dict):
		
		###########################
		# Parse dictionaries. 
		###########################
		sample_traj = input_dictionary['sample_traj']
		sample_action_seq = input_dictionary['sample_action_seq']
		concatenated_traj = input_dictionary['concatenated_traj']
		latent_z_indices = variational_dict['latent_z_indices']
		latent_b = variational_dict['latent_b']

		##########################
		# Set batch mask.
		##########################

		# Set batch mask. For batch_size = 1, this is just an array of ones the length of the trajectory. 
		# For batch_size>1, this is an array of size B x Max length of batch size, with 1's for each batch element till that element's trajectory length. 
		# This way the update policies function is inherited for batch joint training. Only the set mask changes. 
		self.set_batch_mask()

		###########################
		# Initialize variables. 
		###########################

		# Initialize both loglikelihoods to 0. 
		subpolicy_loglikelihood = 0.
		latent_loglikelihood = 0.

		# Need to assemble inputs first - returns a Torch CUDA Tensor.
		# This doesn't need to take in actions, because we can evaluate for all actions then select. 
		assembled_inputs, subpolicy_inputs, padded_action_seq = self.assemble_inputs(concatenated_traj, latent_z_indices, latent_b, sample_action_seq, self.conditional_information)

		###########################
		# Compute learnt subpolicy loglikelihood.
		###########################
		unmasked_learnt_subpolicy_loglikelihoods, entropy = self.policy_network.forward(subpolicy_inputs, padded_action_seq)

		# Clip values. # Comment this out to remove clipping.
		unmasked_learnt_subpolicy_loglikelihoods = torch.clamp(unmasked_learnt_subpolicy_loglikelihoods,min=self.args.subpolicy_clamp_value)

		# Multiplying the likelihoods with the subpolicy ratio before summing.
		unmasked_learnt_subpolicy_loglikelihoods = self.args.subpolicy_ratio*unmasked_learnt_subpolicy_loglikelihoods

		# Averaging until penultimate timestep.
		learnt_subpolicy_loglikelihoods = self.batch_mask*unmasked_learnt_subpolicy_loglikelihoods
		# learnt_subpolicy_loglikelihood = learnt_subpolicy_loglikelihoods[:-1].mean()
		learnt_subpolicy_loglikelihood = learnt_subpolicy_loglikelihoods[:-1].sum()/(self.batch_mask[:-1].sum())

		###########################
		# Compute Latent policy loglikelihood values. 
		###########################		

		# Whether to clone assembled_inputs based on the phase of training. 
		# In phase one it doesn't matter if we use the clone or not, because we never use latent policy loss. 
		# So just clone anyway. 
		# For now, ignore phase 3. This prevents gradients from going into the variational policy from the latent policy.		
		assembled_inputs_copy = assembled_inputs.clone().detach()
		latent_z_copy = latent_z_indices.clone().detach()
		# Consideration for later:
		# if self.training_phase==3:
		# Don't clone.

		if self.args.discrete_z:
			# Return discrete probabilities from latent policy network. 
			latent_z_logprobabilities, latent_b_logprobabilities, latent_b_probabilities, latent_z_probabilities = self.latent_policy.forward(assembled_inputs_copy)
			# # Selects first option for variable = 1, second option for variable = 0. 
			
			# Use this to check if latent_z elements are equal: 
			diff_val = (1-(latent_z_indices==latent_z_indices.roll(1,0))[1:]).to(device).float()
			# We rolled latent_z, we didn't roll diff. This works because latent_b is always guaranteed to be 1 in the first timestep, so it doesn't matter what's in diff_val[0].
			diff_val = diff_val.roll(1,0)

			# Selects first option for variable = 1, second option for variable = 0. 
			latent_z_temporal_logprobabilities = torch.where(latent_b[:-1].byte(), latent_z_logprobabilities[range(len(sample_traj)-1),latent_z_indices[:-1].long()], -self.lambda_likelihood_penalty*diff_val)
			latent_z_logprobability = latent_z_temporal_logprobabilities.mean()

		else:
			# If not, we need to evaluate the latent probabilties of latent_z_indices under latent_policy. 			
			latent_b_logprobabilities, latent_b_probabilities, latent_distributions = self.latent_policy.forward(assembled_inputs_copy, self.epsilon)
			# Evalute loglikelihood of latent z vectors under the latent policy's distributions. 			

			if self.args.batch_size>1:
				latent_z_logprobabilities = latent_distributions.log_prob(latent_z_copy)				
			else:				
				latent_z_logprobabilities = latent_distributions.log_prob(latent_z_copy.unsqueeze(1))				

			# Multiply logprobabilities by the latent policy ratio.
			# First mask the latent_z_temporal_logprobs. 			
			unmasked_latent_z_temporal_logprobabilities = latent_z_logprobabilities[:-1]*self.args.latentpolicy_ratio
			latent_z_temporal_logprobabilities = self.batch_mask[:-1]*unmasked_latent_z_temporal_logprobabilities
			# latent_z_logprobability = latent_z_temporal_logprobabilities.mean()
			latent_z_logprobability = latent_z_temporal_logprobabilities.sum()/(self.batch_mask[:-1].sum())
			latent_z_probabilities = None			

		# LATENT LOGLIKELIHOOD is defined as: 
		# =	\sum_{t=1}^T \log p(\zeta_t | \tau_{1:t}, \zeta_{1:t-1})
		# = \sum_{t=1}^T \log { \phi_t(b_t)} + \log { 1[b_t==1] \eta_t(h_t|s_{1:t}) + 1[b_t==0] 1[z_t==z_{t-1}] } 

		# Adding log probabilities of termination (of whether it terminated or not), till penultimate step. 
		
		# if self.args.batch_size>1: 
		# 	unmasked_latent_b_temporal_logprobabilities = latent_b_logprobabilities.take(latent_b.long())[:-1]
		# else:
		# 	# unmasked_latent_b_temporal_logprobabilities = latent_b_logprobabilities[range(len(sample_traj)-1),latent_b[:-1].long()]
		# 	unmasked_latent_b_temporal_logprobabilities = latent_b_logprobabilities.take(latent_b.long())[:-1]
			
		unmasked_latent_b_temporal_logprobabilities = latent_b_logprobabilities.take(latent_b.long())[:-1]
		latent_b_temporal_logprobabilities = self.batch_mask[:-1]*unmasked_latent_b_temporal_logprobabilities
		# latent_b_logprobability = latent_b_temporal_logprobabilities.mean()
		latent_b_logprobability = latent_b_temporal_logprobabilities.sum()/(self.batch_mask[:-1].sum())
		latent_loglikelihood += latent_b_logprobability
		latent_loglikelihood += latent_z_logprobability

		# DON'T CLAMP, JUST MULTIPLY BY SUITABLE RATIO! Probably use the same lat_z_wt and lat_b_wt ratios from the losses. 
		latent_temporal_loglikelihoods = self.args.lat_b_wt*latent_b_temporal_logprobabilities + self.args.lat_z_wt*latent_z_temporal_logprobabilities.squeeze(1)

		##################################################
		#### Manage merging likelihoods for REINFORCE ####
		##################################################

		if self.training_phase==1: 
			temporal_loglikelihoods = learnt_subpolicy_loglikelihoods[:-1].squeeze(1)
		elif self.training_phase==2 or self.training_phase==3:
			# temporal_loglikelihoods = learnt_subpolicy_loglikelihoods[:-1].squeeze(1) + self.args.temporal_latentpolicy_ratio*latent_temporal_loglikelihoods
			temporal_loglikelihoods = learnt_subpolicy_loglikelihoods[:-1].squeeze(1)

		if self.args.debug:
			if self.iter%self.args.debug==0:
				print("Embedding in the Evaluate Likelihoods Function.")
				embed()

		# Parse return objects into dicitonary. 
		return_dict = {}
		return_dict['latent_loglikelihood'] = latent_loglikelihood
		return_dict['latent_b_logprobabilities'] = latent_b_logprobabilities
		return_dict['latent_z_logprobabilities'] = latent_z_logprobabilities
		return_dict['latent_b_probabilities'] = latent_b_probabilities
		return_dict['latent_z_probabilities'] = latent_z_probabilities
		return_dict['latent_z_logprobability'] = latent_z_logprobability
		return_dict['latent_b_logprobability'] = latent_b_logprobability
		return_dict['learnt_subpolicy_loglikelihood'] = learnt_subpolicy_loglikelihood
		return_dict['learnt_subpolicy_loglikelihoods'] = learnt_subpolicy_loglikelihoods
		return_dict['temporal_loglikelihoods'] = temporal_loglikelihoods
		return_dict['subpolicy_inputs'] = subpolicy_inputs

		return return_dict

	def set_batch_mask(self):

		# self.set_batch_mask = torch.ones((1, self.current_traj_len))
		# Remember, dimensions are time x batch.
		self.batch_mask = torch.ones((self.current_traj_len,1)).to(device).float()

	def update_policies(self, i, input_dictionary, variational_dict, eval_likelihood_dict):

		######################################################
		########### Initialize things for func, ##############
		######################################################

		# Set optimizer gradients to zero.
		self.optimizer.zero_grad()

		######################################################
		############### Parse dictionaries. ##################
		######################################################

		sample_action_seq = input_dictionary['sample_action_seq']

		latent_b = variational_dict['latent_b']*self.batch_mask
		latent_z_indices = variational_dict['latent_z_indices']*self.batch_mask.unsqueeze(2)
		variational_z_logprobabilities = variational_dict['variational_z_logprobabilities']*self.batch_mask
		variational_b_logprobabilities = variational_dict['variational_b_logprobabilities']*self.batch_mask.unsqueeze(2)
		variational_z_probabilities = variational_dict['variational_z_probabilities']
		variational_b_probabilities = variational_dict['variational_b_probabilities']*self.batch_mask.unsqueeze(2)
		kl_divergence = variational_dict['kl_divergence']*self.batch_mask
		prior_loglikelihood = variational_dict['prior_loglikelihood']*self.batch_mask
		
		latent_z_logprobabilities = eval_likelihood_dict['latent_z_logprobabilities']*self.batch_mask
		latent_b_logprobabilities = eval_likelihood_dict['latent_b_logprobabilities']*self.batch_mask.unsqueeze(2)
		latent_z_probabilities = eval_likelihood_dict['latent_z_probabilities']
		latent_b_probabilities = eval_likelihood_dict['latent_b_probabilities']*self.batch_mask.unsqueeze(2)
		learnt_subpolicy_loglikelihood = eval_likelihood_dict['learnt_subpolicy_loglikelihood']
		learnt_subpolicy_loglikelihoods = eval_likelihood_dict['learnt_subpolicy_loglikelihoods']*self.batch_mask
		latent_loglikelihood = eval_likelihood_dict['latent_loglikelihood']
		
		temporal_loglikelihoods = eval_likelihood_dict['temporal_loglikelihoods']*self.batch_mask[:-1]

		loglikelihood = (learnt_subpolicy_loglikelihood+latent_loglikelihood)*self.batch_mask

		# Assemble prior and KL divergence losses. 
		# Since these are output by the variational network, and we don't really need the last z predicted by it. 
		prior_loglikelihood = (self.batch_mask*prior_loglikelihood)[:-1]		
		kl_divergence = (self.batch_mask*kl_divergence)[:-1]

		######################################################
		############## Update latent policy. #################
		######################################################

		# Remember, an NLL loss function takes <Probabilities, Sampled Value> as arguments. 
		# embed()
		self.unmasked_latent_b_loss = self.negative_log_likelihood_loss_function(latent_b_logprobabilities.view(-1,2), latent_b.long().view(-1,)).view(-1,self.args.batch_size)
		self.latent_b_loss = self.batch_mask*self.unmasked_latent_b_loss
		# self.latent_b_loss = self.negative_log_likelihood_loss_function(latent_b_logprobabilities, latent_b.long())		

		if self.args.discrete_z:
			self.latent_z_loss = self.negative_log_likelihood_loss_function(latent_z_logprobabilities, latent_z_indices.long())
		# If continuous latent_z, just calculate loss as negative log likelihood of the latent_z's selected by variational network.
		else:
			self.unmasked_latent_z_loss = -latent_z_logprobabilities.squeeze(1)
			self.latent_z_loss = self.batch_mask*self.unmasked_latent_z_loss

		# Compute total latent loss as weighted sum of latent_b_loss and latent_z_loss.
		# Remember, no need to mask this with batch_mask, because it's just a sum of latent_b_loss and latent_z_loss, which are both already masked.
		self.total_latent_loss = (self.latent_b_loss_weight*self.latent_b_loss+self.latent_z_loss_weight*self.latent_z_loss)[:-1]

		#######################################################
		############# Compute Variational Losses ##############
		#######################################################

		# MUST ALWAYS COMPUTE: # Compute cross entropies. 
		# self.variational_b_loss = self.negative_log_likelihood_loss_function(variational_b_logprobabilities[:-1], latent_b[:-1].long())
		self.unmasked_variational_b_loss = self.negative_log_likelihood_loss_function(variational_b_logprobabilities.view(-1,2), latent_b.long().view(-1,)).view(-1,self.args.batch_size)
		self.variational_b_loss = (self.batch_mask*self.unmasked_variational_b_loss)[:-1]

		# In case of reparameterization, the variational loss that goes to REINFORCE should just be variational_b_loss.
		# Remember, no need to mask this with batch_mask, because it's just derived from variational_b_loss, which is already masked.
		self.variational_loss = self.args.var_loss_weight*self.variational_b_loss

		#######################################################
		########## Compute Variational Reinforce Loss #########
		#######################################################

		# Compute reinforce target based on how we express the objective:
		# The original implementation, i.e. the entropic implementation, uses:
		# (1) \mathbb{E}_{x, z \sim q(z|x)} \Big[ \nabla_{\omega} \log q(z|x,\omega) \{ \log p(x||z) + \log p(z||x) - \log q(z|x) - 1 \} \Big] 

		# The KL divergence implementation uses:
		# (2) \mathbb{E}_{x, z \sim q(z|x)} \Big[ \nabla_{\omega} \log q(z|x,\omega) \{ \log p(x||z) + \log p(z||x) - \log p(z) \} \Big] - \nabla_{\omega} D_{KL} \Big[ q(z|x) || p(z) \Big]

		# Compute baseline target according to NEW GRADIENT, and Equation (2) above. 

		baseline_target = (temporal_loglikelihoods - self.args.prior_weight*prior_loglikelihood).clone().detach()

		if self.baseline is None:
			self.baseline = torch.zeros_like(baseline_target.mean()).to(device).float()
		else:
			# self.baseline = (self.beta_decay*self.baseline)+(1.-self.beta_decay)*baseline_target.mean()
			self.baseline = (self.beta_decay*self.baseline)+(1.-self.beta_decay)*(baseline_target.sum()/(self.batch_mask[:-1].sum()))
			
		# Remember, no need to mask this with batch_mask, because it's just derived from temporal_loglikelihoods and variational_b_loss, which are already masked.
		self.reinforce_variational_loss = self.variational_loss*(baseline_target-self.baseline)

		# If reparam, the variational loss is a combination of three things. 
		# Losses from latent policy and subpolicy into variational network for the latent_z's, the reinforce loss on the latent_b's, and the KL divergence. 
		# But since we don't need to additional compute the gradients from latent and subpolicy into variational network, just set the variational loss to reinforce + KL.
		# self.total_variational_loss = (self.reinforce_variational_loss.sum() + self.args.kl_weight*kl_divergence.squeeze(1).sum()).sum()
		
		# self.total_variational_loss = (self.reinforce_variational_loss + self.args.kl_weight*kl_divergence.squeeze(1)).mean()
		self.total_variational_loss = (self.reinforce_variational_loss + self.kl_weight*kl_divergence.squeeze(1)).sum()/(self.batch_mask[:-1].sum())

		######################################################
		# Set other losses, subpolicy, latent, and prior.
		######################################################

		# Get subpolicy losses.
		self.subpolicy_loss = (-learnt_subpolicy_loglikelihood).mean()

		# Get prior losses. 
		# self.prior_loss = (-self.args.prior_weight*prior_loglikelihood).mean()
		self.prior_loss = (-self.args.prior_weight*prior_loglikelihood).sum()/(self.batch_mask[:-1].sum())

		# Reweight latent loss.
		# self.total_weighted_latent_loss = (self.args.latent_loss_weight*self.total_latent_loss).mean()
		self.total_weighted_latent_loss = (self.args.latent_loss_weight*self.total_latent_loss).sum()/(self.batch_mask[:-1].sum())

		################################################
		# Setting total loss based on phase of training.
		################################################


		# IF PHASE ONE: 
		if self.training_phase==1:
			self.total_loss = self.subpolicy_loss + self.total_variational_loss + self.prior_loss
		# IF DONE WITH PHASE ONE:
		elif self.training_phase==2 or self.training_phase==3:
			# Debugging joint training.
			self.total_loss = self.subpolicy_loss + self.total_variational_loss + self.prior_loss
			# self.total_loss = self.subpolicy_loss + self.total_weighted_latent_loss + self.total_variational_loss + self.prior_loss

		# ################################################
		# # If we're implementing context based training, add. 
		# ################################################
		# if self.args.setting=='context':
		# 	self.total_loss += self.context_loss

		################################################
		if self.args.debug:
			if self.iter%self.args.debug==0:
				print("Embedding in Update Policies")
				embed()
		################################################

		self.total_loss.sum().backward()
		self.optimizer.step()

	def set_env_conditional_info(self):

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.conditional_information = np.zeros((self.conditional_info_size))
		else:			
			obs = self.environment._get_observation()
			self.conditional_information = np.zeros((self.conditional_info_size))
			cond_state = np.concatenate([obs['robot-state'],obs['object-state']])
			self.conditional_information[:cond_state.shape[-1]] = cond_state
			# Also setting particular index in conditional information to 1 for task ID.
			self.conditional_information[-self.number_tasks+self.task_id_for_cond_info] = 1

	def take_rollout_step(self, subpolicy_input, t, use_env=False):

		# Feed subpolicy input into the policy. 
		actions = self.policy_network.get_actions(subpolicy_input,greedy=True,batch_size=1)
		
		# Select last action to execute. 
		action_to_execute = actions[-1].squeeze(1)

		if use_env==True:
			# Take a step in the environment. 
			step_res = self.environment.step(action_to_execute.squeeze(0).detach().cpu().numpy())
			# Get state. 
			observation = step_res[0]
			# Now update conditional information... 
			# self.conditional_information = np.concatenate([new_state['robot-state'],new_state['object-state']])

			gripper_open = np.array([0.0115, -0.0115])
			gripper_closed = np.array([-0.020833, 0.020833])

			# The state that we want is ... joint state?			
			# gripper_finger_values = step_res[0]['gripper_qpos']
			gripper_finger_values = step_res[0][self.visualizer.gripper_key]
			gripper_values = (gripper_finger_values - gripper_open)/(gripper_closed - gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = 2*finger_diff-1

			# Concatenate joint and gripper state. 			
			new_state_numpy = np.concatenate([observation['joint_pos'], np.array(gripper_value).reshape((1,))])
			new_state = torch.tensor(new_state_numpy).to(device).float().view((1,-1))

			# This should be true by default...
			if self.conditional_viz_env:
				self.set_env_conditional_info()
			
		else:
			# Compute next state by adding action to state. 
			new_state = subpolicy_input[t,:self.state_dim]+action_to_execute	

		# return new_subpolicy_input
		return action_to_execute, new_state

	def create_RL_environment_for_rollout(self, environment_name, state=None, task_id=None):

		import robosuite
		self.environment = robosuite.make(environment_name)
		self.task_id_for_cond_info = task_id
		if state is not None:
			self.environment.sim.set_state_from_flattened(state)

	def rollout_variational_network(self, counter, i, variational_dict=None):

		###########################################################
		###########################################################

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now.
		_ , sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)
				
		if self.args.batch_size>1: 
			self.selected_index = 0
			sample_action_seq = sample_action_seq[:,self.selected_index,:]
			concatenated_traj = concatenated_traj[:,self.selected_index,:]
			old_concatenated_traj = old_concatenated_traj[:,self.selected_index,:]
		
		# if self.args.traj_length>0:
		# 	self.rollout_timesteps = self.args.traj_length
		# else:
		# 	self.rollout_timesteps = self.batch_trajectory_lengths[self.selected_index]
		# 	# self.rollout_timesteps = len(concatenated_traj)		
		self.rollout_timesteps = self.batch_trajectory_lengths[self.selected_index]
		
		############# (1) #############
		# Sample latent variables from p(\zeta | \tau).

		if variational_dict is None:
			latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
			variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood = \
				self.variational_policy.forward(torch.tensor(old_concatenated_traj).to(device).float(), self.epsilon, batch_size=1)
		else:
			# Parse dictionary to skip execution. 
			

			latent_b = variational_dict['latent_b'][:,0:1]
			latent_z_indices = variational_dict['latent_z_indices'][:,0]
			variational_z_logprobabilities = variational_dict['variational_z_logprobabilities'][:,0:1]
			variational_b_logprobabilities = variational_dict['variational_b_logprobabilities'][:,0]
			# variational_z_probabilities = variational_dict['variational_z_probabilities'][:,0]
			variational_b_probabilities = variational_dict['variational_b_probabilities'][:,0]
			kl_divergence = variational_dict['kl_divergence'][:,0:1]
			prior_loglikelihood = variational_dict['prior_loglikelihood'][:,0]
		############# (1.5) ###########
		# Doesn't really matter what the conditional information is here... because latent policy isn't being rolled out. 
		# We still call it becasue these assembled inputs are passed to the latnet policy rollout later.

		if self.conditional_viz_env:
			self.set_env_conditional_info()
				
		# Get assembled inputs and subpolicy inputs for variational rollout.
		if self.args.batch_size > 1:
			orig_assembled_inputs, orig_subpolicy_inputs, padded_action_seq = \
				self.assemble_inputs(concatenated_traj[:,np.newaxis,:], latent_z_indices.unsqueeze(1), latent_b, sample_action_seq[:,np.newaxis,:], self.conditional_information, batch_size=1)

		else: 
			orig_assembled_inputs, orig_subpolicy_inputs, padded_action_seq = \
				self.assemble_inputs(concatenated_traj, latent_z_indices, latent_b, sample_action_seq, self.conditional_information)

		###########################################################
		############# (A) VARIATIONAL POLICY ROLLOUT. #############
		###########################################################
	
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()[:self.batch_trajectory_lengths[self.selected_index]]

		print("Rolling out variational network.")
		# For number of rollout timesteps: 
		for t in range(self.rollout_timesteps-1):
			# Take a rollout step. Feed into policy, get action, step, return new input. 
			# print("Rolling out variational policy, timestep: ", t)

			action_to_execute, new_state = self.take_rollout_step(subpolicy_inputs[:(t+1)].view((t+1,-1)), t)
			state_action_tuple = torch.cat([new_state, action_to_execute],dim=1)			
			# Overwrite the subpolicy inputs with the new state action tuple.
			if self.args.batch_size>1:
				subpolicy_inputs[t+1,0,:self.input_size] = state_action_tuple
			else:
				subpolicy_inputs[t+1,:self.input_size] = state_action_tuple
		
		# Get trajectory from this. 
		if self.args.batch_size>1:
			self.variational_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:,:self.state_dim].detach().cpu().numpy())				
		else:
			self.variational_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())				

		return orig_assembled_inputs, orig_subpolicy_inputs, latent_b

	def alternate_rollout_latent_policy(self, counter, i, orig_assembled_inputs, orig_subpolicy_inputs):
		assembled_inputs = orig_assembled_inputs.clone().detach()
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()

		# This version of rollout uses the incremental reparam get actions function. 		
		hidden = None		

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now.
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		# Set rollout length.
		if self.args.traj_length>0:
			self.rollout_timesteps = self.args.traj_length
		else:
			self.rollout_timesteps = len(sample_traj)		

		# For appropriate number of timesteps. 
		for t in range(self.rollout_timesteps-1):

			# First get input row for latent policy. 

			# Feed into latent policy and get z. 

			# Feed z and b into subpolicy. 

			pass

	def rollout_latent_policy(self, orig_assembled_inputs, orig_subpolicy_inputs):
		assembled_inputs = orig_assembled_inputs.clone().detach()[:self.rollout_timesteps]
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()[:self.rollout_timesteps]
		
		# Set the previous b time to 0.
		delta_t = 0

		print("Rolling out latent policy.")
		# embed()

		# For number of rollout timesteps:
		for t in range(self.rollout_timesteps-1):
			
			if t%10==0:
				print("Rolling out latent policy, timestep: ", t)

			##########################################
			#### CODE FOR NEW Z SELECTION ROLLOUT ####
			##########################################

			# Pick latent_z and latent_b. 
			selected_b, new_selected_z = self.latent_policy.get_actions(assembled_inputs[:(t+1)].view((t+1,-1)), greedy=True, delta_t=delta_t, batch_size=1)

			if t==0:
				selected_b = torch.ones_like(selected_b).to(device).float()

			if selected_b[-1]==1:
				# Copy over ALL z's. This is okay to do because we're greedily selecting, and hte latent policy is hence deterministic.
				selected_z = torch.tensor(new_selected_z).to(device).float()

				# If b was == 1, then... reset b to 0.
				delta_t = 0
			else:
				# Increment counter since last time b was 1.
				delta_t += 1

			# Set z's to 0. 
			assembled_inputs[t+1, self.input_size:self.input_size+self.number_policies] = 0.
			# Set z and b in assembled input for the future latent policy passes. 
			if self.args.discrete_z:
				assembled_inputs[t+1, self.input_size+selected_z[-1]] = 1.
			else:
				if self.args.batch_size>1:
					assembled_inputs[t+1, :, self.input_size:self.input_size+self.latent_z_dimensionality] = selected_z[-1]
					assembled_inputs[t+1, :, self.input_size+self.latent_z_dimensionality]	 = selected_b[-1]
				else:
					assembled_inputs[t+1, self.input_size:self.input_size+self.latent_z_dimensionality] = selected_z[-1]
					assembled_inputs[t+1, self.input_size+self.latent_z_dimensionality]	 = selected_b[-1]
			
			# Before copying over, set conditional_info from the environment at the current timestep.

			if self.conditional_viz_env:
				self.set_env_conditional_info()

			if self.conditional_info_size>0:
				if self.args.batch_size>1:
					assembled_inputs[t+1, :, -self.conditional_info_size:] = torch.tensor(self.conditional_information).to(device).float()
				else:
					assembled_inputs[t+1, -self.conditional_info_size:] = torch.tensor(self.conditional_information).to(device).float()

			# Set z and b in subpolicy input for the future subpolicy passes.			
			if self.args.discrete_z:
				# Set z's to 0.
				subpolicy_inputs[t, self.input_size:self.input_size+self.number_policies] = 0.
				subpolicy_inputs[t, self.input_size+selected_z[-1]] = 1.
			else:
				if self.args.batch_size>1:
					subpolicy_inputs[t, :, self.input_size:] = selected_z[-1]
				else:
					subpolicy_inputs[t, self.input_size:] = selected_z[-1]

			# Now pass subpolicy net forward and get action and next state. 
			action_to_execute, new_state = self.take_rollout_step(subpolicy_inputs[:(t+1)].view((t+1,-1)), t, use_env=self.conditional_viz_env)
			state_action_tuple = torch.cat([new_state, action_to_execute],dim=1)

			# Now update assembled input.
			if self.args.batch_size>1:
				assembled_inputs[t+1, :, :self.input_size] = state_action_tuple
				subpolicy_inputs[t+1, :, :self.input_size] = state_action_tuple
			else:
				assembled_inputs[t+1, :self.input_size] = state_action_tuple
				subpolicy_inputs[t+1, :self.input_size] = state_action_tuple

		if self.args.batch_size>1:
			self.latent_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:,:self.state_dim].detach().cpu().numpy())
		else:
			self.latent_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())

		concatenated_selected_b = np.concatenate([selected_b.detach().cpu().numpy(),np.zeros((1))],axis=-1)

		if self.args.debug:
			print("Embedding in Latent Policy Rollout.")
			embed()

		# Clear these variables from memory.
		del subpolicy_inputs, assembled_inputs
		print("Finishing rollout.")
		return concatenated_selected_b

	def rollout_visuals(self, counter, i, variational_dict=None, get_image=True):

		# if self.args.data=='Roboturk':
		if self.conditional_viz_env:
			self.create_RL_environment_for_rollout(self.dataset[i]['environment-name'], self.dataset[i]['flat-state'][0], self.dataset[i]['task-id'],)

		# Rollout policy with 
		# 	a) Latent variable samples from variational policy operating on dataset trajectories - Tests variational network and subpolicies. 
		# 	b) Latent variable samples from latent policy in a rolling fashion, initialized with states from the trajectory - Tests latent and subpolicies. 
		# 	c) Latent variables from the ground truth set (only valid for the toy dataset) - Just tests subpolicies. 

		###########################################################
		############# (A) VARIATIONAL POLICY ROLLOUT. #############
		###########################################################

		orig_assembled_inputs, orig_subpolicy_inputs, variational_segmentation = self.rollout_variational_network(counter, i, variational_dict=variational_dict)

		###########################################################
		################ (B) LATENT POLICY ROLLOUT. ###############
		###########################################################

		if self.args.viz_latent_rollout:
			latent_segmentation = self.rollout_latent_policy(orig_assembled_inputs, orig_subpolicy_inputs)
		
		latent_rollout_image = None

		if get_image==True:
			if self.args.batch_size>1:
				print("Now visualizing rolled out trajectories.")
				if self.args.viz_latent_rollout:					
					latent_rollout_image = self.visualize_trajectory(self.latent_trajectory_rollout[:,self.selected_index,:], segmentations=latent_segmentation, i=i, suffix='Latent')
				variational_rollout_image = self.visualize_trajectory(self.variational_trajectory_rollout[:,self.selected_index,:], segmentations=variational_segmentation.detach().cpu().numpy(), i=i, suffix='Variational')	

			else:				
				if self.args.viz_latent_rollout:						
					latent_rollout_image = self.visualize_trajectory(self.latent_trajectory_rollout, segmentations=latent_segmentation, i=i, suffix='Latent')
				variational_rollout_image = self.visualize_trajectory(self.variational_trajectory_rollout, segmentations=variational_segmentation.detach().cpu().numpy(), i=i, suffix='Variational')	

			return variational_rollout_image, latent_rollout_image
		else:
			return None, None

	def run_iteration(self, counter, i, skip_iteration=False, return_dicts=False, special_indices=None, train=True, input_dictionary=None, bucket_index=None):

		# With learnt discrete subpolicy: 

		####################################	
		# OVERALL ALGORITHM:
		####################################
		# (1) For all epochs:
		# (2)	# For all trajectories:
		# (3)		# Sample z from variational network.
		# (4)		# Evalute likelihood of latent policy, and subpolicy.
		# (5)		# Update policies using likelihoods.		

		self.set_epoch(counter)	
		self.iter = counter

		####################################
		# (1) & (2) get sample from collect inputs function. 
		####################################

		if input_dictionary is None:
			input_dictionary = {}
			input_dictionary['sample_traj'], input_dictionary['sample_action_seq'], input_dictionary['concatenated_traj'], input_dictionary['old_concatenated_traj'] = self.collect_inputs(i, special_indices=special_indices, called_from_train=True, bucket_index=bucket_index)
			if self.args.task_discriminability or self.args.task_based_supervision:
				input_dictionary['sample_task_id'] = self.input_task_id

			# if not(torch.is_tensor(input_dictionary['old_concatenated_traj'])):
			input_dictionary['old_concatenated_traj'] = torch.tensor(input_dictionary['old_concatenated_traj']).to(device).float()
		else:
			pass
			# Things should already be set. 
		# self.batch_indices_sizes = []
		self.batch_indices_sizes.append({'batch_size': input_dictionary['sample_traj'].shape[0], 'i': i})

		if (input_dictionary['sample_traj'] is not None) and not(skip_iteration):

			####################################
			# (3) Sample latent variables from variational network p(\zeta | \tau).
			####################################

			variational_dict = {}
			profile_var_forward = 0
			
			if profile_var_forward:
				# Line profiling
				self.forward_lp = LineProfiler()
				self.forward_lp_wrapper = self.forward_lp(self.variational_policy.forward)
				variational_dict['latent_z_indices'], variational_dict['latent_b'], variational_dict['variational_b_logprobabilities'], variational_dict['variational_z_logprobabilities'], \
				variational_dict['variational_b_probabilities'], variational_dict['variational_z_probabilities'], variational_dict['kl_divergence'], variational_dict['prior_loglikelihood'] = \
					self.forward_lp_wrapper(input_dictionary['old_concatenated_traj'], self.epsilon, batch_trajectory_lengths=self.batch_trajectory_lengths)
				self.forward_lp.print_stats()
			else:
				variational_dict['latent_z_indices'], variational_dict['latent_b'], variational_dict['variational_b_logprobabilities'], variational_dict['variational_z_logprobabilities'], \
				variational_dict['variational_b_probabilities'], variational_dict['variational_z_probabilities'], variational_dict['kl_divergence'], variational_dict['prior_loglikelihood'] = \
					self.variational_policy.forward(input_dictionary['old_concatenated_traj'], self.epsilon, batch_trajectory_lengths=self.batch_trajectory_lengths)

			if self.args.train and train:

				####################################
				# (4) Evaluate Log Likelihoods of actions and options as "Return" for Variational policy.
				####################################
				
				eval_likelihood_dict = self.evaluate_loglikelihoods(input_dictionary, variational_dict)				


				####################################
				# (5) Update policies. 
				####################################
				
				self.update_policies(i, input_dictionary, variational_dict, eval_likelihood_dict)

				####################################
				# (6) Update plots and logging of stats. 
				####################################

				with torch.no_grad():
					self.update_plots(counter, i, input_dictionary, variational_dict, eval_likelihood_dict)

				# print("#########################################")

			if self.args.debug:
				print("Embedding in Run Iteration.")
				embed()		

		if return_dicts:
			if self.args.train and train:
				return input_dictionary, variational_dict, eval_likelihood_dict	
			else:
				return input_dictionary, variational_dict

	def evaluate_metrics(self):
		self.distances = -np.ones((self.test_set_size))

		# Get test set elements as last (self.test_set_size) number of elements of dataset.
		for i in range(self.test_set_size):

			index = i + len(self.dataset)-self.test_set_size
			print("Evaluating ", i, " in test set, or ", index, " in dataset.")

			# Collect inputs. 
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

			# If valid
			if sample_traj is not None:

				# Create environment to get conditional info.
				if self.conditional_viz_env:
					self.create_RL_environment_for_rollout(self.dataset[i]['environment-name'], self.dataset[i]['flat-state'][0])

				# Rollout variational. 
				_, _, _ = self.rollout_variational_network(0, i)

				self.distances[i] = ((sample_traj-self.variational_trajectory_rollout)**2).mean()	

		self.mean_distance = self.distances[self.distances>0].mean()

		# Create save directory:
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Trajectory_Distances_{0}.npy".format(self.args.name)),self.distances)
		np.save(os.path.join(self.dir_name,"Mean_Trajectory_Distance_{0}.npy".format(self.args.name)),self.mean_distance)

	def evaluate(self, model):

		self.set_epoch(0)

		if model:
			if self.args.load_from_transfer:
				# USe args.source_model / args.target_model to figure out what to load.
				self.load_model_from_transfer()
			else:
				self.load_all_models(model)
		
		np.set_printoptions(suppress=True,precision=2)
		
		if self.args.setting in ['context','joint','learntsub','queryjoint','jointtransfer']:
			self.initialize_training_batches()
		else:
			# print("Running Evaluation of State Distances on small test set.")
			self.evaluate_metrics()

		# Visualize space if the subpolicy has been trained...
		# Running even with the fix_subpolicy, so that we can evaluate joint reconstruction.
		if self.args.data in global_dataset_list:

			print("Running Visualization on Robot Data.")	

			########################################
			# Run Joint Eval.
			########################################

			self.visualize_robot_data()

			if self.args.data in ['RealWorldRigid']:
				print("Entering Query Mode")
				embed()
				return 
			
			########################################
			# Run Pretrain Eval.
			########################################

			
			arg_copy = copy.deepcopy(self.args)
			arg_copy.name += "_Eval_Pretrain"
			if self.args.batch_size>1:
				self.pretrain_policy_manager = PolicyManager_BatchPretrain(self.args.number_policies, self.dataset, arg_copy)
			else:
				self.pretrain_policy_manager = PolicyManager_Pretrain(self.args.number_policies, self.dataset, arg_copy)

			self.pretrain_policy_manager.setup()
			self.pretrain_policy_manager.load_all_models(model, only_policy=True)			
			self.pretrain_policy_manager.visualize_robot_data()			

		elif self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			print("Running visualization of embedding space.")
			self.assemble_joint_skill_embedding_space()
			self.visualize_joint_skill_embedding_space()

			# Copy embedding_latent_z_set_array.
			self.global_z_set = copy.deepcopy(self.embedding_latent_z_set_array)

			print("Evaluate contextual representations.")
			self.evaluate_contextual_representations()			

		if self.args.subpolicy_model:
			print("Loading encoder.")
			self.setup_eval_against_encoder()

		# Evaluate NLL and (potentially Expected Value Difference) on Validation / Test Datasets. 		
		self.epsilon = 0.

		# np.set_printoptions(suppress=True,precision=2)
		# for i in range(60):
		# 	self.run_iteration(0, i)

		if self.args.debug:
			print("Embedding in Evaluate.")
			embed()

	def get_trajectory_and_latent_sets(self, get_visuals=False, N=None):
		self.get_latent_trajectory_segmentation_sets(N=N)

	def get_latent_trajectory_segmentation_sets(self, N=None): 

		# print("Getting latent_z, trajectory, segmentation sets.")

		# For N number of random trajectories from MIME: 
		#	# Encode trajectory using encoder into latent_z sequence. 
		# 	# Get distinct latent_z's from this latent_z sequence. 
		# 	# Get corresponding trajectory segment. 
		# Embed distinct latent_z's.

		# Set N:		
		if self.args.debugging_datapoints > -1: 
			self.N = self.args.debugging_datapoints
		else:
			if N is None:
				self.N = 100
			else:
				self.N = N

		# We're going to store 3 sets of things. 
		# (1) The contextual skill embeddings of the latent_z's. 
		# (2) The full trajectory in which the skill is executed, so we can visualize this skill in context.
		# (3) The segmentations so we can highlight the current latent z and differentiate it from context.  

		self.latent_z_set = []
		self.trajectory_set = []
		self.segmentation_set = []
		self.segmented_trajectory_set = []
		# Also logging latent bs and full latent z trajectory now, because recurrent translation model setting needs it..
		self.latent_b_set = []
		self.full_latent_z_trajectory = []
		self.number_distinct_zs = []

		break_var = 0
		self.number_set_elements = 0 

		# Use the dataset to get reasonable trajectories (because without the information bottleneck / KL between N(0,1), cannot just randomly sample.)
		for i in range(self.N//self.args.batch_size+1):

			# print("####################################")
			# print("Embedding in getting Z set")
			# print("####################################")
			# embed()

			# (1) Encoder trajectory. 
			with torch.no_grad():
				input_dict, variational_dict, _ = self.run_iteration(0, i, return_dicts=True, train=False)

			# Getting latent_b_set and full z traj.
			# Don't unbatch them yet.
			# Basically, if we want to use the Z tuple GMM, we need a differentiable version of the z's that we hold on to.. 
			if self.args.recurrent_translation:
				self.latent_b_set.append(variational_dict['latent_b'].clone().detach())
				self.full_latent_z_trajectory.append(variational_dict['latent_z_indices'].clone().detach())

			# Assuming we're running with batch size>1.
			for b in range(self.args.batch_size):

				# If we've computed enough z's, break at the end of this batch.
				if i*self.args.batch_size+b>=self.N:
					break_var = 1
					break

				# Get segmentations.
				distinct_z_indices = torch.where(variational_dict['latent_b'][:,b])[0].clone().detach().cpu().numpy()
				# Get distinct z's.
				distinct_zs = variational_dict['latent_z_indices'][distinct_z_indices, b].clone().detach().cpu().numpy()

				if self.args.z_tuple_gmm:
					self.number_distinct_zs.append(len(distinct_z_indices))
				

				# print("Embedding in Traj Setting Business")
				# embed()

				# Copy over these into lists.
				self.latent_z_set.append(copy.deepcopy(distinct_zs))
				self.trajectory_set.append(copy.deepcopy(input_dict['sample_traj'][:,b]))
				self.segmentation_set.append(copy.deepcopy(distinct_z_indices))
				
				# Add each trajectory segment to segmented_trajectory_set. 
				for k in range(len(distinct_z_indices)-1):					
					traj_segment = input_dict['sample_traj'][distinct_z_indices[k]:distinct_z_indices[k+1],b]
					self.segmented_trajectory_set.append(copy.deepcopy(traj_segment))				
				
				# traj_segment = input_dict['sample_traj'][distinct_z_indices[k+1]:,b]
				traj_segment = input_dict['sample_traj'][distinct_z_indices[len(distinct_z_indices)-1]:,b]
				self.segmented_trajectory_set.append(copy.deepcopy(traj_segment))

				self.number_set_elements += 1 

			if i*self.args.batch_size+b>=self.N or break_var:
				# print("We're breaking at",self.N,i,i*self.args.batch_size,i*self.args.batch_size+b,len(self.latent_b_set)) 
				break
				

		self.avg_reconstruction_error = 0.

		# Log cummulative number of z's..
		if self.args.z_tuple_gmm:			
			self.cummulative_number_zs = np.concatenate([np.zeros(1),np.cumsum(np.array(self.number_distinct_zs))]).astype(int)
			
		# # if self.args.setting=='jointtransfer':
		# # 	self.source_latent_zs = np.concatenate(self.source_manager.latent_z_set)
		# # 	self.target_latent_zs = np.concatenate(self.target_manager.latent_z_set)
		# # else:
		# self.source_latent_zs = self.source_manager.latent_z_set
		# self.target_latent_zs = self.target_manager.latent_z_set
		# if self.args.setting=='jointtransfer':
		# 	self.latent_z_set = np.concatenate(self.latent_z_set)
		# 	self.trajectory_set = np.array(self.trajectory_set)

	def assemble_joint_skill_embedding_space(self, preset_latent_sets=False):
		
		print("Assemble Skill Embedding Space.")		

		#################################
		# First get latent_z's, trajectories, and segmentations.
		#################################

		if not(preset_latent_sets):
			self.get_latent_trajectory_segmentation_sets()		
		# This function sets.. latent_z_set, trajectory_set, and segmentation_set. 

		#################################
		# Create variables. 
		#################################

		self.embedding_latent_z_set = {}
		self.embedding_image_set = {}
		self.embedding_traj_set = {}
		self.embedding_segment_set = {}

		#################################
		# For all the elements in the set, generate image of this element. 
		#################################

		for j in range(len(self.latent_z_set)):
			
			# print("Parsing element {0} in assemble_joint_skill_embedding_space.".format(j))
			#################################
			# Parse element. 
			################################# 
			
			latent_z_seq = self.latent_z_set[j]
			segmentation = self.segmentation_set[j]
			trajectory = self.trajectory_set[j]	
						
			self.embedding_traj_set['traj_{0}'.format(j)] = copy.deepcopy(trajectory)
	
			#################################
			# There may be multiple elements here - check.
			#################################
			if len(latent_z_seq.shape)>1:				
				run_till = len(latent_z_seq)
			else:
				run_till = 1

			# for k in range(len(latent_z_seq)):
			for k in range(run_till):

				#################################
				# For each z.. 
				if run_till==1:
					latent_z = latent_z_seq
				else:
					latent_z = latent_z_seq[k]	
				# self.embedding_latent_z_set.append(copy.deepcopy(latent_z))
				self.embedding_latent_z_set['latent_z_j{0}_k{1}'.format(j,k)] = copy.deepcopy(latent_z)
				
				#################################
				# Get segmentation.
				#################################

				start_index = segmentation[k]
				# if run_till==1:
				# 	end_index = segmentation[k+1]
				# else:
				if k+1 >= len(latent_z_seq):
					# Because this won't evaluate to true if latent_z_seq is a single z, because len(latent_z_seq) = 64. 
					end_index = trajectory.shape[0]
				else:					
					end_index = segmentation[k+1]
								
				# self.embedding_segment_set.append([start_index, end_index])
				self.embedding_segment_set['segment_j{0}_k{1}'.format(j,k)] = [start_index, end_index]

				#################################
				# Get image. 		
				#################################
				# t1 = time.time()
				# image = self.get_skill_visual_in_context(trajectory, segment_start=start_index, segment_end=end_index)
				# t2 = time.time()
				# # self.embedding_image_set['image_{0}'.format(k)] = copy.deepcopy(image)
				# self.embedding_image_set['image_{0}'.format(k)] = image
				
				self.embedding_image_set['image_j{0}_k{1}'.format(j,k)] = self.get_skill_visual_in_context(trajectory, segment_start=start_index, segment_end=end_index)

		# For that, convert to numpy arrays..
		# self.embedding_latent_z_set_array = np.array(self.embedding_latent_z_set)
		self.embedding_latent_z_set_array = np.array(list(self.embedding_latent_z_set.values()))

	def visualize_joint_skill_embedding_space(self, suffix=None, global_z_set=None, image_suffix=None):
		
		#################################
		# Now that we've gotten the entire set of latent_z's and their corresponding trajectory images, plot it all.
		#################################

		print("Visualizing Skill Embedding Space.")

		#################################
		# First project it into a TSNE space.
		#################################

		# If we have kept track of global z's, append those to embedding_latent_z_set to visualize, for contrast. 
		if global_z_set is not None:			
			self.embedding_latent_z_set_array = np.concatenate([self.embedding_latent_z_set_array, global_z_set])

		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=self.args.perplexity)
		embedded_zs = tsne.fit_transform(self.embedding_latent_z_set_array)

		#################################
		# Create save dictionary.
		#################################

		if self.args.model is not None:
			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))		
		else:
			# Create fake directory. 
			model_epoch = "during_training"

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		if suffix is not None:
			self.dir_name = os.path.join(self.dir_name, suffix)

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		#################################
		# Create a scatter plot of the embedding itself. The plot does not seem to work without this. 
		#################################

		matplotlib.rcParams['figure.figsize'] = [4,4]
		fig, ax = plt.subplots()
		ax.scatter(embedded_zs[:,0],embedded_zs[:,1])		
		plt.savefig("{0}/Raw_Embedding_Joint_{1}_perp{2}_{3}.png".format(self.dir_name,self.args.name,self.args.perplexity,image_suffix))
		plt.close()

		#################################
		# Now overlay images.
		#################################

		# matplotlib.rcParams['figure.figsize'] = [8, 8]
		# zoom_factor = 0.04
		# matplotlib.rcParams['figure.figsize'] = [40, 40]		
		# zoom_factor = 0.4
		matplotlib.rcParams['figure.figsize'] = [80, 80]
		# matplotlib.rcParams['figure.figsize'] = [100, 100]
		zoom_factor = 0.5

		fig, ax = plt.subplots()

		# Plot z's.
		ax.scatter(embedded_zs[:,0],embedded_zs[:,1],s=100)

		# Now set some plot parameters. 
		ax.axis('off')
		ax.set_title("Embedding of Latent Representation of our Model",fontdict={'fontsize':5})
		artists = []
		
		#################################
		# For number of samples in TSNE / Embedding, create a Image object for each of them. 
		#################################

		# for k in range(len(self.embedding_image_set)):	
		for k, v in enumerate(self.embedding_image_set):
			# if k%10==0:
			# 	print(k)
			
			# Create offset image (so that we can place it where we choose), with specific zoom. 
			imagebox = OffsetImage(self.embedding_image_set[v].transpose([1,2,0]),zoom=zoom_factor)
			
			# Create an annotation box to put the offset image into. specify offset image, position, and disable bounding frame. 
			ab = AnnotationBbox(imagebox, (embedded_zs[k,0],embedded_zs[k,1]), frameon=False)

			# Add the annotation box artist to the list artists. 
			artists.append(ax.add_artist(ab))	

		#################################
		# With direct trajectory overlays instead of  inset images.
		#################################

		# ratio = 0.2 

		# for k, traj in enumerate(self.embedding_traj_set):

		# 	ax.scatter(embedded_zs[k,0]+ratio*traj[:,0],embedded_zs[k,1]+ratio*traj[:,1],c=range(len(traj)),s=100,cmap='jet',vmin=0,vmax=len(traj))

		# 	# if self.args.setting=='context':
		# 	segment_start = self.embedding_segment_set[k][0]
		# 	segment_end = self.embedding_segment_set[k][1]
			
		# 	ax.scatter(embedded_zs[k,0]+ratio*traj[segment_start:segment_end,0],embedded_zs[k,1]+ratio*traj[segment_start:segment_end,1],c=range(segment_start,segment_end),s=200,cmap='jet',vmin=0,vmax=len(traj),edgecolors='k',linewidth=2.)
	
		#################################
		# Save Figure.
		#################################
		
		plt.savefig("{0}/Embedding_Joint_{1}_perp{2}_{3}.png".format(self.dir_name,self.args.name,self.args.perplexity, image_suffix))
		plt.close()	

	def get_skill_visual_in_context(self, traj, segment_start=None, segment_end=None):
		
		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:

			# If Toy data, then... 
			# Generate image of trajectory. 
			matplotlib.rcParams['figure.figsize'] = [5,5]

			fig = plt.figure()		
			ax = fig.gca()

			# If generating contextual embedding visuals, visualize entire trajectory for context. 
			# Otherwise just visualize the trajectory segment.
			# if self.args.setting=='context':
			ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),s=30,cmap='jet',vmin=0,vmax=len(traj))

			# if self.args.setting=='context':
			ax.scatter(traj[segment_start:segment_end,0],traj[segment_start:segment_end,1],c=range(segment_start,segment_end),s=60,cmap='jet',vmin=0,vmax=len(traj),edgecolors='k',linewidth=2.)
	
			scale = 15
			plt.xlim(-scale,scale)
			plt.ylim(-scale,scale)
	
			fig.canvas.draw()
	
			width, height = fig.get_size_inches() * fig.get_dpi()
			image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
			
			# Already got image data. Now close plot so it doesn't cry.
			# fig.gcf()
			plt.close()

			image = np.transpose(image, axes=[2,0,1])

			return image
		else:
			pass

	def set_context_histogram(self):

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			# When we're using toy data, can use ground truth z's to create a histogram of the context of a particular z.
			# Analyze how the z representation changes over the values in this histogram.

			self.dataset.B_array
			self.dataset.Y_array

			# Get distinct z's and distinct z indices from dataset.
			distinct_zs = []
			distinct_z_indices = []
			
			max_distinct_zs = 0

			for i in range(len(self.dataset.B_array)):
				dist_z_inds = np.where(self.dataset.B_array[i,:])[0]
				distinct_z_indices.append(copy.deepcopy(dist_z_inds))
				dist_zs = self.dataset.Y_array[i,dist_z_inds]
				distinct_zs.append(copy.deepcopy(dist_zs))
				if len(dist_z_inds)>max_distinct_zs:
					max_distinct_zs=len(dist_z_inds)

			# Now pad the zs with trailing -1's to max length.
			for i in range(len(self.dataset.B_array)):
				distinct_zs[i] = np.pad(distinct_zs[i], (0,max_distinct_zs-len(distinct_zs[i])), mode='constant', constant_values=-1)
			# Now make it an array. 
			distinct_zs = np.array(distinct_zs)

			# Now that we have things assembled nicely, get unique sequences,
			# ID of which sequence the particular element in distinct_z is, 
			# Inverse indices, which describe the 
			# and the count of how many of such unique elements exist.
			print("Running unique.")
			self.skill_sequence, self.skill_index, self.skill_inverse_indices, self.skill_counts = np.unique(distinct_zs, axis=0, return_inverse=True, return_index=True, return_counts=True)

	def evaluate_similar_skills_different_skill_sequences(self, specific_index=None):

		####################################
		####################################
		# (1) Evaluate similar skills in different skill sequences. 
		# (1) Using the ground truth option label to ascertain whether skills are "similar" or not.

		# Ideally we'd like to see - 
		# (a) Similar skills have overall close embeddings (compared to global sets), but embeddings change based on context. 
		# (b) Similar contexts / skill sequences result in more similar skill embeddings than different contexts. 
		####################################
		####################################

		print("#####################################################")
		print("Now evaluating embeddings of the same skill in different contexts / amongst different skill sequences.")
		print("#####################################################")

		# Since there are just 4 skills used to generate the data, we should evaluate embeddings for all of them. 
		number_of_datapoints = self.args.batch_size
		# If we look at consider_labels_from# of these, should find batch_size number of relevant z's.
		if self.args.data=='ContinuousNonZero':
			consider_labels_from = 2*number_of_datapoints
		else:
			consider_labels_from = 3*number_of_datapoints

		i=0				
		indices = np.arange(i,i+consider_labels_from)
		self.dataset_b, self.dataset_z = self.dataset.get_latent_variables(indices)

		# Set range to evaluate over. 
		if specific_index is not None:
			# If we have a specific index (likely because the jointtransfer setting is using this function), 
			# Just evaluate for that specific index.
			eval_range = range(specific_index,specific_index+1)
			suffix = "_SPI{0}".format(specific_index)
		else:
			eval_range = range(self.args.number_policies)
			suffix = ""
		# for k in range(self.args.number_policies):
		for k in eval_range:

			# For each skill, get number_of_datapoint number of different trajectories that have the relevant skill. 
			special_indices = []
			relevant_z_indices = np.where((self.dataset_z==k).any(axis=-1))[0]
			special_indices = relevant_z_indices[:number_of_datapoints]
			
			print("#####################################################")
			print("Running evaluations of the same skill in different contexts, for skill ID: ",k)		

			# Now feed into run iteration. 
			input_dict, variational_dict, eval_likelihood_dict = self.run_iteration(0, i, return_dicts=True, special_indices=special_indices, train=False)

			self.latent_z_set = []
			self.trajectory_set = []
			self.segmentation_set = []

			for b in range(number_of_datapoints):
			
				# Get segmentations.		
				distinct_z_indices = torch.where(variational_dict['latent_b'][:,b])[0].clone().detach().cpu().numpy()
				
				# Get distinct z's.
				distinct_zs = variational_dict['latent_z_indices'][distinct_z_indices, b].clone().detach().cpu().numpy()

				# Get timesteps for which it's the relevant skill under the GT labels. 
				relevant_z_indices = np.where((self.dataset_latent_z_labels[b,:]==k))[0]

				# Now add the last timestep 
				histogram_bins = np.concatenate([distinct_z_indices,np.array(input_dict['sample_traj'].shape[0:1])])
				
				# Find out which bin the relevant_z_indices intersect the most with. 
				counts, bins = np.histogram(relevant_z_indices, bins=histogram_bins)
				relevant_bin = counts.argmax()
				relevant_index = bins[relevant_bin]

				# Copy to lists.				
				self.trajectory_set.append(input_dict['sample_traj'][:,b])
				self.latent_z_set.append(distinct_zs[relevant_bin])
				self.segmentation_set.append([bins[relevant_bin],bins[relevant_bin+1]])

			# Now that latent_z_set, trajectory_set and segmentation_set are set, call - 
			self.assemble_joint_skill_embedding_space(preset_latent_sets=True)
			# # Now that we have the dictionaries... embed and visualize them. 
			self.visualize_joint_skill_embedding_space(suffix='SameSkill_DiffContext{0}'.format(suffix), global_z_set=self.global_z_set, image_suffix="Skill{0}".format(k))

	def evaluate_similar_skill_sequences(self, specific_index=None):

		####################################
		####################################
		# (2) Evaluate representations of similar sets of skill sequences, for which there are at least k trajectories.
		# (2) This shows different skills in similar skill sequences. 

		# What does this tell us? For each unique sequence of skills, what does the embedding of the skills executed in this sequence look like? 
		# (a) Ideally want to see that the same skill in the same sequence of skill is embedded together. 
		# (b) Ideally want to see different skills in the same sequence of skill embedded differently. 		
		####################################
		####################################

		self.minimum_trajectories = 20
		# self.max_viz_trajs = 5

		# Figure out where we have enough datapoints to make reasonable comparisons.
		eval_skill_sequence_indices = np.where(self.skill_counts>self.minimum_trajectories)[0][:self.max_viz_trajs]

		print("#####################################################")
		print("Iterating over skill sequences, and visualizing them.")
		print("#####################################################")

		# Set evaluation range. 
		if specific_index is not None:
			eval_range = range(specific_index, specific_index+1)
			suffix = "_SPI{0}".format(specific_index)
		else:
			# For all skill_sequences for which we have these minimum number of trajectories.
			eval_range = range(len(eval_skill_sequence_indices))
			suffix = ""
		# for i, skill_seq in enumerate(eval_skill_sequence_indices):
		for i in eval_range:

			# Since we switched to ranges, set this element based on the current index.
			skill_seq = eval_skill_sequence_indices[i]
								
			print("#####################################################")
			print("Currently operating on skills sequence ", i)			

			# Get dataset indices for this specific skill sequence.
			special_indices = np.where(self.skill_inverse_indices==skill_seq)[0]

			# Now if special_indices > batch_size, randomly sample a batch of them.
			# if special_indices.shape[0] > self.args.batch_size:
				# special_indices = np.random.choice(special_indices, size=self.args.batch_size)

			# Instead of randomly sampling indices, just select the first batch_size of them.	
			special_indices = special_indices[:self.args.batch_size]

			# Now feed into run iteration. 
			input_dict, variational_dict, eval_likelihood_dict = self.run_iteration(0, i, return_dicts=True, special_indices=special_indices, train=False)

			# Reset the latent, trajectory, and segmentation sets.			
			self.latent_z_set = []
			self.trajectory_set = []
			self.segmentation_set = []

			for b in range(self.args.batch_size):
				# Get segmentations.				
				distinct_z_indices = torch.where(variational_dict['latent_b'][:,b])[0].clone().detach().cpu().numpy()
				# Get distinct z's.
				distinct_zs = variational_dict['latent_z_indices'][distinct_z_indices, b].clone().detach().cpu().numpy()
				
				# Copy over these into lists.
				self.latent_z_set.append(copy.deepcopy(distinct_zs))
				self.trajectory_set.append(copy.deepcopy(input_dict['sample_traj'][:,b]))
				self.segmentation_set.append(copy.deepcopy(distinct_z_indices))

			# Now that latent_z_set, trajectory_set and segmentation_set are set, call - 
			self.assemble_joint_skill_embedding_space(preset_latent_sets=True)
			# # Now that we have the dictionaries... embed and visualize them. 
			self.visualize_joint_skill_embedding_space(suffix='SkillSeq{0}'.format(suffix), global_z_set=self.global_z_set, image_suffix="Seq{0}".format(i))
	
	def evaluate_contextual_representations(self, skip_same_skill=False, specific_index=None):
		
		####################################
		print("####################################")
		print("Compute unique sequences of skills., for context evaluation.")		
		print("####################################")
		self.set_context_histogram()
		
		# Things to evaluate. 

		####################################
		# (1) Evaluate similar skills in different skill sequences. 
		####################################
		
		if not(skip_same_skill):
			self.evaluate_similar_skills_different_skill_sequences(specific_index=specific_index)

		####################################
		# (2) Evaluate representations of similar sets of skill sequences, for which there are at least k trajectories.
		####################################

		self.evaluate_similar_skill_sequences(specific_index=specific_index)

class PolicyManager_BatchJoint(PolicyManager_Joint):

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_BatchJoint, self).__init__(number_policies, dataset, args)		

	def create_networks(self):

		# print("Embed in network creation")
		# embed()
		# Create instances of networks. 
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)
		self.latent_policy = ContinuousLatentPolicyNetwork_ConstrainedBPrior(self.input_size+self.conditional_info_size, self.hidden_size, self.args, self.number_layers).to(device)

		if (self.args.setting=='context' or (self.args.setting=='jointtransfer' and self.args.context)):
			if self.args.new_context:
				self.variational_policy = ContinuousNewContextualVariationalPolicyNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.args.var_number_layers).to(device)
			else:
				self.variational_policy = ContinuousContextualVariationalPolicyNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.args.var_number_layers).to(device)
		else:
			if self.args.data in ['RealWorldRigid'] or self.args.split_stream_encoder==1:
				print("Making a Factored Segmenter Network.")
				self.variational_policy = ContinuousSequentialFactoredEncoderNetwork(self.input_size, self.args.var_hidden_size, int(self.latent_z_dimensionality/2), self.args).to(device)
				# self.variational_policy = ContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)
			else:
				self.variational_policy = ContinuousVariationalPolicyNetwork_Batch(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.args.var_number_layers).to(device)
		
	# Batch concatenation functions. 
	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((self.args.batch_size,1,self.output_size)),sample_action_seq],axis=1)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((self.args.batch_size,1,self.output_size))],axis=1)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)
		
	def get_batch_element(self, i):
		# Make data_element a list of dictionaries. 
		# data_element = np.array([self.dataset[b] for b in range(i,i+self.args.batch_size)])
		data_element = np.array([self.dataset[b] for b in self.sorted_indices[i:i+self.args.batch_size]])
		# for b in range(i,i+self.args.batch_size):	
		# 	data_element.append(self.dataset[b])

		return data_element

	def set_batch_mask(self):

		# Initialize with 0's. 
		# self.batch_mask = torch.zeros((self.args.batch_size, self.max_batch_traj_length)).to(device).float()
		self.batch_mask = torch.zeros((self.max_batch_traj_length, self.args.batch_size)).to(device).float()
		# Set batch mask for each batch element as ... 1's for length of that element, and 0 after. 
		for b in range(self.args.batch_size):
			# self.batch_mask[b, :self.batch_trajectory_lengths[b]] = 1.
			self.batch_mask[:self.batch_trajectory_lengths[b], b] = 1.

	# Get batch full trajectory. 
	def collect_inputs(self, i, get_latents=False, special_indices=None, called_from_train=False, bucket_index=None):

		# print("# Debug task ID batching")
		# embed()

		# Toy Data
		if self.args.data in ['ContinuousNonZero','DirContNonZero','DeterGoal','ToyContext']:

			# Sample trajectory segment from dataset. 
			if special_indices is not None:
				sample_traj, sample_action_seq = self.dataset[special_indices]
				self.dataset_latent_b_labels, self.dataset_latent_z_labels = self.dataset.get_latent_variables(special_indices)
			else:
				sample_traj, sample_action_seq = self.dataset[i:i+self.args.batch_size]
				indices = np.arange(i,i+self.args.batch_size)
				self.dataset_latent_b_labels, self.dataset_latent_z_labels = self.dataset.get_latent_variables(indices)
					
			# If the collect inputs function is being called from the train function, 
			# Then we should corrupt the inputs based on how much the input_corruption_noise is set to. 
			# If it's 0., then no corruption. 

			corrupted_sample_action_seq = self.corrupt_inputs(sample_action_seq)
			corrupted_sample_traj = self.corrupt_inputs(sample_traj)
			concatenated_traj = self.concat_state_action(corrupted_sample_traj, corrupted_sample_action_seq)		
			old_concatenated_traj = self.old_concat_state_action(corrupted_sample_traj, corrupted_sample_action_seq)

			if self.args.data=='DeterGoal':
				self.conditional_information = np.zeros((self.args.condition_size))
				self.conditional_information[self.dataset.get_goal(i)] = 1
				self.conditional_information[4:] = self.dataset.get_goal_position[i]
			else:
				self.conditional_information = np.zeros((self.args.condition_size))
			self.batch_trajectory_lengths = concatenated_traj.shape[1]*np.ones((self.args.batch_size)).astype(int)
			self.max_batch_traj_length = concatenated_traj.shape[1]

			return sample_traj.transpose((1,0,2)), sample_action_seq.transpose((1,0,2)), concatenated_traj.transpose((1,0,2)), old_concatenated_traj.transpose((1,0,2))

		elif self.args.data in global_dataset_list:
					   
			if self.args.data in ['MIME','OldMIME'] or self.args.data=='Mocap':

				if self.args.task_discriminability or self.args.task_based_supervision:

					# Don't really need to use digitize, we need to digitize with respect to .. 0, 32, 64, ... 8448. 
					# So.. just use... //32
					if bucket_index is None:
						bucket = i//self.args.batch_size
					else:
						bucket = bucket_index

					data_element = self.dataset[np.array(self.task_based_shuffling_blocks[bucket])]
					self.input_task_id = self.index_task_id_map[bucket]					
				else:
					# data_element = self.dataset[i:i+self.args.batch_size]

					# print("Index actually going into dataset: ",i,i+self.args.batch_size)
					data_element = self.dataset[self.sorted_indices[i:i+self.args.batch_size]]

			else:
				data_element = self.get_batch_element(i)

			# Get trajectory lengths across batch, to be able to create masks for losses. 
			self.batch_trajectory_lengths = np.zeros((self.args.batch_size), dtype=int)
			minl = 10000
			maxl = 0

			# print("Debugging dataset extent")
			# embed()
			
			for x in range(self.args.batch_size):

				if self.args.ee_trajectories:
					self.batch_trajectory_lengths[x] = data_element[x]['endeffector_trajectory'].shape[0]
				else:

					self.batch_trajectory_lengths[x] = data_element[x]['demo'].shape[0]
				maxl = max(maxl,self.batch_trajectory_lengths[x])
				minl = min(minl,self.batch_trajectory_lengths[x])
			# print("For this iteration:",maxl-minl,minl,maxl)

			self.max_batch_traj_length = self.batch_trajectory_lengths.max()

			# Create batch object that stores trajectories. 
			batch_trajectory = np.zeros((self.args.batch_size, self.max_batch_traj_length, self.state_size))
			
			# Copy over data elements into batch_trajectory array.
			for x in range(self.args.batch_size):
				if self.args.ee_trajectories:	
					batch_trajectory[x,:self.batch_trajectory_lengths[x]] = data_element[x]['endeffector_trajectory']
				else:					
					batch_trajectory[x,:self.batch_trajectory_lengths[x]] = data_element[x]['demo']

				if self.args.data in ['RealWorldRigid'] and self.args.images_in_real_world_dataset:
					data_element[x]['subsampled_images'] = data_element[x]['images']
			
			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
				batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value				

			# Set condiitonal information. 
			if self.args.data in ['MIME','OldMIME','GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', 'GRABObject','DAPG', 'DAPGHand', 'DAPGObject','DexMV','DexMVHand','DexMVObject']:
				self.conditional_information = np.zeros((self.conditional_info_size))				

			# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic',\
				'RoboturkObjects','RoboturkRobotObjects','RoboMimicObjects','RoboMimicRobotObjects']:

				if self.args.batch_size==1:

					robot_states = data_element['robot-state']
					object_states = data_element['object-state']

					self.conditional_information = np.zeros((self.conditional_info_size))
					
					# Don't set this if pretraining / baseline.
					if self.args.setting=='learntsub' or self.args.setting=='imitation':
						self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
						self.conditional_information[:,:self.cond_robot_state_size] = robot_states
						# Doing this instead of self.cond_robot_state_size: because the object_states size varies across demonstrations.
						self.conditional_information[:,self.cond_robot_state_size:self.cond_robot_state_size+object_states.shape[-1]] = object_states	
						# Setting task ID too.		
						self.conditional_information[:,-self.number_tasks+data_element['task-id']] = 1.
				else:

					#####################################################################	
					# Set a batch element here..				
					batch_conditional_information = np.zeros((self.args.batch_size, self.max_batch_traj_length, self.conditional_info_size))

					for x in range(self.args.batch_size):

						if data_element[x]['is_valid'] and self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']:

							batch_conditional_information[x,:self.batch_trajectory_lengths[x],:self.cond_robot_state_size] = data_element[x]['robot-state']						
							batch_conditional_information[x,:self.batch_trajectory_lengths[x],self.cond_robot_state_size:self.cond_robot_state_size+data_element[x]['object-state'].shape[-1]] = data_element[x]['object-state']						
							batch_conditional_information[x,:self.batch_trajectory_lengths[x],-self.number_tasks+data_element[x]['task-id']] = 1.
 
					self.conditional_information = np.zeros((self.conditional_info_size))
					#####################################################################

			# Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)

			# If the collect inputs function is being called from the train function, 
			# Then we should corrupt the inputs based on how much the input_corruption_noise is set to. 
			# If it's 0., then no corruption. 
			corrupted_action_sequence = self.corrupt_inputs(action_sequence)
			corrupted_batch_trajectory = self.corrupt_inputs(batch_trajectory)

			concatenated_traj = self.concat_state_action(corrupted_batch_trajectory, corrupted_action_sequence)		
			old_concatenated_traj = self.old_concat_state_action(corrupted_batch_trajectory, corrupted_action_sequence)
			# # Concatenate
			# concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)
			# old_concatenated_traj = self.old_concat_state_action(batch_trajectory, action_sequence)

			# Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence
			

			# If trajectory length is set to something besides -1, restrict trajectory length to this.
			if self.args.traj_length > -1 :			
				batch_trajectory = batch_trajectory.transpose((1,0,2))[:self.args.traj_length]
				scaled_action_sequence = scaled_action_sequence.transpose((1,0,2))[:self.args.traj_length-1]
				concatenated_traj = concatenated_traj.transpose((1,0,2))[:self.args.traj_length]
				old_concatenated_traj = old_concatenated_traj.transpose((1,0,2))[:self.args.traj_length]

				for x in range(self.args.batch_size):
					if self.batch_trajectory_lengths[x] > self.args.traj_length:
						self.batch_trajectory_lengths[x] = self.args.traj_length
				self.max_batch_traj_length = self.batch_trajectory_lengths.max()			

				return batch_trajectory, scaled_action_sequence, concatenated_traj, old_concatenated_traj, data_element

			# If we're using task based discriminability. 
			if self.args.task_discriminability or self.args.task_based_supervision:
				# Set the task ID's. 
				self.batch_task_ids = np.zeros((self.args.batch_size), dtype=int)
				for k in range(self.args.batch_size):
					self.batch_task_ids[k] = data_element[k]['task_id']
				
				# Figure f we should be implementing.. same task ID stuff... probably more important to do smart batching of trjaectory lengths? 
				# What will this need? Maybe... making references to the discriminators? And then calling forward on them? 

			# print("Embed before return in Collect Input Batch")
			# embed()
			return batch_trajectory.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), \
				concatenated_traj.transpose((1,0,2)), old_concatenated_traj.transpose((1,0,2)), \
					data_element
			
	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq, conditional_information=None, batch_size=None):

		if batch_size is None:
			batch_size = self.args.batch_size

		################
		# This seems like it could be a major issue in the contextual embedding training. 
		# If you made this a copy, and added parameters of the subpolicy to the optimizer... 
		# Stops gradients going back into the variational network from this. 
		################
		# if self.training_phase>1:
		# 	# Prevents gradients being propagated through this..
		# 	latent_z_copy = torch.tensor(latent_z_indices).to(device)
		# else:
		# 	latent_z_copy = latent_z_indices

		# INSTEAD, just try the latent_z_copy. 
		latent_z_copy = latent_z_indices
		# latent_z_copy = torch.tensor(latent_z_indices).to(device)

		if conditional_information is None:
			conditional_information = torch.zeros((self.conditional_info_size)).to(device).float()

		# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 					
		assembled_inputs = torch.zeros((input_trajectory.shape[0],batch_size,self.input_size+self.latent_z_dimensionality+1+self.conditional_info_size)).to(device)		
		assembled_inputs[:,:,:self.input_size] = torch.tensor(input_trajectory).to(device).float()
		assembled_inputs[range(1,len(input_trajectory)),:,self.input_size:self.input_size+self.latent_z_dimensionality] = latent_z_copy[:-1]
		
		# We were writing the wrong dimension... should we be running again? :/ 
		assembled_inputs[range(1,len(input_trajectory)),:,self.input_size+self.latent_z_dimensionality] = latent_b[:-1].float()
		# assembled_inputs[range(1,len(input_trajectory)),-self.conditional_info_size:] = torch.tensor(conditional_information).to(device).float()

		# Instead of feeding conditional infromation only from 1'st timestep onwards, we are going to st it from the first timestep. 
		if self.conditional_info_size>0:
			assembled_inputs[:,:,-self.conditional_info_size:] = torch.tensor(conditional_information).to(device).float()

		# Now assemble inputs for subpolicy.
		subpolicy_inputs = torch.zeros((len(input_trajectory),batch_size,self.input_size+self.latent_z_dimensionality)).to(device)
		subpolicy_inputs[:,:,:self.input_size] = torch.tensor(input_trajectory).to(device).float()
		subpolicy_inputs[range(len(input_trajectory)),:,self.input_size:] = latent_z_indices

		# # This method of concatenation is wrong, because it evaluates likelihood of action [0,0] as well. 
		# # Concatenated action sqeuence for policy network. 
		# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
		# This is the right method of concatenation, because it evaluates likelihood
		if torch.is_tensor(sample_action_seq) and self.args.setting=='jointtransfer':
			padded_action_seq = torch.roll(sample_action_seq,-1,dims=0)
			# padded_action_seq = torch.cat([sample_action_seq, torch.zeros((1,batch_size,self.output_size)).to(device).float()],axis=0)
		else:
			padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,batch_size,self.output_size))],axis=0)

		return assembled_inputs, subpolicy_inputs, padded_action_seq
	
	def train(self, model=None):

		# Run some initialization process to manage GPU memory with variable sized batches.
		self.initialize_training_batches()

		# Now run original training function.
		super().train(model=model)

class PolicyManager_BatchJointQueryMode(PolicyManager_BatchJoint):

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_BatchJointQueryMode, self).__init__(number_policies, dataset, args)		

	def run_iteration(self, counter, i, skip_iteration=False, return_dicts=False, special_indices=None, train=True, input_dictionary=None, bucket_index=None):

		# With learnt discrete subpolicy: 

		####################################	
		# OVERALL ALGORITHM:
		####################################
		# (1) For all epochs:
		# (2)	# For all trajectories:
		# (3)		# Sample z from variational network.
		# (4)		# Evalute likelihood of latent policy, and subpolicy.
		# (5)		# Update policies using likelihoods.		

		self.set_epoch(counter)	
		self.iter = counter

		####################################
		# (1) & (2) get sample from collect inputs function. 
		####################################

		if input_dictionary is None:
			input_dictionary = {}
			input_dictionary['sample_traj'], input_dictionary['sample_action_seq'], \
				input_dictionary['concatenated_traj'], input_dictionary['old_concatenated_traj'], \
					input_dictionary['data_element'] = self.collect_inputs(i, special_indices=special_indices, called_from_train=True, bucket_index=bucket_index)
			if self.args.task_discriminability or self.args.task_based_supervision:
				input_dictionary['sample_task_id'] = self.input_task_id

			# if not(torch.is_tensor(input_dictionary['old_concatenated_traj'])):
			input_dictionary['old_concatenated_traj'] = torch.tensor(input_dictionary['old_concatenated_traj']).to(device).float()
		else:
			pass
			# Things should already be set. 
		# self.batch_indices_sizes = []
		self.batch_indices_sizes.append({'batch_size': input_dictionary['sample_traj'].shape[0], 'i': i})

		if (input_dictionary['sample_traj'] is not None) and not(skip_iteration):

			####################################
			# (3) Sample latent variables from variational network p(\zeta | \tau).
			####################################

			variational_dict = {}
			profile_var_forward = 0
			
			variational_dict['latent_z_indices'], variational_dict['latent_b'] = \
				self.variational_policy.forward(input_dictionary['old_concatenated_traj'], self.epsilon)

			if self.args.debug:
				print("Embedding in Run Iteration.")
				embed()		

		if return_dicts:
			return input_dictionary, variational_dict, None

class PolicyManager_BaselineRL(PolicyManager_BaseClass):

	def __init__(self, number_policies=4, dataset=None, args=None):
	
		# super(PolicyManager_BaselineRL, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		super(PolicyManager_BaselineRL, self).__init__()

		# Create environment, setup things, etc. 
		self.args = args		

		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_episodes = self.args.epsilon_over
		self.baseline = None
		self. learning_rate = self.args.learning_rate
		self.max_timesteps = 100
		self.gamma = 0.99
		self.batch_size = 10
		self.number_test_episodes = 100

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_episodes)
		self.number_episodes = 5000000

		# Orhnstein Ullenhbeck noise process parameters. 
		self.theta = 0.15
		self.sigma = 0.2		

		self.gripper_open = np.array([0.0115, -0.0115])
		self.gripper_closed = np.array([-0.020833, 0.020833])


		self.reset_statistics()

	def create_networks(self):

		if self.args.MLP_policy:
			self.policy_network = ContinuousMLP(self.input_size, self.args.hidden_size, self.output_size, self.args).to(device)
			self.critic_network = CriticMLP(self.input_size, self.args.hidden_size, 1, self.args).to(device)
		else:
			# Create policy and critic. 		
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers, small_init=True).to(device)			
			self.critic_network = CriticNetwork(self.input_size, self.args.hidden_size, 1, self.args, self.args.number_layers).to(device)

	def create_training_ops(self):

		self.NLL_Loss = torch.nn.NLLLoss(reduction='none')
		self.MSE_Loss = torch.nn.MSELoss(reduction='none')
		
		# parameter_list = list(self.policy_network.parameters()) + list(self.critic_network.parameters())
		self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
		self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

	def save_all_models(self, suffix):
		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")	
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Critic_Network'] = self.critic_network.state_dict()
		
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, critic=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		if critic:
			self.critic_network.load_state_dict(load_object['Critic_Network'])

	def setup(self):
		# Calling a special RL setup function. This is because downstream classes inherit (and may override setup), but will still inherit RL_setup intact.
		self.RL_setup()

	def RL_setup(self):
		# Create Mujoco environment. 
		import robosuite
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		# self.input_size = self.state_size + self.output_size		
		self.input_size = self.state_size + self.output_size*2
		
		# Create networks. 
		self.create_networks()
		self.create_training_ops()		
		self.initialize_plots()

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size, min_sigma=self.args.OU_min_sigma, max_sigma=self.args.OU_max_sigma)

	def set_parameters(self, episode_counter, evaluate=False):
		if self.args.train and not(evaluate):
			if episode_counter<self.decay_episodes:
				self.epsilon = self.initial_epsilon-self.decay_rate*episode_counter
			else:
				self.epsilon = self.final_epsilon		
		else:
			self.epsilon = 0.

	def reset_lists(self):
		self.reward_trajectory = []
		self.state_trajectory = []
		self.action_trajectory = []
		self.image_trajectory = []
		self.terminal_trajectory = []
		self.cummulative_rewards = None
		self.episode = None

	def get_action(self, hidden=None, random=True, counter=0, evaluate=False):

		# Change this to epsilon greedy...
		whether_greedy = np.random.binomial(n=1,p=0.8)

		if random or not(whether_greedy):
			action = 2*np.random.random((self.output_size))-1
			return action, hidden	

		# The rest of this will only be evaluated or run when random is false and when whether_greedy is true.
		# Assemble states of current input row.
		current_input_row = self.get_current_input_row()

		# Using the incremental get actions. Still get action greedily, then add noise. 		
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(current_input_row).to(device).float(), greedy=True, hidden=hidden)

		if evaluate:
			noise = torch.zeros_like(predicted_action).to(device).float()
		else:
			# Get noise from noise process. 					
			noise = torch.randn_like(predicted_action).to(device).float()*self.epsilon

		# Perturb action with noise. 			
		perturbed_action = predicted_action + noise

		if self.args.MLP_policy:
			action = perturbed_action[-1].detach().cpu().numpy()
		else:
			action = perturbed_action[-1].squeeze(0).detach().cpu().numpy()		

		return action, hidden

	def get_OU_action(self, hidden=None, random=False, counter=0, evaluate=False):

		if random==True:
			action = 2*np.random.random((self.output_size))-1
			return action, hidden
		
		# Assemble states of current input row.
		current_input_row = self.get_current_input_row()
		# Using the incremental get actions. Still get action greedily, then add noise. 		
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(current_input_row).to(device).float(), greedy=True, hidden=hidden)

		# Numpy action
		if self.args.MLP_policy:
			action = predicted_action[-1].detach().cpu().numpy()
		else:
			action = predicted_action[-1].squeeze(0).detach().cpu().numpy()		

		if evaluate:
			perturbed_action = action
		else:
			# Perturb action with noise. 			
			perturbed_action = self.NoiseProcess.get_action(action, counter)

		return perturbed_action, hidden

	def rollout(self, random=False, test=False, visualize=False):
	
		counter = 0		
		eps_reward = 0.	
		state = self.environment.reset()
		terminal = False

		self.reset_lists()
		# Reset the noise process! We forgot to do this! :( 
		self.NoiseProcess.reset()

		if visualize:			
			image = self.environment.sim.render(600,600, camera_name='frontview')
			self.image_trajectory.append(np.flipud(image))
		
		self.state_trajectory.append(state)
		# self.terminal_trajectory.append(terminal)
		# self.reward_trajectory.append(0.)		

		hidden = None

		while not(terminal) and counter<self.max_timesteps:

			if self.args.OU:
				action, hidden = self.get_OU_action(hidden=hidden,random=random,counter=counter, evaluate=test)
			else:
				action, hidden = self.get_action(hidden=hidden,random=random,counter=counter, evaluate=test)			
				
			# Take a step in the environment. 	
			next_state, onestep_reward, terminal, success = self.environment.step(action)
		
			self.state_trajectory.append(next_state)
			self.action_trajectory.append(action)
			self.reward_trajectory.append(onestep_reward)
			self.terminal_trajectory.append(terminal)
				
			# Copy next state into state. 
			state = copy.deepcopy(next_state)

			# Counter
			counter += 1 

			# Append image. 
			if visualize:
				image = self.environment.sim.render(600,600, camera_name='frontview')
				self.image_trajectory.append(np.flipud(image))
		
		print("Rolled out an episode for ",counter," timesteps.")

		# Now that the episode is done, compute cummulative rewards... 
		self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])

		self.episode_reward_statistics = copy.deepcopy(self.cummulative_rewards[0])
		print("Achieved reward: ", self.episode_reward_statistics)
		# print("########################################################")

		# NOW construct an episode out of this..	
		self.episode = RLUtils.Episode(self.state_trajectory, self.action_trajectory, self.reward_trajectory, self.terminal_trajectory)
		# Since we're doing TD updates, we DON'T want to use the cummulative reward, but rather the reward trajectory itself.

	def get_transformed_gripper_value(self, gripper_finger_values):
		gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)			

		finger_diff = gripper_values[1]-gripper_values[0]
		gripper_value = np.array(2*finger_diff-1).reshape((1,-1))
		return gripper_value

	def get_current_input_row(self):
		# Addiong joint states, gripper, actions, and conditional info in addition to just conditional and actions.
		gripper_finger_values = self.state_trajectory[-1]['gripper_qpos']
		conditional = np.concatenate([self.state_trajectory[-1]['robot-state'].reshape((1,-1)),self.state_trajectory[-1]['object-state'].reshape((1,-1))],axis=1)

		if len(self.action_trajectory)>0:				
			state_action = np.concatenate([self.state_trajectory[-1]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[-1].reshape((1,-1))],axis=1)
		else:			
			# state_action = np.concatenate([self.state_trajectory[-1]['robot-state'].reshape((1,-1)),self.state_trajectory[-1]['object-state'].reshape((1,-1)),np.zeros((1,self.output_size))],axis=1)
			state_action = np.concatenate([self.state_trajectory[-1]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), np.zeros((1,self.output_size))],axis=1)
		return np.concatenate([state_action, conditional],axis=1)

	def assemble_inputs(self):
		conditional_sequence = np.concatenate([np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1) for t in range(len(self.state_trajectory))],axis=0)

		state_action_sequence = np.concatenate([np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(self.state_trajectory[t]['gripper_qpos']), self.action_trajectory[t-1].reshape((1,-1))],axis=1) for t in range(1,len(self.state_trajectory))],axis=0)		
		initial_state_action = np.concatenate([self.state_trajectory[0]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(self.state_trajectory[0]['gripper_qpos']), np.zeros((1, self.output_size))],axis=1)

		# Copy initial state to front of state_action seq. 
		state_action_sequence = np.concatenate([state_action_sequence, initial_state_action],axis=0)

		inputs = np.concatenate([state_action_sequence, conditional_sequence],axis=1)
		
		return inputs

	def process_episode(self, episode):
		# Assemble states, actions, targets.

		# First reset all the lists from the rollout now that they've been written to memory. 
		self.reset_lists()

		# Now set the lists. 
		self.state_trajectory = episode.state_list
		self.action_trajectory = episode.action_list
		self.reward_trajectory = episode.reward_list
		self.terminal_trajectory = episode.terminal_list

		assembled_inputs = self.assemble_inputs()

		# Input to the policy should be states and actions. 
		self.state_action_inputs = torch.tensor(assembled_inputs).to(device).float()	

		# Get summed reward for statistics. 
		self.batch_reward_statistics += sum(self.reward_trajectory)

	def set_differentiable_critic_inputs(self):
		# Get policy's predicted actions by getting action greedily, then add noise. 				
		predicted_action = self.policy_network.reparameterized_get_actions(self.state_action_inputs, greedy=True).squeeze(1)
		noise = torch.zeros_like(predicted_action).to(device).float()
		
		# Get noise from noise process. 					
		noise = torch.randn_like(predicted_action).to(device).float()*self.epsilon

		# Concatenate the states from policy inputs and the predicted actions. 
		self.critic_inputs = torch.cat([self.state_action_inputs[:,:self.output_size], predicted_action, self.state_action_inputs[:,2*self.output_size:]],axis=1).to(device).float()

	def update_policies(self):
		######################################
		# Compute losses for actor.
		self.set_differentiable_critic_inputs()		

		self.policy_optimizer.zero_grad()
		self.policy_loss = - self.critic_network.forward(self.critic_inputs[:-1]).mean()
		self.policy_loss_statistics += self.policy_loss.clone().detach().cpu().numpy().mean()
		self.policy_loss.backward()
		self.policy_optimizer.step()

	def set_targets(self):
		if self.args.TD:
			# Construct TD Targets. 
			self.TD_targets = self.critic_predictions.clone().detach().cpu().numpy()
			# Select till last time step, because we don't care what critic says after last timestep.
			self.TD_targets = np.roll(self.TD_targets,-1,axis=0)[:-1]
			# Mask with terminal. 
			self.TD_targets = self.gamma*np.array(self.terminal_trajectory)*self.TD_targets		
			self.TD_targets += np.array(self.reward_trajectory)
			self.critic_targets = torch.tensor(self.TD_targets).to(device).float()
		else:
			self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])
			self.critic_targets = torch.tensor(self.cummulative_rewards).to(device).float()

	def update_critic(self):
		######################################
		# Zero gradients, then backprop into critic.		
		self.critic_optimizer.zero_grad()	
		# Get critic predictions first. 
		if self.args.MLP_policy:
			self.critic_predictions = self.critic_network.forward(self.state_action_inputs).squeeze(1)
		else:
			self.critic_predictions = self.critic_network.forward(self.state_action_inputs).squeeze(1).squeeze(1)

		# Before we actually compute loss, compute targets.
		self.set_targets()

		# We predicted critic values from states S_1 to S_{T+1} because we needed all for bootstrapping. 
		# For loss, we don't actually need S_{T+1}, so throw it out.
		self.critic_loss = self.MSE_Loss(self.critic_predictions[:-1], self.critic_targets).mean()
		self.critic_loss_statistics += self.critic_loss.clone().detach().cpu().numpy().mean()	
		self.critic_loss.backward()
		self.critic_optimizer.step()
		######################################

	def update_networks(self):
		# Update policy network. 
		self.update_policies()
		# Now update critic network.
		self.update_critic()

	def reset_statistics(self):
		# Can also reset the policy and critic loss statistcs here. 
		self.policy_loss_statistics = 0.
		self.critic_loss_statistics = 0.
		self.batch_reward_statistics = 0.
		self.episode_reward_statistics = 0.

	def update_batch(self):

		# Get set of indices of episodes in the memory. 
		batch_indices = self.memory.sample_batch(self.batch_size)

		for ind in batch_indices:

			# Retrieve appropriate episode from memory. 
			episode = self.memory.memory[ind]

			# Set quantities from episode.
			self.process_episode(episode)

			# Now compute gradients to both networks from batch.
			self.update_networks()

	def update_plots(self, counter):

		if counter%self.args.display_freq==0:

			# print("Embedding in Update Plots.")
			
			# Rollout policy.
			self.rollout(random=False, test=True, visualize=True)

		# Now that we've updated these into TB, reset stats. 
		self.reset_statistics()

	def run_iteration(self, counter, evaluate=False):

		# This is really a run episode function. Ignore the index, just use the counter. 
		# 1) 	Rollout trajectory. 
		# 2) 	Collect stats / append to memory and stuff.
		# 3) 	Update policies. 

		self.set_parameters(counter, evaluate=evaluate)
		# Maintain counter to keep track of updating the policy regularly. 			

		# cProfile.runctx('self.rollout()',globals(), locals(),sort='cumtime')
		self.rollout(random=False, test=evaluate)

		if self.args.train and not(evaluate):

			# If training, append to memory. 
			self.memory.append_to_memory(self.episode)
			# Update on batch. 
			self.update_batch()
			# Update plots. 
			self.update_plots(counter)

	def initialize_memory(self):

		# Create memory object. 
		self.memory = RLUtils.ReplayMemory(memory_size=self.args.memory_size)

		# Number of initial episodes needs to be less than memory size. 
		self.initial_episodes = self.args.burn_in_eps

		# While number of transitions is less than initial_transitions.
		episode_counter = 0 
		while episode_counter<self.initial_episodes:

			# Reset the noise process! We forgot to do this! :( 
			self.NoiseProcess.reset()

			print("Initializing Memory Episode: ", episode_counter)
			# Rollout an episode.
			self.rollout(random=self.args.random_memory_burn_in)

			# Add episode to memory.
			self.memory.append_to_memory(self.episode)

			episode_counter += 1			

	def evaluate(self, epoch=None, model=None):

		if model is not None:
			print("Loading model in training.")
			self.load_all_models(model)
			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		else:
			model_epoch = epoch

		self.total_rewards = np.zeros((self.number_test_episodes))

		# For number of test episodes. 
		for eps in range(self.number_test_episodes):
			# Run an iteration (and rollout)...
			self.run_iteration(eps, evaluate=True)
			self.total_rewards[eps] = np.array(self.reward_trajectory).sum()

		# Create save directory to save these results. 
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Total_Rewards_{0}.npy".format(self.args.name)),self.total_rewards)
		np.save(os.path.join(self.dir_name,"Mean_Reward_{0}.npy".format(self.args.name)),self.total_rewards.mean())

	def train(self, model=None):

		# 1) Initialize memory maybe.
		# 2) For number of iterations, RUN ITERATION:
		# 3) 	Rollout trajectory. 
		# 4) 	Collect stats. 
		# 5) 	Update policies. 

		if model:
			print("Loading model in training.")
			self.load_all_models(model)

		print("Starting Main Training Procedure.")
		self.set_parameters(0)

		np.set_printoptions(suppress=True,precision=2)

		# Fixing seeds.
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)		

		print("Initializing Memory.")
		self.initialize_memory()

		for e in range(self.number_episodes):
			
			# Reset the noise process! We forgot to do this! :( 
			self.NoiseProcess.reset()

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			self.run_iteration(e)
			print("#############################")
			print("Running Episode: ",e)

			if e%self.args.eval_freq==0:
				self.evaluate(epoch=e, model=None)

class PolicyManager_DownstreamRL(PolicyManager_BaselineRL):

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_DownstreamRL, self).__init__(number_policies=4, dataset=dataset, args=args)

	def setup(self):
		# Create Mujoco environment. 
		import robosuite
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.environment.action_spec[0].shape[0]
		self.conditional_info_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		# If we are loading policies....
		if self.args.model:
			# Padded conditional info.
			self.conditional_info_size = 53		
		self.input_size = 2*self.state_size
		
		# Create networks. 
		self.create_networks()
		self.create_training_ops()
		
		self.initialize_plots()

		self.gripper_open = np.array([0.0115, -0.0115])
		self.gripper_closed = np.array([-0.020833, 0.020833])

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size)

	def create_networks(self):
		# Copying over the create networks from Joint Policy training. 

		# Not sure if there's a better way to inherit - unless we inherit from both classes.
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers).to(device)				
		self.critic_network = CriticNetwork(self.input_size+self.conditional_info_size, self.args.hidden_size, 1, self.args, self.args.number_layers).to(device)

		if self.args.constrained_b_prior:
			self.latent_policy = ContinuousLatentPolicyNetwork_ConstrainedBPrior(self.input_size+self.conditional_info_size, self.args.hidden_size, self.args, self.args.number_layers).to(device)
		else:
			self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size+self.conditional_info_size, self.args.hidden_size, self.args, self.args.number_layers).to(device)

	def create_training_ops(self):
		
		self.NLL_Loss = torch.nn.NLLLoss(reduction='none')
		self.MSE_Loss = torch.nn.MSELoss(reduction='none')
		
		# If we are using reparameterization, use a global optimizer for both policies, and a global loss function.
		parameter_list = list(self.latent_policy.parameters())
		if not(self.args.fix_subpolicy):
			parameter_list = parameter_list + list(self.policy_network.parameters())		
		# The policy optimizer handles both the low and high level policies, as long as the z's being passed from the latent to sub policy are differentiable.
		self.policy_optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)
		self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

	def save_all_models(self, suffix):
		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")	
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Latent_Policy'] = self.latent_policy.state_dict()
		save_object['Critic_Network'] = self.critic_network.state_dict()
		
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, critic=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		if self.args.load_latent:
			self.latent_policy.load_state_dict(load_object['Latent_Policy'])
		if critic:
			self.critic_network.load_state_dict(load_object['Critic_Network'])

	def reset_lists(self):
		self.reward_trajectory = []
		self.state_trajectory = []
		self.action_trajectory = []
		self.image_trajectory = []
		self.terminal_trajectory = []
		self.latent_z_trajectory = []
		self.latent_b_trajectory = []
		self.cummulative_rewards = None
		self.episode = None

	def get_conditional_information_row(self, t=-1):
		# Get robot and object state.
		conditional_info_row = np.zeros((1,self.conditional_info_size))
		info_value = np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1)		
		conditional_info_row[0,:info_value.shape[1]] = info_value

		return conditional_info_row

	def get_transformed_gripper_value(self, gripper_finger_values):
		gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)			

		finger_diff = gripper_values[1]-gripper_values[0]
		gripper_value = np.array(2*finger_diff-1).reshape((1,-1))
		return gripper_value

	def get_current_input_row(self, t=-1):

		# The state that we want is ... joint state?
		gripper_finger_values = self.state_trajectory[t]['gripper_qpos']

		if len(self.action_trajectory)==0 or t==0:
			return np.concatenate([self.state_trajectory[0]['joint_pos'].reshape((1,-1)), np.zeros((1,1)), np.zeros((1,self.output_size))],axis=1)
		elif t==-1:
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t].reshape((1,-1))],axis=1)
		else: 
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t-1].reshape((1,-1))],axis=1)

	def get_latent_input_row(self, t=-1):
		# If first timestep, z's are 0 and b is 1. 
		if len(self.latent_z_trajectory)==0 or t==0:
			return np.concatenate([np.zeros((1, self.args.z_dimensions)),np.ones((1,1))],axis=1)
		if t==-1:
			return np.concatenate([self.latent_z_trajectory[t].reshape((1,-1)),self.latent_b_trajectory[t].reshape((1,1))],axis=1)
		elif t>0:
			t-=1	
			return np.concatenate([self.latent_z_trajectory[t].reshape((1,-1)),self.latent_b_trajectory[t].reshape((1,1))],axis=1)

	def assemble_latent_input_row(self, t=-1):
		# Function to assemble ONE ROW of latent policy input. 
		# Remember, the latent policy takes.. JOINT_states, actions, z's, b's, and then conditional information of robot-state and object-state. 

		# Assemble these three pieces: 
		return np.concatenate([self.get_current_input_row(t), self.get_latent_input_row(t), self.get_conditional_information_row(t)],axis=1)

	def assemble_latent_inputs(self):
		# Assemble latent policy inputs over time.
		return np.concatenate([self.assemble_latent_input_row(t) for t in range(len(self.state_trajectory))],axis=0)		

	def assemble_subpolicy_input_row(self, latent_z=None, t=-1):
		# Remember, the subpolicy takes.. JOINT_states, actions, z's. 
		# Assemble (remember, without b, and without conditional info).

		if latent_z is not None:
			# return np.concatenate([self.get_current_input_row(t), latent_z.reshape((1,-1))],axis=1)

			# Instead of numpy, use torch. 
			return torch.cat([torch.tensor(self.get_current_input_row(t)).to(device).float(), latent_z.reshape((1,-1))],dim=1)
		else:
			# Remember, get_latent_input_row isn't operating on something that needs to be differentiable, so just use numpy and then wrap with torch tensor. 
			# return torch.tensor(np.concatenate([self.get_current_input_row(t), self.get_latent_input_row(t)[:,:-1]],axis=1)).to(device).float()
			return torch.tensor(np.concatenate([self.get_current_input_row(t), self.latent_z_trajectory[t].reshape((1,-1))],axis=1)).to(device).float()

	def assemble_subpolicy_inputs(self, latent_z_list=None):
		# Assemble sub policy inputs over time.	
		if latent_z_list is None:
			# return np.concatenate([self.assemble_subpolicy_input_row(t) for t in range(len(self.state_trajectory))],axis=0)

			# Instead of numpy, use torch... 
			return torch.cat([self.assemble_subpolicy_input_row(t=t) for t in range(len(self.state_trajectory))],dim=0)
		else:
			# return np.concatenate([self.assemble_subpolicy_input_row(t, latent_z=latent_z_list[t]) for t in range(len(self.state_trajectory))],axis=0)

			# Instead of numpy, use torch... 
			return torch.cat([self.assemble_subpolicy_input_row(t=t, latent_z=latent_z_list[t]) for t in range(len(self.state_trajectory))],dim=0)

	def assemble_state_action_row(self, action=None, t=-1):
		# Get state action input row for critic.
		if action is not None:

			gripper_finger_values = self.state_trajectory[t]['gripper_qpos']
			gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = np.array(2*finger_diff-1).reshape((1,-1))

			# Don't create a torch tensor out of actions. 
			return torch.cat([torch.tensor(self.state_trajectory[t]['joint_pos']).to(device).float().reshape((1,-1)), torch.tensor(gripper_value).to(device).float(), action.reshape((1,-1)), torch.tensor(self.get_conditional_information_row(t)).to(device).float()],dim=1)
		else:		
			# Just use actions that were used in the trajectory. This doesn't need to be differentiable, because it's going to be used for the critic targets, so just make a torch tensor from numpy. 
			return torch.tensor(np.concatenate([self.get_current_input_row(t), self.get_conditional_information_row(t)],axis=1)).to(device).float()

	def assemble_state_action_inputs(self, action_list=None):
		# return np.concatenate([self.assemble_state_action_row(t) for t in range(len(self.state_trajectory))],axis=0)
		
		# Instead of numpy use torch.
		if action_list is not None:
			return torch.cat([self.assemble_state_action_row(t=t, action=action_list[t]) for t in range(len(self.state_trajectory))],dim=0)
		else:
			return torch.cat([self.assemble_state_action_row(t=t) for t in range(len(self.state_trajectory))],dim=0)

	def get_OU_action_latents(self, policy_hidden=None, latent_hidden=None, random=False, counter=0, previous_z=None, test=False, delta_t=0):

		# if random==True:
		# 	action = 2*np.random.random((self.output_size))-1
		# 	return action, 

		# Get latent policy inputs.
		latent_policy_inputs = self.assemble_latent_input_row()
		
		# Feed in latent policy inputs and get the latent policy outputs (z, b, and hidden)
		latent_z, latent_b, latent_hidden = self.latent_policy.incremental_reparam_get_actions(torch.tensor(latent_policy_inputs).to(device).float(), greedy=True, hidden=latent_hidden, previous_z=previous_z, delta_t=delta_t)

		# Perturb latent_z with some noise. 
		z_noise = self.epsilon*torch.randn_like(latent_z)
		# Add noise to z.
		latent_z = latent_z + z_noise

		if latent_b[-1]==1:
			delta_t = 0
		else:
			delta_t += 1

		# Now get subpolicy inputs.
		# subpolicy_inputs = self.assemble_subpolicy_input_row(latent_z.detach().cpu().numpy())
		subpolicy_inputs = self.assemble_subpolicy_input_row(latent_z=latent_z)

		# Feed in subpolicy inputs and get the subpolicy outputs (a, hidden)
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(subpolicy_inputs).to(device).float(), greedy=True, hidden=policy_hidden)

		# Numpy action
		action = predicted_action[-1].squeeze(0).detach().cpu().numpy()		
		
		if test:
			perturbed_action = action
		else:	
			# Perturb action with noise. 			
			if self.args.OU:
				perturbed_action = self.NoiseProcess.get_action(action, counter)

			else:
				# Just regular epsilon
				perturbed_action = action + self.epsilon*np.random.randn(action.shape[-1])

		return perturbed_action, latent_z, latent_b, policy_hidden, latent_hidden, delta_t

	def rollout(self, random=False, test=False, visualize=False):
		
		# Reset the noise process! We forgot to do this! :( 
		self.NoiseProcess.reset()

		# Reset some data for the rollout. 
		counter = 0		
		eps_reward = 0.			
		terminal = False
		self.reset_lists()

		# Reset environment and add state to the list.
		state = self.environment.reset()
		self.state_trajectory.append(state)		

		# If we are going to visualize, get an initial image.
		if visualize:			
			image = self.environment.sim.render(600,600, camera_name='frontview')
			self.image_trajectory.append(np.flipud(image))

		# Instead of maintaining just one LSTM hidden state... now have one for each policy level.
		policy_hidden = None
		latent_hidden = None
		latent_z = None

		delta_t = 0		

		# For number of steps / while we don't terminate:
		while not(terminal) and counter<self.max_timesteps:

			# Get the action to execute, b, z, and hidden states. 
			action, latent_z, latent_b, policy_hidden, latent_hidden, delta_t = self.get_OU_action_latents(policy_hidden=policy_hidden, latent_hidden=latent_hidden, random=random, counter=counter, previous_z=latent_z, test=test, delta_t=delta_t)

			if self.args.debug:
				print("Embed in Trajectory Rollout.")
				embed()

			# Take a step in the environment. 	
			next_state, onestep_reward, terminal, success = self.environment.step(action)
			
			# Append everything to lists. 
			self.state_trajectory.append(next_state)
			self.action_trajectory.append(action)
			self.reward_trajectory.append(onestep_reward)
			self.terminal_trajectory.append(terminal)
			self.latent_z_trajectory.append(latent_z.detach().cpu().numpy())
			self.latent_b_trajectory.append(latent_b.detach().cpu().numpy())

			# Copy next state into state. 
			state = copy.deepcopy(next_state)

			# Counter
			counter += 1 

			# Append image to image list if we are visualizing. 
			if visualize:
				image = self.environment.sim.render(600,600, camera_name='frontview')
				self.image_trajectory.append(np.flipud(image))
				
		# Now that the episode is done, compute cummulative rewards... 
		self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])
		self.episode_reward_statistics = copy.deepcopy(self.cummulative_rewards[0])
		
		print("Rolled out an episode for ",counter," timesteps.")
		print("Achieved reward: ", self.episode_reward_statistics)

		# NOW construct an episode out of this..	
		self.episode = RLUtils.HierarchicalEpisode(self.state_trajectory, self.action_trajectory, self.reward_trajectory, self.terminal_trajectory, self.latent_z_trajectory, self.latent_b_trajectory)

	def process_episode(self, episode):
		# Assemble states, actions, targets.

		# First reset all the lists from the rollout now that they've been written to memory. 
		self.reset_lists()

		# Now set the lists. 
		self.state_trajectory = episode.state_list
		self.action_trajectory = episode.action_list
		self.reward_trajectory = episode.reward_list
		self.terminal_trajectory = episode.terminal_list
		self.latent_z_trajectory = episode.latent_z_list
		self.latent_b_trajectory = episode.latent_b_list

		# Get summed reward for statistics. 
		self.batch_reward_statistics += sum(self.reward_trajectory)

		# Assembling state_action inputs to feed to the Critic network for TARGETS. (These don't need to, and in fact shouldn't, be differentiable).
		self.state_action_inputs = torch.tensor(self.assemble_state_action_inputs()).to(device).float()

	def update_policies(self):
		# There are a few steps that need to be taken. 
		# 1) Assemble latent policy inputs.
		# 2) Get differentiable latent z's from latent policy. 
		# 3) Assemble subpolicy inputs with these differentiable latent z's. 
		# 4) Get differentiable actions from subpolicy. 
		# 5) Assemble critic inputs with these differentiable actions. 
		# 6) Now compute critic predictions that are differentiable w.r.t. sub and latent policies. 
		# 7) Backprop.

		# 1) Assemble latent policy inputs. # Remember, these are the only things that don't need to be differentiable.
		self.latent_policy_inputs = torch.tensor(self.assemble_latent_inputs()).to(device).float()		

		# 2) Feed this into latent policy. 
		latent_z, latent_b, _ = self.latent_policy.incremental_reparam_get_actions(torch.tensor(self.latent_policy_inputs).to(device).float(), greedy=True)

		# 3) Assemble subpolicy inputs with diff latent z's. Remember, this needs to be differentiable. Modify the assembling to torch, WITHOUT creating new torch tensors of z. 

		self.subpolicy_inputs = self.assemble_subpolicy_inputs(latent_z_list=latent_z)

		# 4) Feed into subpolicy. 
		diff_actions, _ = self.policy_network.incremental_reparam_get_actions(self.subpolicy_inputs, greedy=True)

		# 5) Now assemble critic inputs. 
		self.differentiable_critic_inputs = self.assemble_state_action_inputs(action_list=diff_actions)

		# 6) Compute critic predictions. 
		self.policy_loss = - self.critic_network.forward(self.differentiable_critic_inputs[:-1]).mean()

		# Also log statistics. 
		self.policy_loss_statistics += self.policy_loss.clone().detach().cpu().numpy().mean()

		# 7) Now backprop into policy.
		self.policy_optimizer.zero_grad()		
		self.policy_loss.backward()
		self.policy_optimizer.step()

class PolicyManager_DMPBaselines(PolicyManager_Joint):

	# Make it inherit joint policy manager init.
	def __init__(self, number_policies=4, dataset=None, args=None):
		super(PolicyManager_DMPBaselines, self).__init__(number_policies, dataset, args)

	def setup_DMP_parameters(self):
		self.output_size 
		self.number_kernels = 15
		self.window = 15
		self.kernel_bandwidth = 1.5

		self.number_kernels = self.args.baseline_kernels
		self.window = self.args.baseline_window
		self.kernel_bandwidth = self.args.baseline_kernel_bandwidth

	def get_MSE(self, sample_traj, trajectory_rollout):
		# Evaluate MSE between reconstruction and sample trajectory. 
		return ((sample_traj-trajectory_rollout)**2).mean()

	def get_FlatDMP_rollout(self, sample_traj, velocities=None):
		# Reinitialize DMP Class. 
		self.dmp = DMP.DMP(time_steps=len(sample_traj), num_ker=self.number_kernels, dimensions=self.state_size, kernel_bandwidth=self.kernel_bandwidth, alphaz=5., time_basis=True)

		# Learn DMP for particular trajectory.
		self.dmp.learn_DMP(sample_traj)

		# Get rollout. 
		if velocities is not None: 
			trajectory_rollout = self.dmp.rollout(sample_traj[0],sample_traj[-1],velocities)
		else:
			trajectory_rollout = self.dmp.rollout(sample_traj[0],sample_traj[-1],np.zeros((self.state_size)))

		return trajectory_rollout

	def evaluate_FlatDMPBaseline_iteration(self, index, sample_traj):
		trajectory_rollout = self.get_FlatDMP_rollout(sample_traj)
		self.FlatDMP_distances[index] = self.get_MSE(sample_traj, trajectory_rollout)

	def get_AccelerationChangepoint_rollout(self, sample_traj):

		# Get magnitudes of acceleration across time.
		acceleration_norm = np.linalg.norm(np.diff(sample_traj,n=2,axis=0),axis=1)

		# Get velocities. 
		velocities = np.diff(sample_traj,n=1,axis=0,prepend=sample_traj[0].reshape((1,-1)))

		# Find peaks with minimum length = 8.
		window = self.window
		segmentation = find_peaks(acceleration_norm, distance=window)[0]
		
		if len(segmentation)==0:
			segmentation = np.array([0,len(sample_traj)])
		else:
			# Add start and end to peaks. 
			if segmentation[0]<window:
				segmentation[0] = 0
			else:
				segmentation = np.insert(segmentation, 0, 0)
			# If end segmentation is within WINDOW of end, change segment to end. 
			if (len(sample_traj) - segmentation[-1])<window:
				segmentation[-1] = len(sample_traj)
			else:
				segmentation = np.insert(segmentation, len(segmentation), sample_traj.shape[0])

		trajectory_rollout = np.zeros_like(sample_traj)		

		# For every segment.
		for i in range(len(segmentation)-1):
			# Get trajectory segment. 
			trajectory_segment = sample_traj[segmentation[i]:segmentation[i+1]]

			# Get rollout. # Feed velocities into rollout. # First velocity is 0. 
			segment_rollout = self.get_FlatDMP_rollout(trajectory_segment, velocities[segmentation[i]])

			# Copy segment rollout into full rollout. 
			trajectory_rollout[segmentation[i]:segmentation[i+1]] = segment_rollout

		return trajectory_rollout

	def evaluate_AccelerationChangepoint_iteration(self, index, sample_traj):
		trajectory_rollout = self.get_AccelerationChangepoint_rollout(sample_traj)
		self.AccChangepointDMP_distances[index] = self.get_MSE(sample_traj, trajectory_rollout)

	def evaluate_MeanRegression_iteration(self, index, sample_traj):
		mean = sample_traj.mean(axis=0)
		self.MeanRegression_distances[index] = ((sample_traj-mean)**2).mean()

	def get_GreedyDMP_rollout(self, sample_traj):
		pass		

	def evaluate_across_testset(self):

		self.setup_DMP_parameters()
		# Create array for distances. 
		self.FlatDMP_distances = -np.ones((self.test_set_size))
		self.AccChangepointDMP_distances = -np.ones((self.test_set_size))
		self.MeanRegression_distances = -np.ones((self.test_set_size))
		self.lengths = -np.ones((self.test_set_size))

		for i in range(self.test_set_size):

			# Set actual index. 
			index = i + len(self.dataset) - self.test_set_size

			if i%100==0:
				print("Evaluating Datapoint ", i)

			# Get trajectory. 
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

			if sample_traj is not None: 

				# Set sample trajectory to ignore gripper. 
				if self.args.data in ['MIME','OldMIME']:
					sample_traj = sample_traj[:,:-2]
					self.state_size = 14
				elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
					sample_traj = sample_traj[:,:-1]
					self.state_size = 7	

					
					# sample_traj = gaussian_filter1d(sample_traj,3.5,axis=0,mode='nearest')
				# elif self.args.data=='Mocap':
				# 	sample_traj = sample_traj
					
				self.lengths[i] = len(sample_traj)

				# Eval Flat DMP.
				self.evaluate_FlatDMPBaseline_iteration(i, sample_traj)

				# Eval AccChange DMP Baseline.
				self.evaluate_AccelerationChangepoint_iteration(i, sample_traj)

				# Evaluate Mean regression Basleine. 
				self.evaluate_MeanRegression_iteration(i, sample_traj)

		# self.mean_distance = self.distances[self.distances>0].mean()		
		print("Average Distance of Flat DMP Baseline: ", self.FlatDMP_distances[self.FlatDMP_distances>0].mean())
		print("Average Distance of Acceleration Changepoint Baseline: ", self.AccChangepointDMP_distances[self.AccChangepointDMP_distances>0].mean())
		print("Average Distance of Mean Regression Baseline: ", self.MeanRegression_distances[self.MeanRegression_distances>0].mean())

		embed()

class PolicyManager_Imitation(PolicyManager_Pretrain, PolicyManager_BaselineRL):

	def __init__(self, number_policies=4, dataset=None, args=None):	
		super(PolicyManager_Imitation, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		# Explicitly run inits to make sure inheritance is good.
		# PolicyManager_Pretrain.__init__(self, number_policies, dataset, args)
		# PolicyManager_BaselineRL.__init__(self, args)

		# Set train only policy to true.
		self.args.train_only_policy = 1

		# Get task index from task name.
		self.demo_task_index = np.where(np.array(self.dataset.environment_names)==self.args.environment)[0][0]

	def setup(self):
		# Fixing seeds.
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)
		np.set_printoptions(suppress=True,precision=2)

		# Create index list.
		extent = self.dataset.get_number_task_demos(self.demo_task_index)
		self.index_list = np.arange(0,extent)	

		# Create Mujoco environment. 
		import robosuite
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		self.gripper_open = np.array([0.0115, -0.0115])
		self.gripper_closed = np.array([-0.020833, 0.020833])

		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()		
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]

		self.conditional_info_size = self.state_size
		# Input size.. state, action, conditional
		self.input_size = self.state_size + self.output_size*2

		# Create networks. 
		self.create_networks()
		self.create_training_ops()		
		self.initialize_plots()

		self.total_rewards = 0.

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size)

	def create_networks(self):

		# We don't need a decoder.
		# Policy Network is the only thing we need.
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])

	def update_policies(self, logprobabilities):

		# Set gradients to 0.
		self.optimizer.zero_grad()

		# Set policy loss. 
		self.policy_loss = -logprobabilities[:-1].mean()

		# Backward. 
		self.policy_loss.backward()

		# Take a step. 
		self.optimizer.step()

	def update_plots(self, counter, logprobabilities):

		if counter%self.args.display_freq==0:

			# print("Embedding in Update Plots.")
			
			# Rollout policy.
			self.rollout(random=False, test=True, visualize=True)
			
	def run_iteration(self, counter, i):

		self.set_epoch(counter)	
		self.iter = counter

		############# (0) #############
		# Get sample we're going to train on.		
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		if sample_traj is not None:			
			# Now concatenate info with... conditional_information
			policy_inputs = np.concatenate([concatenated_traj, self.conditional_information], axis=1) 	

			# Add zeros to the last action, so that we evaluate likelihood correctly. Since we're using demo actions, no need.
			# padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)

			# Feed concatenated trajectory into the policy. 
			logprobabilities, _ = self.policy_network.forward(torch.tensor(policy_inputs).to(device).float(), sample_action_seq)

			if self.args.train:
				if self.args.debug:
					if self.iter%self.args.debug==0:
						print("Embedding in Train Function.")
						embed()
				
				# Update policy. 						
				self.update_policies(logprobabilities)

				# Update plots.
				self.update_plots(counter, logprobabilities)

	def get_transformed_gripper_value(self, gripper_finger_values):
		gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)					
		finger_diff = gripper_values[1]-gripper_values[0]	
		gripper_value = np.array(2*finger_diff-1).reshape((1,-1))

		return gripper_value

	def get_state_action_row(self, t=-1):

		# The state that we want is ... joint state?
		gripper_finger_values = self.state_trajectory[t]['gripper_qpos']

		if len(self.action_trajectory)==0 or t==0:
			return np.concatenate([self.state_trajectory[0]['joint_pos'].reshape((1,-1)), np.zeros((1,1)), np.zeros((1,self.output_size))],axis=1)
		elif t==-1:
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t].reshape((1,-1))],axis=1)
		else: 
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t-1].reshape((1,-1))],axis=1)

	def get_current_input_row(self, t=-1):
		# Rewrite this funciton so that the baselineRL Rollout class can still use it here...
		# First get conditional information.

		# Get robot and object state.
		conditional_info = np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1)		

		# Get state actions..
		state_action = self.get_state_action_row()

		# Concatenate.
		input_row = np.concatenate([state_action, conditional_info],axis=1)

		return input_row

	def evaluate(self, epoch=None, model=None):

		if model is not None:
			self.load_all_models(model)
			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		else:
			model_epoch = epoch

		self.total_rewards = np.zeros((self.number_test_episodes))

		# Set parameters like epsilon.
		self.set_parameters(0, evaluate=True)

		# For number of test episodes.
		for eps in range(self.number_test_episodes):
			# Now run a rollout. 
			self.rollout(random=False, test=True)

			self.total_rewards[eps] = np.array(self.reward_trajectory).sum()

		# Create save directory to save these results. 
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Total_Rewards_{0}.npy".format(self.args.name)),self.total_rewards)
		np.save(os.path.join(self.dir_name,"Mean_Reward_{0}.npy".format(self.args.name)),self.total_rewards.mean())

	def train(self, model=None):

		if model:
			print("Loading model in training.")
			self.load_all_models(model)		
		counter = 0

		# For number of training epochs. 
		for e in range(self.number_epochs): 
			
			print("Starting Epoch: ",e)

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			# self.automatic_evaluation(e)
			np.random.shuffle(self.index_list)

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			# For every item in the epoch:
			if self.args.setting=='imitation':
				extent = self.dataset.get_number_task_demos(self.demo_task_index)
			else:
				extent = len(self.dataset)-self.test_set_size

			for i in range(extent):

				print("Epoch: ",e," Trajectory:",i, "Datapoint: ", self.index_list[i])
				self.run_iteration(counter, self.index_list[i])				

				counter = counter+1

			if e%self.args.eval_freq==0:
				self.evaluate(e)

		self.write_and_close()

class PolicyManager_Transfer(PolicyManager_BaseClass):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_Transfer, self).__init__()

		# The inherited functions refer to self.args. Also making this to make inheritance go smooth.
		self.args = args

		# Before instantiating policy managers of source or target domains; create copies of args with data attribute changed. 		
		self.source_args = copy.deepcopy(args)
		self.source_args.data = self.source_args.source_domain
		self.source_dataset = source_dataset

		self.target_args = copy.deepcopy(args)
		self.target_args.data = self.target_args.target_domain
		self.target_dataset = target_dataset
		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_epochs = self.args.epsilon_over

		# Now create two instances of policy managers for each domain. Call them source and target domain policy managers. 
		if self.args.setting in ['Transfer']:
			if self.args.batch_size>1:
				self.source_manager = PolicyManager_BatchPretrain(dataset=self.source_dataset, args=self.source_args)
				self.target_manager = PolicyManager_BatchPretrain(dataset=self.target_dataset, args=self.target_args)		
			else:
				self.source_manager = PolicyManager_Pretrain(dataset=self.source_dataset, args=self.source_args)
				self.target_manager = PolicyManager_Pretrain(dataset=self.target_dataset, args=self.target_args)		

			self.source_dataset_size = len(self.source_manager.dataset) - self.source_manager.test_set_size
			self.target_dataset_size = len(self.target_manager.dataset) - self.target_manager.test_set_size


			# Now setup networks for these PolicyManagers. 		
			self.source_manager.setup()
			self.target_manager.setup()

			# Now create variables that we need. 
			self.number_epochs = self.args.epochs
			self.extent = min(self.source_dataset_size, self.target_dataset_size)		
			self.decay_counter = self.decay_epochs*self.extent
			self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)

		# Now define other parameters that will be required for the discriminator, etc. 
		self.input_size = self.args.z_dimensions
		self.hidden_size = self.args.hidden_size
		self.output_size = 2
		self.learning_rate = self.args.learning_rate
		self.already_shuffled = 0

	def set_iteration(self, counter, i=0):

		# Set epsilon.
		if counter<self.decay_counter:
			self.epsilon = self.initial_epsilon-self.decay_rate*counter
		else:
			self.epsilon = self.final_epsilon	

		self.counter = counter
		# Based on what phase of training we are in, set discriminability loss weight, etc. 
		
		# Phase 1 of training: Don't train discriminator at all, set discriminability loss weight to 0.
		if counter<self.args.training_phase_size:
			self.discriminability_loss_weight = 0.
			self.z_trajectory_discriminability_loss_weight = 0.			
			self.vae_loss_weight = 1.
			self.training_phase = 1
			self.skip_vae = 0
			self.skip_discriminator = 1

		# Phase 2 of training: Train the discriminator, and set discriminability loss weight to original.
		else:

			self.vae_loss_weight = self.args.vae_loss_weight
			self.discriminability_loss_weight = self.args.discriminability_weight
			self.z_trajectory_discriminability_loss_weight = self.args.z_trajectory_discriminability_weight

			# if self.args.training_phase_size<=counter and counter<self.args.training_phase_size*2:
			# 	# self.z_transform_discriminability_loss_weight = self.args.z_transform_discriminability_weights
			# 	self.z_transform_discriminability_loss_weight = 0.				
			# else:								
			# 	self.z_transform_discriminability_loss_weight = self.args.z_transform_discriminability_weight

			# Now make discriminator and vae train in alternating fashion. 
			# Set number of iterations of alteration. 

			# Train discriminator for k times as many steps as VAE. Set args.alternating_phase_size as 1 for this. 
			# Instead of using discriminator_phase_size steps for every 1 generator step.
			# Now, training generator / VAE for generator_phase_size. 

			# First get how many alternating phases we've completed so far. 
			completed_alternating_training_phases = (counter//self.args.alternating_phase_size)
			# Now figure out how many stages of discriminator phase sizes and generator phase sizes we've completed.
			modulo_phase = completed_alternating_training_phases%(self.args.discriminator_phase_size+self.args.generator_phase_size)
			# If we haven't yet completed the right number of generator (discriminator) phase sizes is done, train the generator (discriminator). 
			train_generator = modulo_phase<self.args.generator_phase_size
	
			# Now switching to training the discriminator first, because in the case where we're using a translation model, discriminator is useless for first GPS number of phases. 
			# This causes losses to blow up. In the case where we're optimizing represeentations directly, the representations are already trained to reconstruct purely in training phase 1, before this alteranting stuff. 
			train_discriminator = modulo_phase<self.args.discriminator_phase_size
			train_generator = 1-train_discriminator

			if train_generator:
				# print("Training VAE.")
				self.skip_discriminator = 1
				self.skip_vae = 0
			else:
				# print("Training Discriminator.")
				self.skip_discriminator = 0
				self.skip_vae = 1
			
			self.training_phase = 2

		self.source_manager.set_epoch(counter)
		self.target_manager.set_epoch(counter)

		if self.args.new_supervision:
			# Always set to 1 if new sup
			self.supervised_datapoints_multiplier = 1. 
		else:
			
			# Check if i is less than the number of supervised datapoints.
			# If it is, then set the supervised_datapoints_multiplier to 1, otherwise set it to 0. to make sure the supervised loss isn't used for these datapoints..
			if self.args.number_of_supervised_datapoints == -1:
				# If fully supervised case..
				self.supervised_datapoints_multiplier = 1. 
			else:
				if i<self.args.number_of_supervised_datapoints:
					self.supervised_datapoints_multiplier = 1. 
				else:
					self.supervised_datapoints_multiplier = 0.
				
			# print("Iter: ", i, "Sup L W:", self.supervised_datapoints_multiplier)
	
	def create_networks(self):

		# Call create networks from each of the policy managers. 
		self.source_manager.create_networks()
		self.target_manager.create_networks()

		# Now must also create discriminator.
		if self.args.setting=='jointcycletransfer':
			self.discriminator_network = EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size).to(device)
		else:
			if self.args.wasserstein_gan or self.args.lsgan:
				# Implement discriminator as a critic, rather than an actualy classifier network.
				self.discriminator_network = CriticMLP(self.input_size, self.hidden_size, 1, args=self.args).to(device)
			else:
				self.discriminator_network = DiscreteMLP(self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)

	def create_training_ops(self):

		# print("FINALLY RUNNING CREATE TRAIN OPS")
		# # Call create training ops from each of the policy managers. Need these optimizers, because the encoder-decoders get a different loss than the discriminator. 
		if self.args.setting in ['jointtransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
			self.source_manager.learning_rate = self.args.transfer_learning_rate
			self.target_manager.learning_rate = self.args.transfer_learning_rate

		self.source_manager.create_training_ops()
		self.target_manager.create_training_ops()

		# Create BCE loss object. 
		# self.BCE_loss = torch.nn.BCELoss(reduction='None')		
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		
		# Create common optimizer for source, target, and discriminator networks. 
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator_network.parameters(),lr=self.learning_rate, weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):
		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)
		self.save_object = {}

		# Source
		self.save_object['Source_Policy_Network'] = self.source_manager.policy_network.state_dict()
		self.save_object['Source_Encoder_Network'] = self.source_manager.encoder_network.state_dict()
		# Target
		self.save_object['Target_Policy_Network'] = self.target_manager.policy_network.state_dict()
		self.save_object['Target_Encoder_Network'] = self.target_manager.encoder_network.state_dict()
		# Discriminator
		self.save_object['Discriminator_Network'] = self.discriminator_network.state_dict()				

		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Source
		self.source_manager.policy_network.load_state_dict(self.load_object['Source_Policy_Network'])
		self.source_manager.encoder_network.load_state_dict(self.load_object['Source_Encoder_Network'])
		# Target
		self.target_manager.policy_network.load_state_dict(self.load_object['Target_Policy_Network'])
		self.target_manager.encoder_network.load_state_dict(self.load_object['Target_Encoder_Network'])
		# Discriminator
		self.discriminator_network.load_state_dict(self.load_object['Discriminator_Network'])

	def load_domain_models(self):

		if self.args.source_subpolicy_model is not None:
			self.source_manager.load_all_models(self.args.source_subpolicy_model, just_subpolicy=True)
		elif self.args.source_model is not None:
			self.source_manager.load_all_models(self.args.source_model)
		if self.args.target_subpolicy_model is not None:
			self.target_manager.load_all_models(self.args.target_subpolicy_model, just_subpolicy=True)
		elif self.args.target_model is not None:
			self.target_manager.load_all_models(self.args.target_model)

		if self.args.fix_source:
			# # Make optimizer dummy optimizer. 			
			# self.source_dummy_layer = torch.nn.Linear(1,1)
			# self.source_manager.optimizer = torch.optim.Adam(self.source_dummy_layer.parameters(),lr=self.learning_rate)

			# Better way to fix models. 			
			temp_list = self.source_manager.parameter_list + list(self.source_manager.policy_network.parameters())
			for param in temp_list:
				param.requires_grad = False

		if self.args.fix_target:
			# self.target_dummy_layer = torch.nn.Linear(1,1)
			# self.target_manager.optimizer = torch.optim.Adam(self.target_dummy_layer.parameters(),lr=self.learning_rate)

			# Better way to fix models. 
			temp_list = self.target_manager.parameter_list + list(self.target_manager.policy_network.parameters())
			for param in temp_list:
				param.requires_grad = False

	def get_domain_manager(self, domain):
		# Create a list, and just index into this list. 
		domain_manager_list = [self.source_manager, self.target_manager]
		return domain_manager_list[domain]

	def get_trajectory_segment_tuple(self, source_manager, target_manager):

		# Sample indices. 
		source_index = np.random.randint(0, high=self.source_dataset_size)
		target_index = np.random.randint(0, high=self.target_dataset_size)

		# Get trajectory segments. 
		source_trajectory_segment, source_action_seq, _, _ = source_manager.get_trajectory_segment(source_manager.index_list[source_index])
		target_trajectory_segment, target_action_seq, _, _ = target_manager.get_trajectory_segment(target_manager.index_list[target_index])

		return source_trajectory_segment, source_action_seq, target_trajectory_segment, target_action_seq

	def encode_decode_trajectory(self, policy_manager, i, return_trajectory=False, trajectory_input=None):

		# This should basically replicate the encode-decode steps in run_iteration of the Pretrain_PolicyManager. 

		############# (0) #############
		# Sample trajectory segment from dataset. 			

		# Check if the index is too big. If yes, just sample randomly.
		if i >= len(policy_manager.dataset):
			i = np.random.randint(0, len(policy_manager.dataset))

		# This branch is only used in cycle-consistency training.
		if trajectory_input is not None: 			
			# Grab trajectory segment from tuple. 			
			torch_traj_seg = trajectory_input['differentiable_state_action_seq']
			trajectory_segment = trajectory_input['differentiable_state_action_seq'].clone().detach().cpu().numpy()

		else: 
			
			trajectory_segment, sample_action_seq, sample_traj, _ = policy_manager.get_trajectory_segment(i)
			# Torchify trajectory segment.
			torch_traj_seg = torch.tensor(trajectory_segment).to(device).float()
		
		if trajectory_segment is not None:
			############# (1) #############
			# Encode trajectory segment into latent z. 		
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = policy_manager.encoder_network.forward(torch_traj_seg, policy_manager.epsilon)

			########## (2) & (3) ##########
			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = policy_manager.construct_dummy_latents(latent_z)

			# If we are using the pre-computed trajectory input, (in second encode_decode call, from target trajectory to target latent z.)
			# Don't assemble trajectory in numpy, just takze the previous subpolicy_inputs, and then clone it and replace the latent z in it.
			# Remember, this branch is also only used in cycle consistency training. 
			if trajectory_input is not None: 

				# Now assigned trajectory_input['subpolicy_inputs'].clone() to SubPolicy_inputs, and then replace the latent z's.
				subpolicy_inputs = trajectory_input['subpolicy_inputs'].clone()

				if self.args.batch_size>1:
					subpolicy_inputs[:,:,2*policy_manager.state_dim:] = latent_z_seq
					# Now get "sample_action_seq" for forward function. 				
					sample_action_seq = subpolicy_inputs[:,:,policy_manager.state_dim:2*policy_manager.state_dim].clone()
				else:
					subpolicy_inputs[:,2*policy_manager.state_dim:] = latent_z_seq
					# Now get "sample_action_seq" for forward function. 				
					sample_action_seq = subpolicy_inputs[:,policy_manager.state_dim:2*policy_manager.state_dim].clone()

			else:
				# This branch gets executed for plain domain adversarial training, so the changes we made above don't affect domain adversarial training. 
				_, subpolicy_inputs, sample_action_seq = policy_manager.assemble_inputs(trajectory_segment, latent_z_seq, latent_b, sample_action_seq)

			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
			loglikelihoods, _ = policy_manager.policy_network.forward(subpolicy_inputs, sample_action_seq)		
			loglikelihood = loglikelihoods[:-1].mean()

			if return_trajectory:
				return sample_traj, latent_z
			else:
				return subpolicy_inputs, latent_z, loglikelihood, kl_divergence

		if return_trajectory:
			return None, None
		else:
			return None, None, None, None

	def free_memory(self):

		# Deleting objects from get_embeddings()
		del self.source_image, self.target_image, self.shared_image, self.samedomain_shared_embedding_image

		# Deleting copies of objects in update_plots.
		del self.viz_dictionary

		# Deleting objects from get_trajectory_visuals()
		if self.check_toy_dataset():
			del self.source_trajectory_image, self.source_reconstruction_image, self.target_trajectory_image, self.target_reconstruction_image

			# Removing nested objects. 
			del self.source_manager.gt_gif_list, self.source_manager.rollout_gif_list, self.target_manager.gt_gif_list, self.target_manager.rollout_gif_list
			# Setting lists so other functions don't complain.
			self.source_manager.gt_gif_list = [] 
			self.source_manager.rollout_gif_list = []
			self.target_manager.gt_gif_list = []
			self.target_manager.rollout_gif_list = []

			# Remove nested gif objects. 
			del self.source_manager.ground_truth_gif, self.source_manager.rollout_gif, self.target_manager.ground_truth_gif, self.target_manager.rollout_gif

	def check_same_domains(self):

		if (self.args.source_domain==self.args.target_domain) and (self.args.source_domain not in ['MIME']):
			return True
		if (self.args.source_domain==self.args.target_domain) and (self.args.source_domain in ['MIME']) and (self.args.source_single_hand==self.args.target_single_hand):
			return True
		return False

	def log_density_and_chamfer_metrics(self, counter, log_dict, viz_dict=None):

		##################################################
		# Plot density visualizations. Both directions?
		##################################################
	
		if counter%self.args.display_freq==0:
			log_dict = self.construct_density_embeddings(log_dict)		

		if self.check_same_domains() and self.args.eval_transfer_metrics and counter%self.args.metric_eval_freq==0:		
			log_dict['Aggregate Forward GMM Density'], log_dict['Aggregate Reverse GMM Density'] = self.compute_aggregate_GMM_densities()
			log_dict['Aggregate Chamfer Loss'] = self.compute_aggregate_chamfer_loss()

		if log_dict['Domain']==1 and self.check_same_domains() and self.args.supervised_set_based_density_loss:
			log_dict['forward_set_based_supervised_loss'], log_dict['backward_set_based_supervised_loss'] = viz_dict['forward_set_based_supervised_loss'], viz_dict['backward_set_based_supervised_loss']

		return log_dict

	def update_plots(self, counter, viz_dict=None, log=True):

		##################################################
		# Base logging. 
		##################################################

		# print("Running Transfer PM Plots")

		log_dict = {'Domain': viz_dict['domain'], 
					'Total VAE Loss': self.total_VAE_loss,
					'Training Phase': self.training_phase}

		if self.args.setting not in ['densityjointtransfer','densityjointfixembedtransfer']:
			log_dict['Policy Loglikelihood'] = self.likelihood_loss
			log_dict['Encoder KL'] = self.encoder_KL
			log_dict['Unweighted VAE Loss'] = self.unweighted_VAE_loss
			log_dict['VAE Loss'] = self.VAE_loss
			log_dict['Training Discriminator'] = self.skip_vae
			log_dict['Training Embeddings or Translation Models'] = self.skip_discriminator	
			log_dict['Discriminability Loss'] = self.discriminability_loss
			log_dict['Unweighted Discriminability Loss'] = self.unweighted_discriminability_loss
			log_dict['Total Discriminability Loss'] = self.total_discriminability_loss

		##################################################
		# Log discriminator and discriminability losses 
		##################################################
		
		# Plot discriminator values after we've started training it. 
		if self.training_phase>1:

			if self.args.setting not in ['densityjointtransfer','densityjointfixembedtransfer']:

				# Compute discriminator loss and discriminator prob of right action for logging. 
				log_dict['Z Discriminator Loss'], log_dict['Z Discriminator Probability'] = self.discriminator_loss, viz_dict['discriminator_probs']
				log_dict['Total Discriminator Loss'] = self.total_discriminator_loss

				if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
					
					log_dict['Z Trajectory Discriminator Loss'] = self.z_trajectory_discriminator_loss
					log_dict['Unweighted Z Trajectory Discriminator Loss'] = self.unweighted_z_trajectory_discriminator_loss.mean()
					log_dict['Z Trajectory Discriminability Loss'] = self.z_trajectory_discriminability_loss
					log_dict['Z Trajectory Discriminator Probability'] = viz_dict['z_trajectory_discriminator_probs']
					log_dict['Unweighted Z Trajectory Discriminability Loss'] = self.masked_z_trajectory_discriminability_loss.mean()			

				# if self.args.equivariance and viz_dict['domain']==1:
				# 	log_dict['Unweighted Z Equivariance Loss'] = self.unweighted_masked_equivariance_loss
				# 	log_dict['Z Equivariance Loss'] = self.equivariance_loss

				if self.args.task_discriminability:

					log_dict['Unweighted Task Discriminability Loss'] = self.unweighted_task_discriminability_loss.mean()
					log_dict['Task Discriminability Loss'] = self.task_discriminability_loss
					log_dict['Task Discriminator Loss'] = self.task_discriminator_loss
					log_dict['Unweighted Task Discriminator Loss'] = self.unweighted_task_discriminator_loss.mean()
					log_dict['Task Discriminator Domain Probability'] = viz_dict['task_discriminator_probs']

			if self.args.cross_domain_supervision and (viz_dict['domain']==1 or self.args.setting in ['jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']):
				# If cycle, plot cdsl in both directions.
				log_dict['Unweighted Cross Domain Superivision Loss'] = self.unweighted_masked_cross_domain_supervision_loss.mean()
				# Now zero out if we want to use partial supervision..
				log_dict['Datapoint Masked Cross Domain Supervision Loss'] = self.datapoint_masked_cross_domain_supervised_loss
				log_dict['Cross Domain Superivision Loss'] = self.cross_domain_supervision_loss

			if self.args.setting in ['densityjointtransfer','densityjointfixembedtransfer'] and self.args.z_gmm:
				log_dict['Unweighted Cross Domain Density Loss'] = self.unweighted_masked_cross_domain_density_loss.mean()
				log_dict['Cross Domain Density Loss'] = self.cross_domain_density_loss.mean()

				if self.args.setting in ['densityjointfixembedtransfer'] and self.args.z_gmm:
					log_dict['Forward GMM Density Loss'] = viz_dict['forward_density_loss']
					log_dict['Backward GMM Density Loss'] = viz_dict['backward_density_loss']
					log_dict['Weighted Forward GMM Density Loss'] = viz_dict['weighted_forward_density_loss']
					log_dict['Weighted Backward GMM Density Loss'] = viz_dict['weighted_backward_density_loss']

			if self.args.supervised_set_based_density_loss:
				log_dict['Unweighted Supervised Set Based Density Loss'] = self.unweighted_supervised_set_based_density_loss
				log_dict['Datapoint Masked Supervised Set Based Density Loss'] = self.datapoint_masked_supervised_set_based_density_loss
				log_dict['Supervised Set Based Density Loss'] = self.supervised_set_based_density_loss

		##################################################
		# Now visualizing spaces. 
		##################################################


		# print("About to get images in Transfer Plot")

		# If we are displaying things: 
		if counter%self.args.display_freq==0:

			self.gt_gif_list = []
			self.rollout_gif_list = []			
			self.viz_dictionary = {}
					
			##################################################
			# Plot source, target, and shared embeddings via TSNE.
			##################################################

			# # First run get embeddings. 
			# self.viz_dictionary['tsne_source_embedding'], self.viz_dictionary['tsne_target_embedding'], \
			# 	self.viz_dictionary['tsne_combined_embeddings_p5'], self.viz_dictionary['tsne_combined_embeddings_p10'], self.viz_dictionary['tsne_combined_embeddings_p30'], \
			# 	self.viz_dictionary['tsne_combined_traj_embeddings_p5'], self.viz_dictionary['tsne_combined_traj_embeddings_p10'], self.viz_dictionary['tsne_combined_traj_embeddings_p30'] = \
			# 		self.get_embeddings(projection='tsne')

			# First run get embeddings. 
			_, _, self.viz_dictionary['tsne_combined_embeddings_p5'], self.viz_dictionary['tsne_combined_embeddings_p10'], self.viz_dictionary['tsne_combined_embeddings_p30'], _, _, _ = \
					self.get_embeddings(projection='tsne')

			# # If toy domain, plot the trajectories over the embeddings.		
			# if self.check_toy_dataset():
			# 	log_dict['TSNE Source Traj Embedding'], log_dict['TSNE Target Traj Embedding'] = \
			# 		 self.return_wandb_image(self.source_traj_image), self.return_wandb_image(self.target_traj_image)

			# # Add the embeddings to logging dict.
			# log_dict['TSNE Source Embedding'], log_dict['TSNE Target Embedding'], log_dict['TSNE Combined Embedding Perplexity 5'], \
			# 	log_dict['TSNE Combined Embedding Perplexity 10'], log_dict['TSNE Combined Embedding Perplexity 30'] = \
			# 		self.return_wandb_image(self.viz_dictionary['tsne_source_embedding']), self.return_wandb_image(self.viz_dictionary['tsne_target_embedding']), \
			# 		self.return_wandb_image(self.viz_dictionary['tsne_combined_embeddings_p5']), self.return_wandb_image(self.viz_dictionary['tsne_combined_embeddings_p10']), \
			# 		self.return_wandb_image(self.viz_dictionary['tsne_combined_embeddings_p30'])

			# Add the embeddings to logging dict.
			log_dict['TSNE Combined Embedding Perplexity 5'], log_dict['TSNE Combined Embedding Perplexity 10'], log_dict['TSNE Combined Embedding Perplexity 30'] = \
					self.return_wandb_image(self.viz_dictionary['tsne_combined_embeddings_p5']), self.return_wandb_image(self.viz_dictionary['tsne_combined_embeddings_p10']), \
					self.return_wandb_image(self.viz_dictionary['tsne_combined_embeddings_p30'])		

			##################################################
			# Plot source, target, and shared embeddings via PCA. 
			##################################################
			
			# print("Running embeddings PCA.")
			
			# First run get embeddings. 
			_, _, self.viz_dictionary['pca_combined_embeddings'], _ = self.get_embeddings(projection='pca', computed_sets=True)
			# # Add embeddings to logging dict.			
			# log_dict['PCA Source Embedding'], log_dict['PCA Target Embedding'], log_dict['PCA Combined Embedding'] = \
			# 	self.return_wandb_image(self.viz_dictionary['pca_source_embedding']), self.return_wandb_image(self.viz_dictionary['pca_target_embedding']), \
			# 	self.return_wandb_image(self.viz_dictionary['pca_combined_embeddings'])
	
			# Add embeddings to logging dict.			
			
			log_dict['PCA Combined Embedding'] = self.return_wandb_image(self.viz_dictionary['pca_combined_embeddings'])

			# # If toy domain, add to log dict.
			# if self.check_toy_dataset():				
			# 	log_dict['PCA Combined Trajectory Embeddings'], log_dict['TSNE Combined Trajectory Embeddings Perplexity 5'], \
			# 		log_dict['TSNE Combined Trajectory Embeddings Perplexity 10'], log_dict['TSNE Combined Trajectory Embeddings Perplexity 30'] = \
			# 			self.return_wandb_image(self.viz_dictionary['pca_combined_traj_embeddings']), self.return_wandb_image(self.viz_dictionary['tsne_combined_traj_embeddings_p5']), \
			# 			self.return_wandb_image(self.viz_dictionary['tsne_combined_traj_embeddings_p10']), self.return_wandb_image(self.viz_dictionary['tsne_combined_traj_embeddings_p30'])

			##################################################
			# Plot source, target, and shared embeddings via DENSNE.
			##################################################

			# First run get embeddings. 
			_, _, self.viz_dictionary['densne_combined_embeddings_p5'], self.viz_dictionary['densne_combined_embeddings_p10'], self.viz_dictionary['densne_combined_embeddings_p30'], \
				_, _, _ = self.get_embeddings(projection='densne')

			# # If toy domain, plot the trajectories over the embeddings.		
			# if self.check_toy_dataset():
			# 	log_dict['DENSNE Source Traj Embedding'], log_dict['DENSNE Target Traj Embedding'] = \
			# 		 self.return_wandb_image(self.source_traj_image), self.return_wandb_image(self.target_traj_image)

			# Add the embeddings to logging dict.
			log_dict['DENSNE Combined Embedding Perplexity 5'], log_dict['DENSNE Combined Embedding Perplexity 10'], log_dict['DENSNE Combined Embedding Perplexity 30'] = \
				self.return_wandb_image(self.viz_dictionary['densne_combined_embeddings_p5']), self.return_wandb_image(self.viz_dictionary['densne_combined_embeddings_p10']), \
					self.return_wandb_image(self.viz_dictionary['densne_combined_embeddings_p30'])
			##################################################
			# We are also going to log Ground Truth trajectories and their reconstructions in each of the domains, to make sure our networks are learning. 		
			##################################################

			if not(self.args.no_mujoco) and self.check_toy_dataset():
				self.viz_dictionary['source_trajectory'], self.viz_dictionary['source_reconstruction'], self.viz_dictionary['target_trajectory'], self.viz_dictionary['target_reconstruction'] = self.get_trajectory_visuals()

			if 'source_trajectory' in self.viz_dictionary and not(self.args.no_mujoco):
				# Now actually plot the images.
				if self.args.source_domain in ['ContinuousNonZero','DirContNonZero','ToyContext']:
					log_dict['Source Trajectory'], log_dict['Source Reconstruction'] = \
						self.return_wandb_image(self.viz_dictionary['source_trajectory']), self.return_wandb_image(self.viz_dictionary['source_reconstruction'])
				else:
					log_dict['Source Trajectory'], log_dict['Source Reconstruction'] = \
						self.return_wandb_gif(self.viz_dictionary['source_trajectory']), self.return_wandb_gif(self.viz_dictionary['source_reconstruction'])

				if self.args.target_domain in ['ContinuousNonZero','DirContNonZero','ToyContext']:
					log_dict['Target Trajectory'], log_dict['Target Reconstruction'] = \
						self.return_wandb_image(self.viz_dictionary['target_trajectory']), self.return_wandb_image(self.viz_dictionary['target_reconstruction'])
				else:
					log_dict['Target Trajectory'], log_dict['Target Reconstruction'] = \
						self.return_wandb_gif(self.viz_dictionary['target_trajectory']), self.return_wandb_gif(self.viz_dictionary['target_reconstruction'])	

			##################################################			
			# Evaluate metrics and plot them. 
			##################################################

			# Log Average Reconstruction Error.
			log_dict['Source Trajectory Reconstruction Error'], log_dict['Target Trajectory Reconstruction Error'] = \
				self.source_manager.avg_reconstruction_error, self.target_manager.avg_reconstruction_error

			if self.args.eval_transfer_metrics and counter%self.args.metric_eval_freq==0:
				
				# Visualize Source and Translated Target trajectory GIFs, irrespective of whether or not we've same domains. 
				self.visualize_translated_trajectories()

				# Log these GIFs assuming we have MUJOCO.
				if self.args.no_mujoco==0:
					log_dict['Trajectory 0 Original Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj0_OriginalTarget_Traj'])
					log_dict['Trajectory 1 Original Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj1_OriginalTarget_Traj'])					
					log_dict['Trajectory 0 Translated Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj0_TranslatedTarget_Traj'])
					log_dict['Trajectory 1 Translated Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj1_TranslatedTarget_Traj'])

				# If we are actually in same domain, also evaluate average trajectory reconstruction error.
				if self.check_same_domains():
					self.evaluate_correspondence_metrics()
					log_dict['Target To Source Translation Trajectory Error'] = self.average_translated_trajectory_reconstruction_error

					# Log these GIFs assuming we have MUJOCO.
					if self.args.no_mujoco==0:
						log_dict['Trajectory 0 Source GIF'] = self.return_wandb_gif(self.gif_logs['Traj0_Source_Traj'])
						log_dict['Trajectory 0 Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj0_Target_Traj'])
						log_dict['Trajectory 1 Source GIF'] = self.return_wandb_gif(self.gif_logs['Traj1_Source_Traj'])
						log_dict['Trajectory 1 Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj1_Target_Traj'])


			# if self.check_same_domains() and self.args.eval_transfer_metrics and counter%self.args.metric_eval_freq==0:
								
			# 	# # self.evaluate_correspondence_metrics(computed_sets=False)
				
			# 	# # Actually, we've probably computed trajectory and latent sets. 
			# 	# if counter>0:
			# 	self.evaluate_correspondence_metrics()

			# 	# 	log_dict['Source To Target Translation Trajectory Error'] = self.source_target_trajectory_distance
			# 	# 	log_dict['Target To Source Translation Trajectory Error'] = self.target_source_trajectory_distance
			# 	log_dict['Target To Source Translation Trajectory Error'] = self.average_translated_trajectory_reconstruction_error
			# 	# 	log_dict['Source To Target Translation Trajectory Normalized Error'] = self.source_target_trajectory_normalized_distance
			# 	# 	log_dict['Target To Source Translation Trajectory Normalized Error'] = self.target_source_trajectory_normalized_distance
			# 	# 	log_dict['Average Corresponding Z Sequence Error'] = self.average_corresponding_z_sequence_error.mean()
			# 	# 	log_dict['Average Corresponding Z Transition Sequence Error'] = self.average_corresponding_z_transition_sequence_error.mean()
			
			# 	if self.args.no_mujoco==0:

			# 		log_dict['Trajectory 0 Source GIF'] = self.return_wandb_gif(self.gif_logs['Traj0_Source_Traj'])
			# 		log_dict['Trajectory 0 Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj0_Target_Traj'])
			# 		log_dict['Trajectory 1 Source GIF'] = self.return_wandb_gif(self.gif_logs['Traj1_Source_Traj'])
			# 		log_dict['Trajectory 1 Target GIF'] = self.return_wandb_gif(self.gif_logs['Traj1_Target_Traj'])

			##################################################
			# Visualize Z Trajectories.
			##################################################

			log_dict['Source Z Trajectory Joint TSNE Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_tsne_image)
			log_dict['Target Z Trajectory Joint TSNE Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_tsne_image)

			log_dict['Source Z Trajectory PCA Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_pca_image)
			log_dict['Target Z Trajectory PCA Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_pca_image)

			log_dict['Source Z Trajectory Joint DENSNE Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_densne_image)
			log_dict['Target Z Trajectory Joint DENSNE Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_densne_image)

			
			##################################################
			# Visualize z tuple embeddings
			##################################################
		
			log_dict['Joint Z Tuple TSNE Embeddings'] = self.return_wandb_image(copy.deepcopy(self.z_tuple_embedding_image))

			##################################################			
			# Clean up objects consuming memory. 			
			##################################################

			self.free_memory()

		##################################################
		# If we're directly optimizing representations.. call log_density_and_chamfer_metrics..		
		##################################################

		if self.args.setting in ['transfer','cycle_transfer','densityjointtransfer','jointtransfer','jointcycletransfer','densityjointfixembedtransfer']:
			log_dict = self.log_density_and_chamfer_metrics(counter, log_dict, viz_dict)		

		##################################################
		# Now log everything. 
		##################################################

		if log:
			wandb.log(log_dict, step=counter)
		else:
			return log_dict
		
	def get_transform(self, latent_z_set, projection='tsne', shared=False, perplexity=30):

		# if shared:
		# 	# If this set of z's contains z's from both source and target domains, mean-std normalize them independently. 
		# 	normed_z = np.zeros_like(latent_z_set)
		# 	# Normalize source.
		# 	source_mean = latent_z_set[:self.N].mean(axis=0)
		# 	source_std = latent_z_set[:self.N].std(axis=0)
		# 	normed_z[:self.N] = (latent_z_set[:self.N]-source_mean)/source_std
		# 	# Normalize target.
		# 	target_mean = latent_z_set[self.N:].mean(axis=0)
		# 	target_std = latent_z_set[self.N:].std(axis=0)
		# 	normed_z[self.N:] = (latent_z_set[self.N:]-target_mean)/target_std			

		# else:
		# 	# Just normalize z's.
		# 	mean = latent_z_set.mean(axis=0)
		# 	std = latent_z_set.std(axis=0)
		# 	normed_z = (latent_z_set-mean)/std

		# if self.args.z_normalization:
		# 	# ASSUME ALREADY NORMALIZED! 
		# 	normed_z = latent_z_set
		# else:
		# 	# Just normalize z's.
		# 	mean = latent_z_set.mean(axis=0)
		# 	std = latent_z_set.std(axis=0)
		# 	normed_z = (latent_z_set-mean)/std

		# DON'T NORMALIZE THESE SPACES
		# Different STD dev for different dims is weird
		normed_z = latent_z_set	

		if projection=='tsne':
			# Use TSNE to project the data:
			tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=perplexity)
			embedded_zs = tsne.fit_transform(normed_z)

			scale_factor = 1
			scaled_embedded_zs = scale_factor*embedded_zs

			return scaled_embedded_zs, tsne

		elif projection=='pca':
			# Use PCA to project the data:
			pca_object = PCA(n_components=2)
			embedded_zs = pca_object.fit_transform(normed_z)

			return embedded_zs, pca_object

		elif projection=='densne':
			# Use density preserving T-SNE
			# embedded_zs, _, _  = densne.run_densne(normed_z, no_dims=2,randseed=0,perplexity=perplexity, verbose=True)
			embedded_zs, _, _  = densne.run_densne(normed_z, no_dims=2,randseed=0,perplexity=perplexity)
			
			return embedded_zs, None
		
	def transform_zs(self, latent_z_set, transforming_object):
		# Simply just transform according to a fit transforming_object.
		return transforming_object.transform(latent_z_set)

	def set_z_objects(self):
		self.state_dim = 2

		# Use source and target policy manager set computing functions, because they can handle batches.
		self.source_manager.get_trajectory_and_latent_sets(get_visuals=False)
		self.target_manager.get_trajectory_and_latent_sets(get_visuals=False)
				
		# Now assemble them into local variables.
		self.N = self.source_manager.N

		# print("Embed in Set Z Objects")
		# embed()

		if self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
			self.source_latent_zs = np.concatenate(self.source_manager.latent_z_set)
			self.target_latent_zs = np.concatenate(self.target_manager.latent_z_set)
			# First, normalize the sets.. 

			self.source_latent_zs = (self.source_latent_zs-self.source_z_mean.detach().cpu().numpy())/self.source_z_std.detach().cpu().numpy()
			self.target_latent_zs = (self.target_latent_zs-self.target_z_mean.detach().cpu().numpy())/self.target_z_std.detach().cpu().numpy()

			# These are the same z's... this object just retains sequence info. Should be able to find some indexing of concatenate...? 
			self.source_z_trajectory_set = self.source_manager.latent_z_set
			self.target_z_trajectory_set = self.target_manager.latent_z_set

		else:
			self.source_latent_zs = self.source_manager.latent_z_set
			self.target_latent_zs = self.target_manager.latent_z_set
			# First, normalize the sets.. 
			self.source_latent_zs = (self.source_latent_zs-self.source_z_mean.detach().cpu().numpy())/self.source_z_std.detach().cpu().numpy()
			self.target_latent_zs = (self.target_latent_zs-self.target_z_mean.detach().cpu().numpy())/self.target_z_std.detach().cpu().numpy()



		# Try something
		self.source_latent_zs = self.source_latent_zs[:min(500,len(self.source_latent_zs))]
		self.target_latent_zs = self.target_latent_zs[:min(500,len(self.target_latent_zs))]

		self.final_number_of_zs = [self.source_latent_zs.shape[0], self.target_latent_zs.shape[0]]

		# Copy sets so we don't accidentally perform in-place operations on any of the computed sets.
		self.original_source_latent_z_set = copy.deepcopy(self.source_latent_zs)
		self.original_target_latent_z_set = copy.deepcopy(self.target_latent_zs)

		self.shared_latent_zs = np.concatenate([self.source_latent_zs,self.target_latent_zs],axis=0)

		# Also make sure z tuple objects are set
		self.construct_tuple_embeddings(translated_target=False)

		self.z_last_set_by = 'set_z_objects'

	def visualize_embedded_z_trajectories(self, domain, shared_z_embedding, z_trajectory_set_object, projection='tsne'):
		
		# Visualize a set of z trajectories over the shared z embedding space.

		# Get the figure and axes objects, so we can overlay images on to this. 
		domain_list = ['source','target']
		viz_domain = domain_list[domain]

		# print("Embed in viz z traj.")
		# embed()
		fig, ax = self.plot_embedding(shared_z_embedding, "Z Trajectory Image {0}".format(projection), return_fig=True, viz_domain=viz_domain)

		# Add this value to indexing shared_z_embedding, assuming we're actually providing a shared embedding. 		
		# If source, viz_domain = 0, so don't add anything, but if target, add the length of the soruce_latent_z to skip these.
		add_value = len(self.source_latent_zs)*domain

		# embed()
		# for i, z_traj in enumerate(z_trajectory_set_object):
		for i in range(10):
			z_traj = z_trajectory_set_object[i]	
		
			# First get length of this z_trajectory.
			z_traj_len = len(z_traj)

			# Should just be able to get the corresponding embedded z by manipulating indices.. 
			# Assuming len of z_traj is consistent across all elements in z_trajectory_set_object, which would have needed to have been true 
			# for the concatenate in set_z_objects to work.
			# embedded_z_traj = shared_z_embedding[i*z_traj_len:(i+1)*z_traj_len]
			embedded_z_traj = shared_z_embedding[add_value+i*z_traj_len:add_value+(i+1)*z_traj_len]

			# Now that we have the embedded trajectory, come up with some plot for it. 
			ax.scatter(embedded_z_traj[:,0],embedded_z_traj[:,1],s=10,c=domain*np.ones(z_traj_len),cmap='jet',vmin=0,vmax=1)
			diffs = np.diff(embedded_z_traj,axis=0)
			
			ax.quiver(embedded_z_traj[:-1,0],embedded_z_traj[:-1,1],diffs[:,0],diffs[:,1],angles='xy',scale_units='xy',scale=1)

		# Re-generate image.
		fig.canvas.draw()
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)		
		image = np.transpose(image, axes=[2,0,1])

		# Reset plots.
		ax.clear()
		fig.clear()
		plt.close(fig)

		return image	

	def construct_tuple_embeddings(self, translated_target=False):

		# First construct tuples from self.source_z_trajectory_set and self.target_z_trajectory_set		
		self.source_z_tuples = self.construct_tuples_from_z_traj_set(self.source_z_trajectory_set)

		
		if translated_target: 
			self.target_z_tuples = self.construct_tuples_from_z_traj_set(self.translated_target_z_trajectory_set)
		else:
			self.target_z_tuples = self.construct_tuples_from_z_traj_set(self.target_z_trajectory_set)

		# First construct shared original tuples. 
		self.shared_z_tuples = np.concatenate([self.source_z_tuples,self.target_z_tuples])
		
		# Now construct shared source and translated target tuples. 
		# Basically just translate... 
		
		# Next construct embeddings of these tuples. Probably sufficient to just do one joint color coded embeddings.
		self.tsne_embedded_z_tuples, _ = self.get_transform(self.shared_z_tuples)
				
	def construct_tuples_from_z_traj_set(self, z_trajectory_set):

		z_tuple_list = []

		for k, v in enumerate(z_trajectory_set):

			if v.shape[0] == 0:
				print("Running into empty translated target z set issue.")
				embed()

			# First add (0,z_1) tuple. 
			z_tuple_list.append(np.concatenate([np.zeros(self.args.z_dimensions),v[0]]).reshape(1,-1))

			# Add intermediate tuples..
			for j in range(len(v)-1):
				z_tuple_list.append(np.concatenate([v[j],v[j+1]]).reshape(1,-1))

			# Finally, add (z_n,0) tuple.
			z_tuple_list.append(np.concatenate([v[-1],np.zeros(self.args.z_dimensions)]).reshape(1,-1))
		
		z_tuple_array = np.concatenate(z_tuple_list)

		return z_tuple_array

	def get_embeddings(self, projection='tsne', computed_sets=False):
		# Function to visualize source, target, and combined embeddings: 
	
		if computed_sets==False:
			self.set_z_objects()

		# Now that the latent sets for both source and target domains are computed: 
		if projection=='tsne' or projection=='densne':

			# Perplexity lists: 
			if projection=='tsne':
				perplexity_list = [5,10,30]
			if projection=='densne':
				# perplexity_list = [10,30,50]
				perplexity_list = [5,10,30]

			# Use TSNE to transform data.		
			# source_embedded_zs, _ = self.get_transform(self.source_latent_zs, projection)
			# target_embedded_zs, _ = self.get_transform(self.target_latent_zs, projection)
			shared_embedded_zs_p5, _ = self.get_transform(self.shared_latent_zs, projection, shared=True, perplexity=perplexity_list[0])
			shared_embedded_zs_p10, _ = self.get_transform(self.shared_latent_zs, projection, shared=True, perplexity=perplexity_list[1])
			shared_embedded_zs_p30, shared_embedded_zs_p30_tsne = self.get_transform(self.shared_latent_zs, projection, shared=True, perplexity=perplexity_list[2])
			shared_embedded_zs = shared_embedded_zs_p30
			source_embedded_zs = shared_embedded_zs_p30[:len(self.source_latent_zs)]
			target_embedded_zs = shared_embedded_zs_p30[len(self.source_latent_zs):]

			########################################
			# Shared data point visualization. 
			########################################

			self.shared_image_p5 = self.plot_embedding(shared_embedded_zs_p5, "Shared_Embedding Perplexity 5", shared=True)	
			self.shared_image_p10 = self.plot_embedding(shared_embedded_zs_p10, "Shared_Embedding Perplexity 10", shared=True)	
			self.shared_image_p30 = self.plot_embedding(shared_embedded_zs_p30, "Shared_Embedding Perplexity 30", shared=True)	

			########################################
			# Visualizing embedding z trajectories.
			########################################

			# self.set_translated_z_sets()
			# self.source_z_traj_tsne_image = self.visualize_embedded_z_trajectories(0, source_embedded_zs, self.source_z_trajectory_set, projection='tsne')
			# self.target_z_traj_tsne_image = self.visualize_embedded_z_trajectories(1, target_embedded_zs, self.target_z_trajectory_set, projection='tsne')
			if projection=='tsne':
				self.source_z_traj_tsne_image = self.visualize_embedded_z_trajectories(0, shared_embedded_zs_p30, self.source_z_trajectory_set, projection=projection)
				self.target_z_traj_tsne_image = self.visualize_embedded_z_trajectories(1, shared_embedded_zs_p30, self.target_z_trajectory_set, projection=projection)
			else:
				self.source_z_traj_densne_image = self.visualize_embedded_z_trajectories(0, shared_embedded_zs_p30, self.source_z_trajectory_set, projection=projection)
				self.target_z_traj_densne_image = self.visualize_embedded_z_trajectories(1, shared_embedded_zs_p30, self.target_z_trajectory_set, projection=projection)


		elif projection=='pca':
			# Now fit PCA to source.
			source_embedded_zs, pca = self.get_transform(self.source_latent_zs, projection)
			target_embedded_zs = self.transform_zs(self.target_latent_zs, pca)
			shared_embedded_zs = np.concatenate([source_embedded_zs, target_embedded_zs],axis=0)

			########################################
			# Shared data point visualization. 
			########################################

			self.shared_image = self.plot_embedding(shared_embedded_zs, "Shared_Embedding", shared=True)	

			########################################
			# Visualizing embedding z trajectories.
			########################################

			# self.set_translated_z_sets()
			# self.source_z_traj_pca_image = self.visualize_embedded_z_trajectories(0, source_embedded_zs, self.source_z_trajectory_set, projection='pca')
			# self.target_z_traj_pca_image = self.visualize_embedded_z_trajectories(1, target_embedded_zs, self.target_z_trajectory_set, projection='pca')
			self.source_z_traj_pca_image = self.visualize_embedded_z_trajectories(0, shared_embedded_zs, self.source_z_trajectory_set, projection='pca')
			self.target_z_traj_pca_image = self.visualize_embedded_z_trajectories(1, shared_embedded_zs, self.target_z_trajectory_set, projection='pca')
		

		########################################
		# Single domain data point visualization.
		########################################

		self.source_image = self.plot_embedding(shared_embedded_zs,"Source_Embedding",viz_domain='source')
		self.target_image = self.plot_embedding(shared_embedded_zs,"Target_Embedding",viz_domain='target')

		########################################
		# Also visualize tuples of Z's.. 
		########################################

		# Remember, the objects we are interested in are : self.source_z_trajectory_set, and self.target_z_trajectory_set. 
		# First construct necessary tuples
		
		# self.construct_tuple_embeddings()		
		self.z_tuple_embedding_image = self.plot_embedding(self.tsne_embedded_z_tuples, "TSNE Z Tuple Embedding", shared=True, source_length=len(self.source_z_tuples))
		# self.z_tuple_embedding_image = self.plot_embedding(self.tsne_embedded_z_tuples, "TSNE Z Tuple Embedding", shared=True, source_length=len(self.source_z_tuples))

		
		########################################
		# Single domain data point visualization with trajectories.
		########################################

		if self.check_toy_dataset():
			self.source_traj_image = self.plot_embedding(shared_embedded_zs, "Source_Embedding", trajectory=True, viz_domain='source')
			self.target_traj_image = self.plot_embedding(shared_embedded_zs, "Target_Embedding", trajectory=True, viz_domain='target')

		self.samedomain_shared_embedding_image = None


		if projection=='tsne' or projection=='densne':

			if self.check_toy_dataset():

				########################################
				# Shared data point visualization with trajectories.
				########################################

				self.samedomain_shared_embedding_image_p5 = self.plot_embedding(shared_embedded_zs_p5, "SameDomain_Shared_Traj_Embedding Perplexity 5", shared=True, trajectory=True)
				self.samedomain_shared_embedding_image_p10 = self.plot_embedding(shared_embedded_zs_p10, "SameDomain_Shared_Traj_Embedding Perplexity 10", shared=True, trajectory=True)
				self.samedomain_shared_embedding_image_p30 = self.plot_embedding(shared_embedded_zs_p30, "SameDomain_Shared_Traj_Embedding Perplexity 30", shared=True, trajectory=True)
			else:
				self.samedomain_shared_embedding_image_p5  = None
				self.samedomain_shared_embedding_image_p10 = None
				self.samedomain_shared_embedding_image_p30 = None

			return self.source_image, self.target_image, self.shared_image_p5, self.shared_image_p10, self.shared_image_p30, \
				self.samedomain_shared_embedding_image_p5, self.samedomain_shared_embedding_image_p10, self.samedomain_shared_embedding_image_p30

		else:
			if self.check_toy_dataset():

				########################################
				# Shared data point visualization with trajectories.
				########################################

				self.samedomain_shared_embedding_image = self.plot_embedding(shared_embedded_zs, "SameDomain_Shared_Traj_Embedding", shared=True, trajectory=True)
			else:
				self.samedomain_shared_embedding_image = None
		
			return self.source_image, self.target_image, self.shared_image, self.samedomain_shared_embedding_image

	def plot_embedding(self, embedded_zs, title, shared=False, trajectory=False, viz_domain=None, return_fig=False, source_length=None):	
		
		# print("Running plot embedding", title, viz_domain)
		############################################################
		# Setting fig size everywhere so that it doesn't go nuts. 
		############################################################

		matplotlib.rcParams['figure.figsize'] = [5,5]
		fig = plt.figure()
		ax = fig.gca()
		
		############################################################
		# Set colors of embedding plots based on which domain we're plotting.
		############################################################

		if shared:
			colors = 0.2*np.ones((embedded_zs.shape[0]))
			# colors[embedded_zs.shape[0]//2:] = 0.8
			# TRY REPLACE Z.SHAPE//2 by [len(self.source_latent_zs):]
			if source_length is not None:
				colors[source_length:] = 0.8
			else:
				colors[len(self.source_latent_zs):] = 0.8
		else:
			if viz_domain=='source':
				colors = 0.2*np.ones((embedded_zs.shape[0]))
			elif viz_domain=='target':
				colors = 0.8*np.ones((embedded_zs.shape[0]))
			else:
				colors = 0.2*np.ones((embedded_zs.shape[0]))

		############################################################
		# If we're visualizing trajectories in the visualized plots. 
		############################################################

		if trajectory:

			############################################################
			# If we're in the shared embedding setting, assemble the shared_trajectory_set. 
			############################################################

			if shared:

				########################################
				# Create a scatter plot of the embedding.
				########################################

				self.source_manager.get_trajectory_and_latent_sets()
				self.target_manager.get_trajectory_and_latent_sets()						

				if self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
					# self.source_manager.trajectory_set = np.array(self.source_manager.trajectory_set)
					# self.target_manager.trajectory_set = np.array(self.target_manager.trajectory_set)
					# traj_length = len(self.source_manager.trajectory_set[0,:,0])
					# Create a shared trajectory set from both individual segmented_trajectory_set(s). 
					self.shared_trajectory_set = self.source_manager.segmented_trajectory_set+self.target_manager.segmented_trajectory_set					
				else:
					# Assemble shared trajectory set. 
					traj_length = len(self.source_manager.trajectory_set[0,:,0])
					self.shared_trajectory_set = np.zeros((2*self.N, traj_length, self.source_manager.state_dim))
					self.shared_trajectory_set[:self.N] = self.source_manager.trajectory_set
					self.shared_trajectory_set[self.N:] = self.target_manager.trajectory_set		

			else:

				########################################
				# Otherwise just set the shared_trajectory_set.
				########################################

				# Depending on whether we're visualizing the source or target domain
				if viz_domain=='source':
					# embedded_zs = embedded_zs[:(embedded_zs.shape[0]//2)]
					embedded_zs = embedded_zs[:len(self.source_latent_zs)]
					self.shared_trajectory_set = self.source_manager.segmented_trajectory_set					
				elif viz_domain=='target':
					# embedded_zs = embedded_zs[(embedded_zs.shape[0]//2):]
					embedded_zs = embedded_zs[len(self.source_latent_zs):]
					self.shared_trajectory_set = self.target_manager.segmented_trajectory_set
			
			############################################################
			# Now that we've set the trajectory set, plot the trajectories.
			############################################################

			# ratio = 0.4
			# preratio = 0.01
			preratio = 0.005
			ratio = (embedded_zs.max()-embedded_zs.min())*preratio
			color_scaling = 15
			max_traj_length = 6
			color_range_min = 0.2*color_scaling
			color_range_max = 0.8*color_scaling+max_traj_length-1

			# Randomize the order of the plot, so that one domain doesn't overwrite the other in the plot. 
			random_range = list(range(min(embedded_zs.shape[0],len(self.shared_trajectory_set))))
			random.shuffle(random_range)
			
			for i in random_range:
				seg_traj_len = len(self.shared_trajectory_set[i])
				ax.scatter(embedded_zs[i,0]+ratio*self.shared_trajectory_set[i][:,0],embedded_zs[i,1]+ratio*self.shared_trajectory_set[i][:,1], \
					c=colors[i]*color_scaling+range(seg_traj_len),cmap='jet',vmin=color_range_min,vmax=color_range_max,s=15)

		############################################################
		# If we're visualizing just data points in the visualized plots. 
		############################################################

		else:
			########################################
			# Create a scatter plot of the embedding.
			########################################

			s = np.ones(embedded_zs.shape[0])*50
			if viz_domain=='source':
				# s[(embedded_zs.shape[0]//2):] = 1
				# colors[(embedded_zs.shape[0]//2):] = 0.8
				# TRY REPLACE Z.SHAPE//2 by [len(self.source_latent_zs):]
				s[len(self.source_latent_zs):] = 1
				colors[len(self.source_latent_zs):] = 0.8
			elif viz_domain=='target':				
				# s[:(embedded_zs.shape[0]//2)] = 1
				# colors[:(embedded_zs.shape[0])//2] = 0.2
				# TRY REPLACE Z.SHAPE//2 by [:len(self.source_latent_zs)]
				s[:len(self.source_latent_zs)] = 1
				colors[:len(self.source_latent_zs)] = 0.2

			ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet',s=s)
		
		############################################################
		# Now make the plot and generate numpy image from it. 
		############################################################
		ax.set_title("{0}".format(title),fontdict={'fontsize':15})
		fig.canvas.draw()		
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)		
		image = np.transpose(image, axes=[2,0,1])

		# Clear figure from memory.
		if return_fig:
			return fig, ax
		else:
			ax.clear()
			fig.clear()
			plt.close(fig)

			return image

	def get_trajectory_visuals(self):

		i = np.random.randint(0,high=self.extent)

		# First get a trajectory, starting point, and latent z.
		if self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
			source_input_traj, source_var_dict, _ = self.encode_decode_trajectory(self.source_manager, i)
			source_trajectory = source_input_traj['sample_traj']
			source_latent_z = source_var_dict['latent_z_indices']			
		else:
			source_trajectory, source_latent_z = self.encode_decode_trajectory(self.source_manager, i, return_trajectory=True)

		if self.args.batch_size>1:
			source_trajectory = source_trajectory[:,0]
			source_latent_z = source_latent_z[:,0]

		# print("Embedding in Transfer Get Trajectory Visuals.")
		# embed()

		if source_trajectory is not None:
			# Reconstruct using the source domain manager. 

			_, self.source_trajectory_image, self.source_reconstruction_image = self.source_manager.get_robot_visuals(0, source_latent_z, source_trajectory, return_image=True, return_numpy=True, z_seq=(self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']))

			# Now repeat the same for target domain - First get a trajectory, starting point, and latent z.
			if self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle']:
				target_input_dict, target_var_dict, _ = self.encode_decode_trajectory(self.target_manager, i)
				target_trajectory = target_input_dict['sample_traj']
				target_latent_z = target_var_dict['latent_z_indices']
			else:
				target_trajectory, target_latent_z = self.encode_decode_trajectory(self.target_manager, i, return_trajectory=True)

			if self.args.batch_size>1:
				target_trajectory = target_trajectory[:,0]
				target_latent_z = target_latent_z[:,0]

			# Reconstruct using the target domain manager. 
			_, self.target_trajectory_image, self.target_reconstruction_image = self.target_manager.get_robot_visuals(0, target_latent_z, target_trajectory, return_image=True, return_numpy=True, z_seq=(self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']))

			# return np.array(self.source_trajectory_image), np.array(self.source_reconstruction_image), np.array(self.target_trajectory_image), np.array(self.target_reconstruction_image)
			return self.source_trajectory_image, self.source_reconstruction_image, self.target_trajectory_image, self.target_reconstruction_image	
			
		else: 
			return None, None, None, None

	def select_supervised_datapoint_batch_index(self):
		
		# Randomly sample a supervised datapoint batch index. 
		# Sample from.. 
		if self.args.number_of_supervised_datapoints==-1:
			max_value = self.extent
		else:
			max_value = self.args.number_of_supervised_datapoints

		index_choices = np.arange(0,max_value,self.args.batch_size)
		
		# Now sample from these choices. 
		return np.random.choice(index_choices)

	def compute_identity_loss(self, update_dictionary):

		# Feed latent z into translation model, force outputs to be unchanged..
		# Remember, this is only called with domain = 0 here. Here update_dictionary['detached_latent_z'] corresponds to the original source z.
		self.translated_source_z = self.backward_translation_model.forward(update_dictionary['detached_latent_z'])
		unweighted_identity_translation_loss = ((self.translated_source_z - update_dictionary['detached_latent_z'])**2).mean()
		return unweighted_identity_translation_loss		

	def update_networks(self, domain, policy_manager, update_dictionary):

		#########################################################################
		#########################################################################
		# (1) First, update the representation based on reconstruction and discriminability.
		#########################################################################
		#########################################################################

		# Zero out gradients of encoder and decoder (policy).
		if self.args.setting in ['jointfixembed', 'fixembed']:
			# If we are in the translation model setting, use self.optimizer rather either source / target policy manager. 
			self.optimizer.zero_grad()
		else:
			policy_manager.optimizer.zero_grad()

		###########################################################
		# (1a) First, compute reconstruction loss.
		###########################################################

		# Compute VAE loss on the current domain as likelihood plus weighted KL.  
		self.likelihood_loss = -update_dictionary['loglikelihood'].mean()
		self.encoder_KL = update_dictionary['kl_divergence'].mean()
		self.unweighted_VAE_loss = self.likelihood_loss + self.args.kl_weight*self.encoder_KL
		self.VAE_loss = self.vae_loss_weight*self.unweighted_VAE_loss

		###########################################################
		# (1b) Next, compute discriminability loss.
		###########################################################

		# Compute discriminability loss for encoder (implicitly ignores decoder).
		# Pretend the label was the opposite of what it is, and train the encoder to make the discriminator think this was what was true. 
		# I.e. train encoder to make discriminator maximize likelihood of wrong label.
		# domain_label = torch.tensor(1-domain).to(device).long().view(1,)
		domain_label = domain*torch.ones(update_dictionary['discriminator_logprob'].shape[0]*update_dictionary['discriminator_logprob'].shape[1]).to(device).long()
		self.unweighted_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['discriminator_logprob'].view(-1,2), 1-domain_label).mean()
		self.discriminability_loss = self.discriminability_loss_weight*self.unweighted_discriminability_loss
			
		###########################################################
		# (1c) Next, compute z_trajectory discriminability loss.
		###########################################################
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			if self.args.z_trajectory_discriminator:
				traj_domain_label = domain*torch.ones(self.args.batch_size).to(device).long()
				# Overwrite update_dictionary['z_trajectory_weights']. 
				update_dictionary['z_trajectory_weights'] = torch.ones(self.args.batch_size).to(device).float()
			
			elif self.args.z_transform_discriminator:
				# traj_domain_label = domain_label
				# domain*torch.ones(update_dictionary['discriminator_logprob'].shape[0]*update_dictionary['discriminator_logprob'].shape[1]).to(device).long()
				traj_domain_label = domain*torch.ones(update_dictionary['z_trajectory_discriminator_logprob'].shape[0],update_dictionary['z_trajectory_discriminator_logprob'].shape[1]).view(-1,).to(device).long()

			# Set z transform discriminability loss.
			# print("Embedding in update networks, right before computing z traj disc loss")
			# embed()

			self.unweighted_z_trajectory_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['z_trajectory_discriminator_logprob'].view(-1,2), 1-traj_domain_label)
			# self.unweighted_z_trajectory_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['z_trajectory_discriminator_logprob'].view(-1,2), 1-domain_label)
			self.masked_z_trajectory_discriminability_loss = update_dictionary['z_trajectory_weights'].view(-1,)*self.unweighted_z_trajectory_discriminability_loss
			# Mask the z transform discriminability loss based on whether or not this particular latent_z, latent_z transformation tuple should be used to train the representation.
			self.z_trajectory_discriminability_loss = self.z_trajectory_discriminability_loss_weight*self.masked_z_trajectory_discriminability_loss.mean()
		else:
			# Set z transform discriminability loss to dummy value.
			self.z_trajectory_discriminability_loss = 0.
		
		# ###########################################################
		# # (1d) If active, compute equivariance loss. 
		# ###########################################################
		
		# if self.args.equivariance:
		# 	self.unweighted_unmasked_equivariance_loss = self.compute_equivariance_loss(update_dictionary)
		# 	# Now mask by the same temporal masks that we used for the discriminability versions of this idea. 
		# 	self.unweighted_masked_equivariance_loss = (update_dictionary['z_trajectory_weights'].view(-1,)*self.unweighted_unmasked_equivariance_loss).mean()
		# 	self.equivariance_loss = self.args.equivariance_loss_weight*self.unweighted_masked_equivariance_loss

		# else:
		# 	self.equivariance_loss = 0.

		###########################################################
		# (1e) If active, compute cross domain z loss. 
		###########################################################
	
		# Remember, the cross domain gt supervision loss should only be active when... trnaslating, i.e. when we have domain==1.
		if self.args.cross_domain_supervision and domain==1:
			# Call function to compute this. # This function depends on whether we have a translation model or not.. 
			self.unweighted_unmasked_cross_domain_supervision_loss = self.compute_cross_domain_supervision_loss(update_dictionary)
			# Now mask using batch mask.			
			# self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).mean()
			self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).sum()/(policy_manager.batch_mask.sum())
			# Now zero out if we want to use partial supervision..
			self.datapoint_masked_cross_domain_supervised_loss = self.supervised_datapoints_multiplier*self.unweighted_masked_cross_domain_supervision_loss
			# Now weight.			
			self.cross_domain_supervision_loss = self.args.cross_domain_supervision_loss_weight*self.datapoint_masked_cross_domain_supervised_loss
		else:
			self.unweighted_masked_cross_domain_supervision_loss = 0.
			self.datapoint_masked_cross_domain_supervised_loss = 0.
			self.cross_domain_supervision_loss = 0.

		###########################################################
		# (1f) If active, compute task discriminability loss.
		###########################################################

		if self.args.task_discriminability:
			# Set the same kind of label we used in z_trajectory_discriminability..
			traj_domain_label = domain*torch.ones(self.args.batch_size).to(device).long()
			# Create an NLL based on task_discriminator_logprobs...
			self.unweighted_task_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['task_discriminator_logprob'].view(-1,2), 1-traj_domain_label)
			# Weight and average.
			self.task_discriminability_loss = self.args.task_discriminability_loss_weight*self.unweighted_task_discriminability_loss.mean()
		else:
			self.unweighted_task_discriminability_loss = 0.
			self.task_discriminability_loss = 0.

		###########################################################
		# (1g) Compute identity losses.
		###########################################################

		# If "translating" source domain z., 
		# if domain==0:
		# 	self.unweighted_identity_translation_loss = self.compute_identity_loss(update_dictionary)			
		# else:
		# 	self.unweighted_identity_translation_loss = 0.
		self.unweighted_identity_translation_loss = 0.
		self.identity_translation_loss = self.args.identity_translation_loss_weight*self.unweighted_identity_translation_loss

		###########################################################
		# (1h) Compute supervised set based density loss.
		###########################################################

		if domain==1 and self.args.supervised_set_based_density_loss:
			self.forward_set_loss = - (update_dictionary['forward_set_based_supervised_loss']*self.source_manager.batch_mask).sum()/(self.source_manager.batch_mask.sum())
			self.backward_set_loss = - (update_dictionary['backward_set_based_supervised_loss']*self.target_manager.batch_mask).sum()/(self.target_manager.batch_mask.sum())
			self.unweighted_supervised_set_based_density_loss = self.forward_set_loss + self.backward_set_loss
		else:
			self.unweighted_supervised_set_based_density_loss = 0.

		self.datapoint_masked_supervised_set_based_density_loss = self.supervised_datapoints_multiplier*self.unweighted_supervised_set_based_density_loss
		self.supervised_set_based_density_loss = self.args.supervised_set_based_density_loss_weight*self.datapoint_masked_supervised_set_based_density_loss

		###########################################################
		# (1i) Finally, compute total losses. 
		###########################################################

		# Total discriminability loss. 
		self.total_discriminability_loss = self.discriminability_loss + self.z_trajectory_discriminability_loss + self.task_discriminability_loss 

		# Total encoder loss: 
		# self.total_VAE_loss = self.VAE_loss + self.total_discriminability_loss + self.equivariance_loss + self.cross_domain_supervision_loss	
		self.total_VAE_loss = self.VAE_loss + self.total_discriminability_loss + self.cross_domain_supervision_loss	+ self.identity_translation_loss

		if not(self.skip_vae):
			# Go backward through the generator (encoder / decoder), and take a step. 
			self.total_VAE_loss.backward()

			if self.args.setting in ['jointfixembed', 'fixembed']:
				# If we are in the translation model setting, use self.optimizer rather either source / target policy manager. 
				self.optimizer.step()
			else:
				policy_manager.optimizer.step()

		#########################################################################
		#########################################################################
		# (2) Next, update the discriminators based on domain label.
		#########################################################################
		#########################################################################

		# Zero gradients of discriminator(s).
		self.discriminator_optimizer.zero_grad()

		###########################################################
		# (2a) Compute Z-discriminator loss.
		###########################################################

		# If we tried to zero grad the discriminator and then use NLL loss on it again, Pytorch would cry about going backward through a part of the graph that we already \ 
		# went backward through. Instead, just pass things through the discriminator again, but this time detaching latent_z. 
		discriminator_logprob, discriminator_prob = self.discriminator_network(update_dictionary['detached_latent_z'])

		# Compute discriminator loss for discriminator. 
		# self.discriminator_loss = self.negative_log_likelihood_loss_function(discriminator_logprob.squeeze(1), torch.tensor(domain).to(device).long().view(1,))		
		self.unweighted_discriminator_loss = self.negative_log_likelihood_loss_function(discriminator_logprob.view(-1,2), domain_label).mean()
		self.discriminator_loss = self.args.discriminator_weight*self.unweighted_discriminator_loss

		###########################################################
		# (2b) Compute Z-trajectory discriminator loss. 
		###########################################################

		if self.args.z_trajectory_discriminator or self.args.z_transform_discriminator:
			z_trajectory_discriminator_logprob, z_trajectory_discriminator_prob = self.z_trajectory_discriminator.get_probabilities(update_dictionary['z_trajectory'].detach())
			self.unmasked_z_trajectory_discriminator_loss = self.negative_log_likelihood_loss_function(z_trajectory_discriminator_logprob.view(-1,2), traj_domain_label)
			# Mask the z transform discriminator loss based on whether or not this particular latent_z, latent_z transformation tuple should be used to train the discriminator.
			self.unweighted_z_trajectory_discriminator_loss = (update_dictionary['z_trajectory_weights'].view(-1,)*self.unmasked_z_trajectory_discriminator_loss)
			self.z_trajectory_discriminator_loss = self.args.z_trajectory_discriminator_weight*self.unweighted_z_trajectory_discriminator_loss.mean()
		else:
			self.z_trajectory_discriminator_loss = 0.

		###########################################################
		# (2c) Compute Task-Discriminator loss.
		###########################################################
		
		if self.args.task_discriminability:
			task_discriminator_logprob, _ = self.task_discriminators[update_dictionary['sample_task_id']].get_probabilities(update_dictionary['translated_latent_z'].detach())
			self.unweighted_task_discriminator_loss = self.negative_log_likelihood_loss_function(task_discriminator_logprob.view(-1,2), traj_domain_label)
			self.task_discriminator_loss = self.args.task_discriminator_weight*self.unweighted_task_discriminator_loss.mean()
		else:
			self.unweighted_task_discriminator_loss = 0.
			self.task_discriminator_loss = 0.

		###########################################################
		# (2d) Merge discriminator losses.
		###########################################################

		self.total_discriminator_loss = self.discriminator_loss + self.z_trajectory_discriminator_loss + self.task_discriminator_loss

		###########################################################
		# (2e) Now update discriminator(s).
		###########################################################

		if not(self.skip_discriminator):
			# Now go backward and take a step.
			self.total_discriminator_loss.backward()
			self.discriminator_optimizer.step()

	def run_iteration(self, counter, i, domain=None, special_indices=None, skip_viz=False):

		# Phases: 
		# Phase 1:  Train encoder-decoder for both domains initially, so that discriminator is not fed garbage. 
		# Phase 2:  Train encoder, decoder for each domain, and discriminator concurrently. 

		# Algorithm: 
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use (source or target, i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain. 
		# 		# 3) Encode trajectory segments into latent z's and compute likelihood of trajectory actions under the decoder.
		# 		# 4) Feed into discriminator, get likelihood of each domain.
		# 		# 5) Compute and apply gradient updates. 

		# Remember to make domain agnostic function calls to encode, feed into discriminator, get likelihoods, etc. 

		# (0) Setup things like training phases, epislon values, etc.
		self.set_iteration(counter)

		# (1) Select which domain to run on. This is supervision of discriminator.
		# Use same domain across batch for simplicity. 
		if domain is None:
			domain = np.random.binomial(1,0.5)
		self.counter = counter

		# (1.5) Get domain policy manager. 
		policy_manager = self.get_domain_manager(domain)
				
		# (2) & (3) Get trajectory segment and encode and decode. 
		update_dictionary = {}
		update_dictionary['subpolicy_inputs'], update_dictionary['latent_z'], update_dictionary['loglikelihood'], update_dictionary['kl_divergence'] = self.encode_decode_trajectory(policy_manager, i)

		if update_dictionary['latent_z'] is not None:
			# (4) Feed latent z's to discriminator, and get discriminator likelihoods. 
			# In the joint transfer case:
			update_dictionary['discriminator_logprob'], discriminator_prob = self.discriminator_network(update_dictionary['latent_z'])
			update_dictionary['detached_latent_z'] = update_dictionary['latent_z'].detach()
			# (5) Compute and apply gradient updates. 
			# self.update_networks(domain, policy_manager, loglikelihood, kl_divergence, discriminator_logprob, latent_z)
			self.update_networks(domain, policy_manager, update_dictionary)

			# Now update Plots. 			
			# viz_dict = {'domain': domain, 'discriminator_probs': discriminator_prob.squeeze(0).mean(axis=0)[domain].detach().cpu().numpy()}
			viz_dict = {'domain': domain, 'discriminator_probs': discriminator_prob[...,domain].detach().cpu().numpy().mean()}

			if not(skip_viz):
				self.update_plots(counter, viz_dict)

	def compute_neighbors(self, computed_sets=False):
		
		# First make sure neighbor objects are set. 
		self.set_neighbor_objects(computed_sets=computed_sets)

		# Now that neighbor objects are set, compute neighbors. 			
		if self.args.setting in ['jointtransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
			_, self.source_target_neighbors = self.source_neighbors_object.kneighbors(self.target_latent_zs)
			_, self.target_source_neighbors = self.target_neighbors_object.kneighbors(self.source_latent_zs)
		else:
			_, self.source_target_neighbors = self.source_neighbors_object.kneighbors(self.target_manager.latent_z_set)
			_, self.target_source_neighbors = self.target_neighbors_object.kneighbors(self.source_manager.latent_z_set)

	def set_neighbor_objects(self, computed_sets=False):

		with torch.no_grad():
			if not(computed_sets):
				self.source_manager.get_trajectory_and_latent_sets()
				self.target_manager.get_trajectory_and_latent_sets()

		# print("Embed before computing neighbor objects.")
		# embed()
		# Reassembling for neearest neighbor object creation.
		if self.args.setting in ['jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
			# self.source_latent_z_set = np.concatenate(self.source_manager.latent_z_set)
			# self.target_latent_z_set = np.concatenate(self.target_manager.latent_z_set)

			self.source_latent_z_set = []
			self.target_latent_z_set = []

			for b in range(self.args.batch_size):
				self.source_latent_z_set.append(self.source_manager.latent_z_set[b][0])
				self.target_latent_z_set.append(self.target_manager.latent_z_set[b][0])
		else:
			self.source_latent_z_set = self.source_manager.latent_z_set
			self.target_latent_z_set = self.target_manager.latent_z_set

		# Compute nearest neighbors for each set. First build KD-Trees / Ball-Trees. 
		self.source_neighbors_object = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.source_latent_z_set)
		self.target_neighbors_object = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.target_latent_z_set)

		self.neighbor_obj_set = True

	def evaluate_translated_trajectory_segment_distances(self):
		# Evaluate translated trajectory distances - from source to target and target to source domains. 

		# Remember, absolute trajectory differences is meaningless, since the data is randomly initialized across the state space. 
		# Instead, compare actions. I.e. first compute differences along the time dimension. 

		# If we're in the jointtransfer setting, we should be using segmented_trajectory_set(s) instead of trajectory_set.
		source_traj_actions = np.diff(self.source_manager.trajectory_set,axis=1)
		target_traj_actions = np.diff(self.target_manager.trajectory_set,axis=1)

		source_target_trajectory_diffs = (source_traj_actions - target_traj_actions[self.source_target_neighbors.squeeze(1)])
		self.source_target_trajectory_distance = copy.deepcopy(np.linalg.norm(source_target_trajectory_diffs,axis=(1,2)).mean())

		target_source_trajectory_diffs = (target_traj_actions - source_traj_actions[self.target_source_neighbors.squeeze(1)])
		self.target_source_trajectory_distance = copy.deepcopy(np.linalg.norm(target_source_trajectory_diffs,axis=(1,2)).mean())

		self.source_target_trajectory_normalized_distance = self.source_target_trajectory_distance/(np.linalg.norm(source_traj_actions, axis=2).mean())
		self.target_source_trajectory_normalized_distance = self.target_source_trajectory_distance/(np.linalg.norm(target_traj_actions, axis=2).mean())		

	def visualize_translated_trajectories(self):

		# Basically overall flow: (remember, this will be computationally pretty expensive.)
		# 1) For N input trajectories:
		# 	# 0) Get source trajectory (for start state for now).
		# 	# 1) Get target trajectory. 
		# 	# 2) Encode target trajectory as sequence of target z's. 
		#	# 3) Translate sequence of target z's from target domain to source domain. 
		# 	# 4) Feed translated sequence of target z's., along with start state of source trajectory into cross_domain decoding, and decode into a translated target-> source trajectory. 
		# 	# 5) Visualize original target trajectory, and the translated target to source trajectory. 

		number_of_batches = 1
		self.number_of_datapoints_per_batch = self.args.number_of_visualized_translations
		self.start_index = 0  # Changed here

		with torch.no_grad():

			for i in range(number_of_batches):
				
				# 0) Get source trajectory.				
				source_input_dict, _, _ = self.encode_decode_trajectory(self.source_manager, i)

				# 1) Get target trajectory. 
				# 2) Encode target trajectory as sequence of target z's. 
				target_input_dict, target_var_dict, _ = self.encode_decode_trajectory(self.target_manager, i)

				# 3) Translate sequence of target z's from target domain to source domain. 
				translated_latent_z = self.translate_latent_z(target_var_dict['latent_z_indices'], latent_b=target_var_dict['latent_b'])

				# 4) Feed translated sequence of target z's., along with start state of source trajectory into cross_domain decoding, and decode into a translated target-> source trajectory. 
				# Remember, must decode in the output domain of the translation model - usually the source domain (unless cycle setting.)
				cross_domain_decoding_dict = self.cross_domain_decoding(0, self.source_manager, translated_latent_z, start_state=source_input_dict['sample_traj'][0], rollout_length=translated_latent_z.shape[0])

				# Also visualize trajectories 0 and 1 if we have mujoco.
				if self.args.no_mujoco==0 and i==0:

					self.traj_viz_dir_name = os.path.join(self.args.logdir,self.args.name,"TrajVizDict")

					if not(os.path.isdir(self.traj_viz_dir_name)):
						os.mkdir(self.traj_viz_dir_name)

					# First unnormalize the trajectories.
					unnormalized_original_target_traj = (target_input_dict['sample_traj']*self.target_manager.norm_denom_value)+self.target_manager.norm_sub_value														
					# unnormalized_target_traj = (cross_domain_decoding_dict['differentiable_trajectory'].detach().cpu().numpy()*self.target_manager.norm_denom_value)+self.target_manager.norm_sub_value
					# Remember, the cross domain trajectory needs to be unnormalized with the source normalization values.. 					

					unnormalized_translated_target_traj = (cross_domain_decoding_dict['differentiable_trajectory'].detach().cpu().numpy()*self.source_manager.norm_denom_value)+self.source_manager.norm_sub_value
					
					# Smoothen trajectories as needed... 
					if self.args.target_domain in ['Roboturk']:
						unnormalized_original_target_traj = self.smoothen_sawyer_trajectories(unnormalized_original_target_traj)
					if self.args.source_domain in ['Roboturk']:
						# Remember, here, the translated target is in the source domain! 
						unnormalized_translated_target_traj = self.smoothen_sawyer_trajectories(unnormalized_translated_target_traj)

					self.gif_logs = {}

					print("### RUN VIZ TRANS TRAJ")
					# Now for these many trajectories:
					for k in range(self.start_index,self.start_index+self.number_of_datapoints_per_batch):
						# Now visualize the original .. target trajectory. 			

						if self.args.target_domain in ['GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', 'GRABObject']:
							GRAB_gif_name = self.GRAB_trajectory_ID[k].lstrip(self.args.target_datadir+'/')
						else:
							GRAB_gif_name = None
						
						# self.gif_logs['Traj{0}_OriginalTarget_Traj'.format(k)] = np.array(self.target_manager.visualizer.visualize_joint_trajectory(unnormalized_original_target_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="E{0}_C{1}_Traj{2}_OriginalTargetTraj.gif".format(self.current_epoch_running, self.counter, k), return_and_save=True, end_effector=self.args.ee_trajectories))
						self.gif_logs['Traj{0}_OriginalTarget_Traj'.format(k)] = np.array(self.target_manager.visualizer.visualize_joint_trajectory(unnormalized_original_target_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="Traj{0}_OriginalTargetTraj.gif".format(k), return_and_save=True, end_effector=self.args.ee_trajectories, additional_info=GRAB_gif_name))
						# Now visualize the translated target trajectory. 
						# self.gif_logs['Traj{0}_TranslatedTarget_Traj'.format(k)] = np.array(self.source_manager.visualizer.visualize_joint_trajectory(unnormalized_translated_target_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="E{0}_C{1}_Traj{2}_TranslatedTargetTraj.gif".format(self.current_epoch_running, self.counter, k), return_and_save=True, end_effector=self.args.ee_trajectories))
						self.gif_logs['Traj{0}_TranslatedTarget_Traj'.format(k)] = np.array(self.source_manager.visualizer.visualize_joint_trajectory(unnormalized_translated_target_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="Traj{0}_TranslatedTargetTraj.gif".format(k), return_and_save=True, end_effector=self.args.ee_trajectories))

		# Segment these GIFs and save them.
		self.segment_source_target_gifs(target_var_dict['latent_b'], latent_zs=translated_latent_z.detach().cpu().numpy())

	def smoothen_sawyer_trajectories(self, trajectory):

			return gaussian_filter1d(trajectory,self.args.smoothing_kernel_bandwidth,axis=0,mode='nearest')

	def segment_source_target_gifs(self, latent_b, latent_zs=None):

		# print("embedding in segment source target gifs")
		# embed()

		self.segmented_gif_logs = {}
		self.translated_latent_zs_for_downstream = []

		# Remember, we're doing this for 2 batch elements. 
		# for k in range(self.number_of_datapoints_per_batch):
		for k in range(self.start_index,self.start_index+self.number_of_datapoints_per_batch):			
			# Set source and target gif from gif_logs object.
			target_gif = self.gif_logs['Traj{0}_OriginalTarget_Traj'.format(k)]
			translatedtarget_gif = self.gif_logs['Traj{0}_TranslatedTarget_Traj'.format(k)]

			# Now segment these gifs.
			segmentation_indices = torch.where(latent_b[:,k])[0]

			if latent_zs is not None:
				latent_zs_for_element = []

			for j in range(len(segmentation_indices)-1):
				
				self.segmented_gif_logs['Traj{0}_OrigTarget_Segment{1}'.format(k,j)] = target_gif[segmentation_indices[j]:segmentation_indices[j+1]]
				self.segmented_gif_logs['Traj{0}_TranslatedTarget_Segment{1}'.format(k,j)] = translatedtarget_gif[segmentation_indices[j]:segmentation_indices[j+1]]			
			
				# If we have a latent z object, store the z's... 
				if latent_zs is not None:
					latent_zs_for_element.append(latent_zs[segmentation_indices[j],k])
				
			self.segmented_gif_logs['Traj{0}_OrigTarget_Segment{1}'.format(k,len(segmentation_indices)-1)] = target_gif[segmentation_indices[-1]:]			
			self.segmented_gif_logs['Traj{0}_TranslatedTarget_Segment{1}'.format(k,len(segmentation_indices)-1)] = translatedtarget_gif[segmentation_indices[-1]:]

			# If we have a latent z object, store the z's... 
			if latent_zs is not None:
				latent_zs_for_element.append(latent_zs[segmentation_indices[-1],k])

			# Now add this to global set.
			self.translated_latent_zs_for_downstream.append(copy.deepcopy(latent_zs_for_element))

		# Now save this z set. 
		np.save(os.path.join(self.traj_viz_dir_name,"Translated_Zs.npy"),self.translated_latent_zs_for_downstream)

		# if self.args.data not in ['GRAB','GRABHand','GRABArmHand'] and self.args.target_domain not in ['']:
		# if self.args.target_domain not in ['GRAB','GRABHand','GRABArmHand']:
		if True:
			
			# Now save all the gifs we created.
			for key in self.segmented_gif_logs.keys():			
				# Save. 
				imageio.mimsave(os.path.join(self.traj_viz_dir_name,"{0}.gif".format(key)), list(self.segmented_gif_logs[key]))

		# print("embedding in segment source target gifs")
		# embed()

	def evaluate_translated_trajectory_distances(self, just_visualizing=False):

		# Basically overall flow: (remember, this will be computationally pretty expensive.)
		# 1) For N input trajectories:
		# 	# 2) Get source trajectory. 
		# 	# 3) Encode trajectory as sequence of source z's and decode trajectory in same domain to get reconstruction source trajectory. 
		# 	# 4) Get corresponding target trajectory. 
		# 	# 5) Encode target trajectory as sequence of target z's. 
		#	# 6) Translate sequence of target z's from target domain to source domain. 
		# 	# 7) Feed translated sequence of target z's., along with start state of source trajectory into cross_domain decoding, and decode into a translated target-> source trajectory. 
		# 	# 8) Evaluate L2 distance between the reconstructed source trajectory and the reconstructed translated target->source trjaectory. 
		# Alternately: 
		# 	# 8) Evaluate L2 distance between the original source trajectory and the reconstructed translated target->source trjaectory. 
		
		average_trajectory_reconstruction_error = 0.

		with torch.no_grad():

			# for i in range(1):
			# Copy this range over from,.... train.
			
			# self.eval_extent = self.extent
			if just_visualizing:
				self.eval_extent = 1
			else:
				self.eval_extent = 500 		

			for i in range(0,self.eval_extent,self.args.batch_size):

				# t3 = time.time()
				# 2) Get source trajectory.
				# 3) Encode trajectory. 
				source_input_dict, source_var_dict, _ = self.encode_decode_trajectory(self.source_manager, i)

				# 4) Get corresponding target trajectory. 
				# 5) Encode target trajectory as sequence of target z's. 
				target_input_dict, target_var_dict, _ = self.encode_decode_trajectory(self.target_manager, i)

				# 6) Translate sequence of target z's from target domain to source domain. 
				translated_latent_z = self.translate_latent_z(target_var_dict['latent_z_indices'], latent_b=target_var_dict['latent_b'])

				# 7) Feed translated sequence of target z's., along with start state of source trajectory into cross_domain decoding, and decode into a translated target-> source trajectory. 
				# Remember, must decode in the output domain of the translation model - usually the source domain (unless cycle setting.)
				cross_domain_decoding_dict = self.cross_domain_decoding(0, self.source_manager, translated_latent_z, start_state=source_input_dict['sample_traj'][0], rollout_length=translated_latent_z.shape[0])

				# 8) Evaluate L2 distance between the reconstructed source trajectory and the reconstructed translated target->source trjaectory. 				
				# OR:
				# 8) Evaluate L2 distance between the original source trajectory and the reconstructed translated target->source trjaectory. 
				if just_visualizing:
					traj_recon_error = 0.
				else:
					traj_recon_error = ((cross_domain_decoding_dict['differentiable_trajectory'].detach().cpu().numpy() - source_input_dict['sample_traj'])**2).mean(axis=(0,2)).sum()
				average_trajectory_reconstruction_error += traj_recon_error
				
				
				# Also visualize trajectories 0 and 1 if we have mujoco.
				if self.args.no_mujoco==0 and i==0:

					self.traj_viz_dir_name = os.path.join(self.args.logdir,self.args.name,"TrajVizDict")
					if not(os.path.isdir(self.traj_viz_dir_name)):
						os.mkdir(self.traj_viz_dir_name)
									
					# First unnormalize the trajectories.
					unnormalized_source_traj = (source_input_dict['sample_traj']*self.source_manager.norm_denom_value)+self.source_manager.norm_sub_value														
					unnormalized_translatedtarget_traj = (cross_domain_decoding_dict['differentiable_trajectory'].detach().cpu().numpy()*self.target_manager.norm_denom_value)+self.target_manager.norm_sub_value

					self.gif_logs = {}
					# Now for these many trajectories:
					for k in range(2):
						# Now visualize the source trajectory. 
						# source_gif = self.source_manager.visualizer.visualize_joint_trajectory(unnormalized_source_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="Traj{0}_SourceTraj.gif".format(self.current_epoch_running, self.counter, k), return_and_save=True, end_effector=self.args.ee_trajectories)
						source_gif = self.source_manager.visualizer.visualize_joint_trajectory(unnormalized_source_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="Traj{0}_SourceTraj.gif".format(self.current_epoch_running, self.counter, k), return_and_save=True, end_effector=self.args.ee_trajectories)
						self.gif_logs['Traj{0}_Source_Traj'.format(k)] = np.array(source_gif)

						# Now visualize the target trajectory. 
						# THIS IS ACTUALLY IN SOURCE DOMAIN... USE SOURCE VISUALIZER.
						# target_gif = self.source_manager.visualizer.visualize_joint_trajectory(unnormalized_translatedtarget_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="Traj{0}_TargetTranslatedTraj.gif".format(self.current_epoch_running, self.counter, k), return_and_save=True, end_effector=self.args.ee_trajectories)
						target_gif = self.source_manager.visualizer.visualize_joint_trajectory(unnormalized_translatedtarget_traj[:,k], gif_path=self.traj_viz_dir_name, gif_name="Traj{0}_TargetTranslatedTraj.gif".format(k), return_and_save=True, end_effector=self.args.ee_trajectories)
						self.gif_logs['Traj{0}_Target_Traj'.format(k)] = np.array(target_gif)

						# Segment up gifs..
						# self.segment_source_target_gifs(target_var_dict['latent_b'])

		average_trajectory_reconstruction_error /= (self.extent//self.args.batch_size+1)*self.args.batch_size

		return average_trajectory_reconstruction_error

	def evaluate_skill_sequences_across_domains(self):

		################################################ 
		# 1) Evaluate the same skill sequence across both domains. 
		################################################ 

		# For the same skill sequences, evaluate:
		# 1 a) Average discrepancy between corresponding z sequences across domains. 
		# 1 b) Average discrepancy between z transition sequences across domains. 

		self.average_corresponding_z_sequence_error = np.zeros(self.source_manager.max_viz_trajs)
		self.average_corresponding_z_transition_sequence_error = np.zeros(self.source_manager.max_viz_trajs)

		print("Running evaluate_skill_sequences_across_domains for {0} skill sequences.".format(self.source_manager.max_viz_trajs))
		# For some set of skill sequences.
		for k in range(self.source_manager.max_viz_trajs):

			# Run individual source and target managers evaluate_similar_skill_sequences(k). 
			self.source_manager.evaluate_contextual_representations(specific_index=k, skip_same_skill=True)
			self.target_manager.evaluate_contextual_representations(specific_index=k, skip_same_skill=True)
		
			################################################
			# Now evaluate the average discrepancy between corresponding z sequences across domains, measured as RMS error. 
			################################################		
			j=0
			limit = self.args.batch_size
			# for j in range(self.args.batch_size):
			while j<limit:
				if len(self.source_manager.latent_z_set[j])>4 or len(self.target_manager.latent_z_set[j])>4:
					self.source_manager.latent_z_set.pop(j)
					self.target_manager.latent_z_set.pop(j)
					limit-=1
				else:
					j+=1

			source_z = np.concatenate([self.source_manager.latent_z_set])
			target_z = np.concatenate([self.target_manager.latent_z_set])				
			self.average_corresponding_z_sequence_error[k] = ((source_z-target_z)**2).mean()

			################################################
			# Now evaluate the average discrepancy between corresponding z transition sequences across domains, measured as RMS error. 
			################################################

			source_z_diffs = np.diff(source_z,axis=1)
			target_z_diffs = np.diff(target_z,axis=1)
			self.average_corresponding_z_transition_sequence_error[k] = ((source_z_diffs-target_z_diffs)**2).mean()

	def setup_crossdomain_joint_evaluation(self):
		
		# Set up eval for source.
		self.source_manager.assemble_joint_skill_embedding_space()
		self.source_manager.global_z_set = copy.deepcopy(self.source_manager.embedding_latent_z_set_array)

		# Set up eval for target.
		self.target_manager.assemble_joint_skill_embedding_space()
		self.target_manager.global_z_set = copy.deepcopy(self.target_manager.embedding_latent_z_set_array)

		# Setting dummy values for now.
		self.source_target_trajectory_distance = 0
		self.target_source_trajectory_distance = 0

		self.source_target_trajectory_normalized_distance = 0
		self.target_source_trajectory_normalized_distance = 0

	def evaluate_correspondence_metrics(self, computed_sets=True):

		print("Evaluating correspondence metrics.")
		# Evaluate the correspondence and alignment metrics. 

		# Now that the neighbors have been computed, compute translated trajectory reconstruction errors via nearest neighbor z's. 
		# Only use this version of evaluating trajectory distance.
		if not(self.args.setting in ['jointtransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']):		
			
			# Whether latent_z_sets and trajectory_sets are already computed for each manager.
			self.compute_neighbors(computed_sets)
			self.evaluate_translated_trajectory_segment_distances()		

		if self.args.setting in ['jointtransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']:
			
			# If we are actually in joint transfer setting: 

			################################################
			# Evaluate the following correspondence metrics.					
			################################################

			# This evaluates reconstruction error of full length trajectories when translated via z's using the translation model.
			self.average_translated_trajectory_reconstruction_error = self.evaluate_translated_trajectory_distances()

			if self.check_toy_dataset():
				################################################ 
				# 1) First runs some things that are needed for evaluation. 
				################################################ 

				self.setup_crossdomain_joint_evaluation()

				################################################ 
				# 2) Evaluate the same skill sequence across both domains. 
				################################################ 

				self.evaluate_skill_sequences_across_domains()

		##########################################
		# Add more evaluation metrics here. 
		##########################################

		if not(self.args.setting in ['jointtransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer']) or self.check_toy_dataset():
			# self.evaluate_discriminator_accuracy()
			# self.evaluate_cycle_reconstruction_error()

			# Reset variables to prevent memory leaks.
			# source_neighbors_object = None
			# target_neighbors_object = None
			del self.source_neighbors_object
			del self.target_neighbors_object
			self.neighbor_obj_set = False		

	def evaluate(self, model=None):

		# Evaluating Transfer - we just want embeddings of both source and target; so run evaluate of both source and target policy managers. 		

		# Instead of parsing and passing model to individual source and target policy managers, just load using the transfer policy manager, and then run eval. 
		if model is not None: 
			self.load_all_models(model)

		# # Run source policy manager evaluate. 
		# self.source_manager.evaluate(suffix="Source")

		# # Run target policy manager evaluate. 
		# self.target_manager.evaluate(suffix="Target")

		# Evaluate semantic labels.
		self.evaluate_semantic_accuracy()

		# Evaluate metrics. 
		self.evaluate_correspondence_metrics()

	def automatic_evaluation(self, e):
		pass

	def check_toy_dataset(self):
		return self.args.source_domain in ['ContinuousNonZero','ToyContext'] and self.args.target_domain in ['ContinuousNonZero','ToyContext']

	def initialize_training_batches(self, skip=True):

		print("Running Initialize Training Batches for Transfer setting.")
		# Find out which domain has a bigger batch size. 
		# Remember to consider: Input: Statex2 x Time x Z dimensions. 
		# Latent z dimensions are same across both, so sufficient to compare state dims x 2 (i.e. input_size), and time. 

		source_batch_size = self.source_manager.input_size*self.source_manager.max_batch_size
		target_batch_size = self.target_manager.input_size*self.target_manager.max_batch_size

		# In case of tie, doesn't matter. # Actually, not true.
		# max_batch_domain = int(target_batch_size<source_batch_size)
		# We forgot that the translation only actually happens when the domain is 1. 
		max_batch_domain = 1
		max_batch_index = self.get_domain_manager(max_batch_domain).max_batch_size_index		

		# Now compute statistics. 
		self.counter = 0 
		self.set_iteration(self.counter)
		self.compute_z_statistics()

		# Now actually run iteration of the joint transfer / joint embed transfer with these specal indices.
		counter = 0
		print("Running initializing iteration.")
		self.run_iteration(counter, max_batch_index, domain=max_batch_domain, skip_viz=skip)

	def compute_z_statistics(self):

		# Compute z statistics over an epoch, for both the source and target domains. 
		self.source_z_mean = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.source_z_var = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.source_z_std = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.target_z_mean = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.target_z_var = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.target_z_std = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.joint_z_mean = torch.zeros((self.args.z_dimensions)).to(device).float()
		self.joint_z_std = torch.zeros((self.args.z_dimensions)).to(device).float()

		mean_counter = 0
		
		if self.args.z_normalization is None:
			self.source_z_std = torch.ones((self.args.z_dimensions)).to(device).float()
			self.target_z_std = torch.ones((self.args.z_dimensions)).to(device).float()
		else:
			print("Computing Z Statistics.")
			with torch.no_grad():

				for i in range(0,self.extent,self.args.batch_size):
					# Using 1, not batch_size, because we are using z.mean. 
					mean_counter += 1

					# Copy over
					# self.old_source_mean = copy.deepcopy(self.source_z_mean)
					# self.old_target_mean = copy.deepcopy(self.target_z_mean)
					self.old_source_mean = self.source_z_mean.clone().detach()
					self.old_target_mean = self.target_z_mean.clone().detach()

					_ , source_var_dict, _ = self.encode_decode_trajectory(self.source_manager, i, domain=0, initialize_run=True)
					self.source_z_mean = self.source_z_mean + (source_var_dict['latent_z_indices'].mean(axis=(0,1)) - self.source_z_mean)/mean_counter
					self.source_z_var = self.source_z_var + ((source_var_dict['latent_z_indices'].mean(axis=(0,1)) - self.old_source_mean)*(source_var_dict['latent_z_indices'].mean(axis=(0,1))-self.source_z_mean)-self.source_z_var)/mean_counter

					_ , target_var_dict, _ = self.encode_decode_trajectory(self.target_manager, i, domain=1, initialize_run=True)
					self.target_z_mean = self.target_z_mean + (target_var_dict['latent_z_indices'].mean(axis=(0,1)) - self.target_z_mean)/mean_counter
					self.target_z_var = self.target_z_var + ((target_var_dict['latent_z_indices'].mean(axis=(0,1)) - self.old_target_mean)*(target_var_dict['latent_z_indices'].mean(axis=(0,1))-self.target_z_mean)-self.target_z_var)/mean_counter
		
			# Now compute standard deviation. 
			self.source_z_std = torch.sqrt(self.source_z_var)
			self.target_z_std = torch.sqrt(self.target_z_var)

			self.joint_z_mean = (self.source_z_mean+self.target_z_mean)/2
			self.joint_z_var = ((self.source_z_var+self.source_z_mean**2) + (self.target_z_var+self.target_z_mean**2))/2 - self.joint_z_mean**2

			self.joint_z_std = torch.sqrt(self.joint_z_var)		

		if self.args.z_normalization=='global':

			self.source_z_mean = self.joint_z_mean
			self.target_z_mean = self.joint_z_mean
			self.source_z_std = self.joint_z_std
			self.target_z_std = self.joint_z_std
	
	def compute_aggregate_chamfer_loss(self):
		
		with torch.no_grad():
			chamfer_loss = chamfer_distance(torch.tensor(self.source_latent_zs).to(device).unsqueeze(0), torch.tensor(self.target_latent_zs).to(device).unsqueeze(0))[0]
		return chamfer_loss.detach().cpu().numpy()

	def construct_density_embeddings(self, log_dict):

		# Construct density coded embeddings of Z's
		log_dict = self.construct_single_directional_density_embeddings(log_dict, domain=0)
		log_dict = self.construct_single_directional_density_embeddings(log_dict, domain=1)

		# Now construct Z Tuple GMM density coded embeddings of Z Tuples. 
		log_dict = self.construct_single_directional_z_tuple_density_embeddings(log_dict, domain=0)
		log_dict = self.construct_single_directional_z_tuple_density_embeddings(log_dict, domain=1)

		return log_dict

	def construct_single_directional_density_embeddings(self, log_dict, domain):

		##################################################
		# Plot density coded embeddings. 
		##################################################
		
		if domain==0:
			point_set = self.target_latent_zs
			opp_point_set_length = len(self.source_latent_zs)
			prefix = "Forward"
		else:
			point_set = self.source_latent_zs
			opp_point_set_length = len(self.target_latent_zs)
			prefix = "Backward"

		# Evaluate log_probs of target..
		log_probs = self.query_GMM_density(evaluation_domain=domain, point_set=point_set).detach().cpu().numpy()
		color_scale = 50

		# print("Embedding in update plots of dnesity based thing..")		

		# Colors that count sizes..
		if domain==0:
			colors = np.concatenate([color_scale*np.ones(opp_point_set_length), log_probs])
		else:
			colors = np.concatenate([log_probs, color_scale*np.ones(opp_point_set_length)])

		# # This assumes same number of z's in both ... This may not be true.. 
		# if domain==0:
		# 	colors = np.concatenate([color_scale*np.ones_like(log_probs), log_probs])
		# else:
		# 	colors = np.concatenate([log_probs, color_scale*np.ones_like(log_probs)])

		# Embed and transform - just the target_z_tensor? 
		# Do this with just the perplexity set to 30 for now.. 

		tsne_embedded_zs , _ = self.get_transform(self.shared_latent_zs)
		densne_embedded_zs , _ = self.get_transform(self.shared_latent_zs, projection='densne')
		pca_embedded_zs , _ = self.get_transform(self.shared_latent_zs, projection='pca')

		# tsne_embedded_zs , _ = self.get_transform(self.target_latent_zs)
		# densne_embedded_zs , _ = self.get_transform(self.target_latent_zs, projection='densne')

		# if domain==1:
		# 	print("Embedding in construct density embeddings")
		# 	embed()

		tsne_image = self.plot_density_embedding(tsne_embedded_zs, colors, "{0} Density Coded TSNE Embeddings.".format(prefix))
		densne_image = self.plot_density_embedding(densne_embedded_zs, colors, "{0} Density Coded DENSNE Embeddings.".format(prefix))
		pca_image = self.plot_density_embedding(pca_embedded_zs, colors, "{0} Density Coded PCA Embeddings.".format(prefix))

		##################################################
		# Now add to wandb log_dict.
		##################################################

		log_dict['{0} Density Coded TSNE Embeddings Perp30'.format(prefix)] = self.return_wandb_image(tsne_image)
		log_dict['{0} Density Coded DENSNE Embeddings Perp30'.format(prefix)] = self.return_wandb_image(densne_image)
		log_dict['{0} Density Coded PCA Embeddings'.format(prefix)] = self.return_wandb_image(pca_image)

		return log_dict

	def construct_single_directional_z_tuple_density_embeddings(self, log_dict, domain):
		
			##################################################
		# Plot density coded embeddings. 
		##################################################
		
		if domain==0:
			point_set = self.target_z_tuples
			opp_point_set_length = len(self.source_z_tuples)
			prefix = "Forward"
		else:
			point_set = self.source_z_tuples
			opp_point_set_length = len(self.target_z_tuples)
			prefix = "Backward"

		# Evaluate log_probs of target..
		log_probs = self.query_GMM_density(evaluation_domain=domain, point_set=point_set, GMM=self.Z_Tuple_GMM_list[domain]).detach().cpu().numpy()
		color_scale = 50

		# print("Embedding in update plots of dnesity based thing..")		

		# Colors that count sizes..
		if domain==0:
			colors = np.concatenate([color_scale*np.ones(opp_point_set_length), log_probs])
		else:
			colors = np.concatenate([log_probs, color_scale*np.ones(opp_point_set_length)])

		# # This assumes same number of z's in both ... This may not be true.. 
		# if domain==0:
		# 	colors = np.concatenate([color_scale*np.ones_like(log_probs), log_probs])
		# else:
		# 	colors = np.concatenate([log_probs, color_scale*np.ones_like(log_probs)])

		# Embed and transform - just the target_z_tensor? 
		# Do this with just the perplexity set to 30 for now.. 

		tsne_embedded_zs , _ = self.get_transform(self.shared_z_tuples)

		try:
			densne_embedded_zs , _ = self.get_transform(self.shared_z_tuples, projection='densne')
		except:
			densne_embedded_zs = None

		pca_embedded_zs , _ = self.get_transform(self.shared_z_tuples, projection='pca')

		# tsne_embedded_zs , _ = self.get_transform(self.target_latent_zs)
		# densne_embedded_zs , _ = self.get_transform(self.target_latent_zs, projection='densne')

		# if domain==1:
		# 	print("Embedding in construct density embeddings")
		# 	embed()

		tsne_image = self.plot_density_embedding(tsne_embedded_zs, colors, "{0} Density Coded Z Tuple TSNE Embeddings.".format(prefix))
		
		if densne_embedded_zs is not None:
			densne_image = self.plot_density_embedding(densne_embedded_zs, colors, "{0} Density Coded Z Tuple DENSNE Embeddings.".format(prefix))
	
		pca_image = self.plot_density_embedding(pca_embedded_zs, colors, "{0} Density Coded Z Tuple PCA Embeddings.".format(prefix))

		##################################################
		# Now add to wandb log_dict.
		##################################################

		log_dict['{0} Density Coded Z Tuple TSNE Embeddings Perp30'.format(prefix)] = self.return_wandb_image(tsne_image)
		
		if densne_embedded_zs is not None:
			log_dict['{0} Density Coded Z Tuple DENSNE Embeddings Perp30'.format(prefix)] = self.return_wandb_image(densne_image)
	
		log_dict['{0} Density Coded Z Tuple PCA Embeddings'.format(prefix)] = self.return_wandb_image(pca_image)
	

		return log_dict

	def plot_density_embedding(self, embedded_zs, colors, title): 

		# Now visualize TSNE image
		matplotlib.rcParams['figure.figsize'] = [5,5]
		fig = plt.figure()
		ax = fig.gca()
		
		im = ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,edgecolors='k')

		from mpl_toolkits.axes_grid1 import make_axes_locatable
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical')		

		############################################################
		# Now make the plot and generate numpy image from it. 
		############################################################
		ax.set_title("{0}".format(title),fontdict={'fontsize':15})
		fig.canvas.draw()		
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)		
		image = np.transpose(image, axes=[2,0,1])

		return image

	def setup_GMM(self):

		self.GMM_list = [self.create_GMM(evaluation_domain=0), self.create_GMM(evaluation_domain=1)]

		if self.args.z_tuple_gmm:
			
			# Set Z Tuple Means...
			self.differentiable_mean_computation(0)
			# Also set up Z Tuple GMMs
			self.Z_Tuple_GMM_list = [self.create_GMM(evaluation_domain=0, mean_point_set=self.source_z_tuple_set, differentiable_points=True, tuple_GMM=True), \
									 self.create_GMM(evaluation_domain=1, mean_point_set=self.target_z_tuple_set, differentiable_points=True, tuple_GMM=True)]

	def compute_set_based_supervised_GMM_loss(self, z_set_1, z_set_2, differentiable_outputs=False):

		# self.temporary_gmm_model = 

		# Is this metric symmetric? 
		# .. Since the Gaussian is symmetric, isn't this entire metric?

		# Well.. if the GMM components are isometric Guassians, then 
		# both forward and backward likelihoods are equal, correct? 

		forward_gmm = self.create_GMM(evaluation_domain=0,mean_point_set=z_set_1)
		# Don't really need to feed in evaluation_domain.. but might as well.
		backward_gmm = self.create_GMM(evaluation_domain=1,mean_point_set=z_set_2)

		# Now query log probabilities.
		if len(z_set_1.shape)>2 and len(z_set_1.shape)>2:

			#######################################################
			# THIS TRANSPOSING IS WHAT IS WEIRD! 
			#######################################################

			diff_forward_logprobs = torch.transpose(forward_gmm.log_prob(torch.transpose(z_set_2,1,0)),1,0)
			diff_backward_logprobs = torch.transpose(backward_gmm.log_prob(torch.transpose(z_set_1,1,0)),1,0)

			if differentiable_outputs:
				forward_logprobs = diff_forward_logprobs
				backward_logprobs = diff_backward_logprobs
			else:
				forward_logprobs = diff_forward_logprobs.mean().detach().cpu().numpy()
				backward_logprobs = diff_backward_logprobs.mean().detach().cpu().numpy()

		else:
			forward_logprobs = forward_gmm.log_prob(z_set_2).mean().detach().cpu().numpy()
			backward_logprobs = backward_gmm.log_prob(z_set_1).mean().detach().cpu().numpy()

		return forward_logprobs, backward_logprobs

		# The .view(-1,z_dim) was accumulating z's across the batch, which is wrong. Compute set based loss independently across the batch, then do mean reduction.
		# Can we recreate a batch of GMM's? # Test this... 

	def compute_aggregate_GMM_densities(self):

		# print("ABOUT TO RUN COMPUTE AGGREGATE GMM DENSITIES")
		# May need to batch this, depending on memory. 
		forward_density = self.query_GMM_density(evaluation_domain=0, point_set=self.target_latent_zs).mean()
		reverse_density = self.query_GMM_density(evaluation_domain=1, point_set=self.source_latent_zs).mean()

		return forward_density.detach().cpu().numpy(), reverse_density.detach().cpu().numpy()

	def query_GMM_density(self, evaluation_domain=0, point_set=None, differentiable_points=False, GMM=None):
		
		# if GMM is None:
		# 	if differentiable_points:
		# 		return self.GMM_list[evaluation_domain].log_prob(point_set)
		# 	else:
		# 		return self.GMM_list[evaluation_domain].log_prob(torch.tensor(point_set).to(device))
		# else:			
		# 	if differentiable_points:
		# 		return GMM.log_prob(point_set)
		# 	else:
		# 		return GMM.log_prob(torch.tensor(point_set).to(device))
	
		if GMM is None:
			GMM = self.GMM_list[evaluation_domain]

		if differentiable_points==False:
			with torch.no_grad():
				point_set = torch.tensor(point_set).to(device)
				return GMM.log_prob(point_set)

		return GMM.log_prob(point_set)

	def create_GMM(self, evaluation_domain=0, mean_point_set=None, differentiable_points=False, tuple_GMM=False):


		# Overall algorithm.
		# Preprocessing
		# 1) For N samples of datapoints from the source domain. 
		# 	# 2) Feed these input datapoints into the source domain encoder and get source encoding z. 
		#	# 3) Add Z to Source Z Set. 
		# 4) Build GMM with centers around the N Source Z set Z's.
	
		# Remember, for the setting where we have a translation model that translates from TARGET to SOURCe domains (i.e. a backward translation model). 
		# We probably want to evaluate a batch_of_TARGET values, given the source GMM values.

		# Don't actually get z's this here.. Just use the source latent z's...
		# Just make sure we run this after the set_translated_z_sets function is called..
		# # Actually get Z's. Hopefully this goes over representative proportion of dataset.
		# self.source_manager.get_trajectory_and_latent_sets(get_visuals=False)
		# self.target_manager.get_trajectory_and_latent_sets(get_visuals=False)
		

		# Remember, evaluation_domain is the domain from which the components Gaussians come from.
		manager_list = [self.source_manager, self.target_manager]
		policy_manager = manager_list[evaluation_domain]

		###################################
		# Create GMM
		###################################

		# Earlier we were not computing source latent z set because we were assuming it was already computed, but if preprocessing step once, we can't really assume this..
		if mean_point_set is None:
			policy_manager.get_trajectory_and_latent_sets(get_visuals=False)
			mean_point_set = policy_manager.latent_z_set
			gmm_means = torch.tensor(np.concatenate(mean_point_set)).to(device)
		else:

			# print("Embedding in GMM creation")
			# embed()

			if differentiable_points:
				gmm_means = mean_point_set
			else:
				gmm_means = torch.tensor(mean_point_set).to(device)

		if tuple_GMM:
			gmm_variances = self.args.gmm_tuple_variance_value*torch.ones_like(gmm_means).to(device)
		else:
			gmm_variances = self.args.gmm_variance_value*torch.ones_like(gmm_means).to(device)
		
		# Create a mixture that ignores last dimension.. this should be able to handle both batched and non-batched inputs..
		self.mixture_distribution = torch.distributions.Categorical(torch.ones(gmm_means.shape[:-1]).to(device))
		component_distribution = torch.distributions.Independent(torch.distributions.Normal(gmm_means,gmm_variances),1)
		
		GMM = torch.distributions.MixtureSameFamily(self.mixture_distribution, component_distribution)

		# Can now query this GMM for differentiable probability estimate as: 
		# self.GMM.log_prob(batch_of_values)	

		return GMM

	def cross_domain_decoding(self, domain, domain_manager, latent_z, start_state=None, rollout_length=None):

		# If start state is none, first get start state, else use the argument. 
		if start_state is None: 
			# Feed the first latent_z in to get the start state.
			# Get state from... first z in sequence., because we only need one start per trajectory / batch element.
			start_state = self.get_start_state(domain, latent_z[0])
		
		# Now rollout in target domain.		
		cross_domain_decoding_dict = {}
		cross_domain_decoding_dict['differentiable_trajectory'], cross_domain_decoding_dict['differentiable_action_seq'], \
			cross_domain_decoding_dict['differentiable_state_action_seq'], cross_domain_decoding_dict['subpolicy_inputs'] = \
			self.differentiable_rollout(domain_manager, start_state, latent_z, rollout_length=rollout_length)
		# differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs = self.differentiable_rollout(domain_manager, start_state, latent_z)

		# return differentiable_trajectory, subpolicy_inputs
		return cross_domain_decoding_dict

	def differentiable_rollout(self, policy_manager, trajectory_start, latent_z, rollout_length=None):

		# Now implementing a differentiable_rollout function that takes in a policy manager.

		# Copying over from cycle_transfer differentiable rollout. 
		# This function should provide rollout template, but needs modifications to deal with multiple z's being used.

		# Remember, the differentiable rollout is required because the backtranslation / cycle-consistency loss needs to be propagated through multiple sets of translations. 
		# Therefore it must pass through the decoder network(s), and through the latent_z's. (It doesn't actually pass through the states / actions?).		

		# print("Embed in differentiable rollout")
		# embed()

		subpolicy_inputs = torch.zeros((self.args.batch_size,2*policy_manager.state_dim+policy_manager.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[:,:policy_manager.state_dim] = torch.tensor(trajectory_start).to(device).float()
		# subpolicy_inputs[:,2*policy_manager.state_dim:] = torch.tensor(latent_z).to(device).float()
		subpolicy_inputs[:,2*policy_manager.state_dim:] = latent_z[0]

		if self.args.batch_size>1:
			subpolicy_inputs = subpolicy_inputs.unsqueeze(0)

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = policy_manager.rollout_timesteps-1

		for t in range(length):

			# Get actions from the policy.
			actions = policy_manager.policy_network.reparameterized_get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor
			
			# Compute next state. 
			new_state = subpolicy_inputs[t,...,:policy_manager.state_dim]+action_to_execute

			# Create new input row. 
			input_row = torch.zeros((self.args.batch_size, 2*policy_manager.state_dim+policy_manager.latent_z_dimensionality)).to(device).float()
			input_row[:,:policy_manager.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[:,policy_manager.state_dim:2*policy_manager.state_dim] = actions[-1].squeeze(1)
			input_row[:,2*policy_manager.state_dim:] = latent_z[t+1]

			# Now that we have assembled the new input row, concatenate it along temporal dimension with previous inputs. 
			if self.args.batch_size>1:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row.unsqueeze(0)],dim=0)
			else:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[...,:policy_manager.state_dim].detach().cpu().numpy()
		differentiable_trajectory = subpolicy_inputs[...,:policy_manager.state_dim]
		differentiable_action_seq = subpolicy_inputs[...,policy_manager.state_dim:2*policy_manager.state_dim]
		differentiable_state_action_seq = subpolicy_inputs[...,:2*policy_manager.state_dim]

		# For differentiabiity, return tuple of trajectory, actions, state actions, and subpolicy_inputs. 
		return [differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs]

	def setup_task_supervision(self):

		# First construct task lists. 
		self.source_dataset.task_list = np.zeros(len(self.source_dataset))
		self.target_dataset.task_list = np.zeros(len(self.target_dataset))

		for k, v in enumerate(self.source_dataset):
			self.source_dataset.task_list[k] = v['task_id']
		for k, v in enumerate(self.target_dataset):
			self.target_dataset.task_list[k] = v['task_id']

		# Now check for tasks that have trajectories in both domains. 
		# Do we need to do this.. or should we just do this lazily? 

		####################################################
		# Create index lists of tasks and datapoints here. 
		####################################################
		
		# Well.. we don't want to sample tasks that don't have any trajectories across domains..
		if self.args.source_domain=='MIME' or self.args.target_domain=='MIME':
			self.task_feasibility = np.zeros(20)
			
			# Create lists of datapoints in each domain that address each task. 
			self.source_per_task_datapoints = []
			self.target_per_task_datapoints = []

			# Number of datapoints per task..
			self.source_number_of_task_datapoints = np.zeros(20)
			self.target_number_of_task_datapoints = np.zeros(20)

			for k in range(20):

				valid_source_indices = (self.source_dataset.task_list==k)
				valid_target_indices = (self.target_dataset.task_list==k)
				# num_source_tasks = valid_source_indices.sum()
				# num_target_tasks = valid_target_indices.sum()
				
				self.source_number_of_task_datapoints[k] = valid_source_indices.sum()
				self.target_number_of_task_datapoints[k] = valid_target_indices.sum()

				# Compute feasibility as whether there is at least one trajectory of this task in both domains.
				# self.task_feasibility[k] = (num_source_tasks>0) and (num_target_tasks>0)
				self.task_feasibility[k] = (self.source_number_of_task_datapoints[k]>0) and (self.target_number_of_task_datapoints[k]>0)				

				# If feasible, log datapoints..
				if self.task_feasibility[k]:
					self.source_per_task_datapoints.append(list(np.where(valid_source_indices)[0]))
					self.target_per_task_datapoints.append(list(np.where(valid_target_indices)[0]))
				# Otherwise set to dummy. 
				else:					
					self.source_per_task_datapoints.append([])
					self.target_per_task_datapoints.append([])

			# Task Frequencies..
			self.task_datapoint_counts = self.task_feasibility*np.min([self.source_number_of_task_datapoints,self.target_number_of_task_datapoints],axis=0)

			# Listing out task indices that we know are bimanual....
			self.bimanual_indices = [12,14,15]
			# Create a mask that is true for non bimanual tasks..
			self.nonbimanual_tasks = np.ones(20)
			self.nonbimanual_tasks[self.bimanual_indices] = 0

			self.prefreq = self.task_datapoint_counts*self.nonbimanual_tasks
			self.task_frequencies = self.prefreq/self.prefreq.sum()

			# print(self.source_manager.block_index_list_for_task)
			# print(self.target_manager.block_index_list_for_task)

			# print("Embedding in setup task blah")
			# embed()

	def train(self, model=None):

		# Run some initialization process to manage GPU memory with variable sized batches.
		self.current_epoch_running = 0

		if self.args.setting not in ['downstreamtasktransfer']:
			self.initialize_training_batches()

		# Setup GMM.
		self.setup_GMM()

		# If we're using task based supervision. 
		if self.args.task_based_supervision:

			# Setup task supervision. 
			self.setup_task_supervision()

		if self.args.setting in ['densityjointfixembedtransfer']:
			# Specially for this setting, now run initialize_training_batches again without skipping GMM steps.
			self.initialize_training_batches(skip=False)

		# Now run original training function.
		print("About to run train function.")
		super().train(model=model)

	def evaluate_semantic_accuracy(self):
		pass

class PolicyManager_CycleConsistencyTransfer(PolicyManager_Transfer):

	# Inherit from transfer. 
	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_CycleConsistencyTransfer, self).__init__(args, source_dataset, target_dataset)

		self.neighbor_obj_set = False

	# Don't actually need to define these functions since they perform same steps as super functions.
	def create_networks(self):

		super().create_networks()

		# Must also create two discriminator networks; one for source --> target --> source, one for target --> source --> target. 
		# Remember, since these discriminator networks are operating on the trajectory space, we have to 
		# make them LSTM networks, rather than MLPs. 

		if self.args.real_translated_discriminator:
			# # We have the encoder network class that's perfect for this. Output size is 2. 
			self.source_discriminator = EncoderNetwork(self.source_manager.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size).to(device)
			self.target_discriminator = EncoderNetwork(self.target_manager.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size).to(device)

	def set_iteration(self, counter):
		super().set_iteration(counter)

		if counter<self.args.training_phase_size:
			self.real_translated_loss_weight = 0.
		else:
			self.real_translated_loss_weight = self.args.real_trans_loss_weight

	def create_training_ops(self):

		# Call super training ops. 
		super().create_training_ops()

		# Instead of using the individuals policy manager optimizers, use one single optimizer. 		
		self.parameter_list = self.source_manager.parameter_list + self.target_manager.parameter_list
		# Add discriminator parameters if neede.d
		if self.args.real_translated_discriminator:			
			self.parameter_list += list(self.source_discriminator.parameters()) + list(self.target_discriminator.parameters())
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		# Call super save model. 
		super().save_all_models(suffix)

		if self.args.real_translated_discriminator:
			# Now save the individual source / target discriminators. 
			self.save_object['Source_Discriminator_Network'] = self.source_discriminator.state_dict()
			self.save_object['Target_Discriminator_Network'] = self.target_discriminator.state_dict()

			# Overwrite the save from super. 
			torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):

		# Call super load. 
		super().load_all_models(path)

		if self.args.real_translated_discriminator:
			# Now load the individual source and target discriminators. 
			self.source_discriminator.load_state_dict(self.load_object['Source_Discriminator_Network'])
			self.target_discriminator.load_state_dict(self.load_object['Target_Discriminator_Network'])

	def get_start_state(self, domain, source_latent_z):

		# Function to retrieve the start state for differentiable decoding from target domain. 
		# How we do this is first to retrieve the target domain latent z closest to the source_latent_z. 
		# We then select the trajectory corresponding to this target_domain latent_z.
		# We then copy the start state of this trajectory. 

		if not(self.neighbor_obj_set):
			self.set_neighbor_objects()

		# First get neighbor object and trajectory sets. 
		neighbor_object_list = [self.source_neighbors_object, self.target_neighbors_object]
		trajectory_set_list = [self.source_manager.trajectory_set, self.target_manager.trajectory_set]
		
		# Remember, we need _target_ domain. So use 1-domain instead of domain.
		neighbor_object = neighbor_object_list[1-domain]
		trajectory_set = np.array(trajectory_set_list[1-domain])

		# Next get closest target z. 
		_ , target_latent_z_index = neighbor_object.kneighbors(source_latent_z.squeeze(0).detach().cpu().numpy())

		# Don't actually need the target_latent_z, unless we're doing differentiable nearest neighbor transfer. 					
		if self.args.batch_size>1:
			# Now get the corresponding trajectory. 			

			# print("Embedding in start state comp.")
			# embed()	
			trajectory = trajectory_set[target_latent_z_index[:,0]]
			# Finally, pick up first state. 
			start_state = trajectory[:,0]
		else:
			# Now get the corresponding trajectory. 			
			trajectory = trajectory_set[target_latent_z_index[0,0]]
			# Finally, pick up first state. 
			start_state = trajectory[0]

		return start_state

	def differentiable_rollout(self, policy_manager, trajectory_start, latent_z, rollout_length=None):
		# Now implementing a differentiable_rollout function that takes in a policy manager.

		# Copying over from rollout_robot_trajectory. This function should provide rollout template, but may need modifications for differentiability. 

		# Remember, the differentiable rollout is required because the backtranslation / cycle-consistency loss needs to be propagated through multiple sets of translations. 
		# Therefore it must pass through the decoder network(s), and through the latent_z's. (It doesn't actually pass through the states / actions?).		
		subpolicy_inputs = torch.zeros((self.args.batch_size,2*policy_manager.state_dim+policy_manager.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[:,:policy_manager.state_dim] = torch.tensor(trajectory_start).to(device).float()
		# subpolicy_inputs[:,2*policy_manager.state_dim:] = torch.tensor(latent_z).to(device).float()	
		subpolicy_inputs[:,2*policy_manager.state_dim:] = latent_z

		if self.args.batch_size>1:
			subpolicy_inputs = subpolicy_inputs.unsqueeze(0)

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = policy_manager.rollout_timesteps-1

		for t in range(length):

			# Get actions from the policy.
			actions = policy_manager.policy_network.reparameterized_get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor
			
			# Compute next state. 
			new_state = subpolicy_inputs[t,...,:policy_manager.state_dim]+action_to_execute		

			# Create new input row. 
			input_row = torch.zeros((self.args.batch_size, 2*policy_manager.state_dim+policy_manager.latent_z_dimensionality)).to(device).float()
			input_row[:,:policy_manager.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[:,policy_manager.state_dim:2*policy_manager.state_dim] = actions[-1].squeeze(1)
			input_row[:,2*policy_manager.state_dim:] = latent_z

			# Now that we have assembled the new input row, concatenate it along temporal dimension with previous inputs. 
			if self.args.batch_size>1:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row.unsqueeze(0)],dim=0)
			else:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[...,:policy_manager.state_dim].detach().cpu().numpy()
		differentiable_trajectory = subpolicy_inputs[...,:policy_manager.state_dim]
		differentiable_action_seq = subpolicy_inputs[...,policy_manager.state_dim:2*policy_manager.state_dim]
		differentiable_state_action_seq = subpolicy_inputs[...,:2*policy_manager.state_dim]

		# For differentiabiity, return tuple of trajectory, actions, state actions, and subpolicy_inputs. 
		return [differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs]

	def get_source_target_domain_managers(self):

		domain = np.random.binomial(1,0.5)
		# Also Get domain policy manager. 
		source_policy_manager = self.get_domain_manager(domain) 
		target_policy_manager = self.get_domain_manager(1-domain) 

		return domain, source_policy_manager, target_policy_manager

	def cross_domain_decoding(self, domain, domain_manager, latent_z, start_state=None):

		# If start state is none, first get start state, else use the argument. 
		if start_state is None: 
			start_state = self.get_start_state(domain, latent_z)

		# Now rollout in target domain.		
		cross_domain_decoding_dict = {}
		cross_domain_decoding_dict['differentiable_trajectory'], cross_domain_decoding_dict['differentiable_action_seq'], \
			cross_domain_decoding_dict['differentiable_state_action_seq'], cross_domain_decoding_dict['subpolicy_inputs'] = \
			self.differentiable_rollout(domain_manager, start_state, latent_z)
		# differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs = self.differentiable_rollout(domain_manager, start_state, latent_z)

		# return differentiable_trajectory, subpolicy_inputs
		return cross_domain_decoding_dict

	def update_networks(self, dictionary, source_policy_manager):

		# Here are the objectives we have to be considering. 
		# 	1) Reconstruction of inputs under single domain encoding / decoding. 
		#		In this implementation, we just have to use the source_loglikelihood for this. 
		#	2) Discriminability of Z space. This is taken care of from the compute_discriminator_losses function.
		# 	3) Cycle-consistency. This may be implemented as regression (L2), loglikelihood of cycle-reconstructed traj, or discriminability of trajectories.
		#		In this implementation, we just have to use the cross domain decoded loglikelihood. 

		####################################
		# First update encoder decoder networks. Don't train discriminator.
		####################################

		# Zero gradients.
		self.optimizer.zero_grad()		

		####################################
		# (1) Compute single-domain reconstruction loss.
		####################################
	
		# Compute VAE loss on the current domain as negative log likelihood likelihood plus weighted KL.  
		self.source_likelihood_loss = -dictionary['source_loglikelihood'].mean()
		self.source_encoder_KL = dictionary['source_kl_divergence'].mean()
		# self.source_encoder_KL = 0.
		self.source_reconstruction_loss = self.source_likelihood_loss + self.args.kl_weight*self.source_encoder_KL

		####################################
		# (2) Compute discriminability losses.
		####################################

		#	This block first computes discriminability losses:
		#	# a) First, feeds the latent_z into the z_discriminator, that is being trained to discriminate between z's of source and target domains. 
		#	# 	 Gets and returns the loglikelihood of the discriminator predicting the true domain. 
		#	# 	 Also returns discriminability loss, that is used to train the _encoders_ of both domains. 
		#	#		
		#	# b) ####### DON'T NEED TO DO THIS YET: ####### Also feeds either the cycle reconstructed trajectory, or the original trajectory from the source domain, into a separate discriminator. 
		#	# 	 This second discriminator is specific to the domain we are operating in. This discriminator is discriminating between the reconstructed and original trajectories. 
		#	# 	 Basically standard GAN adversarial training, except the generative model here is the entire cycle-consistency translation model.
		#
		#	In addition to this, must also compute discriminator losses to train discriminators themselves. 
		# 	# a) For the z discriminator (and if we're using trajectory discriminators, those too), clone and detach the inputs of the discriminator and compute a discriminator loss with the right domain used in targets / supervision. 
		#	#	 This discriminator loss is what is used to actually train the discriminators.

		# Get z discriminator logprobabilities.		
		# z_discriminator_logprob, self.z_discriminator_prob = self.discriminator_network(dictionary['source_latent_z'])
		z_discriminator_logprob, self.z_discriminator_prob = self.discriminator_network.get_probabilities(dictionary['source_latent_z'])
		# Compute discriminability loss. Remember, this is not used for training the discriminator, but rather the encoders.

		# domain_label = torch.tensor(1-dictionary['domain']).to(device).long().view(1,)
		domain_label = torch.tensor(dictionary['domain']).to(device).long().view(1,)
		# domain_label = torch.ones((timesteps,self.args.batch_size)).to(device).long()*dictionary['domain']
		domain_label = domain_label.repeat(self.args.batch_size)

		# print("Embedding in Update networks.")
		# embed()

		self.z_discriminability_loss = self.discriminability_loss_weight*self.negative_log_likelihood_loss_function(z_discriminator_logprob.squeeze(0), 1-domain_label).mean()

		###### Block that computes discriminability losses assuming we are using trjaectory discriminators. ######

		if self.args.real_translated_discriminator:

			# Get the right trajectory discriminator network.
			discriminator_list = [self.source_discriminator, self.target_discriminator]
			source_discriminator = discriminator_list[dictionary['domain']]
			
			# First select whether we are feeding original or translated trajectory. 
			real_or_translated = np.random.binomial(1,0.5)
			real_trans_label = torch.tensor(1-real_or_translated).to(device).long().repeat(self.args.batch_size)

			# Based on whether original or translated trajectory, set trajectory object to either the original trajectory or cross domain decoded trajectory. 
			if real_or_translated==0:
				trajectory = dictionary['source_subpolicy_inputs_original'][...,:2*source_policy_manager.state_dim]
			else:
				trajectory = dictionary['source_subpolicy_inputs_crossdomain'][...,:2*source_policy_manager.state_dim]

			# # Now feed trajectory to the trajectory discriminator, based on whether it is the source of target discriminator.
			traj_discriminator_logprob, traj_discriminator_prob = source_discriminator.get_probabilities(trajectory)
			
			# # Compute trajectory discriminability loss, based on whether the trajectory was original or reconstructed.
			# self.traj_discriminability_loss = self.negative_log_likelihood_loss_function(traj_discriminator_logprob.squeeze(1), torch.tensor(1-original_or_reconstructed).to(device).long().view(1,))
			self.traj_discriminability_loss = self.negative_log_likelihood_loss_function(traj_discriminator_logprob.squeeze(0), 1-real_trans_label).mean()
			self.weighted_real_translated_loss = self.real_translated_loss_weight*self.traj_discriminability_loss
		else:
			self.weighted_real_translated_loss = 0.

		####################################
		# (3) Compute cycle-consistency losses.
		####################################

		# Must compute likelihoods of original actions under the cycle reconstructed trajectory states. 
		# I.e. evaluate likelihood of original actions under source_decoder (i.e. source subpolicy), with the subpolicy inputs constructed from cycle-reconstruction.
		
		# Get the original action sequence.
		original_action_sequence = dictionary['source_subpolicy_inputs_original'][...,source_policy_manager.state_dim:2*source_policy_manager.state_dim]

		# Now evaluate likelihood of actions under the source decoder.
		self.cycle_reconstructed_loglikelihood, _ = source_policy_manager.policy_network.forward(dictionary['source_subpolicy_inputs_crossdomain'], original_action_sequence)
		# Reweight the cycle reconstructed likelihood to construct the loss.
		self.cycle_reconstruction_loss = -self.args.cycle_reconstruction_loss_weight*self.cycle_reconstructed_loglikelihood.mean()

		####################################
		# Now that individual losses are computed, compute total loss, compute gradients, and then step.
		####################################

		# First combine losses.
		self.total_VAE_loss = self.source_reconstruction_loss + self.z_discriminability_loss + self.cycle_reconstruction_loss + self.weighted_real_translated_loss 

		# If we are in a encoder / decoder training phase, compute gradients and step.  
		if not(self.skip_vae):
			self.total_VAE_loss.backward()
			self.optimizer.step()

		####################################
		# Now compute discriminator losses and update discriminator network(s).
		####################################

		# First zero out the discriminator gradients. 
		self.discriminator_optimizer.zero_grad()

		# Detach the latent z that is fed to the discriminator, and then compute discriminator loss.
		# If we tried to zero grad the discriminator and then use NLL loss on it again, Pytorch would cry about going backward through a part of the graph that we already \ 
		# went backward through. Instead, just pass things through the discriminator again, but this time detaching latent_z. 
		z_discriminator_detach_logprob, z_discriminator_detach_prob = self.discriminator_network.get_probabilities(dictionary['source_latent_z'].detach())

		# Compute discriminator loss for discriminator. 		
		self.z_discriminator_loss = self.negative_log_likelihood_loss_function(z_discriminator_detach_logprob.squeeze(0), domain_label).mean()
		
		####################################
		if self.args.real_translated_discriminator:
	
			# Feed previously set trajectory object to correct discriminator, after detaching it to prevent opposite gradient flow.
			traj_discriminator_logprob, traj_discriminator_prob = source_discriminator.get_probabilities(trajectory.detach())
			# Now compute loss.
			self.real_translated_discriminator_loss = self.negative_log_likelihood_loss_function(traj_discriminator_logprob.squeeze(0), real_trans_label).mean()
		else:
			self.real_translated_discriminator_loss = 0.

		####################################
		self.total_discriminator_loss = self.z_discriminator_loss + self.real_translated_discriminator_loss

		if not(self.skip_discriminator):
			# Now go backward and take a step.
			self.total_discriminator_loss.backward()
			self.discriminator_optimizer.step()

	def update_plots(self, counter, viz_dict):

		# Relabeling losses so that we can use the update_plots of the parent class.
		self.likelihood_loss = self.source_likelihood_loss
		self.encoder_KL = self.source_encoder_KL
		self.VAE_loss = self.source_reconstruction_loss
				
		self.discriminability_loss = self.z_discriminability_loss
		self.discriminator_loss = self.z_discriminator_loss		
		viz_dict['discriminator_probs'] = self.z_discriminator_prob.mean().detach().cpu().numpy()

		# Using super update_plots instead of implementing from scratch. 
		super().update_plots(counter, viz_dict)
		
		# Original trajectory. 
		original_trajectory = viz_dict['source_subpolicy_inputs_original'][:,:,:self.source_manager.state_dim]
		cycle_reconstructed_trajectory = viz_dict['source_subpolicy_inputs_crossdomain'][:,:,:self.source_manager.state_dim]

		self.original_cycle_reconstructed_trajectory_diffs = copy.deepcopy((((original_trajectory - cycle_reconstructed_trajectory).detach().cpu().numpy())**2).mean())

	def run_iteration(self, counter, i, skip_viz=False):

		# Phases: 
		# Phase 1:  Train encoder-decoder for both domains initially, so that discriminator is not fed garbage. 
		# Phase 2:  Train encoder, decoder for each domain, and Z discriminator concurrently. 
		# Phase 3:  Train encoder, decoder for each domain, and the individual source and target discriminators, concurrently.

		# Algorithm (joint training): 
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use as source (i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain. 
		#		# 3) Transfer Steps: 
		#	 		# a) Encode trajectory as latent z (domain 1). 
		#			# b) Use domain 2 decoder to decode latent z into trajectory (domain 2).
		#			# c) Use domain 2 encoder to encode trajectory into latent z (domain 2).
		#			# d) Use domain 1 decoder to decode latent z (domain 2) into trajectory (domain 1).
		# 		# 4) Feed cycle-reconstructed trajectory and original trajectory (both domain 1) into discriminator. 
		#		# 5) Train discriminators to predict whether original or cycle reconstructed trajectory. 
		#		# 	 Alternate: Remember, don't actually need to use trajectory level discriminator networks, can just use loglikelihood cycle-reconstruction loss. Try this first.
		#		# 	 Train z discriminator to predict which domain the latentz sample came from. 
		# 		# 	 Train encoder / decoder architectures with mix of reconstruction loss and discriminator confusing objective. 
		# 		# 	 Compute and apply gradient updates. 

		# Remember to make domain agnostic function calls to encode, feed into discriminator, get likelihoods, etc. 

		####################################
		# (0) Setup things like training phases, epislon values, etc.
		####################################

		self.set_iteration(counter)
		dictionary = {}
		target_dict = {}

		####################################
		# (1) Select which domain to use as source domain (also supervision of z discriminator for this iteration). 
		####################################

		dictionary['domain'], source_policy_manager, target_policy_manager = self.get_source_target_domain_managers()

		####################################
		# (2) & (3 a) Get source trajectory (segment) and encode into latent z. Decode using source decoder, to get loglikelihood for reconstruction objectve. 
		####################################
		
		dictionary['source_subpolicy_inputs_original'], dictionary['source_latent_z'], dictionary['source_loglikelihood'], dictionary['source_kl_divergence'] = self.encode_decode_trajectory(source_policy_manager, i)

		####################################
		# (3 b) Cross domain decoding. 
		####################################
		
		# target_dict['target_trajectory_rollout'], target_dict['target_subpolicy_inputs'] = self.cross_domain_decoding(domain, target_policy_manager, dictionary['source_latent_z'])
		target_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], target_policy_manager, dictionary['source_latent_z'])

		####################################
		# (3 c) Cross domain encoding of target_trajectory_rollout into target latent_z. 
		####################################

		dictionary['target_subpolicy_inputs'], dictionary['target_latent_z'], dictionary['target_loglikelihood'], dictionary['target_kl_divergence'] = self.encode_decode_trajectory(target_policy_manager, i, trajectory_input=target_cross_domain_decoding_dict)

		####################################
		# (3 d) Cross domain decoding of target_latent_z into source trajectory. 
		# Can use the original start state, or also use the reverse trick for start state. Try both maybe.
		####################################
		
		source_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], source_policy_manager, dictionary['target_latent_z'], start_state=dictionary['source_subpolicy_inputs_original'][0,...,:source_policy_manager.state_dim].detach().cpu().numpy())
		# source_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], source_policy_manager, dictionary['target_latent_z'], start_state=dictionary['source_subpolicy_inputs_original'][0,:source_policy_manager.state_dim].detach().cpu().numpy())
		dictionary['source_subpolicy_inputs_crossdomain'] = source_cross_domain_decoding_dict['subpolicy_inputs']

		####################################
		# (4) Compute all losses, reweight, and take gradient steps.
		####################################

		# Update networks.
		self.update_networks(dictionary, source_policy_manager)

		####################################
		# (5) Accumulate and plot statistics of training.
		####################################
		
		if not(skip_viz):
			self.update_plots(counter, dictionary)

		# Encode decode function: First encodes, takes trajectory segment, and outputs latent z. The latent z is then provided to decoder (along with initial state), and then we get SOURCE domain subpolicy inputs. 
		# Cross domain decoding function: Takes encoded latent z (and start state), and then rolls out with target decoder. Function returns, target trajectory, action sequence, and TARGET domain subpolicy inputs. 

class PolicyManager_FixEmbedCycleTransfer(PolicyManager_CycleConsistencyTransfer):
	
	# Inherit from cycle con transfer. 
	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_FixEmbedCycleTransfer, self).__init__(args, source_dataset, target_dataset)

		# if self.args.source_model is not None:
		# 	self.source_manager.load_all_models(self.args.source_model)
		# if self.args.target_model is not None:
		# 	self.target_manager.load_all_models(self.args.target_model)

		self.translation_model_layers = 2
		self.args.real_translated_discriminator = 1

	def setup(self):
		super().setup()

		self.source_manager.rollout_timesteps = 5
		self.target_manager.rollout_timesteps = 5

	def set_iteration(self, counter):

		# Based on what phase of training we are in, set discriminability loss weight, etc. 
		
		# Phase 1 of training: Don't train discriminator at all, set discriminability loss weight to 0.
		if counter<self.args.training_phase_size:
			# self.discriminability_loss_weight = 0.
			# self.real_translated_loss_weight = 0.

			self.discriminability_loss_weight = self.args.discriminability_weight
			self.real_translated_loss_weight = self.args.real_trans_loss_weight

			self.vae_loss_weight = 1.
			self.training_phase = 1
			self.skip_vae = False
			self.skip_discriminator = True			

		# Phase 2 of training: Train the discriminator, and set discriminability loss weight to original.
		else:
			self.discriminability_loss_weight = self.args.discriminability_weight
			self.real_translated_loss_weight = self.args.real_trans_loss_weight

			self.vae_loss_weight = self.args.vae_loss_weight

			# Now make discriminator and vae train in alternating fashion. 
			# Set number of iterations of alteration. 
			# self.alternating_phase_size = self.args.alternating_phase_size*self.extent

			# # If odd epoch, train discriminator. (Just so that we start training discriminator first).
			# if (counter/self.alternating_phase_size)%2==1:			
			# 	self.skip_discriminator = False
			# 	self.skip_vae = True
			# # Otherwise train VAE.
			# else:
			# 	self.skip_discriminator = True
			# 	self.skip_vae = False		

			# Train discriminator for k times as many steps as VAE. Set args.alternating_phase_size as 1 for this. 
			if (counter/self.args.alternating_phase_size)%(self.args.discriminator_phase_size+1)>=1:
				print("Training Discriminator.")
				self.skip_discriminator = False
				self.skip_vae = True
			# Otherwise train VAE.
			else:
				print("Training VAE.")
				self.skip_discriminator = True
				self.skip_vae = False		

			self.training_phase = 2

		self.source_manager.set_epoch(counter)
		self.target_manager.set_epoch(counter)

	def create_networks(self):
		
		 # Call super create networks, to create all the necessary networks. 
		super().create_networks()

		# In addition, now create translation model networks.
		self.forward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)
		self.backward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)

		# Create list of translation models to select from based on source domain.
		self.translation_model_list = [self.forward_translation_model, self.backward_translation_model]

		# Instead of single z discriminator, require two different z discriminators. 
		# del self.discriminator_network

		# Now create individual z discriminators.
		self.source_z_discriminator = DiscreteMLP(self.args.z_dimensions, self.hidden_size, 2, args=self.args).to(device)
		self.target_z_discriminator = DiscreteMLP(self.args.z_dimensions, self.hidden_size, 2, args=self.args).to(device)
		
		# Create lists of discriminators. 
		self.discriminator_list = [self.source_discriminator, self.target_discriminator]
		self.z_discriminator_list = [self.source_z_discriminator, self.target_z_discriminator]
		
	def create_training_ops(self):

		# Don't actually call super().create_training_ops(),
		# Because this creates optimizers with source and target encoder decoder parameters in the optimizer. 

		# Instead, create other things here. 
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')

		# Set regular parameter list. 
		self.parameter_list = list(self.forward_translation_model.parameters()) + list(self.backward_translation_model.parameters())
		# Now create optimizer for translation models. 
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate)

		# Set discriminator parameter list. 
		self.discrimator_parameter_list = list(self.source_z_discriminator.parameters()) + list(self.target_z_discriminator.parameters()) + list(self.source_discriminator.parameters()) + list(self.target_discriminator.parameters())
		# Create common optimizer for source, target, and discriminator networks. 
		self.discriminator_optimizer = torch.optim.Adam(self.discrimator_parameter_list, lr=self.learning_rate)

	def save_all_models(self, suffix):

		# Call super save model. 
		super().save_all_models(suffix)

		# del self.save_object['Discriminator_Network']

		self.save_object['forward_translation_model'] = self.forward_translation_model.state_dict()
		self.save_object['backward_translation_model'] = self.backward_translation_model.state_dict()
		self.save_object['source_z_discriminator'] = self.source_z_discriminator.state_dict()
		self.save_object['target_z_discriminator'] = self.target_z_discriminator.state_dict()

		# Overwrite the save from super. 
		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):

		# Call super load. 
		super().load_all_models(path)

		# Load translation model.
		self.forward_translation_model.load_state_dict(self.load_object['forward_translation_model'])
		self.backward_translation_model.load_state_dict(self.load_object['backward_translation_model'])
		self.source_z_discriminator.load_state_dict(self.load_object['source_z_discriminator'])
		self.target_z_discriminator.load_state_dict(self.load_object['target_z_discriminator'])

	def backtranslate_latent_z(self, domain, latent_z):
		
		# Get translation models.
		forward_translation_model = self.translation_model_list[domain]
		backward_translation_model = self.translation_model_list[1-domain]

		backtranslation_dict = {}

		# Translate Z. 
		translated_latent_z = forward_translation_model.forward(latent_z)
		backtranslated_latent_z = backward_translation_model.forward(translated_latent_z)

		return translated_latent_z, backtranslated_latent_z

	def update_networks(self, dictionary, source_policy_manager):

		# Remember, we don't actually need reconstruction losses now, we just need discriminator and discriminability losses. 

		########################################################################
		# (0) Zero out gradients. 
		########################################################################

		self.optimizer.zero_grad()

		####################################
		# (1) Compute discriminability losses to train translation models.
		####################################

		# Get the correct trajectory discriminator networks.
		source_discriminator = self.discriminator_list[dictionary['domain']]
		source_z_discriminator = self.z_discriminator_list[dictionary['domain']]
		
		# First select whether we are feeding original or translated trajectory. 
		real_or_translated = np.random.binomial(1,0.5)
		# real_trans_label = torch.tensor(1-real_or_translated).to(device).long().repeat(self.args.batch_size)
		real_trans_label = torch.tensor(real_or_translated).to(device).long().repeat(self.args.batch_size)

		# Based on whether original or translated trajectory, set trajectory object to either the original trajectory or cross domain decoded trajectory.
		if real_or_translated==0:
			# If real, set the trajectory and the latent_z as original ones.
			trajectory = dictionary['source_subpolicy_inputs_original'][...,:2*source_policy_manager.state_dim]
			latent_z = dictionary['source_latent_z']
		else:
			# If translated, set the trajectory and latent_z as backtranslated ones. 
			trajectory = dictionary['source_subpolicy_inputs_crossdomain'][...,:2*source_policy_manager.state_dim]
			latent_z = dictionary['backtranslated_latent_z']
		
		
		####################################
		# (1 a) Compute traj_discriminability_loss. 
		####################################

		# Now feed trajectory to the trajectory discriminator, based on whether it is the source of target discriminator.
		self.traj_discriminator_logprob, traj_discriminator_prob = source_discriminator.get_probabilities(trajectory)
		# Logging probability. 
		# self.traj_discriminator_prob = traj_discriminator_prob[...,1-real_or_translated]
		self.traj_discriminator_prob = traj_discriminator_prob[...,real_or_translated]
		
		# Compute trajectory discriminability loss, based on whether the trajectory was original or reconstructed.
		self.traj_discriminability_loss = self.negative_log_likelihood_loss_function(self.traj_discriminator_logprob.squeeze(0), 1-real_trans_label).mean()
		self.weighted_real_translated_loss = self.real_translated_loss_weight*self.traj_discriminability_loss

		####################################
		# (1 b) Compute z_discriminability_loss.
		####################################

		# Now feed latent_z into the z_discriminator. 
		self.z_discriminator_logprob, z_discriminator_prob = source_z_discriminator.forward(latent_z)
		# Logging probability. 
		# self.z_discriminator_prob = z_discriminator_prob[...,1-real_or_translated]
		self.z_discriminator_prob = z_discriminator_prob[...,real_or_translated]

		# Compute z discriminability loss, based on whether the latent_z was original or translated. 
		self.unweighted_z_discriminability_loss = self.negative_log_likelihood_loss_function(self.z_discriminator_logprob.squeeze(0), 1-real_trans_label).mean()
		self.z_discriminability_loss = self.discriminability_loss_weight*self.unweighted_z_discriminability_loss
		####################################
		# (2) Compute cycle-consistency losses.
		####################################

		# Must compute likelihoods of original actions under the cycle reconstructed trajectory states. 
		# I.e. evaluate likelihood of original actions under source_decoder (i.e. source subpolicy), with the subpolicy inputs constructed from cycle-reconstruction.

		# Get the original action sequence.
		original_action_sequence = dictionary['source_subpolicy_inputs_original'][...,source_policy_manager.state_dim:2*source_policy_manager.state_dim]

		# Now evaluate likelihood of actions under the source decoder.
		self.cycle_reconstructed_loglikelihood, _ = source_policy_manager.policy_network.forward(dictionary['source_subpolicy_inputs_crossdomain'], original_action_sequence)
		# Reweight the cycle reconstructed likelihood to construct the loss.
		self.cycle_reconstruction_loss = -self.args.cycle_reconstruction_loss_weight*self.cycle_reconstructed_loglikelihood.mean()


		####################################
		# (3 a) Compute total loss. 
		####################################

		# First combine losses.
		self.total_VAE_loss = self.z_discriminability_loss + self.weighted_real_translated_loss + self.cycle_reconstruction_loss

		####################################
		# (3 a) Compute gradients and step to update translation models.
		####################################
		if not(self.skip_vae):
			self.total_VAE_loss.backward()
			self.optimizer.step()

		########################################################################
		# (4) Now compute discriminator losses and update discriminator network(s).
		########################################################################

		# First zero out the discriminator gradients. 
		self.discriminator_optimizer.zero_grad()

		####################################
		# (4 a) Feed previously set trajectory object to correct discriminator, after detaching it to prevent opposite gradient flow.
		####################################
		traj_discriminator_logprob, traj_discriminator_prob = source_discriminator.get_probabilities(trajectory.detach())
		# Now compute loss.
		self.real_translated_discriminator_loss = self.negative_log_likelihood_loss_function(traj_discriminator_logprob.squeeze(0), real_trans_label).mean()

		####################################
		# (4 b) Feed previously set latent_z object to correct z_discriminator, after detaching it to prevent opposite gradient flow.
		####################################
		z_discriminator_logprob, z_discriminator_prob = source_z_discriminator.forward(latent_z.detach())
		# Now compute loss.
		self.z_discriminator_loss = self.negative_log_likelihood_loss_function(z_discriminator_logprob.squeeze(0), real_trans_label).mean()

		####################################
		# (4 c) Compute total discriminator loss.
		####################################		
		self.total_discriminator_loss = self.z_discriminator_loss + self.real_translated_discriminator_loss

		if not(self.skip_discriminator):
			# Now go backward and take a step.
			self.total_discriminator_loss.backward()
			self.discriminator_optimizer.step()

	def set_translated_z_sets(self):
		# First copy sets so we don't accidentally perform in-place operations on any of the computed sets.
		self.original_source_latent_z_set = copy.deepcopy(self.source_latent_zs)
		self.original_target_latent_z_set = copy.deepcopy(self.target_latent_zs)

		############################################################
		# First use original source latent set, and translated target latent set. 		
		############################################################

		# First translate the target z's. 
		self.target_latent_zs = self.backward_translation_model.forward(torch.tensor(self.original_target_latent_z_set).to(device).float()).detach().cpu().numpy()		
		self.shared_latent_zs = np.concatenate([self.source_latent_zs,self.target_latent_zs],axis=0)

		self.translated_viz_dictionary = {}
		# Get embeddings of source, and backward translated target latent_zs. 	
		_ , _ , self.translated_viz_dictionary['tsne_origsource_transtarget_p5'], self.translated_viz_dictionary['tsne_origsource_transtarget_p10'], self.translated_viz_dictionary['tsne_origsource_transtarget_p30'], \
			self.translated_viz_dictionary['tsne_origsource_transtarget_traj_p5'], self.translated_viz_dictionary['tsne_origsource_transtarget_traj_p10'], self.translated_viz_dictionary['tsne_origsource_transtarget_traj_p30'] = \
				self.get_embeddings(projection='tsne', computed_sets=True)

		############################################################
		# Now use original target latent set, and translated source latent set. 
		############################################################

		# Now reset the target latent zs, and translate the source latent zs.
		self.target_latent_zs = copy.deepcopy(self.original_target_latent_z_set)
		self.source_latent_zs = self.forward_translation_model.forward(torch.tensor(self.original_source_latent_z_set).to(device).float()).detach().cpu().numpy()
		self.shared_latent_zs = np.concatenate([self.source_latent_zs,self.target_latent_zs],axis=0)

		# Get embeddings of forward translated source, and original target latent_zs. 	
		_ , _ , self.translated_viz_dictionary['tsne_transsource_origtarget_p5'], self.translated_viz_dictionary['tsne_transsource_origtarget_p10'], self.translated_viz_dictionary['tsne_transsource_origtarget_p30'], \
			self.translated_viz_dictionary['tsne_transsource_origtarget_traj_p5'], self.translated_viz_dictionary['tsne_transsource_origtarget_traj_p10'], self.translated_viz_dictionary['tsne_transsource_origtarget_traj_p30'] = \
				self.get_embeddings(projection='tsne', computed_sets=True)

	def update_plots(self, counter, viz_dict):

		# Set reconstruction losses to 0, just to recylce plotting function.
		self.source_likelihood_loss = 0.
		self.source_encoder_KL = 0.
		self.source_reconstruction_loss = 0.
		
		super().update_plots(counter, viz_dict)

		############################################################
		# Now implement visualization of original latent set and translated z space in both directions. 
		############################################################	

		if counter%self.args.display_freq==0:
			self.set_translated_z_sets()

	def run_iteration(self, counter, i, skip_viz=False):

		####################################
		# (0) Setup things like training phases, epsilon values, etc.
		####################################

		self.set_iteration(counter)
		dictionary = {}
		target_dict = {}

		####################################
		# (1) Select which domain to use as source domain (also supervision of z discriminator for this iteration). 
		####################################

		dictionary['domain'], source_policy_manager, target_policy_manager = self.get_source_target_domain_managers()

		####################################
		# (2) & (3 a) Get source trajectory (segment) and encode into latent z. Decode using source decoder, to get loglikelihood for reconstruction objectve. 
		####################################
		
		dictionary['source_subpolicy_inputs_original'], dictionary['source_latent_z'], dictionary['source_loglikelihood'], dictionary['source_kl_divergence'] = self.encode_decode_trajectory(source_policy_manager, i)

		####################################
		# (3 b) Back translate the latent z.
		####################################

		dictionary['translated_latent_z'], dictionary['backtranslated_latent_z'] = self.backtranslate_latent_z(dictionary['domain'], dictionary['source_latent_z'])

		####################################
		# (3 c) Decode backtranslated latent_z into source trajectory. 
		####################################
		
		source_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], source_policy_manager, dictionary['backtranslated_latent_z'], \
			start_state=dictionary['source_subpolicy_inputs_original'][0,...,:source_policy_manager.state_dim].detach().cpu().numpy())
		dictionary['source_subpolicy_inputs_crossdomain'] = source_cross_domain_decoding_dict['subpolicy_inputs']

		####################################
		# (4) Compute all losses, reweight, and take gradient steps.
		####################################

		# Update networks.
		self.update_networks(dictionary, source_policy_manager)

		####################################
		# (5) Accumulate and plot statistics of training.
		####################################
		
		if not(skip_viz):
			self.update_plots(counter, dictionary)

class PolicyManager_JointFixEmbedTransfer(PolicyManager_Transfer):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_JointFixEmbedTransfer, self).__init__(args, source_dataset, target_dataset)

		# Before we initialize anything, set: 
		self.args.fix_source = 1
		self.args.fix_target = 1
		self.set_trans_z = 0
		self.global_traj_counter = 0

		# Set source noise to very close to 0...
		# self.source_args.epsilon_from = 0.02
		# self.source_args.epsilon_to = 0.01

		# Now create two instances of policy managers for each domain. Call them source and target domain policy managers. 
		self.source_manager = PolicyManager_BatchJoint(number_policies=4, dataset=self.source_dataset, args=self.source_args)
		self.target_manager = PolicyManager_BatchJoint(number_policies=4, dataset=self.target_dataset, args=self.target_args)

		self.source_dataset_size = len(self.source_manager.dataset) - self.source_manager.test_set_size
		self.target_dataset_size = len(self.target_manager.dataset) - self.target_manager.test_set_size

		# Now create variables that we need. 
		self.number_epochs = self.args.epochs
		self.extent = min(self.source_dataset_size, self.target_dataset_size)

		# Now setup networks for these PolicyManagers. 		
		self.source_manager.setup()
		self.source_manager.initialize_training_batches()
		self.target_manager.setup()
		self.target_manager.initialize_training_batches()	

		self.translation_model_layers = 4
		self.args.real_translated_discriminator = 0
		self.decay_counter = self.decay_epochs*self.extent
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)

		self.translated_z_epsilon = 0.01

	def set_iteration(self, counter, i=0):
		# First call the set iteration of super. 
		super().set_iteration(counter, i)

		# Now make sure VAE loss weight is set to 0, because we can ignore the reconstruction losses in this setting.
		self.vae_loss_weight = 0.

	def set_translated_z_sets_recurrent_translation(self, domain=1):

		# For the recurrent model setting, special translation...
		# self.target_latent_zs = self.backward_translation_model.forward(torch.tensor(self.original_target_latent_z_set, epsilon=0.0001, precomputed_b=).to(device).float()).detach().cpu().numpy()

		self.translated_z_seq_set = []
		self.source_z_seq_set = []

		# For how many every batches there are in the latent_b_set, 
		for k in range(len(self.target_manager.latent_b_set)):
			
			# Feed the z sequence to the translation model.
			normed_z_input = (self.target_manager.full_latent_z_trajectory[k]-self.target_z_mean)/self.target_z_std
			translated_z_seq = self.backward_translation_model.forward(normed_z_input, epsilon=0.0001, precomputed_b=self.target_manager.latent_b_set[k])

			# Now unbatch and add element to set.
			for b in range(self.args.batch_size):				

				# Parse elements from target set. 
				distinct_z_indices = torch.where(self.target_manager.latent_b_set[k][:,b])[0].clone().detach().cpu().numpy()
				distinct_zs = translated_z_seq[distinct_z_indices, b].clone().detach().cpu().numpy()
				self.translated_z_seq_set.append(distinct_zs)
				
				# Also parse elements from source set...
				distinct_source_zs = self.source_manager.full_latent_z_trajectory[k][distinct_z_indices,b].clone().detach().cpu().numpy()
				self.source_z_seq_set.append(distinct_source_zs)

	def set_translated_z_sets(self, domain=1):
	
		self.viz_dictionary = {}

		# No need to copy these sets over, they have been set in set_z_objects. Not changing them here ensures we're not messing around.
		self.source_latent_zs = copy.deepcopy(self.original_source_latent_z_set)
		self.target_latent_zs = copy.deepcopy(self.original_target_latent_z_set)

		if domain==1:
			############################################################
			# Use original source latent set, and translated target latent set. 		
			############################################################
			
			# First translate the target z's. 
			if self.args.recurrent_translation:
				# self.target_latent_zs = self.backward_translation_model.forward(torch.tensor(self.original_target_latent_z_set, epsilon=0.0001, precomputed_b=).to(device).float()).detach().cpu().numpy()
				with torch.no_grad():
					self.set_translated_z_sets_recurrent_translation()
					self.source_latent_zs = np.concatenate(self.source_z_seq_set)
					self.target_latent_zs = np.concatenate(self.translated_z_seq_set)
			else:						
				self.target_latent_zs = self.backward_translation_model.forward(torch.tensor(self.original_target_latent_z_set).to(device).float()).detach().cpu().numpy()

			self.shared_latent_zs = np.concatenate([self.source_latent_zs,self.target_latent_zs],axis=0)

			# Construct translated_target_z_trajectory_set! 
			# make z tuple objects set to the translated ones! 
			self.construct_translated_target_z_trajectory_set()
			self.construct_tuple_embeddings(translated_target=True)

			# Get embeddings of source, and backward translated target latent_zs. 			
			_ , self.viz_dictionary['tsne_transtarget_p30'], self.viz_dictionary['tsne_origsource_transtarget_p05'], self.viz_dictionary['tsne_origsource_transtarget_p10'], self.viz_dictionary['tsne_origsource_transtarget_p30'], \
				self.viz_dictionary['tsne_origsource_transtarget_traj_p05'], self.viz_dictionary['tsne_origsource_transtarget_traj_p10'], self.viz_dictionary['tsne_origsource_transtarget_traj_p30'] = \
					self.get_embeddings(projection='tsne', computed_sets=True)
			# Also set target traj image

			# also run for PCA! 
			_, _, self.viz_dictionary['pca_combined_origsource_transtarget_embeddings'], _ = self.get_embeddings(projection='pca', computed_sets=True)

			# Also do with DENSNE
			# Get embeddings of source, and backward translated target latent_zs. 			
			_ , self.viz_dictionary['densne_transtarget_p30'], self.viz_dictionary['densne_origsource_transtarget_p05'], self.viz_dictionary['densne_origsource_transtarget_p10'], self.viz_dictionary['densne_origsource_transtarget_p30'], \
				self.viz_dictionary['densne_origsource_transtarget_traj_p05'], self.viz_dictionary['densne_origsource_transtarget_traj_p10'], self.viz_dictionary['densne_origsource_transtarget_traj_p30'] = \
					self.get_embeddings(projection='densne', computed_sets=True)

			if self.check_toy_dataset():
				self.viz_dictionary['tsne_transtarget_traj_p30'] = self.target_traj_image
		else:
			
			############################################################
			# Now use original target latent set, and translated source latent set. 
			############################################################
			
			# First translate the target z's. 
			# ASSUME WE AREN'T USING RECURRENT TRANSLATION
			if self.args.recurrent_translation:				
				with torch.no_grad():
					self.set_translated_z_sets_recurrent_translation(domain=domain)
					self.source_latent_zs = np.concatenate(self.translated_z_seq_set)
					self.target_latent_zs = np.concatenate(self.target_z_seq_set)
			else:						
				self.source_latent_zs = self.forward_translation_model.forward(torch.tensor(self.original_source_latent_z_set).to(device).float()).detach().cpu().numpy()

			self.shared_latent_zs = np.concatenate([self.source_latent_zs,self.target_latent_zs],axis=0)

			# Construct translated_target_z_trajectory_set! 
			# make z tuple objects set to the translated ones! 
			self.construct_translated_target_z_trajectory_set()
			self.construct_tuple_embeddings(translated_target=True)

			# # Get embeddings of forward translated source, and original target latent_zs. 	
			# ############################################################
			self.viz_dictionary['tsne_transsource_p30'] , _ , self.viz_dictionary['tsne_transsource_origtarget_p05'], self.viz_dictionary['tsne_transsource_origtarget_p10'], self.viz_dictionary['tsne_transsource_origtarget_p30'], \
				self.viz_dictionary['tsne_transsource_origtarget_traj_p05'], self.viz_dictionary['tsne_transsource_origtarget_traj_p10'], self.viz_dictionary['tsne_transsource_origtarget_traj_p30'] = \
					self.get_embeddings(projection='tsne', computed_sets=True)

			# also run for PCA! 
			_, _, self.viz_dictionary['pca_combined_transsource_origtarget_embeddings'], _ = self.get_embeddings(projection='pca', computed_sets=True)

			# Also do for DENSNE
			self.viz_dictionary['densne_transsource_p30'] , _ , self.viz_dictionary['densne_transsource_origtarget_p05'], self.viz_dictionary['densne_transsource_origtarget_p10'], self.viz_dictionary['densne_transsource_origtarget_p30'], \
				self.viz_dictionary['densne_transsource_origtarget_traj_p05'], self.viz_dictionary['densne_transsource_origtarget_traj_p10'], self.viz_dictionary['densne_transsource_origtarget_traj_p30'] = \
					self.get_embeddings(projection='densne', computed_sets=True)

			# Also set target traj image
			if self.check_toy_dataset():
				self.viz_dictionary['tsne_transsource_traj_p30'] = self.source_traj_image
				# self.viz_dictionary['densne_transsource_traj_p30'] = self.source_traj_image
	

		self.z_last_set_by = 'set_translated_z_sets'
		self.set_trans_z +=1

	def construct_translated_target_z_trajectory_set(self):

		# Use same indexing as what the visualize_embedded_z_trajectories uses to go
		# from elements in self.target_z_trajectory_set to the right elements in ... not self.shared_embedded_zs, but .. self.target_latent_zs / self.shared_latent_zs! 
		# THis is because we need to actually embed these differently. 

		# Copying over some of visualize_embedded_z_trajectories...
		# Add this value to indexing shared_z_embedding, assuming we're actually providing a shared embedding. 		
		# If source, viz_domain = 0, so don't add anything, but if target, add the length of the soruce_latent_z to skip these.

		domain = 1
		add_value = len(self.source_latent_zs)*domain

		self.translated_target_z_trajectory_set = []

		for i, z_traj in enumerate(self.target_z_trajectory_set):
		
			# First get length of this z_trajectory.
			z_traj_len = len(z_traj)

			# Should just be able to get the corresponding embedded z by manipulating indices.. 
			# Assuming len of z_traj is consistent across all elements in z_trajectory_set_object, which would have needed to have been true 
			# for the concatenate in set_z_objects to work.

			# Translated z traj
			# If we've passed the valid number of z's.. just skip.
			if add_value+i*z_traj_len>len(self.shared_latent_zs):
				pass
			else:
				translated_z_traj = self.shared_latent_zs[add_value+i*z_traj_len:min(add_value+(i+1)*z_traj_len,len(self.shared_latent_zs))]

				self.translated_target_z_trajectory_set.append(translated_z_traj)	

	def update_plots(self, counter, viz_dict, log=False):

		# Call super update plots for the majority of the work. Call this with log==false to make sure that wandb only logs things we add in this function. 
		log_dict = super().update_plots(counter, viz_dict, log=False)

		# # Also log identity loss..
		# log_dict['Unweighted Identity Translation Loss'] = self.unweighted_identity_translation_loss 
		# log_dict['Identity Translation Loss'] = self.identity_translation_loss
		if self.args.gradient_penalty:
			log_dict['Unweighted Wasserstein Gradient Penalty'] = self.unweighted_wasserstein_gradient_penalty
			log_dict['Wasserstein Gradient Penalty'] = self.wasserstein_gradient_penalty

		############################################################
		# Now implement visualization of original latent set and translated z space in both directions. 
		############################################################	
		
		if counter%self.args.display_freq==0:
			
			##################################################
			# Visualize Translated Z Trajectories.
			##################################################

			self.set_translated_z_sets()

			log_dict['Source Z Trajectory Joint TargetTranslated TSNE Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_tsne_image)
			log_dict['Target Z Trajectory Joint TargetTranslated TSNE Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_tsne_image)
			# log_dict['Source Z Trajectory JointTargetTranslated PCA Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_pca_image)
			log_dict['Target Z Trajectory JointTargetTranslated PCA Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_pca_image)

			log_dict['Source Z Trajectory Joint TargetTranslated DENSNE Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_densne_image)
			log_dict['Target Z Trajectory Joint TargetTranslated DENSNE Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_densne_image)

			##################################################
			# Now log combined source and translated target visualizations, and if we want, target and translated source.
			##################################################

			# log_dict["TSNE Translated Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transtarget_p30'])	

			log_dict["TSNE Combined Source and Translated Target Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_p05'])
			log_dict["TSNE Combined Source and Translated Target Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_p10'])
			log_dict["TSNE Combined Source and Translated Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_p30'])
			# log_dict["TSNE Combined Translated Source and Target Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_p05'])
			# log_dict["TSNE Combined Translated Source and Target Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_p10'])
			# log_dict["TSNE Combined Translated Source and Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_p30'])

			## Now for DENSNE
			# log_dict["DENSNE Translated Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['densne_transtarget_p30'])	
			log_dict["DENSNE Combined Source and Translated Target Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['densne_origsource_transtarget_p05'])
			log_dict["DENSNE Combined Source and Translated Target Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['densne_origsource_transtarget_p10'])
			log_dict["DENSNE Combined Source and Translated Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['densne_origsource_transtarget_p30'])
			# log_dict["DENSNE Combined Translated Source and Target Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['densne_transsource_origtarget_p05'])
			# log_dict["DENSNE Combined Translated Source and Target Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['densne_transsource_origtarget_p10'])
			# log_dict["DENSNE Combined Translated Source and Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['densne_transsource_origtarget_p30'])


			##################################################
			# Now for PCA
			##################################################

			log_dict["PCA Combined Source and Translated Target Embeddings"] = self.return_wandb_image(self.viz_dictionary['pca_combined_origsource_transtarget_embeddings'])
			# log_dict["PCA Combined Source and Translated Target Embeddings"] = self.return_wandb_image(self.viz_dictionary['pca_combined_transsource_origtarget_embeddings'])

			# if self.check_toy_dataset():					
			# 	log_dict["TSNE Translated Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transtarget_traj_p30'])
			# 	log_dict["TSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_traj_p05'])
			# 	log_dict["TSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_traj_p10'])
			# 	log_dict["TSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_traj_p30'])
			# 	# log_dict["TSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_traj_p05'])
			# 	# log_dict["TSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_traj_p10'])
			# 	# log_dict["TSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_traj_p30'])

			# 	# Now for DENSNE
			# 	log_dict["DENSNE Translated Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['densne_transtarget_traj_p30'])
			# 	log_dict["DENSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['densne_origsource_transtarget_traj_p05'])
			# 	log_dict["DENSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['densne_origsource_transtarget_traj_p10'])
			# 	log_dict["DENSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['densne_origsource_transtarget_traj_p30'])
			# 	# log_dict["DENSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['densne_transsource_origtarget_traj_p05'])
			# 	# log_dict["DENSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['densne_transsource_origtarget_traj_p10'])
			# 	# log_dict["DENSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['densne_transsource_origtarget_traj_p30'])

			##################################################
			# Now visualize TUPLES of Z's.
			##################################################

			# Remember, whatever is plotting z trajectories can also plot z tuple embeddings.
			log_dict['Joint Target Translated Z Tuple TSNE Embeddings'] = self.return_wandb_image(copy.deepcopy(self.z_tuple_embedding_image))

			###################################################
			# Compute Aggregate CDSL
			###################################################	

			# if self.args.eval_transfer_metrics:
			if self.args.eval_transfer_metrics and counter%self.args.metric_eval_freq==0 and self.check_same_domains():
				self.compute_aggregate_supervised_loss()
			
				# IGNORING AGGREGATE CDSL FOR NOW.
				# self.aggregate_cdsl_value = 0.
				# Now log the aggergate CDSL
				log_dict['Aggregated Supervised Z Error'] = self.aggregate_cdsl_value

		#####################################################
		# Now also call log density and chamfer metrics, now that the translated sets are set..
		#####################################################

		# Need to recreate the reverse GMM here. 
		self.GMM_list[1] = self.create_GMM(evaluation_domain=1, mean_point_set=self.target_latent_zs)

		log_dict = self.log_density_and_chamfer_metrics(counter, log_dict, viz_dict=viz_dict)
		
		if log:
			wandb.log(log_dict, step=counter)
		else:
			return log_dict

	def create_networks(self):
		
		 # Call super create networks, to create all the necessary networks. 
		super().create_networks()

		# In addition, now create translation model networks.

		if self.args.recurrent_translation:
			# self.forward_translation_model = ContinuousVariationalPolicyNetwork_Batch(self.args.z_dimensions, self.args.var_hidden_size, self.args.z_dimensions, self.args, number_layers=self.args.var_hidden_size, translation_network=True).to(device)
			# Create recurrent translation model from variational model template.
			self.backward_translation_model = ContinuousVariationalPolicyNetwork_Batch(self.args.z_dimensions, self.args.var_hidden_size, self.args.z_dimensions, self.args, number_layers=self.args.var_hidden_size, translation_network=True).to(device)
		else:
			# self.forward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)
			self.backward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)

		# Create list of translation models to select from based on source domain.
		# self.translation_model_list = [self.forward_translation_model, self.backward_translation_model]
		
		# Create fake list now, that references the backward translation model... since this should only be used when domain = 1
		self.translation_model_list = [None, self.backward_translation_model]

		# Instead of single z discriminator, require two different z discriminators. 
		# Just use the self.discriminator_network for now.
		# self.source_z_discriminator = DiscreteMLP(self.args.z_dimensions, self.hidden_size, 2, args=self.args).to(device)
		# self.target_z_discriminator = DiscreteMLP(self.args.z_dimensions, self.hidden_size, 2, args=self.args).to(device)
		
		# Create lists of discriminators. 
		# self.discrimin	ator_list = [self.source_discriminator, self.target_discriminator]
		# self.z_discriminator_list = [self.source_z_discriminator, self.target_z_discriminator]

		if self.args.z_transform_discriminator:
			# self.source_z_transform_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)
			self.z_trajectory_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)
			# self.target_z_transform_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)

			# self.z_transform_discriminator_list = [self.source_z_transform_discriminator, self.target_z_transform_discriminator]
		elif self.args.z_trajectory_discriminator:
			
			self.z_trajectory_discriminator = EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size, args=self.args).to(device)

		# If we're using task based discriminability
		if self.args.task_discriminability:
			self.task_discriminators = [EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size, args=self.args).to(device) for k in range(self.args.number_of_tasks)]
			
	def create_training_ops(self):

		# Don't actually call super().create_training_ops(),
		# Because this creates optimizers with source and target encoder decoder parameters in the optimizer. 

		# Instead, create other things here. 
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')

		# Set regular parameter list. 
		# self.parameter_list = list(self.forward_translation_model.parameters()) + list(self.backward_translation_model.parameters())
		self.parameter_list = list(self.backward_translation_model.parameters())

		# Now create optimizer for translation models. 		
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)
		# self.optimizer = torch.optim.RMSprop(self.parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

		# Set discriminator parameter list. 
		# self.discriminator_parameter_list = list(self.source_z_discriminator.parameters()) + list(self.target_z_discriminator.parameters())
		# if self.args.z_transform_discriminator:
		# 	self.discriminator_parameter_list += list(self.source_z_transform_discriminator.parameters()) + list(self.target_z_transform_discriminator.parameters())

		self.discriminator_parameter_list = list(self.discriminator_network.parameters())
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			self.discriminator_parameter_list += list(self.z_trajectory_discriminator.parameters())

		if self.args.task_discriminability:
			for k in range(self.args.number_of_tasks):
				self.discriminator_parameter_list += list(self.task_discriminators[k].parameters())

		# Create common optimizer for source, target, and discriminator networks. 
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator_parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)
		# self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator_parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)

		self.save_object = {}

		# del self.save_object['Discriminator_Network']

		# self.save_object['forward_translation_model'] = self.forward_translation_model.state_dict()
		self.save_object['backward_translation_model'] = self.backward_translation_model.state_dict()
		self.save_object['z_discriminator'] = self.discriminator_network.state_dict()
		# self.save_object['source_z_discriminator'] = self.source_z_discriminator.state_dict()
		# self.save_object['target_z_discriminator'] = self.target_z_discriminator.state_dict()
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:		
			self.save_object['z_trajectory_discriminator'] = self.z_trajectory_discriminator.state_dict()

		# Overwrite the save from super. 
		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Load translation model.
		# self.forward_translation_model.load_state_dict(self.load_object['forward_translation_model'])
		self.backward_translation_model.load_state_dict(self.load_object['backward_translation_model'])
		self.discriminator_network.load_state_dict(self.load_object['z_discriminator'])

		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			self.z_trajectory_discriminator.load_state_dict(self.load_object['z_trajectory_discriminator'])

	def get_z_transformation(self, latent_z, latent_b):

		# New transformation... 
		prepadded_z = torch.cat([torch.zeros((1,self.args.batch_size,self.args.z_dimensions)).to(device), latent_z])
		postpadded_z = torch.cat([latent_z, torch.zeros((1,self.args.batch_size,self.args.z_dimensions)).to(device)])
		
		latent_z_transformation_vector = torch.cat([prepadded_z, postpadded_z], dim=-1)
		# latent_z_transformation_weights = torch.cat([latent_b, torch.zeros(1,self.args.batch_size).to(device)])
		# Forgot to change the masks… The mask should now consider the last z tuple / transition, because it is the (z_n,0) pair, not a (z_n,z_n) dummy pair?
		latent_z_transformation_weights = torch.cat([latent_b, torch.ones(1,self.args.batch_size).to(device)])

		return latent_z_transformation_vector, latent_z_transformation_weights, None

	def encode_decode_trajectory(self, policy_manager, i, return_trajectory=False, domain=None, initialize_run=False, bucket_index=None):

		# Check if the index is too big. If yes, just sample randomly.		
		if i >= len(policy_manager.dataset):
			print("Randomly sampling data point because we extended range.")
			i = np.random.randint(0, len(policy_manager.dataset))

		# Since the joint training manager nicely lets us get dictionaries, just use it, but remember not to train. 
		# This does all the steps we need.
		source_input_dict, source_var_dict, source_eval_dict = policy_manager.run_iteration(self.counter, i, return_dicts=True, train=False, bucket_index=bucket_index)		

		# If we're using GRAB data, also remember the trajectory information. to visualize. 
		if policy_manager.args.data in ['GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', 'GRABObject']:
			
			# Use the get batch eleemnt function. 
			batched_data_element = policy_manager.get_batch_element(i)
			# Now log the trajctory ID. 			
			self.GRAB_trajectory_ID = np.array([batched_data_element[b]['file'] for b in range(self.args.batch_size)])			

		if self.args.z_normalization and not(initialize_run):
			if domain==0:
				source_var_dict['latent_z_indices'] = (source_var_dict['latent_z_indices']-self.source_z_mean)/self.source_z_std
			else:
				source_var_dict['latent_z_indices'] = (source_var_dict['latent_z_indices']-self.target_z_mean)/self.target_z_std

		return source_input_dict, source_var_dict, source_eval_dict

	def translate_latent_z(self, latent_z, latent_b=None, domain=1):
		
		# if len(latent_z.shape)==2 or latent_z.shape[0]==1:
		# 	print("Embedding in translate latent z")
		# 	embed()
		
		# Here, domain is the domain they're translating "FROM"

		# Translate Z. 	
		if self.args.recurrent_translation:
			
			# We need a denoising implementation for the recurrent translation model.
			corrupted_latent_z = self.corrupt_inputs(latent_z)

			# Use the corrupted_latent_z instead of the original.
			translated_latent_z = self.translation_model_list[domain].forward(corrupted_latent_z, epsilon=self.translated_z_epsilon, precomputed_b=latent_b)
		else:						
			translated_latent_z = self.translation_model_list[domain].forward(latent_z, action_epsilon=self.translated_z_epsilon)

		# Now normalizing translated latent z. 
		translated_latent_z = (translated_latent_z-self.source_z_mean)/self.source_z_std

		return translated_latent_z

	def compute_aggregate_supervised_loss(self, domain=1):

		# # Well technically shouldn't ahve to run this..
		# self.set_z_objects()
		
		# Aggregate CDSL value
		self.aggregate_cdsl_stat = 0.

		eval_ind_range = np.arange(0,len(self.original_source_latent_z_set),self.args.batch_size)
		eval_ind_range = np.append(eval_ind_range,len(self.original_source_latent_z_set))

		for k, v in enumerate(eval_ind_range[:-1]):	
			with torch.no_grad():

				# DOESN'T THIS IMPLICITLY ASSUME ALIGNED TEMPORAL SEGMENTATION? 
				# USING THE Z SET... RATHER THAN Z'S COMPTUTED FROM CORRESPONDING TRAJECTORIES... ASSUMES ... 
				# well not exactly aligned temporal segmentation, but something weaker - same number of z's across domains across all trajectories. 
				# Which is still bad enough to mess up evaluation... basically ignore aggregate supervised loss? 
				detached_z = torch.tensor(self.original_target_latent_z_set[v:eval_ind_range[k+1]]).to(device)
				cross_domain_z = torch.tensor(self.original_source_latent_z_set[v:eval_ind_range[k+1]]).to(device)

				if self.args.recurrent_translation:
					# self.aggregate_cdsl_stat += (self.translation_model_list[domain].get_probabilities(detached_z, epsilon=self.translated_z_epsilon, precomputed_b=update_dictionary['latent_b'], evaluate_value=cross_domain_z)
					self.aggregate_cdsl_stat += 0.
				else:					
			
					# Is this even legit???!?!?
					# print("Embedding in agg superivsed loss")
					# embed()
					self.aggregate_cdsl_stat += self.translation_model_list[domain].get_probabilities(detached_z, action_epsilon=self.translated_z_epsilon, evaluate_value=cross_domain_z).sum()
					# self.aggregate_cdsl_stat += (self.translation_model_list[domain].get_probabilities(detached_z, action_epsilon=self.translated_z_epsilon, evaluate_value=cross_domain_z)**2).sum()
					
		if self.args.recurrent_translation:
			self.aggregate_cdsl_value = 0.
		else:
			self.aggregate_cdsl_value = (self.aggregate_cdsl_stat/len(self.original_source_latent_z_set)).detach().cpu().numpy()

	def compute_cross_domain_supervision_loss(self, update_dictionary, domain=1):

		# Basically feed in the predicted zs from the translation model, and get likelihoods of the zs from the target domain. 
		# This can be used as a loss function or as an evaluation metric. 


		if self.args.new_supervision:
			# Gather Z statistics.
			detached_z = update_dictionary['supervised_latent_z'].detach()
			cross_domain_z = update_dictionary['cross_domain_supervised_latent_z'].detach()

		else:
			# Gather Z statistics.
			detached_z = update_dictionary['latent_z'].detach()
			cross_domain_z = update_dictionary['cross_domain_latent_z'].detach()

		# # TRYING Z NORMALIZATION THING! 
		# if self.args.z_normalization is None:
		
		# 	concat_zs = torch.cat([detached_z,cross_domain_z])
		# 	z_mean = concat_zs.mean(dim=0)
		# 	z_std = concat_zs.std(dim=0)
		# 	normed_zs = (concat_zs-z_mean)/z_std
		# 	detached_z = normed_zs[:detached_z.shape[0]]
		# 	cross_domain_z = normed_zs[detached_z.shape[0]:]
			
		###############################################	
		
		if self.args.recurrent_translation:	
			unweighted_unmasked_cross_domain_supervision_loss = - self.translation_model_list[domain].get_probabilities(detached_z, epsilon=self.translated_z_epsilon, precomputed_b=update_dictionary['latent_b'], evaluate_value=cross_domain_z)
		else:
			unweighted_unmasked_cross_domain_supervision_loss = - self.translation_model_list[domain].get_probabilities(detached_z, action_epsilon=self.translated_z_epsilon, evaluate_value=cross_domain_z)

		###############################################

		# # Clamp these values. 
		# torch.clamp(unmasked_learnt_subpolicy_loglikelihoods,min=self.args.subpolicy_clamp_value)

		return unweighted_unmasked_cross_domain_supervision_loss

	def compute_triplet_loss(self, update_dictionary):

		# Implement triplet loss based on task ID. 
		# If task ID is similar, minimize representation distance, but if task ID is different, maximize representation distance up to some threshold.
		pass

	def update_networks(self, domain, policy_manager, update_dictionary):

		#########################################################################
		# If we're implementing a regular GAN (as opposed to a Wasserstein GAN), just use super.update_networks.
		#########################################################################	

		if self.args.wasserstein_gan or self.args.lsgan:
			self.alternate_gan_update(domain, policy_manager, update_dictionary)
		else:		
			# Regular GAN update.
			super().update_networks(domain, policy_manager, update_dictionary)		

	def alternate_gan_update(self, domain, policy_manager, update_dictionary):
	
		#########################################################################
		# Here, implement Wasserstein GAN or LSGAN style objective! 
		#########################################################################

		#########################################################################				
		# (1) First, update the representation based on discriminability.
		#########################################################################

		# Since we are in the translation model setting, use self.optimizer rather either source / target policy manager. 
		self.optimizer.zero_grad()

		###########################################################
		# (1a) First, compute reconstruction loss.
		###########################################################

		# Compute VAE loss on the current domain as likelihood plus weighted KL.  
		self.likelihood_loss = 0.
		self.encoder_KL = 0.
		self.unweighted_VAE_loss = 0.
		self.VAE_loss = self.vae_loss_weight*self.unweighted_VAE_loss

		###########################################################
		# (1b) Next, compute discriminability loss.
		###########################################################

		# # Compute discriminability loss for encoder (implicitly ignores decoder).
		# # Pretend the label was the opposite of what it is, and train the encoder to make the discriminator think this was what was true. 
		# # I.e. train encoder to make discriminator maximize likelihood of wrong label.
		# # domain_label = torch.tensor(1-domain).to(device).long().view(1,)
		# domain_label = domain*torch.ones(update_dictionary['discriminator_logprob'].shape[0]*update_dictionary['discriminator_logprob'].shape[1]).to(device).long()
		# self.unweighted_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['discriminator_logprob'].view(-1,2), 1-domain_label).mean()
		# self.discriminability_loss = self.discriminability_loss_weight*self.unweighted_discriminability_loss
			
		if self.args.wasserstein_gan:
			# Wasserstein GAN loss.
			# For wasserstein GAN loss, the translation model is expected to maximize: E_{z \sim p(z)} [D(G(z))], or minimize... E_{z \sim p(z)} [ - D(G(z))]
			# Basically just multiply with - domain here... 
			# self.unweighted_discriminability_loss = -domain*update_dictionary['discriminator_prob'][...,domain].mean()

			# Moving to implementing the discriminator as a critic network rather than a classifier.
			# Still using discriminator "prob" as dictionary key. 
			self.unweighted_discriminability_loss = -domain*update_dictionary['discriminator_prob'].mean()

		elif self.args.lsgan:
			# LSGAN Discriminability loss.
			# Discriminability loss only active when domain = 1.
			self.unweighted_discriminability_loss = domain*((update_dictionary['discriminator_prob']-domain)**2).mean()
		

		# In either case, weight the discirminability loss.
		self.discriminability_loss = self.discriminability_loss_weight*self.unweighted_discriminability_loss

		###########################################################
		# (1c) Next, compute z_trajectory discriminability loss.
		###########################################################
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			if self.args.z_trajectory_discriminator:
				traj_domain_label = domain*torch.ones(self.args.batch_size).to(device).long()
				# Overwrite update_dictionary['z_trajectory_weights']. 
				update_dictionary['z_trajectory_weights'] = torch.ones(self.args.batch_size).to(device).float()
			
			elif self.args.z_transform_discriminator:
				traj_domain_label = domain_label

			# Set z transform discriminability loss.
			self.unweighted_z_trajectory_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['z_trajectory_discriminator_logprob'].view(-1,2), 1-traj_domain_label)
			# self.unweighted_z_trajectory_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['z_trajectory_discriminator_logprob'].view(-1,2), 1-domain_label)
			self.masked_z_trajectory_discriminability_loss = update_dictionary['z_trajectory_weights'].view(-1,)*self.unweighted_z_trajectory_discriminability_loss
			# Mask the z transform discriminability loss based on whether or not this particular latent_z, latent_z transformation tuple should be used to train the representation.
			self.z_trajectory_discriminability_loss = self.z_trajectory_discriminability_loss_weight*self.masked_z_trajectory_discriminability_loss.mean()
		else:
			# Set z transform discriminability loss to dummy value.
			self.unweighted_z_trajectory_discriminability_loss = 0.
			self.z_trajectory_discriminability_loss = 0.

		# ###########################################################
		# # (1d) If active, compute equivariance loss. 
		# ###########################################################
		
		# if self.args.equivariance:
		# 	self.unweighted_unmasked_equivariance_loss = self.compute_equivariance_loss(update_dictionary)
		# 	# Now mask by the same temporal masks that we used for the discriminability versions of this idea. 
		# 	self.unweighted_masked_equivariance_loss = (update_dictionary['z_trajectory_weights'].view(-1,)*self.unweighted_unmasked_equivariance_loss).mean()
		# 	self.equivariance_loss = self.args.equivariance_loss_weight*self.unweighted_masked_equivariance_loss

		# else:
		# 	self.equivariance_loss = 0.

		###########################################################
		# (1e) If active, compute cross domain z loss. 
		###########################################################
	
		# Remember, the cross domain gt supervision loss should only be active when... trnaslating, i.e. when we have domain==1.
		if self.args.cross_domain_supervision and domain==1:
			# Call function to compute this. # This function depends on whether we have a translation model or not.. 
			self.unweighted_unmasked_cross_domain_supervision_loss = self.compute_cross_domain_supervision_loss(update_dictionary)
			# Now mask using batch mask.			
			# self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).mean()
			self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).sum()/(policy_manager.batch_mask.sum())
			# Now zero out if we want to use partial supervision..
			self.datapoint_masked_cross_domain_supervised_loss = self.supervised_datapoints_multiplier*self.unweighted_masked_cross_domain_supervision_loss
			# Now weight.			
			self.cross_domain_supervision_loss = self.args.cross_domain_supervision_loss_weight*self.datapoint_masked_cross_domain_supervised_loss
		else:
			self.unweighted_masked_cross_domain_supervision_loss = 0.
			self.cross_domain_supervision_loss = 0.

		###########################################################
		# (1f) If active, compute task discriminability loss.
		###########################################################

		if self.args.task_discriminability:
			# Set the same kind of label we used in z_trajectory_discriminability..
			traj_domain_label = domain*torch.ones(self.args.batch_size).to(device).long()
			# Create an NLL based on task_discriminator_logprobs...
			self.unweighted_task_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['task_discriminator_logprob'].view(-1,2), 1-traj_domain_label)
			# Weight and average.
			self.task_discriminability_loss = self.args.task_discriminability_loss_weight*self.unweighted_task_discriminability_loss.mean()
		else:
			self.unweighted_task_discriminability_loss = 0.
			self.task_discriminability_loss = 0.

		###########################################################
		# (1g) Compute identity losses.
		###########################################################

		# If "translating" source domain z., 
		if domain==0:
			self.unweighted_identity_translation_loss = self.compute_identity_loss(update_dictionary)			
		else:
			self.unweighted_identity_translation_loss = 0.
		self.identity_translation_loss = self.args.identity_translation_loss_weight*self.unweighted_identity_translation_loss

		###########################################################
		# (1h) Finally, compute total losses. 
		###########################################################

		# Total discriminability loss. 
		self.total_discriminability_loss = self.discriminability_loss + self.z_trajectory_discriminability_loss + self.task_discriminability_loss 

		# Total encoder loss: 
		self.total_VAE_loss = self.VAE_loss + self.total_discriminability_loss + self.cross_domain_supervision_loss	+ self.identity_translation_loss

		if not(self.skip_vae):
			# Go backward through the generator (encoder / decoder), and take a step. 
			self.total_VAE_loss.backward()

			# If we are in the translation model setting, use self.optimizer rather either source / target policy manager. 
			self.optimizer.step()

		#########################################################################
		# (2) Next, update the discriminator based on Wasserstein GAN loss.
		#########################################################################

		# Zero gradients of discriminator(s).
		self.discriminator_optimizer.zero_grad()

		###########################################################
		# (2a) Compute Z-discriminator loss.
		###########################################################

		# If we tried to zero grad the discriminator and then use NLL loss on it again, Pytorch would cry about going backward through a part of the graph that we already \ 
		# went backward through. Instead, just pass things through the discriminator again, but this time detaching latent_z. 
		discriminator_prob = self.discriminator_network(update_dictionary['detached_latent_z'])

		if self.args.wasserstein_gan:
		
			# Wasserstein GAN loss.
			# Here, domain label should just determine sign of loss.. and we have every element of the batch is the same domain.
			# In this case... generated / translated z's (when domain==1) have to have +ve loss, and when domain==0, loss should be -ve. 
			# This is described by.. 2*(domain-0.5) 
			# This is only for .. backward translation... 
			# self.discriminator_loss = ((domain-0.5)*2) * discriminator_prob[...,domain].mean()

			# Moving to implementing the discriminator as a critic network rather than as a classifier.
			self.discriminator_loss = ((domain-0.5)*2) * discriminator_prob.mean()
		elif self.args.lsgan:		
			# LSGAN Discriminator loss.
			self.discriminator_loss = ((update_dictionary['discriminator_prob'] - (1-domain))**2).mean()

		###########################################################
		# (2b) Compute Z-trajectory discriminator loss. 
		###########################################################

		if self.args.z_trajectory_discriminator or self.args.z_transform_discriminator:
			z_trajectory_discriminator_logprob, z_trajectory_discriminator_prob = self.z_trajectory_discriminator.get_probabilities(update_dictionary['z_trajectory'].detach())
			self.unmasked_z_trajectory_discriminator_loss = self.negative_log_likelihood_loss_function(z_trajectory_discriminator_logprob.view(-1,2), traj_domain_label)
			# Mask the z transform discriminator loss based on whether or not this particular latent_z, latent_z transformation tuple should be used to train the discriminator.
			self.unweighted_z_trajectory_discriminator_loss = (update_dictionary['z_trajectory_weights'].view(-1,)*self.unmasked_z_trajectory_discriminator_loss)
			self.z_trajectory_discriminator_loss = self.args.z_trajectory_discriminator_weight*self.unweighted_z_trajectory_discriminator_loss.mean()
		else:
			self.unweighted_z_trajectory_discriminator_loss = 0.
			self.z_trajectory_discriminator_loss = 0.

		###########################################################
		# (2c) Compute Task-Discriminator loss.
		###########################################################
		
		if self.args.task_discriminability:
			task_discriminator_logprob, _ = self.task_discriminators[update_dictionary['sample_task_id']].get_probabilities(update_dictionary['translated_latent_z'].detach())
			self.unweighted_task_discriminator_loss = self.negative_log_likelihood_loss_function(task_discriminator_logprob.view(-1,2), traj_domain_label)
			self.task_discriminator_loss = self.args.task_discriminator_weight*self.unweighted_task_discriminator_loss.mean()
		else:
			self.unweighted_task_discriminator_loss = 0.
			self.task_discriminator_loss = 0.

		###########################################################
		# (2d) Compute Discriminator Gradient Penalty
		###########################################################

		if self.args.gradient_penalty:
			self.unweighted_wasserstein_gradient_penalty = self.compute_wasserstein_gradient_penalty(domain, update_dictionary)
		else:
			self.unweighted_wasserstein_gradient_penalty = 0.
		self.wasserstein_gradient_penalty = self.args.gradient_penalty_weight*self.unweighted_wasserstein_gradient_penalty

		###########################################################
		# (2e) Merge discriminator losses.
		###########################################################

		self.total_discriminator_loss = self.discriminator_loss + self.z_trajectory_discriminator_loss + self.task_discriminator_loss + self.wasserstein_gradient_penalty

		###########################################################
		# (2f) Now update discriminator(s).
		###########################################################

		if not(self.skip_discriminator):
			# Now go backward and take a step.
			self.total_discriminator_loss.backward()
			self.discriminator_optimizer.step()

			if self.args.wasserstein_discriminator_clipping:
				for param in self.discriminator_parameter_list:
					param.data.clamp_(min=-self.args.wasserstein_discriminator_clipping_value,max=self.args.wasserstein_discriminator_clipping_value)

	def compute_wasserstein_gradient_penalty(self, domain, update_dictionary):
		# Compute gradient penalty:

		# Using https://github.com/EmilienDupont/wgan-gp/blob/master/training.py#L100 as a reference. 

		# 1) Get source and translated target input z's. 
		# 2) Compute interpolation of the source and translated target input z's. 
		#	# 2a) For interpolation compute which of the z's is the smaller one, (after reshape), sample an appropraite number of zs from the smaller domain. 
		# 	# 2b) Pad the smaller z with those randomly sampled z's. 
		# 	# 2c) Add the padded zs with the larger z set. 
		# 3) Feed interpolated z's into the discriminator. 
		# 4) Compute gradient of discriminator output with respect to inputs. 
		# 5) Compute norm of gradients.
		# 6) Feed back as penalty. 

		# 0) Setting this penalty to 0 when the domain is source domain, because we're not translating here.
		if domain==0:			
			gradient_penalty = 0
			return gradient_penalty

		# 1) Get source and translated target z's. 
		# Sample a random batch of z's cross domain, so we can enforce a gradient penalty. 
		with torch.no_grad():
			# Get opposite domain policy manager.
			target_policy_manager = self.get_domain_manager(1-domain)
			# Feed in trajectory.
			# Instead of i, use randomly sampled batch. 
			# 
			if self.args.debugging_datapoints>-1:
				index = np.random.randint(0, self.args.debugging_datapoints)
			else:
				index = np.random.randint(0, self.extent)
			
			cross_domain_input_dict, cross_domain_var_dict, cross_domain_eval_dict = self.encode_decode_trajectory(target_policy_manager, index, domain=1-domain)
			
			# Log cross domain latent z in update dictionary. 
			source_latent_z = cross_domain_var_dict['latent_z_indices']

		translated_target_zs = update_dictionary['translated_latent_z'].detach()

		# 2) Compute interpolation of the source and translated target input z's. 
		#	# 2a) For interpolation compute which of the z's is the smaller one, (after reshape), sample an appropraite number of zs from the smaller domain. 
		z_sets = [source_latent_z.view(-1,self.args.z_dimensions), translated_target_zs.view(-1,self.args.z_dimensions)]
		# if source_latent_z.view(-1,self.args.z_dimensions).shape[0] < .shape[0]:
		if z_sets[0].shape[0]<z_sets[1].shape[0]:
			# Source is smaller. 
			smaller_zs = 0
		else:
			smaller_zs = 1

		# First find out how many extra z samples we need. 
		number_of_z_samples = z_sets[1-smaller_zs].shape[0]-z_sets[smaller_zs].shape[0]
		# Now draw these extra z samples.		
		indices = torch.randint(0,high=z_sets[smaller_zs].shape[0],size=(number_of_z_samples,)).to(device)
		extra_z_samples = z_sets[smaller_zs][indices]
		
		# 	# 2b) Padding the smaller z set with the randomly sampled zs.
		padded_smaller_set = torch.cat([z_sets[smaller_zs],extra_z_samples])
		
		#	# 2c) Adding via interp.
		interpolation_alpha = np.random.uniform()
		interpolated_zs = (interpolation_alpha*padded_smaller_set + (1.-interpolation_alpha)*z_sets[1-smaller_zs]).detach()
		interpolated_zs_with_grads = torch.autograd.Variable(interpolated_zs, requires_grad=True)

		# 3) Feeding interpolated z's into the discriminator.
		interpolated_discriminator_probabilities = self.discriminator_network(interpolated_zs_with_grads)

		# 4) Compute gradient of discriminator output with respect to inputs. 
		gradients = torch.autograd.grad(outputs=interpolated_discriminator_probabilities, inputs=interpolated_zs_with_grads,
							   grad_outputs=torch.ones(interpolated_discriminator_probabilities.size()).to(device),
							   create_graph=True, retain_graph=True)[0]

		# 5) Compute norm of gradients.
		gradient_norm = torch.sqrt( (gradients**2).sum(dim=1) + 1e-6)
		gradient_penalty = ((gradient_norm - 1) ** 2).mean()
	

		# 6) Feed back as penalty
		return gradient_penalty

	def visualize_low_likelihood_skills(self, domain, update_dictionary, input_dict): 
			
		self.global_traj_counter_max = 10

		if domain==0 and self.global_traj_counter<self.global_traj_counter_max:
			lps = self.query_GMM_density(evaluation_domain=domain, point_set=update_dictionary['latent_z'])
			if lps.min()<self.lowlikelihood_threshold:	

				# Get z's for which it's low likelihood. 
				tindices = torch.where(lps<self.lowlikelihood_threshold)
				indices = (tindices[0].detach().cpu().numpy(), tindices[1].detach().cpu().numpy())

				# Just directly get the trajectory.
				traj = input_dict['sample_traj'][indices]
				unnorm_traj = (traj*self.source_manager.norm_denom_value) + self.source_manager.norm_sub_value

				# print("embedding in visualize low ll skillss")
				# embed()
				# Now visualize.. 
				self.visualizer.visualize_joint_trajectory(unnorm_traj, gif_path="LowlikelihoodTraj", gif_name="Unaligned_Traj{0}.gif".format(self.global_traj_counter), return_and_save=True, end_effector=self.args.ee_trajectories)

				self.global_traj_counter+=1

	def run_iteration(self, counter, i, domain=None, skip_viz=False):
		
		#################################################
		## Algorithm:
		#################################################
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use (source or target, i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain; Encode trajectory segments into latent z's and compute likelihood of trajectory actions under the decoder.		
		# 		# 3) If we have selected the domain to translate from, translate the z sequence.
		# 		# 4) Feed into discriminator, get likelihood of real / translated. 
		#			# Remember, this mode is slightly different. The discriminator(s) are strictly speaking not differentiating between the domains themselves, 
		# 			# but original and translated versions of zs in each domain,. 
		#			# Remember this when selecting discriminator inputs. 
		# 		# 5) Compute and apply gradient updates. 
		#		# 6) Update plots. 
		#################################################

		#################################################
		## (0) Setup things like training phases, epsilon values, etc.
		#################################################

		self.set_iteration(counter, i)		

		#################################################
		## (1) Select which domain to run on; also supervision of discriminator.
		#################################################

		# Use same domain across batch for simplicity. 
		if domain is None:
			domain = np.random.binomial(1,0.5)
			
		self.counter = counter
		policy_manager = self.get_domain_manager(domain)

		#################################################	
		## (2) Get trajectory segment and encode and decode. 
		#################################################

		update_dictionary = {}
		source_input_dict, source_var_dict, source_eval_dict = self.encode_decode_trajectory(policy_manager, i, domain=domain)
		update_dictionary['subpolicy_inputs'], update_dictionary['latent_z'], update_dictionary['loglikelihood'], update_dictionary['kl_divergence'] = \
			source_eval_dict['subpolicy_inputs'], source_var_dict['latent_z_indices'], source_eval_dict['learnt_subpolicy_loglikelihoods'], source_var_dict['kl_divergence']
		
		if self.args.task_discriminability or self.args.task_based_supervision:
			update_dictionary['sample_task_id'] = source_input_dict['sample_task_id']

		if update_dictionary['latent_z'] is not None:

			# print('Batch Shape:',source_input_dict['old_concatenated_traj'].shape)

			#################################################
			## (3) If domain==Target, translate the latent z(s) to the source domain.
			#################################################

			detached_original_latent_z = update_dictionary['latent_z'].detach()
			if domain==1:
				update_dictionary['translated_latent_z'] = self.translate_latent_z(detached_original_latent_z, source_var_dict['latent_b'].detach())
			else:
				# Otherwise.... set translated z to latent z, because that's what we're going to feed to the discriminator(s). 
				# Detach just to make sure gradients don't pass into the source encoder. 
				update_dictionary['translated_latent_z'] = detached_original_latent_z

			# Set this variable, because this is what the discriminator training uses as input. 
			update_dictionary['detached_latent_z'] = update_dictionary['translated_latent_z'].detach()
			
			#################################################
			## (4) Feed latent z's to discriminator, and get discriminator likelihoods. 
			#################################################

			# In the joint transfer case: this is only for one domain.
			# update_dictionary['discriminator_logprob'], discriminator_prob = self.discriminator_network(update_dictionary['translated_latent_z'])
			# Log the probability as well (haha)

			if self.args.wasserstein_gan or self.args.lsgan:
				update_dictionary['discriminator_prob'] = self.discriminator_network(update_dictionary['translated_latent_z'])
			else:
				update_dictionary['discriminator_logprob'], update_dictionary['discriminator_prob'] = self.discriminator_network(update_dictionary['translated_latent_z'])

			#################################################
			## (4b) If we are using a z_transform discriminator.
			#################################################
			
			if self.args.z_transform_discriminator or self.args.equivariance:
				# Calculate the transformation.
				update_dictionary['z_transformations'], update_dictionary['z_trajectory_weights'], _ = self.get_z_transformation(update_dictionary['translated_latent_z'], source_var_dict['latent_b'])
				update_dictionary['z_trajectory_discriminator_logprob'], z_transform_discriminator_prob = self.z_trajectory_discriminator(update_dictionary['z_transformations'])
				update_dictionary['z_trajectory'] = update_dictionary['z_transformations']
				
				# Add this probability to the dictionary to visualize.
				viz_dict = {'z_trajectory_discriminator_probs': z_transform_discriminator_prob[...,domain].detach().cpu().numpy().mean()}

			elif self.args.z_trajectory_discriminator:
				# Feed the entire z trajectory to the discriminator.
				update_dictionary['z_trajectory_discriminator_logprob'], z_trajectory_discriminator_prob = self.z_trajectory_discriminator.get_probabilities(update_dictionary['translated_latent_z'])
				viz_dict = {'z_trajectory_discriminator_probs': z_trajectory_discriminator_prob[...,domain].detach().cpu().numpy().mean()}
				update_dictionary['z_trajectory'] = update_dictionary['translated_latent_z']
			else:
				viz_dict = {}

			# if domain==0:
			# 	lps = self.query_GMM_density(evaluation_domain=domain, point_set=update_dictionary['latent_z'])
			# 	# if lps.min()<-400:
			# 	if lps.min()<-80:	
			# 		print("Embed in run iter of")
			# 		embed()

			
			####
			self.lowlikelihood_threshold = -80
			self.visualize_low_likelihood_skills(domain, update_dictionary, source_input_dict)


			#################################################
			## (4c) If we are using cross domain supervision.
			#################################################

			if self.args.cross_domain_supervision:

				# Get opposite domain policy manager.
				target_policy_manager = self.get_domain_manager(1-domain)
				# Feed in trajectory.
				cross_domain_input_dict, cross_domain_var_dict, cross_domain_eval_dict = self.encode_decode_trajectory(target_policy_manager, i, domain=1-domain)
				# Log cross domain latent z in update dictionary. 
				update_dictionary['cross_domain_latent_z'] = cross_domain_var_dict['latent_z_indices']
				# also log b's.
				update_dictionary['latent_b'] = source_var_dict['latent_b']
			

			#################################################
			## (4d) If we are using task based discriminability.
			#################################################

			if self.args.task_discriminability:

				# Feed in the source latent z's into the appropriate task discriminatory (based on batch task ID, source_input_dict['batch_task_id'] ). 
				update_dictionary['task_discriminator_logprob'], update_dictionary['task_discriminator_prob'] = self.task_discriminators[source_input_dict['sample_task_id']].get_probabilities(update_dictionary['translated_latent_z'])



			#################################################
			## (4e) Compute set based supervised loss.			
			#################################################

			# update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['latent_z'].view(-1,self.args.z_dimensions), update_dictionary['cross_domain_latent_z'].view(-1,self.args.z_dimensions))

			# The .view(-1,z_dim) was accumulating z's across the batch, which is wrong. Compute set based loss independently across the batch, then do mean reduction.
			# update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['latent_z'], update_dictionary['cross_domain_latent_z'])

			# Need to feed translated_latent_z's rather than the latent_z.. 
			if domain==1 and self.args.supervised_set_based_density_loss:
				# update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['translated_latent_z'], update_dictionary['cross_domain_latent_z'], differentiable_outputs=True)
				update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['cross_domain_latent_z'], update_dictionary['translated_latent_z'], differentiable_outputs=True)

			# print("Embed in JFE Run iter")
			# embed()

			#################################################
			## (5) Compute and apply gradient updates. 			
			#################################################	

			self.update_networks(domain, policy_manager, update_dictionary)			

			#################################################
			## (6) Update Plots. 			
			#################################################
			
			viz_dict['domain'] = domain
			
			if self.args.wasserstein_gan or self.args.lsgan:
				viz_dict['discriminator_probs'] = update_dictionary['discriminator_prob'].detach().cpu().numpy().mean()
			else:
				viz_dict['discriminator_probs'] = update_dictionary['discriminator_prob'][...,domain].detach().cpu().numpy().mean()

			if self.args.task_discriminability:
				viz_dict['task_discriminator_probs'] = update_dictionary['task_discriminator_prob'][...,domain].detach().cpu().numpy().mean()

			if domain==1 and self.args.supervised_set_based_density_loss:
				viz_dict['forward_set_based_supervised_loss'], viz_dict['backward_set_based_supervised_loss'] = update_dictionary['forward_set_based_supervised_loss'].mean().detach().cpu().numpy(), update_dictionary['backward_set_based_supervised_loss'].mean().detach().cpu().numpy()

			if not(skip_viz):
				self.update_plots(counter, viz_dict, log=True)

class PolicyManager_JointFixEmbedCycleTransfer(PolicyManager_JointFixEmbedTransfer):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_JointFixEmbedCycleTransfer, self).__init__(args, source_dataset, target_dataset)

	def create_networks(self):
		
		 # Call super create networks, to create all the necessary networks. 
		super().create_networks()

		# In addition, now create translation model networks.

		if self.args.recurrent_translation:
			# Create recurrent translation model from variational model template.
			self.forward_translation_model = ContinuousVariationalPolicyNetwork_Batch(self.args.z_dimensions, self.args.var_hidden_size, self.args.z_dimensions, self.args, number_layers=self.args.var_hidden_size, translation_network=True).to(device)
			self.backward_translation_model = ContinuousVariationalPolicyNetwork_Batch(self.args.z_dimensions, self.args.var_hidden_size, self.args.z_dimensions, self.args, number_layers=self.args.var_hidden_size, translation_network=True).to(device)
		else:
			self.forward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)
			self.backward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)

		# Create list of translation models to select from based on source domain.
		self.translation_model_list = [self.forward_translation_model, self.backward_translation_model]

		# # Instead of single z discriminator, require two different z discriminators. 
		# # Just use the self.discriminator_network for now.
		# self.source_z_discriminator = DiscreteMLP(self.args.z_dimensions, self.hidden_size, 2, args=self.args).to(device)
		# self.target_z_discriminator = DiscreteMLP(self.args.z_dimensions, self.hidden_size, 2, args=self.args).to(device)
		
		# For now, don't use discriminators.... 
		# Create lists of discriminators. 
		# self.discriminator_list = [self.source_discriminator, self.target_discriminator]
		self.z_discriminator_list = [self.source_z_discriminator, self.target_z_discriminator]

		if self.args.z_transform_discriminator:
			self.source_z_trajectory_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)
			self.target_z_trajectory_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)

			# self.z_trajectory_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)
			self.z_trajectory_discriminator_list = [self.source_z_trajectory_discriminator, self.target_z_trajectory_discriminator]

		elif self.args.z_trajectory_discriminator:
			
			self.source_z_trajectory_discriminator = EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size, args=self.args).to(device)
			self.target_z_trajectory_discriminator = EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size, args=self.args).to(device)

			# self.z_trajectory_discriminator = EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size, args=self.args).to(device)
			self.z_trajectory_discriminator_list = [self.source_z_trajectory_discriminator, self.target_z_trajectory_discriminator]

	def create_training_ops(self):

		# Don't actually call super().create_training_ops(),
		# Because this creates optimizers with source and target encoder decoder parameters in the optimizer. 

		# Instead, create other things here. 
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')

		# Set regular parameter list. 
		self.parameter_list = list(self.forward_translation_model.parameters()) + list(self.backward_translation_model.parameters())
		# self.parameter_list = list(self.backward_translation_model.parameters())

		# Now create optimizer for translation models. 
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

		# For now, don't use discriminators.... 
		
		# Set discriminator parameter list. 
		self.discriminator_parameter_list = list(self.source_z_discriminator.parameters()) + list(self.target_z_discriminator.parameters())
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			self.discriminator_parameter_list += list(self.source_z_trajectory_discriminator.parameters()) + list(self.target_z_trajectory_discriminator.parameters())

		# self.discriminator_parameter_list = list(self.discriminator_network.parameters())
		# if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
		# 	self.discriminator_parameter_list += list(self.z_trajectory_discriminator.parameters())

		# Create common optimizer for source, target, and discriminator networks. 
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator_parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)

		self.save_object = {}

		self.save_object['forward_translation_model'] = self.forward_translation_model.state_dict()
		self.save_object['backward_translation_model'] = self.backward_translation_model.state_dict()

		self.save_object['source_z_discriminator'] = self.source_z_discriminator.state_dict()
		self.save_object['target_z_discriminator'] = self.target_z_discriminator.state_dict()
		
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:		
			# pass
			self.save_object['source_z_trajectory_discriminator'] = self.z_trajectory_discriminator.state_dict()
			self.save_object['target_z_trajectory_discriminator'] = self.z_trajectory_discriminator.state_dict()

		# Overwrite the save from super. 
		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Load translation model.
		self.forward_translation_model.load_state_dict(self.load_object['forward_translation_model'])
		self.backward_translation_model.load_state_dict(self.load_object['backward_translation_model'])

		self.source_discriminator_network.load_state_dict(self.load_object['source_z_discriminator'])
		self.target_discriminator_network.load_state_dict(self.load_object['target_z_discriminator'])

		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			self.source_z_trajectory_discriminator.load_state_dict(self.load_object['source_z_trajectory_discriminator'])
			self.target_z_trajectory_discriminator.load_state_dict(self.load_object['target_z_trajectory_discriminator'])
			# pass 

	def update_plots(self, counter, viz_dict, log=False):

		# Call super update plots for the majority of the work. Call this with log==false to make sure that wandb only logs things we add in this function. 
		log_dict = super().update_plots(counter, viz_dict, log=False)
		domain= viz_dict['domain']

		if self.args.cross_domain_supervision:
			log_dict['Unweighted Cycle Cross Domain Superivision Loss'] = self.unweighted_masked_cycle_cdsl.mean()
			log_dict['Cycle Cross Domain Superivision Loss'] = self.cycle_cross_domain_supervision_loss
			log_dict['Cycle Z Error'] = viz_dict['Cycle Z Error']

		############################################################
		# Now implement visualization of original latent set and translated z space in both directions. 
		############################################################			

		if counter%self.args.display_freq==0:
			
			##################################################
			# Visualize Translated Z Trajectories.
			##################################################

			self.set_translated_z_sets(domain=0)
			# self.construct_tuple_embeddings()

			log_dict['Source Z Trajectory Joint TSNE Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_tsne_image)
			log_dict['Target Z Trajectory Joint TSNE Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_tsne_image)
			log_dict['Source Z Trajectory Joint PCA Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_pca_image)
			# log_dict['Target Z Trajectory Joint PCA Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_pca_image)
			log_dict['Source Z Trajectory Joint DENSNE Embedding Visualizations'] = self.return_wandb_image(self.source_z_traj_densne_image)
			log_dict['Target Z Trajectory Joint DENSNE Embedding Visualizations'] = self.return_wandb_image(self.target_z_traj_densne_image)

			##################################################
			# Now log combined source and translated target visualizations, and if we want, target and translated source.
			##################################################

			log_dict["TSNE Translated Source Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_p30'])	

			# log_dict["TSNE Combined Source and Translated Target Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_p05'])
			# log_dict["TSNE Combined Source and Translated Target Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_p10'])
			# log_dict["TSNE Combined Source and Translated Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_p30'])
			log_dict["TSNE Combined Translated Source and Target Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_p05'])
			log_dict["TSNE Combined Translated Source and Target Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_p10'])
			log_dict["TSNE Combined Translated Source and Target Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_p30'])

			if self.check_toy_dataset():					
				log_dict["TSNE Translated Source Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_traj_p30'])

				# log_dict["TSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_traj_p05'])
				# log_dict["TSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_traj_p10'])
				# log_dict["TSNE Combined Source and Translated Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_origsource_transtarget_traj_p30'])
				log_dict["TSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 05"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_traj_p05'])
				log_dict["TSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 10"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_traj_p10'])
				log_dict["TSNE Combined Translated Source and Target Trajectory Embeddings Perplexity 30"] = self.return_wandb_image(self.viz_dictionary['tsne_transsource_origtarget_traj_p30'])

		if log:
			wandb.log(log_dict, step=counter)
		else:
			return log_dict

	def update_networks(self, domain, policy_manager, update_dictionary):

		#########################################################################
		#########################################################################
		# (1) First, update the representation based on reconstruction and discriminability.
		#########################################################################
		#########################################################################

		# Since we are in the translation model setting, use self.optimizer rather either source / target policy manager. 
		self.optimizer.zero_grad()

		###########################################################
		# (1a) First, compute reconstruction loss.
		###########################################################

		# Compute VAE loss on the current domain as likelihood plus weighted KL.  
		# self.likelihood_loss = -update_dictionary['loglikelihood'].mean()
		# self.encoder_KL = update_dictionary['kl_divergence'].mean()
		# self.unweighted_VAE_loss = self.likelihood_loss + self.args.kl_weight*self.encoder_KL

		self.likelihood_loss = 0.
		self.encoder_KL = 0.		
		self.unweighted_VAE_loss = 0.
		self.VAE_loss = self.vae_loss_weight*self.unweighted_VAE_loss

		###########################################################
		# (1b) Next, compute discriminability loss.
		###########################################################

		# Compute discriminability loss for encoder (implicitly ignores decoder).
		# Pretend the label was the opposite of what it is, and train the encoder to make the discriminator think this was what was true. 
		# I.e. train encoder to make discriminator maximize likelihood of wrong label.
		# domain_label = torch.tensor(1-domain).to(device).long().view(1,)
		# domain_label = domain*torch.ones(update_dictionary['discriminator_logprob'].shape[0]*update_dictionary['discriminator_logprob'].shape[1]).to(device).long()
		# self.unweighted_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['discriminator_logprob'].view(-1,2), 1-domain_label).mean()
		self.unweighted_discriminability_loss = 0.
		self.discriminability_loss = self.discriminability_loss_weight*self.unweighted_discriminability_loss
			
		###########################################################
		# (1c) Next, compute z_trajectory discriminability loss.
		###########################################################
		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			
			if self.args.z_trajectory_discriminator:
				traj_domain_label = domain*torch.ones(self.args.batch_size).to(device).long()
				# Overwrite update_dictionary['z_trajectory_weights']. 
				update_dictionary['z_trajectory_weights'] = torch.ones(self.args.batch_size).to(device).float()
			
			elif self.args.z_transform_discriminator:
				traj_domain_label = domain_label

			# Set z transform discriminability loss.
			# self.unweighted_z_trajectory_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['z_trajectory_discriminator_logprob'].view(-1,2), 1-traj_domain_label)
			# # self.unweighted_z_trajectory_discriminability_loss = self.negative_log_likelihood_loss_function(update_dictionary['z_trajectory_discriminator_logprob'].view(-1,2), 1-domain_label)
			# self.masked_z_trajectory_discriminability_loss = update_dictionary['z_trajectory_weights'].view(-1,)*self.unweighted_z_trajectory_discriminability_loss
			# # Mask the z transform discriminability loss based on whether or not this particular latent_z, latent_z transformation tuple should be used to train the representation.
			# self.z_trajectory_discriminability_loss = self.z_trajectory_discriminability_loss_weight*self.masked_z_trajectory_discriminability_loss.mean()

			self.unweighted_z_trajectory_discriminability_loss = 0.
			self.masked_z_trajectory_discriminability_loss = 0. 
			self.z_trajectory_discriminability_loss = 0.
		else:
			# Set z transform discriminability loss to dummy value.
			self.z_trajectory_discriminability_loss = 0.
		
		###########################################################
		# (1d) If active, compute cross domain z loss. 
		###########################################################
	
		# Remember, the cross domain gt supervision loss should only be active when... trnaslating, i.e. when we have domain==1.
		if self.args.cross_domain_supervision:
			# Call function to compute this. # This function depends on whether we have a translation model or not.. 
			self.unweighted_unmasked_cross_domain_supervision_loss = self.compute_cross_domain_supervision_loss(update_dictionary)
			# Now mask using batch mask.			
			# self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).mean()
			self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).sum()/(policy_manager.batch_mask.sum())
			# Now weight.
			self.cross_domain_supervision_loss = self.args.cross_domain_supervision_loss_weight*self.unweighted_masked_cross_domain_supervision_loss		
		else:
			self.unweighted_unmasked_cross_domain_supervision_loss = 0. 
			self.cross_domain_supervision_loss = 0.

		###########################################################
		# (1e) Compute cycle consistency cross domain supervised losses. 
		###########################################################

		if self.args.cross_domain_supervision:
			# Call function to compute this. # This function depends on whether we have a translation model or not.. 
			self.unweighted_unmasked_cycle_cdsl = self.compute_cycle_cross_domain_supervision_loss(update_dictionary)
			# Now mask using batch mask.			
			# self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).mean()
			self.unweighted_masked_cycle_cdsl = (policy_manager.batch_mask*self.unweighted_unmasked_cycle_cdsl).sum()/(policy_manager.batch_mask.sum())
			# Now weight.
			self.cycle_cross_domain_supervision_loss = self.args.cycle_cross_domain_supervision_loss_weight*self.unweighted_masked_cycle_cdsl
		else:
			self.unweighted_unmasked_cycle_cdsl = 0.
			self.cycle_cross_domain_supervision_loss = 0.

		###########################################################
		# (1f) Finally, compute total losses. 
		###########################################################

		# Total discriminability loss. 
		self.total_discriminability_loss = self.discriminability_loss + self.z_trajectory_discriminability_loss 

		# Total encoder loss: 
		self.total_VAE_loss = self.VAE_loss + self.total_discriminability_loss + self.cross_domain_supervision_loss	+ self.cycle_cross_domain_supervision_loss

		if not(self.skip_vae):
			# Go backward through the generator (encoder / decoder), and take a step. 
			self.total_VAE_loss.backward()

			# Since we are in the translation model setting, use self.optimizer rather either source / target policy manager. 
			self.optimizer.step()

		#########################################################################
		#########################################################################
		# (2) Next, update the discriminators based on domain label.
		#########################################################################
		#########################################################################

		# Zero gradients of discriminator(s).
		# self.discriminator_optimizer.zero_grad()

		###########################################################
		# (2a) Compute Z-discriminator loss.
		###########################################################

		# If we tried to zero grad the discriminator and then use NLL loss on it again, Pytorch would cry about going backward through a part of the graph that we already \ 
		# went backward through. Instead, just pass things through the discriminator again, but this time detaching latent_z. 
		# discriminator_logprob, discriminator_prob = self.discriminator_network(update_dictionary['detached_latent_z'])

		# Compute discriminator loss for discriminator. 
		# self.discriminator_loss = self.negative_log_likelihood_loss_function(discriminator_logprob.squeeze(1), torch.tensor(domain).to(device).long().view(1,))		
		# self.discriminator_loss = self.negative_log_likelihood_loss_function(discriminator_logprob.view(-1,2), domain_label).mean()
		self.discriminator_loss = 0.

		###########################################################
		# (2b) Compute Z-trajectory discriminator loss. 
		###########################################################

		if self.args.z_trajectory_discriminator or self.args.z_transform_discriminator:
			# z_trajectory_discriminator_logprob, z_trajectory_discriminator_prob = self.z_trajectory_discriminator.get_probabilities(update_dictionary['z_trajectory'].detach())
			# self.unmasked_z_trajectory_discriminator_loss = self.negative_log_likelihood_loss_function(z_trajectory_discriminator_logprob.view(-1,2), traj_domain_label)
			# # Mask the z transform discriminator loss based on whether or not this particular latent_z, latent_z transformation tuple should be used to train the discriminator.
			# self.unweighted_z_trajectory_discriminator_loss = (update_dictionary['z_trajectory_weights'].view(-1,)*self.unmasked_z_trajectory_discriminator_loss)

			self.unmasked_z_trajectory_discriminator_loss = 0.
			self.unweighted_z_trajectory_discriminator_loss = 0.
			self.z_trajectory_discriminator_loss = 0.
			# self.z_trajectory_discriminator_loss = self.args.z_trajectory_discriminator_weight*self.unweighted_z_trajectory_discriminator_loss.mean()
		else:
			self.z_trajectory_discriminator_loss = 0.

		###########################################################
		# (2c) Merge discriminator losses.
		###########################################################
		
		self.total_discriminator_loss = self.discriminator_loss + self.z_trajectory_discriminator_loss

		###########################################################
		# (2d) Now update discriminator(s).
		###########################################################

		# if not(self.skip_discriminator):
		# 	# Now go backward and take a step.
		# 	self.total_discriminator_loss.backward()
		# 	self.discriminator_optimizer.step()

	def compute_cycle_cross_domain_supervision_loss(self, update_dictionary, domain=1):

		# Compute a cycle-consistency version of te cross domain supervised loss. 

		# Here, do not detach inputs used for computed these losses, becaue we gradients to be propagated to both translation models. 
		# The individual cross domain superivision losses should prevent this from collapsing to identity / trivial mapping. 
		# Detach outputs.. though this shouldn't make a differnce. 

		# We want to enforce cycle consistency between the back-translated z (in the source domain) and the original source latent z. 
		# Do this by maximizing likelihood of the original source z under the opposite translation model (target to source), given the translated z from source to target. 
	
		# For now, since we aren't using a discriminator for the cycle consistency loss, we don't even really need to evaluate the backtranslated z.

		###############################################	
		
		# Assuming no recurrent translation. 
		# Probably need to use translation_model_list[1-domain] here? 
		# If source domain is 0, the translation_model_list[0] is the forward translation model, and translation_model_list[1-domain] is the backward model. 
		unweighted_unmasked_cycle_cdsl = - self.translation_model_list[1-domain].get_probabilities(update_dictionary['translated_latent_z'], action_epsilon=self.translated_z_epsilon, evaluate_value=update_dictionary['latent_z'].detach())				

		###############################################

		# # Clamp these values. 
		# torch.clamp(unmasked_learnt_subpolicy_loglikelihoods,min=self.args.subpolicy_clamp_value)

		return unweighted_unmasked_cycle_cdsl

	def compute_cross_domain_supervision_loss(self, update_dictionary, domain=1):

		# Basically feed in the predicted zs from the translation model, and get likelihoods of the zs from the target domain. 
		# This can be used as a loss function or as an evaluation metric. 

		# Gather Z statistics.
		detached_z = update_dictionary['latent_z'].detach()
		cross_domain_z = update_dictionary['cross_domain_latent_z'].detach()

		###############################################	

		# Forward CDSL
		forward_cdsl = - self.translation_model_list[domain].get_probabilities(detached_z, action_epsilon=self.translated_z_epsilon, evaluate_value=cross_domain_z)
		# Backward CDSL		
		backward_cdsl = - self.translation_model_list[1-domain].get_probabilities(cross_domain_z, action_epsilon=self.translated_z_epsilon, evaluate_value=detached_z)
		
		###############################################

		# Compute symmetric CDSL
		unweighted_unmasked_cross_domain_supervision_loss = forward_cdsl + backward_cdsl

		return unweighted_unmasked_cross_domain_supervision_loss
	
	def run_iteration(self, counter, i, domain=None, skip_viz=False):

		#################################################
		## Algorithm:
		#################################################
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 0) Setup things like training phases, epsilon values, etc.
		# 		# 1) Select which domain to use (source or target, i.e. with 50% chance, select either domain).
		# 		# 2) Get batch of trajectories from desired domain; Encode trajectories into sequence of latent z's and compute likelihood of trajectory actions under the decoder.		
		# 		# 3) (If supervised) Get the sequence of latent z's this batch of trajectories is encoded as under the target domain encoder. 
		# 		# 4) Translate this sequence of latent z's to the target domain. 
		# 		# 5) Back-translate the translated latent z's to the source domain.
		# 		# 6) Feed into discriminator, get likelihood of real / translated. 
		#			# Remember, this mode is slightly different. The discriminator(s) are strictly speaking not differentiating between the domains themselves, 
		# 			# but original and translated versions of zs in each domain,. Remember this when selecting discriminator inputs. 		
		# 		# 7) Compute and apply gradient updates. 
		#		# 8) Update plots. 
		#################################################

		#################################################
		## (0) Setup things like training phases, epsilon values, etc.
		#################################################

		self.set_iteration(counter)		
		
		#################################################
		## (1) Select which domain to run on; also supervision of discriminator.
		#################################################

		# Use same domain across batch for simplicity. 
		if domain is None:
			domain = np.random.binomial(1,0.5)
			
		self.counter = counter
		policy_manager = self.get_domain_manager(domain)

		#################################################	
		## (2) Get batch of trajectories, encode and decode.
		#################################################

		update_dictionary = {}
		source_input_dict, source_var_dict, source_eval_dict = self.encode_decode_trajectory(policy_manager, i, domain=domain)
		update_dictionary['subpolicy_inputs'], update_dictionary['latent_z'], update_dictionary['loglikelihood'], update_dictionary['kl_divergence'] = \
			source_eval_dict['subpolicy_inputs'], source_var_dict['latent_z_indices'], source_eval_dict['learnt_subpolicy_loglikelihoods'], source_var_dict['kl_divergence']
		
		if update_dictionary['latent_z'] is not None:
			
			#################################################	
			## (3) (If supervised) Get target sequence of latent z's of this batch of trajectories.
			#################################################

			if self.args.cross_domain_supervision:

				# Get opposite domain policy manager.
				target_policy_manager = self.get_domain_manager(1-domain)
				# Feed in trajectory.
				cross_domain_input_dict, cross_domain_var_dict, cross_domain_eval_dict = self.encode_decode_trajectory(target_policy_manager, i, domain=1-domain)
				# Log cross domain latent z in update dictionary. 
				update_dictionary['cross_domain_latent_z'] = cross_domain_var_dict['latent_z_indices']
				# also log b's.
				update_dictionary['latent_b'] = source_var_dict['latent_b']

			#################################################	
			## (4) Translate this sequence of latent z's to the target domain. 
			#################################################

			# Set this variable, because this is what the discriminator training uses as input. 
			update_dictionary['detached_latent_z'] = update_dictionary['latent_z'].detach()			
			update_dictionary['translated_latent_z'] = self.translate_latent_z(update_dictionary['detached_latent_z'] , source_var_dict['latent_b'].detach(), domain=domain)
			update_dictionary['detached_translated_latent_z'] = update_dictionary['translated_latent_z'].detach()

			#################################################	
			## (5) Back-translate the translated latent z's to the source domain.
			#################################################

			update_dictionary['backtranslated_latent_z'] = self.translate_latent_z(update_dictionary['translated_latent_z'] , source_var_dict['latent_b'].detach(), domain=1-domain)			
			viz_dict = {} 
			# Also compute and log the backtranslated z error / cycle reconstruction z errror...
			viz_dict['Cycle Z Error'] = ((update_dictionary['backtranslated_latent_z']-update_dictionary['latent_z'])**2).mean()

			#################################################	
			## (6) Feed into discriminator, get likelihood of real / translated. 
			#################################################

			# Skip for now.
			 
			#################################################
			## (7) Compute and apply gradient updates. 			
			#################################################
			
			self.update_networks(domain, policy_manager, update_dictionary)

			#################################################
			## (8) Update Plots. 			
			#################################################
			
			
			viz_dict['domain'] = domain
			# viz_dict['discriminator_probs'] = discriminator_prob[...,domain].detach().cpu().numpy().mean()
			viz_dict['discriminator_probs'] = None

			if not(skip_viz):
				self.update_plots(counter, viz_dict, log=True)

class PolicyManager_JointTransfer(PolicyManager_Transfer):

	# Inherit from transfer.
	def __init__(self, args=None, source_dataset=None, target_dataset=None):
			
		# The inherited functions refer to self.args. Also making this to make inheritance go smooth.
		super(PolicyManager_JointTransfer, self).__init__(args, source_dataset, target_dataset)

		# Now create two instances of policy managers for each domain. Call them source and target domain policy managers. 
		self.source_manager = PolicyManager_BatchJoint(number_policies=4, dataset=self.source_dataset, args=self.source_args)
		self.target_manager = PolicyManager_BatchJoint(number_policies=4, dataset=self.target_dataset, args=self.target_args)

		self.source_dataset_size = len(self.source_manager.dataset) - self.source_manager.test_set_size
		self.target_dataset_size = len(self.target_manager.dataset) - self.target_manager.test_set_size

		# Now create variables that we need. 
		self.number_epochs = self.args.epochs
		self.extent = min(self.source_dataset_size, self.target_dataset_size)		

		# Now setup networks for these PolicyManagers. 		
		self.source_manager.setup()
		self.source_manager.initialize_training_batches()
		self.target_manager.setup()
		self.target_manager.initialize_training_batches()		

		# Now create variables that we need. 
		self.number_epochs = self.args.epochs
		self.extent = min(self.source_dataset_size, self.target_dataset_size)		
		self.decay_counter = self.decay_epochs*self.extent
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)

		# # Now define other parameters that will be required for the discriminator, etc. 
		# self.input_size = self.args.z_dimensions
		# self.hidden_size = self.args.hidden_size
		# self.output_size = 2
		# self.learning_rate = self.args.learning_rate

	def save_all_models(self, suffix):
		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)

		self.save_object = {}

		# Source
		self.save_object['Source_Policy_Network'] = self.source_manager.policy_network.state_dict()
		self.save_object['Source_Encoder_Network'] = self.source_manager.variational_policy.state_dict()
		# Target
		self.save_object['Target_Policy_Network'] = self.target_manager.policy_network.state_dict()
		self.save_object['Target_Encoder_Network'] = self.target_manager.variational_policy.state_dict()
		# Discriminator
		self.save_object['Discriminator_Network'] = self.discriminator_network.state_dict()				

		if self.args.z_transform_discriminator or self.args.z_trajectory_discriminator:
			self.save_object['Z_Trajectory_Discriminator_Network'] = self.z_trajectory_discriminator.state_dict()

		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Source
		self.source_manager.policy_network.load_state_dict(self.load_object['Source_Policy_Network'])
		self.source_manager.variational_policy.load_state_dict(self.load_object['Source_Encoder_Network'])
		# Target
		self.target_manager.policy_network.load_state_dict(self.load_object['Target_Policy_Network'])
		self.target_manager.variational_policy.load_state_dict(self.load_object['Target_Encoder_Network'])
		# Discriminator
		self.discriminator_network.load_state_dict(self.load_object['Discriminator_Network'])

		if self.args.z_trajectory_discriminator:
			self.z_trajectory_discriminator.load_state_dict(self.load_object['Z_Trajectory_Discriminator_Network'])

	def create_networks(self):

		super().create_networks()

		# Z Trajectory discriminator takes precedence.
		if self.args.z_trajectory_discriminator:
			self.z_trajectory_discriminator = EncoderNetwork(self.input_size, self.hidden_size, self.output_size, batch_size=self.args.batch_size, args=self.args).to(device)
		elif self.args.z_transform_discriminator:
			self.z_trajectory_discriminator = DiscreteMLP(2*self.input_size, self.hidden_size, self.output_size, args=self.args).to(device)

	def create_training_ops(self):

		super().create_training_ops()
		
		discriminator_opt_params = self.discriminator_network.parameters()
		if self.args.z_trajectory_discriminator or self.args.z_transform_discriminator:
			discriminator_opt_params = list(discriminator_opt_params) + list(self.z_trajectory_discriminator.parameters())
		self.discriminator_optimizer = torch.optim.Adam(discriminator_opt_params,lr=self.learning_rate, weight_decay=self.args.regularization_weight)	

	def encode_decode_trajectory(self, policy_manager, i, return_trajectory=False):

		# Check if the index is too big. If yes, just sample randomly.
		if i >= len(policy_manager.dataset):
			i = np.random.randint(0, len(policy_manager.dataset))

		# Since the joint training manager nicely lets us get dictionaries, just use it, but remember not to train. 
		# This does all the steps we need.
		source_input_dict, source_var_dict, source_eval_dict = policy_manager.run_iteration(self.counter, i, return_dicts=True, train=False)

		return source_input_dict, source_var_dict, source_eval_dict

	def get_z_transformation(self, latent_z, latent_b):

		# First compute the differences.
		# No torch diff op, so do it manually. 
		if self.args.z_transform_or_tuple:
			# Actually compute difference.
			latent_z_diff = latent_z[1:] - latent_z[:-1]
		else:
			# Instead of computing differences, we're going to copy over the subsequent / succeeding z's into the diff vector, to form a tuple.
			latent_z_diff = latent_z[1:]

		# Better way to compute weights is just roll latent_b
		with torch.no_grad():					
			latent_z_transformation_weights = latent_b.roll(-1,dims=0)
			# Zero out last weight, to ignore (z_t, 0) tuple at the end. 
			if self.args.ignore_last_z_transform:
				latent_z_transformation_weights[-1] = 0

		# Concatenate 0's to the latent_z_diff. 
		padded_latent_z_diff = torch.cat([latent_z_diff, torch.zeros((1,self.args.batch_size,self.args.z_dimensions)).to(device)],dim=0)

		# Now concatenate the z's themselves... 
		latent_z_transformation_vector = torch.cat([padded_latent_z_diff, latent_z], dim=-1)	

		return latent_z_transformation_vector, latent_z_transformation_weights
		# return latent_z_transformation_vector.view(-1,2*self.args.z_dimensions), latent_z_transformation_weights.view(-1,1)


		
		# Implement naive density based matching here..
		# Losses we need... VAE loss, KL loss, and.. partial supervised loss.
		# Supervised loss needs to be re-implemented in the setting without a translation model...
		# Maybe... treat it as an identity translation to reuse code? 

		# Basically create a dummy Gaussian with fixed variance to evaluate CDSL? 
		# Still need to use this setting to do.. optimizer management etc.
		# Also to prevent detaching to z's., because the translation model.... is not what we want to give gradients to? 
			
	def run_iteration(self, counter, i, domain=None, skip_viz=False):


		# Phases: 
		# Phase 1:  Train encoder-decoder for both domains initially, so that discriminator is not fed garbage. 
		# Phase 2:  Train encoder, decoder for each domain, and discriminator concurrently. 

		# Algorithm: 
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use (source or target, i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain. 
		# 		# 3) Encode trajectory segments into latent z's and compute likelihood of trajectory actions under the decoder.
		# 		# 4) Feed into discriminator, get likelihood of each domain.
		# 		# 5) Compute and apply gradient updates. 

		# Remember to make domain agnostic function calls to encode, feed into discriminator, get likelihoods, etc. 
		# (0) Setup things like training phases, epislon values, etc.
		self.set_iteration(counter)

		# (1) Select which domain to run on. This is supervision of discriminator.
		# Use same domain across batch for simplicity. 
		if domain is None:
			domain = np.random.binomial(1,0.5)
		self.counter = counter

		# (1.5) Get domain policy manager. 
		policy_manager = self.get_domain_manager(domain)
				
		# (2) & (3) Get trajectory segment and encode and decode. 
		update_dictionary = {}
		source_input_dict, source_var_dict, source_eval_dict = self.encode_decode_trajectory(policy_manager, i)
		update_dictionary['subpolicy_inputs'], update_dictionary['latent_z'], update_dictionary['loglikelihood'], update_dictionary['kl_divergence'] = \
			source_eval_dict['subpolicy_inputs'], source_var_dict['latent_z_indices'], source_eval_dict['learnt_subpolicy_loglikelihoods'], source_var_dict['kl_divergence']

		if update_dictionary['latent_z'] is not None:
			# (4) Feed latent z's to discriminator, and get discriminator likelihoods. 
			# In the joint transfer case:
			update_dictionary['discriminator_logprob'], discriminator_prob = self.discriminator_network(update_dictionary['latent_z'])

			# (4b) If we are using a z_transform discriminator.

			if self.args.z_trajectory_discriminator:
				# Feed the entire z trajectory to the discriminator.
				update_dictionary['z_trajectory_discriminator_logprob'], z_trajectory_discriminator_prob = self.z_trajectory_discriminator.get_probabilities(update_dictionary['latent_z'])
				viz_dict = {'z_trajectory_discriminator_probs': z_trajectory_discriminator_prob[...,domain].detach().cpu().numpy().mean()}
				update_dictionary['z_trajectory'] = update_dictionary['latent_z']

			elif self.args.z_transform_discriminator:					
				# Calculate the transformation.
				update_dictionary['z_transformations'], update_dictionary['z_trajectory_weights'] = self.get_z_transformation(update_dictionary['latent_z'], source_var_dict['latent_b'])
				update_dictionary['z_trajectory_discriminator_logprob'], z_trajectory_discriminator_prob = self.z_trajectory_discriminator.get_probabilities(update_dictionary['z_transformations'])
				update_dictionary['z_trajectory'] = update_dictionary['z_transformations']
				
				# Add this probability to the dictionary to visualize.
				viz_dict = {'z_trajectory_discriminator_probs': z_trajectory_discriminator_prob[...,domain].detach().cpu().numpy().mean()}
			else:
				viz_dict = {}

			# Detach. 
			update_dictionary['detached_latent_z'] = update_dictionary['latent_z'].detach()

			# (5) Compute and apply gradient updates. 			
			self.update_networks(domain, policy_manager, update_dictionary)

			# Now update Plots. 			
			# viz_dict = {'domain': domain, 'discriminator_probs': discriminator_prob.squeeze(0).mean(axis=0)[domain].detach().cpu().numpy()}
			viz_dict['domain'] = domain
			viz_dict['discriminator_probs'] = discriminator_prob[...,domain].detach().cpu().numpy().mean()

			if not(skip_viz):
				self.update_plots(counter, viz_dict)

class PolicyManager_DensityJointTransfer(PolicyManager_JointTransfer):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):
		
		# The inherited functions refer to self.args. Also making this to make inheritance go smooth.
		super(PolicyManager_DensityJointTransfer, self).__init__(args, source_dataset, target_dataset)

	def save_all_models(self, suffix):
		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)

		self.save_object = {}

		# Source
		self.save_object['Source_Policy_Network'] = self.source_manager.policy_network.state_dict()
		self.save_object['Source_Encoder_Network'] = self.source_manager.variational_policy.state_dict()
		# Target
		self.save_object['Target_Policy_Network'] = self.target_manager.policy_network.state_dict()
		self.save_object['Target_Encoder_Network'] = self.target_manager.variational_policy.state_dict()

		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Source
		self.source_manager.policy_network.load_state_dict(self.load_object['Source_Policy_Network'])
		self.source_manager.variational_policy.load_state_dict(self.load_object['Source_Encoder_Network'])
		# Target
		self.target_manager.policy_network.load_state_dict(self.load_object['Target_Policy_Network'])
		self.target_manager.variational_policy.load_state_dict(self.load_object['Target_Encoder_Network'])

	def update_networks(self, domain, policy_manager, update_dictionary):		

		#########################################################################
		#########################################################################
		# (1) First, update the representation based on reconstruction and discriminability.
		#########################################################################
		#########################################################################

		# Zero out gradients of encoder and decoder (policy).
		policy_manager.optimizer.zero_grad()

		###########################################################
		# (1a) First, compute reconstruction loss.
		###########################################################

		# Compute VAE loss on the current domain as likelihood plus weighted KL.  
		self.likelihood_loss = -update_dictionary['loglikelihood'].mean()
		self.encoder_KL = update_dictionary['kl_divergence'].mean()
		self.unweighted_VAE_loss = self.likelihood_loss + self.args.kl_weight*self.encoder_KL
		self.VAE_loss = self.vae_loss_weight*self.unweighted_VAE_loss

		###########################################################
		# (1b) Next, compute cross domain density. 
		###########################################################

		# Does this need to be masked? 
		self.unweighted_unmasked_cross_domain_density_loss = update_dictionary['cross_domain_density_loss']
		# Mask..
		self.unweighted_masked_cross_domain_density_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_density_loss).sum()/(policy_manager.batch_mask.sum())
		# Weight this loss.
		self.cross_domain_density_loss = self.args.cross_domain_density_loss_weight*self.unweighted_masked_cross_domain_density_loss

		###########################################################
		# (1c) If active, compute cross domain z loss. 
		###########################################################
	
		# Remember, the cross domain gt supervision loss should only be active when... trnaslating, i.e. when we have domain==1.
		if self.args.cross_domain_supervision:
			# Call function to compute this. # This function depends on whether we have a translation model or not.. 
			self.unweighted_unmasked_cross_domain_supervision_loss = update_dictionary['cross_domain_supervised_loss'].mean(dim=-1)
			# Now mask using batch mask.			
			# self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).mean()
			self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).sum()/(policy_manager.batch_mask.sum())
			# Now zero out if we want to use partial supervision..
			self.datapoint_masked_cross_domain_supervised_loss = self.supervised_datapoints_multiplier*self.unweighted_masked_cross_domain_supervision_loss
			# Now weight.			
			self.cross_domain_supervision_loss = self.args.cross_domain_supervision_loss_weight*self.datapoint_masked_cross_domain_supervised_loss
		else:
			self.unweighted_masked_cross_domain_supervision_loss = 0.
			self.cross_domain_supervision_loss = 0.

		###########################################################
		# (1d) Finally, compute total loss.
		###########################################################

		self.total_VAE_loss = self.VAE_loss + self.cross_domain_supervision_loss + self.cross_domain_density_loss

		# Go backward through the generator (encoder / decoder), and take a step. 
		self.total_VAE_loss.backward()
		policy_manager.optimizer.step()

	def update_plots(self, counter, viz_dict=None, log=True):

		log_dict = super().update_plots(counter, viz_dict, log=False)

		if counter%self.args.display_freq==0:
			log_dict = self.construct_density_embeddings(log_dict)

		if log:
			wandb.log(log_dict, step=counter)
		else:
			return log_dict

	def compute_density_based_loss(self, update_dictionary):

		# print("Embedding in density based loss computation")
		# embed()

		return - self.GMM.log_prob(update_dictionary['latent_z'])

	def compute_cross_domain_supervision_loss(self, i, update_dictionary):

		# Basically feed in the predicted zs from the translation model, and get likelihoods of the zs from the target domain. 
		# This can be used as a loss function or as an evaluation metric. 

		# # Gather Z statistics.
		# detached_z = update_dictionary['latent_z'].detach()
		# cross_domain_z = update_dictionary['cross_domain_latent_z'].detach()

		# # Log cross domain latent z in update dictionary. 
		# update_dictionary['cross_domain_latent_z'] = cross_domain_var_dict['latent_z_indices']

		# Compute the Cross Domain Loss.. here, maybe should just be L2 loss.
		unweighted_unmasked_cross_domain_supervision_loss = ((update_dictionary['cross_domain_latent_z'] - update_dictionary['latent_z'])**2)
		
		return unweighted_unmasked_cross_domain_supervision_loss
	
	def run_iteration(self, counter, i, domain=None, skip_viz=False):
		
		# Overall algorithm.
		# Preprocessing
		# 1) For N samples of datapoints from the source domain. 
		# 	# 2) Feed these input datapoints into the source domain encoder and get source encoding z. 
		#	# 3) Add Z to Source Z Set. 
		# 4) Build GMM with centers around the N Source Z set Z's.

		# Training. 
		# 0) Setup things like training phases, epislon values, etc.
		# 1) For E Epochs:
		# 	# 2) For D datapoints:
		#		# 3) Sample x from target domain.
		#		# 4) Feed x into target domain decoder to get z encoding. 
		# 		# 5) Compute overall objective. 
		# 			# 5a) Compute action likelihood. 		
		# 			# 5b) Compute likelihood of target z encoding under the source domain GMM. 
		# 			# 5c) Compute supervised loss.
		# 		# 6) Compute gradients of objective and then update networks / policies.

		# (0) Setup things like training phases, epislon values, etc.
		self.set_iteration(counter, i=i)
		
		# (3), (4), (5a) Get input datapoint from target domain. One directional in this case.
		source_input_dict, source_var_dict, source_eval_dict = self.encode_decode_trajectory(self.target_manager, i)
		update_dictionary = {}
		update_dictionary['subpolicy_inputs'], update_dictionary['latent_z'], update_dictionary['loglikelihood'], update_dictionary['kl_divergence'] = \
			source_eval_dict['subpolicy_inputs'], source_var_dict['latent_z_indices'], source_eval_dict['learnt_subpolicy_loglikelihoods'], source_var_dict['kl_divergence']

		if not(skip_viz):			
			# 5b) Compute likelihood of target z under source domain.
			# update_dictionary['cross_domain_density_loss'] = self.compute_density_based_loss(update_dictionary)
			update_dictionary['cross_domain_density_loss'] = - self.query_GMM_density(evaluation_domain=0, point_set=update_dictionary['latent_z'], differentiable_points=True)
						
			# Precursor to 5c - run cross domain encode / decode. Running this in run itreation so we have access to variables.
			cross_domain_input_dict, cross_domain_var_dict, cross_domain_eval_dict = self.encode_decode_trajectory(self.source_manager, i)
			update_dictionary['cross_domain_latent_z'] = cross_domain_var_dict['latent_z_indices']

			# 5c) Compute supervised loss..
			update_dictionary['cross_domain_supervised_loss'] = self.compute_cross_domain_supervision_loss(i, update_dictionary)

			# 5d) Compute set based supervised loss.			
			# update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['latent_z'].view(-1,self.args.z_dimensions), update_dictionary['cross_domain_latent_z'].view(-1,self.args.z_dimensions))

			# The .view(-1,z_dim) was accumulating z's across the batch, which is wrong. Compute set based loss independently across the batch, then do mean reduction.
			# update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['latent_z'], update_dictionary['cross_domain_latent_z'])

			if domain==1:
				# update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['translated_latent_z'], update_dictionary['cross_domain_latent_z'])
				update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['cross_domain_latent_z'], update_dictionary['translated_latent_z'])

			# 6) Compute gradients of objective and then update networks / policies.
			self.update_networks(1, self.target_manager, update_dictionary)					

			# 7) Update plots. 
			viz_dict = {}
			viz_dict['domain'] = domain
			if domain==1:
				viz_dict['forward_set_based_supervised_loss'], viz_dict['backward_set_based_supervised_loss'] = update_dictionary['forward_set_based_supervised_loss'].mean().detach().cpu().numpy(), update_dictionary['backward_set_based_supervised_loss'].mean().detach().cpu().numpy()
		
			self.update_plots(counter, viz_dict, log=True)

class PolicyManager_DensityJointFixEmbedTransfer(PolicyManager_JointFixEmbedTransfer):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_DensityJointFixEmbedTransfer, self).__init__(args, source_dataset, target_dataset)
	
	def create_networks(self):
		
		 # Call super create networks, to create all the necessary networks. 
		super().create_networks()

		# self.backward_translation_model = ContinuousMLP(self.args.z_dimensions, self.args.hidden_size, self.args.z_dimensions, args=self.args, number_layers=self.translation_model_layers).to(device)
		# self.translation_model_list = [None, self.backward_translation_model]

	def save_all_models(self, suffix):

		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)

		self.save_object = {}

		# self.save_object['forward_translation_model'] = self.forward_translation_model.state_dict()
		self.save_object['backward_translation_model'] = self.backward_translation_model.state_dict()

		# Overwrite the save from super. 
		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Load translation model.
		# self.forward_translation_model.load_state_dict(self.load_object['forward_translation_model'])
		self.backward_translation_model.load_state_dict(self.load_object['backward_translation_model'])

	def update_networks(self, domain, policy_manager, update_dictionary):		

		#########################################################################
		#########################################################################
		# (1) First, update the representation based on reconstruction and discriminability.
		#########################################################################
		#########################################################################

		# Zero out gradients of encoder and decoder (policy).
		self.optimizer.zero_grad()

		# Log some dummy values
		self.likelihood_loss = 0.
		self.encoder_KL = 0.
		self.unweighted_VAE_loss = self.likelihood_loss + self.args.kl_weight*self.encoder_KL
		self.VAE_loss = self.vae_loss_weight*self.unweighted_VAE_loss		

		###########################################################
		# (1a) First compute cross domain density. 
		###########################################################
		if self.args.z_gmm:
			self.weighted_forward_loss = - self.args.forward_density_loss_weight*update_dictionary['forward_density_loss']
			self.weighted_backward_loss = - self.args.backward_density_loss_weight*update_dictionary['backward_density_loss']

			# Does this need to be masked? 	
			self.unweighted_unmasked_cross_domain_density_loss = self.weighted_forward_loss + self.weighted_backward_loss
			# Mask..
			self.unweighted_masked_cross_domain_density_loss = (self.unsupervised_loss_batch_mask*self.unweighted_unmasked_cross_domain_density_loss).sum()/(policy_manager.batch_mask.sum())

		else:
			self.unweighted_masked_cross_domain_density_loss = 0.
		
		# Weight this loss.
		self.cross_domain_density_loss = self.args.cross_domain_density_loss_weight*self.unweighted_masked_cross_domain_density_loss
		
		###########################################################
		# (1b) Compute cross domain z tuple density. 
		###########################################################

		if self.args.z_tuple_gmm:
			# (1b1) Forward Z tuple density loss.
			self.unweighted_masked_forward_z_tuple_density_loss = - update_dictionary['target_z_trajectory_weights']*update_dictionary['forward_z_tuple_density_loss']
			self.unweighted_forward_z_tuple_density_loss = self.unweighted_masked_forward_z_tuple_density_loss.sum()/update_dictionary['target_z_trajectory_weights'].sum()
			self.forward_z_tuple_density_loss = self.args.forward_tuple_density_loss_weight*self.unweighted_forward_z_tuple_density_loss

			# (1b2) Backward Z tuple density loss.
			self.unweighted_masked_backward_z_tuple_density_loss = - update_dictionary['source_z_trajectory_weights']*update_dictionary['backward_z_tuple_density_loss']
			self.unweighted_backward_z_tuple_density_loss = self.unweighted_masked_backward_z_tuple_density_loss.sum()/update_dictionary['source_z_trajectory_weights'].sum()
			self.backward_z_tuple_density_loss = self.args.backward_tuple_density_loss_weight*self.unweighted_backward_z_tuple_density_loss
			
			# Total Z tuple density loss. 
			self.unweighted_masked_cross_domain_z_tuple_density_loss = self.forward_z_tuple_density_loss + self.backward_z_tuple_density_loss			
		else:
			self.unweighted_masked_cross_domain_z_tuple_density_loss = 0.
			
		self.cross_domain_z_tuple_density_loss = self.args.cross_domain_z_tuple_density_loss_weight*self.unweighted_masked_cross_domain_z_tuple_density_loss

		###########################################################
		# (1c) If active, compute cross domain z loss. 
		###########################################################
	
		# Remember, the cross domain gt supervision loss should only be active when... trnaslating, i.e. when we have domain==1.
		if self.args.cross_domain_supervision:
			# Call function to compute this. # This function depends on whether we have a translation model or not.. 
			self.unweighted_unmasked_cross_domain_supervision_loss = update_dictionary['cross_domain_supervised_loss']
			# Now mask using batch mask.	
			# print("Embed in supervision..")
			# embed() 
				
			# self.unweighted_masked_cross_domain_supervision_loss = (policy_manager.batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).mean()
			self.unweighted_masked_cross_domain_supervision_loss = (self.supervised_loss_batch_mask*self.unweighted_unmasked_cross_domain_supervision_loss).sum()/(self.supervised_loss_batch_mask.sum())
			# Now zero out if we want to use partial supervision..
			self.datapoint_masked_cross_domain_supervised_loss = self.supervised_datapoints_multiplier*self.unweighted_masked_cross_domain_supervision_loss
			# Now weight.			
			self.cross_domain_supervision_loss = self.args.cross_domain_supervision_loss_weight*self.datapoint_masked_cross_domain_supervised_loss
		else:
			self.unweighted_masked_cross_domain_supervision_loss = 0.
			self.cross_domain_supervision_loss = 0.

		###########################################################
		# (1d) If active, compute the task based density loss.
		###########################################################

		if self.args.task_based_supervision:
			self.unweighted_task_based_supervised_loss = self.forward_supervised_set_tuple_based_loss + self.backward_supervised_set_tuple_based_loss			
		else:
			self.unweighted_task_based_supervised_loss = 0.
		self.task_based_supervised_loss = self.args.task_based_supervised_loss_weight*self.unweighted_task_based_supervised_loss

		###########################################################
		# (1e) Finally, compute total loss.
		###########################################################

		# print("Embedding in VAE loss")
		# embed()
		self.total_VAE_loss = self.cross_domain_supervision_loss + self.cross_domain_density_loss + self.cross_domain_z_tuple_density_loss + self.task_based_supervised_loss

		# Go backward through the generator (encoder / decoder), and take a step. 
		self.total_VAE_loss.backward()
		self.optimizer.step()

	def construct_z_tuples_from_z_sets(self, z_set, cummulative_number_zs):

		# Making this a fucntion so that we can use for source and target domains.
		z_trajectories = [z_set[low:high] for low, high in zip(cummulative_number_zs, cummulative_number_zs[1:])]
		z_tuples = None
		for k, v in enumerate(z_trajectories):

			# Get te distinct tuples of z's that occur in this particular z trajecotry. 
			unpadded_z_tuple_list = torch.cat([v[i:i+2].view(-1,2*self.args.z_dimensions) for i in range(v.shape[0]-1)])

			# Also pad this set of z tuples with (0, z1) and (zn, 0), so that we also encourage similarity of initial and terminal z distributions. 
			pretup =  torch.cat([torch.zeros_like(unpadded_z_tuple_list[0,:self.args.z_dimensions]).to(device), unpadded_z_tuple_list[0,:self.args.z_dimensions]]).view(1,-1)
			posttup = torch.cat([unpadded_z_tuple_list[-1, self.args.z_dimensions:],torch.zeros_like(unpadded_z_tuple_list[-1,self.args.z_dimensions:]).to(device)]).view(1,-1)

			z_tuple_list = torch.cat([pretup, unpadded_z_tuple_list, posttup])

			if z_tuples is None:
				z_tuples = z_tuple_list
			else:
				z_tuples =  torch.cat([z_tuples, z_tuple_list])

		# print("embedding in construct tuples frm set")
		# embed()

		return z_tuples		

	def construct_z_tuple_sets(self, domain=None, policy_manager=None, z_set=None):

		######################################################
		# Compute target tuples.
		######################################################

		# If we're also using z tuple GMMs, re-assemble z tuples from the differentiable_target_means, using the self.target_mananger.cummulative_number_zs
		# Remember, cummulative_number_zs is the for N trajectories, but we subsampled 500 (distinct) z's from that. 
		# Needs to be handled - if cummulative_z[-1] > 500 or < 500. 

		if policy_manager.cummulative_number_zs[-1] > self.final_number_of_zs[domain]:

			# Find bucket of self.final_number_of_target_zs, and 
			upper_bucket_index = np.digitize(self.final_number_of_zs[domain], policy_manager.cummulative_number_zs)
			# Select cummulative_number_zs until this bucket. 
			cummulative_number_zs = policy_manager.cummulative_number_zs[:upper_bucket_index]
			# Set last element to self.final_number_of_zs[1]
			cummulative_number_zs[-1] = self.final_number_of_zs[domain]
		else:
			cummulative_number_zs = policy_manager.cummulative_number_zs
		
		# Now reassemble the z tuples from the given z_set. 
		z_tuples = self.construct_z_tuples_from_z_sets(z_set, cummulative_number_zs)
		
		return z_tuples
		
	def differentiable_mean_computation(self, counter):
		
		# Function to differentiably compute means of the reverse GMM, as a function of the trnaslation mdoel. 
		# The original target latent z variable is always pre translation.

		self.inputs_to_translation_model = torch.tensor(self.original_target_latent_z_set).to(device)		
		self.differentiable_target_means = self.backward_translation_model(self.inputs_to_translation_model)

		# Create torch tensor for source means.
		self.differentiable_source_means = torch.tensor(self.original_source_latent_z_set).to(device)

		if self.args.z_tuple_gmm:

			# Get target domain z tuples.
			self.target_z_tuple_set = self.construct_z_tuple_sets(domain=1, policy_manager=self.target_manager, z_set=self.differentiable_target_means)

			# Remember, this mean set doesn't need to be recomputed at every step. 
			# Check if we're running for the first time, otherwise just keep as is. 
			if counter==0:
				# Get source domain z tuples.
				self.source_z_tuple_set = self.construct_z_tuple_sets(domain=0, policy_manager=self.source_manager, z_set=self.differentiable_source_means)

	def update_plots(self, counter, viz_dict, log=False):

		# Call super update plots for the majority of the work. Call this with log==false to make sure that wandb only logs things we add in this function. 		
		log_dict = super().update_plots(counter, viz_dict, log=False)

		if self.args.z_tuple_gmm:
			# Log Z tuple densities. 
			log_dict['Forward Z Tuple Density Loss'] = self.forward_z_tuple_density_loss
			log_dict['Backward Z Tuple Density Loss'] = self.backward_z_tuple_density_loss
			log_dict['Unweighted Forward Z Tuple Density Loss'] = self.unweighted_forward_z_tuple_density_loss
			log_dict['Unweighted Backward Z Tuple Density Loss'] = self.unweighted_backward_z_tuple_density_loss			
			log_dict['Unweighted Z Tuple Density Loss'] = self.unweighted_masked_cross_domain_z_tuple_density_loss
			log_dict['Z Tuple Density Loss'] = self.cross_domain_z_tuple_density_loss 

		if self.args.task_based_supervision:

			# Log task based supervised loss. 
			log_dict['Forward Supervised Set Tuple Based Loss'] = self.forward_supervised_set_tuple_based_loss.detach().cpu().numpy()
			log_dict['Backward Supervised Set Tuple Based Loss'] = self.backward_supervised_set_tuple_based_loss.detach().cpu().numpy()
			log_dict['Unweighted Task Based Supervised Loss'] = self.unweighted_task_based_supervised_loss.detach().cpu().numpy()
			log_dict['Task Based Supervised Loss'] = self.task_based_supervised_loss.detach().cpu().numpy()

		# Actually log. 
		if log:
			wandb.log(log_dict, step=counter)
		else:
			return log_dict

	def compute_cross_domain_supervision_loss(self, update_dictionary):

		if self.args.number_of_supervised_datapoints==0:
			update_dictionary['cross_domain_supervised_loss'] = 0.
			update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = 0., 0.

			return 

		# Compute new style of supervised loss.
		if self.args.new_supervision:

			# First, get supervised datapoint index that satisfies number of supervised datapoints.
			supervised_datapoint_index = self.select_supervised_datapoint_batch_index()
			
			# print("Sup DPI:", supervised_datapoint_index, "Now embedding in run iter")
			# embed()
			# Now collect input and representation for this supervised datapoint.
			source_supervised_input_dict, source_supervised_var_dict, source_supervised_eval_dict = self.encode_decode_trajectory(self.target_manager, supervised_datapoint_index)

			# Create supervised loss batch mask. 
			self.supervised_loss_batch_mask = copy.deepcopy(self.target_manager.batch_mask)
			if self.args.number_of_supervised_datapoints<self.args.batch_size and not(self.args.number_of_supervised_datapoints==-1):
				# Find number of items to zero out. 
				number_of_items = self.args.batch_size - self.args.number_of_supervised_datapoints
				# Zero out appropriate number of batch items.
				self.supervised_loss_batch_mask[:,-number_of_items:] = 0.

			# Now collect cross domain input and representation for this datapoint. 
			cross_domain_supervised_input_dict, cross_domain_supervised_var_dict, cross_domain_supervised_eval_dict = self.encode_decode_trajectory(self.source_manager, supervised_datapoint_index)

			# Log these things in update_dictionary for loss computation. 
			update_dictionary['supervised_latent_z'] = source_supervised_var_dict['latent_z_indices']
			update_dictionary['cross_domain_supervised_latent_z'] = cross_domain_supervised_var_dict['latent_z_indices']

			# Also translate the supervised source latent z for the set based supervised loss.
			update_dictionary['translated_supervised_latent_z'] = self.translate_latent_z(update_dictionary['supervised_latent_z'].detach(), source_supervised_var_dict['latent_b'].detach())

			# Now compute supervised loss.
			update_dictionary['cross_domain_supervised_loss'] = super().compute_cross_domain_supervision_loss(update_dictionary)

			if domain==1:
				update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['cross_domain_supervised_latent_z'], update_dictionary['translated_supervised_latent_z'], differentiable_outputs=True)

		else:
			# Otherwise just compute superivsed loss directly. 
			update_dictionary['cross_domain_supervised_loss'] = super().compute_cross_domain_supervision_loss(update_dictionary)

			# Compute set based superivsed loss.
			if domain==1:
				update_dictionary['forward_set_based_supervised_loss'], update_dictionary['backward_set_based_supervised_loss'] = self.compute_set_based_supervised_GMM_loss(update_dictionary['cross_domain_latent_z'], update_dictionary['translated_latent_z'], differentiable_outputs=True)

	def get_task_datapoint_indices(self, sampled_task_id):

		# Get datapoints in source and target domains for the sampled task ID. 
		# Efficiently doing this requires index to block maps..
		source_bucket = np.random.choice(self.source_manager.block_index_list_for_task[sampled_task_id])
		target_bucket = np.random.choice(self.target_manager.block_index_list_for_task[sampled_task_id])

		return source_bucket, target_bucket

	def compute_task_based_supervision_loss(self, update_dictionary, i):

		###########################################################
		# 1) Select a feasbile task. 
		###########################################################
		task_list = np.arange(0,20)
		sampled_task = np.random.choice(task_list, size=1, p=self.task_frequencies)[0]

		###########################################################
		# 2) Select a batch of datapoints in both domains from this feasible task. 
		###########################################################

		source_bucket, target_bucket = self.get_task_datapoint_indices(sampled_task)		
		
		###########################################################
		# 3) Run skill learning pipeline on this pair of datapoints.
		###########################################################

		source_sup_input_dict, source_sup_var_dict, source_sup_eval_dict = self.encode_decode_trajectory(self.source_manager, i, bucket_index=source_bucket)
		target_sup_input_dict, target_sup_var_dict, target_sup_eval_dict = self.encode_decode_trajectory(self.target_manager, i, bucket_index=target_bucket)

		###########################################################
		# 4) Translate the target domain z's before constructing GMMs. 
		###########################################################

		update_dictionary['translated_supervised_target_domain_zs'] = self.translate_latent_z(target_sup_var_dict['latent_z_indices'].detach(), target_sup_var_dict['latent_b'].detach())		

		###########################################################
		# 5) Need to get Z transformations for this rather than the z's themselves.
		###########################################################

		update_dictionary['source_sup_z_transformations'], update_dictionary['source_sup_z_transformation_weights'], _ = self.get_z_transformation(source_sup_var_dict['latent_z_indices'].detach(), source_sup_var_dict['latent_b'].detach())
		update_dictionary['target_sup_z_transformations'], update_dictionary['target_sup_z_transformation_weights'], _ = self.get_z_transformation(update_dictionary['translated_supervised_target_domain_zs'], target_sup_var_dict['latent_b'].detach())

		###########################################################
		# 6) Create forward and backward GMMs. 
		###########################################################

		# Remember, now transposing the supervised_z_transformation objects, because we need this to handle the batches / timesteps of the z sets correctly. 		
		self.supervised_z_tuple_GMM_list = [self.create_GMM(evaluation_domain=0, mean_point_set=update_dictionary['source_sup_z_transformations'].transpose(1,0), differentiable_points=True, tuple_GMM=True), \
									 		self.create_GMM(evaluation_domain=1, mean_point_set=update_dictionary['target_sup_z_transformations'].transpose(1,0), differentiable_points=True, tuple_GMM=True)]

		###########################################################
		# 7) Now implement tuple / set based losses, by querying these GMMs for likelihoods. 
		###########################################################

		# We DON'T actually want to transpose the z transformation objects here, because they have to be the opposite shape as the mean / component distributons in the GMM's..
		# The likelihoods from this are going to be the same shape as the query, i.e. of the cross domain z transformation object.

		# Remember, now we need to implement tuple based version of set loss..
		self.unmasked_forward_supervised_set_tuple_based_logprobabilities = self.query_GMM_density(evaluation_domain=0, point_set=update_dictionary['target_sup_z_transformations'], differentiable_points=True, GMM=self.supervised_z_tuple_GMM_list[0])
		self.unmasked_backward_supervised_set_tuple_based_logprobabilities = self.query_GMM_density(evaluation_domain=1, point_set=update_dictionary['source_sup_z_transformations'], differentiable_points=True, GMM=self.supervised_z_tuple_GMM_list[1])

		self.forward_supervised_set_tuple_based_loss = - (update_dictionary['target_sup_z_transformation_weights']*self.unmasked_forward_supervised_set_tuple_based_logprobabilities).sum()/update_dictionary['target_sup_z_transformation_weights'].sum()
		self.backward_supervised_set_tuple_based_loss = - (update_dictionary['source_sup_z_transformation_weights']*self.unmasked_backward_supervised_set_tuple_based_logprobabilities).sum()/update_dictionary['source_sup_z_transformation_weights'].sum()
	
	def run_iteration(self, counter, i, domain=None, skip_viz=False):

		# Overall algorithm.
		# Preprocessing
		# 1) For N samples of datapoints from the source domain. 
		# 	# 2) Feed these input datapoints into the source domain encoder and get source encoding z. 
		#	# 3) Add Z to Source Z Set. 
		# 4) Build GMM with centers around the N Source Z set Z's.

		# Training. 
		# 0) Setup things like training phases, epsilon values, etc.
		# 1) For E Epochs:
		# 	# 2) For D datapoints:
		#		# 3) Sample x from target domain.
		#		# 4) Feed x into target domain encoder to get z encoding. 
		# 		# 5) Compute overall objective. 
		#			# 5a) Compute Z GMM likelihoods. 
		# 				# 5a1) Compute likelihood of target z encoding under the source domain GMM. 
		# 				# 5a2) Compute likelihood of source z encoding(s) under the target domain GMM. (Backward GMM Loss / IMLE)
		#			# 5b) Compute Z Tuple GMM likelihoods.
		# 			# 5c) Compute supervised loss.
		# 		# 6) Compute gradients of objective and then update networks / policies.

		################################################
		# (0) Setup things like training phases, epislon values, etc.
		################################################

		self.set_iteration(counter, i=i)
		
		if counter==0:
			self.set_z_objects()

		################################################			
		# (3), (4), (5a) Get input datapoint from target domain. One directional in this case.
		################################################

		domain = 1

		# print("Ebmedding in run iter")
		# embed()
		source_input_dict, source_var_dict, source_eval_dict = self.encode_decode_trajectory(self.target_manager, i)
		update_dictionary = {}
		update_dictionary['subpolicy_inputs'], update_dictionary['latent_z'], update_dictionary['loglikelihood'], update_dictionary['kl_divergence'] = \
			source_eval_dict['subpolicy_inputs'], source_var_dict['latent_z_indices'], source_eval_dict['learnt_subpolicy_loglikelihoods'], source_var_dict['kl_divergence']

		self.unsupervised_loss_batch_mask = copy.deepcopy(self.target_manager.batch_mask)
	
		if not(skip_viz):			

			################################################
			# Precursor to 5c - run cross domain encode / decode. Running this in run iteration so we have access to variables.
			################################################

			cross_domain_input_dict, cross_domain_var_dict, cross_domain_eval_dict = self.encode_decode_trajectory(self.source_manager, i)
			update_dictionary['cross_domain_latent_z'] = cross_domain_var_dict['latent_z_indices']
	
			detached_original_latent_z = update_dictionary['latent_z'].detach()
			update_dictionary['translated_latent_z'] = self.translate_latent_z(detached_original_latent_z, source_var_dict['latent_b'].detach())

			################################################
			# 5a) Compute Z GMM likelihoods. 
			################################################

			################################################
			# 5a1) Compute likelihood of target z encoding under the source domain GMM. 
			################################################
			
			# update_dictionary['cross_domain_density_loss'] = self.compute_density_based_loss(update_dictionary)
			# print("RUNNING QGMMD Forward Z Den")
			update_dictionary['forward_density_loss'] = self.query_GMM_density(evaluation_domain=0, point_set=update_dictionary['translated_latent_z'], differentiable_points=True)

			################################################
			# 5a2) Computing likelihood of source z encoding(s) under the target domain GMM. (Backward GMM Loss / IMLE)
			################################################

			# Step 1: Recompute translated means. 
			self.differentiable_mean_computation(counter)
			# Step 2: Recreate target domain GMM (with translated z's as input).						
			self.GMM_list[domain] = self.create_GMM(evaluation_domain=domain, mean_point_set=self.differentiable_target_means, differentiable_points=True)
			# Step 3: Actually query GMM for likelihoods. Remember, this needs to be done differentiably. 
			# print("RUNNING QGMMD Backward Z Den")
			update_dictionary['backward_density_loss'] = self.query_GMM_density(evaluation_domain=domain, point_set=update_dictionary['cross_domain_latent_z'], differentiable_points=True)

			################################################
			# 5b) Compute Z Tuple GMM likelihood.
			################################################	

			if self.args.z_tuple_gmm:

				################################################
				# 5b1) Compute Translated Z Tuples for this iteration.
				################################################

				# Computing tuples in both domains for this particular batch.
				update_dictionary['target_z_transformations'], update_dictionary['target_z_trajectory_weights'], _ = self.get_z_transformation(update_dictionary['translated_latent_z'], source_var_dict['latent_b'])
				update_dictionary['source_z_transformations'], update_dictionary['source_z_trajectory_weights'], _ = self.get_z_transformation(update_dictionary['cross_domain_latent_z'], cross_domain_var_dict['latent_b'])
				# update_dictionary['z_trajectory'] = update_dictionary['z_transformations']

				################################################
				# 5b2) Compute likelihood of target z encoding tuples under the source domain Z Tuple GMM. 			
				################################################
				
				update_dictionary['forward_z_tuple_density_loss'] = self.query_GMM_density(evaluation_domain=domain, point_set=update_dictionary['target_z_transformations'], differentiable_points=True, GMM=self.Z_Tuple_GMM_list[0])

				################################################
				# 5b3) Compute likelihood of target z encoding tuples under the source domain Z Tuple GMM. 			
				################################################

				# Step 1: Recompute translated means. Remember, this is already done by differentiable mean computation.
				# Step 2: Recreate target domain Z Tuple GMM (wiht translated z tuples as input.). 
				self.Z_Tuple_GMM_list[1] = self.create_GMM(evaluation_domain=domain, mean_point_set=self.target_z_tuple_set, differentiable_points=True)
				# Step 3: Actually query GMM for likelihood. 
				# print("RUNNING QGMMD Backward Z Tup Den")
				update_dictionary['backward_z_tuple_density_loss'] = self.query_GMM_density(evaluation_domain=domain, point_set=update_dictionary['source_z_transformations'], differentiable_points=True, GMM=self.Z_Tuple_GMM_list[1])
				
			################################################
			# 5c) Compute supervised loss.
			################################################

			if self.args.task_based_supervision:
				self.compute_task_based_supervision_loss(update_dictionary, i)
				# For deubgz
				# update_dictionary['cross_domain_supervised_loss'] = 0.
			if self.args.cross_domain_supervision:
				self.compute_cross_domain_supervision_loss(update_dictionary)
				# update_dictionary['cross_domain_supervised_loss'] = 0.

			################################################
			# 6) Compute gradients of objective and then update networks / policies.
			################################################
			
			self.update_networks(1, self.target_manager, update_dictionary)					

			################################################
			# 7) Update plots. 
			################################################
			
			viz_dict = {}
			viz_dict['domain'] = domain

			if self.args.supervised_set_based_density_loss:
				viz_dict['forward_set_based_supervised_loss'], viz_dict['backward_set_based_supervised_loss'] = update_dictionary['forward_set_based_supervised_loss'].mean().detach().cpu().numpy(), update_dictionary['backward_set_based_supervised_loss'].mean().detach().cpu().numpy()
			if self.args.z_gmm:
				viz_dict['forward_density_loss'], viz_dict['backward_density_loss'] = update_dictionary['forward_density_loss'].mean().detach().cpu().numpy(), update_dictionary['backward_density_loss'].mean().detach().cpu().numpy()
				viz_dict['weighted_forward_density_loss'], viz_dict['weighted_backward_density_loss'] = self.weighted_forward_loss.mean().detach().cpu().numpy(), self.weighted_backward_loss.mean().detach().cpu().numpy()

			self.update_plots(counter, viz_dict, log=True)

			# print("Embed in RUn ITer")
			# embed()

	def evaluate_semantic_accuracy(self):

		# Get appropriate set of labels. 
		# Build KDTree for efficient nearest neighbor queries? 

		# 1) Get source labels.
		if self.args.source_domain=='MIME':
			if self.args.source_single_hand=='left':
				self.source_domain_z_set = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_zs.npy")
				self.source_domain_z_inverse_labels = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_labels_and_z_indices.npy", allow_pickle=True)
				self.source_domain_labelled_z_indices = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_labelled_z_indices.npy")				
				self.source_domain_z_labels = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_labels.npy", allow_pickle=True)
			elif self.args.source_single_hand=='right':
				self.source_domain_z_set = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_zs.npy")
				self.source_domain_z_inverse_labels = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_labels_and_z_indices.npy", allow_pickle=True)
				self.source_domain_labelled_z_indices = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_labelled_z_indices.npy")
				self.source_domain_z_labels = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_labels.npy", allow_pickle=True)
		elif self.args.source_domain=='Roboturk':			
			self.source_domain_z_set = np.load("Labelled_Zs/Roboturk_RTP_001_me500_zs.npy")
			self.source_domain_z_inverse_labels = np.load("Labelled_Zs/Roboturk_RTP_001_me500_labels_and_z_indices.npy", allow_pickle=True)
			self.source_domain_labelled_z_indices = np.load("Labelled_Zs/Roboturk_RTP_001_me500_labelled_z_indices.npy")
			self.source_domain_z_labels = np.load("Labelled_Zs/Roboturk_RTP_001_me500_labels.npy", allow_pickle=True)

		# 2) Get target labels.
		if self.args.target_domain=='MIME':
			if self.args.target_single_hand=='left':
				self.target_domain_z_set = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_zs.npy")
				self.target_domain_z_inverse_labels = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_labels_and_z_indices.npy", allow_pickle=True)
				self.target_domain_labelled_z_indices = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_labelled_z_indices.npy")
				self.target_domain_z_labels = np.load("Labelled_Zs/MIME_Left_MBP_094_me500_labels.npy", allow_pickle=True)
			elif self.args.target_single_hand=='right':
				self.target_domain_z_set = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_zs.npy")
				self.target_domain_z_inverse_labels = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_labels_and_z_indices.npy", allow_pickle=True)
				self.target_domain_labelled_z_indices = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_labelled_z_indices.npy")
				self.target_domain_z_labels = np.load("Labelled_Zs/MIME_Right_MBP_094_me500_labels.npy", allow_pickle=True)
		elif self.args.target_domain=='Roboturk':			
			self.target_domain_z_set = np.load("Labelled_Zs/Roboturk_RTP_001_me500_zs.npy")
			self.target_domain_z_inverse_labels = np.load("Labelled_Zs/Roboturk_RTP_001_me500_labels_and_z_indices.npy", allow_pickle=True)
			self.target_domain_labelled_z_indices = np.load("Labelled_Zs/Roboturk_RTP_001_me500_labelled_z_indices.npy")
			self.target_domain_z_labels = np.load("Labelled_Zs/Roboturk_RTP_001_me500_labels.npy", allow_pickle=True)			
	
		# Some preprocessing. 
		number_of_labelled_zs = 50
		# Need to select 50 labelled z's in each domain.
		print("######################################")
		print("Embed in semantic label accuracy eval.")
		print("######################################")
		embed()

		self.source_domain_labelled_z_indices = self.source_domain_labelled_z_indices.astype(int)
		self.target_domain_labelled_z_indices = self.target_domain_labelled_z_indices.astype(int)

		self.source_domain_z_set = self.source_domain_z_set[self.source_domain_labelled_z_indices[:number_of_labelled_zs]]
		self.target_domain_z_set = self.target_domain_z_set[self.target_domain_labelled_z_indices[:number_of_labelled_zs]]	

		# 3) Translate TARGET DOMAIN Z's to source domain, because translation models are trained to translate from target to source.
		torch_target_zs = torch.tensor(self.target_domain_z_set).to(device).float()		
		translated_target_to_source_zs = self.backward_translation_model.forward(torch_target_zs).detach().cpu().numpy()

		# 4) Now for 50 labelled z's... check whether the label of the nearest SOURCE Z (to a given translated target z) matches. 
		# 4a) First construct KDTree.
		kdtree = KDTree(self.source_domain_z_set)

		# 4b) Now query KDtree for neighbors.
		distances, nearest_neighbor_indices = kdtree.query(translated_target_to_source_zs)

		# 4c) Nearest neighbor indices indexes into .. self.source_domain_z_set, which is... length 50. 
		# Use self.source_domain_labelled_z_indices to get indices for the label set.. 
		original_index_nearest_neighbors = self.source_domain_labelled_z_indices[nearest_neighbor_indices]
	
		# make sure same number of labels across domains, otherwise measuring accuracy across different sets...

class PolicyManager_JointCycleTransfer(PolicyManager_CycleConsistencyTransfer):

	# Inherit from transfer.
	def __init__(self, args=None, source_dataset=None, target_dataset=None):
			
		# The inherited functions refer to self.args. Also making this to make inheritance go smooth.
		super(PolicyManager_JointCycleTransfer, self).__init__(args, source_dataset, target_dataset)

		self.args = args

		# Before instantiating policy managers of source or target domains; create copies of args with data attribute changed. 		
		self.source_args = copy.deepcopy(args)
		self.source_args.data = self.source_args.source_domain
		self.source_dataset = source_dataset

		self.target_args = copy.deepcopy(args)
		self.target_args.data = self.target_args.target_domain
		self.target_dataset = target_dataset

		# Now create two instances of policy managers for each domain. Call them source and target domain policy managers. 
		self.source_manager = PolicyManager_BatchJoint(number_policies=4, dataset=self.source_dataset, args=self.source_args)
		self.target_manager = PolicyManager_BatchJoint(number_policies=4, dataset=self.target_dataset, args=self.target_args)

		self.source_dataset_size = len(self.source_manager.dataset) - self.source_manager.test_set_size
		self.target_dataset_size = len(self.target_manager.dataset) - self.target_manager.test_set_size

		# Now create variables that we need. 
		self.number_epochs = self.args.epochs
		self.extent = min(self.source_dataset_size, self.target_dataset_size)		

		# Now setup networks for these PolicyManagers. 		
		self.source_manager.setup()
		self.source_manager.initialize_training_batches()
		self.target_manager.setup()
		self.target_manager.initialize_training_batches()

		if self.args.source_subpolicy_model is not None:
			self.source_manager.load_all_models(self.args.source_subpolicy_model, just_subpolicy=True)
		if self.args.target_subpolicy_model is not None:
			self.target_manager.load_all_models(self.args.target_subpolicy_model, just_subpolicy=True)
		if self.args.source_model is not None:
			self.source_manager.load_all_models(self.args.source_model)
		if self.args.target_model is not None:
			self.target_manager.load_all_models(self.args.target_model)


		# Now define other parameters that will be required for the discriminator, etc. 
		self.input_size = self.args.z_dimensions
		self.hidden_size = self.args.hidden_size
		self.output_size = 2
		self.learning_rate = self.args.learning_rate
	
	def save_all_models(self, suffix):
		# super.save_all_models(self, suffix)
		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)

		self.save_object = {}

		# Source
		self.save_object['Source_Policy_Network'] = self.source_manager.policy_network.state_dict()
		self.save_object['Source_Encoder_Network'] = self.source_manager.variational_policy.state_dict()
		# Target
		self.save_object['Target_Policy_Network'] = self.target_manager.policy_network.state_dict()
		self.save_object['Target_Encoder_Network'] = self.target_manager.variational_policy.state_dict()
		# Discriminator
		self.save_object['Discriminator_Network'] = self.discriminator_network.state_dict()				

		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Source
		self.source_manager.policy_network.load_state_dict(self.load_object['Source_Policy_Network'])
		self.source_manager.variational_policy.load_state_dict(self.load_object['Source_Encoder_Network'])
		# Target
		self.target_manager.policy_network.load_state_dict(self.load_object['Target_Policy_Network'])
		self.target_manager.variational_policy.load_state_dict(self.load_object['Target_Encoder_Network'])
		# Discriminator
		self.discriminator_network.load_state_dict(self.load_object['Discriminator_Network'])

	def encode_decode_trajectory(self, policy_manager, i, return_trajectory=False, trajectory_input_dict=None):

		# If we haven't been provided a trajectory input, sample one and run run_iteration as usual.
		if trajectory_input_dict is None:
			# Check if the index is too big. If yes, just sample randomly.
			if i >= len(policy_manager.dataset):
				i = np.random.randint(0, len(policy_manager.dataset))

			# Since the joint training manager nicely lets us get dictionaries, just use it, but remember not to train. 
			# This does all the steps we need.
			source_input_dict, source_var_dict, source_eval_dict = policy_manager.run_iteration(self.counter, i, return_dicts=True, train=False)
		
		# If we have been provided a trajectory input, supply the relelvant items to the policy_managers.run_iteration. 
		else:
			# Relabelling dictionary keys.
			trajectory_input_dict['sample_traj'] = trajectory_input_dict['differentiable_trajectory']
			trajectory_input_dict['sample_action_seq'] = trajectory_input_dict['differentiable_action_seq']
			trajectory_input_dict['concatenated_traj'] = trajectory_input_dict['differentiable_state_action_seq']
			# Must concatenate for variational network input. 
			# Sample_action_seq here already probably is padded with a 0 at the beginning. 
			
			# DEBUGGING hack:
			# trajectory_input_dict['old_concatenated_traj'] = trajectory_input_dict['differentiable_state_action_seq']

			# THIS IS THE RIGHT THING TO DO:
			trajectory_input_dict['old_concatenated_traj'] = policy_manager.differentiable_old_concat_state_action(trajectory_input_dict['sample_traj'], trajectory_input_dict['sample_action_seq'])

			# Now that we've assembled trajectory input dictionary neatly, feed it to policy_manager.run_iteration.
			source_input_dict, source_var_dict, source_eval_dict = policy_manager.run_iteration(self.counter, i, return_dicts=True, train=False, input_dictionary=trajectory_input_dict)

		return source_input_dict, source_var_dict, source_eval_dict

	def cross_domain_decoding(self, domain, domain_manager, latent_z, start_state=None):

		# If start state is none, first get start state, else use the argument. 
		if start_state is None: 
			# Feed the first latent_z in to get the start state.
			# Get state from... first z in sequence., because we only need one start per trajectory / batch element.
			start_state = self.get_start_state(domain, latent_z[0])
		
		# Now rollout in target domain.		
		cross_domain_decoding_dict = {}
		cross_domain_decoding_dict['differentiable_trajectory'], cross_domain_decoding_dict['differentiable_action_seq'], \
			cross_domain_decoding_dict['differentiable_state_action_seq'], cross_domain_decoding_dict['subpolicy_inputs'] = \
			self.differentiable_rollout(domain_manager, start_state, latent_z)
		# differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs = self.differentiable_rollout(domain_manager, start_state, latent_z)

		# return differentiable_trajectory, subpolicy_inputs
		return cross_domain_decoding_dict

	def differentiable_rollout(self, policy_manager, trajectory_start, latent_z, rollout_length=None):

		# Now implementing a differentiable_rollout function that takes in a policy manager.

		# Copying over from cycle_transfer differentiable rollout. 
		# This function should provide rollout template, but needs modifications to deal with multiple z's being used.

		# Remember, the differentiable rollout is required because the backtranslation / cycle-consistency loss needs to be propagated through multiple sets of translations. 
		# Therefore it must pass through the decoder network(s), and through the latent_z's. (It doesn't actually pass through the states / actions?).		

		subpolicy_inputs = torch.zeros((self.args.batch_size,2*policy_manager.state_dim+policy_manager.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[:,:policy_manager.state_dim] = torch.tensor(trajectory_start).to(device).float()
		# subpolicy_inputs[:,2*policy_manager.state_dim:] = torch.tensor(latent_z).to(device).float()
		subpolicy_inputs[:,2*policy_manager.state_dim:] = latent_z[0]

		if self.args.batch_size>1:
			subpolicy_inputs = subpolicy_inputs.unsqueeze(0)

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = policy_manager.rollout_timesteps-1

		for t in range(length):

			# Get actions from the policy.
			actions = policy_manager.policy_network.reparameterized_get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor
			
			# Compute next state. 
			new_state = subpolicy_inputs[t,...,:policy_manager.state_dim]+action_to_execute		

			# Create new input row. 
			input_row = torch.zeros((self.args.batch_size, 2*policy_manager.state_dim+policy_manager.latent_z_dimensionality)).to(device).float()
			input_row[:,:policy_manager.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[:,policy_manager.state_dim:2*policy_manager.state_dim] = actions[-1].squeeze(1)
			input_row[:,2*policy_manager.state_dim:] = latent_z[t+1]

			# Now that we have assembled the new input row, concatenate it along temporal dimension with previous inputs. 
			if self.args.batch_size>1:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row.unsqueeze(0)],dim=0)
			else:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[...,:policy_manager.state_dim].detach().cpu().numpy()
		differentiable_trajectory = subpolicy_inputs[...,:policy_manager.state_dim]
		differentiable_action_seq = subpolicy_inputs[...,policy_manager.state_dim:2*policy_manager.state_dim]
		differentiable_state_action_seq = subpolicy_inputs[...,:2*policy_manager.state_dim]

		# For differentiabiity, return tuple of trajectory, actions, state actions, and subpolicy_inputs. 
		return [differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs]

	def assemble_dictionary(self, dictionary, source_input_dict, source_var_dict, source_eval_dict, \
		target_input_dict, target_var_dict, target_eval_dict):

		# Source elements.
		dictionary['source_subpolicy_inputs_original'] = source_eval_dict['subpolicy_inputs']
		dictionary['source_latent_z'] = source_var_dict['latent_z_indices']
		dictionary['source_loglikelihood'] = source_eval_dict['learnt_subpolicy_loglikelihoods']
		dictionary['source_kl_divergence'] = source_var_dict['kl_divergence']

		# Target elements.
		dictionary['target_subpolicy_inputs_original'] = target_eval_dict['subpolicy_inputs']
		dictionary['target_latent_z'] = target_var_dict['latent_z_indices']
		dictionary['target_loglikelihood'] = target_eval_dict['learnt_subpolicy_loglikelihoods']
		dictionary['target_kl_divergence'] = target_var_dict['kl_divergence']

		return dictionary

	def run_iteration(self, counter, i, skip_viz=False):

		# Phases: 
		# Phase 1:  Train encoder-decoder for both domains initially, so that discriminator is not fed garbage. 
		# Phase 2:  Train encoder, decoder for each domain, and Z discriminator concurrently. 
		# Phase 3:  Train encoder, decoder for each domain, and the individual source and target discriminators, concurrently.

		# Algorithm (joint training): 
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use as source (i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain. 
		#		# 3) Transfer Steps: 
		#	 		# a) Encode trajectory as latent z (domain 1). 
		#			# b) Use domain 2 decoder to decode latent z into trajectory (domain 2).
		#			# c) Use domain 2 encoder to encode trajectory into latent z (domain 2).
		#			# d) Use domain 1 decoder to decode latent z (domain 2) into trajectory (domain 1).
		# 		# 4) Feed cycle-reconstructed trajectory and original trajectory (both domain 1) into discriminator. 
		#		# 5) Train discriminators to predict whether original or cycle reconstructed trajectory. 
		#		# 	 Alternate: Remember, don't actually need to use trajectory level discriminator networks, can just use loglikelihood cycle-reconstruction loss. Try this first.
		#		# 	 Train z discriminator to predict which domain the latentz sample came from. 
		# 		# 	 Train encoder / decoder architectures with mix of reconstruction loss and discriminator confusing objective. 
		# 		# 	 Compute and apply gradient updates. 

		# Remember to make domain agnostic function calls to encode, feed into discriminator, get likelihoods, etc. 

		####################################
		# (0) Setup things like training phases, epislon values, etc.
		####################################

		self.set_iteration(counter)
		self.counter = counter
		dictionary = {}
		target_dict = {}

		####################################
		# (1) Select which domain to use as source domain (also supervision of z discriminator for this iteration). 
		####################################

		dictionary['domain'], source_policy_manager, target_policy_manager = self.get_source_target_domain_managers()

		####################################
		# (2) & (3 a) Get source trajectory and encode into latent z sequence. Decode using source decoder, to get loglikelihood for reconstruction objectve. 
		####################################
		source_input_dict, source_var_dict, source_eval_dict = self.encode_decode_trajectory(source_policy_manager, i)
		# dictionary['source_subpolicy_inputs_original'], dictionary['source_latent_z'], dictionary['source_loglikelihood'], dictionary['source_kl_divergence'] = self.encode_decode_trajectory(source_policy_manager, i)		
		
		####################################
		# (3 b) Cross domain decoding. 
		####################################

		target_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], target_policy_manager, source_var_dict['latent_z_indices'])
		
		####################################
		# (3 c) Cross domain encoding of target_trajectory_rollout into target latent_z. 
		####################################

		# dictionary['target_subpolicy_inputs'], dictionary['target_latent_z'], dictionary['target_loglikelihood'], dictionary['target_kl_divergence']
		target_input_dict, target_var_dict, target_eval_dict = self.encode_decode_trajectory(target_policy_manager, i, trajectory_input_dict=target_cross_domain_decoding_dict)

		####################################
		# (3 d) Cross domain decoding of target_latent_z into source trajectory. 
		# Can use the original start state, or also use the reverse trick for start state. Try both maybe.
		####################################
		source_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], source_policy_manager, target_var_dict['latent_z_indices'], start_state=source_eval_dict['subpolicy_inputs'][0,...,:source_policy_manager.state_dim].detach().cpu().numpy())
		# source_cross_domain_decoding_dict = self.cross_domain_decoding(dictionary['domain'], source_policy_manager, dictionary['target_latent_z'], start_state=dictionary['source_subpolicy_inputs_original'][0,:source_policy_manager.state_dim].detach().cpu().numpy())
		dictionary['source_subpolicy_inputs_crossdomain'] = source_cross_domain_decoding_dict['subpolicy_inputs']

		####################################
		# Parse dictionary.
		####################################

		dictionary = self.assemble_dictionary(dictionary, source_input_dict, source_var_dict, source_eval_dict, \
			target_input_dict, target_var_dict, target_eval_dict)

		####################################
		# (4) Compute all losses, reweight, and take gradient steps.
		####################################

		# Update networks.
		self.update_networks(dictionary, source_policy_manager)

		####################################
		# (5) Accumulate and plot statistics of training.
		####################################
		
		if not(skip_viz):
			self.update_plots(counter, dictionary)

		# Encode decode function: First encodes, takes trajectory segment, and outputs latent z. The latent z is then provided to decoder (along with initial state), and then we get SOURCE domain subpolicy inputs. 
		# Cross domain decoding function: Takes encoded latent z (and start state), and then rolls out with target decoder. Function returns, target trajectory, action sequence, and TARGET domain subpolicy inputs. 

class PolicyManager_IKTrainer(PolicyManager_BaseClass):	

	def __init__(self, dataset=None, args=None):

		#
		super(PolicyManager_IKTrainer, self).__init__()

		self.args = args
		if self.args.data in ['MIME','OldMIME']:
			self.state_size = 16
			self.IK_state_size = 14
			
		if self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
			self.state_size = 8
			self.IK_state_size = 7

		self.conditional_info_size = 1
		self.dataset = dataset
		self.hidden_size = self.args.hidden_size
		self.number_layers = self.args.number_layers
		self.learning_rate = self.args.learning_rate
		self.test_set_size = 100
		self.extent = len(self.dataset) - self.test_set_size

		# self.extent = (self.extent//self.args.batch_size+1)*self.args.batch_size-self.args.batch_size
		self.number_epochs = self.args.epochs

		# Create a Baxter visualizer object to fiddle with and verify things against..
		self.visualizer = BaxterVisualizer(args=self.args)

	def create_networks(self):
		
		# Create IK network
		self.IK_network = ContinuousMLP(self.IK_state_size, self.hidden_size, self.IK_state_size, args=self.args, number_layers=self.number_layers).to(device)

	def create_training_ops(self):

		# Create parameter list. 
		self.parameter_list = self.IK_network.parameters()

		# Create optimizer. 
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate, weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['IK_Network'] = self.IK_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path):

		load_object = torch.load(path)
		self.IK_network.load_state_dict(load_object['IK_Network'])

	# Get batch full trajectory. 
	def get_js_ee_inputs(self, i, get_latents=False, special_indices=None, called_from_train=False):

		# print("# Debug task ID batching")
		# embed()

		# Toy Data
		# if self.args.data in ['MIME','OldMIME'] or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
		if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
			
			if self.args.data in ['MIME','OldMIME'] or self.args.data=='Mocap':

				if self.args.task_discriminability or self.args.task_based_supervision:

					# Don't really need to use digitize, we need to digitize with respect to .. 0, 32, 64, ... 8448. 
					# So.. just use... //32
					bucket = i//32
					data_element = self.dataset[np.array(self.task_based_shuffling_blocks[bucket])]
					self.input_task_id = self.index_task_id_map[bucket]					
				else:
					# data_element = self.dataset[i:i+self.args.batch_size]
					data_element = self.dataset[self.sorted_indices[i:i+self.args.batch_size]]

			else:
				data_element = self.get_batch_element(i)

			# Get trajectory lengths across batch, to be able to create masks for losses. 
			self.batch_trajectory_lengths = np.zeros((self.args.batch_size), dtype=int)
			minl = 10000
			maxl = 0
			

			# if len(data_element)<self.args.batch_size:
			# 	print("Embed in get js ee")
			# 	embed()

			for x in range(self.args.batch_size):

				# Doesn't really depend on EE or not..
				self.batch_trajectory_lengths[x] = data_element[x]['demo'].shape[0]

				maxl = max(maxl,self.batch_trajectory_lengths[x])
				minl = min(minl,self.batch_trajectory_lengths[x])
			# print("For this iteration:",maxl-minl,minl,maxl)

			self.max_batch_traj_length = self.batch_trajectory_lengths.max()

			# Create batch object that stores trajectories. 
			batch_trajectory = np.zeros((self.args.batch_size, self.max_batch_traj_length, self.state_size))
			batch_ee_trajectory = np.zeros((self.args.batch_size, self.max_batch_traj_length, self.state_size))

			# Copy over data elements into batch_trajectory array.
			for x in range(self.args.batch_size):
				batch_ee_trajectory[x,:self.batch_trajectory_lengths[x]] = data_element[x]['endeffector_trajectory']
				batch_trajectory[x,:self.batch_trajectory_lengths[x]] = data_element[x]['demo']
			
			# If normalization is set to some value.
			# if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			# 	batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value

			return batch_trajectory.transpose((1,0,2)), batch_ee_trajectory.transpose((1,0,2))
			
	def eval_IK(self, input_dictionary, update_dictionary):
				
		bs = 32
		np.set_printoptions(precision=2)

		array_errors = np.zeros(bs)
		array_dataset_vs_recon_ee_pose_error = np.zeros(bs)
		array_dataset_vs_IK_ee_pose_error = np.zeros(bs)

		for k in range(bs):
			
			js = input_dictionary['joint_angle_traj'][:,k,:14]
			ee = input_dictionary['end_effector_traj'][:,k,:14]		

			pjs = update_dictionary['predicted_joint_states'].view(js.shape[0],bs,-1).detach().cpu().numpy()
			vpjs = pjs[:,k]

			errors = np.zeros(js.shape[0])
			js_vs_IK_errors = np.zeros(js.shape[0])
			js_vs_seed_errors = np.zeros(js.shape[0])

			dataset_vs_recon_ee_pose_error = np.zeros(js.shape[0])
			dataset_vs_IK_ee_pose_error = np.zeros(js.shape[0])

			for t in range(js.shape[0]):
				js1 = js[t]
				ee1 = ee[t]
				pjs1 = vpjs[t]

				if (js1==0).all():
					continue				

				ee_pose = ee1
				seed = pjs1 
				self.visualizer.baxter_IK_object.controller.sync_ik_robot(seed)

				peep = np.concatenate(self.visualizer.baxter_IK_object.controller.ik_robot_eef_joint_cartesian_pose())

				dataset_vs_recon_ee_pose_error[t] = (abs(peep-ee1)).mean()

				joint_positions = np.array(self.visualizer.baxter_IK_object.controller.inverse_kinematics(
							target_position_right=ee_pose[:3],
							target_orientation_right=ee_pose[3:7],
							target_position_left=ee_pose[7:10],
							target_orientation_left=ee_pose[10:14],
							rest_poses=seed))

				# if k==7 and t==33:
				# print("embedding in eval")
				# embed()

				# errors[t] = (abs(joint_positions-js1)).mean()

				js_vs_IK_errors[t] = (abs(joint_positions-js1)).mean()
				js_vs_seed_errors[t] = (abs(seed-js1)).mean()

				self.visualizer.baxter_IK_object.controller.sync_ik_robot(joint_positions)
				peep2 = np.concatenate(self.visualizer.baxter_IK_object.controller.ik_robot_eef_joint_cartesian_pose())
				
				dataset_vs_IK_ee_pose_error[t] = (abs(peep2-ee1)).mean()

			# print("BI:",k,errors.max(),dataset_vs_recon_ee_pose_error.max(),dataset_vs_IK_ee_pose_error.max())
			array_errors[k] = errors.max()
			array_dataset_vs_recon_ee_pose_error[k] = dataset_vs_recon_ee_pose_error.max()
			array_dataset_vs_IK_ee_pose_error[k] = dataset_vs_IK_ee_pose_error.max()

			# print("BI:", k, array_errors[k], array_dataset_vs_recon_ee_pose_error[k], array_dataset_vs_IK_ee_pose_error[k], js.shape[0])
		# print(array_errors.mean(), array_dataset_vs_recon_ee_pose_error.mean(), array_dataset_vs_IK_ee_pose_error.mean()) 
		print("{:10.2f}".format(array_errors.mean()), "{:10.2f}".format(array_dataset_vs_recon_ee_pose_error.mean()), "{:10.2f}".format(array_dataset_vs_IK_ee_pose_error.mean()))
		

	def run_iteration(self, counter, i, return_z=False, and_train=True):
				
		# Flow: 
		# 
		# 1) Get batch of trajectories. 
		# 2) Convert to batch of joint angles and ee states. 
		# 3) Feed to network 
		# 4) Compute loss and optimize

		#############################################		
		# 1) Get batch of trajectories.
		#############################################
		
		input_dictionary, update_dictionary, log_dict = {}, {}, {}

		# Get batch of joint angle trajectories and ee trajectories.
		input_dictionary['joint_angle_traj'], input_dictionary['end_effector_traj'] = self.get_js_ee_inputs(i, called_from_train=True)

		#############################################		
		# 2) Batch timesteps.
		#############################################

		input_dictionary['joint_angle_states'] = torch.tensor(input_dictionary['joint_angle_traj'][...,:-2].reshape(-1,self.IK_state_size)).to(device).float()
		input_dictionary['end_effector_states'] =  torch.tensor(input_dictionary['end_effector_traj'][...,:-2].reshape(-1,self.IK_state_size)).to(device).float()

		#############################################		
		# 3) Feed EE states to IK network and get predicted joint states.
		#############################################
		
		update_dictionary['predicted_joint_states'] = self.IK_network(input_dictionary['end_effector_states'])

		#############################################
		# 4) Compute loss.
		#############################################		

		# Compute mask..
		self.batch_mask = np.zeros((input_dictionary['joint_angle_traj'].shape[0],input_dictionary['joint_angle_traj'].shape[1]))
		for k in range(self.args.batch_size):
			self.batch_mask[:self.batch_trajectory_lengths[k],k] = 1. 
		self.torch_batch_mask = torch.tensor(self.batch_mask).to(device).float().reshape(-1,1)

		# Actually computing loss, and then masking it.
		self.joint_state_error = (update_dictionary['predicted_joint_states']-input_dictionary['joint_angle_states'])
		self.joint_state_loss = (self.joint_state_error**2)
		self.total_loss = (self.torch_batch_mask*self.joint_state_loss).sum()/self.torch_batch_mask.sum()

		#############################################		
		# 5) Optimize. 
		#############################################		

		self.optimizer.zero_grad()
		self.total_loss.backward()
		self.optimizer.step()

		#############################################
		# 5) Log loss.
		#############################################

		log_dict['Joint State Loss'] = self.joint_state_loss.detach().cpu().numpy()
		log_dict['Total Loss'] = self.total_loss.detach().cpu().numpy()
		log_dict['Absolute Joint State Error'] = abs(self.joint_state_error).detach().cpu().numpy()				
		wandb.log(log_dict, step=counter)

		#############################################
		# Evaluate.. 
		
		self.eval_IK(input_dictionary, update_dictionary)
		#############################################

		if self.args.debug:
			print("Embedding in run iter")
			embed()

class PolicyManager_DownstreamTaskTransfer(PolicyManager_DensityJointFixEmbedTransfer):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		# Should this inherit from DJFE PM? 
		# Or just have an instance of it... 
		# Difference is probably... how easily it will be to interact with self objects of this class. 
		# Still have access to all of the DJFE PM functions / objects either way... 

		# Inheritance probably easier from a creation of PM point of view?

		super(PolicyManager_DownstreamTaskTransfer, self).__init__(args, source_dataset, target_dataset)
		
	def setup(self):
				
		# Run super set up to construct networks, load domain models, etc. 
		super(PolicyManager_DownstreamTaskTransfer, self).setup()

		# print("Embed before model load.")
		# embed()

		# Also load the translation model trained in DJFE PM. 
		self.load_all_models(self.args.model)

		# Now set things up for running PPO. 
		self.setup_ppo()

		print("Finished Setup PPO.")

		# Now create training ops for PPO. 
		self.setup_ppo_training_ops()
		self.artificial_downsample_factor = 1

		print("Done running Downstream Task Setup.")

	def setup_ppo(self, source_or_target='source'):
		
		####################################################
		# This function should essentially replicate what the Run_Robosuite_PPO file does.
		####################################################
		
		if not(self.args.no_mujoco):
			import robosuite
			from robosuite.wrappers import GymWrapper

		####################################################
		# Create the base environment.
		####################################################

		self.source_or_target = source_or_target
		if source_or_target=='source':
			env = self.args.environment
		else:
			env = self.args.target_environment
		self.set_environment(env)

		####################################################
		# Set some parameters.
		####################################################

		self.latent_z_dimension = 16
		self.steps_per_epoch = 1000
		self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
		self.gamma = 0.99
		self.lam = 0.97
		self.clip_ratio = 0.2
		# Increasing target KL
		self.target_kl = 0.05
		self.target_kl = self.args.target_KL
		self.train_v_iters = 80
		self.train_pi_iters = 80
		self.max_ep_len = 1000

		# 
		self.max_ep_len = 500
		self.steps_per_epoch = 500
		self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
			
		# Logging defaults
		# def hierarchical_ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
		# 		steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
		# 		vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
		# 		target_kl=0.01, logger_kwargs=dict(), save_freq=10, args=None):


		####################################################		
		# Create a log directory.
		####################################################

		self.RL_logdir = "Logs/{0}".format(self.args.name+"_"+self.args.environment)

		#################################################### 
		# Create a policy / critic. 
		####################################################

		self.ActorCritic = partial(exercise1_2_auxiliary.ExerciseActorCritic, actor=MLPGaussianActor)

		####################################################
		# Set some parameters.
		####################################################

		self.ppo_obs_dim = self.gym_env.observation_space.shape
		self.ppo_act_dim = self.gym_env.action_space.shape

		####################################################
		# Initialize things needed for PPO, that were in the PPO Init block.
		####################################################		

		# Special function to avoid certain slowdowns from PyTorch + MPI combo.
		setup_pytorch_for_mpi()

		# Set up logger and save configuration		
		# print("Embed before logger creation")
		# embed()
		logger_kwargs = dict(output_dir=self.RL_logdir)
		self.ppo_logger = EpochLogger(**logger_kwargs)
		# self.ppo_logger.save_config(locals())
	
		#####################################################
		# Changing to implementing as an MLPactorcritic but with a few additional functions.. 
		#####################################################
		
		if True:
			latent_z_dimension = 16 
			# Creating a special action space. 
			action_space_bound = np.ones(self.latent_z_dimension)*np.inf
			action_space = Box(-action_space_bound, action_space_bound)

			# If there is an RL model provided, load that instead of instantiating the actor critic... if we're in source env.
			if self.args.RL_model_path is not None and self.source_or_target=='source':
				
				# Feed to load policy and env with return policy set to True.
				_, _, self.actor_critic = load_policy_and_env(self.args.RL_model_path, return_policy=True)
			else:

				# Otherwise instantiate an actor critic.
				self.actor_critic = self.ActorCritic(self.gym_env.observation_space, action_space, **dict(hidden_sizes=(64,)))

		#####################################################
		# Sync params across processes
		#####################################################
		
		sync_params(self.actor_critic)

		# Count variables
		var_counts = tuple(core.count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.v])
		self.ppo_logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

		#####################################################
		# Set up experience buffer
		#####################################################

		# self.local_steps_per_epoch = int(self.steps_per_epoch / num_procs())
		self.ppo_buffer = PPOBuffer(self.ppo_obs_dim, self.ppo_act_dim, self.local_steps_per_epoch, self.gamma, self.lam)
			
		#####################################################
		# Depending on whether we're in source or target domain...
		#####################################################

		#####################################################
		# Instantiate low level policy, and set up joint limits.
		#####################################################

		if self.source_or_target=='source':
			# Instantiate low level policy from the source domain.
			self.lowlevel_policy = self.source_manager.policy_network

			# Set up joint limits and state size.
			self.lower_joint_limits = self.source_manager.norm_sub_value
			# self.upper_joint_limits = self.source_manager.norm_denom_value
			self.joint_limit_range = self.source_manager.norm_denom_value
			self.state_input_size = self.source_manager.input_size

		else:
			# Instantiate low level policy from the source domain.
			self.lowlevel_policy = self.target_manager.policy_network

			# Set up joint limits and state size.
			self.lower_joint_limits = self.target_manager.norm_sub_value
			# self.upper_joint_limits = self.source_manager.norm_denom_value
			self.joint_limit_range = self.target_manager.norm_denom_value
			self.state_input_size = self.target_manager.input_size
						
		# Here, we don't actuallfy need to create this, because the DJFE PM has done this for us. Just reference this appropriately. 
		# Need to verify that it's the source policy we want to reference here..
		
		self.lowlevel_policy.args.batch_size = 1
		self.lowlevel_policy.batch_size = 1

	def setup_ppo_training_ops(self):
		
		#######################################################
		# Set up optimizers for policy and value function
		#######################################################
		
		# CHANGED: Use all parameters.     
		# Can't just use the ac.parameters(), because we need separate optimizers for the latent and low-level policy optimizers. 
		self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.args.rl_policy_learning_rate)

		self.vf_optimizer = Adam(self.actor_critic.v.parameters(), lr=self.args.rl_critic_learning_rate)

		# Set up model saving
		self.ppo_logger.setup_pytorch_saver(self.actor_critic)

		#######################################################
		# Setup epsilon greedy. 
		#######################################################

		self.decay_counter = self.args.epsilon_over
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)

	def compute_loss_pi(self, data):
		
		#######################################################
		# Set up function for computing PPO policy loss
		#######################################################

		# CHANGED TO sampling just z. 
		# Since we don't need a separate function for evaluating batch_logprob, just use ac.pi.
		obs, z_act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
		# print("embedding in loss")
		# embed()
		pi, logp = self.actor_critic.pi(obs, z_act)

		ratio = torch.exp(logp - logp_old)
		clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
		loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

		# Useful extra info
		approx_kl = (logp_old - logp).mean().item()

		# CHANGE: NOTE: This is entropy of the low level policy distribution. Since this is only used for logging and not training, this is fine. 
		ent = pi.entropy().mean().item()
		clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
		clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
		pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

		return loss_pi, pi_info
	
	def compute_loss_v(self, data):

		#######################################################
		# Critic function loss
		#######################################################
				
		obs, ret = data['obs'], data['ret']
		return ((self.actor_critic.v(obs) - ret)**2).mean()

	def update(self):	

		# Copying over update function definition from hierarchical PPO.

		data = self.ppo_buffer.get()

		pi_l_old, pi_info_old = self.compute_loss_pi(data)
		pi_l_old = pi_l_old.item()
		v_l_old = self.compute_loss_v(data).item()

		# Train policy with multiple steps of gradient descent
		for i in range(self.train_pi_iters):
			self.pi_optimizer.zero_grad()
			loss_pi, pi_info = self.compute_loss_pi(data)
			kl = mpi_avg(pi_info['kl'])
			if kl > 1.5 * self.target_kl:
				self.ppo_logger.log('Early stopping at step %d due to reaching max kl.'%i)
				break
			
			loss_pi.backward()
			mpi_avg_grads(self.actor_critic.pi)    # average grads across MPI processes
			self.pi_optimizer.step()

		self.ppo_logger.store(StopIter=i)

		# Value function learning
		for i in range(self.train_v_iters):
			self.vf_optimizer.zero_grad()
			loss_v = self.compute_loss_v(data)
			loss_v.backward()
			mpi_avg_grads(self.actor_critic.v)    # average grads across MPI processes
			self.vf_optimizer.step()

		# Log changes from update
		kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
		self.ppo_logger.store(LossPi=pi_l_old, LossV=v_l_old,
					 KL=kl, Entropy=ent, ClipFrac=cf,
					 DeltaLossPi=(loss_pi.item() - pi_l_old),
					 DeltaLossV=(loss_v.item() - v_l_old))

	def rollout(self, evaluate=False, visualize=False, z_trajectory=None, greedy=False, finetune=False):

		#######################################################
		# Implementing a rollout fucnction. 
		#######################################################

		# 1) Initialize. 
		# 2) While we haven't exceeded timelimit and are still non-terminal:
		#   # 3) Sample z from z policy. 
		#   # 4) While we haven't exceeded skill timelimit and are still non terminal, and haven't exceeded overall timelimit. 
		#       # 5) Sample low-level action a from low-level policy. 
		#       # 6) Step in environment. 
		#       # 7) Increment counters, reset states, log cummulative rewards, etc. 
		#   # 8) Reset episode if necessary.           
	
		#######################################################
		# 1) Initialize / reset. (State is actually already reset here.)
		#######################################################

		if True:	
			t = 0 
			# reset hidden state for incremental policy forward.
			hidden = None
			terminal = False
			o, ep_ret, ep_len = self.gym_env.reset(), 0, 0			
			self.image_list = []

			if self.source_or_target=='source':
				environment = self.args.environment
			else:
				environment = self.args.target_environment

		##########################################
		# 2) While we haven't exceeded timelimit and are still non-terminal:
		##########################################

		while t<self.local_steps_per_epoch and not(terminal) and t<self.eval_time_limit:
			
			##########################################
			# Set Z action
			##########################################			

			if True:
				if finetune:
					if self.args.finetune_method=='FullRL':
						# Samle action from the high level policy. 
						z_action, v, z_logp = self.actor_critic.step(torch.as_tensor(o, dtype=torch.float32), greedy=greedy)
						# Also get the right timestep z from the translated z trajectory.
						translated_z_action = z_trajectory[t//self.artificial_downsample_factor]
					
					elif self.args.finetune_method=='AdaptZ':
						# How do we set z's here? 
						# How do we set policy here? 
						pass
						# One way to set z's here is create a generative model that repeats z's, and feed in these variables as the parameters to the opitmizer, 
						# So that if any one of them are updated all of the repeated z's are updated identically. 
						# The other option is to treat the .. "ADAPTATION" as either a network or an additive modification to the z trajectory
						# Then we need little change to the code? 

				else:
					if z_trajectory is not None:	
						z_action, v, logp = z_trajectory[t//self.artificial_downsample_factor], 1., 1.					
					else:
						z_action, v, z_logp = self.actor_critic.step(torch.as_tensor(o, dtype=torch.float32), greedy=greedy)

			# First reset skill timer. 
			t_skill = 0

			##########################################
			# 4) While we haven't exceeded skill timelimit and are still non terminal, and haven't exceeded overall timelimit. 
			##########################################
			
			while t_skill<self.skill_time_limit and not(terminal) and t<self.local_steps_per_epoch:
									
				##########################################
				# 5) Sample low-level action a from low-level policy. 
				##########################################

				##########################################
				# 5a) Get joint state from observation.				
				##########################################
				
				# 5a) Get joint state from observation.				
				obs_spec = self.gym_env.observation_spec()
				max_gripper_state = 0.042

				if True:
					if float(robosuite.__version__[:3])>1.:
						pure_joint_state = self.gym_env.sim.get_state()[1][:7]						
						
						# if self.args.environment=='Wipe':
						if environment=='Wipe':
							# here, no gripper, so set dummy joint pos
							gripper_state = np.array([max_gripper_state/2])
						else:

							# if args.env_name[:3] == 'Bax':

							# 	# Assembel gripper state from both left and right gripper states. 
							# 	left_gripper_state = np.array([obs_spec['left_gripper_qpos'][0]-obs_spec['left_gripper_qpos'][1]])
							# 	right_gripper_state = np.array([obs_spec['right_gripper_qpos'][0]-obs_spec['right_gripper_qpos'][1]])
							# 	gripper_state = np.concatenate([right_gripper_state, left_gripper_state])

							# else:
							# 	gripper_state = np.array([obs_spec['robot0_gripper_qpos'][0]-obs_spec['robot0_gripper_qpos'][1]])
							gripper_state = np.array([obs_spec['robot0_gripper_qpos'][0]-obs_spec['robot0_gripper_qpos'][1]])
					else:
										
						# if self.args.environment[:3] =='Bax':
						if environment[:3] =='Bax':

							# Assembel gripper state from both left and right gripper states. 
							left_gripper_state = np.array([obs_spec['left_gripper_qpos'][0]-obs_spec['left_gripper_qpos'][1]])
							right_gripper_state = np.array([obs_spec['right_gripper_qpos'][0]-obs_spec['right_gripper_qpos'][1]])
							gripper_state = np.concatenate([right_gripper_state, left_gripper_state])

							# Assemble joint states by flipping left and right hands. 
							pure_joint_state = np.zeros(14)

							# State from environment comes in as... RIGHT, LEFT. 
							# Policy was trained on... LEFT RIGHT. 
							pure_joint_state[:7] = obs_spec['joint_pos'][7:14]
							pure_joint_state[7:14] = obs_spec['joint_pos'][:7]

						else:
							pure_joint_state = obs_spec['joint_pos']
							gripper_state = np.array([obs_spec['gripper_qpos'][0]-obs_spec['gripper_qpos'][1]])
				
				if True:
					# Norm gripper state from 0 to 1
					gripper_state = gripper_state/max_gripper_state
					# Norm gripper state from -1 to 1. 
					gripper_state = 2*gripper_state-1
					joint_state = np.concatenate([pure_joint_state, gripper_state])
					
					# Normalize joint state according to joint limits (minmax normaization).
					normalized_joint_state = (joint_state - self.lower_joint_limits)/self.joint_limit_range

				##########################################
				# 5b) Assemble input. 
				##########################################

				if True:
					if t==0:
						low_level_action_numpy = np.zeros_like(normalized_joint_state)                    					
					assembled_states = np.concatenate([normalized_joint_state,low_level_action_numpy])
					assembled_input = np.concatenate([assembled_states, z_action])
					
					torch_assembled_input = torch.tensor(assembled_input).to(device).float().view(-1,1,self.state_input_size+self.latent_z_dimension)
					# torch_assembled_input = torch.tensor(assembled_input).to(device).float().view(-1,1,self.lowlevel_policy.input_size+self.latent_z_dimension)

					# 5c) Now actually retrieve action.
					low_level_action, hidden = self.lowlevel_policy.incremental_reparam_get_actions(torch_assembled_input, greedy=True, hidden=hidden)
					low_level_action_numpy = low_level_action.detach().squeeze().squeeze().cpu().numpy()

					# 5d) Unnormalize 
					# unnormalized_low_level_action_numpy = *joint_limit_range 
					# UNNORMALIZING ACTIONS! WE'VE NEVER ..DONE THIS BEFORE? 
					# JUST SCALE UP FOR NOW
					unnormalized_low_level_action_numpy = self.args.action_scale_factor * low_level_action_numpy
					# 5d) Normalize action for benefit of environment. 
					# Output of policy is minmax normalized, which is 0-1 range. 
					# Change to -1 to 1 range. 

					# if self.args.environment[:3]=='Bax':
					if environment[:3]=='Bax':
					
						# If we're in a baxter environmnet, flip the left and right hand actions.
						normalized_low_level_action = np.zeros(16)
						normalized_low_level_action[:7] = unnormalized_low_level_action_numpy[7:14]
						normalized_low_level_action[7:14] = unnormalized_low_level_action_numpy[:7]
						# Flip gripper states too..
						
						normalized_low_level_action[14:] = unnormalized_low_level_action_numpy[14:][::-1]
					else:					
						normalized_low_level_action = unnormalized_low_level_action_numpy

				##########################################
				# 6) Step in environment. 
				##########################################

				# if self.args.environment=='Wipe':
				if environment=='Wipe':
					next_o, r, d, _ = self.gym_env.step(normalized_low_level_action[:-1])
				else:
					next_o, r, d, _ = self.gym_env.step(normalized_low_level_action)

				##########################################
				# 6b) If needed, add auxilliary rewards. 
				##########################################	

				if finetune:					
					if self.args.finetune_method in ['AdaptZ','FullRL']:
						# auxilliary_reward = np.linalg.norm((translated_z_action - z_action).detach().cpu().numpy())
						auxilliary_reward = - np.linalg.norm(translated_z_action - z_action)
					# print("Relative rewards:", r, auxilliary_reward,  r + self.args.auxilliary_reward_weight*auxilliary_reward)
					r = r + self.args.auxilliary_reward_weight*auxilliary_reward

				# Logging images				
				if (visualize) and t%10==0:					

					# if float(robosuite.__version__[:3])>1.:
						# self.image_list.append(np.flipud(env.sim.render(600,600,camera_name='agentview')))
					# else:	
					self.image_list.append(np.flipud(self.gym_env.sim.render(600,600,camera_name='vizview1')))

				##########################################
				# 7) Increment counters, reset states, log cummulative rewards, etc. 
				##########################################

				ep_ret += r
				ep_len += 1
				ep_ret_copy, ep_len_copy = copy.deepcopy(ep_ret), copy.deepcopy(ep_len)

				# save and log
				# CHANGED: Saving the action tuple in the buffer instead of just the action..
				# buf.store(o, action_tuple, r, v, logp_tuple)
				# CHANGING TO STORING Z ACTION AND Z LOGP.   
				
				if evaluate==False:
					self.ppo_buffer.store(o, z_action, r, v, z_logp)

				self.ppo_logger.store(VVals=v)
				
				# Update obs (critical!)
				o = next_o

				timeout = ep_len == self.max_ep_len
				terminal = d or timeout
				epoch_ended = t==self.local_steps_per_epoch-1

				# Also adding to the skill time and overall time, since we're in a while loop now.
				t_skill += 1                    
				t+=1

				if terminal or epoch_ended or t>=self.eval_time_limit:

					if epoch_ended and not(terminal):
						print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
					# if trajectory didn't reach terminal state, bootstrap value target
					if timeout or epoch_ended:
						_, v, _ = self.actor_critic.step(torch.as_tensor(o, dtype=torch.float32))
					else:
						v = 0
					self.ppo_buffer.finish_path(v)
					
					if terminal:
						# only save EpRet / EpLen if trajectory finished
						self.ppo_logger.store(EpRet=ep_ret, EpLen=ep_len)
					o, ep_ret, ep_len = self.gym_env.reset(), 0, 0
			
		# if evaluate	
		return ep_ret_copy, ep_len_copy, self.image_list

	def setup_train(self):
		
		# Set global skill time limit.
		self.skill_time_limit = 14

		self.eval_time_limit = 10000000
		self.downsample_freq = 20
		

		# if self.args.evaluate_translated_zs:
		# 	self.args.epochs = 1
		# 	self.image_list = []
		# 	self.batch_index = 0			
		# 	self.translated_zs = np.load(self.args.translated_z_file)[:,self.batch_index]
		# 	# source_variational_dict = np.load(args.source_variational_dict,allow_pickle=True).item()
		# 	# source_latent_bs = source_variational_dict['latent_b'][:,batch_index]
		# 	self.skill_time_limit = 16
		# 	self.eval_time_limit = self.translated_zs.shape[0]*self.downsample_freq
		self.artificial_downsample_factor = 1
		self.skill_time_limit *= self.downsample_freq
		
		self.eval_episodes = 100

	def evaluate_policy(self, eval_episodes=10, suffix=""):

		# Evaluate policy across #self.eval_episodes number of episodes. 
		self.ppo_eval_logger = EpochLogger()

		self.eval_episodes = eval_episodes
		
		for k in range(self.eval_episodes):
			
			# print("Rollout #",epoch,(epoch==0))
			ep_ret, ep_len, image_list = self.rollout(evaluate=True, visualize=True, greedy=True)

			logdir = "Logs/{0}".format(self.args.name+"_"+self.args.environment)

			path = os.path.join(logdir, "Images")
			if not(os.path.isdir(path)):
				os.mkdir(path)

			print("Saving Render!")
			imageio.mimsave(os.path.join(path,"Rollout_{0}_Traj{1}.gif".format(suffix,k)), image_list)
			print("Finished Saving Render!")

			print('Episode %d \t EpRet %.3f \t EpLen %d'%(k, ep_ret, ep_len))

			# # Log info about epoch
			# self.ppo_eval_logger.log_tabular('Epoch', k)
			# self.ppo_eval_logger.log_tabular('EpRet', with_min_and_max=True)
			# self.ppo_eval_logger.log_tabular('EpLen', average_only=True)
			# self.ppo_eval_logger.dump_tabular()

	def set_epsilon(self, counter):

		if counter<self.decay_counter:
			self.epsilon = self.initial_epsilon-self.decay_rate*counter
		else:
			self.epsilon = self.final_epsilon	

	def get_greedy(self):
		
		if np.random.random()<self.epsilon:
			greedy = False
		else:
			greedy = True

	def train_RL(self, model=None, finetune_mode=False, z_trajectory=None):
		
		#######################################################
		# Actual train loop block.
		#######################################################

		start_time = time.time()

		print("#######################################################")
		print("Starting to run training.")
		print("#######################################################")

		if finetune_mode:
			number_epochs = self.args.finetune_epochs
		else:
			number_epochs = self.args.epochs

		for epoch in range(number_epochs):		

			# Rollout. 			
			print("#######################################################")
			print("Runing Epoch #: ",epoch)        
			print("#######################################################")

			print("Running Rollout.")
			
			# self.rollout(visualize=self.args.render)

			# Epsilon greedy. 
			self.set_epsilon(epoch)
			# Get greedy or not. 
			greedy = self.get_greedy()
			
			# Now rollout.
			self.rollout(greedy=greedy, z_trajectory=z_trajectory, finetune=finetune_mode)			
			
			##########################################
			# 8) Save, update, and log. 
			##########################################

			# Save model
			if (epoch % self.args.save_freq == 0) or (epoch == number_epochs-1):
				self.ppo_logger.save_state({'env': self.gym_env}, None)
			
			if finetune_mode:
				eval_freq = self.args.finetune_eval_freq
			else:
				eval_freq = self.args.eval_freq

			if (epoch%eval_freq==0) and (epoch>0):

				print("#######################################################")
				print("About to evaluate policy over 10 episodes.")
				print("#######################################################")

				self.evaluate_policy(eval_episodes=10)
					
			# Perform PPO update if we have enough buffer items. 
			# print("Embed before update.")
			# embed()
			
			# if buf.ptr>=buf.max_size:
			print("Running update.")
			self.update()

			# Log info about epoch
			self.ppo_logger.log_tabular('Epoch', epoch)
			self.ppo_logger.log_tabular('EpRet', with_min_and_max=True)
			self.ppo_logger.log_tabular('EpLen', average_only=True)
			self.ppo_logger.log_tabular('VVals', with_min_and_max=True)
			self.ppo_logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
			self.ppo_logger.log_tabular('LossPi', average_only=True)
			self.ppo_logger.log_tabular('LossV', average_only=True)
			self.ppo_logger.log_tabular('DeltaLossPi', average_only=True)
			self.ppo_logger.log_tabular('DeltaLossV', average_only=True)
			self.ppo_logger.log_tabular('Entropy', average_only=True)
			self.ppo_logger.log_tabular('KL', average_only=True)
			self.ppo_logger.log_tabular('ClipFrac', average_only=True)
			self.ppo_logger.log_tabular('StopIter', average_only=True)
			self.ppo_logger.log_tabular('Time', time.time()-start_time)
			self.ppo_logger.dump_tabular()

		print("#######################################################")
		print("Finished running training.")
		print("#######################################################")

		print("#######################################################")
		print("About to evaluate policy over 10 episodes.")
		print("#######################################################")

		if finetune_mode:
			suffix = "Finetune"
		else:
			suffix = ""

		self.evaluate_policy(eval_episodes=10, suffix=suffix)

	def set_environment(self, env):

		if self.args.environment in ['Door','Wipe'] and float(robosuite.__version__[:3])>1.:        
			# Specify that we're going to use the Sawyer here..
			self.base_env = robosuite.make(env, robots="Sawyer", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True)        
		else:
			self.base_env = robosuite.make(env, has_renderer=False, use_camera_obs=False, reward_shaping=True)			
		
		# Now make a GymWrapped version of that environment.
		self.gym_env = GymWrapper(self.base_env)

	# def train(self, model=None):
	def evaluate_alignment(self):

		#################################################
		# Assume we've trained an alignment model.
		#################################################

		#################################################
		# Construct z trajectory set in source domain.
		#################################################

		print("#################################################")
		print("About to evaluate alignment!")
		print("#################################################")

		self.z_trajectory_set = []
		self.translated_z_trajectory_set = []
		self.orig_ep_returns = []
		self.translated_z_ep_returns = []
		self.eval_episodes = 10
		for k in range(self.eval_episodes):
					
			# 1) Get a z trajectory from a rollout.
			viz = (self.args.viz_latent_rollout)
			orig_ep_return, _, image_list = self.rollout(visualize=viz, greedy=True)

			# Whether we are visualizing this trajectory. 
			if viz:
				path = os.path.join(self.RL_logdir, "Images")
				if not(os.path.isdir(path)):
					os.mkdir(path)

				imageio.mimsave(os.path.join(path,"Traj{0}_Source_Rollout.gif".format(k)), image_list)

			self.orig_ep_returns.append(orig_ep_return)
		
			# 2) Get all data from PPO buffer. 		
			data = self.ppo_buffer.get()

			# 3) Get z trajectory from the data.
			z_traj = data['act']

			# 4) Add z trajectory to set.
			self.z_trajectory_set.append(z_traj)					

		# 4) Reset PPO policy, environment etc. with  target environment.
		self.setup_ppo(source_or_target='target')
		self.setup_ppo_training_ops()
		
		# print("Embed before translated rollouts")
		# embed()

		#################################################
		# Now translate and evaluate all of the trajectories in the z trajectory set.
		#################################################		
		
		for k in range(self.eval_episodes):

			# 5) Translate the source z trajectory. 
		
			# Retrieve z traj. 
			orig_z_trajectory = self.z_trajectory_set[k]
			# Torch. 
			torch_orig_z_trajectory = torch.tensor(orig_z_trajectory).to(device).float()			
			# Translate.			
			translated_z_trajectory = self.backward_translation_model(torch_orig_z_trajectory, greedy=True).detach().cpu().numpy()
			# Add to list. 
			self.translated_z_trajectory_set.append(translated_z_trajectory)

			# 6) Evaluate the original and translated z trajectory in the target domain.
			viz = self.args.viz_latent_rollout
			source_z_ep_ret, _, source_z_traj_image_list = self.rollout(z_trajectory=orig_z_trajectory, evaluate=True, visualize=viz, greedy=True)

			translated_ep_ret, _, translated_image_list = self.rollout(z_trajectory=translated_z_trajectory, evaluate=True, visualize=viz, greedy=True)
			self.translated_z_ep_returns.append(translated_ep_ret)

			# IF we are visualizing..
			if viz:
				imageio.mimsave(os.path.join(path,"Traj{0}_Target_Rollout_source_zs.gif".format(k)), source_z_traj_image_list)
				imageio.mimsave(os.path.join(path,"Traj{0}_Target_Rollout_translated_zs.gif".format(k)), translated_image_list)

			print("Episode: ",k," Original Return: %3.2f"%self.orig_ep_returns[k], " Source Z Return: %3.2f"%source_z_ep_ret, " Translated Z Return: %3.2f"%translated_ep_ret)

	def finetune_target_RL(self):

		print("##########################################")
		print("Finetuning RL in Target Domain.")
		print("##########################################")

		# Fine-tune for each trajectory in trajectory set... 
		# for k in range(self.eval_episodes):
		for k in range(1):
			
			# Are we training a target policy? 
			# Or adapting Z's? 

			######################################
			# 1) Recreate training ops. 
			######################################

			# Depending on whether we are training a target policy or adapting z's, create training ops appropraitely. 
			# Resetting policy, and then recreating training ops, so that finetuning on one z traj isn't affected by the others. 
			
			if self.args.finetune_method == 'AdaptZ':
				pass
			if self.args.finetune_method == 'FullRL':
				
				action_space_bound = np.ones(self.latent_z_dimension)*np.inf
				action_space = Box(-action_space_bound, action_space_bound)

				# First recreate policy and value function by recreating the actor critic.
				self.actor_critic = self.ActorCritic(self.gym_env.observation_space, action_space, **dict(hidden_sizes=(64,)))

				# Next, recreate training ops. # Resets optimizers.
				self.setup_ppo_training_ops()

			######################################
			# 2) Actually run finetuning. 
			######################################

			self.train_RL(finetune_mode=True, z_trajectory=self.translated_z_trajectory_set[k])
						
			# ######################################
			# # 3) Evaluate finetuned policy.
			# ######################################
			
			# self.evaluate_policy(eval_episodes=10)

		pass
		
	def train(self, model=None):

		# Setup training either way. 
		self.setup_train()

		# Actually train RL.
		self.train_RL(model=model)
		
		# Evaluate alignment over x episodes. 		
		self.evaluate_alignment()

		if self.args.finetune_method is not None:
			# Actually train alignment. 
			self.finetune_target_RL()

	# Here's how we're going to get prior from same tasks... 
	# Get high performing z traj from high level policy on source domain.. 
	# Translate
	# Feed z's.

	def evaluate(self, model=None):

		# Setup training either way. 
		self.setup_train()

		if self.args.debug_RL:
			print("Embedding in evaluate function.")
			embed()

		# Evaluate alignment over x episodes. 
		self.evaluate_alignment()
		
		


