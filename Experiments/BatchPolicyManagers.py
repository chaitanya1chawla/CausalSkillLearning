# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from headers import *
from PolicyNetworks import *
from Visualizers import BaxterVisualizer, SawyerVisualizer, ToyDataVisualizer #, MocapVisualizer
import TFLogger, DMP, RLUtils
from PolicyManagers import PolicyManager_BaseClass, PolicyManager_Pretrain, PolicyManager_Joint

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class BatchPolicyManager_Pretrain(PolicyManager_Pretrain)

	def __init__(self):
		super(BatchPolicyManager_Pretrain, self).__init__()

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

    def get_trajectory_segment(self, i):
	
    	if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':

			# Sample trajectory segment from dataset. 
			sample_traj, sample_action_seq = self.dataset[i:i+self.args.batch_size]

			# Subsample trajectory segment. 		
			start_timepoint = np.random.randint(0,self.args.traj_length-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[:, start_timepoint:end_timepoint]	
			sample_action_seq = sample_action_seq[:, start_timepoint:end_timepoint-1]

			self.current_traj_len = self.traj_length

			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

			return concatenated_traj, sample_action_seq, sample_traj
		
		elif self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			data_element = self.dataset[i:i+self.args.batch_size]

            # Must select common trajectory segment length for batch.
            # Must select different start / end points for items of batch?

			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length            
                        
            batch_trajectory = np.zeros((self.args.batch_size, self.current_traj_len, self.state_size))

            for x in range(self.args.batch_size):
                
                # Select the trajectory for each instance in the batch. 
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

                    batch_trajectory[x] = data_element[x]['demo'][start_timepoint:end_timepoint]
                    if not(self.args.gripper):
                    	batch_trajectory[x] = data_element['demo'][start_timepoint:end_timepoint,:-1]

            # If normalization is set to some value.
            if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
                batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value

            # Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)

			# Concatenate
			concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)

            # Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			return concatenated_traj, scaled_action_sequence, batch_trajectory

	def collect_inputs(self, i, get_latents=False):

		if self.args.data=='DeterGoal':

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
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)
			old_concatenated_traj = self.old_concat_state_action(sample_traj, sample_action_seq)
		
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
	
		elif self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			# If we're imitating... select demonstrations from the particular task.
			if self.args.setting=='imitation' and self.args.data=='Roboturk':
				data_element = self.dataset.get_task_demo(self.demo_task_index, i)
			else:
				data_element = self.dataset[i]

			if not(data_element['is_valid']):
				return None, None, None, None							

			trajectory = data_element['demo']

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

			action_sequence = np.diff(trajectory,axis=0)

			self.current_traj_len = len(trajectory)

			if self.args.data=='MIME':
				self.conditional_information = np.zeros((self.conditional_info_size))				
			elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
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
			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)
			old_concatenated_traj = self.old_concat_state_action(trajectory, action_sequence)

			if self.args.setting=='imitation':
				action_sequence = RLUtils.resample(data_element['demonstrated_actions'],len(trajectory))
				concatenated_traj = np.concatenate([trajectory, action_sequence],axis=1)

			return trajectory, action_sequence, concatenated_traj, old_concatenated_traj

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

        if not(self.args.discrete_z):
			# # Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			# assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1)).to(device)
			# assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()			

			# assembled_inputs[range(1,len(input_trajectory)),self.input_size:-1] = latent_z_indices[:-1]
			# assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
            
            # Create subpolicy inputs tensor. 
			# subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
            subpolicy_inputs = torch.zeros((len(input_trajectory), self.args.batch_size, self.input_size+self.latent_z_dimensionality)).to(device)

            # Now copy over trajectory. 
			# subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()         
            subpolicy_inputs[:,:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.args.batch_size,self.input_size).to(device).float()

            # Now copy over latent z's. 
			subpolicy_inputs[range(len(input_trajectory)),:,self.input_size:] = latent_z_indices

			# # Concatenated action sequence for policy network's forward / logprobabilities function. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
            # View time first and batch second for downstream LSTM.
            sample_action_seq = sample_action_seq.transpose((1,0,2))
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.args.batch_size,self.output_size))],axis=0)

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

	# def run_iteration(self, counter, i, return_z=False, and_train=True):

	# 	# Basic Training Algorithm: 
	# 	# For E epochs:
	# 	# 	# For all trajectories:
	# 	#		# Sample trajectory segment from dataset. 
	# 	# 		# Encode trajectory segment into latent z. 
	# 	# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
	# 	# 		# Update parameters. 

	# 	self.set_epoch(counter)

	# 	############# (0) #############
	# 	# Sample trajectory segment from dataset. 			
	# 	if self.args.traj_segments:			
	# 		trajectory_segment, sample_action_seq, sample_traj  = self.get_trajectory_segment(i)
	# 	else:
	# 		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)				
	# 		# Calling it trajectory segment, but it's not actually a trajectory segment here.
	# 		trajectory_segment = concatenated_traj

	# 	if trajectory_segment is not None:
	# 		############# (1) #############
	# 		torch_traj_seg = torch.tensor(trajectory_segment).to(device).float()
	# 		# Encode trajectory segment into latent z. 		
	# 		latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon)

	# 		########## (2) & (3) ##########
	# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
	# 		latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

	# 		_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(trajectory_segment, latent_z_seq, latent_b, sample_action_seq)

	# 		# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
	# 		loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq)

    #         embed()
            
	# 		loglikelihood = loglikelihoods[:-1].mean()
			 
	# 		if self.args.debug:
	# 			print("Embedding in Train.")
	# 			embed()

	# 		############# (3) #############
	# 		# Update parameters. 
	# 		if self.args.train and and_train:

	# 			# If we are regularizing: 
	# 			# 	(1) Sample another z. 
	# 			# 	(2) Construct inputs and such.
	# 			# 	(3) Compute distances, and feed to update_policies.
	# 			regularization_kl = None
	# 			z_distance = None

	# 			self.update_policies_reparam(loglikelihood, subpolicy_inputs, kl_divergence)

	# 			# Update Plots. 
	# 			self.update_plots(counter, loglikelihood, trajectory_segment)

	# 			if return_z: 
	# 				return latent_z, sample_traj, sample_action_seq

	# 		else:

	# 			if return_z: 
	# 				return latent_z, sample_traj, sample_action_seq
	# 			else:
	# 				np.set_printoptions(suppress=True,precision=2)
	# 				print("###################", i)
	# 				print("Policy loglikelihood:", loglikelihood)
			
	# 		print("#########################################")	
	# 	else: 
	# 		return None, None, None


