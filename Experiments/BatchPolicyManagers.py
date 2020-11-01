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

    def get_batch_trajectory_segment(self, i):
	
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

	def run_iteration(self, counter, i, return_z=False, and_train=True):

		# Basic Training Algorithm: 
		# For E epochs:
		# 	# For all trajectories:
		#		# Sample trajectory segment from dataset. 
		# 		# Encode trajectory segment into latent z. 
		# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		# 		# Update parameters. 

		self.set_epoch(counter)

		############# (0) #############
		# Sample trajectory segment from dataset. 			
		if self.args.traj_segments:			
			trajectory_segment, sample_action_seq, sample_traj  = self.get_trajectory_segment(i)
		else:
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)				
			# Calling it trajectory segment, but it's not actually a trajectory segment here.
			trajectory_segment = concatenated_traj

		if trajectory_segment is not None:
			############# (1) #############
			torch_traj_seg = torch.tensor(trajectory_segment).to(device).float()
			# Encode trajectory segment into latent z. 		
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon)

			embed()

			########## (2) & (3) ##########
			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

			_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(trajectory_segment, latent_z_seq, latent_b, sample_action_seq)

			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
			loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq)
			loglikelihood = loglikelihoods[:-1].mean()
			 
			if self.args.debug:
				print("Embedding in Train.")
				embed()

			############# (3) #############
			# Update parameters. 
			if self.args.train and and_train:

				# If we are regularizing: 
				# 	(1) Sample another z. 
				# 	(2) Construct inputs and such.
				# 	(3) Compute distances, and feed to update_policies.
				regularization_kl = None
				z_distance = None

				self.update_policies_reparam(loglikelihood, subpolicy_inputs, kl_divergence)

				# Update Plots. 
				self.update_plots(counter, loglikelihood, trajectory_segment)

				if return_z: 
					return latent_z, sample_traj, sample_action_seq

			else:

				if return_z: 
					return latent_z, sample_traj, sample_action_seq
				else:
					np.set_printoptions(suppress=True,precision=2)
					print("###################", i)
					print("Policy loglikelihood:", loglikelihood)
			
			print("#########################################")	
		else: 
			return None, None, None


