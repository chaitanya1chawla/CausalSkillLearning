# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from headers import *
from PolicyNetworks import *
from Visualizers import BaxterVisualizer, SawyerVisualizer, ToyDataVisualizer #, MocapVisualizer
import TFLogger, DMP, RLUtils
from PolicyManagers import *

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class PolicyManager_JointTransfer(PolicyManager_CycleConsistencyTransfer):

	# Inherit from transfer.
	def __init__(self, args=None, source_dataset=None, target_dataset=None):
				
		super(PolicyManager_JointTransfer, self).__init__(args, source_dataset, target_dataset)

	def encode_decode_trajectory(self, policy_manager, i, return_trajectory=False, trajectory_input_dict=None):

		# If we haven't been provided a trajectory input, sample one and run run_iteration as usual.
		if trajectory_input_dict is None:
			# Check if the index is too big. If yes, just sample randomly.
			if i >= len(policy_manager.dataset):
				i = np.random.randint(0, len(policy_manager.dataset))

			# Since the joint training manager nicely lets us get dictionaries, just use it, but remember not to train. 
			# This does all the steps we need.
			source_input_dict, source_var_dict, source_eval_dict = policy_manager.run_iteration(counter, i, return_dicts=True, train=False)
		
		# If we have been provided a trajectory input, supply the relelvant items to the policy_managers.run_iteration. 
		else:
			# Relabelling dictionary keys.
			trajectory_input_dict['sample_traj'] = trajectory_input_dict['differentiable_trajectory']
			trajectory_input_dict['sample_action_seq'] = trajectory_input_dict['differentiable_action_seq']
			trajectory_input_dict['concatenated_traj'] = trajectory_input_dict['differentiable_state_action_seq']
			# Must concatenate for variational network input. 
			# Sample_action_seq here already probably is padded with a 0 at the beginning. 
			trajectory_input_dict['old_concatenated_traj'] = policy_manager.differentiable_old_concat_state_action(trajectory_input_dict['sample_traj'], trajectory_input_dict['sample_action_seq'])

			# Now that we've assembled trajectory input dictionary neatly, feed it to policy_manager.run_iteration.
			source_input_dict, source_var_dict, source_eval_dict = policy_manager.run_iteration(counter, i, return_dicts=True, train=False, input_dictionary=trajectory_input_dict)

		return source_input_dict, source_var_dict, source_eval_dict

	def cross_domain_decoding(self, domain, domain_manager, latent_z, start_state=None):

		# If start state is none, first get start state, else use the argument. 
		if start_state is None: 
			# Feed the first latent_z in to get the start state.
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

	def run_iteration(self, counter, i):

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
		target_input_dict, target_var_dict, target_eval_dict = self.encode_decode_trajectory(target_policy_manager, i, trajectory_input=target_cross_domain_decoding_dict)

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
		
		self.update_plots(counter, dictionary)

		# Encode decode function: First encodes, takes trajectory segment, and outputs latent z. The latent z is then provided to decoder (along with initial state), and then we get SOURCE domain subpolicy inputs. 
		# Cross domain decoding function: Takes encoded latent z (and start state), and then rolls out with target decoder. Function returns, target trajectory, action sequence, and TARGET domain subpolicy inputs. 
