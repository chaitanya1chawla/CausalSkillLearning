# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from headers import *
import DataLoaders, MIME_DataLoader, Roboturk_DataLoader, Mocap_DataLoader
from PolicyManagers import *
import TestClass

def return_dataset(args, data=None, create_dataset_variation=False):
	
	# The data parameter overrides the data in args.data. 
	# This is so that we can call return_dataset with source and target data for transfer setting.
	if data is not None:
		args.data = data

	# Define Data Loader.
	if args.data=='ContinuousNonZero':
		dataset = DataLoaders.ContinuousNonZeroToyDataset(args.datadir, create_dataset_variation=create_dataset_variation)
	elif args.data=='DeterGoal':
		dataset = DataLoaders.DeterministicGoalDirectedDataset(args.datadir)			
	elif args.data=='DirContNonZero':
		dataset = DataLoaders.ContinuousDirectedNonZeroToyDataset(args.datadir)
	elif args.data=='ToyContext':
		dataset = DataLoaders.ToyContextDataset(args.datadir)
	elif args.data=='OldMIME':
		dataset = MIME_DataLoader.MIME_NewDataset(short_traj=args.short_trajectories)
	elif args.data=='MIME':
		dataset = MIME_DataLoader.MIME_NewMetaDataset(short_traj=args.short_trajectories)
	elif args.data=='Roboturk':		
		dataset = Roboturk_DataLoader.Roboturk_NewSegmentedDataset(args)
	elif args.data=='OrigRoboturk':
		dataset = Roboturk_DataLoader.Roboturk_Dataset(args)
	elif args.data=='FullRoboturk':
		dataset = Roboturk_DataLoader.Roboturk_FullDataset(args)
	elif args.data=='Mocap':
		dataset = Mocap_DataLoader.Mocap_Dataset(args)

	return dataset

class Master():

	def __init__(self, arguments):
		self.args = arguments 
		
		self.dataset = return_dataset(self.args, create_dataset_variation=self.args.dataset_variation)

		# Now define policy manager.
		if self.args.setting=='learntsub' or self.args.setting=='joint':
			# self.policy_manager = PolicyManager_BatchJoint(self.args.number_policies, self.dataset, self.args)
			if self.args.batch_size > 1: 
				self.policy_manager = PolicyManager_BatchJoint(self.args.number_policies, self.dataset, self.args)
			else:
				self.policy_manager = PolicyManager_Joint(self.args.number_policies, self.dataset, self.args)

		elif self.args.setting=='context':
			# Assume we're going to run with batch size > 1. 
			self.policy_manager = PolicyManager_BatchJoint(self.args.number_policies, self.dataset, self.args)

		elif self.args.setting=='pretrain_sub':
			if self.args.batch_size > 1: # Only setting batch manager for training.
				self.policy_manager = PolicyManager_BatchPretrain(self.args.number_policies, self.dataset, self.args)
			else:
				self.policy_manager = PolicyManager_Pretrain(self.args.number_policies, self.dataset, self.args)

		elif self.args.setting=='baselineRL':
			self.policy_manager = PolicyManager_BaselineRL(args=self.args)

		elif self.args.setting=='downstreamRL':
			self.policy_manager = PolicyManager_DownstreamRL(args=self.args)

		elif self.args.setting=='DMP':			
			self.policy_manager = PolicyManager_DMPBaselines(self.args.number_policies, self.dataset, self.args)

		elif self.args.setting=='imitation':
			self.policy_manager = PolicyManager_Imitation(self.args.number_policies, self.dataset, self.args)

		elif self.args.setting in ['transfer','cycle_transfer','fixembed','jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle']:
		
			source_dataset = return_dataset(self.args, data=self.args.source_domain)
			target_dataset = return_dataset(self.args, data=self.args.target_domain)
		
			# If we're creating a variation in the dataset: 
			if self.args.dataset_variation:
				target_dataset = return_dataset(self.args, data=self.args.target_domain, create_dataset_variation=create_dataset_variation)

			if self.args.setting=='transfer':
				self.policy_manager = PolicyManager_Transfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)
			elif self.args.setting=='cycle_transfer':
				self.policy_manager = PolicyManager_CycleConsistencyTransfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)				
			elif self.args.setting=='fixembed':
				self.policy_manager = PolicyManager_FixEmbedCycleConTransfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)
			elif self.args.setting=='jointfixembed':
				self.policy_manager = PolicyManager_JointFixEmbedTransfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)
			elif self.args.setting=='jointfixcycle':
				self.policy_manager = PolicyManager_JointFixEmbedCycleTransfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)
			elif self.args.setting=='jointtransfer':
				self.policy_manager = PolicyManager_JointTransfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)
			elif self.args.setting=='jointcycletransfer':
				self.policy_manager = PolicyManager_JointCycleTransfer(args=self.args, source_dataset=source_dataset, target_dataset=target_dataset)

		if self.args.debug:
			print("Embedding in Master.")
			embed()
			
		# Create networks and training operations. 
		self.policy_manager.setup()

	def run(self):
		if self.args.setting in ['pretrain_sub','pretrain_prior','imitation','baselineRL','downstreamRL',\
			'transfer','cycle_transfer','jointtransfer','fixembed','jointcycletransfer', 'jointfixembed', 'jointfixcycle']:
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					self.policy_manager.train()
			else:				
				if self.args.setting=='pretrain_prior':
					self.policy_manager.train(self.args.model)
				else:														
					self.policy_manager.evaluate(model=self.args.model)		
				
		# elif self.args.setting=='learntsub' or self.args.setting=='joint' or self.args.setting=='context':
		elif self.args.setting in ['learntsub','joint','context']:
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					if self.args.subpolicy_model:
						print("Just loading subpolicies.")
						self.policy_manager.load_all_models(self.args.subpolicy_model, just_subpolicy=True)
					self.policy_manager.train()
			else:
				# self.policy_manager.train(self.args.model)
				self.policy_manager.evaluate(self.args.model)

		# elif self.args.setting=='baselineRL' or self.args.setting=='downstreamRL':
		# 	if self.args.train:
		# 		if self.args.model:
		
		# 			self.policy_manager.train(self.args.model)
		# 		else:
		# 			self.policy_manager.train()

		elif self.args.setting=='DMP':
			self.policy_manager.evaluate_across_testset()

	def test(self):
		if self.args.test_code:
			loader = TestClass.TestLoaderWithKwargs()
			suite = loader.loadTestsFromTestCase(TestClass.MetaTestClass, policy_manager=self.policy_manager)
			unittest.TextTestRunner().run(suite)

def parse_arguments():
	parser = argparse.ArgumentParser(description='Learning Skills from Demonstrations')

	# Setup training. 
	parser.add_argument('--datadir', dest='datadir',type=str,default='../../Data/Datasets/ContData/')
	parser.add_argument('--train',dest='train',type=int,default=0)
	parser.add_argument('--debug',dest='debug',type=int,default=0)
	parser.add_argument('--notes',dest='notes',type=str)
	parser.add_argument('--name',dest='name',type=str,default=None)
	parser.add_argument('--fake_batch_size',dest='fake_batch_size',type=int,default=1)
	parser.add_argument('--batch_size',dest='batch_size',type=int,default=32)
	parser.add_argument('--training_phase_size',dest='training_phase_size',type=int,default=500000)
	parser.add_argument('--initial_counter_value',dest='initial_counter_value',type=int,default=0)
	parser.add_argument('--data',dest='data',type=str,default='Continuous')
	parser.add_argument('--setting',dest='setting',type=str,default='gtsub')
	parser.add_argument('--test_code',dest='test_code',type=int,default=0)
	parser.add_argument('--model',dest='model',type=str)
	# parser.add_argument('--logdir',dest='logdir',type=str,default='Experiment_Logs/')
	parser.add_argument('--logdir',dest='logdir',type=str,default='ExpWandbLogs/')
	parser.add_argument('--epochs',dest='epochs',type=int,default=500) # Number of epochs to train for. Reduce for Mocap.
	parser.add_argument('--debugging_datapoints',dest='debugging_datapoints',type=int,default=-1,help='How many data points to run training on. If greater than 0, only select that many datapoints for debugging.')
	parser.add_argument('--seed',dest='seed',type=int,default=0,help='Seed value to initialize random processes.')

	# Training setting. 
	parser.add_argument('--discrete_z',dest='discrete_z',type=int,default=0)
	# parser.add_argument('--transformer',dest='transformer',type=int,default=0)	
	parser.add_argument('--z_dimensions',dest='z_dimensions',type=int,default=64)
	parser.add_argument('--number_layers',dest='number_layers',type=int,default=5)
	parser.add_argument('--hidden_size',dest='hidden_size',type=int,default=64)
	parser.add_argument('--var_number_layers',dest='var_number_layers',type=int,default=5)
	parser.add_argument('--var_hidden_size',dest='var_hidden_size',type=int,default=64)
	parser.add_argument('--dropout',dest='dropout',type=float,default=0.,help='Whether to set dropout.') 
	parser.add_argument('--mlp_dropout',dest='mlp_dropout',type=float,default=0.,help='Whether to set dropout.') 
	parser.add_argument('--environment',dest='environment',type=str,default='SawyerLift') # Defines robosuite environment for RL.
	
	# Data parameters. 
	parser.add_argument('--traj_segments',dest='traj_segments',type=int,default=1) # Defines whether to use trajectory segments for pretraining or entire trajectories. Useful for baseline implementation.
	parser.add_argument('--gripper',dest='gripper',type=int,default=1) # Whether to use gripper training in roboturk.
	parser.add_argument('--ds_freq',dest='ds_freq',type=int,default=1) # Additional downsample frequency.
	parser.add_argument('--condition_size',dest='condition_size',type=int,default=4)
	parser.add_argument('--smoothen', dest='smoothen',type=int,default=0) # Whether to smoothen the original dataset. 
	parser.add_argument('--smoothing_kernel_bandwidth', dest='smoothing_kernel_bandwidth',type=float,default=3.5) # The smoothing bandwidth that is applied to data loader trajectories. 

	# Training paradigm parameters. 
	parser.add_argument('--new_gradient',dest='new_gradient',type=int,default=1)
	parser.add_argument('--b_prior',dest='b_prior',type=int,default=1)
	parser.add_argument('--constrained_b_prior',dest='constrained_b_prior',type=int,default=1) # Whether to use constrained b prior var network or just normal b prior one.
	parser.add_argument('--reparam',dest='reparam',type=int,default=1)	
	parser.add_argument('--number_policies',dest='number_policies',type=int,default=4)
	parser.add_argument('--fix_subpolicy',dest='fix_subpolicy',type=int,default=1)
	parser.add_argument('--reset_training',dest='reset_training',type=int,default=0,help='Whether to reset subpolicy training for joint training, used mostly to learn contextual representations.')
	parser.add_argument('--train_only_policy',dest='train_only_policy',type=int,default=0) # Train only the policy network and use a pretrained encoder. This is weird but whatever. 
	parser.add_argument('--load_latent',dest='load_latent',type=int,default=1) # Whether to load latent policy from model or not.
	parser.add_argument('--subpolicy_model',dest='subpolicy_model',type=str)
	parser.add_argument('--traj_length',dest='traj_length',type=int,default=10)
	parser.add_argument('--short_trajectories',dest='short_trajectories',type=int,default=0,help='Whether to restrict training to short trajectories, to massively save GPU memory.')
	parser.add_argument('--skill_length',dest='skill_length',type=int,default=5)
	parser.add_argument('--var_skill_length',dest='var_skill_length',type=int,default=1)

	# Parameters for evaluation. 
	parser.add_argument('--display_freq',dest='display_freq',type=int,default=10000)
	parser.add_argument('--save_freq',dest='save_freq',type=int,default=5)	
	parser.add_argument('--eval_freq',dest='eval_freq',type=int,default=20)	
	parser.add_argument('--metric_eval_freq',dest='metric_eval_freq',type=int,default=10000)	
	parser.add_argument('--perplexity',dest='perplexity',type=float,default=30,help='Value of perplexity fed to TSNE.')
	parser.add_argument('--latent_set_file_path',dest='latent_set_file_path',type=str,help='File path to pre-computed latent sets to visualize.')
	parser.add_argument('--viz_latent_rollout',dest='viz_latent_rollout',type=int,default=0,help='Whether to visualize latent rollout or not.')

	parser.add_argument('--entropy',dest='entropy',type=int,default=0)
	parser.add_argument('--var_entropy',dest='var_entropy',type=int,default=0)
	parser.add_argument('--ent_weight',dest='ent_weight',type=float,default=0.)
	parser.add_argument('--var_ent_weight',dest='var_ent_weight',type=float,default=2.)
	
	parser.add_argument('--pretrain_bias_sampling',type=float,default=0.) # Defines percentage of trajectory within which to sample trajectory segments for pretraining.
	parser.add_argument('--pretrain_bias_sampling_prob',type=float,default=0.)
	parser.add_argument('--action_scale_factor',type=float,default=1)	

	parser.add_argument('--z_exploration_bias',dest='z_exploration_bias',type=float,default=0.)
	parser.add_argument('--b_exploration_bias',dest='b_exploration_bias',type=float,default=0.)
	parser.add_argument('--lat_z_wt',dest='lat_z_wt',type=float,default=0.1)
	parser.add_argument('--lat_b_wt',dest='lat_b_wt',type=float,default=1.)
	parser.add_argument('--z_probability_factor',dest='z_probability_factor',type=float,default=0.1)
	parser.add_argument('--b_probability_factor',dest='b_probability_factor',type=float,default=0.01)
	parser.add_argument('--subpolicy_clamp_value',dest='subpolicy_clamp_value',type=float,default=-5)
	parser.add_argument('--latent_clamp_value',dest='latent_clamp_value',type=float,default=-5)
	parser.add_argument('--min_variance_bias',dest='min_variance_bias',type=float,default=0.01)
	parser.add_argument('--normalization',dest='normalization',type=str,default='None')
	parser.add_argument('--regularization_weight',dest='regularization_weight',type=float,default=0.,help='Value of regularization weight to be added to the model.')

	parser.add_argument('--likelihood_penalty',dest='likelihood_penalty',type=int,default=10)
	parser.add_argument('--subpolicy_ratio',dest='subpolicy_ratio',type=float,default=0.01)
	parser.add_argument('--latentpolicy_ratio',dest='latentpolicy_ratio',type=float,default=0.1)
	parser.add_argument('--temporal_latentpolicy_ratio',dest='temporal_latentpolicy_ratio',type=float,default=0.)
	parser.add_argument('--latent_loss_weight',dest='latent_loss_weight',type=float,default=0.1)	
	parser.add_argument('--var_loss_weight',dest='var_loss_weight',type=float,default=1.)
	parser.add_argument('--prior_weight',dest='prior_weight',type=float,default=0.00001)
	# parser.add_argument('--context_loss_weight',dest='context_loss_weight',type=float,default=1.,help='Weight of context loss.')
	parser.add_argument('--kl_weight',dest='kl_weight',type=float,default=0.01,help='KL weight when constant.')
	parser.add_argument('--kl_schedule',dest='kl_schedule',type=int,default=0,help='Whether to schedule KL weight.')
	parser.add_argument('--initial_kl_weight',dest='initial_kl_weight',type=float,default=0.0,help='Initial KL weight.')
	parser.add_argument('--final_kl_weight',dest='final_kl_weight',type=float,default=1.0,help='Initial KL weight.')
	parser.add_argument('--kl_increment_epochs',dest='kl_increment_epochs',type=int,default=100,help='Number of epochs to increment KL over.')
	parser.add_argument('--kl_begin_increment_epochs',dest='kl_begin_increment_epochs',type=int,default=100,help='Number of epochs after which to increment KL.')
	
	# Cross Domain Skill Transfer parameters. 
	parser.add_argument('--discriminability_weight',dest='discriminability_weight',type=float,default=1.,help='Weight of discriminability loss in cross domain skill transfer.') 
	parser.add_argument('--vae_loss_weight',dest='vae_loss_weight',type=float,default=1.,help='Weight of VAE loss in cross domain skill transfer.') 	
	parser.add_argument('--alternating_phase_size',dest='alternating_phase_size',type=int,default=2000, help='Size of alternating training phases.')
	parser.add_argument('--discriminator_phase_size',dest='discriminator_phase_size',type=int,default=2,help='Factor by which to train discriminator more than generator.')
	parser.add_argument('--generator_phase_size',dest='generator_phase_size',type=int,default=1,help='Factor by which to train generaotr more than discriminator.')
	parser.add_argument('--cycle_reconstruction_loss_weight',dest='cycle_reconstruction_loss_weight',type=float,default=1.,help='Weight of the cycle-consistency reconstruction loss term.')
	parser.add_argument('--real_translated_discriminator',dest='real_translated_discriminator',type=int,default=0,help='Whether to include real-translated discriminator based losses.')
	parser.add_argument('--real_trans_loss_weight',dest='real_trans_loss_weight',type=float,default=1.,help='Weight of discriminability loss between real and (cycle) translated trajectories.')
	parser.add_argument('--z_transform_discriminator',dest='z_transform_discriminator',type=int,default=0,help='Whether to use z transform discriminators.')
	parser.add_argument('--z_trajectory_discriminator',dest='z_trajectory_discriminator',type=int,default=0,help='Whether to use z trajectory discriminators.')
	parser.add_argument('--z_trajectory_discriminability_weight',dest='z_trajectory_discriminability_weight',type=float,default=1.,help='Weight of z trajectory discriminability loss.')
	parser.add_argument('--z_trajectory_discriminator_weight',dest='z_trajectory_discriminator_weight',type=float,default=1.,help='Weight of z trajectory discriminator loss.')
	parser.add_argument('--max_viz_trajs',dest='max_viz_trajs',type=int,default=5,help='How many trajectories to visualize.')
	parser.add_argument('--z_transform_or_tuple',dest='z_transform_or_tuple',type=int,default=0,help='Whether to use the z transform or z tuples.')	
	parser.add_argument('--ignore_last_z_transform',dest='ignore_last_z_transform',type=int,default=0,help='Whether to ignore or last z transform.')

	# Exploration and learning rate parameters. 
	parser.add_argument('--epsilon_from',dest='epsilon_from',type=float,default=0.3)
	parser.add_argument('--epsilon_to',dest='epsilon_to',type=float,default=0.05)
	parser.add_argument('--epsilon_over',dest='epsilon_over',type=int,default=30)
	parser.add_argument('--learning_rate',dest='learning_rate',type=float,default=1e-4)
	parser.add_argument('--transfer_learning_rate',dest='transfer_learning_rate',type=float,default=1e-4)

	# Baseline parameters. 
	parser.add_argument('--baseline_kernels',dest='baseline_kernels',type=int,default=15)
	parser.add_argument('--baseline_window',dest='baseline_window',type=int,default=15)
	parser.add_argument('--baseline_kernel_bandwidth',dest='baseline_kernel_bandwidth',type=float,default=3.5)

	# Reinforcement Learning parameters. 
	parser.add_argument('--TD',dest='TD',type=int,default=0) # Whether or not to use Temporal difference while training the critic network.
	parser.add_argument('--OU',dest='OU',type=int,default=1) # Whether or not to use the Ornstein Uhlenbeck noise process while training.
	parser.add_argument('--OU_max_sigma',dest='OU_max_sigma',type=float,default=0.2) # Max Sigma value of the Ornstein Uhlenbeck noise process.
	parser.add_argument('--OU_min_sigma',dest='OU_min_sigma',type=float,default=0.2) # Min Sigma value of the Ornstein Uhlenbeck noise process.
	parser.add_argument('--MLP_policy',dest='MLP_policy',type=int,default=0) # Whether or not to use MLP policy.
	parser.add_argument('--mean_nonlinearity',dest='mean_nonlinearity',type=int,default=0) # Whether or not to use Tanh activation.
	parser.add_argument('--burn_in_eps',dest='burn_in_eps',type=int,default=500) # How many epsiodes to burn in.
	parser.add_argument('--random_memory_burn_in',dest='random_memory_burn_in',type=int,default=1) # Whether to burn in episodes into memory randomly or not.
	parser.add_argument('--shaped_reward',dest='shaped_reward',type=int,default=0) # Whether or not to use shaped rewards.
	parser.add_argument('--memory_size',dest='memory_size',type=int,default=2000) # Size of replay memory. 2000 is okay, but is still kind of short sighted. 
	parser.add_argument('--no_mujoco',dest='no_mujoco',type=int,default=0,help='Whether we have mujoco installation or not.')

	# Transfer learning domains, etc. 
	parser.add_argument('--source_domain',dest='source_domain',type=str,help='What the source domain is in transfer.')
	parser.add_argument('--target_domain',dest='target_domain',type=str,help='What the target domain is in transfer.')
	parser.add_argument('--source_model',dest='source_model',type=str,help='What model to use for the source domain.',default=None)
	parser.add_argument('--target_model',dest='target_model',type=str,help='What model to use for the target domain.',default=None)
	parser.add_argument('--source_subpolicy_model',dest='source_subpolicy_model',type=str,help='What subpolicy model to use for the source domain.',default=None)
	parser.add_argument('--target_subpolicy_model',dest='target_subpolicy_model',type=str,help='What subpolicy model to use for the target domain.',default=None)
	parser.add_argument('--fix_source',dest='fix_source',type=int,default=0,help='Whether to fix source domain representation.')
	parser.add_argument('--fix_target',dest='fix_target',type=int,default=0,help='Whether to fix target domain representation.')
	parser.add_argument('--load_from_transfer',dest='load_from_transfer',type=int,default=0,help='Whether we are loading joint model from transfer training.')
	parser.add_argument('--dataset_variation',dest='dataset_variation',type=int,default=0,help='Whether to use flipped or original version of the dataset.')
	# parser.add_argument('--reset_subpolicy_training',dest='reset_subpolicy_training',type=int,default=1,help='Whether to reset subpolicy training.')
	parser.add_argument('--residual_translation',dest='residual_translation',type=int,default=0,help='Whether to use a residual model for translation or just a regular one.')
	parser.add_argument('--small_translation_model',dest='small_translation_model',type=int,default=0,help='Whether to use a small model for translation or not. Restricts network capacity.')
	parser.add_argument('--recurrent_translation',dest='recurrent_translation',type=int,default=0,help='Whether to implement a recurrent translation model.')
	parser.add_argument('--input_corruption_noise',dest='input_corruption_noise',type=float,default=0.,help='How much noise to add to the input to corrupt it. Default no corruption.')
	parser.add_argument('--equivariance',dest='equivariance',type=int,default=0,help='Whether to implement equivariance objective in (Joint) Fix Embed setting.')
	parser.add_argument('--equivariance_loss_weight',dest='equivariance_loss_weight',type=float,default=1.,help='Weight associated with the equivariance loss.')
	parser.add_argument('--cross_domain_supervision',dest='cross_domain_supervision',type=int,default=0,help='Whether to use cross domain supervision when operating in pair of same domains.')
	parser.add_argument('--cross_domain_supervision_loss_weight',dest='cross_domain_supervision_loss_weight',type=float,default=0.,help='Weight associated with the cross domain supervision loss.')
	parser.add_argument('--cycle_cross_domain_supervision_loss_weight',dest='cycle_cross_domain_supervision_loss_weight',type=float,default=0.,help='Weight associated with the cycle cross domain supervision loss.')
	parser.add_argument('--z_normalization',dest='z_normalization',type=str,default=None,choices=[None, 'global','ind'],help='What normalization to use for zs.')

	# Wasserstein GAN
	parser.add_argument('--wasserstein_gan',dest='wasserstein_gan',type=int,default=0,help='Whether to implement Wasserstein GAN or not.')
	parser.add_argument('--lsgan',dest='lsgan',type=int,default=0,help='Whether to implement LSGAN or not.')
	parser.add_argument('--gradient_penalty',dest='gradient_penalty',type=int,default=0,help='Whether to implement Wasserstein GAN gradient penalty or not.')
	parser.add_argument('--gradient_penalty_weight',dest='gradient_penalty_weight',type=float,default=10.,help='Relative weight of the Wasserstein discriminator gradient penalty.')
	parser.add_argument('--wasserstein_discriminator_clipping',dest='wasserstein_discriminator_clipping',type=int,default=0,help='Whether to apply clipping of discriminator parameters.')
	parser.add_argument('--wasserstein_discriminator_clipping_value',dest='wasserstein_discriminator_clipping_value',type=float,default=0.01,help='Value to apply clipping of discriminator parameters.')
	parser.add_argument('--identity_translation_loss_weight',dest='identity_translation_loss_weight',type=float,default=10,help='Weight associated with th e regularization of translation model to identity for source zs.')

	# Task ID based discriminability
	parser.add_argument('--task_discriminability',dest='task_discriminability',type=int,default=0,help='Whether or not to implement task based discriminability.')
	parser.add_argument('--number_of_tasks',dest='number_of_tasks',type=int,default=0,help='Number of tasks to be considered in task based discriminability.')
	parser.add_argument('--task_discriminability_loss_weight',dest='task_discriminability_loss_weight',type=float,default=0.,help='Loss weight associated with task based discriminability.')
	parser.add_argument('--task_discriminator_weight',dest='task_discriminator_weight',type=float,default=0.,help='Loss weight associated with task discriminator(s)')

	# Parameters for contextual training. 
	parser.add_argument('--mask_fraction',dest='mask_fraction',type=float,default=0.15,help='What fraction of zs to mask in contextual embedding.')
	parser.add_argument('--context',dest='context',type=int,default=1,help='Whether to implement contextual embedding model or original joint embedding model in Joint Transfer setting.')
	parser.add_argument('--new_context',dest='new_context',type=int,default=1,help='Whether to implement new contextual embedding model or original one.')
	parser.add_argument('--ELMO_embeddings',dest='ELMO_embeddings',type=int,default=0,help='Whether to implement ELMO style embeddings.')
	parser.add_argument('--eval_transfer_metrics',dest='eval_transfer_metrics',type=int,default=0,help='Whether to evaluate correspondence metrics in transfer setting.')

	return parser.parse_args()

def main(args):

	args = parse_arguments()
	# Moving this up.
	wandb.init(project=args.setting, dir=args.logdir, name=args.name)

	master = Master(args)	
	# Add argparse flags to wandb config.
	wandb.config.update(args)

	if args.test_code:
		master.test()
	else:
		master.run()

if __name__=='__main__':
	main(sys.argv)





