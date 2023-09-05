# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from headers import *
# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# torch.cuda.set_device(torch.device('cuda:1'))
# if use_cuda:
# 	torch.cuda.set_device(2)

class PolicyNetwork_BaseClass(torch.nn.Module):
	
	def __init__(self):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(PolicyNetwork_BaseClass, self).__init__()

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample().squeeze(0)
		return sample_action

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		return action_probabilities.argmax()

	# def select_epsilon_greedy_action(self, action_probabilities):
	# 	epsilon = 0.1
	# 	if np.random.random()<epsilon:
	# 		return self.sample_action(action_probabilities)
	# 	else:
	# 		return self.select_greedy_action(action_probabilities)

	def select_epsilon_greedy_action(self, action_probabilities, epsilon=0.1):
		epsilon = epsilon

		whether_greedy = torch.rand(action_probabilities.shape[0]).to(device)
		sample_actions = torch.where(whether_greedy<epsilon, self.sample_action(action_probabilities), self.select_greedy_action(action_probabilities))

		return sample_actions

class PolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional model, this is going to be evaluated for log-probabilities alone. 
	# Forward pass set up for evaluating this already. 

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies, number_layers=4, batch_size=1, whether_latentb_input=False):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(PolicyNetwork, self).__init__()

		if whether_latentb_input:
			self.input_size = input_size+number_subpolicies+1
		else:
			self.input_size = input_size+number_subpolicies
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = number_layers
		self.batch_size = batch_size
		
		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.softmax_layer = torch.nn.Softmax(dim=1)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)		

	def forward(self, input, hidden=None, return_log_probabilities=False):		
		# The argument hidden_input here is the initial hidden state we want to feed to the LSTM. 				
		# Assume inputs is the trajectory sequence.

		# Input Format must be: Sequence_Length x Batch_Size x Input_Size. 

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		outputs, hidden = self.lstm(format_input)

		# Takes softmax of last output. 
		if return_log_probabilities:
			# Computes log probabilities, needed for loss function and log likelihood. 

			preprobability_outputs = self.output_layer(outputs)
			log_probabilities = self.batch_logsoftmax_layer(preprobability_outputs).squeeze(1)
			probabilities = self.batch_softmax_layer(preprobability_outputs).squeeze(1)
			return outputs, hidden, log_probabilities, probabilities
		else:
			# Compute action probabilities for sampling. 
			softmax_output = self.softmax_layer(self.output_layer(outputs[-1]))
			return outputs, hidden, softmax_output

class ContinuousPolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional model, this is going to be evaluated for log-probabilities alone. 
	# Forward pass set up for evaluating this already. 

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	# def __init__(self, input_size, hidden_size, output_size, number_subpolicies, number_layers=4, batch_size=1):
	# def __init__(self, input_size, hidden_size, output_size, z_space_size, number_layers=4, batch_size=1, whether_latentb_input=False):
	def __init__(self, input_size, hidden_size, output_size, args, number_layers=4, whether_latentb_input=False, zero_z_dim=False, small_init=False):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousPolicyNetwork, self).__init__()

		self.hidden_size = hidden_size
		# The output size here must be mean+variance for each dimension. 
		# This is output_size*2. 
		self.args = args
		if self.args is None:
			self.debug = False
			self.latent_z_dimensions = 16
			self.dropout = 0.
		else:
			self.latent_z_dimensions = self.args.z_dimensions
			self.dropout = self.args.dropout
			self.debug = self.args.debug

		self.output_size = output_size
		self.num_layers = number_layers
		self.batch_size = self.args.batch_size

		
		if whether_latentb_input:
			# self.input_size = input_size+self.args.z_dimensions+1
			self.input_size = input_size+self.latent_z_dimensions+1
		else:
			if zero_z_dim:
				self.input_size = input_size
			else:
				# self.input_size = input_size+self.args.z_dimensions
				self.input_size = input_size+self.latent_z_dimensions
		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, dropout=self.dropout)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		# # Try initializing the network to something, so that we can escape the stupid constant output business.
		if small_init:
			for name, param in self.mean_output_layer.named_parameters():
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.xavier_normal_(param,gain=0.0001)

		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.

		self.variance_factor = self.args.variance_factor

	def forward(self, input, action_sequence, epsilon=0.001, batch_size=None, debugging=False):
		# Input is the trajectory sequence of shape: Sequence_Length x 1 x Input_Size. 
		# Here, we also need the continuous actions as input to evaluate their logprobability / probability. 		
		# format_input = torch.tensor(input).view(input.shape[0], self.batch_size, self.input_size).float().to(device)

		if batch_size is None:
			batch_size = self.batch_size

		format_input = input.view((input.shape[0], batch_size, self.input_size))

		hidden = None

		if isinstance(action_sequence,np.ndarray):
			format_action_seq = torch.from_numpy(action_sequence).to(device).float().view(action_sequence.shape[0], batch_size, self.output_size)
		else:
			format_action_seq = action_sequence.view(action_sequence.shape[0], batch_size, self.output_size)

		# format_action_seq = torch.from_numpy(action_sequence).to(device).float().view(action_sequence.shape[0],1,self.output_size)
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			self.mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs)) 
			# + epsilon
		else:
			self.mean_outputs = self.mean_output_layer(lstm_outputs)
		# variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)
		
		if self.args.variance_mode=='Constant':
			variance_outputs = self.args.variance_value*torch.ones_like(self.mean_outputs).to(device)
		elif self.args.variance_mode=='Annealed':
			variance_outputs = (epsilon/self.args.epsilon_scale_factor) * torch.ones_like(self.mean_outputs).to(device)
		elif self.args.variance_mode=='Learned':
			# variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias) + epsilon/self.args.epsilon_scale_factor			
			variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)
		
		# print(variance_outputs)
		# print("Embed after print var")
		# embed()
		# Remember, because of Pytorch's dynamic construction, this distribution can have it's own batch size. 
		# It doesn't matter if batch sizes changes over different forward passes of the LSTM, because we're only going
		# to evaluate this distribution (instance)'s log probability with the same sequence length. 

		# if debugging:
			# embed()		
		covariance_matrix = torch.diag_embed(variance_outputs)

		# Executing distribution creation on CPU and then copying back to GPU.
		dist = torch.distributions.MultivariateNormal(self.mean_outputs.cpu(), covariance_matrix.cpu())
		log_probabilities = dist.log_prob(format_action_seq.cpu()).to(device)

		# dist = torch.distributions.MultivariateNormal(mean_outputs, covariance_matrix)
		# log_probabilities = dist.log_prob(format_action_seq)

		# log_probabilities = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs)).log_prob(format_action_seq)
		entropy = dist.entropy()

		if self.args.debug:
			print("Embedding in the policy network.")		
			embed()
			
		return log_probabilities, entropy

	# @gpu_profile
	def get_actions(self, input, greedy=False, batch_size=None):
		if batch_size is None:
			batch_size = self.batch_size

		format_input = input.view((input.shape[0], batch_size, self.input_size))

		hidden = None
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)

		if greedy:
			return mean_outputs
		else:

			# Remember, because of Pytorch's dynamic construction, this distribution can have it's own batch size. 
			# It doesn't matter if batch sizes changes over different forward passes of the LSTM, because we're only going
			# to evaluate this distribution (instance)'s log probability with the same sequence length. 
			dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

			return dist.sample()

	def reparameterized_get_actions(self, input, greedy=False, action_epsilon=0.):
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))

		hidden = None
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			action = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			action = mean_outputs + variance_outputs * noise

		return action

	def incremental_reparam_get_actions(self, input, greedy=False, action_epsilon=0., hidden=None):
		
		# Input should be a single timestep input here. 
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		# Instead of feeding in entire input sequence, we are feeding in current timestep input and previous hidden state.
		lstm_outputs, hidden = self.lstm(format_input, hidden)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			action = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			action = mean_outputs + variance_outputs * noise

		return action, hidden

	def get_regularization_kl(self, input_z1, input_z2):
		# Input is the trajectory sequence of shape: Sequence_Length x 1 x Input_Size. 
		# Here, we also need the continuous actions as input to evaluate their logprobability / probability. 		
		format_input_z1 = input_z1.view(input_z1.shape[0], self.batch_size, self.input_size)
		format_input_z2 = input_z2.view(input_z2.shape[0], self.batch_size, self.input_size)

		hidden = None
		# format_action_seq = torch.from_numpy(action_sequence).to(device).float().view(action_sequence.shape[0],1,self.output_size)
		lstm_outputs_z1, _ = self.lstm(format_input_z1)
		# Reset hidden? 
		lstm_outputs_z2, _ = self.lstm(format_input_z2)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs_z1 = self.activation_layer(self.mean_output_layer(lstm_outputs_z1))
			mean_outputs_z2 = self.activation_layer(self.mean_output_layer(lstm_outputs_z2))
		else:
			mean_outputs_z1 = self.mean_output_layer(lstm_outputs_z1)
			mean_outputs_z2 = self.mean_output_layer(lstm_outputs_z2)
		variance_outputs_z1 = self.variance_activation_layer(self.variances_output_layer(lstm_outputs_z1))+self.variance_activation_bias
		variance_outputs_z2 = self.variance_activation_layer(self.variances_output_layer(lstm_outputs_z2))+self.variance_activation_bias

		dist_z1 = torch.distributions.MultivariateNormal(mean_outputs_z1, torch.diag_embed(variance_outputs_z1))
		dist_z2 = torch.distributions.MultivariateNormal(mean_outputs_z2, torch.diag_embed(variance_outputs_z2))

		kl_divergence = torch.distributions.kl_divergence(dist_z1, dist_z2)

		return kl_divergence

class LatentPolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional Information model, this is going to be evaluated for log-probabilities alone. 
	# THIS IS STILL A SINGLE DIRECTION LSTM!!

	# This still needs to be written separately from the normal sub-policy network(s) because it also requires termination probabilities. 
	# Must change forward pass back to using lstm() directly on the entire sequence rather than iterating.
	# Now we have the whole input sequence beforehand. 

	# Policy Network inherits from torch.nn.Module.
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, number_subpolicies, number_layers=4, b_exploration_bias=0., batch_size=1):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(LatentPolicyNetwork, self).__init__()

		# Input size is actually input_size + number_subpolicies +1 
		self.input_size = input_size+number_subpolicies+1
		self.offset_for_z = input_size+1
		self.hidden_size = hidden_size
		self.number_subpolicies = number_subpolicies
		self.output_size = number_subpolicies
		self.num_layers = number_layers
		self.b_exploration_bias = b_exploration_bias
		self.batch_size = batch_size

		# Define LSTM. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers).to(device)

		# # Try initializing the network to something, so that we can escape the stupid constant output business.
		for name, param in self.lstm.named_parameters():
			if 'bias' in name:
				torch.nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				torch.nn.init.xavier_normal_(param,gain=5)

		# Transform to output space - Latent z and Latent b. 
		self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.termination_output_layer = torch.nn.Linear(self.hidden_size,2)
		
		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
	
	def forward(self, input):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		latent_z_preprobabilities = self.subpolicy_output_layer(outputs)
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		latent_z_probabilities = self.batch_softmax_layer(latent_z_preprobabilities).squeeze(1)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)

		latent_z_logprobabilities = self.batch_logsoftmax_layer(latent_z_preprobabilities).squeeze(1)
		latent_b_logprobabilities = self.batch_logsoftmax_layer(latent_b_preprobabilities).squeeze(1)
			
		# Return log probabilities. 
		return latent_z_logprobabilities, latent_b_logprobabilities, latent_b_probabilities, latent_z_probabilities

	def get_actions(self, input, greedy=False):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		latent_z_preprobabilities = self.subpolicy_output_layer(outputs)
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		latent_z_probabilities = self.batch_softmax_layer(latent_z_preprobabilities).squeeze(1)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)

		if greedy==True:
			selected_b = self.select_greedy_action(latent_b_probabilities)
			selected_z = self.select_greedy_action(latent_z_probabilities)
		else:
			selected_b = self.sample_action(latent_b_probabilities)
			selected_z = self.sample_action(latent_z_probabilities)
		
		return selected_b, selected_z

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

class ContinuousLatentPolicyNetwork(PolicyNetwork_BaseClass):

	# def __init__(self, input_size, hidden_size, z_dimensions, number_layers=4, b_exploration_bias=0., batch_size=1):
	def __init__(self, input_size, hidden_size, args, number_layers=4):		

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousLatentPolicyNetwork, self).__init__()

		self.args = args
		# Input size is actually input_size + number_subpolicies +1 
		self.input_size = input_size+self.args.z_dimensions+1
		self.offset_for_z = input_size+1
		self.hidden_size = hidden_size
		# self.number_subpolicies = number_subpolicies
		self.output_size = self.args.z_dimensions
		self.num_layers = number_layers
		self.b_exploration_bias = self.args.b_exploration_bias
		self.batch_size = self.args.batch_size

		# Define LSTM. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, dropout=self.args.dropout).to(device)

		# Transform to output space - Latent z and Latent b. 
		# self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.termination_output_layer = torch.nn.Linear(self.hidden_size,2)
		
		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.
			
		self.variance_factor = 0.01

		# # # Try initializing the network to something, so that we can escape the stupid constant output business.
		for name, param in self.lstm.named_parameters():
			if 'bias' in name:
				torch.nn.init.constant_(param, 0.001)
			elif 'weight' in name:
				torch.nn.init.xavier_normal_(param,gain=5)

		# Also initializing mean_output_layer to something large...
		for name, param in self.mean_output_layer.named_parameters():
			if 'bias' in name:
				torch.nn.init.constant_(param, 0.)
			elif 'weight' in name:
				torch.nn.init.xavier_normal_(param,gain=2)

	def forward(self, input, epsilon=0.001):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)
	
		latent_b_preprobabilities = self.termination_output_layer(outputs)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	
		latent_b_logprobabilities = self.batch_logsoftmax_layer(latent_b_preprobabilities).squeeze(1)
			
		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))	

		if self.args.debug:
			print("Embedding in Latent Policy.")
			embed()
		# Return log probabilities. 
		return latent_b_logprobabilities, latent_b_probabilities, self.dists

	def get_actions(self, input, greedy=False, epsilon=0.001):
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)
	
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	
			
		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))	

		if greedy==True:
			selected_b = self.select_greedy_action(latent_b_probabilities)
			selected_z = mean_outputs
		else:
			# selected_b = self.sample_action(latent_b_probabilities)
			selected_b = self.select_greedy_action(latent_b_probabilities)
			selected_z = self.dists.sample()

		return selected_b, selected_z

	def incremental_reparam_get_actions(self, input, greedy=False, action_epsilon=0.001, hidden=None, previous_z=None):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		outputs, hidden = self.lstm(format_input, hidden)

		latent_b_preprobabilities = self.termination_output_layer(outputs)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	
		# Greedily select b. 
		selected_b = self.select_greedy_action(latent_b_probabilities)

		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + action_epsilon

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			selected_z = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			selected_z = mean_outputs + variance_outputs * noise

		# If single input and previous_Z is None, this is the first timestep. So set b to 1, and don't do anything to z. 
		if input.shape[0]==1 and previous_z is None:
			selected_b[0] = 1
		# If previous_Z is not None, this is not the first timestep, so don't do anything to z. If b is 0, use previous. 
		elif input.shape[0]==1 and previous_z is not None:
			if selected_b==0:
				selected_z = previous_z
		elif input.shape[0]>1:
			# Now modify z's as per New Z Selection. 
			# Set initial b to 1. 
			selected_b[0] = 1
			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if selected_b[t]==0:
					selected_z[t] = selected_z[t-1]

		return selected_z, selected_b, hidden

	def reparam_get_actions(self, input, greedy=False, action_epsilon=0.001, hidden=None):

		# Wraps incremental 
		# MUST MODIFY INCREMENTAL ONE TO HANDLE NEW_Z_SELECTION (i.e. only choose new one if b is 1....)

			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

class ContinuousLatentPolicyNetwork_ConstrainedBPrior(ContinuousLatentPolicyNetwork):

	def __init__(self, input_size, hidden_size, args, number_layers=4):		

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousLatentPolicyNetwork_ConstrainedBPrior, self).__init__(input_size, hidden_size, args, number_layers)

		# We can inherit the forward function from the above class... we just need to modify get actions.	

		self.min_skill_time = 12
		self.max_skill_time = 16

	def get_prior_value(self, elapsed_t, max_limit=5):

		skill_time_limit = max_limit-1

		# if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
		# if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
		if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic','GRAB']:
			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				# lens = np.array([12,13,14,15,16])
				np.arange(self.min_skill_time, self.max_skill_time)
				# probabilities = np.array([0.1,0.2,0.4,0.2,0.1])
				prob_biases = np.array([[0.8,0.],[0.4,0.],[0.,0.],[0.,0.4]])				

				max_limit = 16
				skill_time_limit = 12

			else:
				max_limit = 20
				skill_time_limit = max_limit-1	

		prior_value = torch.zeros((1,2)).to(device).float()
		# If at or over hard limit.
		if elapsed_t>=max_limit:
			prior_value[0,1]=1.

		# If at or more than typical, less than hard limit:
		elif elapsed_t>=skill_time_limit:
	
			if self.args.var_skill_length:
				prior_value[0] = torch.tensor(prob_biases[elapsed_t-skill_time_limit]).to(device).float()
			else:
				# Random
				prior_value[0,1]=0. 

		# If less than typical. 
		else:
			# Continue.
			prior_value[0,0]=1.

		return prior_value

	def get_actions(self, input, greedy=False, epsilon=0.001, delta_t=0, batch_size=None):

		if batch_size is None:
			batch_size = self.batch_size

		format_input = input.view((input.shape[0], batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)
	
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias		
			
		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))	

		############################################
		prior_value = self.get_prior_value(delta_t)

		# Now... add prior value.
		# Only need to do this to the last timestep... because the last sampled b is going to be copied into a different variable that is stored.
		latent_b_preprobabilities[-1, :, :] += prior_value		
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	

		# Sample b. 
		selected_b = self.select_greedy_action(latent_b_probabilities)
		############################################

		# Now implementing hard constrained b selection.
		if delta_t < self.min_skill_time:
			# Continue. Set b to 0.
			selected_b[-1] = 0.

		elif (self.min_skill_time <= delta_t) and (delta_t < self.max_skill_time):
			pass

		else: 
			# Stop and select a new z. Set b to 1. 
			selected_b[-1] = 1.

		# Also get z... assume higher level funciton handles the new z selection component. 
		if greedy==True:
			selected_z = mean_outputs
		else:
			selected_z = self.dists.sample()

		return selected_b, selected_z

	def incremental_reparam_get_actions(self, input, greedy=False, action_epsilon=0.001, hidden=None, previous_z=None, delta_t=0):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		outputs, hidden = self.lstm(format_input, hidden)

		latent_b_preprobabilities = self.termination_output_layer(outputs)
		
		############################################
		# GET PRIOR AND ADD. 
		prior_value = self.get_prior_value(delta_t)
		latent_b_preprobabilities[-1, :, :] += prior_value			
		############################################
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	

		# Greedily select b. 
		selected_b = self.select_greedy_action(latent_b_probabilities)

		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + action_epsilon

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			selected_z = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			selected_z = mean_outputs + variance_outputs * noise

		# If single input and previous_Z is None, this is the first timestep. So set b to 1, and don't do anything to z. 
		if input.shape[0]==1 and previous_z is None:
			selected_b[0] = 1
		# If previous_Z is not None, this is not the first timestep, so don't do anything to z. If b is 0, use previous. 
		elif input.shape[0]==1 and previous_z is not None:
			if selected_b==0:
				selected_z = previous_z
		elif input.shape[0]>1:
			# Now modify z's as per New Z Selection. 
			# Set initial b to 1. 
			selected_b[0] = 1
			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if selected_b[t]==0:
					selected_z[t] = selected_z[t-1]

		return selected_z, selected_b, hidden

class VariationalPolicyNetwork(PolicyNetwork_BaseClass):
	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	# def __init__(self, input_size, hidden_size, number_subpolicies, number_layers=4, z_exploration_bias=0., b_exploration_bias=0.,  batch_size=1):
	def __init__(self, input_size, hidden_size, number_subpolicies, args, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(VariationalPolicyNetwork, self).__init__()
		
		self.args = args
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.number_subpolicies = number_subpolicies
		self.output_size = number_subpolicies
		self.num_layers = number_layers	
		self.z_exploration_bias = self.args.z_exploration_bias
		self.b_exploration_bias = self.args.b_exploration_bias
		self.z_probability_factor = self.args.z_probability_factor
		self.b_probability_factor = self.args.b_probability_factor
		self.batch_size = self.args.batch_size

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True)

		# Transform to output space - Latent z and Latent b. 
		# THIS OUTPUT LAYER TAKES 2*HIDDEN SIZE as input because it's bidirectional. 
		self.subpolicy_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.termination_output_layer = torch.nn.Linear(2*self.hidden_size,2)

		# Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
			
	def sample_latent_variables(self, subpolicy_outputs, termination_output_layer):
		# Run sampling layers. 
		sample_z = self.sample_action(subpolicy_outputs)		
		sample_b = self.sample_action(termination_output_layer)
		return sample_z, sample_b 

	def sample_latent_variables_epsilon_greedy(self, subpolicy_outputs, termination_output_layer, epsilon):
		sample_z = self.select_epsilon_greedy_action(subpolicy_outputs, epsilon)
		sample_b = self.select_epsilon_greedy_action(termination_output_layer, epsilon)
		return sample_z, sample_b

	def forward(self, input, epsilon, new_z_selection=True):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_z_preprobabilities = self.subpolicy_output_layer(outputs)*self.z_probability_factor + self.z_exploration_bias
		# variational_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor
		# Add b continuation bias to the continuing option at every timestep. 
		variational_b_preprobabilities[:,0,0] += self.b_exploration_bias

		variational_z_probabilities = self.batch_softmax_layer(variational_z_preprobabilities).squeeze(1)
		variational_b_probabilities = self.batch_softmax_layer(variational_b_preprobabilities).squeeze(1)

		variational_z_logprobabilities = self.batch_logsoftmax_layer(variational_z_preprobabilities).squeeze(1)
		variational_b_logprobabilities = self.batch_logsoftmax_layer(variational_b_preprobabilities).squeeze(1)
		
		# sampled_z_index, sampled_b = self.sample_latent_variables(variational_z_probabilities, variational_b_probabilities)
		sampled_z_index, sampled_b = self.sample_latent_variables_epsilon_greedy(variational_z_probabilities, variational_b_probabilities, epsilon)

		if new_z_selection:
			# Set initial b to 1. 
			sampled_b[0] = 1

			# # Trying cheeky thing to see if we can learn in this setting.
			# sampled_b[1:] = 0

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]

		return sampled_z_index, sampled_b, variational_b_logprobabilities,\
		 variational_z_logprobabilities, variational_b_probabilities, variational_z_probabilities, None

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		# Sampling can handle batched action_probabilities. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample()
		return sample_action

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

	def select_epsilon_greedy_action(self, action_probabilities, epsilon=0.1):
		epsilon = epsilon
		# if np.random.random()<epsilon:
		# 	# return(np.random.randint(0,high=len(action_probabilities)))
		# 	return self.sample_action(action_probabilities)
		# else:
		# 	return self.select_greedy_action(action_probabilities)

		# Issue with the current implementation is that it selects either sampling or greedy selection identically across the entire batch. 
		# This is stupid, use a toch.where instead? 
		# Sample an array of binary variables of size = batch size. 
		# For each, use greedy or ... 

		whether_greedy = torch.rand(action_probabilities.shape[0]).to(device)
		sample_actions = torch.where(whether_greedy<epsilon, self.sample_action(action_probabilities), self.select_greedy_action(action_probabilities))

		return sample_actions

	def sample_termination(self, termination_probability):
		sample_terminal = torch.distributions.Bernoulli(termination_probability).sample().squeeze(0)
		return sample_terminal

class ContinuousVariationalPolicyNetwork(PolicyNetwork_BaseClass):

	# def __init__(self, input_size, hidden_size, z_dimensions, number_layers=4, z_exploration_bias=0., b_exploration_bias=0.,  batch_size=1):
	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):
		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousVariationalPolicyNetwork, self).__init__()
	
		self.args = args	
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = z_dimensions
		self.num_layers = number_layers	
		self.z_exploration_bias = self.args.z_exploration_bias
		self.b_exploration_bias = self.args.b_exploration_bias
		self.z_probability_factor = self.args.z_probability_factor
		self.b_probability_factor = self.args.b_probability_factor
		self.batch_size = self.args.batch_size

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True, dropout=self.args.dropout)

		# Transform to output space - Latent z and Latent b. 
		# THIS OUTPUT LAYER TAKES 2*HIDDEN SIZE as input because it's bidirectional. 
		self.termination_output_layer = torch.nn.Linear(2*self.hidden_size,2)

		# Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=-1)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()

		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.
			
		self.variance_factor = 0.01

	def forward(self, input, epsilon, new_z_selection=True, var_epsilon=0.001):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

		# Add b continuation bias to the continuing option at every timestep. 
		variational_b_preprobabilities[:,0,0] += self.b_exploration_bias
		variational_b_probabilities = self.batch_softmax_layer(variational_b_preprobabilities).squeeze(1)		
		variational_b_logprobabilities = self.batch_logsoftmax_layer(variational_b_preprobabilities).squeeze(1)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + var_epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		sampled_b = self.select_epsilon_greedy_action(variational_b_probabilities, epsilon)

		if epsilon==0.:
			sampled_z_index = mean_outputs.squeeze(1)
		else:

			# Whether to use reparametrization trick to retrieve the latent_z's.
			if self.args.reparam:

				if self.args.train:
					noise = torch.randn_like(variance_outputs)

					# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
					sampled_z_index = mean_outputs + variance_outputs*noise
					# Ought to be able to pass gradients through this latent_z now.

					sampled_z_index = sampled_z_index.squeeze(1)

				# If evaluating, greedily get action.
				else:
					sampled_z_index = mean_outputs.squeeze(1)
			else:
				sampled_z_index = self.dists.sample().squeeze(1)
		
		if new_z_selection:
			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]		

		# Also compute logprobabilities of the latent_z's sampled from this net. 
		variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))
		variational_z_probabilities = None

		# Set standard distribution for KL. 
		self.standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).to(device),torch.eye((self.output_size)).to(device))
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, self.standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = self.standard_distribution.log_prob(sampled_z_index)

		# if self.args.debug:
		# 	print("#################################")
		# 	print("Embedding in Variational Network.")
		# 	embed()

		return sampled_z_index, sampled_b, variational_b_logprobabilities,\
		 variational_z_logprobabilities, variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		# Sampling can handle batched action_probabilities. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample()
		return sample_action

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

	def select_epsilon_greedy_action(self, action_probabilities, epsilon=0.1):
		epsilon = epsilon
		# if np.random.random()<epsilon:
		# 	# return(np.random.randint(0,high=len(action_probabilities)))
		# 	return self.sample_action(action_probabilities)
		# else:
		# 	return self.select_greedy_action(action_probabilities)

		# Issue with the current implementation is that it selects either sampling or greedy selection identically across the entire batch. 
		# This is stupid, use a toch.where instead? 
		# Sample an array of binary variables of size = batch size. 
		# For each, use greedy or ... 

		whether_greedy = torch.rand(action_probabilities.shape[0]).to(device)
		sample_actions = torch.where(whether_greedy<epsilon, self.sample_action(action_probabilities), self.select_greedy_action(action_probabilities))

		return sample_actions

class ContinuousVariationalPolicyNetwork_BPrior(ContinuousVariationalPolicyNetwork):
	
	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousVariationalPolicyNetwork_BPrior, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)

	def get_prior_value(self, elapsed_t, max_limit=5):

		skill_time_limit = max_limit-1

		# if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
		# if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
		if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic','GRAB']:
			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				lens = np.array([12,13,14,15,16])
				# probabilities = np.array([0.1,0.2,0.4,0.2,0.1])
				prob_biases = np.array([[0.8,0.],[0.4,0.],[0.,0.],[0.,0.4]])				

				max_limit = 16
				skill_time_limit = 12

			else:
				max_limit = 20
				skill_time_limit = max_limit-1	
		prior_value = torch.zeros((1,2)).to(device).float()
		# If at or over hard limit.
		if elapsed_t>=max_limit:
			prior_value[0,1]=1.

		# If at or more than typical, less than hard limit:
		elif elapsed_t>=skill_time_limit:
	
			if self.args.var_skill_length:
				prior_value[0] = torch.tensor(prob_biases[elapsed_t-skill_time_limit]).to(device).float()
			else:
				# Random
				prior_value[0,1]=0. 

		# If less than typical. 
		else:
			# Continue.
			prior_value[0,0]=1.

		return prior_value

	def forward(self, input, epsilon, new_z_selection=True):

		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon
		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		prev_time = 0
		# Create variables for prior and probs.
		prior_values = torch.zeros_like(variational_b_preprobabilities).to(device).float()
		variational_b_probabilities = torch.zeros_like(variational_b_preprobabilities).to(device).float()
		variational_b_logprobabilities = torch.zeros_like(variational_b_preprobabilities).to(device).float()
		sampled_b = torch.zeros(input.shape[0]).to(device).int()
		sampled_b[0] = 1
		
		for t in range(1,input.shape[0]):

			# Compute prior value. 
			delta_t = t-prev_time

			# if self.args.debug:
			# 	print("##########################")
			# 	print("Time: ",t, " Prev Time:",prev_time, " Delta T:",delta_t)

			prior_values[t] = self.get_prior_value(delta_t, max_limit=self.args.skill_length)

			# Construct probabilities.
			variational_b_probabilities[t,0,:] = self.batch_softmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
			variational_b_logprobabilities[t,0,:] = self.batch_logsoftmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
			
			sampled_b[t] = self.select_epsilon_greedy_action(variational_b_probabilities[t:t+1], epsilon)

			if sampled_b[t]==1:
				prev_time = t				

			# if self.args.debug:
			# 	print("Sampled b:",sampled_b[t])

		if epsilon==0.:
			sampled_z_index = mean_outputs.squeeze(1)
		else:

			# Whether to use reparametrization trick to retrieve the latent_z's.
			if self.args.reparam:

				if self.args.train:
					noise = torch.randn_like(variance_outputs)

					# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
					sampled_z_index = mean_outputs + variance_outputs*noise
					# Ought to be able to pass gradients through this latent_z now.

					sampled_z_index = sampled_z_index.squeeze(1)

				# If evaluating, greedily get action.
				else:
					sampled_z_index = mean_outputs.squeeze(1)
			else:
				sampled_z_index = self.dists.sample().squeeze(1)
		
		if new_z_selection:
			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]		

		# Also compute logprobabilities of the latent_z's sampled from this net. 
		variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))
		variational_z_probabilities = None

		# Set standard distribution for KL. 
		self.standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).to(device),torch.eye((self.output_size)).to(device))
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, self.standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = self.standard_distribution.log_prob(sampled_z_index)

		if self.args.debug:
			print("#################################")
			print("Embedding in Variational Network.")
			embed()

		return sampled_z_index, sampled_b, variational_b_logprobabilities.squeeze(1), \
		 variational_z_logprobabilities, variational_b_probabilities.squeeze(1), variational_z_probabilities, kl_divergence, prior_loglikelihood

class ContinuousVariationalPolicyNetwork_ConstrainedBPrior(ContinuousVariationalPolicyNetwork_BPrior):

	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousVariationalPolicyNetwork_ConstrainedBPrior, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)

		# if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
		# if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
		if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic','GRAB']:			
			self.min_skill_time = 12
			self.max_skill_time = 16
		else:
			self.min_skill_time = 4
			self.max_skill_time = 6

	def forward(self, input, epsilon, new_z_selection=True, batch_size=1):

		# Input Format must be: Sequence_Length x Batch_Size x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon
		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		# Create variables for prior and probabilities.
		prior_values = torch.zeros_like(variational_b_preprobabilities).to(device).float()
		variational_b_probabilities = torch.zeros_like(variational_b_preprobabilities).to(device).float()
		variational_b_logprobabilities = torch.zeros_like(variational_b_preprobabilities).to(device).float()

		#######################################
		################ Set B ################
		#######################################
		
		# Set the first b to 1, and the time b was == 1. 		
		sampled_b = torch.zeros(input.shape[0]).to(device).int()
		# Changing to batching.. 
		sampled_b = torch.zeros(input.shape[0], self.args.batch_size).to(device).int()

		sampled_b[0] = 1
		prev_time = 0

		for t in range(1,input.shape[0]):
			
			# Compute time since the last b occurred. 			
			delta_t = t-prev_time
			# Compute prior value. 
			prior_values[t] = self.get_prior_value(delta_t, max_limit=self.args.skill_length)
			
			# Construct probabilities.
			variational_b_probabilities[t,0,:] = self.batch_softmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
			variational_b_logprobabilities[t,0,:] = self.batch_logsoftmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
	
			# Now Implement Hard Restriction on Selection of B's. 
			if delta_t < self.min_skill_time:
				# Set B to 0. I.e. Continue. 
				# variational_b_probabilities[t,0,:] = variational_b_probabilities[t,0,:]*0
				# variational_b_probabilities[t,0,0] += 1
				
				sampled_b[t] = 0.

			elif (self.min_skill_time <= delta_t) and (delta_t < self.max_skill_time):		
				# Sample b. 			
				sampled_b[t] = self.select_epsilon_greedy_action(variational_b_probabilities[t:t+1], epsilon)

			elif self.max_skill_time <= delta_t:
				# Set B to 1. I.e. select new z. 
				sampled_b[t] = 1.

			# If b is 1, set the previous time to now. 
			if sampled_b[t]==1:
				prev_time = t				

		#######################################
		################ Set Z ################
		#######################################

		# Now set the z's. If greedy, just return the means. 
		if epsilon==0.:
			sampled_z_index = mean_outputs.squeeze(1)
		# If not greedy, then reparameterize. 
		else:
			# Whether to use reparametrization trick to retrieve the latent_z's.
			if self.args.train:
				noise = torch.randn_like(variance_outputs)

				# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
				sampled_z_index = mean_outputs + variance_outputs*noise
				# Ought to be able to pass gradients through this latent_z now.

				sampled_z_index = sampled_z_index.squeeze(1)

			# If evaluating, greedily get action.
			else:
				sampled_z_index = mean_outputs.squeeze(1)
		
		# Modify z's based on whether b was 1 or 0. This part should remain the same.		
		if new_z_selection:
			
			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]		

		# Also compute logprobabilities of the latent_z's sampled from this net. 
		variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))
		variational_z_probabilities = None

		# Set standard distribution for KL. 
		self.standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).to(device),torch.eye((self.output_size)).to(device))
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, self.standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = self.standard_distribution.log_prob(sampled_z_index)

		if self.args.debug:
			print("#################################")
			print("Embedding in Variational Network.")
			embed()

		return sampled_z_index, sampled_b, variational_b_logprobabilities.squeeze(1), \
		 variational_z_logprobabilities, variational_b_probabilities.squeeze(1), variational_z_probabilities, kl_divergence, prior_loglikelihood

class ContinuousVariationalPolicyNetwork_Batch(ContinuousVariationalPolicyNetwork_ConstrainedBPrior):

	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4, translation_network=False):

		super(ContinuousVariationalPolicyNetwork_Batch, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)

		self.translation_network = translation_network
	
	def get_prior_value(self, elapsed_t, max_limit=5, batch_size=None):

		if batch_size==None:
			batch_size = self.batch_size
		
		skill_time_limit = max_limit-1	
		
		# if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
		if self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic','GRAB']:
			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				lens = np.array([12,13,14,15,16])
				# probabilities = np.array([0.1,0.2,0.4,0.2,0.1])
				prob_biases = np.array([[0.8,0.],[0.4,0.],[0.,0.],[0.,0.4]])				

				max_limit = 16
				skill_time_limit = 12

			else:
				max_limit = 20
				skill_time_limit = max_limit-1	
		else:
			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# probabilities = np.array([0.1,0.2,0.4,0.2,0.1])
				prob_biases = np.array([[0.8,0.],[0.4,0.],[0.,0.],[0.,0.4]])				

				max_limit = 6
				skill_time_limit = 4

			else:
				max_limit = 5
				skill_time_limit = max_limit-1	

		# Compute elapsed time - skill time limit.
		delt = elapsed_t-skill_time_limit

		# Initialize prior vlaues. 
		prior_value = torch.zeros((batch_size,2)).to(device).float()
		
		# Since we're evaluating multiple conditions over the batch, don't do this with if-else structures. 
		# Instead, set values of prior based on which of the following cases they fall into. 			
		# print("Embedding in prior computation!")
		# print("TIME: ",elapsed_t)
		# print("Prior: ",prior_value)
		# embed()	

		######################################
		# CASE 1: If we're at / over the max limit:
		######################################
		condition_1 = torch.tensor((elapsed_t>=max_limit).astype(int)).to(device).float()
		case_1_block = np.array([[0,1]])
		case_1_value = torch.tensor(np.repeat(case_1_block, batch_size, axis=0)).to(device).float()

		######################################
		# CASE 2:  If we're not over max limt, but at/ over the typical skill time length.
		######################################
		condition_2 = torch.tensor((elapsed_t>=skill_time_limit).astype(int)*(elapsed_t<max_limit).astype(int)).to(device).float()

		sel_indices = np.where(elapsed_t>=skill_time_limit)[0]
		intermediate_values = np.min([np.ones_like(delt,dtype=int)*3,np.max([np.zeros_like(delt,dtype=int),delt.astype(int)],axis=0)],axis=0)

		# Create basic building block that's going to repeat, that we use for the var_skill_length=0 case. 
		block = np.array([[0,1]])
		block_repeat = np.repeat(block, batch_size, axis=0)

		# Create array that sets values based on var_skill_length cases. 
		case_2_value = torch.tensor((self.args.var_skill_length*prob_biases[intermediate_values]) + \
			(1-self.args.var_skill_length)*block_repeat).to(device).float()

		######################################
		# CASE 3: If we're not over either the max limit or the typical skill time length.
		######################################
		condition_3 = torch.tensor((elapsed_t<skill_time_limit).astype(int)).to(device).float()
		case_3_block = np.array([[1,0]])
		case_3_value = torch.tensor(np.repeat(case_3_block, batch_size, axis=0)).to(device).float()

		######################################
		# Now set the prior values. 
		######################################		
		prior_value = condition_1.unsqueeze(1)*case_1_value + condition_2.unsqueeze(1)*case_2_value + condition_3.unsqueeze(1)*case_3_value
		
		# Now return prior value. 
		return prior_value

	# @gpu_profile
	# @tprofile(immediate=True)
	def forward(self, input, epsilon, new_z_selection=True, batch_size=None, batch_trajectory_lengths=None, precomputed_b=None, evaluate_z_probability=None):

		##################################################
		##################### Set A ######################
		##################################################

		if batch_size is None:
			batch_size = self.batch_size			

		##################################################
		# Pass through base LSTM. 	
		##################################################

		# Input Format must be: Sequence_Length x Batch_Size x Input_Size. 	
		format_input = input.view((input.shape[0], batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		##################################################
		# If usual variational network, predict b's. 
		##################################################

		if not(self.translation_network):
			# Damping factor for probabilities to prevent washing out of bias. 
			variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

			# Create variables for prior and probabilities.
			prior_values = torch.zeros_like(variational_b_preprobabilities).to(device).float()
			variational_b_probabilities = torch.zeros_like(variational_b_preprobabilities).to(device).float()
			variational_b_logprobabilities = torch.zeros_like(variational_b_preprobabilities).to(device).float()

		##################################################
		# Predict latent z's.
		##################################################

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		##################################################
		##################### Set B ######################
		##################################################
				
		if not(self.translation_network):

			##################################################
			# Initialize b's.
			##################################################

			# Set the first b to 1, and the time b was == 1. 		
			# sampled_b = torch.zeros(input.shape[0]).to(device).int()
			# Changing to batching.. 
			sampled_b = torch.zeros(input.shape[0], batch_size).to(device).int()
			sampled_b[0] = 1

			prev_time = np.zeros((batch_size))

			##################################################
			# Iterate over time and get b's.
			##################################################

			for t in range(1,input.shape[0]):
				
				# Compute time since the last b occurred. 			
				delta_t = t-prev_time
				
				# Compute prior value. 
				# print("SAMPLED B: ", sampled_b[:t])
				prior_values[t] = self.get_prior_value(delta_t, max_limit=self.args.skill_length, batch_size=batch_size)

				# Construct probabilities.
				variational_b_probabilities[t] = self.batch_softmax_layer(variational_b_preprobabilities[t] + prior_values[t])
				variational_b_logprobabilities[t] = self.batch_logsoftmax_layer(variational_b_preprobabilities[t] + prior_values[t])

				############################
				# Batching versions of implementing hard restriction of selection of B's.
				############################

				# CASE 1: If we haven't reached the minimum skill execution time. 
				condition_1 = torch.tensor((delta_t<self.min_skill_time).astype(int)).to(device)
				
				# CASE 2: If execution time is over the minimum skill execution time, but less than the maximum:
				condition_2 = torch.tensor((delta_t>=self.min_skill_time).astype(int)*(delta_t<self.max_skill_time).astype(int)).to(device)
				
				# CASE 3: If we have reached the maximum skill execution time.
				condition_3 = torch.tensor((delta_t>=self.max_skill_time).astype(int)).to(device)

				sampled_b[t] = condition_1*torch.zeros(1).to(device).float() + (condition_2*self.select_epsilon_greedy_action(variational_b_probabilities[t:t+1], epsilon)).squeeze(0) + \
					condition_3*torch.ones(1).to(device).float()

				# Now if sampled_b[t] ==1, set the prev_time of that batch element to current time t. 
				# Otherwise, let prev_time stay prev_time.
				# Maybe a safer way to execute this: 
				prev_time[(torch.where(sampled_b[t]==1)[0]).cpu().detach().numpy()] = t		

		else:
			# Here, we didn't need to actually compute b's. So just assign them from precomputed ones. 
			sampled_b = precomputed_b.detach()

		##################################################
		##################### Set Z ######################
		##################################################

		##################################################
		# Get initial z predictions.
		##################################################
		# Now set the z's. If greedy, just return the means. 
		if epsilon==0. or not(self.args.train):
			sampled_z_index = mean_outputs.squeeze(1)
		# If not greedy, then reparameterize. 
		else:
			# Whether to use reparametrization trick to retrieve the latent_z's.
			noise = torch.randn_like(variance_outputs)

			# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
			sampled_z_index = mean_outputs + variance_outputs*noise
			# Ought to be able to pass gradients through this latent_z now.

			sampled_z_index = sampled_z_index.squeeze(1)
		
		##################################################
		# Modify z's based on whether b was 1 or 0.
		##################################################
		
		if new_z_selection:

			if not(self.translation_network):
				# Set initial b to 1. 
				sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 				
				sampled_z_index[t, torch.where(sampled_b[t]==0)[0]] = sampled_z_index[t-1, torch.where(sampled_b[t]==0)[0]]

			# How to vectorize this op? 
			# In general, need to spend some time rewriting this entire function.?

		##################################################
		# Get z probabilities, KL and prior values.
		##################################################
				
		# Also compute logprobabilities of the latent_z's sampled from this net. 
		if self.args.batch_size>1:
			variational_z_logprobabilities = self.dists.log_prob(sampled_z_index)
		else:
			variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))

		variational_z_probabilities = None

		# Set standard distribution for KL. 
		self.standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).to(device),torch.eye((self.output_size)).to(device))
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, self.standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = self.standard_distribution.log_prob(sampled_z_index)

		if self.args.debug:
			print("#################################")
			print("Embedding in Variational Network.")
			embed()		

		if self.translation_network:
			if evaluate_z_probability is None:
				return sampled_z_index
			else:
				return self.dists.log_prob(evaluate_z_probability)
		else:
			return sampled_z_index, sampled_b, variational_b_logprobabilities.squeeze(1), \
		 	variational_z_logprobabilities, variational_b_probabilities.squeeze(1), variational_z_probabilities, kl_divergence, prior_loglikelihood

	def get_probabilities(self, input, epsilon, precomputed_b=None, evaluate_value=None):
		return self.forward(input, epsilon, precomputed_b=precomputed_b, evaluate_z_probability=evaluate_value)

# class ContinuousVariationalPolicyNetwork_Factored(ContinuousVariationalPolicyNetwork_Batch):

# 	def __init__(self):

class ContinuousContextualVariationalPolicyNetwork(ContinuousVariationalPolicyNetwork_Batch):

	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):
		
		super(ContinuousContextualVariationalPolicyNetwork, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)

		# Define a bidirectional LSTM now.
		self.z_dimensions = self.args.z_dimensions
		self.contextual_lstm = torch.nn.LSTM(input_size=self.args.z_dimensions,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True, dropout=self.args.dropout)
		# self.z_output_layer = torch.nn.Linear(2*self.hidden_size, self.z_dimensions)

		self.contextual_mean_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.contextual_variances_output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)

	def forward(self, input, epsilon, new_z_selection=True, batch_size=None, batch_trajectory_lengths=None):
		
		# First run the forward function of the original variational network. 
		# This runs the initial LSTM and predicts the original embedding of skills. 
		sampled_z_index, sampled_b, variational_b_logprobabilities, \
		 variational_z_logprobabilities, variational_b_probabilities, \
		 variational_z_probabilities, kl_divergence, prior_loglikelihood = \
			 super().forward(input, epsilon, new_z_selection=new_z_selection, batch_size=batch_size, batch_trajectory_lengths=batch_trajectory_lengths)

		# Now parse the sequence of per timestep z's to sequence of z's of length = the number of skills in the trajectory. 
		# The latent_b vector has this information, specified in terms of when b=1. 
		distinct_indices_collection = []
		z_sequence_collection = []
		# Also collect indices to mask, at specified mask fraction
		mask_indices_collection = []
		max_distinct_zs = 0
		
		for j in range(self.args.batch_size):

			# Mask z's that extend past the trajectory length. 
			sampled_b[batch_trajectory_lengths[j]:,j] = 0
			sampled_z_index[batch_trajectory_lengths[j]:,j] = 0.

			# Get times at which we actually observe distinct z's.
			distinct_z_indices = torch.where(sampled_b[:,j])[0].clone().detach().cpu().numpy()

			# Keep track of max, so that we can create a tensor of that size.
			if len(distinct_z_indices)>max_distinct_zs:
				max_distinct_zs = len(distinct_z_indices)

			# mask_indices.append(np.random.choice(distinct_z_indices, size=int(len(distinct_z_indices)*self.args.mask_fraction), replace=False))
			# These mask indices index into the distinct_indices list, so the values in mask indices are positions in the list to be masked.
			# Masking strategy - uniformly randomly sample mask_fraction arbitrarily. 
			number_mask_elements = np.ceil(len(distinct_z_indices)*self.args.mask_fraction).astype(int)
			if number_mask_elements==1 and len(distinct_z_indices)==1:
				number_mask_elements = 0
			mask_indices = np.random.choice(range(len(distinct_z_indices)), size=number_mask_elements, replace=False)
			mask_indices_collection.append(copy.deepcopy(mask_indices))

			# Now copy over the masked indices into a single list. 
			distinct_indices_collection.append(copy.deepcopy(distinct_z_indices))

			# Now actually mask the chosen mask indices.
			masked_z = sampled_z_index[distinct_z_indices,j]
			masked_z[mask_indices] = 0.

			# Now copy over the masked indices into a single list. 
			z_sequence_collection.append(masked_z)

		# Now that we've gotten the distinct z sequence, make padded tensor version of this. 
		self.initial_skill_embedding = torch.zeros((max_distinct_zs, self.args.batch_size, self.args.z_dimensions)).to(device).float()

		# Having created a tensor for this, copy into the tensor. 
		for j in range(self.args.batch_size):
			self.initial_skill_embedding[:len(z_sequence_collection[j]),j] = z_sequence_collection[j]

		# Now that we've gotten the initial skill embeddings (from the distinct z sequence), 
		# Feed it into the contextual LSTM, and predict new contextual embeddings. 
		contextual_outputs, contextual_hidden = self.contextual_lstm(self.initial_skill_embedding)
		# self.contextual_skill_embedding = self.z_output_layer(contextual_outputs)

		# Now recreate distributions, so we can evaluate new KL.
		self.contextual_mean = self.contextual_mean_output_layer(contextual_outputs)
		var_epsilon = 0.001
		self.contextual_variance = self.variance_factor*(self.variance_activation_layer(self.contextual_variances_output_layer(contextual_outputs))+self.variance_activation_bias) + var_epsilon
		self.contextual_dists = torch.distributions.MultivariateNormal(self.contextual_mean, torch.diag_embed(self.contextual_variance))

		if self.args.train:
			noise = torch.randn_like(self.contextual_variance)

			# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
			self.contextual_skill_embedding = (self.contextual_mean + self.contextual_variance*noise).squeeze(1)
			# Ought to be able to pass gradients through this latent_z now.

		# If evaluating, greedily get action.
		else:
			self.contextual_skill_embedding = self.contextual_mean.squeeze(1)

		######### 



		# Now must reconstruct the z vector (sampled_z_indices). # Incidentally this removes need for masking of z's.
		# Must use the original sampled_b to take care of this. 
		new_sampled_z_indices = torch.zeros_like(sampled_z_index).to(device)

		for j in range(self.args.batch_size):
			# Use distinct_z_indices, where we have already computed torch.where(sampled_b[:,j]). 
			# May need to manipulate this to negate the where. 
			for k in range(len(distinct_indices_collection[j])-1):
				new_sampled_z_indices[distinct_indices_collection[j][k]:distinct_indices_collection[j][k+1],j] = self.contextual_skill_embedding[k,j]			
			# new_sampled_z_indices[distinct_indices_collection[j][-1]:batch_trajectory_lengths[j],j] = self.contextual_skill_embedding[k+1,j]
			new_sampled_z_indices[distinct_indices_collection[j][-1]:batch_trajectory_lengths[j],j] = self.contextual_skill_embedding[len(distinct_indices_collection[j])-1,j]	
		
		# Now recompute prior_loglikelihood with the new zs. 
		prior_loglikelihood = self.standard_distribution.log_prob(new_sampled_z_indices)

		# Also recompute the KL. 
		# kl_divergence = torch.distributions.kl_divergence(self.contextual_dists, self.standard_distribution).mean()

		# Return same objects as original forward function. 
		return new_sampled_z_indices, sampled_b, variational_b_logprobabilities, \
		 variational_z_logprobabilities, variational_b_probabilities, \
		 variational_z_probabilities, kl_divergence, prior_loglikelihood

class ContinuousNewContextualVariationalPolicyNetwork(ContinuousVariationalPolicyNetwork_Batch):
	
	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):
		
		super(ContinuousNewContextualVariationalPolicyNetwork, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)
		# Define a bidirectional LSTM now.
		self.z_dimensions = self.args.z_dimensions
		self.contextual_lstm = torch.nn.LSTM(input_size=self.args.z_dimensions,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True, dropout=self.args.dropout)
		# self.z_output_layer = torch.nn.Linear(2*self.hidden_size, self.z_dimensions)

		self.contextual_mean_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.contextual_variances_output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)

	def forward(self, input, epsilon, new_z_selection=True, batch_size=None, batch_trajectory_lengths=None):
		
		#####################
		# First run the forward function of the original variational network. 
		# This runs the initial LSTM and predicts the original embedding of skills. 
		#####################

		sampled_z_index, sampled_b, variational_b_logprobabilities, \
		 variational_z_logprobabilities, variational_b_probabilities, \
		 variational_z_probabilities, kl_divergence, prior_loglikelihood = \
			 super().forward(input, epsilon, new_z_selection=new_z_selection, batch_size=batch_size, batch_trajectory_lengths=batch_trajectory_lengths)

		#####################
		# Now parse the sequence of per timestep z's to sequence of z's of length = the number of skills in the trajectory. 
		# The latent_b vector has this information, specified in terms of when b=1. 
		#####################

		# Create separate masking object.
		contextual_mask = torch.ones_like(sampled_z_index).to(device).float()	
		
		for j in range(self.args.batch_size):
			
			#####################
			# Mask z's that extend past the trajectory length. 
			#####################

			sampled_b[batch_trajectory_lengths[j]:,j] = 0
			sampled_z_index[batch_trajectory_lengths[j]:,j] = 0.
			contextual_mask[batch_trajectory_lengths[j]:,j] = 0.

			#####################
			# Get times at which we actually observe distinct z's.
			#####################

			distinct_z_indices = torch.where(sampled_b[:,j])[0].clone().detach().cpu().numpy()

			#####################
			# These mask indices index into the distinct_indices list, so the values in mask indices are positions in the list to be masked.
			# Masking strategy - uniformly randomly sample mask_fraction arbitrarily. 
			#####################

			number_mask_elements = np.ceil(len(distinct_z_indices)*self.args.mask_fraction).astype(int)
			if number_mask_elements==1 and len(distinct_z_indices)==1:
				number_mask_elements = 0				
			mask_indices = np.random.choice(range(len(distinct_z_indices)), size=number_mask_elements, replace=False)

			#####################			
			# Now actually mask the chosen mask indices.
			#####################
			for k in range(len(mask_indices)):				
				if mask_indices[k]+1 >= len(distinct_z_indices):
					end_index = contextual_mask.shape[0]
				else:
					end_index = distinct_z_indices[mask_indices[k]+1]
				contextual_mask[distinct_z_indices[mask_indices[k]]:end_index,j]  = 0

		#####################
		# Now mask the sampled input to create the masked input. 
		#####################

		self.initial_skill_embedding = contextual_mask*sampled_z_index

		#####################
		# Now that we've gotten the initial skill embeddings (from the distinct z sequence), 
		# Feed it into the contextual LSTM, and predict new contextual embeddings. 
		#####################
	
		contextual_outputs, contextual_hidden = self.contextual_lstm(self.initial_skill_embedding)
		
		#####################
		# Now recreate distributions, so we can evaluate new KL.
		#####################

		self.contextual_mean = self.contextual_mean_output_layer(contextual_outputs)
		var_epsilon = 0.001
		self.contextual_variance = self.variance_factor*(self.variance_activation_layer(self.contextual_variances_output_layer(contextual_outputs))+self.variance_activation_bias) + var_epsilon
		self.contextual_dists = torch.distributions.MultivariateNormal(self.contextual_mean, torch.diag_embed(self.contextual_variance))

		if self.args.train:
			noise = torch.randn_like(self.contextual_variance)
			# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
			self.contextual_skill_embedding = (self.contextual_mean + self.contextual_variance*noise).squeeze(1)
			# Ought to be able to pass gradients through this latent_z now.
		# If evaluating, greedily get action.
		else:
			self.contextual_skill_embedding = self.contextual_mean.squeeze(1)

		#####################
		# Since the contextual embeddings are just predicted by an LSTM, use the same technique of "NEw z selection"
		# as in the original variational network, that copies over the previous timesteps' z, if b at that timestep = 0. (i.e. continue).
		#####################

		for t in range(1,input.shape[0]):
			# If b_t==0, just use previous z. 
			# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
			# sampled_z_index[t, torch.where(sampled_b[t]==0)[0]] = sampled_z_index[t-1, torch.where(sampled_b[t]==0)[0]]
			self.contextual_skill_embedding[t, torch.where(sampled_b[t]==0)[0]] = self.contextual_skill_embedding[t-1, torch.where(sampled_b[t]==0)[0]]

		# Now recompute prior_loglikelihood with the new zs. 
		prior_loglikelihood = self.standard_distribution.log_prob(self.contextual_skill_embedding)

		# Also recompute the KL. 
		kl_divergence = torch.distributions.kl_divergence(self.contextual_dists, self.standard_distribution)

		#######
		# Try ELMO embeddings.
		if self.args.ELMO_embeddings:
			self.elmo_contextual_skill_embedding = self.contextual_skill_embedding + sampled_z_index
		else:
			# If not using ELMO embedding, just use the newly predicted ones.
			self.elmo_contextual_skill_embedding = self.contextual_skill_embedding

		# Return same objects as original forward function. 
		return self.elmo_contextual_skill_embedding, sampled_b, variational_b_logprobabilities, \
		 variational_z_logprobabilities, variational_b_probabilities, \
		 variational_z_probabilities, kl_divergence, prior_loglikelihood

class EncoderNetwork(PolicyNetwork_BaseClass):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies=4, batch_size=1, args=None):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(EncoderNetwork, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.number_subpolicies = number_subpolicies
		self.args = args
		self.batch_size = self.args.batch_size
		self.num_layers = self.args.number_layers		

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True, dropout=self.args.dropout)

		# Define output layers for the LSTM, and activations for this output layer. 

		# Because it's bidrectional, once we compute <outputs, hidden = self.lstm(input)>, we must concatenate: 
		# From reverse LSTM: <outputs[0,:,hidden_size:]> and from the forward LSTM: <outputs[-1,:,:hidden_size]>.
		# (Refer - https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66 )
		# Because of this, the output layer must take in size 2*hidden.
		self.hidden_layer = torch.nn.Linear(2*self.hidden_size, 2*self.hidden_size)
		self.output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)		

		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)

	def forward(self, input, epsilon=0.0001):
		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array. 		
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input)

		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,self.batch_size,-1))	

		# Calculate preprobs.
		preprobabilities = self.output_layer(self.hidden_layer(concatenated_outputs))
		probabilities = self.batch_softmax_layer(preprobabilities)
		logprobabilities = self.batch_logsoftmax_layer(preprobabilities)

		latent_z = self.select_epsilon_greedy_action(probabilities, epsilon=epsilon)

		# Return latentz_encoding as output layer of last outputs. 
		return latent_z, logprobabilities, None, None

	def get_probabilities(self, input):
		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array. 		
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input)

		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,self.batch_size,-1))	

		# Calculate preprobs.
		preprobabilities = self.output_layer(self.hidden_layer(concatenated_outputs))
		probabilities = self.batch_softmax_layer(preprobabilities)
		logprobabilities = self.batch_logsoftmax_layer(preprobabilities)

		# Return latentz_encoding as output layer of last outputs. 
		return logprobabilities, probabilities

class OldContinuousEncoderNetwork(PolicyNetwork_BaseClass):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, args, batch_size=1):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(OldContinuousEncoderNetwork, self).__init__()

		self.args = args
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = self.args.var_number_layers
		self.batch_size = self.args.batch_size

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True)

		# Define output layers for the LSTM, and activations for this output layer. 

		# # Because it's bidrectional, once we compute <outputs, hidden = self.lstm(input)>, we must concatenate: 
		# # From reverse LSTM: <outputs[0,:,hidden_size:]> and from the forward LSTM: <outputs[-1,:,:hidden_size]>.
		# # (Refer - https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66 )
		# # Because of this, the output layer must take in size 2*hidden.
		# self.hidden_layer = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
		# self.output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)		

		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.

		self.variance_factor = 0.01

	def forward(self, input, epsilon=0.001, z_sample_to_evaluate=None):
		# This epsilon passed as an argument is just so that the signature of this function is the same as what's provided to the discrete encoder network.

		# Input format must be: Sequence_Length x Batch_Size x Input_Size. 
		# Assuming input is a numpy array.
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input)

		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,self.batch_size,-1))

		# Predict Gaussian means and variances. 
		# if self.args.mean_nonlinearity:
		# 	mean_outputs = self.activation_layer(self.mean_output_layer(concatenated_outputs))
		# else:
		mean_outputs = self.mean_output_layer(concatenated_outputs)
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(concatenated_outputs))+self.variance_activation_bias) + epsilon

		dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		# Whether to use reparametrization trick to retrieve the 
		if self.args.reparam:
			noise = torch.randn_like(variance_outputs)

			# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
			latent_z = mean_outputs + variance_outputs * noise
			# Ought to be able to pass gradients through this latent_z now.

		else:
			# Retrieve sample from the distribution as the value of the latent variable.
			latent_z = dist.sample()
		# calculate entropy for training.
		entropy = dist.entropy()
		# Also retrieve log probability of the same.
		logprobability = dist.log_prob(latent_z)

		# Set standard distribution for KL. 
		self.standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).to(device),torch.eye((self.output_size)).to(device))
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(dist, self.standard_distribution)

		if self.args.debug:
			print("###############################")
			print("Embedding in Encoder Network.")
			embed()

		if z_sample_to_evaluate is None:

			print("Embedding in forward of Old continuous encoder")
			embed()
			return latent_z, logprobability, entropy, kl_divergence

		else:
			logprobability = dist.log_prob(z_sample_to_evaluate)
			return logprobability

class ContinuousEncoderNetwork(PolicyNetwork_BaseClass):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, args, batch_size=1):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousEncoderNetwork, self).__init__()

		# Define dimensions for both robot and environment streams. 
		self.define_dimensions(input_size, hidden_size, output_size, args)		

		# Define layers. 
		self.define_layers()

		# Instantiate networks. 
		self.instantiate_networks()

	def define_dimensions(self, input_size, hidden_size, output_size, args):

		##############################
		# Setup state sizes etc.
		##############################

		self.args = args

		# Define state sizes for each partition of state space.
		# Keep track of robot input and output state size. 
		self.size_dict = {}
		self.size_dict['state_size'] = int(input_size/2)
		self.size_dict['input_size'] = input_size
		self.size_dict['output_size'] = output_size

		# Other layers.
		self.num_layers = self.args.var_number_layers
		self.hidden_size = hidden_size
		self.batch_size = self.args.batch_size 

	def define_layers(self):

		# Defines layers shared across both streams.
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.
		self.variance_factor = self.args.variance_factor

	def define_networks(self, input_size, output_size):
		
		# Define a bidirectional LSTM now.
		lstm = torch.nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True).to(device)

		# Define output layers for the LSTM, and activations for this output layer. 
		mean_output_layer = torch.nn.Linear(2*self.hidden_size, output_size).to(device)
		variances_output_layer = torch.nn.Linear(2*self.hidden_size, output_size).to(device)

		return lstm, mean_output_layer, variances_output_layer

	def instantiate_networks(self):

		self.network_dict = torch.nn.ModuleDict()
		self.network_dict['lstm'], self.network_dict['mean_output_layer'], self.network_dict['variances_output_layer'] = self.define_networks(self.size_dict['input_size'], self.size_dict['output_size'])

	def forward(self, input, epsilon=0.0001, network_dict=None, size_dict=None, z_sample_to_evaluate=None, artificial_batch_size=None):

		##############################
		# Set default inputs.
		##############################

		if network_dict is None:
			network_dict = self.network_dict
		if size_dict is None:
			size_dict = self.size_dict

		##############################
		# Format input to the appropriate size: Sequence_Length x Batch_Size x Input_Size. 
		##############################

		batch_size = self.batch_size
		if artificial_batch_size is not None:
			batch_size = artificial_batch_size
		format_input = input.view((input.shape[0], batch_size, size_dict['input_size']))

		##############################
		# Forward pass through LSTM. 
		##############################
				
		outputs, hidden = network_dict['lstm'](format_input)
		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,batch_size,-1))

		##############################
		# Predict Gaussian means and variances. 
		##############################

		mean_outputs = network_dict['mean_output_layer'](concatenated_outputs)

		# if self.args.constant_variance:
		# 	variance_outputs = self.args.variance_value*torch.ones_like(mean_outputs).to(device) + epsilon
		# else:
		variance_outputs = self.variance_factor*(self.variance_activation_layer(network_dict['variances_output_layer'](concatenated_outputs))+self.variance_activation_bias) + epsilon
		

		dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))
		noise = torch.randn_like(variance_outputs)

		##############################
		# Get Latent Z.
		##############################
		
		# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise), can pass gradients through it. 
		latent_z = mean_outputs + variance_outputs * noise 

		# calculate entropy for training.
		entropy = dist.entropy()
		# Also retrieve log probability of the same.
		logprobability = dist.log_prob(latent_z)

		##############################
		# Set standard distribution for KL. 
		##############################

		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((size_dict['output_size'])).to(device),torch.eye((size_dict['output_size'])).to(device))
		kl_divergence = torch.distributions.kl_divergence(dist, standard_distribution)

		if self.args.debug:
			print("###############################")
			print("Embedding in Encoder Network.")
			embed()

		if z_sample_to_evaluate is None:
			return latent_z, logprobability, entropy, kl_divergence

		else:
			logprobability = dist.log_prob(z_sample_to_evaluate)
			return logprobability

class ContinuousFactoredEncoderNetwork(ContinuousEncoderNetwork):

	def __init__(self, input_size, hidden_size, output_size, args, batch_size=1):
		
		# Using its own init function.
		super(ContinuousFactoredEncoderNetwork, self).__init__(input_size, hidden_size, output_size, args)

		# Define indices. 
		self.robot_indices = np.concatenate([np.arange(0,self.robot_size_dict['state_size']), \
			np.arange(self.robot_size_dict['state_size']+self.env_size_dict['state_size'],2*self.robot_size_dict['state_size']+self.env_size_dict['state_size'])])
	
		self.env_indices = np.concatenate([ np.arange(self.robot_size_dict['state_size'],self.robot_size_dict['state_size']+self.env_size_dict['state_size']), \
			np.arange( self.robot_size_dict['state_size']*2+self.env_size_dict['state_size'], self.robot_size_dict['state_size']*2+self.env_size_dict['state_size']*2) ])

	def define_dimensions(self, input_size, hidden_size, output_size, args):

		##############################
		# Setup state sizes etc.
		##############################

		self.args = args


		# Define state sizes for each partition of state space.
		# Keep track of robot input and output state size. 
		self.robot_size_dict = {}
		self.robot_size_dict['state_size'] = self.args.robot_state_size
		self.robot_size_dict['input_size'] = 2*self.robot_size_dict['state_size']
		self.robot_size_dict['output_size'] = int(self.args.z_dimensions/2)

		# Keep track of env. input and output state size. 
		self.env_size_dict = {}
		self.env_size_dict['state_size'] = self.args.env_state_size
		self.env_size_dict['input_size'] = 2*self.env_size_dict['state_size']
		self.env_size_dict['output_size'] = int(self.args.z_dimensions/2)

		# Other layers.
		self.num_layers = self.args.var_number_layers
		self.hidden_size = hidden_size
		self.batch_size = self.args.batch_size 

	def instantiate_networks(self):		

		# Define networks for robot stream.
		self.robot_network_dict = torch.nn.ModuleDict()
		self.robot_network_dict['lstm'], self.robot_network_dict['mean_output_layer'], self.robot_network_dict['variances_output_layer'] = self.define_networks(self.robot_size_dict['input_size'], self.robot_size_dict['output_size'])

		# Define networks for environment stream.
		self.env_network_dict = torch.nn.ModuleDict()
		self.env_network_dict['lstm'], self.env_network_dict['mean_output_layer'], self.env_network_dict['variances_output_layer'] = self.define_networks(self.env_size_dict['input_size'], self.env_size_dict['output_size'])

	def split_stream_inputs(self, input):

		# Split input between robot and environment streams. 
		# The joint stream input is of size: (Time Dimensions) x (Batch Size) x (Robot state size + Env state size + Robot action size + Env action size). 

		robot_input = input[:,:,self.robot_indices]
		env_input = input[:,:,self.env_indices]

		return robot_input, env_input

	def run_super_forward(self, input, epsilon=0.001, network_dict={}, size_dict={}, z_sample_to_evaluate=None, artificial_batch_size=None):
		# robot_latent_z, robot_logprob, robot_entropy, robot_kl_divergence = super().forward(robot_input, epsilon, network_dict=self.robot_network_dict, size_dict=self.robot_size_dict, z_sample_to_evaluate=robot_z_sample)

		return super().forward(input, epsilon=epsilon, network_dict=self.robot_network_dict, size_dict=self.robot_size_dict, z_sample_to_evaluate=z_sample_to_evaluate, artificial_batch_size=artificial_batch_size)

	def get_environment_input_representation(self, env_input):
		
		
		# Dummy function that we can override in the case of the soft object.
		return env_input

	def forward(self, input, epsilon=0.001, network_dict={}, size_dict={}, z_sample_to_evaluate=None):

		# (1) Split input. 
		# (2) Run forward on each stream. 
		#	# (2a) Run forward on robot stream. 
		#	# (2b) Run forward on env stream. 
		# (3) Aggregate stream outputs. 
		# (4) Return. 

		##################################
		# (1) Split Inputs. 
		##################################

		robot_input, env_input = self.split_stream_inputs(input)
		if z_sample_to_evaluate is not None:
			robot_z_sample, env_z_sample = z_sample_to_evaluate[:self.robot_size_dict['output_size']], z_sample_to_evaluate[self.robot_size_dict['output_size']:]
		else:
			robot_z_sample, env_z_sample = None, None
		
		##################################
		# (2) Run forward on each stream. 
		##################################

		if z_sample_to_evaluate is None:
			# (2a) Run forward on robot stream.				
			robot_latent_z, robot_logprob, robot_entropy, robot_kl_divergence = super().forward(robot_input, epsilon, network_dict=self.robot_network_dict, size_dict=self.robot_size_dict, z_sample_to_evaluate=robot_z_sample)

			new_env_input = self.get_environment_input_representation(env_input)

			# (2b) Run forward on env stream.							
			env_latent_z, env_logprob, env_entropy, env_kl_divergence = super().forward(new_env_input, epsilon, network_dict=self.env_network_dict, size_dict=self.env_size_dict, z_sample_to_evaluate=env_z_sample)

			##################################
			# (3) Aggregate stream outputs. 
			##################################
					
			# Remember, robot_z and env_z are both [1 x Batch_Size x (2 x Z_Dimensions)]. Concatenate across last dimension. 
			concatenated_latent_z = torch.cat([robot_latent_z, env_latent_z],axis=-1)

			# Aggregate log probabilities.
			aggregated_logprobability = robot_logprob + env_logprob

			# Aggregate KL Divergence
			aggregated_kl_divergence = robot_kl_divergence + env_kl_divergence

			# Aggregated entropy. 
			aggregated_entropy = robot_entropy + env_entropy

			##################################
			# (4) Return. 
			##################################

			return concatenated_latent_z, aggregated_logprobability, aggregated_entropy, aggregated_kl_divergence		

		else: 

			# (2a) Run forward on robot stream.				
			robot_logprob = super().forward(robot_input, epsilon, network_dict=self.robot_network_dict, size_dict=self.robot_size_dict, z_sample_to_evaluate=robot_z_sample)

			# (2b) Run forward on env stream.		
			env_logprob = super().forward(env_input, epsilon, network_dict=self.env_network_dict, size_dict=self.env_size_dict, z_sample_to_evaluate=env_z_sample)

			# Aggregate log probabilities.
			aggregated_logprobability = robot_logprob + env_logprob

			##################################
			# (4) Return. 
			##################################

			return concatenated_latent_z, aggregated_logprobability, aggregated_entropy, aggregated_kl_divergence		

class ContinuousSoftEncoderNetwork(ContinuousEncoderNetwork):

	def __init__(self, input_size, hidden_size, output_size, args, batch_size=1):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousSoftEncoderNetwork, self).__init__()

		sys.path.append('/home/tshankar/Research/Code/PointMAE')

		self.define_soft_input_layer()

	def define_soft_input_layer(self):

		import deploy_model

		self.PointMAE_model = deploy_model.return_model()

	def forward(self, input, epsilon=0.001, network_dict=None, size_dict=None, z_sample_to_evaluate=None, artificial_batch_size=None):
		return super().forward(input, epsilon, network_dict, size_dict, z_sample_to_evaluate, artificial_batch_size)
	
######################################
# Backing up pattern
######################################

# import sys 
# sys.path.append("/home/tshankar/Research/Code/PointMAE")

# import deploy_model
# import torch

# pmm = deploy_model.return_model()
# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# bs = 8
# ptz_size = 512
# ptz = torch.randn(14,bs,ptz_size,3).to(device)

# dpmm = torch.nn.DataParallel(pmm).to(device)

# dpmm(ptz.reshape(-1,ptz_size,3))


######################################
######################################

class ContinuousFactoredSoftEncoderNetwork(ContinuousFactoredEncoderNetwork):

	def __init__(self, input_size, hidden_size, output_size, args, batch_size=1):

		# Using its own init function.
		super(ContinuousFactoredSoftEncoderNetwork, self).__init__(input_size, hidden_size, output_size, args)

		self.define_soft_input_layer()

	def define_soft_input_layer(self):

		sys.path.append('/home/tshankar/Research/Code/PointMAE')
		import deploy_model

		self.PointMAE_model = deploy_model.return_model()
		# Also get the unmasked encoder.
		# self.PointMAE_unmasked_encoder = self.PointMAE_model.MAE_encoder.encoder
		
		self.n_patches = 26
		self.pmae_representation_size = 384
		self.hidden_size_1 = 64
		self.hidden_size_2 = 16
		self.maxpool_kernel_size = 6
		self.representation_size = (self.n_patches//self.maxpool_kernel_size)*self.hidden_size_2
		self.pc_linear_layer_1 = torch.nn.Linear(self.pmae_representation_size, self.hidden_size_1)		
		self.pc_linear_layer_2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)		
		# The max pool layer works on 4D. 
		self.maxpoollayer_1 = torch.nn.MaxPool2d((self.maxpool_kernel_size,1))

	# def post_process_linear_layer
	def get_environment_input_representation(self, input_point_cloud):
	# def compute_point_cloud_representation(self, input_point_cloud):
		
		# For point cloud stream, run PointMAE model forward, get the representation.. 
		# Assume we have element point_cloud.

		# We have two options for the representation of the pointcloud. 
		# Option #1: Find a representation of the all point patches.
		# Option #2: Use representation of visible point patches.
		# The Point-MAE paper has tested out reconstructing entire point clouds from the visible point clouds
		# is perfectly successful, therefore is a useful representation. 
		
		# self.point_cloud_representation = self.PointMAE_unmasked_encoder(input_point_cloud)
		# If 4Dimensional, reshape. 
		if len(input_point_cloud.shape)>3:
			reshaped_point_cloud = input_point_cloud.reshape(-1,input_point_cloud.shape[2],input_point_cloud.shape[3])
			# Actually run representation. Reshaping is fine because it runs independently across batches.
			reshaped_representation = self.PointMAE_model.encoder_forward(reshaped_point_cloud)
			# Reshape representation back.
			pointMAE_representation = reshaped_representation.reshape(input_point_cloud.shape[0], input_point_cloud.shape[1], self.n_patches, self.pmae_representation_size)
		else:		
			# Actually run representation. Reshaping is fine because it runs independently across batches.
			pointMAE_representation = self.PointMAE_model.encoder_forward(input_point_cloud)

		# # 
		# reshaped_point_cloud = input_point_cloud.reshape(-1, input_point_cloud.shape[-2], input_point_cloud.shape[-1])
		# reshaped_representation = self.PointMAE_model.encoder_forward(reshaped_point_cloud)
		# pointMAE_representation = reshaped_representation.reshape()

		# Linear transform 1. 
		transformed_pmae_representation = self.pc_linear_layer_1(pointMAE_representation)
		# MaxPool
		maxpooled_representation = self.maxpoollayer_1(transformed_pmae_representation)
		# Linear transform 2. 
		pointcloud_representation = self.pc_linear_layer_2(maxpooled_representation).reshape(-1,self.representation_size)
		
		return pointcloud_representation
					

class CriticNetwork(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(CriticNetwork, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.number_layers = number_layers
		self.batch_size = 1

		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.number_layers)		

		self.output_layer = torch.nn.Linear(self.hidden_size,self.output_size)		

	def forward(self, input):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))

		hidden = None
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict critic value for each timestep. 
		critic_value = self.output_layer(lstm_outputs)		

		return critic_value

class ContinuousMLP(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(ContinuousMLP, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.args = args

		self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
		self.hidden_layer1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.hidden_layer2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.hidden_layer3 = torch.nn.Linear(self.hidden_size, self.hidden_size)	
		
		if self.args.small_translation_model:
			self.mean_output_layer = torch.nn.Linear(self.input_size,self.output_size)
			self.variances_output_layer = torch.nn.Linear(self.input_size,self.output_size)
		else:
			self.mean_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
			self.variances_output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		self.variance_factor = 0.01
		self.variance_activation_bias = 0.
		

		if self.args.leaky_relu:			
			self.relu_activation = torch.nn.LeakyReLU()
		else:
			self.relu_activation = torch.nn.ReLU()

		self.variance_activation_layer = torch.nn.Softplus()
		
		# self.dropout_layer = torch.nn.Dropout(self.args.mlp_dropout)
		# Don't use dropout for now...
		self.dropout_layer = torch.nn.Dropout(self.args.dropout)
		
		if self.args.batch_norm:
			self.batch_norm_layer1 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer2 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer3 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer4 = torch.nn.BatchNorm1d(self.hidden_size)


	def forward(self, input, greedy=False, action_epsilon=0.0001):

		# Assumes input is Batch_Size x Input_Size.			
		if self.args.small_translation_model:
			# final_layer = self.input_layer(input)
			# Special input to output layer.. 
			self.mean_outputs = self.mean_output_layer(input)		
			self.variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(input))+self.variance_activation_bias) + action_epsilon

		else:

			if self.args.batch_norm:		
				s1 = input.shape[0]
				if len(input.shape)==3:					
					s2 = input.shape[1]				
				else:
					s2 = 1

				h1 = self.dropout_layer(self.relu_activation(self.batch_norm_layer1( self.input_layer(input).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
				h2 = self.dropout_layer(self.relu_activation(self.batch_norm_layer2( self.hidden_layer1(h1).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
				h3 = self.dropout_layer(self.relu_activation(self.batch_norm_layer3( self.hidden_layer2(h2).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
				h4 = self.dropout_layer(self.relu_activation(self.batch_norm_layer4( self.hidden_layer3(h3).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
				h4 = h4.squeeze(1)

			else:
				h1 = self.dropout_layer(self.relu_activation(self.input_layer(input)))
				h2 = self.dropout_layer(self.relu_activation(self.hidden_layer1(h1)))
				h3 = self.dropout_layer(self.relu_activation(self.hidden_layer2(h2)))
				h4 = self.dropout_layer(self.relu_activation(self.hidden_layer3(h3)))

			final_layer = h4
		
			self.mean_outputs = self.mean_output_layer(final_layer)		
			self.variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(final_layer))+self.variance_activation_bias) + action_epsilon

		# self.variance_value = 1e-5
		self.variance_value = 0.05
		self.variance_outputs = self.variance_value*torch.ones_like(self.mean_outputs).to(device).float()

		noise = torch.randn_like(self.variance_outputs)
			
		if greedy: 
			action = self.mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			action = self.mean_outputs + self.variance_outputs * noise

		if self.args.residual_translation:
			return action+input
		else:
			return action

	def reparameterized_get_actions(self, input, greedy=False, action_epsilon=0.0001):
		return self.forward(input, greedy, action_epsilon)

	def get_probabilities(self, input, evaluate_value, action_epsilon):

		# Run forward to set the variance and mean values. 
		_ = self.forward(input, action_epsilon=action_epsilon)

		# Create distribution. 
		self.dists = torch.distributions.MultivariateNormal(self.mean_outputs, torch.diag_embed(self.variance_outputs))
		
		# Evaluate logprobability.
		return self.dists.log_prob(evaluate_value)

class CriticMLP(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(CriticMLP, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_size = 1
		self.args = args

		self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
		self.hidden_layer1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.hidden_layer2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.hidden_layer3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		if self.args.leaky_relu:			
			self.relu_activation = torch.nn.LeakyReLU()
		else:
			self.relu_activation = torch.nn.ReLU()

		self.dropout_layer = torch.nn.Dropout(self.args.mlp_dropout)

		if self.args.batch_norm:
			self.batch_norm_layer1 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer2 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer3 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer4 = torch.nn.BatchNorm1d(self.hidden_size)


	def forward(self, input, greedy=False, action_epsilon=0.0001):

		# Assumes input is Batch_Size x Input_Size.
		if self.args.batch_norm:		
			s1 = input.shape[0]
			if len(input.shape)==3:					
				s2 = input.shape[1]				
			else:
				s2 = 1

			h1 = self.dropout_layer(self.relu_activation(self.batch_norm_layer1( self.input_layer(input).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h2 = self.dropout_layer(self.relu_activation(self.batch_norm_layer2( self.hidden_layer1(h1).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h3 = self.dropout_layer(self.relu_activation(self.batch_norm_layer3( self.hidden_layer2(h2).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h4 = self.dropout_layer(self.relu_activation(self.batch_norm_layer4( self.hidden_layer3(h3).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h4 = h4.squeeze(1)

		else:
			h1 = self.dropout_layer(self.relu_activation(self.input_layer(input)))
			h2 = self.dropout_layer(self.relu_activation(self.hidden_layer1(h1)))
			h3 = self.dropout_layer(self.relu_activation(self.hidden_layer2(h2)))
			h4 = self.dropout_layer(self.relu_activation(self.hidden_layer3(h3)))

		# Predict critic value for each timestep. 
		critic_value = self.output_layer(h4)		

		return critic_value

class DiscreteMLP(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(DiscreteMLP, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.args = args 

		self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
		self.hidden_layer1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.hidden_layer2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.hidden_layer3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		if self.args.leaky_relu:			
			self.relu_activation = torch.nn.LeakyReLU()
		else:
			self.relu_activation = torch.nn.ReLU()

		self.dropout_layer = torch.nn.Dropout(self.args.mlp_dropout)

		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)		


		if self.args.batch_norm:
			self.batch_norm_layer1 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer2 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer3 = torch.nn.BatchNorm1d(self.hidden_size)
			self.batch_norm_layer4 = torch.nn.BatchNorm1d(self.hidden_size)

	def forward(self, input):
				
		# Assumes input is Batch_Size x Input_Size.
		if self.args.batch_norm:		
			s1 = input.shape[0]
			if len(input.shape)==3:					
				s2 = input.shape[1]				
			else:
				s2 = 1

			h1 = self.dropout_layer(self.relu_activation(self.batch_norm_layer1( self.input_layer(input).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h2 = self.dropout_layer(self.relu_activation(self.batch_norm_layer2( self.hidden_layer1(h1).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h3 = self.dropout_layer(self.relu_activation(self.batch_norm_layer3( self.hidden_layer2(h2).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h4 = self.dropout_layer(self.relu_activation(self.batch_norm_layer4( self.hidden_layer3(h3).view(-1,self.hidden_size) ).view(s1, s2, self.hidden_size) ))
			h4 = h4.squeeze(1)

		else:
			h1 = self.dropout_layer(self.relu_activation(self.input_layer(input)))
			h2 = self.dropout_layer(self.relu_activation(self.hidden_layer1(h1)))
			h3 = self.dropout_layer(self.relu_activation(self.hidden_layer2(h2)))
			h4 = self.dropout_layer(self.relu_activation(self.hidden_layer3(h3)))

		# Compute preprobability with output layer.
		preprobability_outputs = self.output_layer(h4)
		
		# Compute probabilities and logprobabilities.
		log_probabilities = self.batch_logsoftmax_layer(preprobability_outputs)
		probabilities = self.batch_softmax_layer(preprobability_outputs)

		return log_probabilities, probabilities

	def get_probabilities(self, input):

		return self.forward(input)

def mlp(sizes, activation, output_activation=torch.nn.Identity):
    """
    Build a multi-layer perceptron in PyTorch.

    Args:
        sizes: Tuple, list, or other iterable giving the number of units
            for each layer of the MLP. 

        activation: Activation function for all layers except last.

        output_activation: Activation function for last layer.

    Returns:
        A PyTorch module that can be called to give the output of the MLP.
        (Use an nn.Sequential module.)

    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j+1]), act()]
    return torch.nn.Sequential(*layers)

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(torch.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return pre_sum.sum(axis=-1)

class MLPGaussianActor(torch.nn.Module):


    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)      
        self.std_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.softplus_activation = torch.nn.Softplus()

    def forward(self, obs, act=None):

        # Create mean and var. 
        mean = self.mu_net(obs)
        standard_deviation = self.softplus_activation(self.std_net(obs))

        # Distribution. 
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(standard_deviation))

        log_prob = None
        if act is not None:
            log_prob = dist.log_prob(act)

        return dist, log_prob

