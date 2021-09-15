from headers import *
from PolicyNetworks import *
from RL_headers import *

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_printoptions(sci_mode=False, precision=2)

class PPOBuffer:
	"""
	A buffer for storing trajectories experienced by a PPO agent interacting
	with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
	for calculating the advantages of state-action pairs.
	"""

	def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
		self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)

		# CHANGE: Probably need to make this a more generic data structure to handle tuples. 
		# self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

		# CHANGED: Making this a list (of arbitrary elements) of length = size. 
		# Since the buffer object itself is just length size for a particular epoch.
		self.act_buf = [[] for i in range(size)]

		self.adv_buf = np.zeros(size, dtype=np.float32)
		self.rew_buf = np.zeros(size, dtype=np.float32)
		self.ret_buf = np.zeros(size, dtype=np.float32)
		self.val_buf = np.zeros(size, dtype=np.float32)
		self.logp_buf = np.zeros(size, dtype=np.float32)
		self.gamma, self.lam = gamma, lam
		self.ptr, self.path_start_idx, self.max_size = 0, 0, size

	def store(self, obs, act, rew, val, logp):
		"""
		Append one timestep of agent-environment interaction to the buffer.
		"""
		assert self.ptr < self.max_size     # buffer has to have room so you can store
		self.obs_buf[self.ptr] = obs

		# CHANGE: May possibly need to change this.   
		# CHANGED: Just need to torchify these actions before storing them. 
		# What we need to worry about is... torch taking up GPU space! Instead, storing them in normal RAM maybe better? 
		# torch_act = [torch.as_tensor(x, dtype=torch.float32) for x in act]
		# self.act_buf[self.ptr] = torch_act

		# CHanging back! Don't know what we were doing earlier
		self.act_buf[self.ptr] = act        
		self.rew_buf[self.ptr] = rew
		self.val_buf[self.ptr] = val
		self.logp_buf[self.ptr] = logp
		self.ptr += 1

	def finish_path(self, last_val=0):
		"""
		Call this at the end of a trajectory, or when one gets cut off
		by an epoch ending. This looks back in the buffer to where the
		trajectory started, and uses rewards and value estimates from
		the whole trajectory to compute advantage estimates with GAE-Lambda,
		as well as compute the rewards-to-go for each state, to use as
		the targets for the value function.

		The "last_val" argument should be 0 if the trajectory ended
		because the agent reached a terminal state (died), and otherwise
		should be V(s_T), the value function estimated for the last state.
		This allows us to bootstrap the reward-to-go calculation to account
		for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
		"""

		path_slice = slice(self.path_start_idx, self.ptr)
		rews = np.append(self.rew_buf[path_slice], last_val)
		vals = np.append(self.val_buf[path_slice], last_val)
		
		# the next two lines implement GAE-Lambda advantage calculation
		deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
		self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
		
		# the next line computes rewards-to-go, to be targets for the value function
		self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
		
		self.path_start_idx = self.ptr

	def get(self):
		"""
		Call this at the end of an epoch to get all of the data from
		the buffer, with advantages appropriately normalized (shifted to have
		mean zero and std one). Also, resets some pointers in the buffer.
		"""
		# assert self.ptr == self.max_size    # buffer has to be full before you can get
		# self.ptr, self.path_start_idx = 0, 0
	
		# Select the first ptr elements of the buffer. 
		valid_adv_buf = self.adv_buf[:self.ptr]
		valid_obs_buf = self.obs_buf[:self.ptr]
		valid_act_buf = self.act_buf[:self.ptr]
		valid_ret_buf = self.ret_buf[:self.ptr]
		valid_logp_buf = self.logp_buf[:self.ptr]
		# Now set ptr to 0.
		self.ptr, self.path_start_idx = 0, 0

		# the next two lines implement the advantage normalization trick
		adv_mean, adv_std = mpi_statistics_scalar(valid_adv_buf)
		valid_adv_buf = (valid_adv_buf - adv_mean) / adv_std
		data = dict(obs=valid_obs_buf, act=valid_act_buf, ret=valid_ret_buf,
					adv=valid_adv_buf, logp=valid_logp_buf)
		
		# print("Embed in get")
		# embed()

		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

