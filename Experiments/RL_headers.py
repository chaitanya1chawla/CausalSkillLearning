from spinup.exercises.pytorch.problem_set_1 import exercise1_1
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary		
from PolicyNetworks import mlp, MLPGaussianActor
from spinup import ppo_pytorch as ppo
# from spinup.algos.pytorch.ppo.hierarchical_ppo import hierarchical_ppo
from spinup.algos.pytorch.ppo.new_hierarchical_ppo import hierarchical_ppo
from spinup.exercises.common import print_result
from functools import partial
import gym, os, pandas as pd, psutil, time
from spinup.utils.test_policy import load_policy_and_env
# if self.args.mujoco:
#     import robosuite
#     from robosuite.wrappers import GymWrapper

from torch.optim import Adam
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# from PolicyNetworks import ContinuousPolicyNetwork
from gym.spaces import Box, Discrete

import robosuite
from robosuite.wrappers import GymWrapper
from spinup.exercises.pytorch.problem_set_1 import exercise1_1
from spinup.exercises.pytorch.problem_set_1 import exercise1_2_auxiliary		
from PolicyNetworks import mlp, MLPGaussianActor
from spinup import ppo_pytorch as ppo
# from spinup.algos.pytorch.ppo.hierarchical_ppo import hierarchical_ppo
from spinup.algos.pytorch.ppo.new_hierarchical_ppo import hierarchical_ppo
from spinup.exercises.common import print_result
from functools import partial
import gym, os, pandas as pd, psutil, time
from spinup.utils.test_policy import load_policy_and_env
# if self.args.mujoco:
#     import robosuite
#     from robosuite.wrappers import GymWrapper

from torch.optim import Adam
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# from PolicyNetworks import ContinuousPolicyNetwork
from gym.spaces import Box, Discrete

import robosuite
from robosuite.wrappers import GymWrapper
import imageio
