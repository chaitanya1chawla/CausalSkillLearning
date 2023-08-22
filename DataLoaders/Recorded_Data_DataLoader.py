from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *


def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]


class Recorded_Data_PreDataset(Dataset):
	def __init__(self, args, ) -> None:
        super().__init__() 
        self.args = args
        if self.args.datadir is None:
            self.dataset_directory = '/scratch/cchawla/rosbag_trials'
	    