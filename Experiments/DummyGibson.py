import os
import json
import h5py
import argparse
import imageio
import numpy as np
import copy

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

##########################################
##########################################

# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

dp = "/data/tanmayshankar/Datasets/MOMART/table_cleanup_to_dishwasher/expert/table_cleanup_to_dishwasher_expert.hdf5"

dummy_spec = dict(
    obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
)
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dp)
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)

##########################################
##########################################

x = np.load("/data/tanmayshankar/Datasets/MOMART/table_cleanup_to_dishwasher/New_Task_Demo_Array.npy", allow_pickle=True)

image_list = []

    def set_state(v):
        sd = {}
        sd['states'] = v
        return sd
# def set_state(v):

#     sd = {}
#     sd['states'] = v

for k, v in enumerate(x[0]['flat-state'][::20]):
    print("State:",k)

    sd = set_state(v)
    env.reset_to(sd)
    image = env.render(mode='rgb', camera_name='rgb', height=400, width=400)
    image_list.append(image)

gif_name = "/data/tanmayshankar/TrainingLogs/Buhtinky.gif"
image_array = np.array(image_list)

##########################################
##########################################

# imageio.v3.imwrite(gif_name, image_list, loop=0)
imageio.v3.imwrite(gif_name, image_array, loop=0)
