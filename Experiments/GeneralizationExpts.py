label_dict = {}
label_dict['000'] = 'PickRight'
label_dict['006'] = 'BoxOpen'
label_dict['008'] = 'ReachRightDrawer'
label_dict['010'] = 'SidewaysBoxOpen'
label_dict['011'] = 'LeftDrawerOpen'
label_dict['012'] = 'PickLeft'
label_dict['013'] = 'Reach'
label_dict['015'] = 'ReachCup'
label_dict['017'] = 'Return'
label_dict['020'] = 'Stir'
label_dict['022'] = 'Place'
label_dict['023'] = 'Place'
label_dict['027'] = 'Stir'
label_dict['034'] = 'MoveCup'
label_dict['036'] = 'ReturnfromRightDrawer'
label_dict['038'] = 'Release'
label_dict['043'] = 'ReturnfromCup'
label_dict['044'] = 'ReachBoxFront'
label_dict['047'] = 'ReachLeftDrawer'
label_dict['050'] = 'PlaceRelease'
label_dict['051'] = 'ReachSpoon'
label_dict['053'] = 'Pour'
label_dict['056'] = 'PickLeft'
label_dict['057'] = 'FinishPour'
label_dict['058'] = 'GraspSpoon'

################################
# For Z_index = 38, Releasing GRipper from cup.
################################
n_samples = 5
state_dim = 21
start_state_set = np.zeros((n_samples, 1, state_dim))
traj_sample_indices = np.random.randint(0, high=95, size=n_samples)

z_index = 38
z_to_replicate = self.latent_z_set[z_index]

for k, v in enumerate(traj_sample_indices):
    start_state_set[k] = self.trajectory_set[v][0]

rollout_list = []
for k, start_state in enumerate(start_state_set):
    traj = self.retrieve_unnormalized_robot_trajectory(start_state, z_to_replicate, rollout_length=14)
    rollout_list.append(traj)

norm_gt = self.trajectory_set[z_index]*self.norm_denom_value + self.norm_sub_value

np.save("Traj{0}_GT.npy".format(str(z_index).zfill(3)), norm_gt)
for k, v in enumerate(traj_sample_indices):
    np.save("Traj_Z{0}_RolloutStartIndex{1}.npy".format(str(z_index).zfill(3), str(v).zfill(3)), rollout_list[k])

# self.traj_dir_name, "Traj{0}_GT.npy".format(kstr)), gt_traj_tuple)

################################
# For Z_index = 57, FinishedPouring
################################
n_samples = 4
state_dim = 21
start_state_set = np.zeros((n_samples, 1, state_dim))
traj_sample_indices = np.random.randint(0, high=95, size=n_samples)
traj_sample_indices = np.array([66, 77, 23, 41])
z_index = 57
z_to_replicate = self.latent_z_set[z_index]

for k, v in enumerate(traj_sample_indices):
    start_state_set[k] = self.trajectory_set[v][0]

rollout_list = []
for k, start_state in enumerate(start_state_set):
    traj = self.retrieve_unnormalized_robot_trajectory(start_state, z_to_replicate, rollout_length=14)
    rollout_list.append(traj)

norm_gt = self.trajectory_set[z_index]*self.norm_denom_value + self.norm_sub_value
norm_gt_tuple = (self.task_name_set[z_index], norm_gt)

np.save("Traj{0}_GT.npy".format(str(z_index).zfill(3)), norm_gt_tuple)
for k, v in enumerate(traj_sample_indices):
    np.save("Traj_Z{0}_RolloutStartIndex{1}.npy".format(str(z_index).zfill(3), str(v).zfill(3)), (self.task_name_set[v], rollout_list[k]))

################################
# For Z_index = 57, FinishedPouring
################################

n_samples = 4
state_dim = 21
start_state_set = np.zeros((n_samples, 1, state_dim))
traj_sample_indices = np.random.randint(0, high=95, size=n_samples)
traj_sample_indices = np.array([66, 77, 23, 41])
z_index = 57
z_to_replicate = self.latent_z_set[z_index]

for k, v in enumerate(traj_sample_indices):
    start_state_set[k] = self.trajectory_set[v][0]

rollout_list = []
for k, start_state in enumerate(start_state_set):
    traj = self.retrieve_unnormalized_robot_trajectory(start_state, z_to_replicate, rollout_length=14)
    rollout_list.append(traj)

norm_gt = self.trajectory_set[z_index]*self.norm_denom_value + self.norm_sub_value
norm_gt_tuple = (self.task_name_set[z_index], norm_gt)


strz = str(z_index).zfill(3)
logdir = "RealRobotLogs/Traj{0}_{1}".format(strz, label_dict[strz])
os.path.mkdir(logdir)


gt_traj_path = "Traj{0}_GT.npy".format(str(z_index).zfill(3))
np.save(os.path.join(logdir, gt_traj_path), norm_gt_tuple)
for k, v in enumerate(traj_sample_indices):
    rollout_path = os.path.join( ,"Traj_Z{0}_RolloutStartIndex{1}.npy".format(str(z_index).zfill(3), str(v).zfill(3)))
    np.save(os.path.join(logdir, rollout_path), (self.task_name_set[v], rollout_list[k]))

