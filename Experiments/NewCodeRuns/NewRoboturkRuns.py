# Template MIME run
# CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_111 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_094/saved_models/Model_epoch470 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

# Roboturk Joint Runs
from shutil import SpecialFileError


CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=RT_001 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=learntsub --name=RT_002 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_002/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=learntsub --name=RT_003 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_003/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

# Original Joint Run
# python Master.py --train=1 --setting=learntsub --name=RJ135 --training_phase_size=100000 --initial_counter_value=200000 --latent_loss_weight=1. --lat_z_wt=0.0001 --subpolicy_ratio=0.1 --data=Roboturk --subpolicy_model=Experiment_Logs/R93/saved_models/Model_epoch20 --model=Experiment_Logs/RJ84/saved_models/Model_epoch50 --load_latent=0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --number_layers=8 --hidden_size=128 --kl_weight=0.001 --b_probability_factor=0.01

# Debug
python Master.py --train=1 --setting=learntsub --name=RT_debug --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100
# 

# Eval
python Master.py --train=0 --setting=learntsub --name=RT_001_eval --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RT_001/saved_models/Model_epoch500

python Master.py --train=0 --setting=learntsub --name=RT_002_eval --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_002/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RT_002/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=RT_003 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_003/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RT_003/saved_models/Model_epoch500


###############################################################
###############################################################
# # Reviving Roboturk
python Master.py --train=0 --setting=learntsub --name=RT_001_eval_notrain_butsmoo --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RT_001/saved_models/Model_epoch500 --dataset_traj_length_limit=100 --smoothen=1

# viz latent spae
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=RT_001_eval_notrain_butsmoo_viz --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RT_001/saved_models/Model_epoch500 --dataset_traj_length_limit=100 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk

python Master.py --train=0 --setting=learntsub --name=RT_002_eval --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_002/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RT_002/saved_models/Model_epoch500



###############################
# Eval RTP_001 to viz SpecialFileError
# python Master.py --train=1 --setting=learntsub --name=RT_001_viz --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500




#########################################
# Run new joint RT with 051-053 runs..
python cluster_run.py --name='RTP_051' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_051 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_052' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_052 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_053' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_053 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

# Rerunning with new RTP_051 pretrain series..
# debug
python Master.py --train=1 --setting=learntsub --name=RTdeb --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --epochs=5000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --metric_eval_freq=100000000000

# run
python cluster_run.py --name='RT_051' --cmd='python Master.py --train=1 --setting=learntsub --name=RT_051 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --epochs=5000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RT_052' --cmd='python Master.py --train=1 --setting=learntsub --name=RT_052 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_052/saved_models/Model_epoch340 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --epochs=5000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RT_053' --cmd='python Master.py --train=1 --setting=learntsub --name=RT_053 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=Roboturk  --subpolicy_model=ExpWandbLogs/RTP_053/saved_models/Model_epoch340 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --epochs=5000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'
