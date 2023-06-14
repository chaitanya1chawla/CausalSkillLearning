# 
# Sample
// python Master.py --train=1 --setting=pretrain_sub --name=RMP_105 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000


# Actually run the RMRO 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_000 --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

#############################
# Eval
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_000_Env --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_000/saved_models/Model_epoch1995 --embedding_visualization_stream='env' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_000_Robot --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_000/saved_models/Model_epoch1995 --embedding_visualization_stream='robot' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_000 --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_000/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0


# Eval 001
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Env --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --embedding_visualization_stream='env' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Robot --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --embedding_visualization_stream='robot' --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0
    
# with video
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Vid --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0
##########################################
# Run split stream encoder 
##########################################


# Actually run the RMRO 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMROP_000 --data=RoboMimicRobotObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMROP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1

# Eval
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMROP_000 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RMROP_000/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0


// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMROP_001 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RMROP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0

# 
scp /data/tanmayshankar/TrainingLogs/RMROP_001/saved_models/Model_epoch1995 tshankar@bach:~/../../data/tanmayshankar/TrainingLogs/RMROP_001/saved_models/

scp /data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 tshankar@bach:~/../../data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_Vid --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1--epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=1 --viz_gt_sim_rollout=0 --seed=0

#########################################
# Copying template roboturk run
/ CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_NewObjectPoseset_v121_p0501_n05500 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_104/saved_models/Model_epoch2000 --sim_viz_action_scale_factor=1 --sim_viz_step_repetition=20 --viz_sim_rollout=1 --viz_gt_sim_rollout=0 --seed=2

#############################
# Rerun RMOP
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_NewViz2 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0 --viz_gt_sim_rollout=0 --seed=0

# 
#############################
# Rerun RMOP
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_001_NewViz3 --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0 --viz_gt_sim_rollout=0 --seed=0

#############################
# Rerun RMOP
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_002_TestNNmodels --data=RoboMimicRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0 --viz_gt_sim_rollout=0 --seed=0

#############################
# Run with high KL... 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_100 --data=RoboMimicRobotObjects --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_101 --data=RoboMimicRobotObjects --kl_weight=1.0 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_102 --data=RoboMimicRobotObjects --kl_weight=10.0 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_103 --data=RoboMimicRobotObjects --kl_weight=100.0 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

#############################
# Debug relative state recon loss
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_debugrel --data=RoboMimicRobotObjects --kl_weight=1.0 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

#############################
# Debug relative state recon loss. 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_200 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=0.01

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_201 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=0.1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_202 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=1.0

#############################
# Eval
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_200 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=0.01 --model=/data/tanmayshankar/TrainingLogs/RMOP_200/saved_models/Model_epoch2000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_201 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=0.1 --model=/data/tanmayshankar/TrainingLogs/RMOP_201/saved_models/Model_epoch2000

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_202 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=1.0 --model=/data/tanmayshankar/TrainingLogs/RMOP_202/saved_models/Model_epoch2000

# RECREATE RESULTS FILE WITH SIM ROLLOUT = 0 
#############################
# Eval
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_200 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=0.01 --model=/data/tanmayshankar/TrainingLogs/RMOP_200/saved_models/Model_epoch2000 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_201 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=0.1 --model=/data/tanmayshankar/TrainingLogs/RMOP_201/saved_models/Model_epoch2000 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_202 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --relative_state_reconstruction_loss_weight=1.0 --model=/data/tanmayshankar/TrainingLogs/RMOP_202/saved_models/Model_epoch2000 --viz_sim_rollout=0

# Run relative state reconstruction loss, but with larger number of epochs. 
#############################
# Debug relative state recon loss. 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_203 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.01

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_204 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.1

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_205 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=1.0


# debug big env
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOP_debugbig --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.01
 

#############################
# Eval
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_203 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.01 --model=ExpWandbLogs/RMOP_203/saved_models/Model_epoch4400 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_204 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.1 --model=ExpWandbLogs/RMOP_204/saved_models/Model_epoch4400 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_205 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=1.0 --model=ExpWandbLogs/RMOP_205/saved_models/Model_epoch4400 --viz_sim_rollout=0

#############################
# Eval
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_203 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.01 --model=ExpWandbLogs/RMOP_203/saved_models/Model_epoch20000 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_204 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=0.1 --model=ExpWandbLogs/RMOP_204/saved_models/Model_epoch20000 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOP_205 --data=RoboMimicRobotObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=10000 --epochs=20000 --save_freq=100 --datadir=/home/tshankar/Research/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=ExpWandbLogs/ --relative_state_reconstruction_loss_weight=1.0 --model=ExpWandbLogs/RMOP_205/saved_models/Model_epoch20000 --viz_sim_rollout=0
