# GRAB Hand Pretrain run that seems to work fine.
// python Master.py --train=1 --setting=pretrain_sub --name=GHP_001_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 
# --input_corruption_noise=0.1

# Viz
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=GHP_001_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --model=ExpWandbLogs/GHP_001_Right/saved_models/Model_epoch4000

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=GHP_001_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --model=ExpWandbLogs/GHP_001_Right/saved_models/Model_epoch2000


#################
# Template GRAB dataset run... 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GBP_002 --data=GRAB --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --normalization=minmax --epochs=2000
#################

# Without noise, but now with new epsilon range., and default network sizes.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=GHP_002_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90

# With noise
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=GHP_003_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --input_corruption_noise=0.01

# // CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GHP_004_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --input_corruption_noise=0.05

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=GHP_005_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --input_corruption_noise=0.1

# // CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=GHP_006_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --input_corruption_noise=0.2

### Eval.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=GHP_002_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --model=ExpWandbLogs/GHP_002_Right/saved_models/Model_epoch2400

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=GHP_002_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --model=ExpWandbLogs/GHP_002_Right/saved_models/Model_epoch3700

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=GHP_003_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --model=ExpWandbLogs/GHP_003_Right/saved_models/Model_epoch2400

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=GHP_005_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --model=ExpWandbLogs/GHP_005_Right/saved_models/Model_epoch2400

###################
# Continuing training 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=GHP_002_Right_cont --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90

# With noise
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=GHP_003_Right_cont --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --input_corruption_noise=0.01

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=GHP_005_Right_cont --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --input_corruption_noise=0.1


##########
# Retry
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=GHP_006_Right --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --logdir=/data/tanmayshankar/TrainingLogs/

# Zero out wrist joint 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GHP_007_Right_ZeroWrist --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --logdir=/data/tanmayshankar/TrainingLogs/

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GHP_008_Right_ZeroWrist --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --logdir=/data/tanmayshankar/TrainingLogs/

# Eval
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GHP_007_Right_ZeroWrist --data=GRABHand --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --human_pos_normalization=wrist --normalization=minmax --epochs=4000 --single_hand=right --dataset_traj_length_limit=90 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/GHP_007_Right_ZeroWrist/saved_models/Model_epoch
