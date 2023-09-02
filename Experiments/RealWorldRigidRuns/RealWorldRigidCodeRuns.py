# Generate dataset
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWR_debug --data=RealWorldRigidPreproc --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWR_debugsingle --data=RealWorldRigidPreproc --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16


# Now test actual dataset.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWR_debug_dataset --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# Debug training
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWR_debug_training --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

###################################

# Run with normalization, with and without factored encoding.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_003 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_004 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

###################################
# Run without normalization, with and without factored encoding.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_005 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_006 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

###################################
# Run with normalization, with and without factored encoding, with no KL
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_007 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --kl_weight=0. --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_008 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --kl_weight=0. --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

##################################
##################################
# Run without normalization, with and without factored encoding.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_011 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_012 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

###################################
# Run with normalization, with and without factored encoding, with no KL
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_013 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --kl_weight=0. --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_014 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --kl_weight=0. --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

###################################
# Run with normalization, with and without factored encoding.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_015 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_016 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

##########################################################################################
##########################################################################################

# # Run these again with higher epsilon over values.. 
# ##################################
# # No Norm, No FactoredEnc, Yes KL
# // CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_021 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# # No Norm, FactoredEnc, Yes KL
# // CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_022 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

# # No Norm, No FactoredEnc, No KL
# // CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_023 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.

# # No Norm, FactoredEnc, No KL
# // CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_024 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --kl_weight=0.

# # Run with normalization, with and without factored encoding.
# // CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_025 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax

# // CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_026 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

# # Run with normalization, with and without factored encoding, with no KL
# // CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_027 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --kl_weight=0.

# // CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_028 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=50000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --kl_weight=0.

#############################################
# Seemed like the best, so run for longer.. 
# Run with normalization, with and without factored encoding.
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_020 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=100000 --save_freq=2000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RWRP_021 --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=100000 --save_freq=2000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14

# 
# Eval
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RWRP_021_debugeval --data=RealWorldRigid --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=5000 --display_freq=2000 --epochs=100000 --save_freq=2000 --datadir=/data/tanmayshankar/Datasets/RealWorldRigid/RealWorldRigidNumpyOnly/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --normalization=minmax --split_stream_encoder=1 --robot_state_size=7 --env_state_size=14 --model=/data/tanmayshankar/TrainingLogs/RWR_021/saved_models/Model_epoch64000
