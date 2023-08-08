# Generate dataset
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FK_debug --data=FrankaKitchenPreproc --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# Try training
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FK_debug --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# 
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_000 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# Debug viz
// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_000_debugviz --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --model=/data/tanmayshankar/TrainingLogs/FKROP_000/saved_models/Model_epoch4000

// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_000_Eval --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --model=/data/tanmayshankar/TrainingLogs/FKROP_000/saved_models/Model_epoch4000

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_001 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21

# Eval
// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_001_Eval --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21 --model=/data/tanmayshankar/TrainingLogs/FKROP_001/saved_models/Model_epoch4000


# Run for longer
# 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_002 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --model=/data/tanmayshankar/TrainingLogs/FKROP_000/saved_models/Model_epoch4000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_003 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21  --model=/data/tanmayshankar/TrainingLogs/FKROP_001/saved_models/Model_epoch4000

# Run without KL
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_004 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.

//  