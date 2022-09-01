# Trial
// python Master.py --train=1 --setting=pretrain_sub --name=RTO_trial --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=1

# We need to figure out why it doesn't work.. 
# Factors to vary.. smoothening and the KL weight.
// python Master.py --train=1 --setting=pretrain_sub --name=RTO_001 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=1

// python Master.py --train=1 --setting=pretrain_sub --name=RTO_002 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=1

// python Master.py --train=1 --setting=pretrain_sub --name=RTO_003 --data=RoboturkObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_004 --data=RoboturkObjects --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_005 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_006 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_007 --data=RoboturkObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_008 --data=RoboturkObjects --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

## Testing out visualization setup... 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_005_viz --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_005/saved_models/Model_epoch500

//CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_005_viz2 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_005/saved_models/Model_epoch500

##
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_005_viz --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_005/saved_models/Model_epoch2000

##############################
##############################

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_005_viz_highres --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_005/saved_models/Model_epoch250 --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_006_viz_highres --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_006/saved_models/Model_epoch250 --task_based_shuffling=1

# 
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_007_viz_highres --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_007/saved_models/Model_epoch250 --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RTO_008_viz_highres --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_008/saved_models/Model_epoch250 --task_based_shuffling=1

#####################################################
#####################################################

# Run without relative pose.
# debugging no rel pose
python Master.py --train=1 --setting=pretrain_sub --name=RTO_debug --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_010 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_011 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

# // CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_012 --data=RoboturkObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

# // CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_013 --data=RoboturkObjects --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

# RErun on bach
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RTO_012 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_013 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RTO_014 --data=RoboturkObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTO_015 --data=RoboturkObjects --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0


// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RTO_012 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_012/saved_models/Model_epoch400 --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTO_013 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_013/saved_models/Model_epoch600 --task_based_shuffling=1

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RTO_014 --data=RoboturkObjects --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_014/saved_models/Model_epoch1500 --task_based_shuffling=1

# Eval later models..
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RTO_012 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_012/saved_models/Model_epoch1000 --task_based_shuffling=1 --perplexity=10

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=0 --setting=pretrain_sub --name=RTO_013 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_013/saved_models/Model_epoch1000 --task_based_shuffling=1 --perplexity=10

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RTO_012 --data=RoboturkObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_012/saved_models/Model_epoch1800 --task_based_shuffling=1 --perplexity=10

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=0 --setting=pretrain_sub --name=RTO_013 --data=RoboturkObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --model=ExpWandbLogs/RTO_013/saved_models/Model_epoch1800 --task_based_shuffling=1 --perplexity=10

#######################
# Run on Joint Roboturk Robot Object data.
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTROP_001 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/