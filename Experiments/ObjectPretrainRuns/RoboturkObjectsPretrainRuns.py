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

# 