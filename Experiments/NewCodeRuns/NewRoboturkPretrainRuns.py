# Template MIME run
python Master.py --train=1 --setting=pretrain_sub --name=MBP_094 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Actual roboturk ..
python Master.py --train=1 --setting=pretrain_sub --name=RTP_001 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=RTP_002 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=RTP_003 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=RTP_001 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=RTP_002 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --model=ExpWandbLogs/RTP_002/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=RTP_003 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --model=ExpWandbLogs/RTP_003/saved_models/Model_epoch500

# Roboturk + smoothen
# Actual roboturk ..
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTP_004 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTP_005 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTP_006 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

# Eval 
CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RTP_004 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1 --model=ExpWandbLogs/RTP_004/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RTP_005 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1 --model=ExpWandbLogs/RTP_005/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTP_006 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1 --model=ExpWandbLogs/RTP_006/saved_models/Model_epoch500

# Try with and without smoothen to see whether something changed..
python Master.py --train=1 --setting=pretrain_sub --name=RTP_007 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=RTP_008 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

# Smoothen without normalize
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTP_007 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTP_008 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTP_009 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=RTP_007 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1 --model=ExpWandbLogs/RTP_007/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=RTP_008 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1 --model=ExpWandbLogs/RTP_008/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=RTP_009 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --smoothen=1 --model=ExpWandbLogs/RTP_009/saved_models/Model_epoch500

#####################################################################################
#####################################################################################
# Retrain pretrain roboturk
python Master.py --train=1 --setting=pretrain_sub --name=RTP_010 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/

python Master.py --train=1 --setting=pretrain_sub --name=RTP_011 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/

python Master.py --train=1 --setting=pretrain_sub --name=RTP_012 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/

# debug
python Master.py --train=1 --setting=pretrain_sub --name=RTP_debug --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/

# # 
# python Master.py --train=1 --setting=pretrain_sub --name=RTP_010 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 

# python Master.py --train=1 --setting=pretrain_sub --name=RTP_011 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 

# python Master.py --train=1 --setting=pretrain_sub --name=RTP_012 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 

# Try with smoothen and normalize

# Retrain pretrain roboturk
python Master.py --train=1 --setting=pretrain_sub --name=RTP_010 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1

python Master.py --train=1 --setting=pretrain_sub --name=RTP_011 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1

python Master.py --train=1 --setting=pretrain_sub --name=RTP_012 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1

# 
python Master.py --train=1 --setting=pretrain_sub --name=RTP_010 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --model=ExpWandbLogs/RTP_010/saved_models/Model_epoch


# Try... 
python Master.py --train=1 --setting=pretrain_sub --name=RTP_debug --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/




################
python Master.py --train=1 --setting=pretrain_sub --name=RTP_001_viz --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=1 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --model=ExpWandbLogs/RTP_001/saved_models/Model_epoch500