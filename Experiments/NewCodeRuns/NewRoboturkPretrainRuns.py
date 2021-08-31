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

################################################################################
################################################################################
# Running new RTP runs with more epochs..

python cluster_run.py --name='RTP_021' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_021 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_022' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_022 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_023' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_023 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

# Now with smoothen
python cluster_run.py --name='RTP_031' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_031 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_032' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_032 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_033' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_033 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

# Now with smoothen and action scaling
python cluster_run.py --name='RTP_041' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_041 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --action_scale_factor=10'

python cluster_run.py --name='RTP_042' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_042 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --action_scale_factor=10'

python cluster_run.py --name='RTP_043' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_043 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --smoothen=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --action_scale_factor=10'

##########################################
# Now run with higher eps.
python cluster_run.py --name='RTP_051' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_051 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_052' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_052 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_053' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_053 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

# Higher eps, higher KL..
python cluster_run.py --name='RTP_061' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_061 --data=FullRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_062' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_062 --data=FullRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_063' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_063 --data=FullRoboturk --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

# Higher eps, even higher KL..
python cluster_run.py --name='RTP_071' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_071 --data=FullRoboturk --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_072' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_072 --data=FullRoboturk --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

python cluster_run.py --name='RTP_073' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_073 --data=FullRoboturk --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/'

# Eval 22
python Master.py --train=0 --setting=pretrain_sub --name=RTP_022 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_022/saved_models/Model_epoch75

#########################
# Eval 50 series
python Master.py --train=0 --setting=pretrain_sub --name=RTP_051_eval_p10 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=RTP_052_eval_p10 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_052/saved_models/Model_epoch340 --perplexity=10

# +/- 5model
python Master.py --train=0 --setting=pretrain_sub --name=RTP_051_eval_p10_m335 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_051/saved_models/Model_epoch335 --perplexity=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTP_052_eval_p10_m335 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_052/saved_models/Model_epoch335 --perplexity=10

# 
# Eval 50 series
python Master.py --train=0 --setting=pretrain_sub --name=RTP_051_eval_p10 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --perplexity=10 --debug=1


######################################################
######################################################

# Now finetune on smoothened things...
# Now run with higher eps.
# Debug
python Master.py --train=1 --setting=pretrain_sub --name=RTP_debug_load --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --smoothen=1

python cluster_run.py --name='RTP_101' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_101 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --smoothen=1'

python cluster_run.py --name='RTP_102' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_102 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_052/saved_models/Model_epoch340 --smoothen=1'

python cluster_run.py --name='RTP_103' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=RTP_103 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_053/saved_models/Model_epoch340 --smoothen=1'

# Uncluster
python Master.py --train=1 --setting=pretrain_sub --name=RTP_201 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_051/saved_models/Model_epoch340 --smoothen=1

python Master.py --train=1 --setting=pretrain_sub --name=RTP_202 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=1 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_052/saved_models/Model_epoch340 --smoothen=1

python Master.py --train=1 --setting=pretrain_sub --name=RTP_203 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=2 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_053/saved_models/Model_epoch340 --smoothen=1

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=RTP_201_eval_m5 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_201/saved_models/Model_epoch5 --smoothen=1

python Master.py --train=0 --setting=pretrain_sub --name=RTP_201_eval_m10 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_201/saved_models/Model_epoch10 --smoothen=1

python Master.py --train=0 --setting=pretrain_sub --name=RTP_201_eval_m15 --data=FullRoboturk --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --normalization=minmax --no_mujoco=0 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --model=ExpWandbLogs/RTP_201/saved_models/Model_epoch15 --smoothen=1
