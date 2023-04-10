# VizTrial
// python Master.py --train=0 --setting=pretrain_sub --name=DAPG_Pretraining_005 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --model=ExpWandbLogs/DAPG_Pretraining002/saved_models/Model_epoch0 --viz_sim_rollout=0

# 
// python Master.py --train=0 --setting=pretrain_sub --name=DAPG_Pretraining_007 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --model=ExpWandbLogs/DAPG_Pretraining002/saved_models/Model_epoch0 --viz_sim_rollout=0

# Playing with training shuffling
// python Master.py --train=1 --setting=pretrain_sub --name=DAPG_Pretraining_102 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --viz_sim_rollout=0 

# Rerun
// python Master.py --train=1 --setting=pretrain_sub --name=DAPG_Pretraining_200 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --viz_sim_rollout=0 --datadir=/home/tshankar/Research/Data/Datasets/DAPG/

####################################
# 
// python Master.py --train=1 --setting=pretrain_sub --name=DAPG_Pretraining_200 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --viz_sim_rollout=0 --datadir=/home/tshankar/Research/Data/Datasets/DAPG/ --epochs=120000 --save_freq=5000 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=30000

// python Master.py --train=1 --setting=pretrain_sub --name=DAPG_Pretraining_201 --data=DAPG --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --viz_sim_rollout=0 --datadir=/home/tshankar/Research/Data/Datasets/DAPG/ --epochs=120000 --save_freq=5000 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=30000