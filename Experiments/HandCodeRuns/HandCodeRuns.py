# VizTrial
// python Master.py --train=0 --setting=pretrain_sub --name=DAPG_Pretraining_005 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --model=ExpWandbLogs/DAPG_Pretraining002/saved_models/Model_epoch0 --viz_sim_rollout=0

# 
// python Master.py --train=0 --setting=pretrain_sub --name=DAPG_Pretraining_007 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --model=ExpWandbLogs/DAPG_Pretraining002/saved_models/Model_epoch0 --viz_sim_rollout=0

# Playing with training shuffling
// python Master.py --train=1 --setting=pretrain_sub --name=DAPG_Pretraining_102 --data=DAPG --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=8 --hidden_size=128 --save_freq=500 --normalization=minmax --viz_sim_rollout=0 