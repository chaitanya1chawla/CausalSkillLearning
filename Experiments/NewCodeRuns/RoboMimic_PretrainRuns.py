python Master.py --train=1 --setting=pretrain_sub --name=RM_debug_dataset --data=OrigRoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# 
python Master.py --train=1 --setting=pretrain_sub --name=RM_debug_dataset --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1


python Master.py --train=1 --setting=pretrain_sub --name=RM_debug_train --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Actually run
python Master.py --train=1 --setting=pretrain_sub --name=RMP_001 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMP_002 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=RMP_003 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMP_004 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar

# Run meanvar and minmax for longer
# These runs are good! 
python Master.py --train=1 --setting=pretrain_sub --name=RMP_005 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000

python Master.py --train=1 --setting=pretrain_sub --name=RMP_006 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMP_007 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=2000

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMP_008 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=2000

# Trial
python Master.py --train=1 --setting=pretrain_sub --name=RMP_trial --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000

# Visualize RMP_007
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMP_007_Eval2 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=2000 --model=ExpWandbLogs/RMP_007/saved_models/Model_epoch2000

# Viz RMP_005
python Master.py --train=0 --setting=pretrain_sub --name=RMP_005_Eval2 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000 --model=ExpWandbLogs/RMP_005/saved_models/Model_epoch2000

#### 
# Running for longer.
python Master.py --train=1 --setting=pretrain_sub --name=RMP_010 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=5000 --save_freq=100

python Master.py --train=1 --setting=pretrain_sub --name=RMP_011 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=5000 --save_freq=100

# Rerun 005 / 007 for longer as well
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMP_012 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=2000 --model=ExpWandbLogs/RMP_007/saved_models/Model_epoch2000 --save_freq=100

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMP_013 --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=2000 --model=ExpWandbLogs/RMP_005/saved_models/Model_epoch2000 --save_freq=100

# Evaluate RMP_010 and 011
python Master.py --train=0 --setting=pretrain_sub --name=RMP_010_Eval --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=minmax --epochs=5000 --save_freq=100 --model=ExpWandbLogs/RMP_010/saved_models/Model_epoch5000

python Master.py --train=0 --setting=pretrain_sub --name=RMP_011_Eval --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=5000 --save_freq=100 --model=ExpWandbLogs/RMP_011/saved_models/Model_epoch5000

# play with dataset
python Master.py --train=1 --setting=pretrain_sub --name=RMP_play --data=RoboMimic --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --normalization=meanvar --epochs=5000 --save_freq=100
