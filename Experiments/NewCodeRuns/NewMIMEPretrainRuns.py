# Why is model reconstruction so bad? 
python Master.py --train=1 --setting=pretrain_sub --name=MBP_050 --data=MIME --number_layers=4 --hidden_size=48 --var_hidden_size=48 --var_number_layers=4 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=50 --display_freq=2000 --action_scale_factor=10

python Master.py --train=1 --setting=pretrain_sub --name=MBP_051 --data=MIME --number_layers=4 --hidden_size=48 --var_hidden_size=48 --var_number_layers=4 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=50 --display_freq=2000 --action_scale_factor=10

python Master.py --train=1 --setting=pretrain_sub --name=MBP_052 --data=MIME --number_layers=4 --hidden_size=48 --var_hidden_size=48 --var_number_layers=4 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=50 --display_freq=2000 --action_scale_factor=10

#
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MB_Pretrain_047 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=16 --eval_freq=500 --normalization=minmax --model=GoodModels/Experiment_Logs/MB_Pretrain_047/saved_models/Model_epoch499

#####
python Master.py --train=1 --setting=pretrain_sub --name=MBP_070 --data=MIME --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --display_freq=2000 --epochs=50

# 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_071 --data=MIME --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --display_freq=2000 --epochs=50

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_072 --data=MIME --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --display_freq=2000 --epochs=50

python Master.py --train=1 --setting=pretrain_sub --name=MBP_073 --data=MIME --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --display_freq=2000 --epochs=50
#
python Master.py --train=1 --setting=pretrain_sub --name=MBP_074 --data=MIME --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --epochs=50

# Rerun 63-65
python Master.py --train=1 --setting=pretrain_sub --name=MBP_075 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_076 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_077 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000

# Trying.. 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_080 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_081 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_082 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=10

# 81 with diff noise
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_083 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Eval
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_081 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_081/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_083 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_083/saved_models/Model_epoch500

# The minmax normalization error... good, but.. reconstruction isn't really that good.. 
# Try with varying size.
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_090 --data=MIME --number_layers=3 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_091 --data=MIME --number_layers=3 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_092 --data=MIME --number_layers=3 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_093 --data=MIME --number_layers=3 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_094 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_095 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_096 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.5 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_097 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=1. --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Now with KL schedule
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_100 --data=MIME --number_layers=3 --hidden_size=32 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0. --final_kl_weight=1. --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_101 --data=MIME --number_layers=3 --hidden_size=32 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0. --final_kl_weight=1. --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_102 --data=MIME --number_layers=3 --hidden_size=48 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0. --final_kl_weight=1. --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_103 --data=MIME --number_layers=3 --hidden_size=48 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0. --final_kl_weight=1. --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_104 --data=MIME --number_layers=4 --hidden_size=48 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0.001 --final_kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_105 --data=MIME --number_layers=4 --hidden_size=48 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0.001 --final_kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_106 --data=MIME --number_layers=4 --hidden_size=48 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0.001 --final_kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_107 --data=MIME --number_layers=4 --hidden_size=48 --kl_schedule=1 --kl_begin_increment_epochs=100 --kl_increment_epochs=100 --initial_kl_weight=0.001 --final_kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Eval all of these models
# 90
CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=MBP_090 --data=MIME --number_layers=3 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1  --model=ExpWandbLogs/MBP_090/saved_models/Model_epoch285 --perplexity=10
# 91
CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=MBP_091 --data=MIME --number_layers=3 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1  --model=ExpWandbLogs/MBP_091/saved_models/Model_epoch285 --perplexity=10
# 94
python Master.py --train=0 --setting=pretrain_sub --name=MBP_094 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_094/saved_models/Model_epoch470 --perplexity=10
# 95
python Master.py --train=0 --setting=pretrain_sub --name=MBP_095 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_095/saved_models/Model_epoch405 --perplexity=10

#######################
# Rerun 94 / 95 with different seeds.
python Master.py --train=1 --setting=pretrain_sub --name=MBP_110 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_111 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_112 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_113 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Eval
# 111
python Master.py --train=0 --setting=pretrain_sub --name=MBP_111 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_111/saved_models/Model_epoch485 --perplexity=10
# 113
python Master.py --train=0 --setting=pretrain_sub --name=MBP_113 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_113/saved_models/Model_epoch470 --perplexity=10

###### RERUN WITH NEW MIME - this is basically just slightly differently normalized gripper values

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_200 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_201 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_202 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Repeat
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_203 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_204 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_205 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

# Eval
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_203 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_203/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_204 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_204/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_205 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_205/saved_models/Model_epoch500

# Debug
python Master.py --train=1 --setting=pretrain_sub --name=MBPdebug --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1

### With higher KL
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_500 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_501 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_502 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_503 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_504 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_505 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=1. --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

# Slightly lower KL... 
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_506 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_507 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=MBP_508 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.01 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1


# Eval high KL runs..
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_500 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.1 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --no_mujoco=1 --model=ExpWandbLogs/MBP_500/saved_models/Model_epoch70 --no_mujoco=1

#########################################
# RUN PRETRAIN WITH UNNORMED TSNE...

# Repeat
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_300 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --no_mujoco=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_301 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --no_mujoco=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_302 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --no_mujoco=1

##########################################
# Run MIME pretrain with EE
python Master.py --train=1 --setting=pretrain_sub --name=MPE_000 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1

python Master.py --train=1 --setting=pretrain_sub --name=MPE_001 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1

python Master.py --train=1 --setting=pretrain_sub --name=MPE_002 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1

#######################################
# Now run MIME Pretrain with EE
python Master.py --train=1 --setting=pretrain_sub --name=MPE_000 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

python Master.py --train=1 --setting=pretrain_sub --name=MPE_001 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

python Master.py --train=1 --setting=pretrain_sub --name=MPE_002 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

# Run wihtout normalization
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MPE_006 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MPE_007 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MPE_008 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

####################################
# Eval MPE_000
python Master.py --train=0 --setting=pretrain_sub --name=MPE_000 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/MPE_000/saved_models/Model_epoch215


#########################################################################
#########################################################################
# Run on actual correct EE data
python cluster_run.py --name='MPE_010' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MPE_010 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MPE_011' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MPE_011 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MPE_012' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MPE_012 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MPE_013' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MPE_013 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MPE_014' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MPE_014 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MPE_015' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MPE_015 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

# 
# Visualize skills 
python Master.py --train=0 --setting=pretrain_sub --name=MPE_010_Eval --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/MPE_010/saved_models/Model_epoch500

# Debug 
python Master.py --train=0 --setting=pretrain_sub --name=MPE_010_debug --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/MPE_010/saved_models/Model_epoch500

# 
python Master.py --train=0 --setting=pretrain_sub --name=MPE_010_debug --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --model=ExpWandbLogs/MPE_010/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=MPE_010_debug --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --ee_trajectories=1 --model=ExpWandbLogs/MPE_010/saved_models/Model_epoch500

##########################################################################
##########################################################################
# Debug
python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_debug --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"

python cluster_run.py --name='MP_LR_001' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_001 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"'

python cluster_run.py --name='MP_LR_002' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_002 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"'

python cluster_run.py --name='MP_LR_003' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_003 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"'

python cluster_run.py --name='MP_LR_004' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_004 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right"'

python cluster_run.py --name='MP_LR_005' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_005 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right"'

python cluster_run.py --name='MP_LR_006' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_006 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right"'

# 
python cluster_run.py --name='MP_LR_001' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_001 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"'

python cluster_run.py --name='MP_LR_002' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_002 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"'

python cluster_run.py --name='MP_LR_003' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_003 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left"'

python cluster_run.py --name='MP_LR_004' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_004 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right"'

python cluster_run.py --name='MP_LR_005' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_005 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right"'

python cluster_run.py --name='MP_LR_006' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_006 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right"'

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=MP_LR_006_eval --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right" --model=ExpWandbLogs/MP_LR_006/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=MP_LR_004_eval --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right" --model=ExpWandbLogs/MP_LR_004/saved_models/Model_epoch500

python Master.py --train=0 --setting=pretrain_sub --name=MP_LR_003_eval --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left" --model=ExpWandbLogs/MP_LR_003/saved_models/Model_epoch500

# 
python Master.py --train=0 --setting=pretrain_sub --name=MP_LR_004_eval3 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="right" --model=ExpWandbLogs/MP_LR_004/saved_models/Model_epoch80

python Master.py --train=0 --setting=pretrain_sub --name=MP_LR_003_eval3 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand="left" --model=ExpWandbLogs/MP_LR_003/saved_models/Model_epoch80


# RErun

python cluster_run.py --name='MP_LR_011' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_011 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=left'

python cluster_run.py --name='MP_LR_012' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_012 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=left'

python cluster_run.py --name='MP_LR_013' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_013 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=left'

python cluster_run.py --name='MP_LR_014' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_014 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=right'

python cluster_run.py --name='MP_LR_015' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_015 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=right'

python cluster_run.py --name='MP_LR_016' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_016 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=right'

# Rerun for much longer, lower KL weight.. 
python cluster_run.py --name='MP_LR_021' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_021 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=left --epochs=4000 --save_freq=20'

python cluster_run.py --name='MP_LR_022' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_022 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=left --epochs=4000 --save_freq=20'

python cluster_run.py --name='MP_LR_023' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_023 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=left --epochs=4000 --save_freq=20'

python cluster_run.py --name='MP_LR_024' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_024 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=right --epochs=4000 --save_freq=20'

python cluster_run.py --name='MP_LR_025' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_025 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=right --epochs=4000 --save_freq=20'

python cluster_run.py --name='MP_LR_026' --cmd='python Master.py --train=1 --setting=pretrain_sub --name=MP_LR_026 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --single_hand=right --epochs=4000 --save_freq=20'

# Debug

python Master.py --train=1 --setting=pretrain_sub --name=MP_try --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --epochs=4000 --save_freq=20

# 
python Master.py --train=1 --setting=pretrain_sub --name=MP_try --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/home/tanmay/Research/Code/Data/Datasets/MIME/ --epochs=4000 --save_freq=20

##################################
# Getting labels for MBP_094_viz2
python Master.py --train=0 --setting=pretrain_sub --name=MBP_094_viz2 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_094/saved_models/Model_epoch500

# 
python Master.py --train=0 --setting=pretrain_sub --name=MBP_094_LeftLabels --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_094/saved_models/Model_epoch500 --single_hand=left

python Master.py --train=0 --setting=pretrain_sub --name=MBP_094_RightLabels --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --model=ExpWandbLogs/MBP_094/saved_models/Model_epoch500 --single_hand=right