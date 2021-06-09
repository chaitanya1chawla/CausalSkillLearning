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


#########################################
# RUN PRETRAIN WITH UNNORMED TSNE...

# Repeat
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_300 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --no_mujoco=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_301 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --no_mujoco=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_302 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --no_mujoco=1
