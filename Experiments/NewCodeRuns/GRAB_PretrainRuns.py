from audioop import minmax


python Master.py --train=1 --setting=pretrain_sub --name=GB_debug_dataset --data=GRABPreproc --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/

python Master.py --train=1 --setting=pretrain_sub --name=GB_debug_dataset --data=GRAB --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/

# Runs
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GBP_001 --data=GRAB --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --epochs=2000

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GBP_002 --data=GRAB --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --normalization=minmax --epochs=2000

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=GBP_003 --data=GRAB --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200 --display_freq=2000 --action_scale_factor=1 --datadir=/data/tanmayshankar/Datasets/GRAB_Joints/ --normalization=meanvar --epochs=2000