// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FK_debug --data=FrankaKitchenPreproc --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16