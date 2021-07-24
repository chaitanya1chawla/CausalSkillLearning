7# Pretrain
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_001 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_002 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_003 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2

# Eval
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_001 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --model=ExpWandbLogs/MBP_001/saved_models/Model_epoch185 --perplexity=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_002 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --model=ExpWandbLogs/MBP_002/saved_models/Model_epoch185 --perplexity=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_002 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --model=ExpWandbLogs/MBP_002/saved_models/Model_epoch235 --perplexity=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_003 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --model=ExpWandbLogs/MBP_003/saved_models/Model_epoch185 --perplexity=10

# Pretrain with slightly bigger z
python Master.py --train=1 --setting=pretrain_sub --name=MBP_004 --data=MIME --number_layers=4 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0

python Master.py --train=1 --setting=pretrain_sub --name=MBP_005 --data=MIME --number_layers=4 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_006 --data=MIME --number_layers=4 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2

# JOint
python Master.py --train=1 --setting=learntsub --name=MJ_001 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=GoodModels/Experiment_Logs/MB_Pretrain_047/saved_models/Model_epoch499 --latent_loss_weight=0.0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000 --number_layers=8 --hidden_size=128 --batch_size=16

# smaller joint
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_003 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_002/saved_models/Model_epoch235 --latent_loss_weight=0.0 --z_dimensions=8 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=3 --hidden_size=20 --batch_size=32 --var_number_layers=4 --var_hidden_size=32

# Eval 4-6.
python Master.py --train=0 --setting=pretrain_sub --name=MBP_004 --data=MIME --number_layers=4 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --model=ExpWandbLogs/MBP_004/saved_models/Model_epoch170 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_005 --data=MIME --number_layers=4 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_006 --data=MIME --number_layers=4 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --model=ExpWandbLogs/MBP_006/saved_models/Model_epoch170 --perplexity=10

# Run joint with MJ_005..
python Master.py --train=1 --setting=learntsub --name=MJ_004 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=16 --number_layers=4 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32

# debug joint
python Master.py --train=1 --setting=learntsub --name=MJ_debug --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=16 --number_layers=4 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32

#
python Master.py --train=1 --setting=pretrain_sub --name=MBP_007 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_008 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_009 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.5 --epsilon_to=0.1

python Master.py --train=1 --setting=pretrain_sub --name=MBP_010 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=1. --epsilon_to=0.1

# Eval for 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_007_eval --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_007/saved_models/Model_epoch60

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_008_eval --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_008/saved_models/Model_epoch60

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_007_eval_p10 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_007/saved_models/Model_epoch60 --perplexity=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_008_eval_p10 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_008/saved_models/Model_epoch60 --perplexity=10

# Dense
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_007_eval_m150 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_007/saved_models/Model_epoch150 --perplexity=10

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_008_eval_m150 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_008/saved_models/Model_epoch150 --perplexity=10

# Trying out MJ with a variety of noise values..
python Master.py --train=1 --setting=learntsub --name=MJ_005 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=16 --number_layers=4 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=learntsub --name=MJ_006 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=16 --number_layers=4 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01

python Master.py --train=1 --setting=learntsub --name=MJ_007 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=16 --number_layers=4 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1

python Master.py --train=1 --setting=learntsub --name=MJ_008 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_005/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=16 --number_layers=4 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.5 --epsilon_to=0.1

######
# Rerun with different seeds. 
python Master.py --train=1 --setting=pretrain_sub --name=MBP_010 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_011 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=1 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_012 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=2 --epsilon_from=0.1 --epsilon_to=0.01

# Minmax
python Master.py --train=1 --setting=pretrain_sub --name=MBP_013 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_014 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_015 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01

# Run joint with MBP_007 Model 150
python Master.py --train=1 --setting=learntsub --name=MJ_010 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_007/saved_models/Model_epoch150 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=learntsub --name=MJ_011 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_007/saved_models/Model_epoch150 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01

# Eval
python Master.py --train=0 --setting=learntsub --name=MJ_010_eval --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --model=ExpWandbLogs/MJ_010/saved_models/Model_epoch25 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.01 --epsilon_to=0.01

##
# Now running MIME Joint with MBP013-015.
python Master.py --train=1 --setting=learntsub --name=MJ_013 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_013/saved_models/Model_epoch235 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --seed=0

python Master.py --train=1 --setting=learntsub --name=MJ_014 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_014/saved_models/Model_epoch235 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --seed=1

python Master.py --train=1 --setting=learntsub --name=MJ_015 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_015/saved_models/Model_epoch235 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --seed=2

# Now with MBP_010-012.
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_016 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_010/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --seed=0

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_017 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --seed=1

python Master.py --train=1 --setting=learntsub --name=MJ_018 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_012/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --seed=2

# Eval for these  joint runs...
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_016 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --model=ExpWandbLogs/MJ_016/saved_models/Model_epoch105 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.01 --epsilon_to=0.01 --seed=0

python Master.py --train=0 --setting=learntsub --name=MJ_017 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --model=ExpWandbLogs/MJ_017/saved_models/Model_epoch105 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.01 --epsilon_to=0.01 --seed=1

python Master.py --train=0 --setting=learntsub --name=MJ_018 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --model=ExpWandbLogs/MJ_018/saved_models/Model_epoch90 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.01 --epsilon_to=0.01 --seed=2

#######################
# Experiments with more noise., to compare with MJ_017.
python Master.py --train=1 --setting=learntsub --name=MJ_020 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=50 --seed=1

python Master.py --train=1 --setting=learntsub --name=MJ_021 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_022 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=50 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_023 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50 --seed=1

# Eval noisy runs
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_022 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --model=ExpWandbLogs/MJ_022/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=50 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_023 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --model=ExpWandbLogs/MJ_023/saved_models/Model_epoch170 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50 --seed=1

###########################
# Eval MBP010-15.
###########################

python Master.py --train=0 --setting=pretrain_sub --name=MBP_010 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_010/saved_models/Model_epoch270 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_011 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270

python Master.py --train=0 --setting=pretrain_sub --name=MBP_012 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_012/saved_models/Model_epoch350 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_013 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_013/saved_models/Model_epoch235

python Master.py --train=0 --setting=pretrain_sub --name=MBP_014 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=1 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_014/saved_models/Model_epoch235

python Master.py --train=0 --setting=pretrain_sub --name=MBP_015 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=minmax --seed=2 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_015/saved_models/Model_epoch235

# ####
# Need more noise...
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_024 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=100 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_025 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_026 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.5 --epsilon_to=0.01 --epsilon_over=100 --seed=1

# Different capacity
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_027 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=5 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=100 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_028 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=5 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_029 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --subpolicy_model=ExpWandbLogs/MBP_011/saved_models/Model_epoch270 --latent_loss_weight=0.0 --z_dimensions=8 --number_layers=3 --hidden_size=20 --var_number_layers=5 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --seed=0 --epsilon_from=0.5 --epsilon_to=0.01 --epsilon_over=100 --seed=1


# Eval for Good MIME model..
python Master.py --train=0 --setting=pretrain_sub --name=MBP_047_prev --data=MIME --number_layers=8 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_010/saved_models/Model_epoch270 --perplexity=30

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MB_Pretrain_047 --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=16 --eval_freq=500 --normalization=minmax --model=GoodModels/Experiment_Logs/MB_Pretrain_047/saved_models/Model_epoch499

# Speed compare 8 layer hidden size 128 and z_dim 64 model pretrain speed. 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MB_Pretrain_047_replicate --data=MIME --number_layers=8 --hidden_size=128 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=16 --eval_freq=500 --normalization=minmax

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=Small_MIME --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=16 --eval_freq=500 --normalization=minmax

# Speed trials
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=ST1 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=16 --eval_freq=500 --normalization=minmax

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=ST2 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=16 --eval_freq=500 --normalization=minmax

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=ST3 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=16 --eval_freq=500 --normalization=minmax

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=ST4 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=16 --eval_freq=500 --normalization=minmax

# Speed trials with joint training
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJS2 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --latent_loss_weight=0.0 --z_dimensions=64 --number_layers=3 --hidden_size=20 --var_number_layers=4 --var_hidden_size=32 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=16 --seed=0 --epsilon_from=0.5 --epsilon_to=0.01 --epsilon_over=100 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJS3 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --latent_loss_weight=0.0 --z_dimensions=64 --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --batch_size=16 --seed=0 --epsilon_from=0.5 --epsilon_to=0.01 --epsilon_over=100 --seed=1

##########################################
# Z sizes with small nets
python Master.py --train=1 --setting=pretrain_sub --name=MBP_020 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_021 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_022 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

python Master.py --train=1 --setting=pretrain_sub --name=MBP_023 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

# Slightly bigger nets
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_024 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_025 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_026 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_027 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=MBP_025 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_025/saved_models/Model_epoch230 --perplexity=5

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=MBP_027 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_027/saved_models/Model_epoch230 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_023 --data=MIME --number_layers=3 --hidden_size=20 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --model=ExpWandbLogs/MBP_023/saved_models/Model_epoch230 --perplexity=5


####################################################
# Rerun joint with small trajectoreis
python Master.py --train=1 --setting=learntsub --name=MJ_100 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=GoodModels/Experiment_Logs/MB_Pretrain_047/saved_models/Model_epoch499 --latent_loss_weight=0.0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000 --number_layers=8 --hidden_size=128 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50

# 
python Master.py --train=1 --setting=learntsub --name=MJ_101 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=GoodModels/Experiment_Logs/MB_Pretrain_047/saved_models/Model_epoch499 --latent_loss_weight=0.0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000 --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50

# NEed a MIME model with small z... 
python Master.py --train=1 --setting=pretrain_sub --name=MBP_030 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_031 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --display_freq=2000

# python Master.py --train=1 --setting=pretrain_sub --name=MBP_032 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --display_freq=2000

# python Master.py --train=1 --setting=pretrain_sub --name=MBP_033 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_040 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_041 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000



# Rerun with slightly bigger models
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_034 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=MBP_035 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_038 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_039 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000


# python Master.py --train=1 --setting=pretrain_sub --name=MBP_036 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000

# python Master.py --train=1 --setting=pretrain_sub --name=MBP_037 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=64 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000

# Really what I should be doing is running with Z dim = 8 / 16, and finding parameters that work well for recon / separation.


python Master.py --train=1 --setting=pretrain_sub --name=MBP_dummy --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000


####################################################
# Eval new runs
####################################################

python Master.py --train=0 --setting=pretrain_sub --name=MBP_030 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_030/saved_models/Model_epoch260 --perplexity=5

python Master.py --train=0 --setting=pretrain_sub --name=MBP_031 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_031/saved_models/Model_epoch205 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_040 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_040/saved_models/Model_epoch165 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_041 --data=MIME --number_layers=4 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_041/saved_models/Model_epoch155 --perplexity=10

python Master.py --train=0 --setting=pretrain_sub --name=MBP_034 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_034/saved_models/Model_epoch200 --perplexity=5

python Master.py --train=0 --setting=pretrain_sub --name=MBP_035 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_035/saved_models/Model_epoch200 --perplexity=5

python Master.py --train=0 --setting=pretrain_sub --name=MBP_038 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_038/saved_models/Model_epoch200 --perplexity=5

python Master.py --train=0 --setting=pretrain_sub --name=MBP_039 --data=MIME --number_layers=4 --hidden_size=48 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --display_freq=2000 --model=ExpWandbLogs/MBP_039/saved_models/Model_epoch200 --perplexity=5

# Eval MJ101
python Master.py --train=0 --setting=learntsub --name=MJ_101_Eval --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_101/saved_models/Model_epoch185 --latent_loss_weight=0.0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000 --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50

# Eval MJ100 and MJ101 at end of training
# 100
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_100 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_100/saved_models/Model_epoch495 --latent_loss_weight=0.0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000 --number_layers=8 --hidden_size=128 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50

# 101
python Master.py --train=0 --setting=learntsub --name=MJ_101_Eval --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_101/saved_models/Model_epoch495 --latent_loss_weight=0.0 --z_dimensions=64 --traj_length=-1 --var_skill_length=1 --training_phase_size=200000 --number_layers=8 --hidden_size=128 --var_number_layers=8 --var_hidden_size=128 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=50

########################## NEw runs
# Z sizes with small nets
# python Master.py --train=1 --setting=pretrain_sub --name=MBP_060 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

# python Master.py --train=1 --setting=pretrain_sub --name=MBP_061 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

# python Master.py --train=1 --setting=pretrain_sub --name=MBP_062 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01

#
python Master.py --train=1 --setting=pretrain_sub --name=MBP_063 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_064 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --display_freq=2000

python Master.py --train=1 --setting=pretrain_sub --name=MBP_065 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=32 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --display_freq=2000

# Eval
python Master.py --train=0 --setting=pretrain_sub --name=MBP_063 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --display_freq=2000 --model=ExpWandbLogs/MBP_063/saved_models/Model_epoch20

python Master.py --train=0 --setting=pretrain_sub --name=MBP_063 --data=MIME --number_layers=3 --hidden_size=32 --kl_weight=0.001 --var_skill_length=1 --z_dimensions=8 --batch_size=32 --eval_freq=500 --normalization=meanvar --seed=0 --epsilon_from=0.1 --epsilon_to=0.01 --display_freq=2000 --model=ExpWandbLogs/MBP_063/saved_models/Model_epoch25

##########################
# New MIME joint runs with small z values..
# Run with MBP95 and MBP94 models..
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_110 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_095/saved_models/Model_epoch405 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_111 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_094/saved_models/Model_epoch470 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

# slightly diff noise values
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_112 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_095/saved_models/Model_epoch405 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_113 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_094/saved_models/Model_epoch470 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200

# More joint values with different seeds
python Master.py --train=1 --setting=learntsub --name=MJ_114 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_111/saved_models/Model_epoch485 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200

python Master.py --train=1 --setting=learntsub --name=MJ_115 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_113/saved_models/Model_epoch470 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200

# Rerun with less noise? Or more noise but decays earlier on? 
python Master.py --train=1 --setting=learntsub --name=MJ_116 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_111/saved_models/Model_epoch485 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100

python Master.py --train=1 --setting=learntsub --name=MJ_117 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_113/saved_models/Model_epoch470 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100

# Eval... 
CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=learntsub --name=MJ_111 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100
# Eval for 113
CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=learntsub --name=MJ_113 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_113/saved_models/Model_epoch240 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.1 --epsilon_to=0.01 --epsilon_over=200


# Evaluate 114 and 115
python Master.py --train=0 --setting=learntsub --name=MJ_114 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_114/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200

python Master.py --train=0 --setting=learntsub --name=MJ_115 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_114/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200


# Rerun MJ_111 with .. reset training. 
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=learntsub --name=MJ_120 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_111/saved_models/Model_epoch245 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=200 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --fix_subpolicy=0

# CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_121 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_111/saved_models/Model_epoch245 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=10000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --reset_training=1

# Eval MJ_116 and MJ120
python Master.py --train=0 --setting=learntsub --name=MJ_116 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_116/saved_models/Model_epoch490 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_120 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_120/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=200 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --fix_subpolicy=0


# Profile joint..
python -m cProfile Master.py --train=1 --setting=learntsub --name=MJ_profile --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_111/saved_models/Model_epoch245 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=200 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --fix_subpolicy=0 --epochs=2

# smart batch
python Master.py --train=1 --setting=learntsub --name=MJ_smart_batch --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_111/saved_models/Model_epoch245 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=200 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --fix_subpolicy=1 --epochs=2

# 
python Master.py --train=1 --setting=learntsub --name=MJ_122 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --model=ExpWandbLogs/MJ_111/saved_models/Model_epoch245 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=200 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --fix_subpolicy=1 --epochs=500


#####################################################
# DEBUGGING VARIABLE LENGTH LOSSES
python Master.py --train=1 --setting=learntsub --name=MJ_debug --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_111/saved_models/Model_epoch485 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100

python Master.py --train=1 --setting=learntsub --name=Joint_Toy_debug --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=8 --hidden_size=16 --number_layers=3 --var_hidden_size=32 --var_number_layers=4 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=ExpWandbLogs/SKT_Toy53/saved_models/Model_epoch20 --context=0 --seed=0

# Running with adaptation to variable length losses
python Master.py --train=1 --setting=learntsub --name=MJ_200 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_094/saved_models/Model_epoch470 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

python Master.py --train=1 --setting=learntsub --name=MJ_201 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_111/saved_models/Model_epoch485 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100

# 
python Master.py --train=0 --setting=learntsub --name=MJ_200 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/MJ_200/saved_models/Model_epoch35

python Master.py --train=0 --setting=learntsub --name=MJ_201 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/MJ_201/saved_models/Model_epoch35

# DEBUG VERDI
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_debugverdi --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME   --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100 --no_mujoco=1

# Debug new dataset
python Master.py --train=1 --setting=learntsub --name=MJ_debugdataset --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME   --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100 --task_discriminability=1 --number_of_tasks=20

######################################################
# Rerun MJ111 with NEW DATASET
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_300 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_203/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=MJ_301 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_204/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=learntsub --name=MJ_302 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_205/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100

# Eval
######################################################
# Rerun MJ111 with NEW DATASET
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_300 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_203/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/MJ_300/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=MJ_301 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_204/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/MJ_301/saved_models/Model_epoch500

CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=learntsub --name=MJ_302 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_205/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --model=ExpWandbLogs/MJ_302/saved_models/Model_epoch500

######################################################
# Now run joint MIME on End Effector 
# debug
python Master.py --train=1 --setting=learntsub --name=MJ_debug --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/

python cluster_run.py --name='MJ_00' --cmd='python Master.py --train=1 --setting=learntsub --name=MJ_400 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJ_01' --cmd='python Master.py --train=1 --setting=learntsub --name=MJ_401 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJ_02' --cmd='python Master.py --train=1 --setting=learntsub --name=MJ_402 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_002/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJ_03' --cmd='python Master.py --train=1 --setting=learntsub --name=MJ_403 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJ_04' --cmd='python Master.py --train=1 --setting=learntsub --name=MJ_404 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_001/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJ_05' --cmd='python Master.py --train=1 --setting=learntsub --name=MJ_405 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_002/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

# 
python cluster_run.py --name='MJE_010' --cmd='python Master.py --train=1 --setting=learntsub --name=MJE_010 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_010/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJE_011' --cmd='python Master.py --train=1 --setting=learntsub --name=MJE_011 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_011/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJE_012' --cmd='python Master.py --train=1 --setting=learntsub --name=MJE_012 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_012/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJE_013' --cmd='python Master.py --train=1 --setting=learntsub --name=MJE_013 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_013/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=0 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJE_014' --cmd='python Master.py --train=1 --setting=learntsub --name=MJE_014 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_014/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'

python cluster_run.py --name='MJE_015' --cmd='python Master.py --train=1 --setting=learntsub --name=MJE_015 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MPE_015/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=100 --ee_trajectories=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/'
