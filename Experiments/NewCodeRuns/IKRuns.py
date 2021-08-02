# Run IK trainer..
python Master.py --train=1 --setting=iktrainer --name=IK_debug --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1

# 
python cluster_run.py --name='IK_000' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_001 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

python cluster_run.py --name='IK_001' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_001 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

python cluster_run.py --name='IK_002' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_001 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=2 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

# 
python cluster_run.py --name='IK_003' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_004 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

python cluster_run.py --name='IK_004' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_005 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

python cluster_run.py --name='IK_005' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_006 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=2 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

# Play eval
python Master.py --train=1 --setting=iktrainer --name=IK_004_Eval --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_004/saved_models/Model_epoch500 --debug=1

# Rerunning after fixing stupid mistake
python cluster_run.py --name='IK_010' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_010 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

python cluster_run.py --name='IK_011' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_011 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

python cluster_run.py --name='IK_012' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_012 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=2 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1'

# 
python Master.py --train=1 --setting=iktrainer --name=IK_010_Eval --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_010/saved_models/Model_epoch500 --debug=1

# 
python Master.py --train=1 --setting=iktrainer --name=IK_013 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1

python Master.py --train=1 --setting=iktrainer --name=IK_013_Eval --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_013/saved_models/Model_epoch135 --debug=1

# Test .. 
python Master.py --train=1 --setting=iktrainer --name=IK_010_Eval --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_010/saved_models/Model_epoch500 --debug=1

####################
# Training the IK networks further.. 
python cluster_run.py --name='IK_020' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_020 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_010/saved_models/Model_epoch500 --epochs=2000 --save_freq=50'

python cluster_run.py --name='IK_021' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_021 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_011/saved_models/Model_epoch500 --epochs=2000 --save_freq=50'

python cluster_run.py --name='IK_022' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_022 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=2 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --model=ExpWandbLogs/IK_012/saved_models/Model_epoch500 --epochs=2000 --save_freq=50'

######################### 
# Rerunning with masked losses. .# Rerunning after fixing stupid mistake
python cluster_run.py --name='IK_030' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_030 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --dataset_traj_length_limit=200'

python cluster_run.py --name='IK_031' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_031 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --dataset_traj_length_limit=200'

python cluster_run.py --name='IK_032' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_032 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=2 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --dataset_traj_length_limit=200'

# Rerunning with masked losses. .# Rerunning after fixing stupid mistake
python cluster_run.py --name='IK_040' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_040 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=0 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --dataset_traj_length_limit=200 --normalization=mixmax'

python cluster_run.py --name='IK_041' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_041 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=1 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --dataset_traj_length_limit=200 --normalization=mixmax'

python cluster_run.py --name='IK_042' --cmd='python Master.py --train=1 --setting=iktrainer --name=IK_042 --data=MIME --number_layers=4 --hidden_size=48 --batch_size=32 --eval_freq=500 --seed=2 --datadir=/private/home/tanmayshankar/Research/Code/Data/Datasets/MIME/ --short_trajectories=1 --dataset_traj_length_limit=200 --normalization=mixmax'
