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