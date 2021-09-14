# 
CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="RL_Bax_021" --hierarchical=0 --env="BaxterLeftHandLift"

CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="RL_Bax_022" --hierarchical=0 --env="BaxterRightHandLift"

# 
CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="HRL_Bax_021" --hierarchical=1 --env="BaxterLeftHandLift" --data=MIME --lowlevel_policy_model="/home/tshankar/Research/Code/CausalSkillLearning/Experiments/ExpWandbLogs/MJ_116/saved_models/Model_epoch500" --action_scaling=1.0 --epochs=500

CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="HRL_Bax_022" --hierarchical=1 --env="BaxterRightHandLift" --data=MIME --lowlevel_policy_model="/home/tshankar/Research/Code/CausalSkillLearning/Experiments/ExpWandbLogs/MJ_116/saved_models/Model_epoch500" --action_scaling=1.0 --epochs=500

# 
CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="HRL_271" --hierarchical=1 --env="Door" --data=Roboturk --lowlevel_policy_model="/home/tshankar/Research/Code/CausalSkillLearning/Experiments/ExpWandbLogs/RTP_051/saved_models/Model_epoch340" --action_scaling=2. --epochs=500



# Just running flat baselines...
CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="RLB_Saw_101" --hierarchical=0 --env="SawyerPush" --data=Roboturk

CUDA_VISIBLE_DEVICES=1 python Run_Robosuite_PPO.py --run_name="RLB_Saw_102" --hierarchical=0 --env="SawyerReach" --data=Roboturk

# 
CUDA_VISIBLE_DEVICES=1 python Run_Robosuite_PPO.py --run_name="RLB_LBax_103" --hierarchical=0 --env="BaxterLeftHandPush"

CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="RLB_LBax_104" --hierarchical=0 --env="BaxterLeftHandReach"

# 
CUDA_VISIBLE_DEVICES=0 python Run_Robosuite_PPO.py --run_name="RLB_RBax_105" --hierarchical=0 --env="BaxterRightHandPush"

CUDA_VISIBLE_DEVICES=1 python Run_Robosuite_PPO.py --run_name="RLB_RBax_106" --hierarchical=0 --env="BaxterRightHandReach"