#  Run again.. 
python Master.py --train=1 --setting=context --name=Con_Toy002 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=2000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy01/saved_models/Model_epoch100 --debugging_datapoints=5000

# Comparing these settings with learntsub setting. 
python Master.py --train=1 --setting=learntsub --name=Joint_Toy001 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy01/saved_models/Model_epoch100 --debugging_datapoints=5000

################
# Training learntsub setting with updated toy models.
python Master.py --train=1 --setting=learntsub --name=Joint_Toy001 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --debugging_datapoints=5000


#################
# Now run a contextual run with these settings...
# Hinton
python Master.py --train=1 --setting=context --name=Con_Toy001 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --debugging_datapoints=5000

# Run with larger training phase size and see whether it can reconstruct at all...
# If not, it's an architectural thing.. 
python Master.py --train=1 --setting=context --name=Con_Toy002 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --debugging_datapoints=5000

###################
# New masking strategy that masks at least one, unless there literally is one unique z (which we can't mask then, otherwise reconstruction is uninformed)...

python Master.py --train=1 --setting=context --name=Con_Toy003 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --debugging_datapoints=5000

# Now new contextual embedding...
python Master.py --train=1 --setting=context --name=Con_Toy004 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --debugging_datapoints=5000 --new_context=1

# Running with reduced capacity variational network.. 
###################
# New masking strategy that masks at least one, unless there literally is one unique z (which we can't mask then, otherwise reconstruction is uninformed)...
python Master.py --train=1 --setting=context --name=Con_Toy005 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --debugging_datapoints=5000

# Now new contextual embedding... 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy006 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --debugging_datapoints=5000 --new_context=1

# Now new contextual embedding but no masking
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy007 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --debugging_datapoints=5000 --new_context=1 --mask_fraction=0.

###############
# Just training for longer....
# Now new contextual embedding... 
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=context --name=Con_Toy008 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --model=Experiment_Logs/Con_Toy006/saved_models/Model_epoch495 --initial_counter_value=40000 

# Now new contextual embedding but no masking
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy009 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0. --model=Experiment_Logs/Con_Toy007/saved_models/Model_epoch495 --initial_counter_value=40000 

###############
# Take the Con_Toy007 model, now train with masking, and lower learning rate (achieved by just starting in training phase 2.)
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=context --name=Con_Toy010 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --debugging_datapoints=5000 --new_context=1 --mask_fraction=0.1 --model=Experiment_Logs/Con_Toy007/saved_models/Model_epoch495 --initial_counter_value=40000

# Now different mask fraction...
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy011 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --debugging_datapoints=5000 --new_context=1 --mask_fraction=0.2 --model=Experiment_Logs/Con_Toy007/saved_models/Model_epoch495 --initial_counter_value=40000

###################################
###################################
# Now new contextual embedding... 
python Master.py --train=1 --setting=context --name=Con_Toy012 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=50000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --ELMO_embeddings=1

# Now new contextual embedding but no masking
python Master.py --train=1 --setting=context --name=Con_Toy013 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=50000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0. --ELMO_embeddings=1


###################################
# Generating embeddings / debugging embeddings
###################################

# For Joint Model - Joint_Toy001
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=Joint_Toy001_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Joint_Toy001/saved_models/Model_epoch495

# For Con_Toy008
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=context --name=Con_Toy008_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --model=Experiment_Logs/Con_Toy008/saved_models/Model_epoch420 --initial_counter_value=40000 

# For Con_Toy009
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=context --name=Con_Toy009_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0. --model=Experiment_Logs/Con_Toy009/saved_models/Model_epoch450 --initial_counter_value=40000 

#
# Visualize at different perplexities.
#
# For Joint Model - Joint_Toy001
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=Joint_Toy001_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Joint_Toy001/saved_models/Model_epoch495 --perplexity=10 --logdir=InspectionModels/

# For Con_Toy008
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=context --name=Con_Toy008_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --model=Experiment_Logs/Con_Toy008/saved_models/Model_epoch420 --initial_counter_value=40000 --perplexity=10 --logdir=InspectionModels/

# For Con_Toy009
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=context --name=Con_Toy009_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0. --model=Experiment_Logs/Con_Toy009/saved_models/Model_epoch450 --initial_counter_value=40000 --perplexity=10 --logdir=InspectionModels/


##########################
# New contextually generated data. 
##########################

python Master.py --train=1 --setting=pretrain_sub --name=SKT_Toy10 --entropy=0 --data=DirContNonZero --kl_weight=0.01 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --batch_size=32

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=SKT_Toy11 --entropy=0 --data=DirContNonZero --kl_weight=0.001 --z_dimensions=64 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=30 --batch_size=32

# SKT_Toy11 is great, SKT_Toy10 is trash. 
# Now... train equivalents of Joint_Toy001, Con_Toy008, Con_Toy009.
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=Joint_Toy002 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy11/saved_models/Model_epoch430

# Noq equivalent of contextual 008 / 009.
python Master.py --train=1 --setting=context --name=Con_Toy015 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy11/saved_models/Model_epoch430 --var_number_layers=3 --var_hidden_size=32 --new_context=1

# Now new contextual embedding but no masking
python Master.py --train=1 --setting=context --name=Con_Toy016 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy11/saved_models/Model_epoch430 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.

# Now run with increased masking fraction.
python Master.py --train=1 --setting=context --name=Con_Toy017 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy11/saved_models/Model_epoch430 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.2

# Now even further increased masking fraction.
python Master.py --train=1 --setting=context --name=Con_Toy018 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=Experiment_Logs/SKT_Toy11/saved_models/Model_epoch430 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.25


###### 
# Rerun Con_Toy015 - 018 on bach for speedy training.
# Noq equivalent of contextual 008 / 009.
CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=context --name=Con_Toy020 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Con_Toy015/saved_models/Model_epoch90 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --initial_counter_value=140000

# Now new contextual embedding but no masking
CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=context --name=Con_Toy021 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Con_Toy015/saved_models/Model_epoch90 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0. --initial_counter_value=140000

# Now run with increased masking fraction.
CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=context --name=Con_Toy022 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Con_Toy015/saved_models/Model_epoch90 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.2 --initial_counter_value=140000

# Now even further increased masking fraction.
CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=context --name=Con_Toy023 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=DirContNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Con_Toy015/saved_models/Model_epoch90 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.25 --initial_counter_value=140000


###################
# Evaluate with new... skill based evaluating.
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=Joint_Toy001_new_contextual_eval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Joint_Toy001/saved_models/Model_epoch495

################################
################################
# ADding more types of evaluation. 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=learntsub --name=Joint_Toy001_new_contextual_eval2 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=4000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Joint_Toy001/saved_models/Model_epoch495

# Now run for Con_Toy008 and Con_Toy009 models.
# For Con_Toy008
CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=context --name=Con_Toy008_neweval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --model=Experiment_Logs/Con_Toy008/saved_models/Model_epoch420

# For Con_Toy009
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=context --name=Con_Toy009_neweval --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0. --model=Experiment_Logs/Con_Toy009/saved_models/Model_epoch450

###############################################################################################
###############################################################################################
# NEw Contextual model was implemented inheriting the old context model for some reason. This messes up the forward function. Rerun ConToy8/9.
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy020 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1

CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=context --name=Con_Toy021 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy022 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.2

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy0debug_grads --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --mask_fraction=0.2 --epochs=1

#################################################
# Evaluate Con_Toy020.
CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=context --name=Con_Toy020 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --model=Experiment_Logs/Con_Toy020/saved_models/Model_epoch50 --var_number_layers=3 --var_hidden_size=32 --new_context=1


##############################################################################################
# Doing different seeded runs of Con_Toy006/8.
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy030 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --seed=0

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy031 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --seed=1

CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=context --name=Con_Toy032 --kl_weight=0.0 --subpolicy_ratio=1.0 --latentpolicy_ratio=0.01 --b_probability_factor=0.01 --data=ContinuousNonZero --latent_loss_weight=0.0 --z_dimensions=64 --var_skill_length=1 --training_phase_size=40000 --batch_size=32 --short_trajectories=1 --display_freq=50 --viz_latent_rollout=0 --traj_length=-1 --subpolicy_model=GoodModels/Experiment_Logs/SKT_Toy05/saved_models/Model_epoch15 --var_number_layers=3 --var_hidden_size=32 --new_context=1 --seed=2

