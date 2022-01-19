
python Master.py --name=DJFE_LRMIME_unsup_010 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=50 --datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --source_single_hand=right --target_single_hand=left

python Master.py --name=DJFE_LRMIME_unsup_011 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=50 --datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --source_single_hand=left --target_single_hand=right

CUDA_VISIBLE_DEVICES=1 python Master.py --name=DJFE_LRMIME_unsup_012 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=200 --datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --source_single_hand=right --target_single_hand=left

CUDA_VISIBLE_DEVICES=1 python Master.py --name=DJFE_LRMIME_unsup_013 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=200 --datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --source_single_hand=left --target_single_hand=right

########################################
# Continue training unsup 010, 011
python Master.py --name=DJFE_LRMIME_unsup_010_cont --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=50 --datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --source_single_hand=right --target_single_hand=left --model=ExpWandbLogs/DJFE_LRMIME_unsup_010/saved_models/

python Master.py --name=DJFE_LRMIME_unsup_011_cont --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=500 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=50 --datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --source_single_hand=left --target_single_hand=right

