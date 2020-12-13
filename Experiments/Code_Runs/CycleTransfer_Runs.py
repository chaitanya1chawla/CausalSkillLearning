# Copyright (c) Facebook, Inc. and its affiliates.

# Debugging cycle consistency transfer.

python Master.py --name=CTdebug --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001

# Getting cycle-consistency running again.
python Master.py --name=CTdebug --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001

# First run
# Getting cycle-consistency running again.
python Master.py --name=CT000 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=0.5 --discriminability_weight=1.0 --kl_weight=0.001

python Master.py --name=CT001 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001

python Master.py --name=CT002 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001

# Rerun for profiling
python Master.py --name=CT_profile --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10 --display_freq=10 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001

# Writing batched implementation of CCT. 
python Master.py --name=CTbatch --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=4

# Running batched implementation of CCT across different batch sizes. 
CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_000 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=4

CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_001 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=16

CUDA_VISIBLE_DEVICES=2 python Master.py --name=CT_002 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=32

CUDA_VISIBLE_DEVICES=2 python Master.py --name=CT_003 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64

# Debug
CUDA_VISIBLE_DEVICES=3 python Master.py --name=CT_debug --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=100 --display_freq=1000 --eval_freq=4 --alternating_phase_size=20 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64

#######
# Debug with real-translated trajectory discriminator losses. 
CUDA_VISIBLE_DEVICES=0 python Master.py --name=CT_debugdisc --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=100 --display_freq=1000 --eval_freq=4 --alternating_phase_size=20 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.

# Run with real translated discriminator with different weights and full training phase sizes. 
CUDA_VISIBLE_DEVICES=0 python Master.py --name=CT_disc000 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.

CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_disc001 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=0.1

# CUDA_VISIBLE_DEVICES=0 python Master.py --name=CT_disc001 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.

# Run with initial few epochs only training reconstruction.
# Run with real translated discriminator with different weights and full training phase sizes. 
CUDA_VISIBLE_DEVICES=0 python Master.py --name=CT_disc002 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.

CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_disc003 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=0.1

### Train with initial few epochs for reconstruction, decreasing weights of other losses, and various batch sizes.
CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_disc004 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=16 --real_translated_discriminator=1 --real_trans_loss_weight=1.0

CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_disc005 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=32 --real_translated_discriminator=1 --real_trans_loss_weight=1.0

CUDA_VISIBLE_DEVICES=2 python Master.py --name=CT_disc006 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.0

# 
CUDA_VISIBLE_DEVICES=2 python Master.py --name=CT_disc007 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=0.5 --kl_weight=0.001 --batch_size=16 --real_translated_discriminator=1 --real_trans_loss_weight=1.0

CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_disc008 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=0.5 --kl_weight=0.001 --batch_size=32 --real_translated_discriminator=1 --real_trans_loss_weight=1.0

CUDA_VISIBLE_DEVICES=1 python Master.py --name=CT_disc009 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=0.5 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.0

###########
# Run with cycle reconstruction loss weight = 0 
python Master.py --name=DT000 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=16 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0.

python Master.py --name=DT001 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=32 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0.

python Master.py --name=DT002 --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0.

# Debug with viz
python Master.py --name=Debugv --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0.

# Now run VIZ for DT000-DT002
CUDA_VISIBLE_DEVICES=1 python Master.py --name=DT000_eval --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=100 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=16 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0. --epochs=2 --model=Experiment_Logs/DT000/saved_models/Model_epoch110 --initial_counter_value=10000

CUDA_VISIBLE_DEVICES=1 python Master.py --name=DT001_eval --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=100 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=32 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0. --epochs=2 --model=Experiment_Logs/DT001/saved_models/Model_epoch220 --initial_counter_value=10000

CUDA_VISIBLE_DEVICES=1 python Master.py --name=DT002_eval --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=100 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=1.0 --kl_weight=0.001 --batch_size=64 --real_translated_discriminator=1 --real_trans_loss_weight=1.0 --cycle_reconstruction_loss_weight=0. --epochs=2 --model=Experiment_Logs/DT002/saved_models/Model_epoch440 --initial_counter_value=10000
