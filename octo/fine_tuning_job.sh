#!/bin/bash
#
#SBATCH --job-name=fine_tune_octo
#SBATCH --partition=a6000
#SBATCH --qos=2gpu3d
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=fine_tune_octo.txt

# full fine-tuning rlbench
python examples/02_finetune_new_observation_action_rl_bench.py \
  --pretrained_path=hf://rail-berkeley/octo-base-1.5 \
  --data_dir=~/tensorflow_datasets \
  --save_dir=~/octo-rlbench-fine-tuning/octo/checkpoint_rlbench \
  --batch_size=60
