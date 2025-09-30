#!/bin/bash

# models=("unet_fast")
first_channel=(4 8 16 32 64)

# for model in "${models[@]}"; do
  for d in "${first_channel[@]}"; do
  # echo $model
  #
  sbatch cluster_scripts/setoff.sh "photo_gen/main.py model=unet_fast train=unet_fast model.first_channels=$d"
  done
# done
