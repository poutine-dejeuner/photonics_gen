#!/bin/bash

# models=("wgan" "standard_gan" "vae" "simple_unet")
models=("unet_fast")
first_channel=(4 )
# first_channel=(4 8 16 32)

for model in "${models[@]}"; do
  for d in "${first_channel[@]}"; do
  echo $model
  python photo_gen/main.py model=$model train=$model evaluation/functions=debug logger.enabled=False debug=True model.first_channels=$d
  done
done
