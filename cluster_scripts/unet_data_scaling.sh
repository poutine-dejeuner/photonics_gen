#!/bin/bash

train_size=(4 24 44 64 84 104 124 144 164 184 204 224 244 264 284)

for s in "${train_size[@]}"; do
  for i in {1..3}; do
    sbatch cluster_scripts/setoff.sh "photo_gen/main.py train_set_size=$s"
  done
done
