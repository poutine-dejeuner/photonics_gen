#!/bin/bash
#SBATCH --array=1-16
#SBATCH --gres=gpu:40gb
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 60:00:00
#SBATCH --output slurm/%j.out
#SBATCH --error slurm/%j.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --comment="diffusion"

python train3.py --multirun 'model.n_samples="uniform(16, 284, discrete=True)"'
# orion hunt -n diffusion_d_scaling python train3.py 'n_samples="uniform(16, 284)"'
# -lr_fom~'uniform(1e16, 1e18)' \
