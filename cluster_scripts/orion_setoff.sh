#!/bin/bash
#SBATCH --gres=gpu:40gb
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 09:00:00                                 
#SBATCH --output slurm/%j.out
#SBATCH --error slurm/%j.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --comment="diffusion"


#module load python/3.8
# conda activate cphoto

orion hunt -n diffusion python train3.py \
-lr_fom~'uniform(1e16, 1e18)' \
