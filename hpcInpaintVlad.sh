#!/bin/bash

#SBATCH --job-name=VLAD_resized
#SBATCH --output=output_vlad_resized.log
#SBATCH --mem=32G
#SBATCH --time=1-23:59:00
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mail-user=tiago.alves@aicos.fraunhofer.pt
#SBATCH --mail-type=ALL

# mkdir -p /hpc/scratch/$user
# load modules
module load cuda11.2/toolkit/11.2.2

conda activate automatic

python img2img.py