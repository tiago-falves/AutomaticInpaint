#!/bin/bash

#SBATCH --job-name=automatic_inpaint_10000CellAllTypes_HandPick_first_20_masks_fullRes_640_pad50
#SBATCH --output=output_pad50.log
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mail-user=tiago.alves@aicos.fraunhofer.pt
#SBATCH --mail-type=ALL

# mkdir -p /hpc/scratch/$user
# load modules
module load cuda11.2/toolkit/11.2.2

conda activate automatic

python img2img.py

