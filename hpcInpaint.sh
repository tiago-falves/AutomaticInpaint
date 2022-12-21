#!/bin/bash

#SBATCH --job-name=automatic_inpaint_10000CellAllTypes_HandPick_all640_CFG4
#SBATCH --output=output_CFG4_All.log
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mail-user=tiago.alves@aicos.fraunhofer.pt
#SBATCH --mail-type=ALL

# mkdir -p /hpc/scratch/$user
# load modules
module load cuda11.2/toolkit/11.2.2

conda activate automatic

python img2img.py \
    --model_name '2022-12-20T11-43-15_ASCUS_1_training_images_10000_max_training_steps_ASCUS_token_cell_class_word.ckpt' \
    --input_folder "vlad_w_masks" \
    --output_dir "Ascus_Vlad" \
    --prompt "ASCUS cell"

