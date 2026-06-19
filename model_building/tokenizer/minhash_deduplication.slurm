#!/bin/bash

#SBATCH --job-name=minhash_deduplication
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --time 04:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=ilr@cpu            # cpu accounting

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
# module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
conda activate data-proc
# conda activate bert24

input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/nachos

# srun python postprocess.py \
python minhash_deduplication.py \
    --input_path "$input_path" \
    --total_tasks 16 \


echo "END TIME: $(date)"
