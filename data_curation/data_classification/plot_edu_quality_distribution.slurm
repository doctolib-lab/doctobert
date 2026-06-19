#!/bin/bash

#SBATCH --job-name=plot
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=ilr@cpu            # cpu accounting

set -x -e

echo "START TIME: $(date)"

# set up environment
module purges
# module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
conda activate data-proc
# conda activate bert24

save_dir="/lustre/fswork/projects/rech/ilr/commun/doctobert/assets/data_classification"

python plot_edu_quality_distribution.py \
    --save_dir "${save_dir}" \
    --num_workers 8 \

    # --max_samples 10000 \

echo "END TIME: $(date)"
