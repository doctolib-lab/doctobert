#!/bin/bash

#SBATCH --job-name=plot
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=kit@cpu            # cpu accounting

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

dataset_path="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored"

# srun python plot.py \
python plot_distribution.py \
    --dataset_path "${dataset_path}" \
    --num_workers 8 \

    # --max_samples 10000 \

echo "END TIME: $(date)"
