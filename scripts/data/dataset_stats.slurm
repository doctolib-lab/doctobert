#!/bin/bash

#SBATCH --job-name=dataset_stats
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=kit@cpu            # cpu accounting

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

stat_func () {
    local input_path=$1

    # --max_samples 10000 \
    srun python dataset_stats.py \
        --input_path "${input_path}" \
        --num_proc 16 \

}

input_path="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health"
stat_func $input_path

input_path="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science"
stat_func $input_path

input_path="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics"
stat_func $input_path

echo "END TIME: $(date)"
