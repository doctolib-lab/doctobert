#!/bin/bash

#SBATCH --job-name=filter_domain
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3             # QoS
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

# dataset_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified/004_00026.parquet"
# dataset_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified"
# dataset_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified"
dataset_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified"

output_dir="${dataset_path}_filtered"

    # --target_labels "Health,Computers_and_Electronics,Science" \
    # --target_labels "Science" \

srun python filter_by_domain.py \
    --dataset_path "${dataset_path}" \
    --output_dir "${output_dir}" \
    --target_labels "Health" \
    --num_workers 8

echo "END TIME: $(date)"
