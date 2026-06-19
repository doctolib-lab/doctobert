#!/bin/bash

#SBATCH --job-name=prep_spm_input
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --time 02:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev            # QoS
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

export PYTHONPATH=$PYTHONPATH:/lustre/fswork/projects/rech/ilr/commun/doctobert

output_dir_root=/lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer

for dataset in fineweb-2 finepdfs finewiki nachos transcorpus_bio_fr; do
    # output_dir="${output_dir_root}/${dataset}_med01_edu4"
    output_dir="${output_dir_root}/${dataset}_med01_edu4_biocli5"

    # srun python prep_spm_input.py \
    python prep_spm_input.py \
        --dataset "$dataset" \
        --output_dir "$output_dir" \
        --num_workers 16
        # --max_samples 1000000 \
done

echo "END TIME: $(date)"
