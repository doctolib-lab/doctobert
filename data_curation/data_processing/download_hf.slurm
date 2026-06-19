#!/bin/bash

#SBATCH --job-name=download
#SBATCH --output=logs/%x/%j.out                 # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err                  # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1                      # number of cores per tasks
#SBATCH --time=2:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev                        # QoS
#SBATCH --hint=nomultithread                    # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --account=ilr@cpu

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
# module load arch/h100
module load git-lfs
module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
conda activate data-proc

num_proc="64"

# huggingface-cli download \
#     HuggingFaceFW/fineweb-2 \
#     --repo-type dataset \
#     --local-dir /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2 \
#     --max-workers $num_proc \
#     --include "data/fra_Latn/train/*" \

# huggingface-cli download \
#     HuggingFaceFW/finepdfs \
#     --repo-type dataset \
#     --local-dir /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs \
#     --max-workers $num_proc \
#     --include "data/fra_Latn/train/*" \

huggingface-cli download \
    HuggingFaceFW/finewiki \
    --repo-type dataset \
    --local-dir /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki \
    --max-workers $num_proc \
    --include "data/frwiki/*" \

# huggingface-cli download \
#     jknafou/TransCorpus-bio-fr \
#     --repo-type dataset \
#     --local-dir /lustre/fsn1/projects/rech/ilr/commun/corpus/transcorpus_bio_fr \
#     --max-workers $num_proc \

echo "END TIME: $(date)"
