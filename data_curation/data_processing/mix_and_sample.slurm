#!/bin/bash

#SBATCH --job-name=mix_and_sample
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
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


input_paths=(
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored
    /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored
    /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/multilingual_medical_corpus/health_domain_classified_edu_quality_scored
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/E3C/layer3/health_domain_classified_edu_quality_scored
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/final/v1
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/final/v2
)
input_paths_str="${input_paths[*]}"
output_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k/file.parquet"

srun python mix_and_sample.py \
    --input_paths "${input_paths_str}" \
    --output_path "${output_path}" \
    --max_samples 500000 \
    --output_shard_size 50000 \

echo "END TIME: $(date)"
