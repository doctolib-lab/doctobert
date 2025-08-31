#!/bin/bash

#SBATCH --job-name=postprocess_annotation
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

# task="topic"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_30b_a3b_instruct_2507"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_p2"

task="quality"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_edu_quality_health_annotated_qwen3_30b_a3b_instruct_2507"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_edu_quality_health_annotated_qwen3_235b_a22b_instruct_2507_fp8"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_science_annotated_qwen3_30b_a3b_instruct_2507"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_science_annotated_qwen3_235b_a22b_instruct_2507_fp8"
# input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_science_annotated_qwen3_30b_a3b_instruct_2507"
input_dir="/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_science_annotated_qwen3_235b_a22b_instruct_2507_fp8"

output_dir="${input_dir}_postprocessed"

srun python postprocess_annotation.py \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --task $task \
    --num_workers 8 \
    --output_shard_size 1000000 \
    --test_split_size 10000 \

echo "END TIME: $(date)"
