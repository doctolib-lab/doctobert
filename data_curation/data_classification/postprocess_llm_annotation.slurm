#!/bin/bash

#SBATCH --job-name=postprocess_llm_annotation
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4           # number of cores per tasks
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev             # QoS
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

# task="topic"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_30b_a3b_instruct_2507"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_p2"

# task="quality"
# # input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_edu_quality_health_annotated_qwen3_30b_a3b_instruct_2507"
# # input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_edu_quality_health_annotated_qwen3_235b_a22b_instruct_2507_fp8"
# # input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_science_annotated_qwen3_30b_a3b_instruct_2507"
# # input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_science_annotated_qwen3_235b_a22b_instruct_2507_fp8"
# # input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_science_annotated_qwen3_30b_a3b_instruct_2507"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_science_annotated_qwen3_235b_a22b_instruct_2507_fp8"

# edu-quality, subtopic
# output_path="${input_path}_postprocessed/shard.parquet"

# srun python postprocess_llm_annotation.py \
#     --input_path "${input_path}" \
#     --output_path "${output_path}" \
#     --task $task \
#     --num_workers 8 \
#     --output_shard_size 1000000 \


task="medical_entities_v2_1"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8"
input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_reviewed_qwen3_235b_a22b_instruct_2507_fp8"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_reviewed_qwen3_235b_a22b_instruct_2507_fp8_p3p4"

# medical entities
output_path="${input_path}_postprocessed/data_300k.jsonl"

# output_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_next_80b_a3b_instruct_postprocessed_medical/data.jsonl
# output_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_235b_a22b_instruct_2507_fp8_postprocessed_medical/data.jsonl

# srun python postprocess_llm_annotation.py \
python postprocess_llm_annotation.py \
    --input_path "${input_path}" \
    --output_path "${output_path}" \
    --task $task \
    --text_column_name text_truncated \
    --num_workers 8 \

    # --test_split_size 10000 \

echo "END TIME: $(date)"
