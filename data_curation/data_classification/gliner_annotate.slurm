#!/bin/bash

#SBATCH --job-name=gliner_annotate
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --gres=gpu:1                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3        # QoS
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=ilr@h100           # account to use, with xyz the 3 letter code of your project

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
# module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.8.0
# env
# conda activate bert24
conda activate data-proc

export HF_HUB_OFFLINE=1

# model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/models/fastino/gliner2-multi-v1
# model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/classifiers/gliner2-multi-v1-ft-ep20-bs64-encoderlr1e5-tasklr5e4-wd001-qwen3-235b-v3-description/final
model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/classifiers/gliner2-multi-v1-ft-ep20-bs64-encoderlr5e5-tasklr1e3-wd001-qwen3-235b-v4/final

# input_path=$1
# input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k
# input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored
# input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1
input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_qwen3_235b_a22b_instruct_2507_fp8_postprocessed_processed

# input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_test.jsonl
# input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_test.jsonl
# input_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_reviewed_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_test.jsonl

# output_path=${input_path}_extracted_gliner_multi_v1/shard.parquet
# output_path=${input_path}_extracted_gliner_multi_v1_pretrained/shard.parquet
output_path=${input_path}_extracted_gliner/shard.parquet

# infer
# only entity name
    # --infer_batch_size 6 \
# entity name + description
    # --infer_batch_size 5 \
    # --use_description \
# entity name + description + truncation_max_tokens
    # --infer_batch_size 5 \
    # --use_description \
    # --truncation_max_tokens 512 \

    # --text_column text \
srun python gliner_annotate.py \
    --input_path "$input_path" \
    --output_path "$output_path" \
    --model_name_or_path "$model_name_or_path" \
    --dtype bfloat16 \
    --text_column rewriting_text \
    --chunking_max_tokens 512 \
    --truncation_max_tokens 512 \
    --threshold 0.5 \
    --hf_map_batch_size 128 \
    --infer_batch_size 8 \
    --use_description \
    --shuffle True \
    --max_samples 10000 \

    # --output_shard_size 500000 \
    # --max_samples 1000 \

# infer + eval
# srun python gliner_annotate.py \
#     --input_path "$input_path" \
#     --output_path "$output_path" \
#     --model_name_or_path "$model_name_or_path" \
#     --dtype bfloat16 \
#     --text_column input \
#     --chunking_max_tokens 512 \
#     --truncation_max_tokens 512 \
#     --threshold 0.5 \
#     --hf_map_batch_size 128 \
#     --infer_batch_size 8 \
#     --use_description \

# srun python eval_ner.py \
#     --input_path "$output_path" \

echo "END TIME: $(date)"
