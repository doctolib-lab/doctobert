#!/bin/bash

#SBATCH --job-name=llm_annotate_medical_entity
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# data
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p1
dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p3

# prompts

# ner extraction
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_classification/prompts/annotate_medical_entities_system_prompt_v3.txt
# user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_classification/prompts/annotate_medical_entities_user_prompt.txt

# ner review
system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_classification/prompts/annotate_medical_entities_review_system_prompt.txt
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_classification/prompts/annotate_medical_entities_review_user_prompt.txt

# model

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/medgemma-27b-text-it
# )

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Instruct-2507
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
#     --enable_thinking False
# )

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-Next-80B-A3B-Instruct
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
#     # --max_model_len 16384
#     # --max_num_seqs 32
#     # --speculative_config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
#     --gpu_memory_utilization 0.85
# )

model_args=(
    --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
    --temperature 0.7
    --top_p 0.8
    --top_k 20
    --min_p 0.0
    --enable_thinking False
    --max_model_len 16384
    --gpu_memory_utilization 0.85
)

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
#     --temperature 0.6
#     --top_p 0.95
#     --top_k 20
#     --min_p 0.0
#     --gpu_memory_utilization 0.85
#     --reasoning_parser deepseek_r1

# )

# output_dir=${dataset_path}_extracted_qwen3_235b_a22b_instruct_2507_fp8_p1
# output_dir=${dataset_path}_extracted_qwen3_235b_a22b_instruct_2507_fp8_p3
output_dir=${dataset_path}_reviewed_qwen3_235b_a22b_instruct_2507_fp8

# edu-quality, subtopic
# python llm_annotate.py \
# srun python llm_annotate.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_dir $output_dir \
#     --system_prompt_file $system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --url_column_name NOT_EXISTED \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_tokens 2048 \
#     --json_schema medical_entities_extraction \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100 \

# medical entities
# srun python llm_annotate.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_dir $output_dir \
#     --system_prompt_file $system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --url_column_name NOT_EXISTED \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_tokens 4096 \
#     --json_schema medical_entities_extraction_v3 \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_words_per_sample 200 \
#     --max_samples 100000 \
#     --start_idx 100000 \

# review medical entities
srun python llm_annotate.py \
    "${model_args[@]}" \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --system_prompt_file $system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name text_truncated \
    --url_column_name NOT_EXISTED \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_tokens 8192 \
    --json_schema medical_entities_review \
    --batch_size 10000 \
    --shuffle True \

    # --max_samples 1000 \


echo "END TIME: $(date)"
