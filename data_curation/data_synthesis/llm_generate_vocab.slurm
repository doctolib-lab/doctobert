#!/bin/bash

#SBATCH --job-name=llm_generate
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

# hf
export HF_HUB_OFFLINE=1

# task name
export GEN_NAME=vocabulary

# prompts
system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/vocabulary/system_prompt_rewrite.txt
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/vocabulary/user_prompt_rewrite.txt

# data
# 56233 entries
dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/vocabulary/academie_medecine_dictionary/academie_medecine_dictionary.jsonl
text_column_name=text

# model
model_args=(
    --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
    --temperature 0.7
    --top_p 0.8
    --top_k 20
    --min_p 0.0
    --enable_thinking False
)

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/medgemma-27b-text-it
# )

model_suffix=$(basename ${model_args[1]} | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# output path
output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/vocabulary/academie_medecine_dictionary/synthesized_${model_suffix}

# python llm_generate.py \
srun python llm_generate.py \
    "${model_args[@]}" \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --system_prompt_file $system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name $text_column_name \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_tokens 8192 \
    --batch_size 5000 \
    --max_num_seqs 64 \
    --shuffle True \
    --start_idx 20000 \

    # --max_samples 20000 \
    # --max_samples 100 \


echo "END TIME: $(date)"
