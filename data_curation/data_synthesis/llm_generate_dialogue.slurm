#!/bin/bash

#SBATCH --job-name=llm_generate_dialogue
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
export GEN_NAME=dialogue
# prompts
system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/dialogue/system_prompt_generate_dialogue.txt
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/dialogue/user_prompt_generate_dialogue.txt

# data
# pmc patients v2
dataset_path=(
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/pmc_patients_v2/synthesized_medgemma_27b_text_it
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/pmc_patients_v2/synthesized_qwen3_235b_a22b_instruct_2507_fp8
)
dataset_path_str="$(IFS=+; printf '%s' "${dataset_path[*]}")"
text_column_name=output

# model
# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
#     --enable_thinking False
# )

model_args=(
    --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/medgemma-27b-text-it
)

model_suffix=$(basename ${model_args[1]} | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# output path
output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/dialogue/pmc_patients_v2/synthesized_${model_suffix}


# python llm_generate.py \
srun python llm_generate.py \
    "${model_args[@]}" \
    --dataset_path $dataset_path_str \
    --output_dir $output_dir \
    --system_prompt_file $system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name $text_column_name \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_tokens 8192 \
    --batch_size 10000 \
    --shuffle True \
    --max_samples 50000 \

    # --start_idx 50000 \


echo "END TIME: $(date)"
