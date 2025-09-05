#!/bin/bash

#SBATCH --job-name=llm_generate
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4        # QoS
#SBATCH --time=48:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=kit@h100           # account to use, with xyz the 3 letter code of your project

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
export GEN_NAME=clinical_case
# prompts
system_prompt_file=/lustre/fswork/projects/rech/kit/commun/doctobert/scripts/data_synthesis/prompts/clinical_case/system_prompt_rewrite_clinical_case.txt
user_prompt_file=/lustre/fswork/projects/rech/kit/commun/doctobert/scripts/data_synthesis/prompts/clinical_case/user_prompt_rewrite_clinical_case.txt

# data
dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/en/PMC-Patients/pmc_patients_v2.parquet
text_column_name=patient

# model
# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/kit/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
#     --enable_thinking False
# )

model_args=(
    --model_name_or_path /lustre/fswork/projects/rech/kit/commun/pretrained/models/google/medgemma-27b-text-it
)

# output
output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fr/generated/clinical_case/pmc_patients_v2/synthesized_medgemma_27b_text_it
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fr/generated/clinical_case/pmc_patients_v2/synthesized_qwen3_235b_a22b_instruct_2507_fp8

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
    --batch_size 10000 \
    --shuffle True \
    --max_samples 1000 \


echo "END TIME: $(date)"
