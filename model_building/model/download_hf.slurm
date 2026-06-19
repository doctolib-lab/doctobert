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

root_dir="/lustre/fswork/projects/rech/ilr/commun/pretrained/models"

model_names=(
    # openai/gpt-oss-120b
    # openai/gpt-oss-20b
    # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    # mistralai/Ministral-3-14B-Reasoning-2512
    # mistralai/Ministral-3-14B-Instruct-2512
    # mistralai/Mistral-Small-3.2-24B-Instruct-2506
    # Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
    # Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
    # Qwen/Qwen3-30B-A3B-Thinking-2507
    # Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
    # Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
    # Qwen/Qwen3.5-122B-A10B-FP8
    # Qwen/Qwen3.5-35B-A3B-FP8
    # Qwen/Qwen3.5-27B-FP8
    # Qwen/Qwen3.5-9B
    Qwen/Qwen3.6-35B-A3B-FP8
    # PrimeIntellect/INTELLECT-3-FP8
    # MiniMaxAI/MiniMax-M2.1
    # baichuan-inc/Baichuan-M3-235B-GPTQ-INT4
    # zai-org/GLM-4.7-Flash
    # marksverdhei/GLM-4.7-Flash-FP8
    # GadflyII/GLM-4.7-Flash-NVFP4
    # google/gemma-4-31B-it
    # google/gemma-4-26B-A4B-it
)

for model_name in "${model_names[@]}"; do
    download_args=(
        ${model_name}
        --local-dir ${root_dir}/${model_name}
        #     --exclude metal/*
        #     --exclude original/*
    )

    # huggingface-cli download \
    hf download \
        ${download_args[@]} \
        --max-workers $num_proc \

    done

echo "END TIME: $(date)"
