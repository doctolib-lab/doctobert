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
module load pytorch-gpu/py3/2.7.0

export HF_HUB_OFFLINE=1

model_args=(
    --model /lustre/fswork/projects/rech/kit/commun/pretrained/models/google/medgemma-27b-text-it
)

srun python llm_gen.py \
    "${model_args[@]}" \
    --json data/cim11_definitions.json \
    --prompts_dir prompts \
    --out_dir pages_out \
    --p_textbook 0.5 \
    --batch_size 32 \
    --temperature 0.7 \
    --top_p 0.95 \
    --include_style_in_filename \
    --tensor_parallel_size 4 \
    --max_examples 32

echo "END TIME: $(date)"