#!/bin/bash

#SBATCH --job-name=llm_annotate
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

export HF_HUB_OFFLINE=1

dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health

# system_prompt_file=/lustre/fswork/projects/rech/kit/commun/doctobert/scripts/data_classification/prompts/annotate_topic_system_prompt.txt
# system_prompt_file=/lustre/fswork/projects/rech/kit/commun/doctobert/scripts/data_classification/prompts/annotate_topic_system_prompt_refined.txt
system_prompt_file=/lustre/fswork/projects/rech/kit/commun/doctobert/scripts/data_classification/prompts/annotate_topic_system_prompt_refined_v2.txt

user_prompt_file=/lustre/fswork/projects/rech/kit/commun/doctobert/scripts/data_classification/prompts/annotate_topic_user_prompt.txt

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/kit/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Instruct-2507
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
#     --enable_thinking False
# )

model_args=(
    --model_name_or_path /lustre/fswork/projects/rech/kit/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
    --temperature 0.7
    --top_p 0.8
    --top_k 20
    --min_p 0.0
    --enable_thinking False
)

# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_annotated_qwen3_30b_a3b_instruct_2507
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_refined_annotated_qwen3_30b_a3b_instruct_2507
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_30b_a3b_instruct_2507
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8
output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_p2

# python llm_annotate.py \
srun python llm_annotate.py \
    "${model_args[@]}" \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --system_prompt_file $system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name text \
    --url_column_name url \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_tokens 1024 \
    --json_schema domain_classification \
    --batch_size 10000 \
    --shuffle True \
    --start_idx 1100000 \
    --max_samples 400000 \


echo "END TIME: $(date)"
