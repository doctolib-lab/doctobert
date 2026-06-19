#!/bin/bash

#SBATCH --job-name=train_gliner2
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
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# wandb
export WANDB_MODE=offline
export WANDB_PROJECT=2025-12-doctobert-gliner2

model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/models/fastino/gliner2-multi-v1

# train_data_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_next_80b_a3b_instruct_postprocessed/data.jsonl
# train_data_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_train.jsonl
# validation_data_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_test.jsonl
# train_data_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_reviewed_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_train.jsonl
train_data_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_reviewed_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_train_300k.jsonl
validation_data_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_reviewed_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/data_test.jsonl
# pretrained_checkpoint_path=""

root_output_dir=/lustre/fswork/projects/rech/ilr/commun/classifiers

run_name=gliner2-multi-v1-ft-ep20-bs64-encoderlr5e5-tasklr1e3-wd001-qwen3-235b-v4
output_dir=$root_output_dir/$run_name


    # --encoder_lr 1e-5 \
    # --task_lr 5e-4 \

# python train_gliner2.py \
srun torchrun --nproc_per_node=4 train_gliner2.py \
    --model_name_or_path "$model_name_or_path" \
    --train_data_path "$train_data_path" \
    --validation_data_path "$validation_data_path" \
    --do_augmentation True \
    --use_description_train False \
    --use_description_eval True \
    --output_dir "$output_dir" \
    --experiment_name "$run_name" \
    --num_epochs 20 \
    --batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --encoder_lr 5e-5 \
    --task_lr 1e-3 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --scheduler_type cosine \
    --bf16 True \
    --num_workers 4 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --save_total_limit 5 \
    --save_best True \
    --metric_for_best f1 \
    --greater_is_better True \
    --logging_steps 10 \
    --report_to_wandb True \
    --wandb_project "$WANDB_PROJECT" \
    --max_eval_samples 8192 \

echo "END TIME: $(date)"
