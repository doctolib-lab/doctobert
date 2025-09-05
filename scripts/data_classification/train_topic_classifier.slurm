#!/bin/bash

#SBATCH --job-name=train_classifier
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4        # QoS
#SBATCH --time=100:00:00              # maximum execution time (HH:MM:SS)
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
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# wandb
export WANDB_MODE=offline
export WANDB_PROJECT=2025-domain-classifier

model_name_or_path=/lustre/fswork/projects/rech/kit/commun/pretrained/models/almanach/moderncamembert-base
# model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5

train_dir=(
    # /lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_30b_a3b_instruct_2507_postprocessed
    /lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/train
    /lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_p2_postprocessed
)
eval_dir=(
    /lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/test
)

train_dir="$(IFS=+; printf '%s' "${train_dir[*]}")"
eval_dir="$(IFS=+; printf '%s' "${eval_dir[*]}")"

root_output_dir=/lustre/fsn1/projects/rech/kit/commun/classifiers
# run_name=moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5
run_name=moderncamembert_base_ft_health_topic_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr2e5_500k
# run_name=moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_topic_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr2e5
# run_name=moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_topic_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5_500k
output_dir=$root_output_dir/$run_name

    # --gradient_checkpointing \
    # --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    # --max_train_samples 2048 \
    # --max_eval_samples 2048 \
    # --max_predict_samples 2048 \

# accelerate launch train_classifier.py \
srun accelerate launch train_classifier.py \
    --model_name_or_path $model_name_or_path \
    --train_file $train_dir \
    --validation_file $eval_dir \
    --test_file $eval_dir \
    --text_column_names text \
    --url_column_name url \
    --label_column_name topic \
    --max_seq_length 8192 \
    --preprocessing_num_workers 32 \
    --pad_to_max_length false \
    --shuffle_train_dataset \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --optim adamw_torch_fused \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --bf16 \
    --ddp_find_unused_parameters false \
    --dataloader_num_workers 8 \
    --metric_name accuracy \
    --metric_for_best_model accuracy \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --report_to all \
    --run_name $run_name \
    --do_train \
    --do_eval \
    --do_predict \

echo "END TIME: $(date)"
