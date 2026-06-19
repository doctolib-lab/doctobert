#!/bin/bash

#SBATCH --job-name=eval_classifier
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=2:00:00
#SBATCH --hint=nomultithread
#SBATCH --constraint=h100
#SBATCH --account=ilr@h100

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
module load anaconda-py3/2024.06
module load cuda/12.8.0
conda activate data-proc

# hf
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

model=/lustre/fswork/projects/rech/ilr/commun/classifiers/moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_topic_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5_500k
eval_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_topic_refined_v2_annotated_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/test
eval_dir="$(IFS=+; printf '%s' "${eval_dir[*]}")"
srun python eval_classifier.py \
    --model_name_or_path $model \
    --dataset_path $eval_dir \
    --text_column text \
    --url_column url \
    --label_column topic \
    --task_name "Medical subdomain" \
    --task_type classification \
    --max_seq_length 8192 \
    --batch_size 128 \
    --num_workers 8 \
    --output_dir $model/eval \

    # --max_samples 100

model=/lustre/fswork/projects/rech/ilr/commun/classifiers/moderncamembert_base_ft_health_edu_quality_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_edu_quality_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5
eval_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_edu_quality_health_annotated_qwen3_235b_a22b_instruct_2507_fp8_postprocessed/test
eval_dir="$(IFS=+; printf '%s' "${eval_dir[*]}")"
srun python eval_classifier.py \
    --model_name_or_path $model \
    --dataset_path $eval_dir \
    --text_column text \
    --url_column not_existing \
    --label_column score \
    --task_name "Educational quality" \
    --task_type regression \
    --max_seq_length 8192 \
    --batch_size 128 \
    --num_workers 8 \
    --output_dir $model/eval

echo "END TIME: $(date)"
