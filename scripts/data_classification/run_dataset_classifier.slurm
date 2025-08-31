#!/bin/bash

#SBATCH --job-name=run_dataset_classifier
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --gres=gpu:1                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3        # QoS
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
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

# task_type=classification
# task_name=domain_classification
# model_name_or_path=/lustre/fswork/projects/rech/kit/commun/pretrained/models/nvidia/multilingual-domain-classifier
# dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train/000_00000.parquet
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified

# task_type=classification
# task_name=health_domain_classification
# model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_topic_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5_500k
# dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health/00000.parquet
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified

# task_type=regression
# task_name=edu_quality
# model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_health_edu_quality_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_edu_quality_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5
# dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified/00000.parquet
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored

task_type=regression
task_name=edu_quality
model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_science_edu_quality_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_science_edu_quality_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5
dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science
output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_scored
# dataset_path=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_scored

output_path="${output_dir}/${dataset_path##*/}"

# nemo classifier
# srun python run_dataset_classifier.py \
#     --dataset_path $dataset_path \
#     --output_path $output_path \
#     --model_name_or_path $model_name_or_path \
#     --task_type $task_type \
#     --task_name $task_name \
#     --text_column text \
#     --url_column not_existing \
#     --bucket_size 32768 \
#     --batch_size 1024 \
#     --num_workers 1 \
#     --output_shard_size 1000000 \

# domain classifier
## srun python run_dataset_classifier.py \
# python run_dataset_classifier.py \
#     --dataset_path $dataset_path \
#     --output_path $output_path \
#     --model_name_or_path $model_name_or_path \
#     --task_type $task_type \
#     --task_name $task_name \
#     --text_column text \
#     --url_column url \
#     --batch_size 128 \
#     --num_workers 8 \
#     --get_num_words True \

# domain classifier
# srun python run_dataset_classifier.py \
python run_dataset_classifier.py \
    --dataset_path $dataset_path \
    --output_path $output_path \
    --model_name_or_path $model_name_or_path \
    --task_type $task_type \
    --task_name $task_name \
    --text_column text \
    --url_column not_existing \
    --batch_size 128 \
    --num_workers 8 \
    --max_samples 10000 \


echo "END TIME: $(date)"
