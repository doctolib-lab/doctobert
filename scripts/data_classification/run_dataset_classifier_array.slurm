#!/bin/bash

#SBATCH --job-name=run_dataset_classifier_array
#SBATCH --output=logs/%x/%A_%a.out    # %A = array job ID, %a = array task ID
#SBATCH --error=logs/%x/%A_%a.err     # %A = array job ID, %a = array task ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --gres=gpu:1                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3        # QoS
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=kit@h100           # account to use
#SBATCH --array=0-999%50             # Process up to 1000 files, max 50 concurrent jobs

set -x -e

echo "START TIME: $(date)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

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
# input_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified

# task_type=classification
# task_name=health_domain_classification
# model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_health_topic_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_topic_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5_500k
# input_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified

# task_type=regression
# task_name=edu_quality
# model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_health_edu_quality_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_health_edu_quality_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5
# input_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored

task_type=regression
task_name=edu_quality
model_name_or_path=/lustre/fsn1/projects/rech/kit/commun/classifiers/moderncamembert_base_ft_science_edu_quality_annotated_qwen3_30b_a3b_ep3_bs256_lr2e5_ft_science_edu_quality_annotated_qwen3_235b_a22b_fp8_ep3_bs256_lr1e5
# input_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science
# output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_scored
input_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics
output_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_scored

# Get list of all parquet files and select the one for this array task
mapfile -t files < <(find $input_dir -name "*.parquet" | sort)
total_files=${#files[@]}
echo "Total files found: $total_files"

# Check if this array task has a corresponding file
if [ $SLURM_ARRAY_TASK_ID -ge $total_files ]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds number of files ($total_files). Exiting."
    exit 0
fi

# Get the file for this array task
dataset_path=${files[$SLURM_ARRAY_TASK_ID]}

output_path="${output_dir}/${dataset_path##*/}"

echo "Processing file: $dataset_path"
echo "Output will be saved to: $output_path"

# Check if output already exists (to avoid reprocessing)
if [ -f "$output_path" ]; then
    echo "Output file already exists. Skipping: $output_path"
    exit 0
fi

# tmp
# if [ $SLURM_ARRAY_TASK_ID -eq 0 -o $SLURM_ARRAY_TASK_ID -eq 1 ]; then
#     exit 0;
# fi

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
# srun python run_dataset_classifier.py \
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
srun python run_dataset_classifier.py \
    --dataset_path $dataset_path \
    --output_path $output_path \
    --model_name_or_path $model_name_or_path \
    --task_type $task_type \
    --task_name $task_name \
    --text_column text \
    --url_column not_existing \
    --batch_size 128 \
    --num_workers 8 \
    --get_num_words True \


echo "END TIME: $(date)"
