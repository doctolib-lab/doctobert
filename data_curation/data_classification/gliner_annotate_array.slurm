#!/bin/bash

#SBATCH --job-name=gliner_annotate_array
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
#SBATCH --account=ilr@h100           # account to use
#SBATCH --array=0-999%99             # Default array spec (overridden by sbatch --array)

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

input_dir=$1

# model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/models/fastino/gliner2-multi-v1
# model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/classifiers/gliner2-multi-v1-ft-ep20-bs64-encoderlr1e5-tasklr5e4-wd001-qwen3-235b-v3-description/final
model_name_or_path=/lustre/fswork/projects/rech/ilr/commun/classifiers/gliner2-multi-v1-ft-ep20-bs64-encoderlr5e5-tasklr1e3-wd001-qwen3-235b-v4/final

# Get list of all parquet files and select the one for this array task
mapfile -t files < <(find "$input_dir" -name "*.parquet" | sort)
total_files=${#files[@]}
echo "Total files found: $total_files"

# Check if this array task has a corresponding file
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_files" ]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds number of files ($total_files). Exiting."
    exit 0
fi

# Get the file for this array task
dataset_path=${files[$SLURM_ARRAY_TASK_ID]}

output_dir="${input_dir}_extracted_gliner"
output_path="${output_dir}/${dataset_path##*/}"

echo "Processing file: $dataset_path"
echo "Output will be saved to: $output_path"
echo "Using model: $model_name_or_path"

# Check if output already exists (to avoid reprocessing)
if [ -f "$output_path" ]; then
    echo "Output file already exists. Skipping: $output_path"
    exit 0
fi

# tmp
# if [ $SLURM_ARRAY_TASK_ID -eq 0 -o $SLURM_ARRAY_TASK_ID -eq 1 ]; then
#     exit 0;
# fi

# infer
    # --infer_batch_size 8 \
    # --infer_batch_size 1 \
srun python gliner_annotate.py \
    --input_path "$dataset_path" \
    --output_path "$output_path" \
    --model_name_or_path "$model_name_or_path" \
    --dtype bfloat16 \
    --text_column text \
    --chunking_max_tokens 512 \
    --truncation_max_tokens 512 \
    --threshold 0.5 \
    --hf_map_batch_size 128 \
    --infer_batch_size 8 \
    --use_description \


echo "END TIME: $(date)"
