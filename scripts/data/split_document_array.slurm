#!/bin/bash

#SBATCH --job-name=split_document_array
#SBATCH --output=logs/%x/%A_%a.out    # %A = array job ID, %a = array task ID
#SBATCH --error=logs/%x/%A_%a.err     # %A = array job ID, %a = array task ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=kit@cpu            # cpu accounting

set -x -e

echo "START TIME: $(date)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# set up environment
module purge
# module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
# conda activate bert24
conda activate data-proc

# tokenizer_name_or_path=/lustre/fswork/projects/rech/kit/commun/pretrained/models/almanach/moderncamembert-base
tokenizer_name_or_path=/lustre/fswork/projects/rech/kit/commun/pretrained/tokenizers/drbert_modified
tokenizer_name="${tokenizer_name_or_path##*/}"
tokenizer_name="${tokenizer_name//-/_}"

max_tokens=1024
input_dir=/lustre/fsn1/projects/rech/kit/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored
output_dir=${input_dir}_split_max_${max_tokens}_tokens_${tokenizer_name}

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
input_path=${files[$SLURM_ARRAY_TASK_ID]}

output_path="${output_dir}/${input_path##*/}"

echo "Processing file: $input_path"
echo "Output will be saved to: $output_path"

# Check if output already exists (to avoid reprocessing)
if [ -f "$output_path" ]; then
    echo "Output file already exists. Skipping: $output_path"
    exit 0
fi


python split_document.py \
    --input_path "${input_path}" \
    --output_path "${output_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --max_tokens $max_tokens \
    --shuffle True \
    --num_proc 64 \


echo "END TIME: $(date)"
