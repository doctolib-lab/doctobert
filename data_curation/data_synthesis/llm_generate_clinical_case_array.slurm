#!/bin/bash

#SBATCH --job-name=llm_generate_array
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4        # QoS
#SBATCH --time=32:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=ilr@h100           # account to use, with xyz the 3 letter code of your project

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
# conda activate data-proc
conda activate /lustre/fsn1/projects/rech/ilr/uca79wu/conda-envs/vllm

# hf
export HF_HUB_OFFLINE=1

# task name
export GEN_NAME=clinical_case
# prompts
system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/clinical_case/system_prompt_rewrite_clinical_case.txt
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/clinical_case/user_prompt_rewrite_clinical_case.txt

# data
# pmc patients v2
# 250,294 patient summaries
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/en/PMC-Patients/pmc_patients_v2.parquet
# text_column_name=patient
# mimic-iv-notes
# 331,794 de-identified discharge summaries
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv
# 2,321,355 radiology reports
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv.gz
# mimic-iii
# 2,083,180 notes
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimiciii/1.4/noteevents_shards
# text_column_name=text

# array
# input_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge_shards
input_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimiciii/1.4/noteevents_shards
text_column_name=TEXT

# model
# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
# )

# debug
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
model_args=(
    --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-Next-80B-A3B-Instruct
    --temperature 0.7
    --top_p 0.8
    --top_k 20
    --min_p 0.0
    # --speculative_config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
    --max_model_len 16384
    # --max_model_len 262144
    # --max_num_seqs 32
    --gpu_memory_utilization 0.85
)

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Instruct-2507
#     --temperature 0.7
#     --top_p 0.8
#     --top_k 20
#     --min_p 0.0
# )

# model_args=(
#     --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/medgemma-27b-text-it
# )

model_suffix=$(basename ${model_args[1]} | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# output path
# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/pmc_patients_v2/synthesized_${model_suffix}
# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/discharge/synthesized_${model_suffix}
# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/radiology/synthesized_${model_suffix}

# array
output_dir=${input_dir}_synthesized_${model_suffix}

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

# srun python llm_generate.py \
python llm_generate.py \
    "${model_args[@]}" \
    --dataset_path $dataset_path \
    --output_dir $output_path \
    --system_prompt_file $system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name $text_column_name \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_tokens 8192 \
    --batch_size 1000 \
    --shuffle True \

    # --max_samples 1000 \


echo "END TIME: $(date)"
