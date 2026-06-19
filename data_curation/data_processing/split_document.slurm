#!/bin/bash

#SBATCH --job-name=split_document
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 10:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=ilr@cpu            # cpu accounting

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
# module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
conda activate data-proc
# conda activate bert24

# train/test split
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/DOCUMENT_BY_DOCUMENT.txt"
# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/line_by_line.txt"
# python - <<PY
# from datasets import load_dataset

# ds = load_dataset("text", data_files="$input_path", split="train")
# ds = ds.train_test_split(test_size=5_000, seed=42)
# ds["train"].to_parquet("${input_path%.*}_train.parquet")
# ds["test"].to_parquet("${input_path%.*}_test.parquet")
# # ds["train"].save_to_disk("${input_path%.*}_train")
# # ds["test"].save_to_disk("${input_path%.*}_test")
# # ds["train"].to_json("${input_path%.*}_train.jsonl", orient="records", lines=True, force_ascii=False)
# # ds["test"].to_json("${input_path%.*}_test.jsonl", orient="records", lines=True, force_ascii=False)
# PY

# input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/DOCUMENT_BY_DOCUMENT_TRAIN.parquet"
# max_words=630
# output_path="${input_path%.*}_SPLIT_MAX_${max_words}_WORDS.parquet"

# srun python split_document.py \
#     --input_path "${input_path}" \
#     --output_path "${output_path}" \
#     --max_words $max_words \
#     --num_proc 40 \

split_func () {
    local input_path=$1
    local max_tokens=$2
    local tokenizer_name_or_path=$3

    tokenizer_name="${tokenizer_name_or_path##*/}"
    tokenizer_name="${tokenizer_name//-/_}"
    output_path="${input_path%.*}_split_max_${max_tokens}_tokens_${tokenizer_name}.parquet"

    python split_document.py \
        --input_path "${input_path}" \
        --output_path "${output_path}" \
        --tokenizer_name_or_path "${tokenizer_name_or_path}" \
        --max_tokens $max_tokens \
        --shuffle True \
        --num_proc 64 \
        --output_shard_size 1000000 \

}

# tokenizer_name_or_path="/lustre/fswork/projects/rech/ilr/commun/pretrained/models/almanach/moderncamembert-base"
tokenizer_name_or_path="/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/drbert_modified"

max_tokens=512
input_path="/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/line_by_line.txt"
split_func $input_path $max_tokens $tokenizer_name_or_path

echo "END TIME: $(date)"
