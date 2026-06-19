#!/bin/bash

#SBATCH --job-name=split_document_array
#SBATCH --output=logs/%x/%A_%a.out    # %A = array job ID, %a = array task ID
#SBATCH --error=logs/%x/%A_%a.err     # %A = array job ID, %a = array task ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=ilr@cpu            # cpu accounting

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

# tokenizer_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/models/almanach/moderncamembert-base
# tokenizer_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/drbert_modified
# tokenizer_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/doctobert_vocab_50280
# final
tokenizer_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf
# tokenizer_name_or_path=/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_50280_hf

tokenizer_name="${tokenizer_name_or_path##*/}"
tokenizer_name="${tokenizer_name//-/_}"
# Override with a short label — the auto-derived basename (~75 chars) pushes the
# output dir past the 255-byte FS limit on the rewritten-corpora chain.
# Comment out to use the long auto-derived name (only safe for non-rewritten input).
tokenizer_name=vocab_32680
# tokenizer_name=vocab_50280

max_tokens=510
# max_tokens=1022
# max_tokens=8192

# input_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored
input_dir=$1

output_dir=${input_dir}_split_max_${max_tokens}_tokens_${tokenizer_name}

# tmp: shorten the output basename to stay under the 255-byte FS limit on the
# rewritten-corpora chain. Gated on `_rewritten` so existing raw 1022 outputs
# (at the long name) are not orphaned by future raw re-runs.
# Downstream readers (convert_to_mds.slurm etc.) must reference these aliased
# forms for rewritten paths.
if [[ "$input_dir" == *_rewritten_* ]]; then
    output_dir="${output_dir//_split_max_8192_tokens_drbert_modified/_split_max_8192}"
    output_dir="${output_dir//_qwen3.5_35b_a3b_fp8/_qwen3.5}"
fi

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
    --num_proc 40 \

echo "END TIME: $(date)"
