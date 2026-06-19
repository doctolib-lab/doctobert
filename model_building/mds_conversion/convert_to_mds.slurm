#!/bin/bash

#SBATCH --job-name=convert_mds
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
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

echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

export TMPDIR=/lustre/fsn1/projects/rech/ilr/commun/tmp

# tokenizer_name_or_path is optional — leave unset to skip the per-worker tokenizer
# load (faster, since the tokenization map is currently disabled in worker code).
# tokenizer_name_or_path="/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/drbert_modified"

# One `name|path` per line. The `name` tag is the row-name used by _filter_func_p1:
#   "fineweb2_rewritten" → short-circuits the quality gates (universal hygiene still applies)
#   anything else         → routes through _filter_func_p1_quality (gated by `mode` knobs below)
inputs=(
    # split with max 1024 tokens (drbert_modified tokenizer)
    # "fineweb2|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1024_tokens_drbert_modified"
    # "finepdfs|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1024_tokens_drbert_modified"
    # "finewiki|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1024_tokens_drbert_modified"
    # "nachos|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1024_tokens_drbert_modified"
    #
    # # # final v2 (split with max 8192 tokens) — INPUT FOR p2/p3 ONLY
    "fineweb2|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified"
    "finepdfs|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified"
    "finewiki|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified"
    # # "nachos|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified"
    # # "transcorpus|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner"
    # # rewritten
    "fineweb2_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified"
    "finepdfs_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified_10shards"
    "finewiki_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified"
    #
    # final v2 (split with max 510 tokens) — INPUT FOR p1 (resuming 860981)
    # raw filtered (med01 ∧ edu4 already applied at split time)
    # "fineweb2|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_510_tokens_fineweb_2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf"
    # "finepdfs|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_510_tokens_fineweb_2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf"
    # "finewiki|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_510_tokens_fineweb_2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf"
    # "nachos|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_510_tokens_fineweb_2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf"
    # "transcorpus|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner"
    # rewritten
    # "fineweb2_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_510_tokens_vocab_32680_hf"
    # "finepdfs_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_postprocessed2_extracted_gliner_split_max_8192_10shards_split_max_510_tokens_vocab_32680"
    # "finewiki_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_510_tokens_vocab_32680_hf"
    #
    # final v2 (split with max 1022 tokens) — INPUT FOR p1 (1024-context pretraining)
    # # raw filtered (med01 ∧ edu4 applied inside split_document.py before chunking)
    "fineweb2|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1022_tokens_vocab_50280_hf"
    "finepdfs|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1022_tokens_vocab_50280_hf"
    "finewiki|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1022_tokens_vocab_50280_hf"
    # # "nachos|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_split_max_1022_tokens_vocab_50280_hf"
    # # "transcorpus|/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner"
    # # rewritten
    "fineweb2_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_postprocessed2_extracted_gliner_split_max_8192_split_max_1022_tokens_vocab_50280"
    "finepdfs_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_postprocessed2_extracted_gliner_split_max_8192_10shards_split_max_1022_tokens_vocab_50280"
    "finewiki_rewritten|/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_postprocessed2_extracted_gliner_split_max_8192_split_max_1022_tokens_vocab_50280"
)

# Split into parallel `names` / `paths` arrays, then re-join as CSV strings for python.
names=()
paths=()
for entry in "${inputs[@]}"; do
    names+=("${entry%%|*}")
    paths+=("${entry#*|}")
done
name_csv=$(IFS=','; echo "${names[*]}")
input_paths_csv=$(IFS=','; echo "${paths[*]}")
name_joined=$(IFS=_; echo "${names[*]}")

# Filter knobs (substring-matched in _filter_func_p1_quality):
#   subtopic_biocli5 / edu2 / edu4 / med01 / med02 / union
# Mode is split on `_plus_rewritten` into per-side knob strings:
#   <raw-knobs>_plus_rewritten_<rewritten-knobs>
# Either side may be empty:
#   "raw_edu4_med01"                       → raw=edu4+med01, no rewritten in mix
#   "raw_edu4_med01_plus_rewritten"        → raw=edu4+med01, rewritten passes through (universal hygiene only)
# Stage gates (substring-matched in the worker, NOT in _filter_func_p1_quality):
#   p2down → length-aware downsample for p2 context-extension corpus
#            (keeps all docs >= p2_long_threshold, drops < p2_short_floor,
#             samples shorts down to shrink each parquet by p2_shrink_alpha).
# "default" → no filters, no p2 downsample.
# mode="raw_edu4_med01"
# mode="raw_edu4_med01_subtopic_biocli5"
# mode="raw_edu4_med01_plus_rewritten"                                                  # rewritten: no gates
# mode="raw_edu4_med02_plus_rewritten_edu4_med02"                                       # symmetric gates (old `rwquality`)
# mode="raw_edu4_med01_plus_rewritten_plus_n_t"
# mode="raw_edu4_med01_plus_rewritten_p2down"                                           # activates p2 length-aware downsample
# mode="raw_edu4_med01_subtopic_biocli5_plus_rewritten_subtopic_biocli5"                # separate: raw gated, rewritten biocli5 only (the 3.56B variant)
mode="raw_edu4_med01_subtopic_biocli5_plus_rewritten_edu4_med01_subtopic_biocli5"     # symmetric gates on both sides (explicit; reproduces the legacy rwquality 3.03B data)

# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/final_mds_v2/${name_joined}_split_max_1024_tokens_drbert_modified_${mode}
# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/final_mds_v2/${name_joined}_split_max_510_tokens_vocab_32680_${mode}
# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/final_mds_v2/${name_joined}_split_max_1022_tokens_vocab_50280_${mode}
# mix: 1022 short fragments + 8192 whole long docs (name_joined doubles with 12 sources, so hardcode prefix)
output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/final_mds_v2/fineweb2_finepdfs_finewiki_fineweb2_rewritten_finepdfs_rewritten_finewiki_rewritten_split_max_mix_1022_8192_vocab_50280_${mode}
# output_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/final_mds_v2/${name_joined}_split_max_8192_tokens_${mode}

python convert_to_mds.py \
    --input_paths "${input_paths_csv}" \
    --name "${name_csv}" \
    --mode "${mode}" \
    --output_dir "${output_dir}" \
    --num_workers 32 \
    --test_size 0.00001 \

echo "END TIME: $(date)"
