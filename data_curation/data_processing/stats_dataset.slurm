#!/bin/bash

#SBATCH --job-name=stats_dataset
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3             # QoS
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


# Old per-corpus loop — disabled; per-corpus rephrased stats already on disk
# at scripts/data/logs/stats_dataset/1057561.out, and source/8k-split per-corpus
# stats are covered by older logs (827818, 823220, 826886).
# input_paths=(
#     # fineweb-2
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/science
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
#     # finepdfs
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/science_edu_quality_scored
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/computers_and_electronics_edu_quality_scored
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
#     # finewiki
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
#     # NACHOS
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
#     # transcorpus_bio_fr
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner
#     # others (disabled in rewriting pipeline)
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/multilingual_medical_corpus/health_domain_classified_edu_quality_scored_extracted_gliner
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/E3C/layer3/health_domain_classified_edu_quality_scored_extracted_gliner
#     # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/final/v2_extracted_gliner
#
#     # Rephrased + edu-rescored — edu_quality_* = NEW scores; original_edu_quality_* = pre-rephrase scores
#     # fw2  (edu rescore wave 2, job 868063)
#     /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_edu_quality_scored
#     # finepdfs (edu rescore wave 3, job 1044323; note _10shards reshard suffix — input resplit 2→10 for the wave-3 retry)
#     /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_10shards_edu_quality_scored
#     # finewiki (edu rescore wave 2, job 868064)
#     /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_edu_quality_scored
# )
#
# for input_path in "${input_paths[@]}"; do
#     echo
#     echo "============================================================"
#     echo "Dataset: ${input_path}"
#     echo "============================================================"
#
#     python stats_dataset.py \
#         --input_path "${input_path}" \
#         --text_column_name "text" \
#         --num_workers 16
# done


# ----------------------------------------------------------------------------
# Per-corpus FineMed-filtered (source `_extracted_gliner/` + med01 ∧ edu4
# applied in-memory). Doc-level filter — matches the population that downstream
# rewriting expanded into the rephrased corpus. NACHOS / transcorpus omitted
# because the LaTeX FineMed-filtered family only covers fw2 + finepdfs + finewiki.
# ----------------------------------------------------------------------------

filtered_per_corpus=(
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
)

for input_path in "${filtered_per_corpus[@]}"; do
    echo
    echo "============================================================"
    echo "Per-corpus filtered (med01 ∧ edu4): ${input_path}"
    echo "============================================================"

    python stats_dataset.py \
        --input_path "${input_path}" \
        --apply_med01_edu4_filter True \
        --text_column_name "text" \
        --num_workers 16
done


# ----------------------------------------------------------------------------
# Family aggregates: FineMed (source) / FineMed-filtered / FineMed-rephrased.
# Each invocation passes a comma-joined list of paths so stats_dataset.py
# concatenates them on the column-name intersection and treats the union as
# a single corpus.
#
# FineMed-filtered has no on-disk path — the canonical Stage-1 filter
# (med ≥ 0.10 ∧ edu ≥ 4) is applied in-memory on the pre-8k-split source
# corpora (same input set as the per-corpus filtered loop above), giving
# doc-level filter survivors.
# ----------------------------------------------------------------------------

# --- FineMed (source aggregate: fw2 + finepdfs + finewiki, no filter) ---
finemed_source="\
/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner,\
/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner,\
/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner"

echo
echo "============================================================"
echo "Group: FineMed (source aggregate)"
echo "============================================================"
python stats_dataset.py \
    --input_path "${finemed_source}" \
    --text_column_name "text" \
    --num_workers 16

# --- FineMed-filtered (pre-8k-split source + med01 ∧ edu4 applied in-memory) ---
echo
echo "============================================================"
echo "Group: FineMed-filtered (source + med01 ∧ edu4 in-memory)"
echo "============================================================"
python stats_dataset.py \
    --input_path "${finemed_source}" \
    --apply_med01_edu4_filter True \
    --text_column_name "text" \
    --num_workers 16

# --- FineMed-rephrased (post-edu-rescore aggregate, no filter) ---
finemed_rephrased="\
/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_edu_quality_scored,\
/lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_10shards_edu_quality_scored,\
/lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_edu_quality_scored"

echo
echo "============================================================"
echo "Group: FineMed-rephrased (post-edu-rescore aggregate)"
echo "============================================================"
python stats_dataset.py \
    --input_path "${finemed_rephrased}" \
    --text_column_name "text" \
    --num_workers 16


echo "END TIME: $(date)"
