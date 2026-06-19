#!/bin/bash

set -euo pipefail

dataset_paths=(
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_qwen3.5_122b_a10b_fp8_postprocessed2_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_qwen3.5_27b_fp8_postprocessed2_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_qwen3.5_9b_postprocessed2_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_gemma_4_26b_a4b_it_postprocessed2_extracted_gliner
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner_rewritten_gemma_4_31b_it_postprocessed2_extracted_gliner
)

for dataset_path in "${dataset_paths[@]}"; do
    # sbatch llm_rewrite_eval.slurm $dataset_path
    # tmp: script to summarize eval stats after rewriting (compression, entity density, LLM score)
    python llm_rewrite_eval_stats.py --dataset_path "${dataset_path}_eval_intellect_3_fp8" --top_k 5000 --num_proc 16
done
