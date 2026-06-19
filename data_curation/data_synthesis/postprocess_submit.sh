#!/bin/bash


set -e

input_paths=(
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/dialogue/pmc_patients_v2/synthesized_medgemma_27b_text_it
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/dialogue/pmc_patients_v2/synthesized_qwen3_235b_a22b_instruct_2507_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/pmc_patients_v2/synthesized_medgemma_27b_text_it
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/pmc_patients_v2/synthesized_qwen3_235b_a22b_instruct_2507_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/discharge/synthesized_medgemma_27b_text_it
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/discharge/synthesized_medgemma_27b_text_it_part2
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/discharge/synthesized_qwen3_235b_a22b_instruct_2507_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge_shards_synthesized_qwen3_235b_a22b_instruct_2507_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/radiology/synthesized_medgemma_27b_text_it
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/radiology/synthesized_qwen3_235b_a22b_instruct_2507_fp8_250k
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/clinical_case/mimic_iv_notes/radiology/synthesized_qwen3_235b_a22b_instruct_2507_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/icd/cim11_definitions/synthesized_medgemma_27b_text_it
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/vocabulary/academie_medecine_dictionary/synthesized_medgemma_27b_text_it
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/vocabulary/academie_medecine_dictionary/synthesized_qwen3_235b_a22b_instruct_2507_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/en/MIMIC/physionet.org/files/mimiciii/1.4/noteevents_shards_synthesized_qwen3_next_80b_a3b_instruct
    # # final v1
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_new
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten
    # Full-corpus V4.2 array-pipeline outputs (llm_rewrite_array.slurm; nested per-shard subdirs under one base path).
    # Postprocess once per dataset; postprocess_extract.py's recursive **.parquet glob discovers all batches.
    /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8
)

for input_path in "${input_paths[@]}"; do
    echo "Submitting job for $input_path"
    sbatch postprocess.slurm $input_path
done
