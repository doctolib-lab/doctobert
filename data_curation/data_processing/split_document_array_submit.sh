#!/bin/bash

# Helper script to submit the array job with correct number of files

set -e

# Parse debug flag from args or env
DEBUG="${DEBUG:-0}"
for arg in "$@"; do
    if [ "$arg" = "--debug" ] || [ "$arg" = "-d" ]; then
        DEBUG=1
        break
    fi
done

# Set maximum concurrent jobs (adjust based on your quota/needs)
max_concurrent=100

# SLURM script to submit
slurm_script=split_document_array.slurm

submit_job () {
    local input_dir=$1

    # Count the number of parquet files
    echo "Counting parquet files in: $input_dir"
    file_count=$(find "$input_dir" -name "*.parquet" | wc -l)

    if [ "$file_count" -eq 0 ]; then
        echo "Error: No parquet files found in $input_dir"
        exit 1
    fi

    echo "Found $file_count parquet files"

    # Calculate array range (0-based indexing)
    array_end=$((file_count - 1))

    # Create array specification
    # Debug mode: submit only the first task if DEBUG=1 (env or parsed from args)
    if [ "$DEBUG" = "1" ]; then
        echo "DEBUG mode enabled: only submitting the first array task (0)"
        array_spec="0"
    else
        if [ "$file_count" -le "$max_concurrent" ]; then
            array_spec="0-$array_end"
        else
            array_spec="0-$array_end%$max_concurrent"
        fi
    fi

    echo "Submitting array job with specification: $array_spec"

    # Submit the job
    # sbatch --array="$array_spec" $slurm_script
    sbatch --array="$array_spec" $slurm_script $input_dir

    echo "Array job submitted successfully!"
    echo "Use 'squeue -u \$USER' to monitor job status"
    echo "Use 'sacct -j <JOB_ID>' to see detailed job information"

}

# Directory containing input files
input_dirs=(
    # filtered v2 (raw)
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/transcorpus_bio_fr/transcorpus_bio_fr_edu_quality_scored_health_domain_classified_extracted_gliner
    # filtered v2 (8192-split)
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/processed/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified
    # rewritten v2 (raw)
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner
    # rewritten v2 (8192-split)
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified
    ## /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified
    /lustre/fsn1/projects/rech/ilr/commun/corpus/finepdfs/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified_10shards
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/finewiki/data/frwiki_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified_rewritten_mga_stage2_v4_2_3m1n_qwen3.5_35b_a3b_fp8_postprocessed2_extracted_gliner_split_max_8192_tokens_drbert_modified
    # others
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/multilingual_medical_corpus/health_domain_classified_edu_quality_scored_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/E3C/layer3/health_domain_classified_edu_quality_scored_extracted_gliner
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/fr/synthesized/final/v2_extracted_gliner
)

for input_dir in "${input_dirs[@]}"; do
    submit_job $input_dir
done
