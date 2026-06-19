#!/bin/bash

# Helper script to submit the array job with correct number of files

set -e

# Set maximum concurrent jobs (adjust based on your quota/needs)
max_concurrent=99

# SLURM script to submit
slurm_script=gliner_annotate_array.slurm

# Directory containing input files
# input_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k
# input_dir=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored
input_dir=$1

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
# Debug mode: submit only the first task if --debug arg or DEBUG=1
if [ "$2" = "--debug" ] || [ "$DEBUG" = "1" ]; then
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
sbatch --array="$array_spec" "$slurm_script" "$input_dir"

# echo "Array job submitted successfully!"
# echo "Use 'squeue -u \$USER' to monitor job status"
# echo "Use 'sacct -j <JOB_ID>' to see detailed job information"
