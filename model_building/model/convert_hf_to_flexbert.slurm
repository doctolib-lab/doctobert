#!/bin/bash

#SBATCH --job-name=convert_hf_to_flexbert
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --time 02:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev            # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=ilr@cpu            # cpu accounting


set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
# module load arch/h100
# module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
conda activate bert24

convert () {
    local hf_model_path=$1
    local root_output_dir=$2

    python convert_hf_to_flexbert.py \
        --hf_model_path ${hf_model_path} \
        --output_path ${root_output_dir}/${hf_model_path##*/} \

}

root_output_dir=/lustre/fsn1/projects/rech/ilr/commun/checkpoints/03_fr/doctobert-fr-base-pretrain-66b-converted

hf_model_paths=(
    /lustre/fswork/projects/rech/ilr/commun/DrBenchmark/models/doctobert-03-fr-base-pretrain-66b-hf-ep0-ba4876-rank0
    /lustre/fswork/projects/rech/ilr/commun/DrBenchmark/models/doctobert-03-fr-base-pretrain-66b-hf-ep1-ba9327-rank0
    /lustre/fswork/projects/rech/ilr/commun/DrBenchmark/models/doctobert-03-fr-base-pretrain-66b-hf-ep3-ba14625-rank0
    /lustre/fswork/projects/rech/ilr/commun/DrBenchmark/models/doctobert-03-fr-base-pretrain-66b-hf-ep3-ba18379-rank0
    /lustre/fswork/projects/rech/ilr/commun/DrBenchmark/models/doctobert-03-fr-base-pretrain-66b-hf-ep4-ba21983-rank0
)

for hf_model_path in ${hf_model_paths[@]}; do
    convert ${hf_model_path} ${root_output_dir}
done

echo "END TIME: $(date)"
