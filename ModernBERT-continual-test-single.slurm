#!/bin/bash
#SBATCH --job-name=modernbert_continual_single
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --partition=gpu_p6
#SBATCH --account=kit@h100
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --qos=qos_gpu_h100-dev

# Load modules
module purge
module load arch/h100 pytorch-gpu/py3/2.6.0

# Create log directory if it doesn't exist
mkdir -p log

# Print some useful information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"

srun python -u main.py yamls/test/modernbertcamembert-base.yaml 