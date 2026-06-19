#!/bin/bash

#SBATCH --job-name=eval_tokenizer
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
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
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
conda activate data-proc
# conda activate bert24

export TMPDIR=/lustre/fsn1/projects/rech/ilr/commun/tmp

# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/NACHOS/DOCUMENT_BY_DOCUMENT_SPLITTED_750_WORDS.txt

tokenizer_paths=(
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/models/thomas-sounack/BioClinical-ModernBERT-base
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/models/almanach/moderncamembert-base
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Dr-BERT/DrBERT-7GB
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/models/almanach/camembert-bio-base
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/models/jknafou/TransBERT-bio-fr
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb2_health_science_nachos_vocab_50280_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb2_health_science_nachos_vocab_32680_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_vocab_32680_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_vocab_50280_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb2_selected_domains_edu2_nachos_edu2_vocab_50280_n_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_raw_vocab_50280_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_raw_vocab_32680_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_raw_vocab_32680_unigram_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_fineweb2_clinical_finepdfs_clinical_transcorpus_bio_fr_clinical_fineweb2_others_05_finepdfs_others_05_transcorpus_bio_fr_others_vocab_50280_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_fineweb2_clinical_finepdfs_clinical_transcorpus_bio_fr_clinical_fineweb2_others_05_finepdfs_others_05_transcorpus_bio_fr_others_vocab_50280_new_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_clinical_fineweb2_clinical_finepdfs_clinical_transcorpus_bio_fr_clinical_vocab_50280_200m_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_edu4_selected_domains_fineweb2_edu4_selected_domains_finepdfs_edu4_selected_domains_transcorpus_bio_fr_edu4_selected_domains_vocab_50280_150m_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_50280_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_nachos_med01_edu4_transcorpus_bio_fr_med01_edu4_vocab_50280_150m_lines_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/nachos_med01_edu4_vocab_32680_hf
    # /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_biocli5_finepdfs_med01_edu4_biocli5_finewiki_med01_edu4_biocli5_vocab_32680_hf
    /lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers/fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_roberta_hf
)

for tokenizer_name_or_path in "${tokenizer_paths[@]}"; do
    tokenizer_base=${tokenizer_name_or_path##*/}  # strip leading path
    tokenizer_base=${tokenizer_base//-/_}  # replace - by _
    tokenizer_base=${tokenizer_base,,}  # to lowercase

    # srun python eval_tokenizer.py \
    python eval_tokenizer.py \
        --tokenizer_name_or_path "$tokenizer_name_or_path" \
        --output_file "tmp_results/tokenizer_stats_${tokenizer_base}.csv" \
        --num_workers 8
done


echo "END TIME: $(date)"
