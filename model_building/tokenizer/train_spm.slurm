#!/bin/bash

#SBATCH --job-name=train_spm
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.out       # error file (%j = job ID)
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

# set up environment
module purge
# module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
# module load cuda/12.8.0
# env
# conda activate data-proc
conda activate bert24

# install spm cli
# conda install -n data-proc -c anaconda sentencepiece-spm

output_root_dir=/lustre/fswork/projects/rech/ilr/commun/pretrained/tokenizers

corpus_root_dirs=(
    # v1.1
    # # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/nachos
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/nachos_clinical
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/fineweb2_clinical
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finepdfs_clinical
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/transcorpus_bio_fr_clinical
    # # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/fineweb2_others_05
    # # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finepdfs_others_05
    # # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/transcorpus_bio_fr_others
    # v1.2
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/nachos_edu4_selected_domains
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/fineweb2_edu4_selected_domains
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finepdfs_edu4_selected_domains
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/transcorpus_bio_fr_edu4_selected_domains
    # v1.3: med01 + edu4 (drops select_domains), gliner-extracted sources, adds finewiki
    # Active: this is the corpus used to train the existing `fineweb-2_med01_edu4_finepdfs_med01_edu4_finewiki_med01_edu4_vocab_32680_hf` tokenizer.
    /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/fineweb-2_med01_edu4
    /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finepdfs_med01_edu4
    /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finewiki_med01_edu4
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/nachos_med01_edu4
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/transcorpus_bio_fr_med01_edu4
    # v1.4: v1.3 + bio+clinical 5-subtopic select_domains filter
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/fineweb-2_med01_edu4_biocli5
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finepdfs_med01_edu4_biocli5
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/finewiki_med01_edu4_biocli5
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/nachos_med01_edu4_biocli5
    # /lustre/fsn1/projects/rech/ilr/commun/corpus/tokenizer/transcorpus_bio_fr_med01_edu4_biocli5
)

# vocab_size=50280  #50254  # match ModernBERT base size
# final_vocab_size=50368

vocab_size=32680
final_vocab_size=32768

# `for_roberta`: if True, train_spm.py reserves [UNK]/[PAD]/[CLS]/[SEP] at SPM ids 0-3
# and [MASK] at id 4, and convert_spm_to_hf.py registers them by name (no re-append).
# Required for HF RoBERTa pretraining so `cumsum + pad_id` fits the position table.
# Leave False for ModernBERT / RoPE archs where pad_id placement doesn't matter.
for_roberta=True

corpus_files=()
for root_dir in "${corpus_root_dirs[@]}"; do
    while IFS= read -r -d '' f; do
        corpus_files+=("$f")
    done < <(find "$root_dir" -type f -name "*.txt" -print0)
done

# Join with commas for SPM input (preserves spaces in paths)
printf -v corpus '%s,' "${corpus_files[@]}"; corpus=${corpus%,}

# Derive output tag from root directory basenames
roots_basenames=()
for r in "${corpus_root_dirs[@]}"; do
    roots_basenames+=("$(basename "$r")")
done
corpus_tag=$(IFS=_; echo "${roots_basenames[*]}")

output_suffix=""
[ "$for_roberta" = "True" ] && output_suffix="_roberta"
output_dir="${output_root_dir}/${corpus_tag}_vocab_${vocab_size}${output_suffix}"
[ -d "$output_dir" ] || mkdir -p "$output_dir"

# train_args=(
#     # I/O
#     "--input=$corpus"                           # input files (comma-separated)
#     "--model_prefix=$output_dir/model"          # output prefix
#     --input_sentence_size=1000000              # sample 5M lines
#     --shuffle_input_sentence=true              # uniform sampling boosts coverage
#     # Model / vocab
#     --model_type=bpe                            # unigram or bpe
#     "--vocab_size=$vocab_size"                  # try 30k-50k
#     --byte_fallback=true                        # guarantees zero UNKs via byte backoff
#     --character_coverage=0.9995                 # by default: 0.9995, or 0.9999, 1.0
#     --hard_vocab_limit=false                    # soft-limit so it won't crash on small corpora
#     # Normalization & whitespace behavior
#     # --normalization_rule_name=identity          # nmt_nfkc or identity; default nmt_nfkc maps all Unicode whitespace to a space and collapses repeats
#     --remove_extra_whitespaces=false            # removes leading, trailing, and duplicate internal whitespace
#     --allow_whitespace_only_pieces=true         # allows whitespace-only pieces to be added to the vocabulary
#     --add_dummy_prefix=true                     # standard "▁" word boundary marker
#     # --split_by_whitespace=false               # whitespace is not used as a delimiter but is encoded into tokens via the meta-symbol
#     # --split_by_unicode_script=false           # let BPE learn across Latin + Greek letters
#     # Number / digit splitting
#     --split_by_number=true                      # "120 mg" → ["120", " mg"]
#     --split_digits=true                         # "2025" → ["2", "0", "2", "5"]
#     # '--required_chars=0123456789'               # ensure digits are covered
#     # Limits & performance
#     --max_sentence_length=8192                  # for long lines
#     --num_threads=${SLURM_CPUS_PER_TASK:-16}    # align with allocated CPU cores
#     # --num_threads=8    # align with allocated CPU cores
#     # Special tokens
#     --unk_id=0 "--unk_piece=<unk>"              # keep SPM's internal <unk> (not used in HF)
#     --bos_id=-1 --eos_id=-1 --pad_id=-1         # disable BOS, EOS, PAD
#     # '--user_defined_symbols=\n,\r\n,\t'              # literal backslash-n and backslash-t
# )

# spm_train "${train_args[@]}" \
    # --user_defined_symbols=$'\n',$'\r\n',$'\t'


python train_spm.py \
    --corpus "$corpus" \
    --model_prefix "$output_dir/model" \
    --vocab_size "$vocab_size" \
    --num_threads "${SLURM_CPUS_PER_TASK:-16}" \
    --for_roberta "$for_roberta" \

    # --max_lines 150000000
    # --max_lines 170000000
    # --max_lines 100000

# convert to HF
################################################################################
hf_output_dir="${output_dir}_hf"
[ -d "$hf_output_dir" ] || mkdir -p "$hf_output_dir"

python convert_spm_to_hf.py \
    --spm_path "$output_dir/model.model" \
    --output_dir "$hf_output_dir" \
    --target_vocab_size "$final_vocab_size" \
    --for_roberta "$for_roberta"

# check the tokenizer
################################################################################
HF_OUTPUT_DIR="${hf_output_dir}" python - <<'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ["HF_OUTPUT_DIR"])
print(len(tok), tok.special_tokens_map)
s = """Patient avec hypercholestérolémie familiale Patient patient, 500 mg/jour."
Bonjour docteur !
  Comment allez-vous ?"""
print(tok.tokenize(s))
PY

echo "END TIME: $(date)"
