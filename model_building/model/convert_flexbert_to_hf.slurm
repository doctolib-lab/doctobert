#!/bin/bash

#SBATCH --job-name=convert_flexbert_to_hf
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

# moderncamembert
# convert_args=(
#     --bos-token-id 1
#     --eos-token-id 2
#     --cls-token-id 1
#     --sep-token-id 2
#     --pad-token-id 0
#     --mask-token-id 4
#     --max-length 1024
# )

# drbert (also used by 07_fr_v2 exp-* RoBERTa runs trained on the legacy drbert_modified
# tokenizer — vocab_size=32064, max_seq_len=512, max_position_embeddings=514).
# Activate this block AND pass `--model-type roberta` on the checkpoint line.
# [PAD] is already at id 1 in drbert_modified — no pad override needed for RoBERTa's
# `position_id = cumsum + pad_id` math (max position = 1+512 = 513, fits in 514 table).
# convert_args=(
#     --bos-token-id 5
#     --eos-token-id 6
#     --cls-token-id 32005
#     --sep-token-id 32006
#     --pad-token-id 1
#     --mask-token-id 32004
#     --max-length 512
# )

# transbert-bio
# convert_args=(
#     --bos-token-id 0
#     --eos-token-id 2
#     --cls-token-id 0
#     --sep-token-id 2
#     --pad-token-id 1
#     --mask-token-id 32001
#     --max-length 512
# )

# doctobert_vocab_50280
# convert_args=(
#     --bos-token-id 50283
#     --eos-token-id 50281
#     --cls-token-id 50283
#     --sep-token-id 50281
#     --pad-token-id 50282
#     --mask-token-id 50284
##     --max-length 1024
#     --max-length 8192
# )

# 07_fr_v2 — ModernBERT (FlexBERT, vocab_50280). Tokenizer dump (transformers):
#   bos/cls=[CLS]=50283  eos/sep=[SEP]=50281  pad=[PAD]=50282  mask=[MASK]=50284  unk=[UNK]=50280
# RoPE means max-length doesn't shape a position table; pick the trained max_seq_len.
# Use 1024 for phase-1 ckpts (100B/200B base runs), 8192 for 8192-from-scratch runs.
convert_args=(
    --bos-token-id 50283
    --eos-token-id 50281
    --cls-token-id 50283
    --sep-token-id 50281
    --pad-token-id 50282
    --mask-token-id 50284
    # --max-length 1024     # phase-1 (max_seq_len=1024)
    --max-length 8192   # 8192-from-scratch (max_seq_len=8192)
)

# 07_fr_v2 — RoBERTa (hf_bert, vocab_32680). Tokenizer dump:
#   bos/cls=[CLS]=32683  eos/sep=[SEP]=32681  pad=[PAD]=32682  mask=[MASK]=32684  unk=[UNK]=32680
# Learned absolute positions → cap at 512 (training max_seq_len=510 + 2 specials).
# NOTE: pad-token-id is FORCED to 1 (not 32682) because RoBERTa's embeddings compute
# position_ids = cumsum(non_pad_mask) + pad_token_id. With pad=32682 and max_pos_emb=512,
# the first non-pad token lookup overflows the 512-entry position table → CUDA device-side
# assert. Setting pad=1 keeps positions in [2, 511]. This is the standard RoBERTa idiom;
# the tokenizer's actual [PAD] mapping (32682) is unaffected. (Failure observed in array 1041908.)
# Activate this block AND pass `--model-type roberta` on the checkpoint line.
# convert_args=(
#     --bos-token-id 32683
#     --eos-token-id 32681
#     --cls-token-id 32683
#     --sep-token-id 32681
#     --pad-token-id 1
#     --mask-token-id 32684
#     --max-length 512
# )

# output_dir=/lustre/fsn1/projects/rech/ilr/commun/checkpoints
output_dir=/lustre/fswork/projects/rech/ilr/commun/DrBenchmark/models

convert () {
    local input_checkpoint=$1
    shift
    local extra_args=("$@")

    # model_name="${input_checkpoint%/*}"
    # model_name="${model_name##*/}-hf"

    model_name=${input_checkpoint%.*}
    model_name="${model_name##*/}"
    parent_dir="${input_checkpoint%/*}"
    parent_dir="${parent_dir##*/}"

    model_name="${parent_dir}-hf-${model_name}"

    # python convert_mosaicbert_to_hf.py \
    python convert_flexbert_to_hf.py \
        --input-checkpoint ${input_checkpoint} \
        --output-dir ${output_dir} \
        --output-name ${model_name} \
        ${convert_args[@]} \
        ${extra_args[@]}

}


input_checkpoints=(
    # =====================================================================================
    # ModernBERT (FlexBERT) — use the vocab_50280 convert_args above.
    #   P1 / large-P1 / Rewarmup-P1 are 1024 -> set --max-length 1024; p2/p3/8192-from-scratch are 8192.
    # Uncomment the checkpoint(s) to convert.
    # =====================================================================================
    # -- P1 base pretrain (1024) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4/ep16-ba21833-rank0.pt"                        # 100B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten/ep7-ba21832-rank0.pt"          # 100B +rewritten
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-plus-n-t/ep5-ba21832-rank0.pt" # 100B +rewritten +NACHOS+transcorpus
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-200b/ep33-ba43665-rank0.pt"                   # 200B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b/ep15-ba43663-rank0.pt"    # 200B +rewritten
    # -- P2 context-extension (8192) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b/ep5-ba4242-rank0.pt"
    # -- P3 annealing (8192, 20B unless noted) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4/ep3-ba4241-rank0.pt"                       # raw med01+edu4
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten/ep1-ba4240-rank0.pt"        # +rewritten
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli/ep4-ba4240-rank0.pt" # +biocli (base recipe)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-dyn-mlm/ep4-ba4240-rank0.pt" # +biocli, dynamic MLM
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med02-edu4-plus-rewritten-rwquality/ep5-ba4240-rank0.pt"      # med02 + rwquality
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-biocli-plus-rewritten-biocli/ep3-ba4240-rank0.pt"  # biocli x biocli
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli/ep10-ba10600-rank0.pt"       # biocli 50B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-mlm03/ep10-ba10600-rank0.pt" # biocli 50B, MLM 0.30
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-100b-med01-edu4-plus-rewritten-biocli/ep21-ba21200-rank0.pt"      # biocli 100B
    # -- P3 doc-length rework (8192) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-shortdoc1024/ep4-ba4240-rank0.pt"       # short-doc
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-20b-med01-edu4-plus-rewritten-biocli-mixdoc1024doc8192/ep2-ba4240-rank0.pt"  # mix-doc 20B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli-mixdoc1024doc8192/ep5-ba10599-rank0.pt" # mix-doc 50B
    # -- 8192-from-scratch --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4/ep15-ba44288-rank0.pt"                     # decay 8e-4->3e-4, P1 200B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-biocli-20b/ep4-ba4240-rank0.pt"            #   -> biocli cooldown 20B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-3e4-biocli-50b/ep10-ba10600-rank0.pt"          #   -> biocli cooldown 50B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0/ep15-ba43665-rank0.pt"                       # decay 8e-4->0, P1 200B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-biocli-br150b-2e4/ep4-ba4240-rank0.pt"       #   -> biocli branch @150B, 20B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-8e4-0-biocli-br150b-2e4-50b/ep10-ba10600-rank0.pt" #   -> biocli branch @150B, 50B
    # -- decay lineage (Rewarmup / ContDecay) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay/ep15-ba43663-rank0.pt"                             # Rewarmup P1 (1024), linear decay->0
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-p2-20b/ep5-ba4242-rank0.pt"                        # p2 ctx-ext
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-p2-20b-p3-exp-50b-med01-edu4-plus-rewritten-biocli/ep4-ba4240-rank0.pt"  # p3 biocli
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b/ep5-ba4242-rank0.pt"                   # ContDecay p2 (branch @126B)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-p3-general-54b-resume2/ep2-ba6359-rank0.pt" # ContDecay p3 general
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-200b-decay-126b-p2-20b-p3-biocli-54b-resume2/ep2-ba2544-rank0.pt"  # ContDecay p3 biocli
    # -- short-trunk / longer-P3 (idea2) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten/ep3-ba9115-rank0.pt"            # WSD P1 ep3 (~42B, 1024)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b/ep2-ba2121-rank0.pt"     # p2 ctx-ext 10B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-general-100b/ep7-ba21197-rank0.pt" # p3 general 100B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-biocli-100b/ep21-ba21200-rank0.pt" # p3 biocli 100B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-finemed-med01-edu4-plus-rewritten-p2-10b-p3-biocli-20b/ep4-ba4240-rank0.pt"    # p3 biocli 20B
    # -- LARGE (28L/1024H) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten/ep7-ba22458-rank0.pt"          # P1 (1024)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-p2-20b/ep5-ba3638-rank0.pt"    # P2 ctx-ext
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-large-pretrain-finemed-med01-edu4-plus-rewritten-p2-20b-p3-biocli-20b/ep4-ba3636-rank0.pt" # P3 biocli
    #
    # =====================================================================================
    # RoBERTa (hf_bert) — swap convert_args to the vocab_32680 block; pass --model-type roberta.
    # =====================================================================================
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4/ep32-ba92308-rank0.pt --model-type roberta"                  # P1 200B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten/ep14-ba106710-rank0.pt --model-type roberta"  # P1 200B +rewritten
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-500b-new/ep36-ba266775-rank0.pt --model-type roberta"  # P1 500B +rewritten
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-1t-lr1e4/ep73-ba533549-rank0.pt --model-type roberta"  # P1 1T
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality/ep19-ba47690-rank0.pt --model-type roberta"      # biocli continual 100B (from 200B P1)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-200b/ep39-ba95379-rank0.pt --model-type roberta" # biocli continual 200B (from 200B P1)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-200b-cp/ep39-ba95379-rank0.pt --model-type roberta" # biocli continual 200B (from 500B P1) -- final/shipped
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-plus-rewritten-biocli-rwquality-500b-cp/ep47-ba114455-rank0.pt --model-type roberta" # biocli continual, last pre-spike (~300B)
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-fr-v2-base-pretrain-roberta-finemed-med01-edu4-biocli-plus-rewritten-biocli/ep16-ba50085-rank0.pt --model-type roberta"          # biocli x biocli 100B
    # -- RoBERTa ablations (legacy drbert_modified tokenizer, fineweb-2 only) --
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-exp-base-pretrain-roberta-fw2-med01-edu4/ep249-ba93146-rank0.pt --model-type roberta"               # fw2 med01+edu4 200B
    # "/lustre/fsn1/projects/rech/ilr/commun/checkpoints/07_fr_v2/doctobert-exp-base-pretrain-roberta-fw2-med01-edu4-plus-rewritten/ep794-ba136637-rank0.pt --model-type roberta" # fw2 +rewritten 200B
)

for input_checkpoint in "${input_checkpoints[@]}"; do
    convert ${input_checkpoint}
done

# Clean up temporary tokenizer directories created by composer
rm -rf tokenizer-save-dir-*

echo "END TIME: $(date)"a
