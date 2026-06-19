#!/bin/bash

#SBATCH --job-name=llm_rewrite
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3        # QoS
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=ilr@h100           # account to use, with xyz the 3 letter code of your project

set -euo pipefail

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
# module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.8.0
# env
# conda activate bert24
conda activate data-proc

export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR=/lustre/fsn1/projects/rech/ilr/commun/tmp

model_name=${1:-qwen3.5-35b-a3b}

# user prompt (shared by all rewriting variants)
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/user_prompt.txt

# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1
# dataset used to benchmark rewriting models
# dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_multi_v1_extracted_gliner
# final fineweb-2 data
dataset_path=/lustre/fsn1/projects/rech/ilr/commun/corpus/fineweb-2/data/fra_Latn/train_domain_classified_filtered/health_domain_classified_edu_quality_scored_extracted_gliner_split_max_8192_tokens_drbert_modified

# medgemma
if [ "$model_name" == "medgemma-27b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/medgemma-27b-text-it
        --max_num_batched_tokens 8192
    )
    suffix=medgemma_27b
# gemma-4
elif [ "$model_name" == "gemma-4-31b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/gemma-4-31B-it
        --temperature 1.0
        --top_p 0.95
        --top_k 64
        --enable_thinking False
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=gemma_4_31b_it
elif [ "$model_name" == "gemma-4-26b-a4b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/gemma-4-26B-A4B-it
        --temperature 1.0
        --top_p 0.95
        --top_k 64
        --enable_thinking False
        --max_num_batched_tokens 8192
    )
    suffix=gemma_4_26b_a4b_it
# qwen3
elif [ "$model_name" == "qwen3-30b-a3b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Instruct-2507
        # --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        # --enable_thinking False
        --max_num_batched_tokens 8192
        # --gpu_memory_utilization 0.85
    )
    suffix=qwen3_30b_a3b_instruct_2507
elif [ "$model_name" == "qwen3-30b-a3b-thinking" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Thinking-2507
        # --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
        --temperature 0.6
        --top_p 0.95
        --top_k 20
        --min_p 0.0
        --max_num_batched_tokens 8192
        # --gpu_memory_utilization 0.85
    )
    suffix=qwen3_30b_a3b_thinking_2507
elif [ "$model_name" == "qwen3-next-80b-a3b" ]; then
    # export VLLM_WORKER_MULTIPROC_METHOD=spawn
    model_args=(
        # --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-Next-80B-A3B-Instruct
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        # --max_model_len 16384
        # --max_num_seqs 32
        # --speculative_config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=qwen3_next_80b_a3b_instruct_fp8
elif [ "$model_name" == "qwen3-235b-a22b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        --enable_thinking False
        # --max_model_len 16384
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=qwen3_235b_a22b_instruct_2507_fp8
# qwen3.5
elif [ "$model_name" == "qwen3.5-9b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3.5-9B
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        # --presence_penalty 1.5
        --enable_thinking False
        --max_num_batched_tokens 8192
    )
    suffix=qwen3.5_9b
elif [ "$model_name" == "qwen3.5-27b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3.5-27B-FP8
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        # --presence_penalty 1.5
        --enable_thinking False
        --max_num_batched_tokens 8192
    )
    suffix=qwen3.5_27b_fp8
elif [ "$model_name" == "qwen3.5-35b-a3b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3.5-35B-A3B-FP8
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        # --presence_penalty 1.5
        --enable_thinking False
        # --max_num_batched_tokens 8192
        --max_num_batched_tokens 16384
        # --max_num_seqs 512
        # --kv_cache_dtype fp8
        # --speculative_config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
        # --speculative_config '{"method": "suffix", "num_speculative_tokens": 32}'
    )
    suffix=qwen3.5_35b_a3b_fp8
elif [ "$model_name" == "qwen3.5-122b-a10b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/Qwen/Qwen3.5-122B-A10B-FP8
        --temperature 0.7
        --top_p 0.8
        --top_k 20
        --min_p 0.0
        # --presence_penalty 1.5
        --enable_thinking False
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=qwen3.5_122b_a10b_fp8
# gpt-oss
elif [ "$model_name" == "gpt-oss-20b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/openai/gpt-oss-20b
        --reasoning_effort low
        # --reasoning_effort medium
        --max_num_batched_tokens 8192
        # --gpu_memory_utilization 0.85
    )
    suffix=gpt_oss_20b_low
    # suffix=gpt_oss_20b_medium
elif [ "$model_name" == "gpt-oss-120b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/openai/gpt-oss-120b
        --reasoning_effort low
        # --reasoning_effort medium
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=gpt_oss_120b_low
    # suffix=gpt_oss_120b_medium
# nemotron
elif [ "$model_name" == "nemotron-3-nano-30b-a3b" ]; then
    # export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_FLASHINFER_MOE_FP8=1
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
        # --max_num_seqs 8
        # --tensor_parallel_size 1
        # --max_model_len 262144
        # --trust_remote_code
        # --enable_auto_tool_choice
        # --tool_call_parser qwen3_coder
        # --reasoning_parser_plugin /lustre/fswork/projects/rech/ilr/commun/pretrained/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/nano_v3_reasoning_parser.py
        # --reasoning_parser nano_v3
        # --enable_reasoning True
        --kv_cache_dtype fp8
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=nemotron_3_nano_30b_a3b_fp8
# mistral
elif [ "$model_name" == "mistral-small-3.2-24b-instruct" ]; then
    # export VLLM_WORKER_MULTIPROC_METHOD=spawn
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/mistralai/Mistral-Small-3.2-24B-Instruct-2506
        --tokenizer_mode mistral
        --config_format mistral
        --load_format mistral
        # --enable-auto-tool-choice
        # --tool-call-parser mistral
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=mistral_small_3_2_24b_instruct
elif [ "$model_name" == "ministral-3-14b-instruct" ]; then
    # export VLLM_WORKER_MULTIPROC_METHOD=spawn
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/mistralai/Ministral-3-14B-Instruct-2512
        --tokenizer_mode mistral
        --config_format mistral
        --load_format mistral
        # --enable-auto-tool-choice
        # --tool-call-parser mistral
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=ministral_3_14b_instruct
elif [ "$model_name" == "ministral-3-14b-reasoning" ]; then
    # export VLLM_WORKER_MULTIPROC_METHOD=spawn
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/mistralai/Ministral-3-14B-Reasoning-2512
        --tokenizer_mode mistral
        --config_format mistral
        --load_format mistral
        # --enable_reasoning True
        --reasoning_parser mistral
        # --enable-auto-tool-choice
        # --tool-call-parser mistral
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=ministral_3_14b_reasoning
# intellect
elif [ "$model_name" == "intellect-3" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/PrimeIntellect/INTELLECT-3-FP8
        # --tool_call_parser qwen3_coder
        --reasoning_parser deepseek_r1
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=intellect_3_fp8
# minimax
elif [ "$model_name" == "minimax-m2.1" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/MiniMaxAI/MiniMax-M2.1
        --temperature 1.0
        --top_p 0.95
        --top_k 40
        # --enable_auto_tool_choice
        # --tool_call_parser minimax_m2
        --reasoning_parser minimax_m2_append_think
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=minimax_m2.1
# glm
elif [ "$model_name" == "glm-4.7-flash" ]; then
    # export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/zai-org/GLM-4.7-Flash
        # --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/marksverdhei/GLM-4.7-Flash-FP8
        # --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/GadflyII/GLM-4.7-Flash-NVFP4
        --temperature 1.0
        --top_p 0.95
        --speculative_config '{"method": "mtp", "num_speculative_tokens": 1}'
        # --enable_auto_tool_choice
        # --tool_call_parser glm47
        --reasoning_parser glm45
        # --served_model_name glm-4.7-fp8
        --enable_thinking False
        # --max_num_seqs 255
        # --max_num_batched_tokens 8192
        # --disable_custom_all_reduce True
        # --enforce_eager True
        # --gpu_memory_utilization 0.85
    )
    suffix=glm_4.7_flash
# baichuan-m
elif [ "$model_name" == "baichuan-m3-235b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/baichuan-inc/Baichuan-M3-235B-GPTQ-INT4
        --reasoning_parser deepseek_r1
        --enable_thinking False
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    suffix=baichuan_m3_235b_gptq_int4
else
    echo "Invalid model name: $model_name"
    exit 1
fi

# v3 sampling-based rewriting
## system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt.txt
## system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt_v2.txt
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt_v3.txt
# output_path=${dataset_path}_rewritten_${suffix}

# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $output_path \
#     --system_prompt_file $system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \

    # --max_samples 10000 \
    # --start_idx 400000 \


# v3.1 style rewriting (cleaned style dims + view dim)
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt_v3_1.txt
# output_path=${dataset_path}_rewritten_v3_1_${suffix}

# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $output_path \
#     --system_prompt_file $system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --rewriting_version v3_1 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \


# v3.2 style rewriting (v3.1 + verbosity dim to fix over-compression)
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt_v3_2.txt
# output_path=${dataset_path}_rewritten_v3_2_${suffix}

# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $output_path \
#     --system_prompt_file $system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --rewriting_version v3_2 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \


# v4 MGA style rewriting prompts
# stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt.txt
# stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt.txt
# n_stage1_pairs=1
# stage1_output_path=${dataset_path}_rewritten_mga_stage1_${n_stage1_pairs}p_${suffix}
# stage2_output_path=${dataset_path}_rewritten_mga_stage2_${n_stage1_pairs}p_${suffix}

# Stage 1: generate genre/audience pairs
# echo "=== STAGE 1: Generating genre/audience pairs ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $stage1_output_path \
#     --system_prompt_file $stage1_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 1024 \
#     --rewriting_version v4 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \
#     --stage 1 \
#     --n_stage1_pairs $n_stage1_pairs \

# echo "=== STAGE 1 COMPLETE ==="

# Stage 2: reformulate per genre/audience pair
# echo "=== STAGE 2: Reformulating ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $stage1_output_path \
#     --output_path $stage2_output_path \
#     --system_prompt_file $stage2_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --rewriting_version v4 \
#     --language French \
#     --batch_size 10000 \
#     --stage 2 \


# v4.1 MGA rewriting (constrained closed-list genre/audience + style overlay)
# stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt_v4_1.txt
# stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt_v4_1.txt
# n_stage1_pairs=1
# stage1_output_path=${dataset_path}_rewritten_mga_stage1_v4_1_${n_stage1_pairs}p_${suffix}
# stage2_output_path=${dataset_path}_rewritten_mga_stage2_v4_1_${n_stage1_pairs}p_${suffix}

# # Stage 1 v4.1: generate genre/audience pairs from closed lists
# echo "=== STAGE 1 v4.1: Generating genre/audience pairs (closed list) ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $stage1_output_path \
#     --system_prompt_file $stage1_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 1024 \
#     --rewriting_version v4_1 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \
#     --stage 1 \
#     --n_stage1_pairs $n_stage1_pairs \

# echo "=== STAGE 1 v4.1 COMPLETE ==="

# # Stage 2 v4.1: density-focused rewrite with genre/audience + register/abbreviation overlay
# echo "=== STAGE 2 v4.1: Reformulating ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $stage1_output_path \
#     --output_path $stage2_output_path \
#     --system_prompt_file $stage2_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --language French \
#     --batch_size 10000 \
#     --stage 2 \
#     --rewriting_version v4_1 \


# v4.1+ MGA rewriting (V4.1 closed list + 3 high-impact prompt fixes only — same architecture as V4.1).
# Fixes vs V4.1: (a) realizability reframed as content-sufficiency, (b) `natural` marked last-resort,
# (c) corpus-level anti-bias instruction. Stage 2 unchanged from V4.1.
# Output suffix uses `v4_1_plus` to keep V4.1's existing dataset distinct.
# stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt_v4_1_plus.txt
# stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt_v4_1.txt
# n_stage1_pairs=1
# stage1_output_path=${dataset_path}_rewritten_mga_stage1_v4_1_plus_${n_stage1_pairs}p_${suffix}
# stage2_output_path=${dataset_path}_rewritten_mga_stage2_v4_1_plus_${n_stage1_pairs}p_${suffix}

# # Stage 1 v4.1+: closed-list with 3 prompt fixes (realizability reframe, last-resort natural, anti-bias)
# echo "=== STAGE 1 v4.1+: Generating genre/audience pairs (closed list, fixed prompt) ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $stage1_output_path \
#     --system_prompt_file $stage1_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 1024 \
#     --rewriting_version v4_1 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \
#     --stage 1 \
#     --n_stage1_pairs $n_stage1_pairs \

# echo "=== STAGE 1 v4.1+ COMPLETE ==="

# # Stage 2 v4.1+: same as V4.1 (density-focused + register/abbreviation overlay)
# echo "=== STAGE 2 v4.1+: Reformulating ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $stage1_output_path \
#     --output_path $stage2_output_path \
#     --system_prompt_file $stage2_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --language French \
#     --batch_size 10000 \
#     --stage 2 \
#     --rewriting_version v4_1 \


# v4.2 MGA rewriting (open genre/audience + 3-step reasoning structure + uniform per-doc sampling).
# Stage 1 prompt forces: enumerate genres → enumerate audiences → couple n_stage1_pairs=6 with per-dim breadth.
# Stage 2 samples n_stage2_pairs pairs per doc (without replacement) from stage 1 proposals.
# n_stage2_pairs=1 → corpus same size as input (max distribution control).
# n_stage2_pairs=2..6 → corpus grows by k× (more rewrites per source, more compute).
# Stage 2 prompt is reused from V4.1 (density-focused + register/abbreviation overlay).
stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt_v4_2.txt
stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt_v4_1.txt
n_stage1_pairs=3
n_stage2_pairs=1
stage1_output_path=${dataset_path}_rewritten_mga_stage1_v4_2_${n_stage1_pairs}m${n_stage2_pairs}n_${suffix}
stage2_output_path=${dataset_path}_rewritten_mga_stage2_v4_2_${n_stage1_pairs}m${n_stage2_pairs}n_${suffix}

# # Stage 1 v4.2: open generation with 3-step reasoning, n_stage1_pairs=6, per-dim diversity rules
# echo "=== STAGE 1 v4.2: Generating diverse genre/audience pairs ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $stage1_output_path \
#     --system_prompt_file $stage1_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 3072 \
#     --rewriting_version v4_2 \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \
#     --stage 1 \
#     --n_stage1_pairs $n_stage1_pairs \

echo "=== STAGE 1 v4.2 COMPLETE ==="

# Stage 2 v4.2: sample n_stage2_pairs pairs per doc + V4.1-style density-focused rewrite
echo "=== STAGE 2 v4.2: Reformulating (n_stage2_pairs pairs per doc, without replacement) ==="
srun python llm_rewrite.py \
    "${model_args[@]}" \
    --dataset_path $stage1_output_path \
    --output_path $stage2_output_path \
    --system_prompt_file $stage2_system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name text \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_model_len 32768 \
    --max_tokens 16384 \
    --language French \
    --batch_size 10000 \
    --stage 2 \
    --rewriting_version v4_2 \
    --n_stage2_pairs $n_stage2_pairs \


# # mga_official — Official MGA (paper-exact stage 1 from arXiv:2502.04235 Appendix E.2).
# # Stage 1 prompt is the verbatim Prompt 3 (5 pairs hardcoded). Output uses paper's flat-key
# # JSON schema (audience_1, genre_1, ..., audience_5, genre_5), via make_mga_stage1_official_schema.
# # Stage 2 prompt is the verbatim SLM-Base reformulation (Prompt 2) with English→{language}.
# # Stage 2 SAMPLES 1 pair of the 5 per doc (n_stage2_pairs=1) instead of exploding all 5
# # — saves 5× stage-2 compute, matches our matched-scale comparison setup against V4.2 dedup.
# stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt_official.txt
# stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt_official.txt
# n_stage1_pairs=5
# n_stage2_pairs=1
# stage1_output_path=${dataset_path}_rewritten_mga_stage1_mga_official_${n_stage1_pairs}m${n_stage2_pairs}n_${suffix}
# stage2_output_path=${dataset_path}_rewritten_mga_stage2_mga_official_${n_stage1_pairs}m${n_stage2_pairs}n_${suffix}

# # Stage 1 mga_official: official MGA — sentence-level genre/audience descriptions, n=5 (paper-exact)
# echo "=== STAGE 1 mga_official (official MGA): Generating 5 sentence-level genre/audience pairs ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $dataset_path \
#     --output_path $stage1_output_path \
#     --system_prompt_file $stage1_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 4096 \
#     --rewriting_version mga_official \
#     --language French \
#     --batch_size 10000 \
#     --shuffle True \
#     --max_samples 100000 \
#     --stage 1 \
#     --n_stage1_pairs $n_stage1_pairs \

# echo "=== STAGE 1 mga_official COMPLETE ==="

# # Stage 2 mga_official: sample 1 of 5 pairs per doc + official MGA SLM-Base reformulation
# echo "=== STAGE 2 mga_official (official MGA): Reformulating (sampled 1 of 5 pairs per doc) ==="
# srun python llm_rewrite.py \
#     "${model_args[@]}" \
#     --dataset_path $stage1_output_path \
#     --output_path $stage2_output_path \
#     --system_prompt_file $stage2_system_prompt_file \
#     --user_prompt_file $user_prompt_file \
#     --text_column_name text \
#     --output_column_name output \
#     --tensor_parallel_size 4 \
#     --max_model_len 32768 \
#     --max_tokens 16384 \
#     --language French \
#     --batch_size 10000 \
#     --stage 2 \
#     --rewriting_version mga_official \
#     --n_stage2_pairs $n_stage2_pairs \


echo "END TIME: $(date)"
