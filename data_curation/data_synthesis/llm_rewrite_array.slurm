#!/bin/bash

#SBATCH --job-name=llm_rewrite_array
#SBATCH --output=logs/%x/%A_%a.out    # %A = array job ID, %a = array task ID
#SBATCH --error=logs/%x/%A_%a.err     # %A = array job ID, %a = array task ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4        # QoS
#SBATCH --time=100:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=ilr@h100           # account to use, with xyz the 3 letter code of your project
#SBATCH --array=0-999%99             # Default array spec (overridden by sbatch --array)

set -euo pipefail

echo "START TIME: $(date)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

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

# Per-task isolated torch.inductor / triton caches — avoid race condition when many array tasks
# (e.g., 69 concurrent × 4 worker ranks = ~276 processes) write to the same /tmp/.../torchinductor_${USER}/.
# Symptom in shared cache: FileNotFoundError on .tmp files → vLLM EngineCore dies during model compile.
# See logs/llm_rewrite_array/694246_*.err for the failure pattern that motivated this isolation.
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TRITON_CACHE_DIR="${TMPDIR}/triton_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Raise vLLM internal RPC timeout for long-running stage-2 generation.
# Default ~30-60s on `sample_tokens` is too tight for batch=10000 with V4.2 stage-2 long outputs.
# History:
#   - 600000ms (10 min): still hit `sample_tokens` timeout on 739032_{4,5,13} and 734199_19
#   - 1800000ms (30 min): bumped after the second wave of failures
export VLLM_RPC_TIMEOUT=1800000

# OLD V3 single-stage rewriting prompts (kept for reference):
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt.txt
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt_v2.txt
# system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/system_prompt_v3.txt
# OLD original-MGA prompts (kept for reference):
# stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt.txt
# stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt.txt

# V4.2 prompt files (open-list 3-step reasoning stage-1 + density-focused stage-2)
stage1_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage1_system_prompt_v4_2.txt
stage2_system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/mga_stage2_system_prompt_v4_1.txt
# user prompt (shared)
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting/user_prompt.txt

# V4.2 stage params (3 stage-1 pairs sampled to 1 → 1 rewrite per source doc)
n_stage1_pairs=3
n_stage2_pairs=1

input_dir=$1
# model_name=${2:-qwen3-30b-a3b}

# Get list of all parquet files and select the one for this array task
mapfile -t files < <(find "$input_dir" -name "*.parquet" | sort)
total_files=${#files[@]}
echo "Total files found: $total_files"

# Check if this array task has a corresponding file
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_files" ]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds number of files ($total_files). Exiting."
    exit 0
fi

# Get the file for this array task
dataset_path=${files[$SLURM_ARRAY_TASK_ID]}

# OLD 3-way model split (medgemma / gpt-oss-120b / qwen3-next — kept for reference):
# third=$((total_files / 3))
# two_thirds=$((2 * total_files / 3))
# if [ "$SLURM_ARRAY_TASK_ID" -lt "$third" ]; then
#     model_name="medgemma-27b"
# elif [ "$SLURM_ARRAY_TASK_ID" -lt "$two_thirds" ]; then
#     model_name="gpt-oss-120b"
# else
#     model_name="qwen3-next-80b-a3b"
# fi

# OLD 50/50 split (Qwen3.5-35B / Gemma-4-26B — kept for reference):
# half=$((total_files / 2))
# if [ "$SLURM_ARRAY_TASK_ID" -lt "$half" ]; then
#     model_name="qwen3.5-35b-a3b"
# else
#     model_name="gemma-4-26b-a4b"
# fi

# NEW: use Qwen3.5-35B for ALL parquets (single-rewriter at full corpus scale)
model_name="qwen3.5-35b-a3b"

echo "Array task $SLURM_ARRAY_TASK_ID / $total_files -> model $model_name"

# OLD output path naming (single-stage):
# output_dir="${input_dir}_rewritten"
# output_path="${output_dir}/${dataset_path##*/}"

echo "Processing file: $dataset_path"
echo "Using model: $model_name"

# Check if output already exists (to avoid reprocessing)
# if [ -f "$output_path" ]; then
#     echo "Output file already exists. Skipping: $output_path"
#     exit 0
# fi

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
        # OLD: --max_num_batched_tokens 8192
        --max_num_batched_tokens 16384  # 2× the default — absorbs prefill bursts on long V4.2 prompts (verified ~2× speedup at 1M-doc scale)
        # NOTE: do NOT add `kv_cache_dtype=fp8` / `max_num_seqs=512` / `speculative_config=suffix-32` —
        # they slow Qwen 35B on V4.2 rewriting workload (free-form paraphrase has near-zero suffix overlap).
        --enforce_eager True  # disable CUDA graph capture; primary mitigation for shm_broadcast / sample_tokens timeout hangs (vLLM issues #36921, #30682, #41530). ~10-15% throughput cost, eliminates most multi-hour crashes.
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

# OLD V3 single-stage rewriting (kept for reference):
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
#     --json_schema rewriting \
#     --language French \
#     --batch_size 50000 \
#     --shuffle True \

# NEW V4.2 two-stage rewriting (3m1n: 3 stage-1 pairs sampled to 1 → 1 rewrite per source doc).
# Output dirs are per-model (suffix differs); each parquet shard's rewritten output preserves the source filename.
# Note: `llm_rewrite.py` applies the cheap prefilter (density>=0.01 AND edu>=1) before stage-1 LLM calls — saves ~23% LLM compute.
stage1_output_dir="${input_dir}_rewritten_mga_stage1_v4_2_${n_stage1_pairs}m${n_stage2_pairs}n_${suffix}"
stage2_output_dir="${input_dir}_rewritten_mga_stage2_v4_2_${n_stage1_pairs}m${n_stage2_pairs}n_${suffix}"
parquet_name="${dataset_path##*/}"
parquet_stem="${parquet_name%.parquet}"

# OLD: stage paths ended in .parquet → process_dataset stripped ".parquet" and wrote BATCH FILES
# as siblings in stage1_dir (e.g., 00000_000000000.parquet, ...). Stage 2 then couldn't find a single
# input file matching ${parquet_name}. Resulting in FileNotFoundError when loading stage1 output.
# stage1_output_path="${stage1_output_dir}/${parquet_name}"
# stage2_output_path="${stage2_output_dir}/${parquet_name}"

# NEW: stage paths are per-shard SUB-DIRECTORIES (no .parquet ext) → process_dataset writes
# batch files into ${output_path}/{i:09d}.parquet. Stage 2 then loads the directory.
stage1_output_path="${stage1_output_dir}/${parquet_stem}"
stage2_output_path="${stage2_output_dir}/${parquet_stem}"

echo "Stage 1 output: ${stage1_output_path}"
echo "Stage 2 output: ${stage2_output_path}"

# Stage 1 — skip if a .STAGE1_DONE marker exists for this shard (set after a successful previous run).
# This saves the expensive stage-1 work when re-running due to a stage-2 crash.
# For shards where stage 1 actually completed in a prior run but no marker was written,
# touch ${stage1_output_path}/.STAGE1_DONE manually before re-submitting (see header docs).
stage1_marker="${stage1_output_path}/.STAGE1_DONE"

if [ -f "$stage1_marker" ]; then
    echo "=== STAGE 1 v4.2 SKIPPED (marker found: $stage1_marker) ==="
else
    # Partial stage-1 output from a prior failed run is OK to leave: process_dataset writes
    # {i:09d}.parquet for i=0..N-1, so new run overwrites old files of the same name and
    # produces ≥ files than the old (same batch_size). No orphans, no rm needed.
    # CAVEAT: if you change `--batch_size` between runs, manually rm the dir first.
    echo "=== STAGE 1 v4.2 (task ${SLURM_ARRAY_TASK_ID}, ${model_name}) ==="
    srun python llm_rewrite.py \
        "${model_args[@]}" \
        --dataset_path $dataset_path \
        --output_path $stage1_output_path \
        --system_prompt_file $stage1_system_prompt_file \
        --user_prompt_file $user_prompt_file \
        --text_column_name text \
        --output_column_name output \
        --tensor_parallel_size 4 \
        --max_model_len 32768 \
        --max_tokens 3072 \
        --rewriting_version v4_2 \
        --language French \
        --batch_size 10000 \
        --stage 1 \
        --n_stage1_pairs $n_stage1_pairs \

    touch "$stage1_marker"
    echo "=== STAGE 1 v4.2 COMPLETE (marker written: $stage1_marker) ==="
fi

# Stage 2: partial output from a prior crash is OK — same overwrite-by-filename guarantee as stage 1.

# Stage 2: sample 1 of 3 pairs per doc + density-focused reformulation (drops has_medical_content=False rows)
echo "=== STAGE 2 v4.2 (task ${SLURM_ARRAY_TASK_ID}, ${model_name}) ==="
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

echo "=== STAGE 2 v4.2 COMPLETE ==="
echo "END TIME: $(date)"
