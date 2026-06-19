#!/bin/bash

#SBATCH --job-name=llm_rewrite_eval
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev        # QoS
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
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

model_name=intellect-3
# model_name=qwen3-235b-a22b

system_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting_eval/system_prompt.txt
user_prompt_file=/lustre/fswork/projects/rech/ilr/commun/doctobert/data_curation/data_synthesis/prompts/rewriting_eval/user_prompt.txt

dataset_path=$1

# medgemma
if [ "$model_name" == "medgemma-27b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/google/medgemma-27b-text-it
        --max_num_batched_tokens 8192
    )
    suffix=medgemma_27b
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
# gpt-oss
elif [ "$model_name" == "gpt-oss-20b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/openai/gpt-oss-20b
        # --reasoning_effort low
        --reasoning_effort medium
        --max_num_batched_tokens 8192
        # --gpu_memory_utilization 0.85
    )
    # suffix=gpt_oss_20b_low
    suffix=gpt_oss_20b_medium
elif [ "$model_name" == "gpt-oss-120b" ]; then
    model_args=(
        --model_name_or_path /lustre/fswork/projects/rech/ilr/commun/pretrained/models/openai/gpt-oss-120b
        # --reasoning_effort low
        --reasoning_effort medium
        --max_num_batched_tokens 8192
        --gpu_memory_utilization 0.85
    )
    # suffix=gpt_oss_120b_low
    suffix=gpt_oss_120b_medium
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

output_dir=${dataset_path}_eval_${suffix}

# rewriting
srun python llm_rewrite.py \
    "${model_args[@]}" \
    --dataset_path $dataset_path \
    --output_path $output_dir \
    --system_prompt_file $system_prompt_file \
    --user_prompt_file $user_prompt_file \
    --text_column_name original_text \
    --output_column_name output \
    --tensor_parallel_size 4 \
    --max_model_len 32768 \
    --max_tokens 16384 \
    --json_schema rewriting_eval \
    --batch_size 10000 \
    --shuffle True \


    # --max_samples 10000 \
    # --start_idx 400000 \


echo "END TIME: $(date)"
