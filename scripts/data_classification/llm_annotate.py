"""Use a local vllm engine to annotate a dataset with structured output."""

import os
import random
import re
import time

from datasets import Dataset, load_dataset
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class DomainClassificationOutput(BaseModel):
    reasoning: str
    topic: str


class QualityClassificationOutput(BaseModel):
    reasoning: str
    criterion_basic_health_info: bool
    criterion_some_useful_detail: bool
    criterion_general_accuracy_and_coherence: bool
    criterion_high_relevance_and_evidence_based: bool
    criterion_outstanding_authoritative: bool
    final_score: int

class QualityClassificationOutputV2(BaseModel):
    reasoning: str
    score: int


OUTPUT_SCHEMAS = {
    "domain_classification": DomainClassificationOutput.model_json_schema(),
    "quality_classification": QualityClassificationOutput.model_json_schema(),
    "quality_classification_v2": QualityClassificationOutputV2.model_json_schema(),
}


def shuffle_topics_in_prompt(
    prompt: str,
    start_tag: str = "<topics>",
    end_tag: str = "</topics>",
) -> str:
    """Shuffle the order of bullet labels inside the <topics>...</topics> section.

    Preserves the original formatting of each bullet line and only reorders them.
    If no <topics> section is found, returns the prompt unchanged.
    """
    pattern = rf"({re.escape(start_tag)}\s*)([\s\S]*?)(\s*{re.escape(end_tag)})"
    match = re.search(pattern, prompt)
    if not match:
        return prompt

    prefix, body, suffix = match.groups()

    # Split inner body into lines
    lines = body.strip("\n").splitlines()

    # If nothing to shuffle, return as-is
    if len(lines) <= 1:
        return prompt

    random.shuffle(lines)
    new_body = "\n".join(lines)
    new_section = f"{prefix}{new_body}{suffix}"
    return re.sub(pattern, new_section, prompt, count=1)


def load_local_dataset(input_path: str) -> Dataset:
    """Load Hugging Face dataset from local file."""
    if input_path.endswith(".txt"):
        ds = load_dataset("text", data_files=input_path, split="train")
    elif input_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=input_path, split="train")
    elif os.path.isdir(input_path):
        ds = load_dataset(input_path, split="train")
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")
    return ds


def process_batch(
    batch: list[dict],
    llm: LLM,
    params: SamplingParams,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    text_column_name: str = "text",
    url_column_name: str = "url",
    output_column_name: str = "output",
    enable_thinking: bool = False,
) -> list[dict]:
    """Process a batch of data using local vllm engine."""

    prompts = []
    for item in batch:
        # format instruction
        user_prompt_text = user_prompt.format(text=item[text_column_name], url=item[url_column_name])
        messages = [{"role": "user", "content": user_prompt_text}]

        # read system message from prompt file
        # not None or not empty
        if system_prompt:
            # shuffle labels to avoid LLM bias
            randomized_system_prompt = shuffle_topics_in_prompt(system_prompt, start_tag="<topics>", end_tag="</topics>")
            messages.insert(0, {"role": "system", "content": randomized_system_prompt})

        # apply chat template
        formatted_prompt = llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(formatted_prompt)

    # todo: llm.chat
    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        item[output_column_name] = outputs[i].outputs[0].text.strip()

    return batch


# Generate outputs, update dataset in batches, and overwrite checkpoint
def process_dataset(
    dataset: Dataset,
    llm: LLM,
    params: SamplingParams,
    output_dir: str,
    batch_size: int = 512,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    text_column_name: str = "text",
    url_column_name: str = "url",
    output_column_name: str = "output",
    enable_thinking: bool = False,
):
    """Process dataset in batches."""

    # Calculate total number of batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        # Slice dataset to avoid materializing entire dataset rows eagerly
        shard = dataset.select(range(start_idx, end_idx))
        batch_records = list(shard)

        processed_records = process_batch(
            batch_records,
            llm,
            params,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            text_column_name=text_column_name,
            url_column_name=url_column_name,
            output_column_name=output_column_name,
            enable_thinking=enable_thinking,
        )

        shard_ds = Dataset.from_list(processed_records)
        shard_ds.to_parquet(f"{output_dir}/{i:09d}.parquet")


# Main function to control workflow
def main(
    dataset_path: str,
    model_name_or_path: str,
    # i/o
    output_dir: str,
    system_prompt_file: str | None = None,
    user_prompt_file: str | None = None,
    text_column_name: str = "text",
    url_column_name: str = "url",
    output_column_name: str = "output",
    # load params
    # dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    # max_model_len: int = 2048,
    # max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.95,
    # infer params
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    # repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
    json_schema: str | None = None,
    # run params
    batch_size: int = 512,
    shuffle: bool = False,
    seed: int = 42,
    start_idx: int = 0,
    max_samples: int | None = None,
):
    # seed for reproducibility of topic shuffling
    random.seed(seed)

    # load dataset
    dataset = load_local_dataset(dataset_path)
    print(f"Loaded {dataset.num_rows:,d} examples from {dataset_path}")

    # shuffle dataset
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
        print("Shuffled dataset")

    if start_idx > 0:
        dataset = dataset.select(range(start_idx, len(dataset)))
        print(f"Skipped the first {start_idx:,d} examples")

    # take max samples
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # load llm
    print("Start Local vllm engine...")
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        # dtype=dtype,
        # max_model_len=max_model_len,  # limited by kv-cache
        # max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # gen params
    guided_decoding_params_json = None
    if json_schema is not None:
        output_json_schema = OUTPUT_SCHEMAS.get(json_schema)
        guided_decoding_params_json = GuidedDecodingParams(json=output_json_schema)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        # repetition_penalty=repetition_penalty,
        # stop_token_ids=stop_token_ids,
        guided_decoding=guided_decoding_params_json,
    )

    # load prompt
    system_prompt = None
    if system_prompt_file is not None:
        with open(system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read()
    user_prompt = None
    if user_prompt_file is not None:
        with open(user_prompt_file, encoding="utf-8") as f:
            user_prompt = f.read()

    start_time = time.perf_counter()

    process_dataset(
        dataset,
        llm,
        params,
        output_dir,
        batch_size=batch_size,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        text_column_name=text_column_name,
        url_column_name=url_column_name,
        output_column_name=output_column_name,
        enable_thinking=enable_thinking,
    )

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}.\n"
        f"Generated data is saved in {output_dir}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
