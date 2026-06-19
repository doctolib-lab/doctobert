"""Use a local vllm engine to generate data."""

import os
import random
import time
import zipfile
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# tmp control
gen_name = os.environ.get("GEN_NAME")


def file_generator(zip_path, encoding="utf-8"):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".txt"):
                with zf.open(name) as f:
                    yield {"text": f.read().decode(encoding)}


def load_prompt(p: Path | str) -> str:
    """Load prompt from local file."""
    p = str(p) if isinstance(p, Path) else p
    with open(p, encoding="utf-8") as f:
        return f.read()


def load_local_dataset(input_path: str) -> Dataset:
    """Load Hugging Face dataset from local file."""
    if input_path.endswith(".txt"):
        return load_dataset("text", data_files=input_path, split="train")
    elif input_path.endswith(".jsonl") or input_path.endswith(".json"):
        return load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith(".csv") or input_path.endswith(".csv.gz"):
        return load_dataset("csv", data_files=input_path, split="train")
    elif input_path.endswith(".parquet"):
        return load_dataset("parquet", data_files=input_path, split="train")
    elif os.path.isdir(input_path):
        return load_dataset(input_path, split="train")
    elif input_path.endswith(".zip"):
        return Dataset.from_generator(lambda: file_generator(input_path))
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


def load_local_datasets(input_paths: str, sep: str = "+") -> Dataset:
    """Load Hugging Face datasets from local files."""
    datasets = []
    for input_path in input_paths.split(sep):
        ds = load_local_dataset(input_path)
        datasets.append(ds)
    return concatenate_datasets(datasets)


def process_batch(
    batch: list[dict],
    llm: LLM,
    params: SamplingParams,
    system_prompt: str | None = None,
    system_prompt_suffixes: dict[str, str] | None = None,
    user_prompt: str | None = None,
    text_column_name: str = "text",
    output_column_name: str = "output",
    enable_thinking: bool = False,
    max_model_len: int | None = None,
) -> list[dict]:
    """Process a batch of data using local vllm engine."""

    tokenizer = llm.get_tokenizer()
    max_text_tokens = max_model_len // 2 if max_model_len is not None else None

    prompts = []
    for item in batch:
        # Truncate text using actual tokenization
        text = item.get(text_column_name, "")
        if max_text_tokens is not None and text:
            text_token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(text_token_ids) > max_text_tokens:
                text_token_ids = text_token_ids[:max_text_tokens]
                text = tokenizer.decode(text_token_ids, skip_special_tokens=False)

        # Build the user message
        if user_prompt:
            if gen_name in ["clinical_case", "ehr", "dialogue"]:
                user_prompt_text = user_prompt.format(text=text)
                if gen_name == "dialogue":
                    # backup clinical case text (use truncated text)
                    item["clinical_note"] = text
            elif gen_name == "icd":
                labels = item.get("labels", {})
                ctx = {
                    "definition_fr": item.get("definition_fr", ""),
                    "label_fr": labels.get("fr", item.get("label_fr", "")),
                    "skos_notation": item.get("skos_notation", ""),
                }
                user_prompt_text = user_prompt.format(**ctx)
            elif gen_name == "vocabulary":
                ctx = {
                    "term": item.get("term", ""),
                    "definition": item.get("definition", ""),
                    "synonyms": item.get("synonyms", ""),
                    "grammatical_category": item.get("grammatical_category", ""),
                }
                user_prompt_text = user_prompt.format(**ctx)
            else:
                raise ValueError(f"Unsupported generation name: {gen_name}")

        messages = [{"role": "user", "content": user_prompt_text}]

        # Add system message
        gen_style = None
        if system_prompt:
            if gen_name == "clinical_case":
                gen_style = "report" if random.random() < 0.2 else "note"
            elif gen_name in ["icd", "vocabulary"]:
                gen_style = "wiki" if random.random() < 0.2 else "textbook"
            elif gen_name in ["ehr", "dialogue"]:
                # strict system prompt
                pass
            else:
                raise ValueError(f"Unsupported generation name: {gen_name}")

            system_prompt_suffix = system_prompt_suffixes.get(gen_style, "") if system_prompt_suffixes and gen_style else ""
            system_msg = system_prompt + system_prompt_suffix
            messages.insert(0, {"role": "system", "content": system_msg})
        item["gen_style"] = gen_style

        # apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        # this truncation cut off chat template suffix, which is not what we want
        # def _format_and_truncate(llm, messages, max_model_len=None, enable_thinking=False):
        #     tokenizer = llm.get_tokenizer()
        #     # first build ids via the chat template
        #     token_ids = tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=True,
        #         add_generation_prompt=True,
        #         enable_thinking=enable_thinking,
        #     )
        #     if max_model_len is not None and len(token_ids) > max_model_len:
        #         # todo: keep head or throw
        #         token_ids = token_ids[:max_model_len // 2]
        #     # re-decode to a string prompt that vLLM will re-tokenize (same template)
        #     return tokenizer.decode(token_ids, skip_special_tokens=False)

        # formatted_prompt = _format_and_truncate(llm, messages, max_model_len, enable_thinking)
            
        prompts.append(formatted_prompt)

    # todo: llm.chat
    outputs = llm.generate(prompts, params)

    if len(outputs) != len(batch):
        raise RuntimeError(f"Number of outputs ({len(outputs)}) does not match batch size ({len(batch)}).")

    for item, out, prompt in zip(batch, outputs, prompts):
        item[output_column_name] = out.outputs[0].text.strip()
        item["gen_configs"] = {
            "model": llm.llm_engine.model_config.model,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "max_tokens": params.max_tokens,
            "gen_style": item.pop("gen_style", None),
            "prompt": prompt,  # debug
        }

    return batch


# Generate outputs, update dataset in batches, and overwrite checkpoint
def process_dataset(
    dataset: Dataset,
    llm: LLM,
    params: SamplingParams,
    output_dir: str,
    batch_size: int = 512,
    system_prompt: str | None = None,
    system_prompt_suffixes: dict[str, str] | None = None,
    user_prompt: str | None = None,
    text_column_name: str = "text",
    output_column_name: str = "output",
    enable_thinking: bool = False,
    max_model_len: int | None = None,
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
            system_prompt_suffixes=system_prompt_suffixes,
            user_prompt=user_prompt,
            text_column_name=text_column_name,
            output_column_name=output_column_name,
            enable_thinking=enable_thinking,
            max_model_len=max_model_len,
        )

        shard_ds = Dataset.from_list(processed_records)
        # tmp
        if output_dir.endswith(".parquet"):
            shard_ds.to_parquet(f"{output_dir[:-8]}_{i:09d}.parquet")
        else:
            shard_ds.to_parquet(f"{output_dir}/{i:09d}.parquet")
        # shard_ds.to_parquet(f"{output_dir}/{i:09d}.parquet")


# Main function to control workflow
def main(
    dataset_path: str,
    model_name_or_path: str,
    # i/o
    output_dir: str,
    system_prompt_file: str | None = None,
    user_prompt_file: str | None = None,
    text_column_name: str = "text",
    output_column_name: str = "output",
    # load params
    # dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float = 0.95,
    speculative_config: dict | None = None,
    # infer params
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    # repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
    # json_schema: str | None = None,
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
    dataset = load_local_datasets(dataset_path)
    print(f"Loaded {dataset.num_rows:,d} examples")

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

    # tmp
    # if gen_name == "ehr":
    #     dataset = dataset.map(lambda x: {"num_words": len(x[text_column_name].split())}, num_proc=32)
    #     dataset = dataset.filter(lambda x: x["num_words"] < 25_000, num_proc=32)
    #     print(f"Filtered to {dataset.num_rows:,d} examples")

    # load llm
    print("Start Local vllm engine...")
    llm_kwargs = {
        "trust_remote_code": True,
        # "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        print(f"Using max model len: {max_model_len}")
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
        print(f"Using max num seqs: {max_num_seqs}")
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
        print(f"Using speculative config: {speculative_config}")
    llm = LLM(model=model_name_or_path, **llm_kwargs)

    # gen params
    # guided_decoding_params_json = None
    # if json_schema is not None:
    #     output_json_schema = OUTPUT_SCHEMAS.get(json_schema)
    #     guided_decoding_params_json = GuidedDecodingParams(json=output_json_schema)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        # repetition_penalty=repetition_penalty,
        # stop_token_ids=stop_token_ids,
        # guided_decoding=guided_decoding_params_json,
    )

    # load prompts
    # system prompt
    system_prompt = load_prompt(system_prompt_file) if system_prompt_file else None
    # tmp: extra system prompt
    system_prompt_suffixes = {}
    if system_prompt_file:
        system_prompt_file_p = Path(system_prompt_file)
        for p in system_prompt_file_p.parent.glob(system_prompt_file_p.stem + "_style_*.txt"):
            suffix_name = p.stem.split("_style_")[-1]
            system_prompt_suffixes[suffix_name] = load_prompt(p)
    # user prompt
    user_prompt = load_prompt(user_prompt_file) if user_prompt_file else None

    start_time = time.perf_counter()

    process_dataset(
        dataset,
        llm,
        params,
        output_dir,
        batch_size=batch_size,
        system_prompt=system_prompt,
        system_prompt_suffixes=system_prompt_suffixes,
        user_prompt=user_prompt,
        text_column_name=text_column_name,
        output_column_name=output_column_name,
        enable_thinking=enable_thinking,
        max_model_len=max_model_len,
    )

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}.\n"
        f"Generated data is saved in {output_dir}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
