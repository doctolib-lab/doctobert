"""Use a local vllm engine to annotate a dataset with structured output."""

import glob
import os
import random
import re
import time
from typing import Literal

from datasets import Dataset, load_dataset, load_from_disk
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


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

class MedicalEntity(BaseModel):
    # high medical term density subset of the 15 UMLS entity groups
    entity: Literal["disorders", "chemicals_drugs", "anatomy", "procedures", "physiology", "devices", "genes_molecular_sequences"]
    text: str

class MedicalEntitiesExtractionOutput(BaseModel):
    entities: list[MedicalEntity]

class MedicalEntityV2(BaseModel):
    # high medical term density subset of the 15 UMLS entity groups
    entity: Literal["disorders", "chemicals_drugs", "anatomy", "procedures", "physiology", "devices", "genes_molecular_sequences"]
    text: list[str]

class MedicalEntitiesExtractionOutputV2(BaseModel):
    entities: list[MedicalEntityV2]

class MedicalEntityV3(BaseModel):
    # simplified entity groups for medical term density calculation
    entity: Literal["disease", "drug", "body_part", "medical_procedure", "molecular_marker", "clinical_device", "vital_function", "living_beings"]
    text: list[str]

class MedicalEntitiesExtractionOutputV3(BaseModel):
    entities: list[MedicalEntityV3]


class MedicalEntitiesReviewReasoning(BaseModel):
    false_positives: str
    reclassifications: str
    missed_entities: str

class MedicalEntitiesReviewOutput(BaseModel):
    reasoning: MedicalEntitiesReviewReasoning
    entities: list[MedicalEntityV3]


OUTPUT_SCHEMAS = {
    "domain_classification": DomainClassificationOutput.model_json_schema(),
    "quality_classification": QualityClassificationOutput.model_json_schema(),
    "quality_classification_v2": QualityClassificationOutputV2.model_json_schema(),
    "medical_entities_extraction": MedicalEntitiesExtractionOutput.model_json_schema(),
    "medical_entities_extraction_v2": MedicalEntitiesExtractionOutputV2.model_json_schema(),
    "medical_entities_extraction_v3": MedicalEntitiesExtractionOutputV3.model_json_schema(),
    "medical_entities_review": MedicalEntitiesReviewOutput.model_json_schema(),
}


def shuffle_topics_in_prompt(
    prompt: str,
    start_tag: str = "<topics>",
    end_tag: str = "</topics>",
    sep: str = "\n",
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
    lines = body.strip(sep).split(sep)

    # If nothing to shuffle, return as-is
    if len(lines) <= 1:
        return prompt

    random.shuffle(lines)
    new_body = sep.join(lines)
    new_section = f"{prefix}{new_body}{suffix}"
    return re.sub(pattern, new_section, prompt, count=1)


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
        parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if parquet_files:
            # Treat directory of parquet files as a parquet dataset
            return load_dataset("parquet", data_files=parquet_files, split="train")
        dataset_info_path = os.path.join(input_path, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            # Load dataset saved with `Dataset.save_to_disk`
            return load_from_disk(input_path)
        raise ValueError(f"Directory has no parquet files or HF dataset metadata: {input_path}")
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


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
    max_words_per_sample: int | None = None,
) -> list[dict]:
    """Process a batch of data using local vllm engine."""

    prompts = []
    for item in batch:
        text = item[text_column_name]
        if max_words_per_sample is not None:
            text = " ".join(text.split()[:max_words_per_sample])
            item[f"{text_column_name}_truncated"] = text

        # format instruction
        # tmp: classification, initial ner
        # user_prompt_text = user_prompt.format(text=text, url=item.get(url_column_name, ""))
        # tmp: review ner
        user_prompt_text = user_prompt.format(text=text, initial_extraction=item["output"])
        item["output_old"] = item["output"]

        messages = [{"role": "user", "content": user_prompt_text}]

        # read system message from prompt file
        # not None or not empty
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

            # tmp: classification
            # shuffle labels to avoid LLM bias
            # randomized_system_prompt = shuffle_topics_in_prompt(system_prompt, start_tag="<topics>", end_tag="</topics>", sep="\n")
            # randomized_system_prompt = shuffle_topics_in_prompt(system_prompt, start_tag="<entity_groups>", end_tag="</entity_groups>", sep="\n\n")
            # messages.insert(0, {"role": "system", "content": randomized_system_prompt})

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
    max_words_per_sample: int | None = None,
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
            max_words_per_sample=max_words_per_sample,
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
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float = 0.95,
    reasoning_parser: str | None = None,
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
    max_words_per_sample: int | None = None,
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
    if reasoning_parser is not None:
        llm_kwargs["reasoning_parser"] = reasoning_parser
        print(f"Using reasoning parser: {reasoning_parser}")
    # if speculative_config is not None:
    #     llm_kwargs["speculative_config"] = speculative_config
    #     print(f"Using speculative config: {speculative_config}")
    llm = LLM(model=model_name_or_path, **llm_kwargs)

    # gen params
    structured_outputs = None
    if json_schema is not None:
        output_json_schema = OUTPUT_SCHEMAS.get(json_schema)
        structured_outputs = StructuredOutputsParams(json=output_json_schema)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        # repetition_penalty=repetition_penalty,
        # stop_token_ids=stop_token_ids,
        structured_outputs=structured_outputs,
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
        max_words_per_sample=max_words_per_sample,
    )

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}.\n"
        f"Generated data is saved in {output_dir}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
