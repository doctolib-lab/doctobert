"""Classify a dataset using a pretrained model."""

import math
import os
from typing import Any, Dict, List, Literal

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """
    Lightweight classification head on top of a pretrained backbone.

    For nvidia/multilingual-domain-classifier.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config.get("fc_dropout", 0.1))
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features[:, 0, :])  # CLS token
        logits = self.fc(dropped)
        return torch.softmax(logits, dim=-1)


def collate_fn_factory(tokenizer, text_column: str, url_column: str, get_num_words: bool = False):
    """
    Return a collate function that tokenises a list of raw dataset examples.

    The collate function keeps the *raw* examples so the caller can merge the
    model predictions back into the original records later on.
    """

    def collate_fn(batch: List[Dict[str, Any]]):
        # Prepend url if it exists
        texts = []
        for ex in batch:
            text = ex.get(text_column)
            text = text if isinstance(text, str) else ""
            url = ex.get(url_column)
            if isinstance(url, str) and url:
                text = url + "\n\n" + text
            texts.append(text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",  # pad only to the longest seq inside *this* batch
            truncation=True,
        )

        # todo: tmp func
        if get_num_words:
            for ex in batch:
                ex["num_words"] = len(ex[text_column].split()) if isinstance(ex[text_column], str) else 0

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "raw_examples": batch,  # keep originals for later merge
        }

    return collate_fn


class BucketSortedIterableDataset(IterableDataset):
    """
    Yield examples in approx. descending length order within sliding windows.

    The class takes an HF streaming dataset (or any Python iterable of dicts)
    and, inside ``__iter__``, gathers ``bucket_size`` examples, sorts that
    mini-batch by sequence length, and then yields them one by one.

    This greatly reduces the amount of padding within every *real* batch that
    the DataLoader will assemble, while keeping the memory footprint small and
    preserving the streaming property (we never have more than
    ``bucket_size`` items in RAM).
    """

    def __init__(self, hf_iterable, text_column: str, bucket_size: int = 4096):
        super().__init__()
        self.data = hf_iterable
        self.text_column = text_column
        self.bucket_size = bucket_size

    def __iter__(self):
        cache = []
        for ex in self.data:
            cache.append(ex)
            if len(cache) == self.bucket_size:
                # sort shortest -> longest so each subsequent .pop() returns the longest
                cache.sort(key=lambda e: len(e[self.text_column].split()))
                while cache:
                    yield cache.pop()  # pop() is O(1) on list tail
        # Flush remainder
        cache.sort(key=lambda e: len(e[self.text_column].split()))
        while cache:
            yield cache.pop()


def load_local_dataset(input_path: str) -> Dataset:
    """Load Hugging Face dataset from local file."""
    if input_path.endswith(".txt"):
        ds = load_dataset("text", data_files=input_path, split="train")
    elif input_path.endswith(".jsonl") or input_path.endswith(".json"):
        ds = load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=input_path, split="train")
    elif os.path.isdir(input_path):
        ds = load_dataset(input_path, split="train")
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")
    return ds


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:05d}{ext}"


def main(
    output_path: str,
    task_name: str,
    # data
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "test",
    dataset_path: str | None = None,
    text_column: str = "text",
    url_column: str = "url",
    max_samples: int | None = None,
    # model
    model_name_or_path: str = "nvidia/multilingual-domain-classifier",
    task_type: Literal["classification", "regression"] = "classification",
    batch_size: int = 32,
    bucket_size: int | None = None,
    num_workers: int = 1,
    # save
    output_shard_size: int | None = None,  # 1_000_000
    # tmp
    get_num_words: bool = False,
):
    # Load dataset
    if dataset_name is not None:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
        )
    elif dataset_path is not None:
        # dataset = load_dataset(
        #     "parquet",
        #     data_files={"train": dataset_path},
        #     split="train",
        #     streaming=True,
        # )
        dataset = load_local_dataset(dataset_path)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided.")

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    if "nvidia/multilingual-domain-classifier" in model_name_or_path:
        model = CustomModel.from_pretrained(model_name_or_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # model = torch.compile(model)

    if bucket_size is not None:
        # Wrap the streaming dataset so that examples arrive roughly sorted by length.
        dataset = BucketSortedIterableDataset(dataset, text_column=text_column, bucket_size=bucket_size)
        num_workers = 1
        print("Setting num_workers to 1")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_factory(tokenizer, text_column, url_column, get_num_words=get_num_words),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    def _normalize_regression_score(score: float | int) -> int:
        """Round regression score to 0-5."""
        return int(round(max(0, min(score, 5))))

    # Run inference
    buffer = []
    shard_idx = 0  # index for output shards when saving multiple times

    for batch in tqdm(dataloader, desc="Running inference", unit="batch"):
        raw_examples = batch.pop("raw_examples")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = model(batch["input_ids"], batch["attention_mask"])

        if task_type == "classification":
            # If Hugging Face model implementation
            if "nvidia/multilingual-domain-classifier" not in model_name_or_path:
                outputs = outputs.logits.softmax(dim=-1)

            max_scores, max_indices = torch.max(outputs, dim=-1)
            # Build combined records containing original fields + predictions
            for orig, p, idx, s in zip(raw_examples, outputs.tolist(), max_indices, max_scores):
                record = {
                    **orig,  # all original key/value pairs
                    f"{task_name}_scores": p,
                    f"{task_name}_best_class": config.id2label[idx.item()],
                    f"{task_name}_best_score": s.item(),
                }
                buffer.append(record)
        elif task_type == "regression":
            scores = outputs.logits.squeeze(-1).float().tolist()
            for orig, score in zip(raw_examples, scores):
                record = {
                    **orig,  # all original key/value pairs
                    f"{task_name}_score": score,
                    f"{task_name}_normalized_score": _normalize_regression_score(score),
                }
                buffer.append(record)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # save on the fly
        if output_shard_size is not None and len(buffer) >= output_shard_size:
            shard_path = _make_shard_path(output_path, shard_idx)
            Dataset.from_list(buffer).to_parquet(shard_path)
            shard_idx += 1
            buffer.clear()

    print("Saving processed dataset...")
    if buffer:
        if output_shard_size is not None:
            # save on the fly
            # after loop flush whatever is left
            shard_path = _make_shard_path(output_path, shard_idx)
            Dataset.from_list(buffer).to_parquet(shard_path)

            # save by the end
            # cons: buffer grows on the fly and becomes too large
            # output_dataset = Dataset.from_list(buffer)
            # num_shards = math.ceil(len(output_dataset) / output_shard_size)
            # for shard_idx in range(num_shards):
            #     shard = output_dataset.shard(index=shard_idx, num_shards=num_shards)
            #     shard.to_parquet(_make_shard_path(output_path, shard_idx))
        else:
            Dataset.from_list(buffer).to_parquet(output_path)

    print(f"Saved processed dataset with predictions to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
