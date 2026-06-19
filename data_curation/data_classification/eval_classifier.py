"""Evaluate a trained classifier or regressor on a held-out test set.

Prints and saves classification report + confusion matrix. For regression
checkpoints (e.g. quality), predictions and labels are rounded to int (clamped
to 0-5) before scoring, matching `run_dataset_classifier.py`'s
``_normalize_regression_score`` and `train_classifier.py`'s ``--do_predict``.
"""

import os
from typing import Any, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def collate_fn_factory(tokenizer, text_column: str, url_column: str, label_column: str, max_seq_length: int):
    """Tokenise a batch of raw examples; keep originals so we can pull labels later."""

    def collate_fn(batch: List[Dict[str, Any]]):
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
            padding="longest",
            truncation=True,
            max_length=max_seq_length,
        )

        labels = [ex[label_column] for ex in batch]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "raw_examples": batch,
        }

    return collate_fn


def load_local_dataset(input_path: str) -> Dataset:
    """Load Hugging Face dataset from local file or directory."""
    if input_path.endswith(".txt"):
        return load_dataset("text", data_files=input_path, split="train")
    elif input_path.endswith((".jsonl", ".json")):
        return load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith((".csv", ".csv.gz")):
        return load_dataset("csv", data_files=input_path, split="train")
    elif input_path.endswith(".parquet"):
        return load_dataset("parquet", data_files=input_path, split="train")
    elif os.path.isdir(input_path):
        return load_dataset(input_path, split="train")
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


def maybe_load_multiple_local_datasets(dataset_paths: str) -> Dataset:
    """Load datasets from one or more paths joined with '+', then concatenate."""
    datasets = []
    for path in dataset_paths.split("+"):
        ds = load_local_dataset(path)
        print(f"Loaded {ds.num_rows:,d} examples from {path}")
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    final = concatenate_datasets(datasets)
    print(f"Concatenated into {final.num_rows:,d} examples")
    return final


def _normalize_regression_score(score: float | int) -> int:
    """Round regression score to 0-5."""
    return int(round(max(0, min(score, 5))))


def plot_confusion_matrix(cm: np.ndarray, target_names: List[str], out_path: str, title: str | None = None):
    """Save a white→blue heatmap of the confusion matrix (raw counts, row-normalised colour)."""
    n = len(target_names)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    side = max(6, 0.7 * n + 2)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised proportion")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title:
        ax.set_title(title)

    threshold = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > threshold else "black"
            ax.text(
                j, i,
                f"{cm[i, j]:d}",
                ha="center", va="center", color=color, fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix figure to {out_path}")


def main(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    # data
    text_column: str = "text",
    url_column: str = "url",
    label_column: str = "label",
    max_samples: int | None = None,
    # model / inference
    task_type: Literal["classification", "regression"] = "classification",
    task_name: str | None = None,
    max_seq_length: int = 8192,
    batch_size: int = 32,
    num_workers: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset (supports multiple paths joined with '+')
    dataset = maybe_load_multiple_local_datasets(dataset_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    print(f"Mode: {task_type} | max_seq_length: {max_seq_length}")
    if task_type == "classification":
        print(f"label2id: {config.label2id}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_factory(tokenizer, text_column, url_column, label_column, max_seq_length),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # Run inference
    y_true: list[int] = []
    y_pred: list[int] = []
    label2id = {str(k): int(v) for k, v in config.label2id.items()} if task_type == "classification" else None

    for batch in tqdm(dataloader, desc="Running inference", unit="batch"):
        raw_labels = batch["labels"]
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if task_type == "classification":
            preds = outputs.logits.argmax(dim=-1).tolist()
            y_pred.extend(preds)
            y_true.extend(int(l) if isinstance(l, (int, np.integer)) else label2id[str(l)] for l in raw_labels)
        elif task_type == "regression":
            scores = outputs.logits.squeeze(-1).float().tolist()
            y_pred.extend(_normalize_regression_score(s) for s in scores)
            y_true.extend(_normalize_regression_score(float(l)) for l in raw_labels)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    all_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if task_type == "classification":
        id2label = {int(k): str(v) for k, v in config.id2label.items()}
        target_names = [id2label[i] for i in all_classes]
    else:
        target_names = [str(c) for c in all_classes]

    report = classification_report(
        y_true, y_pred, labels=all_classes, target_names=target_names, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{n}" for n in target_names],
        columns=[f"pred:{n}" for n in target_names],
    )

    print("\n=== Classification Report ===")
    print(report)
    print("\n=== Confusion Matrix ===")
    print(cm_df.to_string())

    out_path = os.path.join(output_dir, "predict_results.txt")
    with open(out_path, "w") as f:
        f.write(f"Model: {model_name_or_path}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Num samples: {len(y_true)}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n=== Confusion Matrix ===\n")
        f.write(cm_df.to_string())
        f.write("\n")
    print(f"Saved results to {out_path}")

    fig_path = os.path.join(output_dir, "confusion_matrix.png")
    title = f"Confusion matrix - {task_name} ({task_type})" if task_name else f"Confusion matrix ({task_type})"
    plot_confusion_matrix(cm, target_names, fig_path, title=title)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
