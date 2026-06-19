"""Postprocess synthesized dataset, filter with language and gopher repetition rules."""

import copy
import glob
import json
import math
import os
import re
import random
from typing import Any, Callable

import pandas as pd
from datasets import Dataset, Features, Value, load_dataset
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import GopherRepetitionFilter, LanguageFilter
from datatrove.pipeline.readers.base import BaseReader
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter
from tqdm import tqdm

pattern_dialogue = re.compile(r"^\s*#{1,6}\s*conversation\b.*$", re.IGNORECASE | re.MULTILINE)
# Regex to match JSON objects - finds outermost braces with content
pattern_json_object = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


# Canonical 8 GLiNER medical entity types (matches the upstream extraction schema).
# Used by `normalize_medical_entities` to guarantee a uniform struct schema
# (each entity-type field is `list<string>`, defaulting to []) across all rows so that
# datatrove's Parquet writer doesn't choke on schema mismatches between workers/shards.
# Full per-type prompt definitions live in the GLiNER extraction script.
MEDICAL_ENTITIES: tuple[str, ...] = (
    "disease",
    "drug",
    "body_part",
    "medical_procedure",
    "molecular_marker",
    "clinical_device",
    "vital_function",
    "living_beings",
)


def extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using regex. Returns the last occurrence if multiple found."""
    matches = pattern_json_object.findall(text)
    if not matches:
        return None
    # Take the last match
    json_str = matches[-1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def remove_surrogates(text: str | None) -> str | None:
    """Remove surrogate characters that cannot be encoded as UTF-8."""
    if text is None:
        return None
    # Encode to UTF-8, replacing surrogates with replacement character, then decode back
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def strip_harmony_prefix(text: str | None) -> str | None:
    """Strip gpt-oss harmony output prefix: "analysis<reasoning>assistantfinal<final answer>".
    Returns the text after "assistantfinal" if found, otherwise the original text unchanged.
    Safe no-op for outputs without the marker (other LLMs).
    """
    if text is None:
        return None
    idx = text.find("assistantfinal")
    if idx >= 0:
        return text[idx + len("assistantfinal"):].lstrip()
    return text


# ----------------------------------------------------------------------------
# Recipe-specific transforms (V3 / dialogue / mimic). Not active for V4.x rewriting
# (which produces plain-text `output`, no JSON wrapper). Kept here as utilities;
# call from main() when reprocessing the relevant recipes' outputs.
# ----------------------------------------------------------------------------


def extract_rewriting_json(example: dict, text_column_name: str = "output") -> dict:
    """Parse JSON-wrapped rewriting output (V3 / V3.1 / V3.2 schemas).

    JSON fields by version:
      v3:   is_rewritable, format, rewritten_text, title
      v3_1: is_rewritable, rewritten_text
      v3_2: is_rewritable, rewritten_text
    For v3_1/v3_2, json_output.get("format")/title return None — schema still aligns.
    """
    text = example[text_column_name]
    # offline LLM.generate() returns plain text; <think>...</think> stripped earlier if needed
    try:
        json_output = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        json_output = extract_json_from_text(text) or {}

    title = json_output.get("title")  # v3 only; v3_1/v3_2 → None
    rewritten_text = json_output.get("rewritten_text")
    # Validate types (LLM sometimes outputs lists)
    if (title is not None and not isinstance(title, str)) or (
        rewritten_text is not None and not isinstance(rewritten_text, str)
    ):
        return {
            "is_rewritable": None, "rewriting_format": None, "rewriting_style": None, "rewriting_model": None,
            "rewriting_title": None, "rewriting_text": None, "text": None,
            "medical_entities": json.dumps(example.get("medical_entities", {})),
            "original_medical_entities": json.dumps(example.get("original_medical_entities", {})),
            "original_medical_entities_pretrained": json.dumps(example.get("original_medical_entities_pretrained", {})),
        }
    title = remove_surrogates(title)
    rewritten_text = remove_surrogates(rewritten_text)
    return {
        "is_rewritable": json_output.get("is_rewritable"),
        "rewriting_format": json_output.get("format"),  # v3: string | v3_1/v3_2: None
        "rewriting_style": example.get("gen_configs", {}).get("rewriting_style"),
        "rewriting_model": example.get("gen_configs", {}).get("model"),
        "rewriting_title": title,  # v3: string | v3_1/v3_2: None
        "rewriting_text": rewritten_text,
        "text": rewritten_text,
        # datatrove writing compatibility — serialize struct columns
        "medical_entities": json.dumps(example.get("medical_entities", {})),
        "original_medical_entities": json.dumps(example.get("original_medical_entities", {})),
        "original_medical_entities_pretrained": json.dumps(example.get("original_medical_entities_pretrained", {})),
    }


def drop_all_null_columns(dataset: Dataset, candidates: tuple[str, ...]) -> Dataset:
    """Drop columns from `candidates` that are entirely None in `dataset`.

    Used after `extract_rewriting_json` to drop `rewriting_format` / `rewriting_title`
    when processing v3_1/v3_2 outputs (those schemas have no `format`/`title` fields).
    Uses arrow null_count metadata (O(1) per column).
    """
    cols_to_drop = [
        col for col in candidates
        if col in dataset.column_names and dataset.data.column(col).null_count == dataset.num_rows
    ]
    if cols_to_drop:
        dataset = dataset.remove_columns(cols_to_drop)
        print(f"Dropped all-None columns: {cols_to_drop}")
    return dataset


def extract_dialogue(example: dict) -> dict:
    """Strip the `# Conversation` header and surrounding blank lines from dialogue outputs.

    Used for synthesized dialogue corpora (pmc_patients_v2 dialogue generation).
    """
    text = example.get("text", "")
    if not isinstance(text, str):
        return {"raw_text": text, "text": text}

    parts = pattern_dialogue.split(text, maxsplit=1)
    content_after_header = parts[1] if len(parts) > 1 else text

    cleaned_lines = [ln for ln in content_after_header.splitlines() if ln.strip()]
    cleaned_text = "\n".join(cleaned_lines).strip()
    return {"raw_text": text, "text": cleaned_text}


def _short_model_name(s: str | None) -> str | None:
    """Keep only the last `org/name` path components.

    'Qwen/Qwen3.5-35B-A3B-FP8'                          -> unchanged
    '/lustre/.../models/Qwen/Qwen3.5-35B-A3B-FP8'       -> 'Qwen/Qwen3.5-35B-A3B-FP8'
    """
    if not s:
        return s
    parts = s.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def normalize_medical_entities(example: dict) -> dict:
    """Coerce `medical_entities` to a uniform `dict[str, list[str]]` with all 8 canonical keys.

    Guarantees every row has the same struct schema for the Parquet writer:
    - Missing or None entire `medical_entities` → all 8 keys defaulting to []
    - Missing or None per-entity lists → []
    Without this, multi-worker datatrove can write shards with non-unifiable struct
    schemas (None vs populated), tripping pyarrow's schema check at read time.
    """
    me = example.get("medical_entities") or {}
    return {"medical_entities": {k: (me.get(k) or []) for k in MEDICAL_ENTITIES}}


def pack_v4_rewriting_config(example: dict) -> dict:
    """Pack V4.x rewriting metadata into a single `rewriting_config` struct.

    Reads from `gen_configs.{model, rewriting_style.{abbreviation, register}}` and the
    top-level `genre` / `audience` columns (propagated from stage 1). Returned with the
    field added; caller passes `remove_columns=["gen_configs", "genre", "audience", ...]`
    to drop the now-redundant raw columns.

    Used on V4.1+ / V4.2 stage-2 outputs in main().
    """
    gc = example.get("gen_configs") or {}
    style = gc.get("rewriting_style", {}) or {}
    return {
        "rewriting_config": {
            "model": _short_model_name(gc.get("model")),
            "genre": example.get("genre"),
            "audience": example.get("audience"),
            "abbreviation": style.get("abbreviation"),
            "register": style.get("register"),
        },
    }


def normalize_schema_nulls(example: dict) -> dict:
    """Coerce nulls to empty strings / sentinel values for MIMIC + vocabulary schemas.

    Datatrove's Parquet writer chokes on nullable struct/list columns; this normalizes
    the per-corpus optional columns that may contain Nones.
    """
    result = {}
    # mimic-iv-note discharge summary null hadm_id
    if "hadm_id" in example:
        result["hadm_id"] = example["hadm_id"] if example["hadm_id"] is not None else -1
    # vocabulary
    if "definition" in example:
        result["definition"] = example["definition"] if example["definition"] is not None else ""
    if "etym" in example:
        result["etym"] = example["etym"] if example["etym"] is not None else ""
    if "synonyms" in example:
        if not example["synonyms"]:
            example["synonyms"] = [""]
    if "translation_en" in example:
        result["translation_en"] = example["translation_en"] if example["translation_en"] is not None else ""
    # mimic-iii noteevents
    for col in ("ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME",
                "CATEGORY", "DESCRIPTION", "CGID", "ISERROR"):
        if col in example:
            result[col] = example[col] if example[col] is not None else ""
    return result


def load_local_dataset(input_path: str, num_proc: int = 1) -> Dataset:
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
        # OLD (kept for reference): rely on HF datasets auto-loader.
        # Works only if HF datasets version uses recursive `**.parquet` pattern by default.
        # return load_dataset(input_path, split="train")  #, num_proc=num_proc)

        # NEW: explicit glob — try flat top-level first (original behavior), then recursive
        # (for nested per-shard subdirs produced by llm_rewrite_array.slurm).
        parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if not parquet_files:
            parquet_files = sorted(glob.glob(os.path.join(input_path, "**", "*.parquet"), recursive=True))
        if parquet_files:
            return load_dataset("parquet", data_files=parquet_files, split="train")
        return load_dataset(input_path, split="train")  #, num_proc=num_proc)
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    # return f"{root}_{idx:09d}{ext}"
    os.makedirs(os.path.dirname(root), exist_ok=True)
    return f"{root}/{idx:09d}{ext}"


def save_hf_dataset(ds: Dataset, output_path: str, output_shard_size: int | None = None):
    """Save Hugging Face dataset to local file."""
    if output_path.endswith(".txt"):
        with open(output_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="Writing"):
                f.write(row["text"] + "\n")
    elif output_path.endswith(".jsonl"):
        ds.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif output_path.endswith(".parquet"):
        if output_shard_size is not None:
            num_shards = math.ceil(len(ds) / output_shard_size)
            for shard_idx in range(num_shards):
                shard = ds.shard(index=shard_idx, num_shards=num_shards)
                shard.to_parquet(_make_shard_path(output_path, shard_idx))
        else:
            ds.to_parquet(output_path)
    else:
        ds.save_to_disk(output_path)


class HuggingFaceDatasetInstanceReader(BaseReader):
    """Read data from an *already loaded* HuggingFace dataset (Dataset or IterableDataset).
       Will read each row as a separate document.

    Args:
        dataset: datasets.Dataset or datasets.IterableDataset (NOT a string). If you have a DatasetDict, select a split first.
        source_name: a label used in metadata to identify the source (default: 'hf_dataset_instance')
        limit: limit the number of rows to read
        skip: skip the first n rows
        batch_size: the batch size to use when iterating
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes: data: dict, path: str, id_in_file: int | str
            Returns: dict with at least a "text" key
        text_key: key to use for the text in the default adapter (default: "text"). Ignored if you provide your own `adapter`
        id_key: key to use for the id in the default adapter (default: "id"). Ignored if you provide your own `adapter`
        default_metadata: default metadata to add to all documents
        shuffle_files: shuffle underlying files/shards if supported by the dataset.
            Mostly used for data viz; avoid when doing dedup.
    """

    name = "🤗 HuggingFaceInstance"
    _requires_dependencies = ["datasets"]

    def __init__(
        self,
        dataset: Any,  # Dataset | IterableDataset
        *,
        source_name: str = "hf_dataset_instance",
        limit: int = -1,
        skip: int = 0,
        batch_size: int = 1000,
        doc_progress: bool = False,
        adapter: Callable | None = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(limit, skip, adapter, text_key, id_key, default_metadata)
        self.dataset = dataset
        self.source_name = source_name
        self.batch_size = batch_size
        self.doc_progress = doc_progress
        self.shuffle_files = shuffle_files

    def get_document_from_dict(self, data: dict, source_file: str, id_in_file: int | str):
        document = super().get_document_from_dict(data, source_file, id_in_file)
        if document:
            document.metadata.setdefault("dataset", source_file)
        return document

    def _get_dataset_shard(self, dst, rank: int, world_size: int):
        from datasets import Dataset, IterableDataset
        from datasets.distributed import split_dataset_by_node

        if isinstance(dst, Dataset):
            return dst.shard(world_size, rank, contiguous=False)
        elif isinstance(dst, IterableDataset) and getattr(dst, "n_shards", 0) > 1:
            # Shard at the data-source/file level if possible
            if rank >= dst.n_shards:
                print(f"Warning: requested shard {rank} of a streaming dataset, but it only has {dst.n_shards} shards.")
                return None
            ex_iterable = dst._ex_iterable.shard_data_sources(index=rank, num_shards=world_size, contiguous=False)
            return IterableDataset(
                ex_iterable=ex_iterable,
                info=dst._info.copy(),
                split=dst._split,
                formatting=dst._formatting,
                shuffling=copy.deepcopy(dst._shuffling),
                distributed=copy.deepcopy(dst._distributed),
                token_per_repo_id=dst._token_per_repo_id,
            )
        else:
            # Fallback to inter-file sharding (handles single-shard IterableDataset too)
            return split_dataset_by_node(dataset=dst, rank=rank, world_size=world_size)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        from datasets import DatasetDict, IterableDataset

        if data:
            yield from data

        ds = self.dataset

        # Enforce: pass a concrete split, not a dict
        if isinstance(ds, DatasetDict):
            raise ValueError(
                f"HuggingFaceDatasetInstanceReader expects a concrete split (Dataset/IterableDataset). "
                f"Select one first, e.g., ds = ds['train']. Available splits: {list(ds.keys())}"
            )

        # Optional shuffling (mirrors your original semantics)
        if self.shuffle_files:
            if isinstance(ds, IterableDataset):
                ds = ds.shuffle(seed=42, buffer_size=1000)
            else:
                ds = ds.shuffle(seed=42)

        shard = self._get_dataset_shard(ds, rank, world_size)
        if not shard:
            return

        with tqdm(total=self.limit if self.limit != -1 else None, disable=not self.doc_progress) as pbar:
            li = 0
            for batch in shard.iter(self.batch_size):
                if self.limit != -1 and li >= self.limit:
                    break
                documents = []
                with self.track_time("batch"):
                    # HF batches: dict of columns -> lists; zip to get row dicts
                    for line in (dict(zip(batch, t)) for t in zip(*batch.values())):
                        if self.limit != -1 and li >= self.limit:
                            break
                        document = self.get_document_from_dict(line, self.source_name, f"{rank:05d}/{li}")
                        if not document:
                            continue
                        documents.append(document)
                        self.update_doc_stats(document)
                        self.stat_update("documents")
                        li += 1
                        pbar.update()
                yield from documents


def main(
    input_path: str,
    output_path: str | None = None,
    filtering_output_path: str | None = None,
    text_column_name: str = "text",
    languages: str
    | list[str] = "fr",  # lang code https://fasttext.cc/docs/en/language-identification.html#list-of-supported-languages
    num_workers: int = 16,
    # output_shard_size: int | None = None,
    max_samples: int | None = None,
):
    if output_path is None:
        output_path = f"{input_path}_postprocessed"
    if filtering_output_path is None:
        filtering_output_path = f"{input_path}_removed"

    dataset = load_local_dataset(input_path)  # , num_proc=num_workers)
    print(f"Loaded {dataset.num_rows:,d} documents")

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    if text_column_name != "text":
        if "text" in dataset.column_names:
            dataset = dataset.rename_column("text", "original_text")
        # ds = ds.add_column("text", ds[text_column_name])

    # Strip gpt-oss harmony prefix ("analysis...assistantfinal<text>") if present.
    # No-op for outputs without the marker (other LLMs unchanged).
    # dataset = dataset.map(
    #     lambda x: {"text": strip_harmony_prefix(x["text"])},
    #     num_proc=num_workers,
    #     desc="Stripping harmony prefix (no-op for non-gpt-oss)...",
    # )

    if "medical_entities" in dataset.column_names:
        # # Optional defense-in-depth: ensure 8-key uniform struct before serializing.
        # # Redundant since postprocess_datatrove.py re-applies the same {k: list or []}
        # # logic when deserializing the JSON. Kept here for reference; uncomment if
        # # upstream V4.x outputs ever produce non-uniform structs.
        # dataset = dataset.map(
        #     normalize_medical_entities,
        #     num_proc=num_workers,
        #     desc="Normalizing medical_entities to uniform 8-key struct...",
        # )

        # JSON-serialize so datatrove's per-worker ParquetWriter doesn't trip on
        # `list<null>` schemas inferred from all-[] columns (uneven across workers).
        # Parsed back to dict in postprocess_datatrove.py.
        dataset = dataset.map(
            lambda x: {"medical_entities": json.dumps(x["medical_entities"])},
            num_proc=num_workers,
            desc="JSON-serializing medical_entities (datatrove struct-inference workaround)...",
        )

        # Rename source-doc GLiNER extraction to `original_*` to free up the canonical names
        # for downstream re-extraction (gliner_annotate.py on the rewritten `text` column).
        dataset = dataset.rename_column("medical_entities", "original_medical_entities")
        dataset = dataset.rename_column("medical_entity_density", "original_medical_entity_density")

    # ---- Optional recipe-specific transforms (off by default; uncomment per recipe) ----
    # 
    # V4.1+ rewriting outputs — pack rewriting config into a single struct column, drop raw config columns:
    # 
    # no postprocess
    dataset = dataset.rename_column(text_column_name, "text")
    # Pack V4.x rewriting metadata into a single `rewriting_config` struct, drop the raw
    # source columns. Logical grouping; see pack_v4_rewriting_config() docstring.
    if "gen_configs" in dataset.column_names:
        cols_to_drop = [c for c in ("gen_configs", "genre", "audience", "sampled_from_n")
                        if c in dataset.column_names]
        dataset = dataset.map(
            pack_v4_rewriting_config,
            num_proc=num_workers,
            remove_columns=cols_to_drop,
            desc=f"Packing rewriting_config + dropping {cols_to_drop}...",
        )

    #
    # V3 / V3.1 / V3.2 rewriting outputs are JSON-wrapped — parse, drop empty-version cols, filter:
    # dataset = dataset.map(
    #     extract_rewriting_json,
    #     fn_kwargs={"text_column_name": text_column_name},
    #     num_proc=num_workers,
    #     remove_columns=[text_column_name, "gen_configs"],
    #     desc="Extracting V3 JSON...",
    # )
    # dataset = drop_all_null_columns(dataset, ("rewriting_format", "rewriting_title"))
    # dataset = dataset.filter(
    #     lambda x: x["is_rewritable"] and x["text"],
    #     num_proc=num_workers,
    #     desc="Filtering rewritable examples...",
    # )
    #
    # Dialogue corpora (pmc_patients_v2 dialogue generation) — strip `# Conversation` header:
    # dataset = dataset.map(extract_dialogue, num_proc=num_workers, desc="Extracting dialogues...")
    #
    # MIMIC / vocabulary corpora — coerce nullable optional columns to sentinels:
    # dataset = dataset.map(normalize_schema_nulls, num_proc=num_workers, desc="Normalizing samples...")

    # update num_words
    # Preserve the source-doc word count under `original_num_words` before the
    # active flow recomputes `num_words` from the new `text` (the rewritten output).
    if "num_words" in dataset.column_names:
        dataset = dataset.rename_column("num_words", "original_num_words")
    # if "num_words" not in dataset.column_names:
    dataset = dataset.map(
        lambda x: {"num_words": len(x["text"].split())},
        num_proc=num_workers,
        desc="Counting words...",
    )

    # filter with gopher repetition rules
    LocalPipelineExecutor(
        pipeline=[
            HuggingFaceDatasetInstanceReader(dataset=dataset),
            LanguageFilter(
                # languages=languages,  # comment to compare whatever top1 prob with threshold
                languages=languages,
                language_threshold=0.5,
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/bad_language"),
            ),
            GopherRepetitionFilter(
                exclusion_writer=JsonlWriter(f"{filtering_output_path}/gopher_rep"),
            ),
            ParquetWriter(output_path),
        ],
        tasks=num_workers,
        # workers=-1,
    ).run()

    # Save results
    # save_hf_dataset(ds, output_path, output_shard_size=output_shard_size)
    # print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
