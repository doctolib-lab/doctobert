"""Annotate medical entities in text using GLiNER2."""

import math
import os
import re
from typing import Any

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from gliner2 import GLiNER2
from tqdm import tqdm
from transformers import AutoTokenizer


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:09d}{ext}"
    # os.makedirs(os.path.dirname(root), exist_ok=True)
    # return f"{root}/{idx:09d}{ext}"


def load_local_dataset(input_path: str) -> Dataset:
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
    else:
        raise ValueError(f"Unsupported path or file extension: {input_path}")


def save_hf_dataset(ds: Dataset, output_path: str, output_shard_size: int | None = None):
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


# umls entity groups
# MEDICAL_ENTITIES: dict[str, str] = {
#     # high density
#     "disorders": "Pathological conditions, abnormalities, malfunctions, including diseases, syndromes, injuries, neoplastic processes, signs or symptoms, mental or behavioral dysfunctions",
#     "chemicals_drugs": "Chemical substances for health use, including pharmacologic, biologic, hormones, enzymes, vitamins, immunologic factors, toxins",
#     "anatomy": "Biological structures and body substances: organs, regions, tissues, cells, anatomical components, physiological fluids or materials",
#     "procedures": "Deliberate health-related activities for prevention, diagnosis, treatment, laboratory evaluation, or clinical care",
#     "genes_molecular_sequences": "Genes, gene products, molecular sequences, variants, mutations, genomic or proteomic identifiers",
#     "devices": "Manufactured objects for medical or research use, including implants, instruments, and delivery systems",
#     # medium density
#     "physiology": "Normal biological or mental functions and processes at organism, system, cellular, or molecular levels",
#     # "phenomena": "Natural or human-caused phenomena or processes relevant to health context",
#     # "living_beings": "Organisms and population groups in clinical context, including patients and pathogenic or experimental organisms",
#     # low density
#     # "geographic_areas": "Named physical locations and care settings or units relevant to clinical or epidemiologic context",
#     # "organizations": "Administrative or institutional entities in health care, public health, research, or professional coordination",
#     # "occupations": "Professional roles and biomedical or health-related disciplines",
#     # "concepts": "Abstract entities such as documents, standards, guidelines, regulations, or intellectual products",
#     # "activities": "Human behaviors and routine activities relevant to health status, risk, adherence, or lifestyle",
#     # "objects": "Inanimate physical objects not classified as medical devices, including consumables and non-medical manufactured items",
# }

# MEDICAL_ENTITIES: dict[str, str] = {
#     # high density
#     "disease": "Diagnosed pathology: infection, cancer, syndrome, injury, mental disorder, symptom",
#     "drug": "Pharmaceutical substance: prescription medication, vaccine, active compound, therapeutic agent",
#     "body_part": "Human anatomical structure: organ, tissue, bone, muscle, vessel, nerve, cell",
#     "medical_procedure": "Clinical intervention: surgery, diagnostic test, medical examination, treatment",
#     "molecular_marker": "Genetic and molecular: gene, protein, mutation, biomarker, DNA/RNA sequence",
#     "clinical_device": "Medical instrument: surgical tool, implant, diagnostic equipment, monitoring device",
#     # medium density
#     "vital_function": "Measurable body function: heart rate, blood pressure, hormone level, lab value",
# }

# MEDICAL_ENTITIES: dict[str, str] = {
#     "disease": "Diagnosed human pathology: infection, cancer, syndrome, injury, disorder, clinical symptom",
#     "drug": "Named pharmaceutical: prescription medication, vaccine name, therapeutic compound",
#     "body_part": "Human body structure only: organ, tissue, bone, muscle, blood vessel, nerve, cell",
#     "medical_procedure": "Clinical healthcare intervention: surgery, diagnostic test, medical examination, therapy",
#     "molecular_marker": "Biological molecule: gene name, protein, enzyme, biomarker, receptor, DNA sequence",
#     "clinical_device": "Medical instrument only: surgical tool, implant, diagnostic scanner, monitoring equipment",
# }

MEDICAL_ENTITIES: dict[str, str] = {
    "disease": "Pathological condition: disease, syndrome, infection, cancer, injury, symptom, clinical finding, mental disorder",
    "drug": "Chemical substance for therapy: prescription medication, vaccine, therapeutic compound, drug class, contrast agent",
    "body_part": "Anatomical structure: organ, tissue, bone, muscle, blood vessel, nerve, cell, body fluid, anatomical region",
    "medical_procedure": "Clinical action with methodology: surgery, diagnostic test, medical examination, laboratory test, imaging procedure",
    "molecular_marker": "Molecular entity or biochemical substance: gene, protein, enzyme, receptor, genetic variant, biochemical analyte",
    "clinical_device": "Manufactured medical object: surgical tool, implant, prosthetic, diagnostic scanner, monitoring equipment",
    "vital_function": "Physiological parameter name: heart rate, blood pressure, respiratory rate, temperature, oxygen saturation",
    "living_beings": "Non-human organism in biomedical context: bacterium, virus, fungus, parasite, pathogen, model organism",
}

# _WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _output_features(base_features: Features) -> Features:
    """Define stable output schema to avoid null inference when batches are empty.

    Only declares the two columns gliner_annotate *computes* (`medical_entities` and
    `medical_entity_density`). `original_*` columns are inherited from `base_features` —
    populated upstream by postprocess_extract.py via column rename — so HF passes them
    through unchanged. Redeclaring them here risks a type-cast mismatch if HF's pyarrow
    inference differs from the literal Features struct we'd write.
    """
    feats = base_features.copy()
    feats["medical_entities"] = Features(
        {label: Sequence(Value("string")) for label in MEDICAL_ENTITIES.keys()}
    )
    feats["medical_entity_density"] = Value("float32")
    return feats


def _chunk_text_no_stride(tokenizer, text: str, chunking_max_tokens: int, max_chars_per_doc: int = 1_000_000) -> list[str]:
    """Non-overlapping token chunks; preserves decode boundaries reasonably."""
    if not text:
        return []
    
    # Safety: if document is ridiculously long, handle in blocks to avoid OOM during tokenization
    all_chunks = []
    for start in range(0, len(text), max_chars_per_doc):
        block = text[start : start + max_chars_per_doc]
        enc = tokenizer(block, add_special_tokens=False, truncation=False)
        ids = enc["input_ids"]
        if not ids:
            continue
        for i in range(0, len(ids), chunking_max_tokens):
            sub = ids[i : i + chunking_max_tokens]
            all_chunks.append(tokenizer.decode(sub, skip_special_tokens=True))
    return all_chunks


def _empty_entity_dict() -> dict[str, list[str]]:
    return {k: [] for k in MEDICAL_ENTITIES.keys()}


def _merge_entity_dicts(dicts: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {k: set() for k in MEDICAL_ENTITIES.keys()}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for label, items in d.items():
            if label not in merged:
                continue
            if isinstance(items, list):
                for x in items:
                    if x:
                        merged[label].add(str(x))
    return {k: sorted(list(v)) for k, v in merged.items()}


def _entity_char_count(entities: dict[str, list[str]]) -> int:
    # count chars inside the extracted entity STRINGS (dedup already handled in merge)
    n = 0
    for items in entities.values():
        for s in items:
            n += len(s)
    return n


def _entity_occurrence_char_count(entities: dict[str, list[str]], text: str) -> int:
    """Deduplicate entities across all types, then count chars of all occurrences in text."""
    # collect unique entity strings across all types
    unique_entities = set()
    for items in entities.values():
        unique_entities.update(items)
    # count total chars of all occurrences in text
    n = 0
    for entity in unique_entities:
        if entity:
            n += text.count(entity) * len(entity)
    return n


# def _entity_word_count(entities: dict[str, list[str]]) -> int:
#     # count words inside the extracted entity STRINGS (dedup already handled in merge)
#     n = 0
#     for items in entities.values():
#         for s in items:
#             n += len(_WORD_RE.findall(s))
#     return n


def main(
    input_path: str,
    output_path: str,
    model_name_or_path: str = "fastino/gliner2-multi-v1",
    tokenizer_name_or_path: str | None = None,
    dtype: str | None = None,  # model dtype
    text_column: str = "text",  # text column name
    use_description: bool = False,  # whether to use description for NER
    truncation_max_tokens: int | None = None,  # truncate text to head max_tokens and compute density based on truncated text
    chunking_max_tokens: int = 512,  # max tokens per chunk for sliding window
    threshold: float = 0.5,  # NER threshold
    hf_map_batch_size: int = 128,  # HF dataset map batch size
    infer_batch_size: int = 16,  # GLiNER2 inference batch size
    num_proc: int | None = None,  # HF dataset map number of processes (None=single process, required for CUDA)
    shuffle: bool = False,  # shuffle dataset
    max_samples: int | None = None,  # max samples to process
    output_shard_size: int | None = None,  # output shard size
):
    dataset = load_local_dataset(input_path)
    out_features = _output_features(dataset.features)

    if shuffle:
        dataset = dataset.shuffle(seed=42)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Column '{text_column}' not found. Available columns: {dataset.column_names}"
        )

    extractor = GLiNER2.from_pretrained(model_name_or_path)
    if dtype:
        extractor.to(getattr(torch, dtype))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.to(device)
    extractor.eval()

    tok_name = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

    def process_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts: list[str] = [("" if x is None else str(x)) for x in batch[text_column]]
        num_docs = len(texts)

        # truncate text to middle truncation_max_tokens and compute density based on truncated text
        # middle truncation avoids boilerplate at head (nav, headers) and tail (footers, legal)
        if truncation_max_tokens is not None and truncation_max_tokens > 0:
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
            )
            truncated_ids = []
            for ids in enc["input_ids"]:
                if len(ids) <= truncation_max_tokens:
                    truncated_ids.append(ids)
                else:
                    # take middle tokens
                    start = (len(ids) - truncation_max_tokens) // 2
                    truncated_ids.append(ids[start : start + truncation_max_tokens])
            texts = tokenizer.batch_decode(truncated_ids, skip_special_tokens=True)

        # 1) chunk every doc (no overlap), flatten chunks for batched inference
        flat_chunks: list[str] = []
        flat_meta: list[int] = []  # doc_idx for each chunk
        for i, t in enumerate(texts):
            chunks = _chunk_text_no_stride(tokenizer, t, chunking_max_tokens=chunking_max_tokens)
            flat_chunks.extend(chunks)
            flat_meta.extend([i] * len(chunks))

        # 2) run extraction in batches
        chunk_out: list[dict[str, Any]] = []
        if flat_chunks:
            with torch.inference_mode():
                # many installs return: List[ { "entities": {label: [str,...]}} , ... ]
                chunk_out = extractor.batch_extract_entities(
                    flat_chunks,
                    MEDICAL_ENTITIES if use_description else list(MEDICAL_ENTITIES.keys()),
                    batch_size=infer_batch_size,
                    threshold=threshold,
                )
        
        # 3) merge + calculate density in a single pass over docs to save memory
        per_doc_results: list[list[dict[str, list[str]]]] = [[] for _ in range(num_docs)]
        for res, doc_idx in zip(chunk_out, flat_meta):
            if isinstance(res, dict) and "entities" in res:
                per_doc_results[doc_idx].append(res["entities"])
            elif isinstance(res, dict): # gliner v1 fallback
                per_doc_results[doc_idx].append(res)
        
        # Free up some memory
        del chunk_out
        del flat_chunks
        del flat_meta

        merged_entities: list[dict[str, list[str]]] = []
        density_list: list[float] = []

        for i in range(num_docs):
            t = texts[i]
            ents_list = per_doc_results[i]
            
            merged = _merge_entity_dicts(ents_list) if ents_list else _empty_entity_dict()
            merged_entities.append(merged)

            # total_words = len(_WORD_RE.findall(t))
            # entity_words = _entity_word_count(merged)
            # density = (entity_words / total_words) if total_words > 0 else 0.0

            total_chars = len(t)
            # count chars of all entities
            # entity_chars = _entity_char_count(merged)
            # count chars of all occurrences in text
            entity_chars = _entity_occurrence_char_count(merged, t)
            density = (entity_chars / total_chars) if total_chars > 0 else 0.0
            density_list.append(density)
            
            # Clear doc data as we go
            per_doc_results[i] = None

        return {
            "medical_entities": merged_entities,
            "medical_entity_density": density_list,
            # `original_medical_entities` / `original_medical_entity_density` are passed
            # through from input (renamed upstream in postprocess_extract.py) — no backup
            # logic needed here.
        }

    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=hf_map_batch_size,
        num_proc=num_proc,
        desc="GLiNER2 medical extraction",
        features=out_features,
    )

    save_hf_dataset(processed_dataset, output_path, output_shard_size=output_shard_size)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
