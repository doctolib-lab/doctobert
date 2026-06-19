"""
Postprocess LLM topic/quality annotations.

For topic annotations:
1. Normalize the topics to the canonical categories.
2. Filter out the topics that are not in the canonical categories.
3. Save the dataset to parquet shards.

For quality annotations:
1. Parse the quality annotation and filter out invalid scores.
2. Save the dataset to parquet shards.

For medical entity annotations:
- `medical_entities`: convert to GLiNER token/span format.
- `medical_entities_v2`: convert to GLiNER2 JSONL format (input + entities dict).
"""

import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Literal

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

CANONICAL = [
    "Clinical cases & vignettes",
    "Clinical guidelines & pathways",
    "Patient education & lifestyle",
    "Wellness, supplements & CAM",
    "Public health, policy & programs",
    "Commercial & promotional",
    "Drugs, trials & regulation",
    "Biomedical & mechanistic science",
    "Medical devices, diagnostics & imaging",
    "Health IT, telemedicine & operations",
    "Occupational health & safety",
    "Health workforce education & training",
    "Health services & facilities",
    "Other health",
    "Others",
]

# Lowercased lookup for exact canonical names
_CANON_LUT = {c.lower(): c for c in CANONICAL}

# Hand-tuned mappings for your off-list labels + common variants
_DIRECT_MAP = {
    # trials/research variants
    "clinical trials & regulation": "Drugs, trials & regulation",
    "clinical trials & research": "Drugs, trials & regulation",
    "clinical trials & pathways": "Drugs, trials & regulation",
    "clinical research & trials": "Drugs, trials & regulation",
    "clinical research & studies": "Drugs, trials & regulation",
    "clinical research": "Drugs, trials & regulation",
    # diagnostics/imaging variants
    "clinical diagnostics & imaging": "Medical devices, diagnostics & imaging",
    "clinical diagnostics & testing": "Medical devices, diagnostics & imaging",
    # telemedicine variants
    "telemedicine & operations": "Health IT, telemedicine & operations",
    "telemedicine & remote care": "Health IT, telemedicine & operations",
    "telemedicine & digital health": "Health IT, telemedicine & operations",
    "telemedicine & health it": "Health IT, telemedicine & operations",
    "telemedicine & health services": "Health IT, telemedicine & operations",
    "telemedicine & virtual care": "Health IT, telemedicine & operations",
    # services/facilities
    "clinical services & facilities": "Health services & facilities",
    # procedures
    "clinical procedures & interventions": "Clinical guidelines & pathways",
    "clinical procedures & techniques": "Clinical guidelines & pathways",
    # education-ish
    "clinical definitions & terminology": "Health workforce education & training",
    "clinical anatomy & morphology": "Health workforce education & training",
    # french terms
    "clauses de conscience": "Public health, policy & programs",
    "radioprotection": "Occupational health & safety",
    "addictions comportementales": "Other health",
    # null-ish
    # "none": "Others",
    # "": "Others",
}

MEDICAL_ENTITY_MAP = {
    "DISO": "disorders",
    "CHEM": "chemicals_drugs",
    "ANAT": "anatomy",
    "PROC": "procedures",
    "LIVB": "living_beings",
    "PHYS": "physiology",
    "DEVI": "devices",
    "PHEN": "phenomena",
    "GEOG": "geographic_areas",
    "ORGA": "organizations",
    "OCCU": "occupations",
    "CONC": "concepts",
    "ACTI": "activities",
    "OBJC": "objects",
    "GENE": "genes_molecular_sequences",
}

# v2 entity types (UMLS-based)
# MEDICAL_ENTITY_DESCRIPTIONS_V2: dict[str, str] = {
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
#     ## low density
#     # "geographic_areas": "Named physical locations and care settings or units relevant to clinical or epidemiologic context",
#     # "organizations": "Administrative or institutional entities in health care, public health, research, or professional coordination",
#     # "occupations": "Professional roles and biomedical or health-related disciplines",
#     # "concepts": "Abstract entities such as documents, standards, guidelines, regulations, or intellectual products",
#     # "activities": "Human behaviors and routine activities relevant to health status, risk, adherence, or lifestyle",
#     # "objects": "Inanimate physical objects not classified as medical devices, including consumables and non-medical manufactured items",
# }

# v3 entity types (simplified for medical term density)
MEDICAL_ENTITY_DESCRIPTIONS: dict[str, str] = {
    "disease": "Pathological condition: disease, syndrome, infection, cancer, injury, symptom, clinical finding, mental disorder",
    "drug": "Chemical substance for therapy: prescription medication, vaccine, therapeutic compound, drug class, contrast agent",
    "body_part": "Anatomical structure: organ, tissue, bone, muscle, blood vessel, nerve, cell, body fluid, anatomical region",
    "medical_procedure": "Clinical action with methodology: surgery, diagnostic test, medical examination, laboratory test, imaging procedure",
    "molecular_marker": "Molecular entity or biochemical substance: gene, protein, enzyme, receptor, genetic variant, biochemical analyte",
    "clinical_device": "Manufactured medical object: surgical tool, implant, prosthetic, diagnostic scanner, monitoring equipment",
    "vital_function": "Physiological parameter name: heart rate, blood pressure, respiratory rate, temperature, oxygen saturation",
    "living_beings": "Non-human organism in biomedical context: bacterium, virus, fungus, parasite, pathogen, model organism",
}

MEDICAL_ENTITY_GROUPS = list(MEDICAL_ENTITY_DESCRIPTIONS.keys())


def deduplicate_dataset(ds: Dataset, field_name: str, num_workers: int = 8, seed: int = 42) -> Dataset:
    """Deduplicate dataset by field, keeping first occurrence after shuffle."""

    def _get_hash(example):
        """Get hash of content field."""
        return {"_hash": hash(example[field_name])}

    def _check_uniques(example, uniques):
        """Check if current hash is still in set of unique hashes and remove if true."""
        if example["_hash"] in uniques:
            uniques.remove(example["_hash"])
            return True
        else:
            return False

    ds = ds.shuffle(seed=seed)
    ds = ds.map(_get_hash, num_proc=num_workers, desc="get hash")
    uniques = set(ds.unique("_hash"))
    ds = ds.filter(_check_uniques, fn_kwargs={"uniques": uniques}, desc="dedup data")
    ds = ds.remove_columns("_hash")
    return ds


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:09d}{ext}"
    # os.makedirs(os.path.dirname(root), exist_ok=True)
    # return f"{root}/{idx:09d}{ext}"


def save_hf_dataset(ds: Dataset, output_path: str, output_shard_size: int | None = None):
    """Save Hugging Face dataset to local file."""

    def _remove_none(obj):
        if isinstance(obj, dict):
            return {k: _remove_none(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [_remove_none(x) for x in obj]
        return obj

    if output_path.endswith(".txt"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc="Writing"):
                f.write(row["text"] + "\n")
    elif output_path.endswith(".jsonl"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # ds.to_json(output_path, orient="records", lines=True, force_ascii=False)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in tqdm(ds, desc=f"Writing {os.path.basename(output_path)}"):
                f.write(json.dumps(_remove_none(row), ensure_ascii=False) + "\n")
    elif output_path.endswith(".parquet"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_shard_size is not None:
            num_shards = math.ceil(len(ds) / output_shard_size)
            for shard_idx in range(num_shards):
                shard = ds.shard(index=shard_idx, num_shards=num_shards)
                shard.to_parquet(_make_shard_path(output_path, shard_idx))
        else:
            ds.to_parquet(output_path)
    else:
        ds.save_to_disk(output_path)


def _normalize_text(s: str) -> str:
    """Lower, trim, collapse spaces, strip accents, normalize &/and."""
    s = s or ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" and ", " & ")
    return s


def normalize_topic(label: str | None) -> str | None:
    """Normalize a topic label to the canonical categories."""
    if label is None:
        return None

    norm = _normalize_text(label)
    if norm in _CANON_LUT:
        return _CANON_LUT[norm]

    # Direct mapping of known variants
    if norm in _DIRECT_MAP:
        return _DIRECT_MAP[norm]

    return label


def postprocess_topic(ds: Dataset, num_workers: int = 8) -> Dataset:
    """Parse the topic annotation and normalize it to the canonical categories."""

    def process_func(example):
        try:
            json_output = json.loads(example["output"])
        except json.JSONDecodeError:
            json_output = {}

        topic = json_output.get("topic")
        normalized_topic = normalize_topic(topic)

        return {"topic": normalized_topic}

    ds = ds.map(process_func, num_proc=num_workers)

    # filter out invalid topics
    ds = ds.filter(
        lambda x: x["topic"] in CANONICAL,
        num_proc=num_workers,
    )
    print(f"Filtered to {ds.num_rows:,d} examples")

    return ds


def postprocess_quality(ds: Dataset, num_workers: int = 8) -> Dataset:
    """Parse the quality annotation and filter out invalid scores."""

    def process_func(example):
        try:
            json_output = json.loads(example["output"])
        except json.JSONDecodeError:
            json_output = {}

        score = json_output.get("score")
        if score is not None:
            score = int(score)

        return {"score": score}

    ds = ds.map(process_func, num_proc=num_workers)

    # score counts
    score_counts = Counter(ds["score"])
    print(score_counts)

    # filter out invalid scores
    allowed_scores = set(range(6))
    ds = ds.filter(
        lambda x: x["score"] in allowed_scores,
        num_proc=num_workers,
    )
    print(f"Filtered to {ds.num_rows:,d} examples")

    return ds

def postprocess_medical_entities(ds: Dataset, num_workers: int = 8, text_column_name: str = "text") -> Dataset:
    """Parse the medical entities annotation and convert to GLiNER format."""

    def _normalize_for_match(text: str) -> str:
        return (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("‛", "'")
            .replace("–", "-")
            .replace("—", "-")
        )

    def tokenize_with_spans(text: str):
        tokens = []
        spans = []
        for match in re.finditer(r"\w+|[^\w\s]", text):
            tokens.append(match.group(0))
            spans.append((match.start(), match.end()))
        return tokens, spans

    def find_token_matches(
        norm_tokens: list[str], norm_span_tokens: list[str]
    ) -> list[tuple[int, int]]:
        """Find all start/end token indices (inclusive) where span_tokens appears."""
        if not norm_span_tokens:
            return []
        matches: list[tuple[int, int]] = []
        span_len = len(norm_span_tokens)
        for i in range(len(norm_tokens) - span_len + 1):
            if norm_tokens[i : i + span_len] == norm_span_tokens:
                matches.append((i, i + span_len - 1))
        return matches

    def process_func(example):
        try:
            parsed = json.loads(example["output"])
        except json.JSONDecodeError:
            parsed = []

        if isinstance(parsed, dict) and "entities" in parsed:
            medical_entities = parsed.get("entities", [])
        elif isinstance(parsed, list):
            medical_entities = parsed
        else:
            medical_entities = []

        tokens, token_spans = tokenize_with_spans(example[text_column_name])
        norm_tokens = [_normalize_for_match(tok).lower() for tok in tokens]
        norm_text_for_search = _normalize_for_match(example[text_column_name]).lower()

        token_match_cache: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        used_counts: dict[tuple[str, str, tuple[str, ...]], int] = {}
        ner: list[list] = []

        for med_ent in medical_entities:
            if not isinstance(med_ent, dict):
                continue
            label = med_ent.get("entity")
            # label = MEDICAL_ENTITY_MAP.get(label, label)
            span_text = (med_ent.get("text") or "").strip()
            if label not in MEDICAL_ENTITY_GROUPS or not span_text:
                continue

            span_tokens, _ = tokenize_with_spans(span_text)
            norm_span_tokens = [_normalize_for_match(tok).lower() for tok in span_tokens]
            if not norm_span_tokens:
                continue
            span_tokens_key = tuple(norm_span_tokens)
            norm_span_text = _normalize_for_match(span_text).lower()

            if span_tokens_key not in token_match_cache:
                token_match_cache[span_tokens_key] = find_token_matches(
                    norm_tokens, norm_span_tokens
                )

            matches = token_match_cache[span_tokens_key]
            key = (label, norm_span_text, span_tokens_key)
            use_idx = used_counts.get(key, 0)
            if use_idx >= len(matches):
                # fallback: try raw string search for an additional occurrence
                raw_matches = list(
                    re.finditer(re.escape(norm_span_text), norm_text_for_search)
                )
                if use_idx < len(raw_matches):
                    # map char span to token span
                    char_span = raw_matches[use_idx].span()
                    overlapping = [
                        idx
                        for idx, (start, end) in enumerate(token_spans)
                        if not (end <= char_span[0] or start >= char_span[1])
                    ]
                    if overlapping:
                        ner.append(
                            {
                                "start": overlapping[0],
                                "end": overlapping[-1],
                                "label": label,
                            }
                        )
                        used_counts[key] = use_idx + 1
                    continue
                else:
                    continue

            start_tok, end_tok = matches[use_idx]
            ner.append(
                {
                    "start": start_tok,
                    "end": end_tok,
                    "label": label,
                }
            )
            used_counts[key] = use_idx + 1

        return {
            "tokenized_text": tokens,
            "ner": ner,
            "ner_labels": MEDICAL_ENTITY_GROUPS,
        }

    return ds.map(process_func, num_proc=num_workers)


def postprocess_medical_entities_v2(ds: Dataset, num_workers: int = 8, text_column_name: str = "text") -> Dataset:
    """Parse the medical entities annotation and convert to GLiNER2 text+entities format."""

    def process_func(example):
        text = example[text_column_name] or ""
        try:
            parsed = json.loads(example["output"])
        except (json.JSONDecodeError, TypeError):
            parsed = []

        if isinstance(parsed, dict) and "entities" in parsed:
            medical_entities = parsed.get("entities", [])
        elif isinstance(parsed, list):
            medical_entities = parsed
        else:
            medical_entities = []

        entities_dict: dict[str, list[str]] = {}
        for med_ent in medical_entities:
            if not isinstance(med_ent, dict):
                continue
            label = med_ent.get("entity")
            # label = MEDICAL_ENTITY_MAP.get(label, label)
            span_text = (med_ent.get("text") or "").strip()

            # 1. Validation: valid label, non-empty span, and exact match in text
            if label not in MEDICAL_ENTITY_GROUPS or not span_text:
                continue
            if span_text not in text:
                continue

            # 2. Deduplicate spans per label
            if label not in entities_dict:
                entities_dict[label] = []
            if span_text not in entities_dict[label]:
                entities_dict[label].append(span_text)

        # 3. Only include descriptions for labels present in this example
        filtered_descriptions = {
            label: MEDICAL_ENTITY_DESCRIPTIONS[label]
            for label in entities_dict.keys()
            if label in MEDICAL_ENTITY_DESCRIPTIONS
        }

        return {
            "input": text,
            "output": {
                "entities": entities_dict,
                "entity_descriptions": filtered_descriptions,
            },
        }

    # 4. Remove all original columns to keep the dataset clean for GLiNER2 training
    return ds.map(process_func, num_proc=num_workers, remove_columns=ds.column_names)


def postprocess_medical_entities_v2_1(ds: Dataset, num_workers: int = 8, text_column_name: str = "text") -> Dataset:
    """Parse the medical entities annotation (v2.1 format) and convert to GLiNER2 text+entities format.
    
    v2.1 format: each entity group has a list of text spans
    {
        "entities": [
            {"entity": "disorders", "text": ["span1", "span2", ...]},
            {"entity": "anatomy", "text": ["span3", ...]},
            ...
        ]
    }
    """

    def process_func(example):
        text = example[text_column_name] or ""
        try:
            parsed = json.loads(example["output"])
        except (json.JSONDecodeError, TypeError):
            parsed = {"input": None}

        if isinstance(parsed, dict) and "entities" in parsed:
            medical_entities = parsed.get("entities", [])
        elif isinstance(parsed, list):
            medical_entities = parsed
        else:
            medical_entities = []

        entities_dict: dict[str, list[str]] = {}
        for med_ent in medical_entities:
            if not isinstance(med_ent, dict):
                continue
            label = med_ent.get("entity")
            text_spans = med_ent.get("text", [])

            # 1. Validation: valid label
            if label not in MEDICAL_ENTITY_GROUPS:
                continue

            # 2. Handle text as list of spans
            if isinstance(text_spans, str):
                text_spans = [text_spans]
            if not isinstance(text_spans, list):
                continue

            for span_text in text_spans:
                if not isinstance(span_text, str):
                    continue
                span_text = span_text.strip()
                if not span_text:
                    continue

                # 3. Validation: exact match in text
                if span_text not in text:
                    continue

                # 4. Deduplicate spans per label
                if label not in entities_dict:
                    entities_dict[label] = []
                if span_text not in entities_dict[label]:
                    entities_dict[label].append(span_text)

        # 5. Only include descriptions for labels present in this example
        filtered_descriptions = {
            label: MEDICAL_ENTITY_DESCRIPTIONS[label]
            for label in entities_dict.keys()
            if label in MEDICAL_ENTITY_DESCRIPTIONS
        }

        return {
            "input": text,
            "output": {
                "entities": entities_dict,
                "entity_descriptions": filtered_descriptions,
            },
        }

    # Remove all original columns to keep the dataset clean for GLiNER2 training
    ds = ds.map(process_func, num_proc=num_workers, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["input"], num_proc=num_workers)
    print(f"Filtered to {ds.num_rows:,d} examples after keeping only non-empty input")

    # deduplicate on input
    ds = deduplicate_dataset(ds, field_name="input", num_workers=num_workers)
    print(f"Deduplicated to {ds.num_rows:,d} examples")

    return ds


def main(
    input_path: str,
    output_path: str,
    task: Literal["topic", "quality", "medical_entities", "medical_entities_v2", "medical_entities_v2_1"],
    num_workers: int = 8,
    output_shard_size: int | None = None,
    test_split_size: int | None = None,
    text_column_name: str = "text",
):
    """Main function."""
    # ds = load_dataset(input_path, split="train")
    # print(f"Loaded {ds.num_rows:,d} examples")

    # from datasets import concatenate_datasets
    ds_paths = [
        # "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_next_80b_a3b_instruct_a",
        # "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_next_80b_a3b_instruct_b",
        # "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_235b_a22b_instruct_2507_fp8_c",
        # "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_medical_entities_extracted_qwen3_235b_a22b_instruct_2507_fp8_d",
        # "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p1",
        # "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p2",
        "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p1_reviewed_qwen3_235b_a22b_instruct_2507_fp8",
        "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p2_reviewed_qwen3_235b_a22b_instruct_2507_fp8",
        "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p3_reviewed_qwen3_235b_a22b_instruct_2507_fp8",
        "/lustre/fsn1/projects/rech/ilr/commun/corpus/fr/merged_sampled_500k_extracted_qwen3_235b_a22b_instruct_2507_fp8_p4_reviewed_qwen3_235b_a22b_instruct_2507_fp8",

    ]
    ds_list = [load_dataset(path, split="train") for path in ds_paths]
    ds = concatenate_datasets(ds_list)
    print(f"Concatenated {ds.num_rows:,d} examples")

    if task == "topic":
        ds = postprocess_topic(ds, num_workers)
    elif task == "quality":
        ds = postprocess_quality(ds, num_workers)
    elif task == "medical_entities":
        ds = postprocess_medical_entities(ds, num_workers, text_column_name=text_column_name)
    elif task == "medical_entities_v2":
        ds = postprocess_medical_entities_v2(ds, num_workers, text_column_name=text_column_name)
    elif task == "medical_entities_v2_1":
        ds = postprocess_medical_entities_v2_1(ds, num_workers, text_column_name=text_column_name)
    else:
        raise ValueError(f"Invalid task: {task}")

    if test_split_size is not None:
        ds = ds.train_test_split(test_size=test_split_size)
        root, ext = os.path.splitext(output_path)
        save_hf_dataset(ds["train"], f"{root}_train{ext}", output_shard_size)
        save_hf_dataset(ds["test"], f"{root}_test{ext}", output_shard_size)
    else:
        save_hf_dataset(ds, output_path, output_shard_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
