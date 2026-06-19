"""
Use a local vllm engine to generate data.

System prompt versions:
- v1: Predefined style list (~20 styles). Randomly selects eligible styles from shuffled style blocks.
      Output: eligible_styles, selected_style, rewritten_text, title
- v2: Same style-based approach as v1, but adds explicit noise filtering and "Density & Volume Protocol"
      for lossless medical compression. Falls back to generic_dense_compression if no style fits.
- v3: Parameterized style dimensions (register, abbreviation, structure, verbosity, noise) sampled
      externally and injected via template. Adds is_rewritable check and format detection.
      Output: is_rewritable, format, rewritten_text, title
- v4 (MGA): Two-stage genre/audience approach. Stage 1 generates (genre, audience) pairs,
      Stage 2 reformulates per pair. Style emerges from genre+audience naturally.
      Stage 1 output: proposals (list of genre/audience pairs)
      Stage 2 output: plain text
- v3_1: Cleaned v3 + view dim. Drops format detection, noise, structure, verbosity, title
      (structure/verbosity overlap with view; noise inverts negation; format prevents multi-sample
      diversity). Adds view dim (entity_centric | temporal | causal | decision | natural) as
      orthogonal information-organizing axis, plus soft entity-emphasis instruction and markdown/
      bullet ban. Style dims: register × abbreviation × view (Option B clinical-note-boosted weights).
      Output: is_rewritable, rewritten_text
- v3_2: v3_1 + verbosity dim restored (compressed | standard | expanded) to address v3_1's
      over-compression problem (mean output 183 words vs v3's 215; 5.24% <50-word outputs vs
      v3's 2.10%). Verbosity weights skew toward standard/expanded (0.20/0.45/0.35) to push the
      model away from pathological compression. Style dims: register × abbreviation × verbosity
      × view = 90 combinations (vs v3_1's 30, v3's 144). Same output schema as v3_1.
- v4_1 (MGA constrained): Two-stage like v4, but stage 1 uses Pydantic Literal enums for genre
      (13 options) and audience (7 options) — guided decoding enforces closed-list membership,
      eliminating v4's 11K-genre fragmentation. Stage 2 uses density-focused v3_1-style prompt
      (no expert-writer role) and overlays register × abbreviation sampled per call to break
      per-genre template collapse.
      Stage 1 output: proposals (list of genre/audience pairs, from closed lists)
      Stage 2 output: plain text
"""

import glob
import json
import os
import random
import re
import time
import zipfile
from pathlib import Path
from typing import Literal

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# Regex to match JSON objects (up to 3 levels of nested braces) — duplicated from postprocess_extract.py
# to avoid cross-module import. Used as a fallback when raw JSON parsing fails (e.g., when the LLM output
# has a reasoning prefix like gpt-oss harmony format: "analysis...assistantfinal{json}").
_JSON_OBJECT_PATTERN = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text using regex. Returns the last occurrence if multiple found."""
    matches = _JSON_OBJECT_PATTERN.findall(text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None

# v1, v2 prompt
# class RewritingOutput(BaseModel):
#     eligible_styles: list[str | None]
#     selected_style: str | None
#     rewritten_text: str
#     title: str

# v3 prompt
class RewritingOutput(BaseModel):
    is_rewritable: bool
    format: str | None
    rewritten_text: str | None
    title: str | None


# v3_1 prompt (cleaned style dims, view dim added, no format/title)
class RewritingOutputV3_1(BaseModel):
    is_rewritable: bool
    rewritten_text: str | None


# Evaluation output schema
class RewritingEvalOutput(BaseModel):
    reasoning: str
    factuality: int
    faithfulness: int
    style_adherence: int


# v4 (MGA) stage 1 output schema
class GenreAudiencePair(BaseModel):
    genre: str
    audience: str


class MGAStage1Output(BaseModel):
    reasoning: str
    has_medical_content: bool
    proposals: list[GenreAudiencePair]


# Official MGA stage 1 output schema (Appendix E.2, Prompt 3) — 5 hardcoded pairs, flat keys.
class MGAStage1OutputOfficial(BaseModel):
    audience_1: str
    genre_1: str
    audience_2: str
    genre_2: str
    audience_3: str
    genre_3: str
    audience_4: str
    genre_4: str
    audience_5: str
    genre_5: str


# v4_1 (MGA with curated closed lists) stage 1 output schema
MedicalGenreV4_1 = Literal[
    # Clinical documentation
    "clinical_note",
    "discharge_summary",
    "case_report",
    "consultation_letter",
    # Diagnostics
    "diagnostic_report",
    "prescription",
    # Regulatory / pharmaceutical
    "drug_information_sheet",
    "patient_leaflet",
    # Research / education
    "research_abstract",
    "clinical_guideline",
    "patient_education",
    "medical_qa",
    # Escape
    "natural",
]

MedicalAudienceV4_1 = Literal[
    "medical_specialist",
    "general_practitioner",
    "nurse_or_allied_health",
    "medical_student",
    "researcher",
    "patient_or_layperson",
    "public_health_official",
]


class GenreAudiencePairV4_1(BaseModel):
    genre: MedicalGenreV4_1
    audience: MedicalAudienceV4_1


class MGAStage1OutputV4_1(BaseModel):
    reasoning: str
    has_medical_content: bool
    proposals: list[GenreAudiencePairV4_1]


OUTPUT_SCHEMAS = {
    "rewriting": RewritingOutput.model_json_schema(),
    "rewriting_v3_1": RewritingOutputV3_1.model_json_schema(),
    # v3_2 has identical output schema to v3_1 (is_rewritable + rewritten_text);
    # only style sampling differs. Registered as a separate name for traceability.
    "rewriting_v3_2": RewritingOutputV3_1.model_json_schema(),
    "rewriting_eval": RewritingEvalOutput.model_json_schema(),
}


def get_output_schema(schema_name: str, n_stage1_pairs: int = 3) -> dict | None:
    """Get JSON schema for structured output, with dynamic constraints for MGA stage 1."""
    if schema_name == "mga_stage1":
        schema = MGAStage1Output.model_json_schema()
        # Enforce maxItems so guided decoding caps proposals at n_stage1_pairs.
        # minItems defaults to 0 (allows empty proposals when has_medical_content=False).
        schema["properties"]["proposals"]["maxItems"] = n_stage1_pairs
        return schema
    if schema_name == "mga_stage1_v4_1":
        schema = MGAStage1OutputV4_1.model_json_schema()
        schema["properties"]["proposals"]["maxItems"] = n_stage1_pairs
        return schema
    if schema_name == "mga_stage1_official":
        # Faithful to MGA paper Appendix E.2 (Prompt 3) — flat keys audience_1, genre_1, ..., audience_5, genre_5.
        return MGAStage1OutputOfficial.model_json_schema()
    return OUTPUT_SCHEMAS.get(schema_name)


def default_schema_for(rewriting_version: str, stage: int | None) -> str | None:
    """Deterministic schema selector from (rewriting_version, stage).

    Returns None when no structured output is needed (V4/V4.1 stage 2 produce plain text).
    Used as default when --json_schema is not passed explicitly.
    """
    if stage is None:
        if rewriting_version == "v3":
            return "rewriting"
        if rewriting_version == "v3_1":
            return "rewriting_v3_1"
        if rewriting_version == "v3_2":
            return "rewriting_v3_2"
    elif stage == 1:
        if rewriting_version == "v4":
            return "mga_stage1"
        if rewriting_version == "v4_1":
            return "mga_stage1_v4_1"
        if rewriting_version == "v4_2":
            # V4.2 reuses V4's open-string schema; diversity is enforced by prompt + sampling.
            return "mga_stage1"
        if rewriting_version == "mga_official":
            # Official MGA per arXiv:2502.04235: genre/audience are sentence-level
            # natural-language descriptions; uses paper's flat-key JSON schema
            # (audience_1, genre_1, ..., audience_5, genre_5) instead of our nested proposals list.
            return "mga_stage1_official"
    # stage == 2 (v4 / v4_1 / v4_2 / mga_official) → plain text, no schema
    return None


# Style sampling configuration with default probabilities
STYLE_OPTIONS = {
    "register": {
        "options": ["formal", "telegraphic"],
        "weights": [0.5, 0.5],
    },
    "abbreviation": {
        "options": ["expanded", "moderate", "heavy"],
        "weights": [0.3, 0.4, 0.3],
    },
    "structure": {
        "options": ["prose", "bullets", "key_value", "hybrid"],
        "weights": [0.6, 0.1, 0.1, 0.2],
    },
    "verbosity": {
        "options": ["compressed", "standard", "expanded"],
        "weights": [0.2, 0.6, 0.2],
    },
    "noise": {
        "options": ["clean", "realistic"],
        "weights": [0.7, 0.3],
    },
}


def _normalize_stage1_proposals(raw: str) -> list[dict] | None:
    """Parse stage 1 JSON output and normalize to a list of {"genre": str, "audience": str} dicts.

    Tries direct json.loads first, falls back to regex-based extract_json_from_text for outputs
    with reasoning prefixes (e.g., gpt-oss harmony format: "analysis...assistantfinal{json}").

    Handles two schema formats:
    - V4 / V4.1 / V4.2 nested: {"reasoning", "has_medical_content", "proposals": [{"genre", "audience"}]}
    - mga_official flat: {"audience_1", "genre_1", ..., "audience_5", "genre_5"}

    Returns None to signal the document should be skipped (parse error, non-medical, or empty proposals).
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = _extract_json_from_text(raw)
        if parsed is None:
            return None
    if "proposals" in parsed:
        # Nested-list format (V4 / V4.1 / V4.2)
        if not parsed.get("has_medical_content", False):
            return None
        pairs = parsed.get("proposals", [])
        return pairs if pairs else None
    # Flat-key official MGA format — extract audience_N / genre_N pairs in order
    pairs = []
    for i in range(1, 100):  # generous upper bound
        a_key, g_key = f"audience_{i}", f"genre_{i}"
        if a_key in parsed and g_key in parsed:
            pairs.append({"genre": parsed[g_key], "audience": parsed[a_key]})
        else:
            break
    return pairs if pairs else None


def explode_stage1_output(dataset: Dataset, output_column_name: str = "output") -> Dataset:
    """Parse stage 1 JSON output and explode: one row per (genre, audience) pair.

    Filters out non-medical documents and parse failures.
    """
    exploded = []
    skipped_parse_or_empty = 0
    for row in dataset:
        raw = row[output_column_name]
        pairs = _normalize_stage1_proposals(raw)
        if pairs is None:
            skipped_parse_or_empty += 1
            continue
        for pair in pairs:
            new_row = {**row}
            new_row["genre"] = pair["genre"]
            new_row["audience"] = pair["audience"]
            exploded.append(new_row)
    print(f"Exploded {len(dataset):,d} documents → {len(exploded):,d} rows")
    print(f"  Skipped: {skipped_parse_or_empty} (parse error / non-medical / empty proposals)")
    return Dataset.from_list(exploded)


def sample_stage1_output(
    dataset: Dataset,
    output_column_name: str = "output",
    k: int = 1,
    seed: int = 0,
) -> Dataset:
    """Parse stage 1 JSON output and sample k (genre, audience) pairs per doc without replacement.

    Used by V4.2: 1 doc → k rows. Per-doc cap = min(k, len(proposals)).
    k=1: corpus stays same size as input (max distribution control).
    k>1: corpus grows by ~k× (more rewrites per source); equivalent to explode when k >= max n_stage1_pairs.
    """
    rng = random.Random(seed)
    sampled = []
    skipped_parse_or_empty = 0
    for row in dataset:
        raw = row[output_column_name]
        pairs = _normalize_stage1_proposals(raw)
        if pairs is None:
            skipped_parse_or_empty += 1
            continue
        n_sample = min(k, len(pairs))
        chosen = rng.sample(pairs, n_sample)
        for pair in chosen:
            new_row = {**row}
            new_row["genre"] = pair["genre"]
            new_row["audience"] = pair["audience"]
            new_row["sampled_from_n"] = len(pairs)
            sampled.append(new_row)
    print(f"Sampled {len(dataset):,d} documents → {len(sampled):,d} rows (k={k} of N proposals per doc, uniform without replacement)")
    print(f"  Skipped: {skipped_parse_or_empty} (parse error / non-medical / empty proposals)")
    return Dataset.from_list(sampled)


def sample_rewriting_style() -> dict[str, str]:
    """Sample a style combination from all dimensions."""
    return {
        dim: random.choices(cfg["options"], weights=cfg["weights"], k=1)[0]
        for dim, cfg in STYLE_OPTIONS.items()
    }


# v3_1: cleaned style dims + view dim (Option B clinical-note-boosted weights)
# Dropped: structure (overlaps with register/verbosity, bullets was LLM artifact)
# Dropped: noise (harmful — inverts negation per prior doc analysis)
# Dropped: verbosity (overlaps with view)
# Added: view (information organizing principle — orthogonal to surface style)
STYLE_OPTIONS_V3_1 = {
    "register": {
        "options": ["formal", "telegraphic"],
        "weights": [0.5, 0.5],
    },
    "abbreviation": {
        "options": ["expanded", "moderate", "heavy"],
        "weights": [0.3, 0.4, 0.3],
    },
    "view": {
        "options": ["entity_centric", "temporal", "causal", "decision", "natural"],
        # clinical-note-boosted — entity_centric + temporal match real clinical text distribution
        "weights": [0.28, 0.22, 0.18, 0.18, 0.14],
    },
}


def sample_rewriting_style_v3_1() -> dict[str, str]:
    """Sample a v3_1 style combination (register × abbreviation × view)."""
    return {
        dim: random.choices(cfg["options"], weights=cfg["weights"], k=1)[0]
        for dim, cfg in STYLE_OPTIONS_V3_1.items()
    }


# v3_2: v3_1 + verbosity dim restored to address over-compression observed in v3_1
# (mean 183 words vs v3's 215; 5.24% <50-word outputs vs v3's 2.10%).
# verbosity directly controls how much surrounding context the model keeps per entity.
# = 2 × 3 × 3 × 5 = 90 combinations (vs v3_1's 30 and v3's 144).
STYLE_OPTIONS_V3_2 = {
    "register": {
        "options": ["formal", "telegraphic"],
        "weights": [0.5, 0.5],
    },
    "abbreviation": {
        "options": ["expanded", "moderate", "heavy"],
        "weights": [1 / 3, 1 / 3, 1 / 3],
    },
    "verbosity": {
        "options": ["compressed", "standard", "expanded"],
        # Bias toward standard/expanded to counter v3_1's over-compression tendency.
        # compressed is still kept (~20%) — useful for high-density telegraphic outputs.
        "weights": [0.20, 0.45, 0.35],
    },
    "view": {
        "options": ["entity_centric", "temporal", "causal", "decision", "natural"],
        # Same Option B weights as v3_1 — clinical-note-boosted.
        "weights": [0.28, 0.22, 0.18, 0.18, 0.14],
    },
}


def sample_rewriting_style_v3_2() -> dict[str, str]:
    """Sample a v3_2 style combination (register × abbreviation × verbosity × view)."""
    return {
        dim: random.choices(cfg["options"], weights=cfg["weights"], k=1)[0]
        for dim, cfg in STYLE_OPTIONS_V3_2.items()
    }


# v4_1: stage 2 style overlay applied on top of (genre, audience) from stage 1.
# View is not included here because `genre` drives the organizing principle.
STYLE_OPTIONS_V4_1 = {
    "register": {
        "options": ["formal", "telegraphic"],
        "weights": [0.5, 0.5],
    },
    "abbreviation": {
        "options": ["expanded", "moderate", "heavy"],
        "weights": [1 / 3, 1 / 3, 1 / 3],
    },
}


def sample_rewriting_style_v4_1() -> dict[str, str]:
    """Sample a v4_1 stage 2 style overlay (register × abbreviation)."""
    return {
        dim: random.choices(cfg["options"], weights=cfg["weights"], k=1)[0]
        for dim, cfg in STYLE_OPTIONS_V4_1.items()
    }


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
        parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if parquet_files:
            # Treat directory of parquet files as a parquet dataset
            return load_dataset("parquet", data_files=parquet_files, split="train")
        dataset_info_path = os.path.join(input_path, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            # Load dataset saved with `Dataset.save_to_disk`
            return load_from_disk(input_path)
        # Fallback to loading as a local dataset directory
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


def shuffle_styles_in_prompt(
    prompt: str,
    start_tag: str = "<styles_definition>",
    end_tag: str = "</styles_definition>",
    sep: str = "\n\n",
) -> str:
    """Randomly select and shuffle style blocks inside the <styles_definition> section.

    Selects one style from the primary groups and a random number of styles from the rest.
    If no <styles_definition> section is found, returns the prompt unchanged.
    """
    pattern = rf"({re.escape(start_tag)}\s*)([\s\S]*?)(\s*{re.escape(end_tag)})"
    match = re.search(pattern, prompt)
    if not match:
        return prompt

    tag_open, body, tag_close = match.groups()

    # Split inner body into blocks
    blocks = body.strip(sep).split(sep)

    # If nothing to shuffle, return as-is
    if len(blocks) <= 1:
        return prompt

    target_prefixes = [
        "rapid_telegraphic_note",
        "clinical_report_formal",
        "structured_ehr_export",
        "dense_scientific_abstract_or_methods",
    ]

    # Initialize groups
    grouped_blocks = {prefix: [] for prefix in target_prefixes}
    rest_blocks = []

    for b in blocks:
        stripped_b = b.strip()
        matched = False
        for prefix in target_prefixes:
            if stripped_b.startswith(prefix):
                grouped_blocks[prefix].append(b)
                matched = True
                break
        if not matched:
            rest_blocks.append(b)

    selected_blocks = []
    
    # Pick one from each target group if available
    for prefix in target_prefixes:
        if grouped_blocks[prefix]:
            selected_blocks.append(random.choice(grouped_blocks[prefix]))

    # Pick 1 to len(rest_blocks) from the rest if available
    if rest_blocks:
        k = random.randint(1, len(rest_blocks))
        selected_blocks.extend(random.sample(rest_blocks, k))

    # Shuffle the combined selection
    random.shuffle(selected_blocks)

    new_body = sep.join(selected_blocks)
    new_section = f"{tag_open}{new_body}{tag_close}"
    return re.sub(pattern, new_section, prompt, count=1)


def process_batch(
    batch: list[dict],
    llm: LLM,
    params: SamplingParams,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    text_column_name: str = "text",
    output_column_name: str = "output",
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    max_model_len: int | None = None,
    language: str = "the input language",
    stage: int | None = None,
    n_stage1_pairs: int = 3,
    rewriting_version: str = "v3",
) -> list[dict]:
    """Process a batch of data using local vllm engine."""

    tokenizer = llm.get_tokenizer()
    max_text_tokens = max_model_len // 2 if max_model_len is not None else None

    prompts = []
    for item in batch:
        text = item[text_column_name]

        # Truncate text using actual tokenization
        if max_text_tokens is not None:
            text_token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(text_token_ids) > max_text_tokens:
                text_token_ids = text_token_ids[:max_text_tokens]
                text = tokenizer.decode(text_token_ids, skip_special_tokens=False)

        # Build the user message
        user_prompt_text = text
        if user_prompt:
            user_prompt_text = user_prompt.format(text=user_prompt_text)
            # tmp: eval rewriting
            # user_prompt_text = user_prompt.format(original_text=user_prompt_text, rewriting_text=item["text"], rewriting_style=item["rewriting_style"])
        messages = [{"role": "user", "content": user_prompt_text}]

        # Add system message
        if system_prompt:
            if stage == 1:
                # MGA stage 1: format with n_stage1_pairs
                styled_system_prompt = system_prompt.format(n_stage1_pairs=n_stage1_pairs, language=language)
            elif stage == 2:
                if rewriting_version in ("v4_1", "v4_2"):
                    # V4.1/V4.2 stage 2: genre + audience from stage 1, register/abbreviation sampled fresh
                    # (V4.2 reuses V4.1's stage-2 prompt; only stage 1 differs — open vs closed lists)
                    v4_1_style = sample_rewriting_style_v4_1()
                    styled_system_prompt = system_prompt.format(
                        genre=item["genre"],
                        audience=item["audience"],
                        language=language,
                        **v4_1_style,
                    )
                    item["rewriting_style"] = v4_1_style
                else:
                    # V4 stage 2: format with genre/audience from item
                    styled_system_prompt = system_prompt.format(
                        genre=item["genre"],
                        audience=item["audience"],
                        language=language,
                    )
            else:
                # tmp: eval rewriting
                # styled_system_prompt = system_prompt

                # tmp: v1, v2: shuffle predefined style blocks
                # styled_system_prompt = shuffle_styles_in_prompt(
                #     system_prompt, start_tag="<styles_definition>", end_tag="</styles_definition>", sep="\n\n"
                # )

                # tmp: v3 (default) / v3_1 / v3_2: sample style dimensions and inject into prompt template
                if rewriting_version == "v3_2":
                    rewriting_style = sample_rewriting_style_v3_2()
                elif rewriting_version == "v3_1":
                    rewriting_style = sample_rewriting_style_v3_1()
                else:
                    rewriting_style = sample_rewriting_style()
                styled_system_prompt = system_prompt.format(**rewriting_style, language=language)
                item["rewriting_style"] = rewriting_style
            messages.insert(0, {"role": "system", "content": styled_system_prompt})

        # apply chat template
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        formatted_prompt = tokenizer.apply_chat_template(messages, **chat_template_kwargs)

        # this truncation cut off chat template suffix, which is not what we want
        # def _format_and_truncate(messages, max_model_len=None, enable_thinking=False):
        #     # first build ids via the chat template
        #     token_ids = tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=True,
        #         add_generation_prompt=True,
        #         enable_thinking=enable_thinking,
        #     )
        #     if max_model_len is not None and len(token_ids) > max_model_len:
        #         print(f"Truncating prompt from {len(token_ids)} tokens to {max_model_len // 2} tokens")
        #         # todo: keep head or throw
        #         token_ids = token_ids[:max_model_len // 2]
        #     # re-decode to a string prompt that vLLM will re-tokenize (same template)
        #     return tokenizer.decode(token_ids, skip_special_tokens=False)

        # formatted_prompt = _format_and_truncate(messages, max_model_len, enable_thinking)

        prompts.append(formatted_prompt)

    # todo: llm.chat
    outputs = llm.generate(prompts, params)

    if len(outputs) != len(batch):
        raise RuntimeError(f"Number of outputs ({len(outputs)}) does not match batch size ({len(batch)}).")

    for item, out, prompt in zip(batch, outputs, prompts):
        # item[f"original_{text_column_name}"] = item[text_column_name]
        item[output_column_name] = out.outputs[0].text.strip()
        gen_configs = {
            # "model": llm.llm_engine.model_config.model,
            "model": llm.llm_engine.model_config.model.replace("/lustre/fswork/projects/rech/ilr/commun/pretrained/models/", ""),
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "max_tokens": params.max_tokens,
            # "prompt": prompt,  # debug
        }
        if stage == 2:
            gen_configs["genre"] = item.get("genre")
            gen_configs["audience"] = item.get("audience")
            if rewriting_version in ("v4_1", "v4_2"):
                gen_configs["rewriting_style"] = item.pop("rewriting_style", None)
        elif stage is None:
            gen_configs["rewriting_style"] = item.pop("rewriting_style", None)
        item["gen_configs"] = gen_configs

    return batch


# Generate outputs, update dataset in batches, and overwrite checkpoint
def process_dataset(
    dataset: Dataset,
    llm: LLM,
    params: SamplingParams,
    output_path: str,
    batch_size: int = 512,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    text_column_name: str = "text",
    output_column_name: str = "output",
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    max_model_len: int | None = None,
    language: str = "the input language",
    stage: int | None = None,
    n_stage1_pairs: int = 3,
    rewriting_version: str = "v3",
):
    """Process dataset in batches. Skips batch `i` if its output parquet already exists
    (deterministic resume: same seed/batch_size/dataset ⇒ same rows → identical content).
    """

    # Calculate total number of batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    def _batch_path(i: int) -> str:
        if output_path.endswith(".parquet"):
            return f"{output_path[:-8]}_{i:09d}.parquet"
        return f"{output_path}/{i:09d}.parquet"

    n_skipped = 0
    for i in tqdm(range(num_batches)):
        out_file = _batch_path(i)
        if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
            n_skipped += 1
            continue

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
            output_column_name=output_column_name,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            max_model_len=max_model_len,
            language=language,
            stage=stage,
            n_stage1_pairs=n_stage1_pairs,
            rewriting_version=rewriting_version,
        )

        shard_ds = Dataset.from_list(processed_records)
        shard_ds.to_parquet(out_file)

    if n_skipped:
        print(f"[process_dataset] skipped {n_skipped}/{num_batches} batches (resume — output already on disk)")


# Main function to control workflow
def main(
    dataset_path: str,
    model_name_or_path: str,
    # i/o
    output_path: str,
    system_prompt_file: str | None = None,
    user_prompt_file: str | None = None,
    text_column_name: str = "text",
    output_column_name: str = "output",
    # load params
    # dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    gpu_memory_utilization: float = 0.9,
    async_scheduling: bool | None = None,
    kv_cache_dtype: str | None = None,
    tokenizer_mode: str | None = None,
    config_format: str | None = None,
    load_format: str | None = None,
    reasoning_parser: str | None = None,
    speculative_config: dict | None = None,
    disable_custom_all_reduce: bool | None = None,
    enforce_eager: bool | None = None,
    # infer params
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    # repetition_penalty: float = 1.0,
    json_schema: str | None = None,
    # chat template params
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    # run params
    batch_size: int = 512,
    shuffle: bool = False,
    seed: int = 42,
    start_idx: int = 0,
    max_samples: int | None = None,
    language: str = "the input language",
    # MGA stage params
    stage: int | None = None,  # None=v3 style-based, 1=MGA stage1, 2=MGA stage2
    n_stage1_pairs: int = 3,          # stage 1: max number of (genre, audience) pairs to generate per doc
    n_stage2_pairs: int = 1,         # stage 2 (V4.2 only): number of pairs to sample per doc from stage 1 proposals
    # rewriting version selector (only used when stage is None)
    rewriting_version: str = "v3",  # "v3" | "v3_1" | "v3_2" | "v4" | "v4_1" | "v4_2" | "mga_official" — drives schema selection and stage 2 style overlay
):
    # seed for reproducibility of topic shuffling
    random.seed(seed)

    # load dataset
    dataset = load_local_datasets(dataset_path)
    print(f"Loaded {dataset.num_rows:,d} examples")

    # tmp: eval rewriting
    # dataset = dataset.sort("medical_entity_density", reverse=True)
    # dataset = dataset.select(range(5000))
    # print(f"Sampled {dataset.num_rows:,d} examples with highest medical entity density")

    # OLD hard-coded prefilter:
    # def filter_func(example):
    #     return example["edu_quality_normalized_score"] >= 1 and example["medical_entity_density"] >= 0.05
    # dataset = dataset.filter(filter_func, num_proc=32)
    # print(f"Filtered to {dataset.num_rows:,d} examples")

    # tmp pre-filter (approx has_medical_content via cheap upstream annotations)
    # density>=0.01 AND edu>=1 recovers ~89% of has_medical=True at ~23% LLM-cost saving (see docs/17).
    if stage != 2:  # only filter raw source on stage-1 / single-stage; stage-2 input is already filtered
        def filter_func(example):
            return example["edu_quality_normalized_score"] >= 1 and example["medical_entity_density"] >= 0.01
        before = len(dataset)
        dataset = dataset.filter(filter_func, num_proc=32)
        print(f"Prefiltered: kept {len(dataset):,d}/{before:,d} ({100*len(dataset)/before:.1f}%)")

    # Stage 2 preprocessing: collapse stage 1 output to (genre, audience) per row.
    # V4/V4.1: explode (1 doc → N rows, one per proposal) — corpus inflates by N.
    # V4.2 / mga_official: uniform sample-without-replacement (1 doc → n_stage2_pairs rows from N proposals).
    #       n_stage2_pairs=1 keeps corpus size; n_stage2_pairs>1 grows it by ~k× (capped per-doc by len(proposals)).
    #       mga_official uses paper-exact stage-1 (5 pairs), then samples k=1 of 5 for stage-2 training.
    if stage == 2 and "genre" not in dataset.column_names:
        if rewriting_version in ("v4_2", "mga_official"):
            dataset = sample_stage1_output(dataset, output_column_name, k=n_stage2_pairs, seed=seed)
        else:
            dataset = explode_stage1_output(dataset, output_column_name)

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
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        print(f"Using max num batched tokens: {max_num_batched_tokens}")
    if async_scheduling is not None:
        llm_kwargs["async_scheduling"] = async_scheduling
        print(f"Using async scheduling: {async_scheduling}")
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config
        print(f"Using speculative config: {speculative_config}")
    if kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
        print(f"Using kv cache dtype: {kv_cache_dtype}")
    if tokenizer_mode is not None:
        llm_kwargs["tokenizer_mode"] = tokenizer_mode
        print(f"Using tokenizer mode: {tokenizer_mode}")
    if config_format is not None:
        llm_kwargs["config_format"] = config_format
        print(f"Using config format: {config_format}")
    if load_format is not None:
        llm_kwargs["load_format"] = load_format
        print(f"Using load format: {load_format}")
    if reasoning_parser is not None:
        llm_kwargs["reasoning_parser"] = reasoning_parser
        print(f"Using reasoning parser: {reasoning_parser}")
    if disable_custom_all_reduce is not None:
        llm_kwargs["disable_custom_all_reduce"] = disable_custom_all_reduce
        print(f"Using disable custom all reduce: {disable_custom_all_reduce}")
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager
        print(f"Using enforce eager: {enforce_eager}")
    llm = LLM(model=model_name_or_path, **llm_kwargs)

    # gen params
    # json_schema is an optional override; if not set, derive from (rewriting_version, stage).
    # Default resolution returns None for v4/v4_1 stage 2 (plain-text output).
    resolved_schema_name = json_schema or default_schema_for(rewriting_version, stage)
    structured_outputs = None
    if resolved_schema_name is not None:
        output_json_schema = get_output_schema(resolved_schema_name, n_stage1_pairs=n_stage1_pairs)
        structured_outputs = StructuredOutputsParams(json=output_json_schema)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        # repetition_penalty=repetition_penalty,
        # stop_token_ids=stop_token_ids,
        structured_outputs=structured_outputs,
    )

    # load prompts
    # system prompt
    system_prompt = load_prompt(system_prompt_file) if system_prompt_file else None
    # user prompt
    user_prompt = load_prompt(user_prompt_file) if user_prompt_file else None

    start_time = time.perf_counter()

    process_dataset(
        dataset,
        llm,
        params,
        output_path,
        batch_size=batch_size,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        text_column_name=text_column_name,
        output_column_name=output_column_name,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_model_len=max_model_len,
        language=language,
        stage=stage,
        n_stage1_pairs=n_stage1_pairs,
        rewriting_version=rewriting_version,
    )

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}.\n"
        f"Generated data is saved in {output_path}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
