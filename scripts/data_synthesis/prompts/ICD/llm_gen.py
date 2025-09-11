# -*- coding: utf-8 -*-
"""
Simplified vLLM generator:
- Input: one JSON file containing an array of records with keys:
    {
      "definition_fr": "...",
      "labels": {"fr": "..."},
      "skos_notation": "CODE",
      "icd_code": "CODE"   # optional
    }
- Output: one .txt file per record in out_dir, e.g. CODE_wiki.txt or CODE_textbook.txt
- Prompts loaded from disk:
    - system_base.txt
    - system_base_style_wiki.txt
    - system_base_style_textbook.txt
    - user_template.txt
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict

from vllm import LLM, SamplingParams


# -------------------- Simple helpers --------------------

def load_json_records(path: str) -> List[Dict]:
    """Load a single JSON file containing an array of ICD records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of records.")
    return data

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_quotes(s: str) -> str:
    return (
        s.replace("’", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("«", '"')
         .replace("»", '"')
    )

def normalize_page(text: str) -> str:
    """Remove bullet markers/numbering, collapse blank lines, trim."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized = []
    for ln in lines:
        ln2 = re.sub(r"^\s*[-•\u2022]\s*", "", ln)
        ln2 = re.sub(r"^\s*\d+\s*[.)]\s*", "", ln2)
        normalized.append(ln2)
    page = "\n".join(normalized)
    page = re.sub(r"\n{3,}", "\n\n", page).strip()
    return page

def definition_present(page: str, definition_fr: str) -> bool:
    """Check verbatim presence (with whitespace/quote normalization)."""
    defn = re.sub(r"\s+", " ", normalize_quotes(definition_fr.strip()))
    pg = re.sub(r"\s+", " ", normalize_quotes(page))
    return defn in pg

def ensure_definition(page: str, definition_fr: str) -> str:
    """Ensure a Définition section contains the exact definition."""
    if not definition_fr:
        return page
    if definition_present(page, definition_fr):
        return page
    section = f"\n\nDéfinition\n{definition_fr}\n"
    patt = re.compile(r"(\A|\n\n)(Introduction[^\n]*\n)", flags=re.IGNORECASE)
    if patt.search(page):
        page = patt.sub(r"\1\2" + section, page, count=1)
    else:
        page = page.rstrip() + section
    return normalize_page(page)

def ensure_reference_top(page: str, title: str, skos: str) -> str:
    """
    Ensure first line == title and second line == 'Référence: CODE'.
    If title already first, insert reference below; else prepend both.
    """
    ref_line = f"Référence: {skos}" if skos else ""
    if not title and not ref_line:
        return page
    lines = page.split("\n")
    idx = next((i for i, l in enumerate(lines) if l.strip()), 0) if lines else 0
    first = lines[idx].strip() if lines else ""
    second = lines[idx + 1].strip() if len(lines) > idx + 1 else ""

    need_title = (title and first != title)
    need_ref = (ref_line and second != ref_line)

    if not need_title and not need_ref:
        return page

    rest = "\n".join(lines[idx + (0 if need_title else 1) + (0 if need_ref else 1):]).strip()
    header = []
    if title:
        header.append(title)
    if ref_line:
        header.append(ref_line)
    rebuilt = "\n".join(header) + ("\n\n" + rest if rest else "")
    return normalize_page(rebuilt)

def sanitize_filename(base: str) -> str:
    safe = "".join(ch for ch in (base or "page") if ch.isalnum() or ch in ("-", "_", ".")).strip("._")
    return safe or "page"

def render_messages(system_base: str, style_suffix: str, user_tmpl: str,
                    label_fr: str, definition_fr: str, skos_notation: str):
    system_msg = system_base.rstrip() + "\n\n" + style_suffix.strip()
    user_msg = user_tmpl.format(
        label_fr=label_fr,
        definition_fr=definition_fr,
        skos_notation=skos_notation,
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# -------------------- Core generation --------------------

def build_prompts(records: List[Dict],
                  llm: LLM,
                  system_base: str,
                  style_wiki: str,
                  style_textbook: str,
                  user_template: str,
                  p_textbook: float,
                  rng: random.Random) -> List[str]:
    """
    Build model-ready prompts for all records (using chat template).
    Each record is annotated with _gen_style for filename suffix.
    """
    prompts = []
    for rec in records:
        label = (rec.get("labels") or {}).get("fr", "")
        definition = rec.get("definition_fr", "")
        skos = rec.get("skos_notation", "")
        style = "textbook" if rng.random() < p_textbook else "wiki"
        style_suffix = style_textbook if style == "textbook" else style_wiki

        rec["_gen_style"] = style
        rec["_label_fr"] = label
        rec["_definition_fr"] = definition
        rec["_skos_notation"] = skos

        messages = render_messages(system_base, style_suffix, user_template, label, definition, skos)
        formatted = llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)
    return prompts

def generate_pages(
    json_path: str,
    model_name_or_path: str,
    prompts_dir: str,
    out_dir: str,
    p_textbook: float = 0.5,
    batch_size: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    max_tokens: int = 8192,
    seed: int = 42,
    include_style_in_filename: bool = True,
    tensor_parallel_size: int = 1,
    max_examples: int | None = None,  # NEW
):
    rng = random.Random(seed)

    # Load records from one JSON file (array)
    records = load_json_records(json_path)
    total = len(records)

    # NEW: cap the number of examples
    if max_examples is not None:
        records = records[:max_examples]

    print(f"Loaded {total:,d} records from {json_path}; processing {len(records):,d} examples")

    # Load prompts
    system_base = read_text(os.path.join(prompts_dir, "system_prompt_augment_base.txt"))
    style_wiki = read_text(os.path.join(prompts_dir, "system_prompt_augment_ICD_wiki.txt"))
    style_textbook = read_text(os.path.join(prompts_dir, "system_prompt_augment_ICD_textbook.txt"))
    user_template = read_text(os.path.join(prompts_dir, "user_template_augment_ICD.txt"))

    # Init vLLM (tensor parallel supported)
    print("Starting vLLM engine...")
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,  # pass through to vLLM
    )
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        prompts = build_prompts(
            batch, llm, system_base, style_wiki, style_textbook, user_template,
            p_textbook=p_textbook, rng=rng
        )

        # Generate
        results = llm.generate(prompts, params)

        # Post-process and write
        for i, rec in enumerate(batch):
            text = results[i].outputs[0].text.strip()
            page = normalize_page(text)
            page = ensure_reference_top(page, title=rec.get("_label_fr", ""), skos=rec.get("_skos_notation", ""))
            page = ensure_definition(page, rec.get("_definition_fr", ""))

            base = rec.get("skos_notation") or rec.get("icd_code") or rec.get("_label_fr") or "page"
            base = sanitize_filename(base)
            if include_style_in_filename:
                base = f"{base}_{rec.get('_gen_style', 'wiki')}"
            out_path = os.path.join(out_dir, f"{base}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(page + "\n")

    dur = time.perf_counter() - t0
    print(f"Done in {time.strftime('%Hh%Mm%Ss', time.gmtime(dur))}. Files saved to {out_dir}")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Generate FR Wikipedia/Textbook-style pages from a single JSON array.")
    ap.add_argument("--json", required=True, help="Path to input JSON file (array of records).")
    ap.add_argument("--model", required=True, help="Path/name of local model for vLLM.")
    ap.add_argument("--prompts_dir", required=True, help="Directory with system_base.txt, style files, user_template.txt")
    ap.add_argument("--out_dir", default="pages_out", help="Output directory for .txt files.")
    ap.add_argument("--p_textbook", type=float, default=0.5, help="Probability to choose textbook style.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--min_p", type=float, default=0.05)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_style_in_filename", action="store_true")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel degree (e.g., number of GPUs).")
    ap.add_argument("--max_examples", type=int, default=None, help="Limit the number of examples to process.")  # NEW
    args = ap.parse_args()

    generate_pages(
        json_path=args.json,
        model_name_or_path=args.model,
        prompts_dir=args.prompts_dir,
        out_dir=args.out_dir,
        p_textbook=args.p_textbook,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        include_style_in_filename=args.include_style_in_filename,
        tensor_parallel_size=args.tensor_parallel_size,
        max_examples=args.max_examples,  # NEW
    )

if __name__ == "__main__":
    main()