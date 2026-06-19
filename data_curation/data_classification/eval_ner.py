"""Evaluate NER metrics for medical entities."""

import fire
import os
import json
import math
from datasets import Dataset, load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

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
#     "phenomena": "Natural or human-caused phenomena or processes relevant to health context",
#     "physiology": "Normal biological or mental functions and processes at organism, system, cellular, or molecular levels",
#     # "living_beings": "Organisms and population groups in clinical context, including patients and pathogenic or experimental organisms",
#     # low density
#     # "geographic_areas": "Named physical locations and care settings or units relevant to clinical or epidemiologic context",
#     # "organizations": "Administrative or institutional entities in health care, public health, research, or professional coordination",
#     # "occupations": "Professional roles and biomedical or health-related disciplines",
#     # "concepts": "Abstract entities such as documents, standards, guidelines, regulations, or intellectual products",
#     # "activities": "Human behaviors and routine activities relevant to health status, risk, adherence, or lifestyle",
#     # "objects": "Inanimate physical objects not classified as medical devices, including consumables and non-medical manufactured items",
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


def _make_shard_path(base_path: str, idx: int) -> str:
    """Return a new path like 'file_00005.parquet' for shard index idx."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_{idx:09d}{ext}"


def load_local_dataset(input_path: str) -> Dataset:
    """Load local dataset based on file extension."""
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


def main(input_path: str):
    """
    Load dataset and compute NER metrics (Precision, Recall, F1) for medical entities.
    """
    dataset = load_local_dataset(input_path)

    target_labels = sorted(MEDICAL_ENTITIES.keys())
    
    # We will collect y_true and y_pred as multi-label indicator matrices
    # Each "instance" is a (doc_id, entity_string) pair
    y_true_all = []
    y_pred_all = []
    
    # Track counts per example for MAE/Bias calculation
    gold_counts = {label: [] for label in target_labels}
    pred_counts = {label: [] for label in target_labels}

    for row in tqdm(dataset, desc="Evaluating NER"):
        # Predictions from medical_entities column (predicted by GLiNER)
        preds_dict = row.get("medical_entities", {})
        if not isinstance(preds_dict, dict):
            preds_dict = {}

        # Ground truth from output.entities (provided in input data)
        output_data = row.get("output", {})
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except json.JSONDecodeError:
                output_data = {}

        golds_dict = output_data.get("entities", {}) if isinstance(output_data, dict) else {}
        if not isinstance(golds_dict, dict):
            golds_dict = {}

        # 1. Normalize and clean both dicts
        clean_preds = {}
        clean_golds = {}
        all_unique_entities_in_doc = set()

        for label in target_labels:
            # Normalize and deduplicate: lowercase + strip
            p_set = {str(p).strip().lower() for p in (preds_dict.get(label, []) or []) if p and str(p).strip()}
            g_set = {str(g).strip().lower() for g in (golds_dict.get(label, []) or []) if g and str(g).strip()}
            clean_preds[label] = p_set
            clean_golds[label] = g_set
            all_unique_entities_in_doc |= p_set
            all_unique_entities_in_doc |= g_set

        # 2. For each unique entity found in this document, check its labels in pred vs gold
        for ent in all_unique_entities_in_doc:
            true_labels = [1 if ent in clean_golds[label] else 0 for label in target_labels]
            pred_labels = [1 if ent in clean_preds[label] else 0 for label in target_labels]
            y_true_all.append(true_labels)
            y_pred_all.append(pred_labels)

        # 3. Track counts for this document
        for label in target_labels:
            gold_counts[label].append(len(clean_golds[label]))
            pred_counts[label].append(len(clean_preds[label]))

    # Print results
    print(f"\nNER Evaluation Results for: {input_path}")
    print("Metrics based on set-of-strings matching per entity class.")
    
    if not y_true_all:
        print("No entities found for evaluation.")
        return

    # Use classification_report for detailed per-class and averaged metrics
    # Note: for multi-label, it provides micro, macro, and weighted averages
    report = classification_report(
        y_true_all, 
        y_pred_all, 
        target_names=target_labels, 
        zero_division=0,
        digits=4
    )
    print(report)

    # 4. Compute count metrics (MAE, Bias) per example
    print("\nEntity Count Metrics (per example):")
    header = f"{'Label':<30} | {'Avg Gold':<10} | {'Avg Pred':<10} | {'MAE':<10} | {'Bias':<10}"
    print(header)
    print("-" * len(header))
    
    for label in target_labels:
        g_arr = gold_counts[label]
        p_arr = pred_counts[label]
        n = len(g_arr)
        if n == 0:
            continue
        avg_gold = sum(g_arr) / n
        avg_pred = sum(p_arr) / n
        mae = sum(abs(p - g) for p, g in zip(g_arr, p_arr)) / n
        bias = sum(p - g for g, p in zip(g_arr, p_arr)) / n
        print(f"{label:<30} | {avg_gold:<10.2f} | {avg_pred:<10.2f} | {mae:<10.2f} | {bias:<10.2f}")

    # Total entities count metrics
    all_labels_gold_counts = [sum(counts) for counts in zip(*(gold_counts[l] for l in target_labels))]
    all_labels_pred_counts = [sum(counts) for counts in zip(*(pred_counts[l] for l in target_labels))]
    if all_labels_gold_counts:
        n = len(all_labels_gold_counts)
        avg_gold = sum(all_labels_gold_counts) / n
        avg_pred = sum(all_labels_pred_counts) / n
        mae = sum(abs(p - g) for p, g in zip(all_labels_pred_counts, all_labels_gold_counts)) / n
        bias = sum(p - g for p, g in zip(all_labels_pred_counts, all_labels_gold_counts)) / n
        print("-" * len(header))
        print(f"{'TOTAL':<30} | {avg_gold:<10.2f} | {avg_pred:<10.2f} | {mae:<10.2f} | {bias:<10.2f}")


if __name__ == "__main__":
    fire.Fire(main)
