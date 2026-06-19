"""Train GLiNER2 on medical NER task."""

import os
import random

import fire
import torch
import torch.distributed as dist
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from sklearn.metrics import classification_report
from tqdm import tqdm

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
#     "phenomena": "Natural or human-caused phenomena or processes relevant to health context",
#     "living_beings": "Organisms and population groups in clinical context, including patients and pathogenic or experimental organisms",
#     # low density
#     "geographic_areas": "Named physical locations and care settings or units relevant to clinical or epidemiologic context",
#     "organizations": "Administrative or institutional entities in health care, public health, research, or professional coordination",
#     "occupations": "Professional roles and biomedical or health-related disciplines",
#     "concepts": "Abstract entities such as documents, standards, guidelines, regulations, or intellectual products",
#     "activities": "Human behaviors and routine activities relevant to health status, risk, adherence, or lifestyle",
#     "objects": "Inanimate physical objects not classified as medical devices, including consumables and non-medical manufactured items",
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


def augment_example(example: InputExample) -> list[InputExample]:
    """Create augmented versions of an example."""
    augmented = [example]  # Original
    
    # Shuffle entity order
    if len(example.entities) > 1:
        items = list(example.entities.items())
        random.shuffle(items)
        shuffled_entities = dict(items)
        augmented.append(InputExample(
            text=example.text,
            entities=shuffled_entities
        ))
    
    return augmented

def compute_metrics(model, eval_dataset, batch_size=8, use_description=True):
    """Custom metric computation function for medical entities."""
    target_labels = sorted(MEDICAL_ENTITIES.keys())
    # Use full dict with descriptions or just label keys
    entity_schema = MEDICAL_ENTITIES if use_description else list(MEDICAL_ENTITIES.keys())

    y_true_all = []
    y_pred_all = []

    # Track counts per example for MAE/Bias calculation
    gold_counts = {label: [] for label in target_labels}
    pred_counts = {label: [] for label in target_labels}

    # 1. Collect all texts and gold entities from eval_dataset
    # ExtractorDataset.__getitem__ returns (text, schema/output)
    texts = []
    all_golds = []
    for i in tqdm(range(len(eval_dataset)), desc="Preparing eval data"):
        text, output = eval_dataset[i]
        texts.append(text)

        # Ground truth entities are in 'entities' key or it is the dict itself
        all_golds.append(output.get("entities", output))

    # 2. Batch predict entities using the model
    model.eval()
    with torch.no_grad():
        try:
            # batch_extract_entities returns a list of results: [{"entities": {label: [str,...]}}, ...]
            preds = model.batch_extract_entities(
                texts,
                entity_schema,
                batch_size=batch_size,
                threshold=0.5,
            )
        except Exception as e:
            print(f"Error during batch_extract_entities: {e}")
            return {}

    # 3. Compare predictions with gold entities (same logic as eval_ner.py)
    for i in range(len(texts)):
        preds_dict = preds[i]["entities"]
        golds_dict = all_golds[i]

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
            
            # Track counts
            gold_counts[label].append(len(g_set))
            pred_counts[label].append(len(p_set))

        for ent in all_unique_entities_in_doc:
            true_labels = [1 if ent in clean_golds[label] else 0 for label in target_labels]
            pred_labels = [1 if ent in clean_preds[label] else 0 for label in target_labels]
            y_true_all.append(true_labels)
            y_pred_all.append(pred_labels)

    if not y_true_all:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "avg_gold": 0.0, "avg_pred": 0.0, "mae": 0.0, "bias": 0.0}

    # Use classification_report for detailed metrics
    report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=target_labels,
        zero_division=0,
        output_dict=True
    )

    # Compute aggregate count metrics for TOTAL across all labels
    all_labels_gold_counts = [sum(counts) for counts in zip(*(gold_counts[l] for l in target_labels))]
    all_labels_pred_counts = [sum(counts) for counts in zip(*(pred_counts[l] for l in target_labels))]
    
    avg_gold, avg_pred, mae, bias = 0.0, 0.0, 0.0, 0.0
    if all_labels_gold_counts:
        n = len(all_labels_gold_counts)
        avg_gold = sum(all_labels_gold_counts) / n
        avg_pred = sum(all_labels_pred_counts) / n
        mae = sum(abs(p - g) for p, g in zip(all_labels_pred_counts, all_labels_gold_counts)) / n
        bias = sum(p - g for p, g in zip(all_labels_pred_counts, all_labels_gold_counts)) / n

    # Return weighted averages and count metrics for logging
    return {
        "f1": report["weighted avg"]["f1-score"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        # "avg_gold_entities": avg_gold,
        "avg_pred_entities": avg_pred,
        "avg_mae_entities": mae,
        "avg_bias_entities": bias,
    }

def main(
    output_dir: str,
    train_data_path: str,
    validation_data_path: str | None = None,
    model_name_or_path: str = "fastino/gliner2-multi-v1",
    experiment_name: str = "gliner2",
    num_epochs: int = 10,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    encoder_lr: float = 1e-5,
    task_lr: float = 5e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "cosine",
    bf16: bool = True,
    fp16: bool = False,
    # gradient_checkpointing: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    eval_strategy: str = "steps",
    eval_steps: int = 1000,
    # save_strategy: str = "steps",
    # save_steps: int = 1000,
    save_total_limit: int = 5,
    save_best: bool = True,
    metric_for_best: str = "eval_loss",
    greater_is_better: bool = False,
    # logging_strategy: str = "steps",
    logging_steps: int = 10,
    report_to_wandb: bool = False,
    wandb_project: str | None = None,
    max_train_samples: int = -1,
    max_eval_samples: int = -1,
    do_augmentation: bool = False,
    pretrained_checkpoint_path: str | None = None,
    use_description_train: bool = False,
    use_description_eval: bool = False,
):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    model = GLiNER2.from_pretrained(model_name_or_path)

    # Load and optionally process training data
    if do_augmentation or not use_description_train:
        train_data = TrainingDataset.load(train_data_path)
        
        # Strip entity_descriptions if not using descriptions
        if not use_description_train:
            print("Stripping entity_descriptions from training data...")
            stripped_examples = []
            for ex in train_data:
                stripped_examples.append(InputExample(
                    text=ex.text,
                    entities=ex.entities,
                    entity_descriptions=None,
                ))
            train_data = TrainingDataset(stripped_examples)
        
        if do_augmentation:
            print(f"Applying data augmentation on {len(train_data)} examples...")
            augmented_examples = []
            for ex in tqdm(train_data, desc="Augmenting training dataset"):
                augmented_examples.extend(augment_example(ex))
            train_data = TrainingDataset(augmented_examples)
            print(f"Dataset size after augmentation: {len(train_data)}")
            
            # Since we doubled the dataset size, we halve the number of epochs 
            # to keep the total optimization steps roughly the same.
            old_epochs = num_epochs
            num_epochs = max(1, num_epochs // 2)
            print(f"Reduced num_epochs from {old_epochs} to {num_epochs} due to augmentation.")
        
        # train_data.validate(raise_on_error=False)
        train_data.print_stats()
    else:
        train_data = train_data_path

    # Strip entity_descriptions from validation data if not using descriptions
    eval_data = validation_data_path
    if not use_description_eval and validation_data_path is not None:
        print("Stripping entity_descriptions from validation data...")
        eval_dataset = TrainingDataset.load(validation_data_path)
        stripped_eval_examples = []
        for ex in eval_dataset:
            stripped_eval_examples.append(InputExample(
                text=ex.text,
                entities=ex.entities,
                entity_descriptions=None,
            ))
        eval_data = TrainingDataset(stripped_eval_examples)

    config = TrainingConfig(
        # output
        output_dir=output_dir,
        experiment_name=experiment_name,
        # training
        num_epochs=num_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        # optimizer
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        # scheduler
        warmup_ratio=warmup_ratio,
        scheduler_type=scheduler_type,
        # efficient training
        bf16=bf16,
        fp16=fp16,
        # # gradient_checkpointing=gradient_checkpointing,
        # distributed training
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # Auto-detect DDP
        # dataloader
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        # checkpointing
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        # save_strategy=save_strategy,
        # save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_best=save_best,
        metric_for_best=metric_for_best,
        greater_is_better=greater_is_better,
        # logging
        # logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        report_to_wandb=report_to_wandb,
        wandb_project=wandb_project,
    )

    trainer = GLiNER2Trainer(
        model,
        config,
        compute_metrics=lambda m, d: compute_metrics(m, d, batch_size=config.eval_batch_size, use_description=use_description_eval),
    )
    if pretrained_checkpoint_path:
        trainer.load_checkpoint(pretrained_checkpoint_path)
        print(f"Loaded checkpoint from {pretrained_checkpoint_path}")
    results = trainer.train(
        train_data=train_data,
        eval_data=eval_data,
    )
    print("Training completed!")
    print(f"Best metric: {results['best_metric']:.4f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Training time: {results['total_time_seconds']/60:.1f} minutes")

    if local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main)