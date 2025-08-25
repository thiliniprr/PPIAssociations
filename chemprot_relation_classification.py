# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:00:43 2025

@author: pereran
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification,AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

dataset = load_dataset("bigbio/chemprot", "chemprot_full_source")

train_dataset = dataset['train']
valid_dataset = dataset["validation"]
test_dataset  = dataset["test"]


def prepare_relation_examples(dataset, positive_only=True):
    samples = []
    for item in dataset:
        text = item['text']
        entity_ids = item['entities']['id']
        entity_types = item['entities']['type']
        entity_texts = item['entities']['text']
        entity_offsets = item['entities']['offsets']
        has_relations = bool(item['relations']['type'])

        # Make (chem, gene) pairs
        for i, type1 in enumerate(entity_types):
            for j, type2 in enumerate(entity_types):
                if i == j:
                    continue
                # Consider only Chem-Gene pairs
                if type1 == 'CHEMICAL' and type2.startswith('GENE'):
                    # Mark entities in text
                    chem_text = entity_texts[i]
                    gene_text = entity_texts[j]
                    mod_text = text
                    # For simple marking (be careful if substrings overlap)
                    if chem_text in mod_text and gene_text in mod_text:
                        # Mark first occurrence for both entities
                        mod_text = mod_text.replace(chem_text, "[CHEM]%s[/CHEM]" % chem_text, 1)
                        mod_text = mod_text.replace(gene_text, "[GENE]%s[/GENE]" % gene_text, 1)
                    else:  # If not found, skip
                        continue

                    # Label (default: no relation)
                    label = 0
                    if has_relations:
                        # Some ChemProt versions use arg ids, here it's empty, but in case it changes:
                        rel_found = False
                        for k, arg1 in enumerate(item['relations'].get('arg1', [])):
                            if (
                                (item['relations']['arg1'][k] == entity_ids[i] and item['relations']['arg2'][k] == entity_ids[j])
                            ):
                                # If type is a 'true' label, use it. For binary, just set to 1
                                label = 1 # Or more fine-grained if needed
                                rel_found = True
                        if positive_only and not rel_found:
                            continue

                    # Append example
                    samples.append({
                        'sentence': mod_text,
                        'chem': chem_text,
                        'gene': gene_text,
                        'label': label
                    })
    return samples


def tokenize_fn(example):
    # Tokenize the marked-sentence
    return tokenizer(
        example['sentence'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

train_samples = prepare_relation_examples(train_dataset, positive_only=False)
validate_samples = prepare_relation_examples(valid_dataset, positive_only=False)
test_samples = prepare_relation_examples(test_dataset, positive_only=False)

# Convert to HuggingFace Dataset objects
train_hf = Dataset.from_list(train_samples)
val_hf = Dataset.from_list(validate_samples)
test_hf = Dataset.from_list(test_samples)

# Labels must be integers
train_hf = train_hf.map(tokenize_fn, batched=False)
val_hf = val_hf.map(tokenize_fn, batched=False)
test_hf = test_hf.map(tokenize_fn, batched=False)


model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    num_labels=2  # Binary: 0 or 1
)

training_args = TrainingArguments(
    output_dir="./bio-chemprot-bert",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none"
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

preds = trainer.predict(test_hf)
pred_labels = preds.predictions.argmax(-1)
print(pred_labels)  # 0 (no relation) or 1 (relation)


import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))


