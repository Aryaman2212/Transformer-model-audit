"""
Transformer Model Audit - diagnosing eight text classifiers
=====================================================
This script contains all code and analysis for diagnosing issues with 
machine learning models across 8 client projects.
"""

# ============================================================================
# Setup
# ============================================================================
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from collections import Counter

def evaluate_model(model_name, dataset_name, splits=['train','validation','test']):
    """Evaluate a model on all splits and return metrics."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    ds = load_dataset(dataset_name)
    
    results = {}
    for split in splits:
        if split not in ds:
            continue
        texts = list(ds[split]['text'])
        true_labels = list(ds[split]['label'])
        pred_labels = []
        
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=512)
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)
                pred_labels.extend([model.config.id2label[p.item()] for p in preds])
        
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        results[split] = {'accuracy': acc, 'macro_f1': f1, 
                         'true': true_labels, 'pred': pred_labels}
        
        print(f"\n{split.upper()}: Accuracy={acc:.3f}, Macro-F1={f1:.3f}")
        print(classification_report(true_labels, pred_labels, zero_division=0))
    
    return results, ds, model, tokenizer

# ============================================================================
# Q1 - OrbitalForge: Identify the tokenizer
# ============================================================================
print("=" * 80)
print("Q1 - OrbitalForge: Tokenizer Identification")
print("=" * 80)

ds_q1 = load_dataset("TextAsData/Q1-OrbitalForge-dataset")
sample_ids = ds_q1['train'][0]['input_ids']

candidates = [
    "roberta-base", "microsoft/deberta-base", "bert-base-uncased",
    "distilbert-base-uncased", "bert-base-cased", "allenai/scibert_scivocab_uncased",
]

for name in candidates:
    tok = AutoTokenizer.from_pretrained(name)
    decoded = tok.decode(sample_ids[:50])
    print(f"\n{name}:")
    print(f"  Decoded: {decoded[:200]}")

print("\n>>> ANSWER: The tokenizer is bert-base-cased")
print("    It is the only tokenizer that produces coherent English text when decoding")
print("    the tokenized input_ids. All others produce gibberish because the token IDs")
print("    map to different words in their vocabularies.")

# ============================================================================
# Q2 - NeuroBloom: What's wrong with the tokenizer?
# ============================================================================
print("\n" + "=" * 80)
print("Q2 - NeuroBloom: Tokenizer Issue")
print("=" * 80)

tok_q2 = AutoTokenizer.from_pretrained("TextAsData/Q2-NeuroBloom-model")
ds_q2 = load_dataset("TextAsData/Q2-NeuroBloom-dataset")

print(f"Vocab size: {tok_q2.vocab_size}")
print(f"Special tokens: {tok_q2.special_tokens_map}")

# Show vocab structure
vocab = tok_q2.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
print(f"\nFirst 30 vocab entries (single characters):")
for token, idx in sorted_vocab[:30]:
    print(f"  {idx}: '{token}'")

# Demonstrate the UNK problem
sample_text = ds_q2['train'][0]['text']
tokens = tok_q2.tokenize(sample_text)
unk_count = sum(1 for t in tokens if t == tok_q2.unk_token)
print(f"\nSample tokenization:")
print(f"  Text: {sample_text[:150]}...")
print(f"  UNK tokens: {unk_count}/{len(tokens)} ({unk_count/len(tokens)*100:.1f}%)")

# Overall UNK rate
total_tokens = 0
total_unks = 0
for i in range(min(100, len(ds_q2['train']))):
    tokens = tok_q2.tokenize(ds_q2['train'][i]['text'])
    total_tokens += len(tokens)
    total_unks += sum(1 for t in tokens if t == tok_q2.unk_token)
print(f"\nOverall UNK rate (100 examples): {total_unks}/{total_tokens} = {total_unks/total_tokens:.3f}")

# Common words test
common_words = ['the', 'and', 'hello', 'email', 'meeting', 'project']
print(f"\nCommon word tokenization:")
for w in common_words:
    print(f"  '{w}' -> {tok_q2.tokenize(w)}")

print("\n>>> ANSWER: The tokenizer has a critically small vocabulary of only 1,145 tokens.")
print("    The vocabulary consists almost entirely of single characters (a-z) and some")
print("    character bigrams (two-letter combinations). It lacks actual word-level or")
print("    proper subword tokens. As a result, approximately 78% of all tokens are mapped")
print("    to [UNK], meaning most words in the input text cannot be represented.")
print("    Common English words like 'the', 'and', 'meeting', 'project' are all unknown.")
print("    This makes the tokenizer (and any model using it) essentially useless for NLP.")

# ============================================================================
# Q3 - HydroVine: Model performance issue
# ============================================================================
print("\n" + "=" * 80)
print("Q3 - HydroVine: Model Performance Diagnosis")
print("=" * 80)

results_q3, ds_q3, _, _ = evaluate_model("TextAsData/Q3-HydroVine-model", 
                                          "TextAsData/Q3-HydroVine-dataset")

print(f"\nSUMMARY:")
print(f"{'Split':<12} {'Accuracy':>10} {'Macro-F1':>10}")
for split in ['train','validation','test']:
    r = results_q3[split]
    print(f"{split:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")

print("\n>>> ANSWER: The model is severely overfitting.")
print("    Train: Accuracy=1.000, Macro-F1=1.000 (perfect)")
print("    Val:   Accuracy=0.470, Macro-F1=0.468")
print("    Test:  Accuracy=0.497, Macro-F1=0.493")
print("    The model achieves perfect performance on training data but performs barely")
print("    above chance (~0.2 for 5 classes) on unseen data. It has memorized the training")
print("    examples rather than learning generalizable patterns. This could be caused by")
print("    training for too many epochs, insufficient regularization, learning rate too")
print("    high, or insufficient training data (only 1,400 examples).")

# ============================================================================
# Q4 - LumenGrid: Deployment performance discrepancy
# ============================================================================
print("\n" + "=" * 80)
print("Q4 - LumenGrid: Deployment Performance Investigation")
print("=" * 80)

results_q4, ds_q4, _, _ = evaluate_model("TextAsData/Q4-LumenGrid-model", 
                                          "TextAsData/Q4-LumenGrid-dataset")

print(f"\nSUMMARY:")
print(f"{'Split':<12} {'Accuracy':>10} {'Macro-F1':>10}")
for split in ['train','validation','test']:
    r = results_q4[split]
    print(f"{split:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")

# Data leakage check
train_texts = set(ds_q4['train']['text'])
test_texts = set(ds_q4['test']['text'])
val_texts = set(ds_q4['validation']['text'])
print(f"\nData overlap check:")
print(f"  Train-Test exact overlap: {len(train_texts & test_texts)}")
print(f"  Train-Val exact overlap: {len(train_texts & val_texts)}")

# Subject overlap
train_subjects = set(t.split('\n')[0] for t in ds_q4['train']['text'] if 'SUBJECT:' in t)
test_subjects = set(t.split('\n')[0] for t in ds_q4['test']['text'] if 'SUBJECT:' in t)
print(f"  Train-Test subject overlap: {len(train_subjects & test_subjects)}/{len(test_subjects)}")

print("\n>>> ANSWER: The test set performance (0.982/0.983) is suspiciously much higher")
print("    than train (0.822/0.826) and validation (0.818/0.822) performance. This is")
print("    the opposite of what we'd normally expect — test performance should be equal")
print("    to or lower than training performance. This strongly suggests data leakage:")
print("    the test data was likely used during training (or is derived from training data).")
print("    When deployed on truly new data, the model will achieve ~82% accuracy (similar")
print("    to the validation set), NOT the reported ~98% test accuracy. The client is")  
print("    right to be disappointed — the reported test performance is artificially inflated.")

# ============================================================================
# Q5 - AeroSynth: Unusual performance results
# ============================================================================
print("\n" + "=" * 80)
print("Q5 - AeroSynth: Unusual Results Investigation")
print("=" * 80)

results_q5, ds_q5, _, _ = evaluate_model("TextAsData/Q5-AeroSynth-model", 
                                          "TextAsData/Q5-AeroSynth-dataset")

print(f"\nSUMMARY:")
print(f"{'Split':<12} {'Accuracy':>10} {'Macro-F1':>10}")
for split in ['train','validation','test']:
    r = results_q5[split]
    print(f"{split:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")

# Show the key finding: label prefixes in text
print("\n=== KEY FINDING: Label names in text ===")
for label in sorted(set(ds_q5['train']['label'])):
    examples = [t for t, l in zip(ds_q5['train']['text'], ds_q5['train']['label']) if l == label]
    print(f"\n{label}:")
    for e in examples[:2]:
        print(f"  {e[:100]}...")

# Confusion matrix
labels_sorted = sorted(set(ds_q5['test']['label']))
cm = confusion_matrix(results_q5['test']['true'], results_q5['test']['pred'], labels=labels_sorted)
print(f"\nTest Confusion Matrix:")
print(f"Labels: {labels_sorted}")
print(cm)

print("\n>>> ANSWER: The unusual result is that HR & Internal Admin and Product &")
print("    Engineering achieve 100% precision and recall, while Client & Partner")
print("    Communications and Sales & Marketing perform much worse and are heavily")
print("    confused with each other.")
print("    CAUSE: Some categories have the label name PREFIXED to the email text.")
print("    HR & Internal Admin emails begin with 'HR & Internal Admin SUBJECT:...'")
print("    and Product & Engineering emails begin with 'Product & Engineering SUBJECT:...'")
print("    This is label leakage — the model just needs to read the first few tokens")
print("    to perfectly classify these categories. The other categories (Client, Sales,")
print("    Operations) don't have this prefix, so they are classified based on content")
print("    only, leading to much lower (and more realistic) performance.")

# ============================================================================
# Q6 - CryoNest: Dataset evaluation issue
# ============================================================================
print("\n" + "=" * 80)
print("Q6 - CryoNest: Dataset Issue Diagnosis")
print("=" * 80)

results_q6, ds_q6, _, _ = evaluate_model("TextAsData/Q6-CryoNest-model", 
                                          "TextAsData/Q6-CryoNest-dataset")

print(f"\nSUMMARY:")
print(f"{'Split':<12} {'Accuracy':>10} {'Macro-F1':>10}")
for split in ['train','validation','test']:
    r = results_q6[split]
    print(f"{split:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")

# Show overlap
train_texts = set(ds_q6['train']['text'])
val_texts = set(ds_q6['validation']['text'])
test_texts = set(ds_q6['test']['text'])
print(f"\nData overlap:")
print(f"  Train-Val overlap: {len(train_texts & val_texts)} / {len(train_texts)}")
print(f"  Train-Test overlap: {len(train_texts & test_texts)} / {len(train_texts)}")
print(f"  Val-Test overlap: {len(val_texts & test_texts)} / {len(val_texts)}")

print("\n>>> ANSWER: The train, validation, and test splits all contain the IDENTICAL data.")
print("    All 2800 texts overlap completely between every pair of splits, producing")
print("    identical metrics across all three splits (0.855/0.861).")
print("    This means there is no proper train/validation/test split — the evaluation")
print("    is meaningless because the model is being tested on the same data it was")
print("    trained on. The performance cannot be trusted as a measure of generalization.")
print("    A proper random split should be performed to enable meaningful evaluation.")

# ============================================================================
# Q7 - NeonPixel: Deployment readiness
# ============================================================================
print("\n" + "=" * 80)
print("Q7 - NeonPixel: Deployment Readiness Assessment")
print("=" * 80)

results_q7, ds_q7, _, _ = evaluate_model("TextAsData/Q7-NeonPixel-model", 
                                          "TextAsData/Q7-NeonPixel-dataset")

print(f"\nSUMMARY:")
print(f"{'Split':<12} {'Accuracy':>10} {'Macro-F1':>10}")
for split in ['train','validation','test']:
    r = results_q7[split]
    print(f"{split:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")

# Show the label mismatch
print(f"\nTraining labels: {sorted(set(ds_q7['train']['label']))}")
print(f"Val labels: {sorted(set(ds_q7['validation']['label']))}")
print(f"Test labels: {sorted(set(ds_q7['test']['label']))}")

for split in ['train','validation','test']:
    print(f"\n{split} label distribution:")
    for label, count in sorted(Counter(ds_q7[split]['label']).items()):
        print(f"  {label}: {count}")

print("\n>>> ANSWER: The model is NOT ready for deployment.")
print("    The training data only contains 3 of the 5 required categories:")
print("    - Client & Partner Communications (957)")
print("    - Product & Engineering (1050)")
print("    - Sales & Marketing (917)")
print("    HR & Internal Admin and Operations & Maintenance are ENTIRELY MISSING from")
print("    the training data. The model never sees these classes during training and")
print("    therefore can never predict them (0.00 F1 for both).")
print("    While the model achieves 0.936 accuracy on training data (which only has 3")
print("    classes), it drops to 0.591 on test data that includes all 5 classes.")
print("    The model needs to be retrained on data that includes ALL 5 categories.")

# ============================================================================
# Q8 - SolaraMesh: 94% accuracy claim
# ============================================================================
print("\n" + "=" * 80)
print("Q8 - SolaraMesh: Performance Report Verification")
print("=" * 80)

results_q8, ds_q8, _, _ = evaluate_model("TextAsData/Q8-SolaraMesh-model", 
                                          "TextAsData/Q8-SolaraMesh-dataset")

print(f"\nSUMMARY:")
print(f"{'Split':<12} {'Accuracy':>10} {'Macro-F1':>10}")
for split in ['train','validation','test']:
    r = results_q8[split]
    print(f"{split:<12} {r['accuracy']:>10.3f} {r['macro_f1']:>10.3f}")

# Show class imbalance
for split in ['train','validation','test']:
    counts = Counter(ds_q8[split]['label'])
    total = sum(counts.values())
    print(f"\n{split} label distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

# Show prediction distribution
print(f"\nTest prediction distribution: {Counter(results_q8['test']['pred'])}")

print("\n>>> ANSWER: The performance report was NOT made in good faith.")
print("    While the model does achieve ~94% accuracy on training data, this metric")
print("    is deeply misleading. The dataset is extremely imbalanced:")
print("    - Sales & Marketing: 955 examples (92.4% of training data)")
print("    - All other categories combined: only 78 examples (7.6%)")
print("    The model has essentially learned to predict 'Sales & Marketing' for almost")
print("    everything. Reporting only accuracy on such an imbalanced dataset is misleading.")
print("    The macro-F1 score tells the real story:")
print("    - Train: Accuracy=0.941, but Macro-F1=0.292")
print("    - Test:  Accuracy=0.923, but Macro-F1=0.224")
print("    The model achieves 0.00 F1 for Client, HR, and Operations categories.")
print("    A naive baseline that always predicts 'Sales & Marketing' would get ~92%")
print("    accuracy. The model is essentially no better than this majority-class baseline.")
print("    Responsible reporting should include macro-F1 and per-class metrics.")
