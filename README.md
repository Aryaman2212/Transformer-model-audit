# Transformer Model Audit: Diagnosing Eight Email Classifiers

MSc coursework (Text as Data), University of Glasgow, 2026.

This project is an audit, not a build. I was given eight pre-trained transformer email classifiers,
each belonging to a fictional client with a complaint, and asked **what is actually wrong with
each one**. Every model and dataset is loaded from the Hugging Face Hub, and every finding comes
from evidence: split-level metrics, overlap checks, label distributions and tokenizer
inspection.

## Findings

| Client | Complaint | Diagnosis | Evidence |
|---|---|---|---|
| OrbitalForge | Unknown tokenizer | It's `bert-base-cased` | Only this vocabulary decodes the provided token IDs back into coherent English |
| NeuroBloom | Model behaves badly | **Broken tokenizer vocabulary**: only 1,145 tokens, mostly single characters | About 78% of tokens map to `[UNK]`, including "the", "and" and "meeting" |
| HydroVine | Poor real-world performance | **Severe overfitting** | Train F1 1.000; validation 0.468; test 0.493 (chance is 0.2) |
| LumenGrid | Deployed model underperforms | **Train-test leakage** | Test F1 0.983 is *higher* than train (0.826) and validation (0.822), so real performance is about 82% |
| AeroSynth | Some classes perfect, others poor | **Label leakage**: label names are prefixed to the email text | "HR & Internal Admin SUBJECT: ..." gives the answer away in the first few tokens |
| CryoNest | Suspicious evaluation | **No real split**: train, validation and test are identical | All 2,800 texts overlap across every split, so every split shows the same 0.855 / 0.861 |
| NeonPixel | "Ready to deploy?" | **No**: two of the five classes never appear in training | Train accuracy 0.936 drops to 0.591 on a five-class test set; 0.00 F1 on both missing classes |
| SolaraMesh | Reported **94% accuracy** | **Majority-class model on a 92/8 imbalanced dataset** | Accuracy 0.941 but macro-F1 0.292; zero F1 on three classes |

### The 94% accuracy claim

SolaraMesh reported 94% accuracy. But 92.4% of the training data is a single class
(Sales & Marketing), so a model that always predicts that class scores about 92%. Macro-F1
tells the real story: **0.292 on train and 0.224 on test**, with zero F1 on three of the five
categories. The model does no better than a majority-class baseline. The report was
accurate, but it measured the wrong thing.

## What this demonstrates

Checking whether a reported metric means what it appears to mean. The same failure modes
(leakage, class imbalance, overfitting and train/test contamination) come up in real model
validation work.

## Running it

```bash
pip install -r requirements.txt
python audit.py
```

The models and datasets are downloaded from the Hugging Face Hub (`TextAsData/...`) on first run.
