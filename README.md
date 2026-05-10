# CS476 — Arabic Legal Document Classification

> NLP pipeline for classifying Saudi commercial court cases using the ALARB dataset.

**Course:** CS476 – Natural Language Processing  
**University:** Prince Sultan University, Riyadh  
**Students:** Abeer AlShehri · Rawan Ibrahim  
**Instructor:** Dr. Fatima AlShannaq  

---

## Overview

This project builds an end-to-end Arabic NLP pipeline to automatically classify legal documents from Saudi commercial courts. We address two classification tasks using the [ALARB dataset](https://huggingface.co/datasets/THIQAH-RD/ALARB), which contains 13,028 real court cases.

| Task | Description | Target |
|------|-------------|--------|
| **Task A** | Verdict Prediction | Accepted / Rejected / Settlement / Lapsed / Jurisdictional Dismissal |
| **Task B** | Legal Topic Classification | Applicable law category |

---

## Results

### Task A — Best Model

| Model | Features | Accuracy | Macro F1 |
|-------|----------|----------|----------|
| **Linear SVM** ⭐ | TF-IDF Unigram + Bigram | 80.03% | **0.7225** |
| Logistic Regression | TF-IDF Unigram + Bigram | 79.27% | 0.7205 |
| BiLSTM | Word2Vec Embeddings | 75.96% | 0.7064 |
| ANN | TF-IDF (20k features) | 73.16% | 0.6896 |

### Task A — All 14 Experiment Scenarios (sorted by Macro F1)

| Scenario | Accuracy | Macro F1 |
|----------|----------|----------|
| Linear SVM + TF-IDF Uni+Bi | 0.8003 | 0.7225 |
| Logistic Regression + TF-IDF Uni+Bi | 0.7927 | 0.7205 |
| Logistic Regression + TF-IDF 20k | 0.7815 | 0.7172 |
| Linear SVM + TF-IDF 20k | 0.7823 | 0.7113 |
| Linear SVM + Word Bigrams | 0.7891 | 0.7049 |
| Logistic Regression + Word Bigrams | 0.7843 | 0.7030 |
| Linear SVM + TF-IDF Unigram | 0.7384 | 0.6479 |
| Logistic Regression + TF-IDF Unigram | 0.7105 | 0.6338 |
| Logistic Regression + Char Trigrams | 0.6901 | 0.6027 |
| Logistic Regression + BoW | 0.6913 | 0.5965 |
| Linear SVM + Word2Vec Embeddings | 0.6993 | 0.5955 |
| Linear SVM + BoW | 0.6961 | 0.5937 |
| Linear SVM + Char Trigrams | 0.6885 | 0.5878 |
| Logistic Regression + Word2Vec | 0.6110 | 0.5388 |

---

## Pipeline

```
Raw ALARB Data
     ↓
Phase 1 — EDA & Dataset Loading
     ↓
Phase 2 — Label Engineering (Task A & B)
     ↓
Phase 3 — Arabic Text Preprocessing
          · Normalization · Legal noise removal
          · Stopword removal · ISRIStemmer
     ↓
Phase 4 — Feature Extraction
          · BoW · TF-IDF (uni/bi/20k)
          · N-grams (word & char) · Word2Vec
     ↓
Phase 5 — Model Training
          · Logistic Regression · Linear SVM
          · ANN · BiLSTM
     ↓
Phase 6 — Evaluation & Error Analysis
```

---

## Dataset

**ALARB** — Arabic Legal Argument Reasoning Benchmark  
- 13,028 Saudi commercial court cases  
- Source: [THIQAH-RD/ALARB on HuggingFace](https://huggingface.co/datasets/THIQAH-RD/ALARB)  
- Reference: Abu Shairah et al. (2025). *ALARB: An Arabic Legal Argument Reasoning Benchmark*. Proceedings of The Third Arabic NLP Conference.

---

## Files

| File | Description |
|------|-------------|
| `CS476_Project_Arabic_Legal_NLP.ipynb` | Full implementation notebook (Google Colab) |
| `CS476_Report.pdf` | IEEE-format research report |

---

## Key Findings

- Classical ML (SVM, LR) consistently outperforms deep learning on this dataset
- TF-IDF Unigram + Bigram is the most effective feature representation
- Task B (topic classification) is harder than Task A due to vocabulary overlap between legal categories
- Main error sources: negation patterns and class imbalance
