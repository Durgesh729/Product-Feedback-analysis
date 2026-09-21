# MULTI-DOMAIN PRODUCT FEEDBACK SENTIMENT ANALYSIS SYSTEM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.1-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![HTML/CSS](https://img.shields.io/badge/UI-HTML5%20%2F%20CSS3-red.svg)](https://developer.mozilla.org/)

An automated Natural Language Processing (NLP) system designed to analyze multi-domain e-commerce product reviews across 31 categories, validate input domain relevance, and classify feedback sentiment into **Positive**, **Negative**, and **Neutral** categories using a leakage-free supervised machine learning pipeline.

---

## 📌 Single-Domain Limitation & Multi-Domain Redesign

- **Original Limitation**: The initial model (Model B) was trained on a single-domain electronics dataset (4,955 reviews). While achieving 91.62% accuracy on electronics, it failed to generalize to other product domains (e.g. personal care items like dental/interdental brushes) and suffered from a severe Neutral F1 score limitation (0.0408).
- **Multi-Domain Expansion**: The system has been upgraded with a comprehensive multi-domain dataset (`train.csv`) containing **200,000 English Amazon product reviews** spanning **31 diverse product categories** (`home`, `apparel`, `wireless`, `beauty`, `drugstore`, `kitchen`, `automotive`, `pc`, `electronics`, `personal_care_appliances`, etc.).

---

## 🎯 Main Objective

Develop a practical, explainable, and statistically rigorous multi-domain NLP system that accepts user input, validates its domain relevance, and automatically classifies product feedback into:
- 🟢 **Positive**: High satisfaction, praising product quality and usability (4–5 stars).
- 🔴 **Negative**: Product defects, damage, or customer dissatisfaction (1–2 stars).
- 🟡 **Neutral**: Informational, delivery, or non-emotional observations (3 stars).

---

## 🛠️ Technology Stack

- **Programming Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **NLP & Preprocessing**: NLTK (Tokenization, Negation-Preserving Stop-word removal, Lemmatization)
- **Domain Validation**: Pattern & Lexicon-based input verification layer across 31 product domains
- **Machine Learning**: Scikit-Learn (Unigram+Bigram TF-IDF Vectorizer, Logistic Regression with balanced class weights)
- **Backend API Framework**: FastAPI + Uvicorn
- **Frontend UI**: Responsive HTML5, CSS3, Vanilla JavaScript
- **Model Serialization**: Pickle / Joblib

---

## 🔄 Complete NLP & ML Workflow

```
Raw Input Text (Review Title + Review Body)
      ↓
Domain Validation Layer (Filters out non-product text e.g., "I am not feeling well")
      ↓ [Valid Product Feedback]
Text Cleaning & Normalization (Lowercasing, Contraction Expansion, Special Symbol Removal)
      ↓
Tokenization & Stop-Word Filtering (Preserves negation words: 'not', 'no', 'never', 'don't')
      ↓
Lemmatization (Reduces words to root dictionary form while preserving sentiment context)
      ↓
TF-IDF Feature Extraction (Unigram + Bigram sparse matrix representation, min_df=2, max_features=50,000)
      ↓
Supervised Machine Learning Classifier (Logistic Regression with class_weight='balanced')
      ↓
Predicted Sentiment & Model Probabilities (Positive / Negative / Neutral)
```

---

## 📊 Dataset Audit & Data Quality Pipeline

### 1. Comprehensive Audit Findings (`train.csv`)
- **Total Rows**: 200,000 | **Total Columns**: 8
- **Language**: 100% English (`en`)
- **Star Ratings**: Uniformly distributed (40,000 rows per star rating 1 through 5).
- **Derived Sentiment Mapping**:
  - Negative (1–2 Stars): 80,000 (40.00%)
  - Neutral (3 Stars): 40,000 (20.00%)
  - Positive (4–5 Stars): 80,000 (40.00%)
- **Unique Identifiers**: 200,000 `review_id`s, 185,541 `product_id`s, 196,745 `reviewer_id`s.
- **Missing Values**: 33 missing `review_title` entries (replaced with empty string `""`).

### 2. Documented Duplicate Review Policy
- **Category A (Identical text, same label)**: 1,109 rows across 535 unique texts → Deduplicated to 1 representative record to prevent split contamination.
- **Category B (Identical text, conflicting labels)**: 20 rows across 10 unique texts → REMOVED ALL INSTANCES to eliminate contradictory label noise.
- **Category C (Cross-product duplicate text)**: 834 rows across 408 unique texts → Deduplicated to ensure strict product-level isolation.
- **Cleaned Dataset**: Saved to `data/cleaned_multi_domain_reviews.csv` (199,415 rows).

### 3. Leakage-Safe Stratified Group Split
- **Grouping Variable**: `product_id` (Ensures products in test set NEVER appear in training).
- **Partition Ratio**: 70% Train (139,590 rows) / 15% Validation (29,912 rows) / 15% Test (29,913 rows).
- **Product Leakage**: **VERIFIED 0% OVERLAP** across Train, Val, and Test splits.
- **Exact Split Class Proportions**: Exactly 40.00% Negative, 20.00% Neutral, 40.00% Positive across all three splits (Random Seed = 42).

---

## 🤖 Model Experiments, Baseline Comparison & Evaluation

### 1. Text Field Representation Experiment (Validation Set)
- **Option A (`review_body` alone)**: Acc = 70.42%, Macro F1 = 0.6775
- **Option B (`review_title + review_body`) [SELECTED]**: Acc = 73.03%, Macro F1 = **0.7032** (Title provides crucial strong sentiment signals).

### 2. Final Model vs Model B Baseline Metrics (Untouched Test Set, 29,913 samples)

| Metric | Old Model B Baseline (Electronics Only) | New Multi-Domain Model (31 Product Categories) |
|---|---|---|
| **Accuracy** | 91.62% | **73.19%** |
| **Macro Precision** | 0.5484 | **0.6885** |
| **Macro Recall** | 0.5897 | **0.7225** |
| **Macro F1-Score** | 0.5657 | **0.7042 (+13.85%)** |
| **Weighted F1-Score** | 0.9133 | **0.7388** |
| **Negative F1** | 0.6974 | **0.8049 (P: 0.7989, R: 0.8110)** |
| **Neutral F1** | 0.0408 | **0.5117 (P: 0.4705, R: 0.5609)** *(12.5x Improvement)* |
| **Positive F1** | 0.9590 | **0.7960 (P: 0.7960, R: 0.7957)** |

### 3. Domain-Wise Evaluation Across 31 Categories
- Macro F1 remains consistently high (0.6622 to 0.7280) across all 30 large product domains (`home`, `apparel`, `wireless`, `beauty`, `drugstore`, `kitchen`, `sports`, `pc`, `electronics`, `automotive`, etc.).
- `personal_care_appliances` (11 test samples) is explicitly marked as `[UNSTABLE - LOW N]` in `reports/MULTI_DOMAIN_DOMAIN_EVALUATION.txt`.

### 4. 135-Example Unseen Manual Sanity Test & Dental Brush Case
- **Structured 135-Example Benchmark**: 9 domains × 3 sentiments × 5 examples with **0% text overlap** against training data achieved **89.63% Accuracy** and **0.8931 Macro F1**.
- **Qualitative Dental Brush Review Test**:
  - *Text*: "The angle provided in this interdental brush is not accurate. It hurts the gums. The bristle pin length is very short for pre molar and molar teeth. So difficult clean teeth from one side. Poor quality plastic is used."
  - *Predicted Sentiment*: **Negative**
  - *Predicted Probabilities*: Negative: **80.37%**, Neutral: 7.66%, Positive: 11.97% (Successfully generalizes to personal care domain without hardcoding).

---

## 🚀 Execution & Verification Commands

```bash
# Run complete multi-domain pipeline (Phases 1 through 11)
python src/run_pipeline.py

# Run FastAPI backend web server
python app/main.py
```

---

## 🔬 Scientific & Academic Disclaimers

1. **Predicted Class Probabilities**: Model probabilities reflect internal soft-max / logistic probability distributions and must not be described as guaranteed factual certainty.
2. **Domain Boundaries**: Very low-sample categories (e.g. `personal_care_appliances` with 75 total training reviews) carry higher statistical variance.
3. **Model Artifact Separation**: Legacy single-domain artifacts (`models/sentiment_model.pkl`) remain untouched to maintain full project reproduciblity.
