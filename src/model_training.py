import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def run_model_training_experiments(df_train, df_val):
    print("--- STARTING PHASE 7 & 8: TEXT REPRESENTATION EXPERIMENTS & MODEL SELECTION ---")

    # Define text representation options
    # Option A: review_body only
    # Option B: review_title + " " + review_body

    df_train = df_train.copy()
    df_val = df_val.copy()

    df_train['text_A'] = df_train['review_body'].fillna('')
    df_train['text_B'] = (df_train['review_title'].fillna('') + " " + df_train['review_body'].fillna('')).str.strip()

    df_val['text_A'] = df_val['review_body'].fillna('')
    df_val['text_B'] = (df_val['review_title'].fillna('') + " " + df_val['review_body'].fillna('')).str.strip()

    y_train = df_train['sentiment']
    y_val = df_val['sentiment']

    experiments = [
        {"name": "Exp 1: Option A (Body Only), Unigram (1,1), LogisticRegression(balanced)", "text_col": "text_A", "ngram": (1,1), "class_weight": "balanced"},
        {"name": "Exp 2: Option A (Body Only), Unigram+Bigram (1,2), LogisticRegression(balanced)", "text_col": "text_A", "ngram": (1,2), "class_weight": "balanced"},
        {"name": "Exp 3: Option B (Title+Body), Unigram (1,1), LogisticRegression(balanced)", "text_col": "text_B", "ngram": (1,1), "class_weight": "balanced"},
        {"name": "Exp 4: Option B (Title+Body), Unigram+Bigram (1,2), LogisticRegression(balanced)", "text_col": "text_B", "ngram": (1,2), "class_weight": "balanced"},
        {"name": "Exp 5: Option B (Title+Body), Unigram+Bigram (1,2), LogisticRegression(None)", "text_col": "text_B", "ngram": (1,2), "class_weight": None}
    ]

    results = []

    for exp in experiments:
        t0 = time.time()
        print(f"\nRunning {exp['name']}...")
        vec = TfidfVectorizer(ngram_range=exp['ngram'], min_df=2, max_features=50000)

        # Fit vectorizer strictly on TRAIN data
        X_train_vec = vec.fit_transform(df_train[exp['text_col']])
        X_val_vec = vec.transform(df_val[exp['text_col']])

        clf = LogisticRegression(class_weight=exp['class_weight'], max_iter=1000, random_state=42)
        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_val_vec)

        acc = accuracy_score(y_val, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
        p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

        # Per class metrics
        labels = ['Negative', 'Neutral', 'Positive']
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_val, y_pred, labels=labels)

        res = {
            "name": exp['name'],
            "text_representation": "Option B (Title+Body)" if exp['text_col'] == "text_B" else "Option A (Body Only)",
            "ngram_range": str(exp['ngram']),
            "class_weight": str(exp['class_weight']),
            "accuracy": acc,
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weight,
            "neg_precision": p_class[0], "neg_recall": r_class[0], "neg_f1": f1_class[0],
            "neu_precision": p_class[1], "neu_recall": r_class[1], "neu_f1": f1_class[1],
            "pos_precision": p_class[2], "pos_recall": r_class[2], "pos_f1": f1_class[2],
            "fit_time_sec": time.time() - t0
        }
        results.append(res)
        print(f"  --> Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Neg F1: {f1_class[0]:.4f} | Neu F1: {f1_class[1]:.4f} | Pos F1: {f1_class[2]:.4f}")

    # Build comparison report
    lines = []
    lines.append("==================================================")
    lines.append("   MULTI-DOMAIN MODEL CANDIDATE EXPERIMENTS       ")
    lines.append("==================================================")
    lines.append("Evaluation Split: Validation Set (30,000 samples, Grouped by product_id)")
    lines.append("All vectorizers fitted STRICTLY on Training Set (140,000 samples).")
    lines.append("")

    lines.append(f"{'Experiment Name':<55} | {'Acc':<7} | {'MacroF1':<7} | {'Neg F1':<7} | {'Neu F1':<7} | {'Pos F1':<7}")
    lines.append("-" * 105)

    for r in results:
        lines.append(f"{r['name']:<55} | {r['accuracy']:<7.4f} | {r['macro_f1']:<7.4f} | {r['neg_f1']:<7.4f} | {r['neu_f1']:<7.4f} | {r['pos_f1']:<7.4f}")

    comparison_report = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/MULTI_DOMAIN_MODEL_COMPARISON.txt", "w", encoding="utf-8") as f:
        f.write(comparison_report)

    # Select best model based on Macro F1
    best_exp = max(results, key=lambda x: x['macro_f1'])
    print(f"\n--- SELECTION COMPLETE ---")
    print(f"Selected Best Architecture: {best_exp['name']}")
    print(f"  Macro F1: {best_exp['macro_f1']:.4f} | Accuracy: {best_exp['accuracy']:.4f}")

    return best_exp, results

def train_final_multi_domain_model(df_train, df_val, best_exp_config):
    print("\n--- RETRAIN FINAL MODEL ON COMBINED TRAIN + VALIDATION SET ---")

    # Combine Train + Val
    df_train_val = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    text_col = best_exp_config['text_representation']
    if "Option B" in text_col:
        text_series = (df_train_val['review_title'].fillna('') + " " + df_train_val['review_body'].fillna('')).str.strip()
    else:
        text_series = df_train_val['review_body'].fillna('')

    y_train_val = df_train_val['sentiment']

    # Vectorizer & Model setup
    ngram_range = eval(best_exp_config['ngram_range'])
    cw = None if best_exp_config['class_weight'] == 'None' else 'balanced'

    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_features=50000)
    X_train_val_vec = vec.fit_transform(text_series)

    clf = LogisticRegression(class_weight=cw, max_iter=1000, random_state=42)
    clf.fit(X_train_val_vec, y_train_val)

    # Save final multi-domain model artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/multi_domain_sentiment_model.pkl"
    vec_path = "models/multi_domain_tfidf_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    with open(vec_path, "wb") as f:
        pickle.dump(vec, f)

    print(f"SUCCESS: Saved final multi-domain model to {model_path} and vectorizer to {vec_path}")
    return clf, vec

if __name__ == "__main__":
    from data_split import perform_stratified_group_split
    df_tr, df_v, df_te = perform_stratified_group_split()
    best_cfg, _ = run_model_training_experiments(df_tr, df_v)
    train_final_multi_domain_model(df_tr, df_v, best_cfg)
