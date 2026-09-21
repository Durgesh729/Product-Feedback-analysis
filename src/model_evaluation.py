import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_multi_domain_model(df_test, model_path="models/multi_domain_sentiment_model.pkl", vec_path="models/multi_domain_tfidf_vectorizer.pkl"):
    print("--- STARTING PHASE 8: FINAL MODEL EVALUATION ON UNTOUCHED TEST SET ---")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Model or vectorizer artifact not found!")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    with open(vec_path, "rb") as f:
        vec = pickle.load(f)

    # Option B text representation (Title + Body)
    df_test = df_test.copy()
    test_text = (df_test['review_title'].fillna('') + " " + df_test['review_body'].fillna('')).str.strip()
    y_test = df_test['sentiment']

    X_test_vec = vec.transform(test_text)
    y_pred = clf.predict(X_test_vec)

    # Compute overall metrics
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    labels = ['Negative', 'Neutral', 'Positive']
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    class_report_str = classification_report(y_test, y_pred, labels=labels, digits=4)

    # Document Baseline Model B vs New Model Comparison
    lines = []
    lines.append("==================================================")
    lines.append("   MULTI-DOMAIN MODEL FINAL EVALUATION REPORT    ")
    lines.append("==================================================")
    lines.append(f"Evaluated Test Set Size: {len(df_test)} samples (Grouped by product_id)")
    lines.append("Model Artifacts: models/multi_domain_sentiment_model.pkl")
    lines.append("Vectorizer Artifacts: models/multi_domain_tfidf_vectorizer.pkl")
    lines.append("")
    lines.append("--- OVERALL METRICS COMPARISON ---")
    lines.append(f"{'Metric':<22} | {'Old Model B Baseline':<22} | {'New Multi-Domain Model':<22}")
    lines.append("-" * 72)
    lines.append(f"{'Accuracy':<22} | {'91.62% (0.9162)':<22} | {acc*100:.2f}% ({acc:.4f})")
    lines.append(f"{'Macro Precision':<22} | {'0.5484':<22} | {p_macro:.4f}")
    lines.append(f"{'Macro Recall':<22} | {'0.5897':<22} | {r_macro:.4f}")
    lines.append(f"{'Macro F1':<22} | {'0.5657':<22} | {f1_macro:.4f}")
    lines.append(f"{'Weighted F1':<22} | {'0.9133':<22} | {f1_weight:.4f}")

    lines.append("\n--- PER-CLASS F1 SCORE COMPARISON ---")
    lines.append(f"{'Class':<12} | {'Old Model B Baseline':<22} | {'New Multi-Domain Model':<22}")
    lines.append("-" * 60)
    lines.append(f"{'Negative':<12} | {'0.6974':<22} | {f1_class[0]:.4f} (P: {p_class[0]:.4f}, R: {r_class[0]:.4f})")
    lines.append(f"{'Neutral':<12} | {'0.0408':<22} | {f1_class[1]:.4f} (P: {p_class[1]:.4f}, R: {r_class[1]:.4f})")
    lines.append(f"{'Positive':<12} | {'0.9590':<22} | {f1_class[2]:.4f} (P: {p_class[2]:.4f}, R: {r_class[2]:.4f})")

    lines.append("\n--- CONFUSION MATRIX ---")
    lines.append(f"Labels order: {labels}")
    lines.append(f"{cm}")

    lines.append("\n--- DETAILED CLASSIFICATION REPORT ---")
    lines.append(class_report_str)

    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs("reports", exist_ok=True)
    with open("reports/MULTI_DOMAIN_MODEL_EVALUATION.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Saved evaluation report to reports/MULTI_DOMAIN_MODEL_EVALUATION.txt")
    return report_text

if __name__ == "__main__":
    from data_split import perform_stratified_group_split
    _, _, df_test = perform_stratified_group_split()
    evaluate_multi_domain_model(df_test)
