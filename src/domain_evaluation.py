import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_domain_wise_evaluation(df_test, model_path="models/multi_domain_sentiment_model.pkl", vec_path="models/multi_domain_tfidf_vectorizer.pkl"):
    print("--- STARTING PHASE 9: DOMAIN-WISE PERFORMANCE EVALUATION ---")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Model artifacts not found!")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    with open(vec_path, "rb") as f:
        vec = pickle.load(f)

    df_test = df_test.copy()
    test_text = (df_test['review_title'].fillna('') + " " + df_test['review_body'].fillna('')).str.strip()
    df_test['pred'] = clf.predict(vec.transform(test_text))

    categories = df_test['product_category'].unique()

    domain_results = []
    labels = ['Negative', 'Neutral', 'Positive']

    for cat in sorted(categories):
        cat_df = df_test[df_test['product_category'] == cat]
        cnt = len(cat_df)

        y_true = cat_df['sentiment']
        y_pred = cat_df['pred']

        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

        is_unstable = cnt < 100  # Explicit threshold for statistical instability

        domain_results.append({
            "category": cat,
            "count": cnt,
            "accuracy": acc,
            "neg_f1": f1_class[0],
            "neu_f1": f1_class[1],
            "pos_f1": f1_class[2],
            "macro_f1": f1_macro,
            "statistically_unstable": is_unstable
        })

    # Sort results by count descending
    domain_results.sort(key=lambda x: x['count'], reverse=True)

    lines = []
    lines.append("==================================================")
    lines.append("   MULTI-DOMAIN CATEGORY-WISE EVALUATION REPORT   ")
    lines.append("==================================================")
    lines.append("Evaluated on Held-out Test Set (30,000 samples)")
    lines.append("Categories with < 100 test samples are explicitly marked as [STATISTICALLY UNSTABLE].")
    lines.append("")
    lines.append(f"{'Category':<26} | {'Count':<6} | {'Accuracy':<8} | {'Neg F1':<8} | {'Neu F1':<8} | {'Pos F1':<8} | {'Macro F1':<8} | {'Stability Status'}")
    lines.append("-" * 115)

    for r in domain_results:
        status = "[UNSTABLE - LOW N]" if r['statistically_unstable'] else "Reliable"
        lines.append(f"{r['category']:<26} | {r['count']:<6} | {r['accuracy']*100:6.2f}%  | {r['neg_f1']:<8.4f} | {r['neu_f1']:<8.4f} | {r['pos_f1']:<8.4f} | {r['macro_f1']:<8.4f} | {status}")

    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs("reports", exist_ok=True)
    with open("reports/MULTI_DOMAIN_DOMAIN_EVALUATION.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("Saved domain evaluation report to reports/MULTI_DOMAIN_DOMAIN_EVALUATION.txt")
    return domain_results

if __name__ == "__main__":
    from data_split import perform_stratified_group_split
    _, _, df_test = perform_stratified_group_split()
    run_domain_wise_evaluation(df_test)
