import os
import json
import pandas as pd
import numpy as np

def run_dataset_audit(data_path="train.csv"):
    print(f"--- STARTING PHASE 1: DATASET AUDIT ON {data_path} ---")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist!")

    df = pd.read_csv(data_path)

    # 1. Total rows & 2. Total columns
    total_rows, total_cols = df.shape

    # 3. Column data types
    col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 4. Missing values
    missing_vals = df.isnull().sum().to_dict()

    # 5. Empty strings / whitespace-only
    empty_strings = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            cnt = (df[col].astype(str).str.strip() == '').sum()
            empty_strings[col] = int(cnt)

    # 6. Invalid star values (outside 1..5)
    valid_stars = set([1, 2, 3, 4, 5])
    actual_stars = df['stars'].unique().tolist()
    invalid_star_count = int((~df['stars'].isin(valid_stars)).sum())

    # 7. Language distribution
    lang_dist = df['language'].value_counts().to_dict()

    # 8. Unique review IDs
    uniq_review_ids = int(df['review_id'].nunique())

    # 9. Unique product IDs
    uniq_product_ids = int(df['product_id'].nunique())

    # 10. Unique reviewer IDs
    uniq_reviewer_ids = int(df['reviewer_id'].nunique())

    # 11. Duplicate complete rows
    dup_complete_rows = int(df.duplicated().sum())

    # 12. Duplicate review_body values
    dup_review_bodies = int(df['review_body'].duplicated().sum())

    # 13. Duplicate review_title values
    dup_review_titles = int(df['review_title'].dropna().duplicated().sum())

    # 14. Product reviews per product summary
    prod_counts = df['product_id'].value_counts()
    prod_reviews_stats = {
        "mean": float(prod_counts.mean()),
        "std": float(prod_counts.std()),
        "min": int(prod_counts.min()),
        "max": int(prod_counts.max()),
        "median": float(prod_counts.median()),
        "q25": float(prod_counts.quantile(0.25)),
        "q75": float(prod_counts.quantile(0.75))
    }

    # 15. Reviews per reviewer summary
    rev_counts = df['reviewer_id'].value_counts()
    reviewer_stats = {
        "mean": float(rev_counts.mean()),
        "std": float(rev_counts.std()),
        "min": int(rev_counts.min()),
        "max": int(rev_counts.max()),
        "median": float(rev_counts.median()),
        "q25": float(rev_counts.quantile(0.25)),
        "q75": float(rev_counts.quantile(0.75))
    }

    # 16. Review text length distribution
    body_char_len = df['review_body'].astype(str).str.len()
    body_word_len = df['review_body'].astype(str).str.split().str.len()
    body_length_stats = {
        "char_mean": float(body_char_len.mean()),
        "char_std": float(body_char_len.std()),
        "char_min": int(body_char_len.min()),
        "char_max": int(body_char_len.max()),
        "char_median": float(body_char_len.median()),
        "word_mean": float(body_word_len.mean()),
        "word_std": float(body_word_len.std()),
        "word_min": int(body_word_len.min()),
        "word_max": int(body_word_len.max()),
        "word_median": float(body_word_len.median())
    }

    # 17. Review title length distribution
    title_char_len = df['review_title'].fillna('').astype(str).str.len()
    title_word_len = df['review_title'].fillna('').astype(str).str.split().str.len()
    title_length_stats = {
        "char_mean": float(title_char_len.mean()),
        "char_std": float(title_char_len.std()),
        "char_min": int(title_char_len.min()),
        "char_max": int(title_char_len.max()),
        "char_median": float(title_char_len.median()),
        "word_mean": float(title_word_len.mean()),
        "word_std": float(title_word_len.std()),
        "word_min": int(title_word_len.min()),
        "word_max": int(title_word_len.max()),
        "word_median": float(title_word_len.median())
    }

    # 18. Product category distribution
    cat_dist = df['product_category'].value_counts().to_dict()

    # 19. Star distribution
    star_dist = df['stars'].value_counts().sort_index().to_dict()

    # 20. Derived sentiment distribution
    def map_sentiment(s):
        if s in [1, 2]:
            return 'Negative'
        elif s == 3:
            return 'Neutral'
        elif s in [4, 5]:
            return 'Positive'
        return 'Invalid'

    df['derived_sentiment'] = df['stars'].apply(map_sentiment)
    sentiment_dist = df['derived_sentiment'].value_counts().to_dict()
    sentiment_pct = (df['derived_sentiment'].value_counts(normalize=True) * 100).to_dict()

    # 21. Category x sentiment distribution
    cat_sent_cross = pd.crosstab(df['product_category'], df['derived_sentiment']).to_dict(orient='index')

    # 22. Category x star distribution
    cat_star_cross = pd.crosstab(df['product_category'], df['stars']).to_dict(orient='index')

    audit_json = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_dtypes": col_dtypes,
        "missing_values": missing_vals,
        "empty_strings": empty_strings,
        "invalid_star_values_count": invalid_star_count,
        "language_distribution": lang_dist,
        "unique_review_ids": uniq_review_ids,
        "unique_product_ids": uniq_product_ids,
        "unique_reviewer_ids": uniq_reviewer_ids,
        "duplicate_complete_rows": dup_complete_rows,
        "duplicate_review_body_count": dup_review_bodies,
        "duplicate_review_title_count": dup_review_titles,
        "product_reviews_per_product_stats": prod_reviews_stats,
        "reviews_per_reviewer_stats": reviewer_stats,
        "review_body_length_stats": body_length_stats,
        "review_title_length_stats": title_length_stats,
        "product_category_distribution": cat_dist,
        "star_distribution": star_dist,
        "derived_sentiment_distribution": sentiment_dist,
        "derived_sentiment_percentages": sentiment_pct,
        "category_x_sentiment_distribution": cat_sent_cross,
        "category_x_star_distribution": cat_star_cross
    }

    # Format text report
    lines = []
    lines.append("==================================================")
    lines.append("       MULTI-DOMAIN DATASET AUDIT REPORT          ")
    lines.append("==================================================")
    lines.append(f"Source Dataset: {data_path}")
    lines.append(f"1. Total Rows: {total_rows}")
    lines.append(f"2. Total Columns: {total_cols}")
    lines.append("\n3. Column Data Types:")
    for col, dt in col_dtypes.items():
        lines.append(f"   - {col}: {dt}")

    lines.append("\n4. Missing Values:")
    for col, mv in missing_vals.items():
        lines.append(f"   - {col}: {mv}")

    lines.append("\n5. Empty / Whitespace-only Strings:")
    for col, es in empty_strings.items():
        lines.append(f"   - {col}: {es}")

    lines.append(f"\n6. Invalid Star Values Count: {invalid_star_count}")
    lines.append(f"7. Language Distribution: {lang_dist}")
    lines.append(f"8. Unique Review IDs: {uniq_review_ids} (Total rows: {total_rows})")
    lines.append(f"9. Unique Product IDs: {uniq_product_ids}")
    lines.append(f"10. Unique Reviewer IDs: {uniq_reviewer_ids}")
    lines.append(f"11. Duplicate Complete Rows: {dup_complete_rows}")
    lines.append(f"12. Duplicate review_body Values: {dup_review_bodies}")
    lines.append(f"13. Duplicate review_title Values: {dup_review_titles}")

    lines.append("\n14. Product Reviews per Product Statistics:")
    for k, v in prod_reviews_stats.items():
        lines.append(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

    lines.append("\n15. Reviews per Reviewer Statistics:")
    for k, v in reviewer_stats.items():
        lines.append(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

    lines.append("\n16. Review Body Text Length Distribution:")
    for k, v in body_length_stats.items():
        lines.append(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

    lines.append("\n17. Review Title Text Length Distribution:")
    for k, v in title_length_stats.items():
        lines.append(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

    lines.append("\n18. Product Category Distribution:")
    for cat, cnt in cat_dist.items():
        pct = (cnt / total_rows) * 100
        lines.append(f"   - {cat:<26}: {cnt:>6} ({pct:5.2f}%)")

    lines.append("\n19. Star Distribution:")
    for star, cnt in star_dist.items():
        pct = (cnt / total_rows) * 100
        lines.append(f"   - Star {star}: {cnt:>6} ({pct:5.2f}%)")

    lines.append("\n20. Derived Sentiment Distribution (1-2: Neg, 3: Neu, 4-5: Pos):")
    for sent, cnt in sentiment_dist.items():
        pct = sentiment_pct[sent]
        lines.append(f"   - {sent:<10}: {cnt:>6} ({pct:5.2f}%)")

    lines.append("\n21. Category x Sentiment Distribution:")
    lines.append(f"   {'Category':<26} | {'Negative':<8} | {'Neutral':<8} | {'Positive':<8}")
    lines.append("   " + "-" * 58)
    for cat, s_dict in cat_sent_cross.items():
        neg = s_dict.get('Negative', 0)
        neu = s_dict.get('Neutral', 0)
        pos = s_dict.get('Positive', 0)
        lines.append(f"   {cat:<26} | {neg:<8} | {neu:<8} | {pos:<8}")

    lines.append("\n22. Category x Star Distribution:")
    lines.append(f"   {'Category':<26} | {'1 Star':<6} | {'2 Star':<6} | {'3 Star':<6} | {'4 Star':<6} | {'5 Star':<6}")
    lines.append("   " + "-" * 68)
    for cat, st_dict in cat_star_cross.items():
        s1 = st_dict.get(1, 0)
        s2 = st_dict.get(2, 0)
        s3 = st_dict.get(3, 0)
        s4 = st_dict.get(4, 0)
        s5 = st_dict.get(5, 0)
        lines.append(f"   {cat:<26} | {s1:<6} | {s2:<6} | {s3:<6} | {s4:<6} | {s5:<6}")

    text_report_content = "\n".join(lines)

    # Save to data/MULTI_DOMAIN_DATASET_AUDIT.txt and reports/
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open("data/MULTI_DOMAIN_DATASET_AUDIT.txt", "w", encoding="utf-8") as f:
        f.write(text_report_content)

    with open("reports/MULTI_DOMAIN_DATASET_AUDIT.txt", "w", encoding="utf-8") as f:
        f.write(text_report_content)

    with open("reports/MULTI_DOMAIN_DATASET_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2)

    print("Phase 1 Audit complete. Created data/MULTI_DOMAIN_DATASET_AUDIT.txt, reports/MULTI_DOMAIN_DATASET_AUDIT.txt, and reports/MULTI_DOMAIN_DATASET_AUDIT.json")
    return audit_json

if __name__ == "__main__":
    run_dataset_audit()
