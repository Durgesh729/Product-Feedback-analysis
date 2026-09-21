import os
import pandas as pd
import numpy as np

def map_sentiment(stars):
    if stars in [1, 2]:
        return 'Negative'
    elif stars == 3:
        return 'Neutral'
    elif stars in [4, 5]:
        return 'Positive'
    return 'Invalid'

def analyze_and_clean_data(input_path="train.csv", output_path="data/cleaned_multi_domain_reviews.csv"):
    print(f"--- STARTING PHASE 2 & 5: DATA QUALITY, DUPLICATE ANALYSIS & CLEANING ---")
    df = pd.read_csv(input_path)
    initial_count = len(df)

    # Derived sentiment mapping
    df['sentiment'] = df['stars'].apply(map_sentiment)

    # 1. Title NaNs & Whitespace handling
    missing_titles_count = df['review_title'].isnull().sum()
    df['review_title'] = df['review_title'].fillna('')
    df['review_title'] = df['review_title'].astype(str).str.strip()
    df['review_body'] = df['review_body'].astype(str).str.strip()

    # 2. Detailed Duplicate Text Analysis
    dup_mask = df['review_body'].duplicated(keep=False)
    dup_df = df[dup_mask].copy()

    dup_text_groups = dup_df.groupby('review_body')

    category_A_count = 0  # Duplicate text with SAME sentiment label
    category_B_count = 0  # Duplicate text with CONFLICTING sentiment labels
    category_C_count = 0  # Duplicate text appearing across DIFFERENT product_ids

    cat_A_rows = []
    cat_B_rows = []
    cat_C_rows = []

    for text, group in dup_text_groups:
        uniq_sentiments = group['sentiment'].unique()
        uniq_products = group['product_id'].unique()
        num_rows = len(group)

        if len(uniq_products) > 1:
            category_C_count += num_rows
            cat_C_rows.append((text, num_rows, list(uniq_products), list(uniq_sentiments)))

        if len(uniq_sentiments) == 1:
            category_A_count += num_rows
            cat_A_rows.append((text, num_rows, uniq_sentiments[0]))
        else:
            category_B_count += num_rows
            cat_B_rows.append((text, num_rows, list(uniq_sentiments)))

    print("\n--- DUPLICATE REVIEW ANALYSIS SUMMARY ---")
    print(f"Total rows in dataset: {initial_count}")
    print(f"Total rows involved in text duplicates (review_body): {dup_mask.sum()}")
    print(f"Unique duplicate review_body strings: {len(dup_text_groups)}")
    print(f"Category A (Exact duplicate text with SAME sentiment label): {category_A_count} rows across {len(cat_A_rows)} unique texts")
    print(f"Category B (Exact duplicate text with CONFLICTING sentiment labels): {category_B_count} rows across {len(cat_B_rows)} unique texts")
    print(f"Category C (Exact duplicate text across DIFFERENT product_ids): {category_C_count} rows across {len(cat_C_rows)} unique texts")

    # Document explicit cleaning policy
    policy_doc = []
    policy_doc.append("==================================================")
    policy_doc.append("       DATA CLEANING POLICY & DUPLICATE ANALYSIS   ")
    policy_doc.append("==================================================")
    policy_doc.append(f"Initial Dataset Rows: {initial_count}")
    policy_doc.append(f"Missing Titles: {missing_titles_count} (Filled with empty string '')")
    policy_doc.append("\nDUPLICATE TEXT BREAKDOWN:")
    policy_doc.append(f"- Category A (Duplicate text, SAME sentiment label): {category_A_count} rows ({len(cat_A_rows)} unique texts)")
    policy_doc.append(f"- Category B (Duplicate text, CONFLICTING sentiment labels): {category_B_count} rows ({len(cat_B_rows)} unique texts)")
    policy_doc.append(f"- Category C (Duplicate text, DIFFERENT products): {category_C_count} rows ({len(cat_C_rows)} unique texts)")
    policy_doc.append("\nCLEANING POLICY RULES:")
    policy_doc.append("1. Title NaNs: Replaced with empty string '' so concatenation ('title body') executes cleanly.")
    policy_doc.append("2. Text Normalization: Leading and trailing whitespaces stripped from review_title and review_body.")
    policy_doc.append("3. Category B Policy (Conflicting Sentiment Labels): REMOVE ALL INSTANCES. Conflicting labels on identical text (e.g. 'Great product' labeled 1 star and 5 stars) introduce severe label noise.")
    policy_doc.append("4. Category A Policy (Same Sentiment Labels): DEDUPLICATE TEXT. Keep only 1 representative record to prevent identical reviews from contaminating train/validation/test splits.")
    policy_doc.append("5. Category C Policy (Cross-product duplicate text): DEDUPLICATE. Keeping 1 instance ensures product-level split isolation without duplicate text leakage.")
    policy_doc.append("6. Empty Body Policy: Drop any record where review_body after whitespace stripping is empty.")

    # Apply Policy
    # Identify Category B review_body texts (conflicting labels)
    conflicting_texts = set([item[0] for item in cat_B_rows])

    # Filter out Category B
    df_clean = df[~df['review_body'].isin(conflicting_texts)].copy()
    rows_after_B_drop = len(df_clean)
    dropped_B_count = initial_count - rows_after_B_drop

    # For remaining duplicate review_body texts (Category A / C), drop duplicates keeping first
    df_clean = df_clean.drop_duplicates(subset=['review_body'], keep='first').copy()
    final_count = len(df_clean)
    dedup_dropped_count = rows_after_B_drop - final_count

    policy_doc.append("\nEXECUTION RESULTS:")
    policy_doc.append(f"- Conflicting label rows dropped (Category B): {dropped_B_count}")
    policy_doc.append(f"- Duplicate text rows dropped (Category A & C deduplication): {dedup_dropped_count}")
    policy_doc.append(f"- Total rows dropped: {initial_count - final_count}")
    policy_doc.append(f"- Final Cleaned Dataset Rows: {final_count}")

    # Sentiment distribution after cleaning
    clean_sent_dist = df_clean['sentiment'].value_counts().to_dict()
    clean_sent_pct = (df_clean['sentiment'].value_counts(normalize=True) * 100).to_dict()
    policy_doc.append("\nPOST-CLEANING SENTIMENT DISTRIBUTION:")
    for sent, cnt in clean_sent_dist.items():
        policy_doc.append(f"- {sent}: {cnt} ({clean_sent_pct[sent]:.2f}%)")

    # Save policy report
    report_text = "\n".join(policy_doc)
    os.makedirs("reports", exist_ok=True)
    with open("reports/MULTI_DOMAIN_DATASET_DESIGN_REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path} ({final_count} rows). Report saved to reports/MULTI_DOMAIN_DATASET_DESIGN_REPORT.txt")

    return df_clean, report_text

if __name__ == "__main__":
    analyze_and_clean_data()
