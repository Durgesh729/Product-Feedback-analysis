import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def map_sentiment(stars):
    if stars in [1, 2]:
        return 'Negative'
    elif stars == 3:
        return 'Neutral'
    elif stars in [4, 5]:
        return 'Positive'
    return 'Invalid'

def perform_stratified_group_split(data_path="data/cleaned_multi_domain_reviews.csv", seed=42):
    print(f"--- STARTING PHASE 6: LEAKAGE-SAFE STRATIFIED GROUP SPLITTING ---")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found! Run cleaning step first.")

    df = pd.read_csv(data_path)
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['stars'].apply(map_sentiment)

    total_len = len(df)
    print(f"Loaded {total_len} clean reviews for splitting.")

    # Objective: 70% Train, 15% Validation, 15% Test
    # Grouped by product_id so NO product_id crosses splits.
    # We evaluate candidate group splits to find the partition with minimal class distribution divergence.

    pop_dist = df['sentiment'].value_counts(normalize=True)

    best_splits = None
    min_divergence = float('inf')

    # We iterate seeds around 42 to find optimal product grouping stratification
    for trial_seed in range(seed, seed + 100):
        # Step 1: Split into Train (70%) and Temp (30%)
        gss_outer = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=trial_seed)
        train_idx, temp_idx = next(gss_outer.split(df, groups=df['product_id']))

        df_train = df.iloc[train_idx]
        df_temp = df.iloc[temp_idx]

        # Step 2: Split Temp into Validation (50% of Temp = 15% total) and Test (50% of Temp = 15% total)
        gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=trial_seed)
        val_idx_rel, test_idx_rel = next(gss_inner.split(df_temp, groups=df_temp['product_id']))

        df_val = df_temp.iloc[val_idx_rel]
        df_test = df_temp.iloc[test_idx_rel]

        # Calculate class distribution divergence from population
        train_dist = df_train['sentiment'].value_counts(normalize=True)
        val_dist = df_val['sentiment'].value_counts(normalize=True)
        test_dist = df_test['sentiment'].value_counts(normalize=True)

        div = 0.0
        for cls in ['Negative', 'Neutral', 'Positive']:
            div += abs(train_dist.get(cls, 0) - pop_dist.get(cls, 0))
            div += abs(val_dist.get(cls, 0) - pop_dist.get(cls, 0))
            div += abs(test_dist.get(cls, 0) - pop_dist.get(cls, 0))

        if div < min_divergence:
            min_divergence = div
            best_splits = (df_train, df_val, df_test, trial_seed)

    df_train, df_val, df_test, selected_seed = best_splits

    # Verify zero product_id leakage
    train_prods = set(df_train['product_id'])
    val_prods = set(df_val['product_id'])
    test_prods = set(df_test['product_id'])

    train_val_overlap = len(train_prods.intersection(val_prods))
    train_test_overlap = len(train_prods.intersection(test_prods))
    val_test_overlap = len(val_prods.intersection(test_prods))

    assert train_val_overlap == 0, f"Leakage detected! Train-Val product overlap: {train_val_overlap}"
    assert train_test_overlap == 0, f"Leakage detected! Train-Test product overlap: {train_test_overlap}"
    assert val_test_overlap == 0, f"Leakage detected! Val-Test product overlap: {val_test_overlap}"

    print(f"\nSUCCESS: Zero product leakage verified!")
    print(f"Optimal Stratified Group Split selected (Candidate seed: {selected_seed}, Class divergence score: {min_divergence:.6f})")

    # Save split indices / split column to CSV or return splits
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'

    df_split_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

    # Document exact resulting distributions
    lines = []
    lines.append("==================================================")
    lines.append("   LEAKAGE-SAFE STRATIFIED GROUP SPLIT REPORT     ")
    lines.append("==================================================")
    lines.append(f"Splitting Group Column: product_id")
    lines.append(f"Base Random Seed: {seed} (Selected Trial Seed: {selected_seed})")
    lines.append(f"Zero Product Leakage: VERIFIED (Train-Val: 0, Train-Test: 0, Val-Test: 0)")
    lines.append("\nSPLIT SIZES & PERCENTAGES:")
    lines.append(f"Total Cleaned Rows: {total_len}")
    lines.append(f"Train Set:      {len(df_train):>7} rows ({len(df_train)/total_len*100:5.2f}%) | Unique Products: {len(train_prods)}")
    lines.append(f"Validation Set: {len(df_val):>7} rows ({len(df_val)/total_len*100:5.2f}%) | Unique Products: {len(val_prods)}")
    lines.append(f"Test Set:       {len(df_test):>7} rows ({len(df_test)/total_len*100:5.2f}%) | Unique Products: {len(test_prods)}")

    lines.append("\nEXACT SENTIMENT CLASS DISTRIBUTIONS:")
    lines.append(f"{'Split':<12} | {'Negative':<18} | {'Neutral':<18} | {'Positive':<18}")
    lines.append("-" * 72)

    for name, s_df in [('Population', df), ('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
        counts = s_df['sentiment'].value_counts()
        pcts = s_df['sentiment'].value_counts(normalize=True) * 100
        neg_str = f"{counts.get('Negative',0)} ({pcts.get('Negative',0):.2f}%)"
        neu_str = f"{counts.get('Neutral',0)} ({pcts.get('Neutral',0):.2f}%)"
        pos_str = f"{counts.get('Positive',0)} ({pcts.get('Positive',0):.2f}%)"
        lines.append(f"{name:<12} | {neg_str:<18} | {neu_str:<18} | {pos_str:<18}")

    split_report = "\n".join(lines)
    print(split_report)

    # Append to design report
    with open("reports/MULTI_DOMAIN_DATASET_DESIGN_REPORT.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + split_report)

    return df_train, df_val, df_test

if __name__ == "__main__":
    perform_stratified_group_split()
