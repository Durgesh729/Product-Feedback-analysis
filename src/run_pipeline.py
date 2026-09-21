import sys
import os
import argparse

# Add src to python path
sys.path.insert(0, os.path.dirname(__file__))

from data_audit import run_dataset_audit
from data_cleaning import analyze_and_clean_data
from data_split import perform_stratified_group_split
from model_training import run_model_training_experiments, train_final_multi_domain_model
from model_evaluation import evaluate_multi_domain_model
from domain_evaluation import run_domain_wise_evaluation
from unseen_test import generate_and_evaluate_unseen_test
from api_verification import verify_direct_vs_api_parity

def run_full_pipeline():
    print("==========================================================")
    print("  EXECUTING MULTI-DOMAIN DATASET AUDIT & REDESIGN PIPELINE")
    print("==========================================================")

    # Phase 1: Dataset Audit
    audit_json = run_dataset_audit("train.csv")

    # Phase 2 & 5: Data Quality Analysis, Duplicate Classification & Cleaning
    df_clean, report_text = analyze_and_clean_data("train.csv", "data/cleaned_multi_domain_reviews.csv")

    # Phase 6: Leakage-Safe Stratified Group Split
    df_train, df_val, df_test = perform_stratified_group_split("data/cleaned_multi_domain_reviews.csv", seed=42)

    # Phase 7 & 8: Text Representation Experiments & Model Selection
    best_exp_config, exp_results = run_model_training_experiments(df_train, df_val)

    # Retrain Final Model on Train + Validation
    clf, vec = train_final_multi_domain_model(df_train, df_val, best_exp_config)

    # Phase 8: Final Model Evaluation on untouched Test Set
    evaluate_multi_domain_model(df_test)

    # Phase 9: Domain-Wise Performance Evaluation
    run_domain_wise_evaluation(df_test)

    # Phase 10: 135-Example Unseen Manual Test & Qualitative Dental Test
    generate_and_evaluate_unseen_test("data/cleaned_multi_domain_reviews.csv")

    # Phase 11: Direct Model vs API Parity Verification
    verify_direct_vs_api_parity()

    print("\n==========================================================")
    print("  ALL PIPELINE PHASES (1 THROUGH 11) EXECUTED SUCCESSFULLY!")
    print("==========================================================")

if __name__ == "__main__":
    run_full_pipeline()
