"""
Model Evaluation Module for Product Feedback Analysis

This module calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score),
generates plots (Confusion Matrix, Sentiment Distribution), and exports structured evaluation reports.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def compute_metrics(y_true, y_pred) -> dict:
    """
    Computes overall accuracy, macro and weighted precision, recall, and F1 scores.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    return {
        'accuracy': acc,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1
    }


def plot_confusion_matrix(y_true, y_pred, labels=('Negative', 'Neutral', 'Positive'), output_path='outputs/confusion_matrix.png'):
    """
    Generates and saves a publication-quality Confusion Matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title('Confusion Matrix — Sentiment Classification', fontsize=14, pad=12)
    plt.xlabel('Predicted Sentiment', fontsize=12)
    plt.ylabel('Actual Sentiment', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = 'sentiment', output_path: str = 'outputs/sentiment_distribution.png'):
    """
    Generates and saves a sentiment distribution bar chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = df[sentiment_col].value_counts()
    
    colors = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
    bar_colors = [colors.get(label, '#3498db') for label in counts.index]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values, color=bar_colors, edgecolor='black', alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts.values)*0.01,
                 f'{height}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.title('Dataset Sentiment Distribution', fontsize=14, pad=12)
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Sentiment distribution plot saved to: {output_path}")


def save_evaluation_report(y_true, y_pred, model_name: str = "Logistic Regression", output_path: str = "outputs/evaluation_results.txt"):
    """
    Exports a detailed classification report and metric summary into a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_str = classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
    metrics = compute_metrics(y_true, y_pred)
    
    content = f"""==================================================
MODEL EVALUATION REPORT
==================================================
Model: {model_name}
Dataset Size: {len(y_true)} test samples

--------------------------------------------------
METRICS SUMMARY
--------------------------------------------------
Accuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
Weighted Precision:  {metrics['weighted_precision']:.4f}
Weighted Recall:     {metrics['weighted_recall']:.4f}
Weighted F1-Score:   {metrics['weighted_f1']:.4f}

Macro Precision:     {metrics['macro_precision']:.4f}
Macro Recall:        {metrics['macro_recall']:.4f}
Macro F1-Score:      {metrics['macro_f1']:.4f}

--------------------------------------------------
DETAILED CLASSIFICATION REPORT
--------------------------------------------------
{report_str}
==================================================
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Evaluation report saved to: {output_path}")
    return content
