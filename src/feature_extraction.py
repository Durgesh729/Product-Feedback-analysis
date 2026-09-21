"""
Feature Extraction Module for Product Feedback Analysis

This module handles converting preprocessed text into numerical feature vectors
using Term Frequency - Inverse Document Frequency (TF-IDF) Vectorization.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 10000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Creates and configures a TF-IDF Vectorizer instance.
    - max_features: caps vocabulary size to top features
    - ngram_range: includes unigrams and bigrams (e.g. 'great', 'great quality')
    - min_df: minimum document frequency (set to 1 to retain rare negative/neutral terms)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1
    )
    return vectorizer


def fit_transform_tfidf(vectorizer: TfidfVectorizer, corpus: list):
    """
    Fits vectorizer on text corpus and returns feature matrix.
    """
    return vectorizer.fit_transform(corpus)


def transform_tfidf(vectorizer: TfidfVectorizer, corpus: list):
    """
    Transforms text corpus using an already fitted vectorizer.
    """
    return vectorizer.transform(corpus)


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str):
    """
    Saves fitted TF-IDF vectorizer model using joblib serialization.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(vectorizer, filepath)
    print(f"TF-IDF Vectorizer successfully saved to {filepath}")


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """
    Loads saved TF-IDF vectorizer model from file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Vectorizer file not found at: {filepath}")
    return joblib.load(filepath)
