"""
Prediction Engine & Inference Module for Product Feedback Analysis

This module loads trained sentiment model and vectorizer to perform:
1. Single review sentiment prediction with confidence probability score.
2. Batch review prediction for uploaded CSV datasets with auto-detected text column.
3. Keyword extraction for customer feedback insights.
"""

import os
import sys
import datetime
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import clean_text, tokenize_text, remove_stopwords, lemmatize_tokens, preprocess_pipeline
from src.domain_validation import validate_product_feedback_domain


class SentimentPredictor:
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        # Default to multi_domain artifacts if present, fallback to legacy Model B
        if model_path is None:
            if os.path.exists('models/multi_domain_sentiment_model.pkl'):
                model_path = 'models/multi_domain_sentiment_model.pkl'
            else:
                model_path = 'models/sentiment_model.pkl'

        if vectorizer_path is None:
            if os.path.exists('models/multi_domain_tfidf_vectorizer.pkl'):
                vectorizer_path = 'models/multi_domain_tfidf_vectorizer.pkl'
            else:
                vectorizer_path = 'models/tfidf_vectorizer.pkl'

        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_resources()

    def load_resources(self):
        """
        Loads saved model and vectorizer from disk and prints startup logging.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please train the model first.")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {self.vectorizer_path}. Please train the model first.")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(self.model_path)).strftime("%Y-%m-%d %H:%M:%S")
        meta_path = 'models/model_metadata.pkl'
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        build_timestamp = meta.get('build_timestamp', mtime)

        print("\n==================================================")
        print("API MODEL INITIALIZATION LOG")
        print("==================================================")
        print(f"Loaded sentiment model:          {os.path.abspath(self.model_path)}")
        print(f"Loaded vectorizer:               {os.path.abspath(self.vectorizer_path)}")
        print(f"Model classes:                   {list(self.model.classes_)}")
        print(f"Model trained version/timestamp: {build_timestamp} (File MTime: {mtime})")
        print("==================================================\n")

    def predict_single(self, text: str) -> dict:
        """
        Predicts sentiment for a single user review text string with Domain Validation and NLP step breakdown.
        """
        if not text or not str(text).strip():
            return {
                'is_valid_domain': False,
                'domain_status': 'Outside Product Feedback Domain',
                'domain_message': 'Input text is empty. Please enter feedback about a product or service.',
                'sentiment': 'Neutral',
                'predicted_class_probability': 0.0,
                'confidence': 0.0,
                'probabilities': {'Positive': 33.33, 'Negative': 33.33, 'Neutral': 33.34},
                'cleaned_text': '',
                'nlp_steps': {}
            }

        # 1. Domain Validation Step
        domain_info = validate_product_feedback_domain(text)
        
        # Breakdown NLP steps
        cleaned = clean_text(text)
        tokens = tokenize_text(text)
        filtered = remove_stopwords(tokens)
        lemmas = lemmatize_tokens(filtered)
        processed = ' '.join(lemmas)
        
        vec = self.vectorizer.transform([processed if processed else cleaned])
        nnz_features = vec.nnz
        vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 5000

        nlp_steps = {
            'original_text': text,
            'cleaned_text': cleaned,
            'tokens': tokens,
            'filtered_tokens': filtered,
            'lemmatized_tokens': lemmas,
            'processed_text': processed,
            'num_tokens': len(tokens),
            'num_nonzero_features': int(nnz_features),
            'vocabulary_size': int(vocab_size),
            'model_used': "TF-IDF + Logistic Regression (class_weight='balanced')"
        }

        if not domain_info['is_product_feedback']:
            return {
                'is_valid_domain': False,
                'domain_status': domain_info['domain_status'],
                'domain_message': domain_info['message'],
                'sentiment': None,
                'predicted_class_probability': 0.0,
                'confidence': 0.0,
                'probabilities': {'Positive': 0.0, 'Negative': 0.0, 'Neutral': 0.0},
                'cleaned_text': cleaned,
                'nlp_steps': nlp_steps
            }

        # 2. ML Sentiment Inference for Valid Domain Inputs
        pred_label = self.model.predict(vec)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = self.model.classes_
            prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
            max_prob = float(np.max(probs))
        else:
            prob_dict = {pred_label: 1.0}
            max_prob = 1.0

        prob_dict_percent = {k: round(v * 100, 2) for k, v in prob_dict.items()}
        max_prob_percent = round(max_prob * 100, 2)

        return {
            'is_valid_domain': True,
            'domain_status': domain_info['domain_status'],
            'domain_message': domain_info['message'],
            'sentiment': pred_label,
            'predicted_class_probability': max_prob_percent,
            'confidence': max_prob_percent,  # Backward compatibility key
            'probabilities': prob_dict_percent,
            'cleaned_text': processed if processed else cleaned,
            'nlp_steps': nlp_steps
        }


    def predict_batch(self, df: pd.DataFrame, text_column: str = None) -> tuple:
        """
        Auto-detects text column if not specified, cleans reviews, predicts sentiment,
        and adds 'predicted_sentiment' and 'confidence' columns.
        Returns (processed_df, detected_column_name).
        """
        possible_cols = ['reviewText', 'review', 'review_text', 'text', 'comment', 'feedback', 'summary', 'body']
        
        selected_col = text_column
        if not selected_col:
            for col in possible_cols:
                if col in df.columns:
                    selected_col = col
                    break
                    
        if not selected_col:
            # Fallback: select first string object column
            string_cols = df.select_dtypes(include=['object', 'string']).columns
            if len(string_cols) > 0:
                selected_col = string_cols[0]
            else:
                raise ValueError("No text column found in CSV. Please ensure your file has a review column.")

        out_df = df.copy()
        
        print(f"Processing batch predictions using review column: '{selected_col}'...")
        cleaned_texts = out_df[selected_col].astype(str).apply(lambda t: preprocess_pipeline(t))
        vecs = self.vectorizer.transform(cleaned_texts)
        
        preds = self.model.predict(vecs)
        out_df['predicted_sentiment'] = preds
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vecs)
            confidences = np.max(probs, axis=1) * 100
            out_df['confidence'] = np.round(confidences, 2)
        else:
            out_df['confidence'] = 100.0
            
        return out_df, selected_col

    def extract_top_words(self, df: pd.DataFrame, sentiment_label: str, text_col: str = 'processed_text', top_n: int = 10) -> list:
        """
        Extracts most frequent words for a given sentiment category for basic feedback insights.
        """
        if text_col not in df.columns:
            return []
            
        sub_df = df[df['sentiment'] == sentiment_label] if 'sentiment' in df.columns else df[df['predicted_sentiment'] == sentiment_label]
        if sub_df.empty:
            return []
            
        all_words = ' '.join(sub_df[text_col].dropna()).split()
        if not all_words:
            return []
            
        freq_series = pd.Series(all_words).value_counts()
        return freq_series.head(top_n).to_dict()
