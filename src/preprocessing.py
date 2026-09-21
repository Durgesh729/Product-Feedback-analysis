"""
NLP Preprocessing Module for Product Feedback Analysis

This module performs fundamental NLP text preprocessing steps:
1. Text Cleaning (lowercasing, special characters, URLs, numbers removal)
2. Tokenization (regex / NLTK)
3. Stop-word Removal
4. Stemming / Lemmatization (self-contained & offline-safe)
"""

import re
import nltk

# Self-contained english stopwords set for guaranteed offline operation
DEFAULT_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
    'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he',
    "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
    'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'let',
    "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only',
    'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd",
    "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
    "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we',
    "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's",
    'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd",
    "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
}

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set(DEFAULT_STOPWORDS)

# Crucial NLP Fix: Exclude negation words from stop-words set to retain negation context in TF-IDF
NEGATION_WORDS = {
    'no', 'nor', 'not', 'never', 'none', 'neither', 'cannot', 'cant', 'dont',
    'isnt', 'didnt', 'wasnt', 'wouldnt', 'shouldnt', 'couldnt', 'hasnt',
    'havent', 'hadnt', 'without', "can't", "don't", "isn't", "didn't",
    "wasn't", "wouldn't", "shouldn't", "couldn't", "hasn't", "haven't", "hadn't"
}
stop_words = stop_words - NEGATION_WORDS

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
    stemmer = None

try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
except Exception:
    lemmatizer = None


def expand_contractions(text: str) -> str:
    """
    Expands English negative contractions before punctuation removal so that
    negation words ('not', 'cannot') are preserved for tokenization.
    """
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"\bcan['’]t\b", "cannot", text)
    text = re.sub(r"\bwon['’]t\b", "will not", text)
    text = re.sub(r"\bshan['’]t\b", "shall not", text)
    text = re.sub(r"\b(\w+)n['’]t\b", r"\1 not", text)
    return text


def clean_text(text: str) -> str:
    """
    Cleans raw input text:
    - Expands contractions (e.g., don't -> do not)
    - Converts to string and lowercases
    - Removes HTML tags and URLs
    - Removes non-alphabetic characters (numbers, punctuation)
    - Normalizes extra spaces
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    text = expand_contractions(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def tokenize_text(text: str) -> list:
    """
    Fast regex tokenization dividing text into individual word tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return re.findall(r'\b[a-z]{2,}\b', cleaned)


def remove_stopwords(tokens: list) -> list:
    """
    Removes standard English stop-words from token list.
    """
    return [token for token in tokens if token not in stop_words and len(token) > 1]


def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatizes word tokens to root form with robust fallback.
    """
    if lemmatizer:
        try:
            return [lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            pass
    # Fast suffix stripping fallback
    result = []
    for token in tokens:
        if token.endswith('ing') and len(token) > 5:
            result.append(token[:-3])
        elif token.endswith('ed') and len(token) > 4:
            result.append(token[:-2])
        elif token.endswith('es') and len(token) > 4:
            result.append(token[:-2])
        elif token.endswith('s') and not token.endswith('ss') and len(token) > 3:
            result.append(token[:-1])
        else:
            result.append(token)
    return result


def stem_tokens(tokens: list) -> list:
    """
    Stems word tokens using PorterStemmer.
    """
    if stemmer:
        return [stemmer.stem(token) for token in tokens]
    return tokens


def preprocess_pipeline(text: str, method: str = 'lemmatize') -> str:
    """
    Complete NLP Preprocessing Pipeline:
    Raw Text -> Cleaning -> Tokenization -> Stop-word Removal -> Lemmatization/Stemming -> Processed Text
    """
    tokens = tokenize_text(text)
    filtered = remove_stopwords(tokens)
    
    if method == 'stem':
        processed = stem_tokens(filtered)
    else:
        processed = lemmatize_tokens(filtered)
        
    return ' '.join(processed)


if __name__ == "__main__":
    sample = "The product quality is EXCELLENT! It arrived fast and I am very satisfied."
    print("Original Text: ", sample)
    print("Processed Text:", preprocess_pipeline(sample))
