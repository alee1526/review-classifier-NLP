# src/embeddings/tfidf.py
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import EmbeddingGenerator
import joblib
import logging

class TfidfEmbedding(EmbeddingGenerator):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self.is_fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming.")
        return self.vectorizer.transform(texts).toarray()

    def save(self, filepath):
        """Save the TfidfEmbedding instance to a file."""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a TfidfEmbedding instance from a file."""
        data = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = data['vectorizer']
        instance.is_fitted = data['is_fitted']
        return instance