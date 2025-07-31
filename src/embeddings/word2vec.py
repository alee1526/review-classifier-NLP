# src/embeddings/word2vec.py
import gensim
import numpy as np
from .base import EmbeddingGenerator
import joblib
import logging

class Word2VecEmbedding(EmbeddingGenerator):
    def __init__(self, vector_size=200, window=10, min_count=5):
        super().__init__()
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, texts):
        self.model = gensim.models.Word2Vec(
            sentences=[text.split() for text in texts],
            vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4, sg=1
        )
        self.is_fitted = True
        return self

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming.")
        def document_vector(doc):
            words = [word for word in doc.split() if word in self.model.wv.key_to_index]
            return np.mean([self.model.wv[word] for word in words] or [np.zeros(self.vector_size)], axis=0)
        return np.array([document_vector(text) for text in texts])

    def save(self, filepath):
        """Save the Word2VecEmbedding instance to a file."""
        joblib.dump({
            'model': self.model,
            'is_fitted': self.is_fitted,
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a Word2VecEmbedding instance from a file."""
        data = joblib.load(filepath)
        instance = cls(vector_size=data['vector_size'], window=data['window'], min_count=data['min_count'])
        instance.model = data['model']
        instance.is_fitted = data['is_fitted']
        return instance