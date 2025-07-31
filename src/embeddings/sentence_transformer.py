# src/embeddings/sentence_transformer.py
from sentence_transformers import SentenceTransformer
from .base import EmbeddingGenerator
import joblib
import logging

class SentenceTransformerEmbedding(EmbeddingGenerator):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # Guardar el nombre del modelo como atributo de la clase
        self.is_fitted = True  # Preentrenado, no requiere fit adicional

    def fit(self, texts):
        # No necesita fit, ya que el modelo está preentrenado
        return self

    def transform(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)

    def save(self, filepath):
        """Save the SentenceTransformerEmbedding instance to a file."""
        # No serializamos el modelo completo (es grande y se recarga), solo el nombre y el estado
        joblib.dump({
            'model_name': self.model_name,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a SentenceTransformerEmbedding instance from a file."""
        data = joblib.load(filepath)
        instance = cls(model_name=data['model_name'])
        instance.is_fitted = data['is_fitted']
        return instance