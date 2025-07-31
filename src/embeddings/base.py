from abc import ABC, abstractmethod

class EmbeddingGenerator(ABC):
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, texts):
        pass
    
    @abstractmethod
    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)