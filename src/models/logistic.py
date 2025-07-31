from sklearn.linear_model import LogisticRegression
from .base import MLModel

class LogisticModel(MLModel):
    def __init__(self, C=1.0):
        super().__init__()
        self.model = LogisticRegression(C=C, max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)