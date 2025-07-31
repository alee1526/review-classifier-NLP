from xgboost import XGBClassifier
from .base import MLModel

class XGBModel(MLModel):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        super().__init__()
        self.model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)