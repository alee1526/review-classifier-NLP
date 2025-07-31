from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report

class MLModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report