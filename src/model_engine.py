import joblib
import os
from sklearn.linear_model import LogisticRegression
import numpy as np

class ChurnClassifierModel:
    """
    Manages training, prediction probability parsing, and storage 
    of the single-neuron linear classifier.
    """
    def __init__(self) -> None:
        self.model = LogisticRegression()

    def train(self, X_train: np.ndarray, y_train: list) -> None:
        """Trains the linear classifier using log-loss minimization."""
        self.model.fit(X_train, y_train)
        print("Classifier engine training complete.")
        print(f"Logit Intercept (Bias): {self.model.intercept_[0]:.4f}")
        
        for idx, weight in enumerate(self.model.coef_[0]):
            print(f"Feature [{idx}] Normalized Weight (Log-Odds): {weight:.4f}")

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predicts hard classification labels (0 or 1) using a default 0.5 threshold."""
        return self.model.predict(X)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Extracts raw Sigmoid activation output probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Classifier saved successfully to {filepath}")