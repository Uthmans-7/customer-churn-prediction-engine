from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def compute_classification_scorecard(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Computes a complete categorization evaluation suite including Accuracy, 
    Precision, Recall, and F1-Score.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # FIX: Explicitly enforce a 2x2 matrix shape by mapping all valid classes
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    print("\n--- Binary Classification Scorecard ---")
    print(f"Accuracy (Overall Correctness):   {acc * 100:.2f}%")
    print(f"Precision (False Alarm Defense):  {prec * 100:.2f}%")
    print(f"Recall (Missed Target Defense):   {rec * 100:.2f}%")
    print(f"F1-Score (Harmonic Balance):      {f1:.4f}")
    
    print("\n--- Raw Confusion Matrix Layout ---")
    print(f"True Negatives (Stayed Correctly):  {cm[0][0]} | False Positives (False Alarms): {cm[0][1]}")
    print(f"False Negatives (Missed Churn):     {cm[1][0]} | True Positives (Churn Correct):  {cm[1][1]}")
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}