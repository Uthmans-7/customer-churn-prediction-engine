import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_and_prep_classification_data(
    filepath: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Loads customer usage metrics, isolates features from targets,
    and returns scaled train and test splits optimized for classification.
    """
    df = pd.read_csv(filepath)
    
    # Isolate independent variables and the categorical target
    X = df[["tenure_months", "monthly_charges"]]
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features to equalize gradient updates and avoid sigmoid saturation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test