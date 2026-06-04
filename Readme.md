
#  Telecom Customer Churn Predictor

A modular, production-grade binary classification engine demonstrating the application of parametric logistic models to predict customer subscription churn. The pipeline transforms continuous usage features through a Sigmoid activation layer to compute explicit Bernoulli probabilities, optimized via Binary Cross-Entropy minimization.

## Project Architecture
This repository follows  modular software engineering separation of concerns, isolating data parsing, optimization engines, and categorical scoring evaluation suites.

```text
customer_churn_predictor/
│
├── data/                       # Structural user account usage logs
│   └── customer_churn.csv
│
├── models/                     # Serialized binary model weights (.pkl)
│
├── src/
│   ├── data_pipeline.py        # Ingestion, train/test splitting, and feature normalization
│   ├── model_engine.py         # Logistic solver training, logit tracking, and disk serialization
│   └── evaluator.py            # Categorical scorecard computation (Precision, Recall, F1, Confusion Matrix)
│
├── main.py                     # Primary execution conductor
└── requirements.txt            # Explicit dependency tracker

```

## Core Machine Learning Mechanics

### 1. Sigmoid Saturation & Feature Normalization

Because Logistic Regression passes the linear combination $z = W^T X + b$ through the non-linear Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$, feature scaling via `StandardScaler` is computationally essential. Unscaled inputs can create massive raw $z$ values, pushing the model into the flat outer saturation zones of the Sigmoid curve where gradients approach zero ($ \nabla \approx 0 $). This stalls weight updates during optimization.

### 2. Optimization via Convex Log-Loss

Standard Mean Squared Error (MSE) combined with a non-linear Sigmoid function results in a non-convex error surface full of local minima and saddle points. To guarantee global convergence via gradient descent, this engine optimizes parameters by minimizing Binary Cross-Entropy (Log Loss):

$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

This penalizes the model exponentially when it outputs highly confident incorrect classifications, mirroring backpropagation setups in deep learning multi-layer perceptrons (MLPs).

### 3. Log-Odds Parameter Analysis

The model transforms inputs into a linear decision boundary hyperplane where the threshold probability is exactly $0.5$ (occurring precisely when $W^T X + b = 0$). The trained model weights correspond directly to log-odds shifts:

* **Tenure Months (Weight: -0.8007):** Negative correlation. As user tenure increases, the log-odds of churning drop significantly, mapping long-term customer stability.
* **Monthly Charges (Weight: 1.0445):** Positive correlation. High subscription costs serve as the primary mathematical accelerator driving user cancellation.

## Classification Scorecard

| Metric | Valuation | Production Significance |
| --- | --- | --- |
| **Accuracy** | 100.00% | Overall correctness score across the verification subset. |
| **Precision** | 100.00% | False Alarm defense rate; essential for optimizing retention budget allocations. |
| **Recall** | 100.00% | Missed Target defense rate; crucial for catching at-risk accounts before termination. |
| **F1-Score** | 1.0000 | Harmonic mean of precision and recall, balancing predictions under class imbalances. |

## How to Execute

Ensure your virtual environment is active and project dependencies are installed, then trigger the master pipeline orchestrator directly from the terminal:

```bash
# Activate the virtual environment sandbox
source venv/bin/activate

# Execute the complete pipeline from ingestion to serialized output
python main.py

```

