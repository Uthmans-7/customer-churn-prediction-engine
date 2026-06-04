from src.data_pipeline import load_and_prep_classification_data
from src.model_engine import ChurnClassifierModel
from src.evaluator import compute_classification_scorecard

def execute_classification_pipeline():
    DATA_PATH = "data/customer_churn.csv"
    MODEL_OUTPUT_PATH = "models/churn_classifier.pkl"

    print("==================================================")
    print("STARTING TELECOM CHURN CLASSIFICATION PIPELINE")
    print("==================================================")

    # 1. Pipeline data ingestion and scalable prep
    X_train, X_test, y_train, y_test = load_and_prep_classification_data(DATA_PATH)
    print(f"Data split: {len(X_train)} training records, {len(X_test)} verification records.")

    # 2. Train the single-neuron classification engine
    classifier = ChurnClassifierModel()
    classifier.train(X_train, y_train)

    # 3. Perform inference on hidden validation records
    y_pred_classes = classifier.predict_classes(X_test)
    
    #y_pred_probs = classifier.predict_probabilities(X_test)
    #print(y_pred_probs)
    #print(y_pred_classes)


    # 4. Extract classification metrics
    compute_classification_scorecard(y_test, y_pred_classes)

    # 5. Persist classification model to disk
    classifier.save(MODEL_OUTPUT_PATH)
    print("==================================================")

if __name__ == "__main__":
    execute_classification_pipeline()