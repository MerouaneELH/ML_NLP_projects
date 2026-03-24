import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd # Added for easier handling of collected data

# Assuming data_processing.py is in the same directory (src)
from data_processing import load_and_preprocess_data, clean_text

# Decorator for logging (placeholder for now, as per todo.md Phase 2)
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        # import time # Would be needed for actual timing
        # start_time = time.time()
        print(f"Executing {func.__name__}...")
        result = func(*args, **kwargs)
        # end_time = time.time()
        # print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        print(f"{func.__name__} execution finished.")
        return result
    return wrapper

@log_execution_time
def train_model(data_file_path="sample_data.csv", model_dir="../models"):
    """
    Trains a text classification model.
    """
    print("Starting model training process...")
    
    # Load and preprocess data
    # Collect all data from the generator for scikit-learn training
    all_texts = []
    all_labels = []
    data_generator = load_and_preprocess_data(data_file_path, batch_size=10) # Use a larger batch for collection
    for texts_batch, labels_batch in data_generator:
        if texts_batch:
            all_texts.extend(texts_batch)
            all_labels.extend(labels_batch)

    if not all_texts or not all_labels:
        print("No data loaded. Exiting training.")
        return

    print(f"Loaded {len(all_texts)} samples for training.")

    # Split data (optional for this small dataset, but good practice)
    if len(all_texts) > 1:
        texts_train, texts_test, labels_train, labels_test = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels if len(set(all_labels)) > 1 else None
        )
    else: # Handle case with only one sample, avoid splitting
        texts_train, labels_train = all_texts, all_labels
        texts_test, labels_test = [], []
        print("Warning: Dataset too small to split. Training on the full dataset.")

    # Feature extraction (TF-IDF)
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(texts_train)
    if texts_test:
        X_test = vectorizer.transform(texts_test)

    # Model training (Logistic Regression)
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, labels_train)

    # Evaluate model (optional, but good for checking)
    if texts_test and labels_test:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(labels_test, predictions)
        print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    else:
        print("No test set to evaluate. Skipping evaluation.")

    # Save the model and vectorizer
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, "text_classifier_model.joblib")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    print("Model training process completed.")

if __name__ == "__main__":
    # Ensure paths are relative to the src directory when running this script directly
    # The default data_file_path and model_dir are already set for this structure.
    train_model()

