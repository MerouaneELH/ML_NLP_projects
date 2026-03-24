import pandas as pd
import re

# Basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

# Data loading and preprocessing using a generator
def load_and_preprocess_data(file_path, batch_size=2):
    """
    Loads data from a CSV file, preprocesses it, and yields it in batches.
    Uses a generator for memory-efficient processing.
    """
    try:
        # Read the entire CSV at once for simplicity in this example
        # For very large files, consider reading chunk by chunk with pandas
        df = pd.read_csv(file_path)
        
        # Apply basic text cleaning
        df["text"] = df["text"].apply(clean_text)
        
        # Convert labels to numerical (e.g., positive: 1, negative: 0)
        # This is a simple example; more robust label encoding might be needed
        df["label"] = df["label"].apply(lambda x: 1 if x == "positive" else 0)

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            texts = batch_df["text"].tolist()
            labels = batch_df["label"].tolist()
            yield texts, labels
            
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        yield [], [] # Yield empty lists in case of error
    except Exception as e:
        print(f"An error occurred during data loading or preprocessing: {e}")
        yield [], [] # Yield empty lists in case of error

if __name__ == "__main__":
    # Example usage:
    data_file = "../sample_data.csv" # Relative path from src to project root
    print(f"Loading and preprocessing data from {data_file}...")
    
    data_generator = load_and_preprocess_data(data_file, batch_size=2)
    
    batch_num = 1
    for texts_batch, labels_batch in data_generator:
        if not texts_batch: # Handle cases where generator might yield empty due to error
            print("No data in batch, possibly due to an error during loading.")
            break
        print(f"\nBatch {batch_num}:")
        for i in range(len(texts_batch)):
            print(f"  Text: {texts_batch[i]}, Label: {labels_batch[i]}")
        batch_num += 1
    
    if batch_num == 1 and not texts_batch: # If no batches were processed at all
        print("Failed to process any data.")
    else:
        print("\nData loading and preprocessing example finished.")

