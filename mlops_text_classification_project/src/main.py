from fastapi import FastAPI, HTTPException
import joblib
import os
from pydantic import BaseModel
from logger import log_request, log_info, log_error



# Assuming data_processing.py is in the same directory (src)
from data_processing import clean_text # For preprocessing input text

# --- Model Loading (Singleton Pattern - Conceptual) ---
class ModelSingleton:
    _instance = None
    _model = None
    _vectorizer = None

    def __new__(cls, model_path="../models/text_classifier_model.joblib", 
                vectorizer_path="../models/tfidf_vectorizer.joblib"):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            try:
                print(f"Loading model from: {model_path}")
                print(f"Loading vectorizer from: {vectorizer_path}")
                
                # Adjust paths to be absolute or correctly relative to main.py execution
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cls._model = joblib.load(os.path.join(current_dir, model_path))
                cls._vectorizer = joblib.load(os.path.join(current_dir, vectorizer_path))
                print("Model and vectorizer loaded successfully.")
            except FileNotFoundError as e:
                print(f"Error loading model/vectorizer: {e}. Ensure paths are correct and model is trained.")
                cls._model = None
                cls._vectorizer = None
            except Exception as e:
                print(f"An unexpected error occurred during model loading: {e}")
                cls._model = None
                cls._vectorizer = None
        return cls._instance

    def get_model(self):
        return self._model

    def get_vectorizer(self):
        return self._vectorizer

# Initialize the singleton instance when the app starts
# Note: For a production FastAPI app, you might load this during startup events.
model_loader = ModelSingleton()
# --- End Model Loading ---

app = FastAPI(
    title="Text Classification API",
    description="An API for classifying text as positive or negative.",
    version="0.1.0"
)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    label: int

@app.on_event("startup")
async def startup_event():
    # This is a good place to ensure the model is loaded.
    # The ModelSingleton already attempts to load on first instantiation.
    # We can add a check here if needed.
    if model_loader.get_model() is None or model_loader.get_vectorizer() is None:
        log_info("Warning: Model or vectorizer not loaded at startup. Predictions will fail.")
    else:
        log_info("FastAPI startup: Model and vectorizer are ready.")

@app.post("/predict/", response_model=PredictionResponse)
async def predict_text(request: TextRequest):
    """
    Predicts the sentiment of a given text.
    - **text**: The input text to classify.
    """
    model = model_loader.get_model()
    vectorizer = model_loader.get_vectorizer()

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first or check server logs.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Preprocess the input text (same as training)
        cleaned_text = clean_text(request.text)
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction_label = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        sentiment = "positive" if prediction_label == 1 else "negative"
        
        log_info(f"Input: 	'{request.text}'")
        log_info(f"Cleaned: 	'{cleaned_text}'")
        log_info(f"Predicted Label: {prediction_label} ({sentiment}), Confidence: {prediction_proba}")
        
        return {
            "text": request.text,
            "prediction": sentiment,
            "label": int(prediction_label)
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Text Classification API. Use the /predict/ endpoint to classify text."}

# To run this app (from the mlops_text_classification_project/src directory):
# uvicorn main:app --reload

if __name__ == "__main__":
    # This part is for direct execution (e.g., python main.py) and might not be ideal for uvicorn deployment
    # For development, it's better to use: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")
    # Note: model paths in ModelSingleton are relative to this script's location.
    # Ensure 'models' directory is one level up from 'src' (e.g., ../models/)
    uvicorn.run(app, host="0.0.0.0", port=8000)

