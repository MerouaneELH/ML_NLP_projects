# MLOps Text Classification Project

A sample text classification pipeline built for learning MLOps practices with scikit-learn, FastAPI, and model persistence.

## Overview

This project demonstrates an end-to-end sentiment classification workflow:
- Data ingestion and preprocessing (`src/data_processing.py`)
- TF-IDF feature extraction with `sklearn.feature_extraction.text.TfidfVectorizer`
- Logistic Regression model training (`src/train.py`)
- Model + vectorizer persistence via `joblib` (`models/text_classifier_model.joblib` and `models/tfidf_vectorizer.joblib`)
- REST API serving with FastAPI (`src/main.py`)
- Basic request logging (`src/logger.py`)

## Repository structure

```
mlops_text_classification_project/
├── .dvc/
├── Dockerfile
├── logs/
├── models/
│   ├── text_classifier_model.joblib
│   └── tfidf_vectorizer.joblib
├── sample_data.csv
├── requirements.txt
└── src/
    ├── data_processing.py
    ├── logger.py
    ├── main.py
    └── train.py
```

## Prerequisites

- Python 3.9+
- pip

## Setup

```bash
cd mlops_text_classification_project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
cd mlops_text_classification_project/src
python train.py
```

Model artifacts will be saved in `../models/`.

## Run API server

```bash
cd mlops_text_classification_project/src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Prediction endpoint

POST `http://127.0.0.1:8000/predict/`

Body:
```json
{ "text": "This is a great product!" }
```

Response:
```json
{
  "text": "This is a great product!",
  "prediction": "positive",
  "label": 1
}
```

## Local test data

- `sample_data.csv`: simple two-label dataset used for training.

## Notes

- `src/data_processing.py` includes text cleaning logic and label mapping (positive -> 1, negative -> 0).
- If models are missing, run `python train.py` first.
- Logging is persisted in `logs/api.log`.

## Optional

Build and run with Docker:

```bash
docker build -t mlops-text-classifier .
docker run -p 8000:8000 mlops-text-classifier
```
