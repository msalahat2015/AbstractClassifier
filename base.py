# ============================================================
# arXiv MLP Paper Classifier (Base)
# ============================================================

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import CSVLogger

# ============================================================
# Logging
# ============================================================
BASE_DIR = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/base"
os.makedirs(BASE_DIR, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, f"mlp_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================
# Hyperparameters
# ============================================================
JSON_PATH = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Dataset/arxiv-metadata-oai-snapshot.json"
VOCAB_SIZE = 15000
EPOCHS = 10
BATCH_SIZE = 64
SAMPLE_SIZE = 100000  # samples to load
HISTORY_LOG_FILE = os.path.join(BASE_DIR, "mlp_learning_curves.csv")

# ============================================================
# Base Categories
# ============================================================
BASE_CATEGORIES = [
    "astro-ph", "cond-mat", "cs", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math", "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "q-bio", "q-fin",
    "quant-ph", "stat"
]

# ============================================================
# Load Data
# ============================================================
def load_arxiv_json(path, limit=SAMPLE_SIZE):
    data = []
    logger.info(f"🔍 Streaming first {limit} rows from JSON...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found at {path}")
        
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            item = json.loads(line)
            cats = item.get("categories", "").split()
            if not cats:
                continue
            main_cat = cats[0].split(".")[0]
            if main_cat not in BASE_CATEGORIES:
                continue
            data.append({"full_text": item['title'] + " " + item['abstract'], "category": main_cat})
    return pd.DataFrame(data)

# ============================================================
# Run Experiment
# ============================================================
def run_mlp_experiment():
    try:
        df = load_arxiv_json(JSON_PATH, SAMPLE_SIZE)

        # Label Encoding
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['category'])
        num_classes = len(le.classes_)
        logger.info(f"🧾 Number of classes: {num_classes}")

        # Train/Test Split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['full_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english')
        X_train = tfidf.fit_transform(X_train_text).toarray()
        X_test = tfidf.transform(X_test_text).toarray()
        joblib.dump(tfidf, os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))

        # MLP Model
        model = Sequential([
            Input(shape=(VOCAB_SIZE,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Training
        csv_logger = CSVLogger(HISTORY_LOG_FILE, append=False)
        logger.info(f"🚀 Training MLP. Learning curves -> {HISTORY_LOG_FILE}")
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[csv_logger], verbose=1)

        # Evaluation
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
        acc = accuracy_score(y_test, y_pred)
        logger.info("\n" + "="*50 + "\nCLASSIFICATION REPORT\n" + "="*50)
        logger.info("\n" + report)
        logger.info(f"✅ Final Accuracy: {acc:.4f}")

        # Save Assets
        model.save(os.path.join(BASE_DIR, 'arxiv_mlp_model.h5'))
        joblib.dump(le, os.path.join(BASE_DIR, 'label_encoder.pkl'))
        logger.info("💾 All MLP assets saved successfully.")

    except Exception as e:
        logger.error("❌ MLP experiment failed", exc_info=True)

if __name__ == "__main__":
    run_mlp_experiment()
