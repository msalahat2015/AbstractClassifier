# ============================================================
# arXiv CNN Paper Classifier (Target)
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger

# ============================================================
# Logging
# ============================================================
TARGET_DIR = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/Target"
os.makedirs(TARGET_DIR, exist_ok=True)

LOG_FILE = os.path.join(TARGET_DIR, f"cnn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
MAX_LINES = 100000
VOCAB_SIZE = 20000
MAX_LEN = 256
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2

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
def load_arxiv_json(path, limit):
    data = []
    logger.info(f"📥 Loading first {limit} records from arXiv JSON")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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
def run_cnn_experiment():
    try:
        df = load_arxiv_json(JSON_PATH, MAX_LINES)

        # Label Encoding
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['category'])
        num_classes = len(le.classes_)
        logger.info(f"🧾 Number of classes: {num_classes}")

        # Tokenization
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
        tokenizer.fit_on_texts(df["full_text"])
        sequences = tokenizer.texts_to_sequences(df["full_text"])
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
        y = to_categorical(df["label"], num_classes)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=df["label"]
        )

        # CNN Model
        model = Sequential([
            Embedding(VOCAB_SIZE, EMBEDDING_DIM),
            Conv1D(128, 5, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Training
        csv_logger = CSVLogger(os.path.join(TARGET_DIR, "cnn_learning_curves.csv"))
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[csv_logger], verbose=1)

        # Evaluation
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
        acc = accuracy_score(y_true, y_pred)
        logger.info("\n" + "="*60 + "\nCLASSIFICATION REPORT\n" + "="*60)
        logger.info("\n" + report)
        logger.info(f"✅ Final Accuracy: {acc:.4f}")

        # Save Assets
        model.save(os.path.join(TARGET_DIR, 'arxiv_cnn_model.h5'))
        joblib.dump(tokenizer, os.path.join(TARGET_DIR, 'tokenizer.pkl'))
        joblib.dump(le, os.path.join(TARGET_DIR, 'label_encoder.pkl'))
        logger.info("💾 All CNN assets saved successfully.")

    except Exception as e:
        logger.error("❌ CNN experiment failed", exc_info=True)

if __name__ == "__main__":
    run_cnn_experiment()
