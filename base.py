# ============================================================
# arXiv MLP Paper Classifier
# Top-20 Main Categories + True Balanced Dataset + CSV + CPU
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU

import json
import logging
import joblib
import psutil
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
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

# ============================================================
# Paths & Logging
# ============================================================
BASE_DIR = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/base"
os.makedirs(BASE_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    BASE_DIR, f"mlp_top20_maincat_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

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

SAMPLES_PER_CLASS = 10000   # Number of samples per class after balancing
VOCAB_SIZE = 15000
EPOCHS = 20
BATCH_SIZE = 64
HISTORY_LOG_FILE = os.path.join(BASE_DIR, "mlp_learning_curves.csv")

TOP_K = 20  # Top 20 main categories

# ============================================================
# Step 1: Identify Top 20 main categories from the entire dataset
# ============================================================
def get_top_categories(path, top_k=20):
    counts = {}
    logger.info("🔍 Counting categories in full dataset...")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cats = item.get("categories", "").split()
            if not cats:
                continue
            main_cat = cats[0].split('.')[0]
            counts[main_cat] = counts.get(main_cat, 0) + 1

    counts_series = pd.Series(counts).sort_values(ascending=False)
    top_categories = counts_series.head(top_k).index.tolist()
    logger.info(f"🏷 Top-{top_k} main categories:")
    logger.info(top_categories)
    return top_categories

# ============================================================
# Step 2: Load data for Top 20 + Balance + Save CSV
# ============================================================
def load_balanced_top20_dataset(path, top_categories, samples_per_class):
    logger.info("🔍 Streaming JSON for Top 20 categories...")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cats = item.get("categories", "").split()
            if not cats:
                continue
            main_cat = cats[0].split('.')[0]
            if main_cat not in top_categories:
                continue
            text = item.get("title", "") + " " + item.get("abstract", "")
            rows.append({"text": text, "category": main_cat})

    df = pd.DataFrame(rows)

    # Downsampling each class to achieve balance
    df_balanced = (
        df.groupby("category", group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42))
          .reset_index(drop=True)
    )

    logger.info("📊 Balanced distribution after downsampling:")
    logger.info(df_balanced['category'].value_counts().to_string())

    # Save dataset as CSV
    csv_path = os.path.join(BASE_DIR, "arxiv_top20_maincat_balanced.csv")
    df_balanced.to_csv(csv_path, index=False)
    logger.info(f"💾 Balanced dataset saved to CSV: {csv_path}")

    return df_balanced

# ============================================================
# Step 3: Train MLP on CPU
# ============================================================
def run_experiment():
    try:
        mem = psutil.virtual_memory().available / (1024**3)
        logger.info(f"💾 Available RAM: {mem:.2f} GB")

        # ---------------------------------------
        # 1) Top categories
        # ---------------------------------------
        top_categories = get_top_categories(JSON_PATH, top_k=TOP_K)

        # ---------------------------------------
        # 2) Load & Balance dataset
        # ---------------------------------------
        df = load_balanced_top20_dataset(JSON_PATH, top_categories, SAMPLES_PER_CLASS)

        # ---------------------------------------
        # 3) Encode labels
        # ---------------------------------------
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['category'])
        num_classes = len(le.classes_)
        logger.info(f"🧾 Number of classes: {num_classes}")

        # ---------------------------------------
        # 4) Train/test split (stratified)
        # ---------------------------------------
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
        )

        # ---------------------------------------
        # 5) TF-IDF
        # ---------------------------------------
        tfidf = TfidfVectorizer(max_features=VOCAB_SIZE, stop_words='english')
        logger.info("⚡ TF-IDF vectorization (CPU)...")
        X_train = tfidf.fit_transform(X_train_text).toarray()
        X_test = tfidf.transform(X_test_text).toarray()

        joblib.dump(tfidf, os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

        # ---------------------------------------
        # 6) MLP Model (CPU)
        # ---------------------------------------
        with tf.device("/CPU:0"):
            model = Sequential([
                Input(shape=(VOCAB_SIZE,)),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(32, activation="relu"),
                Dropout(0.5),
                Dense(num_classes, activation="softmax")
            ])

            model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            csv_logger = CSVLogger(HISTORY_LOG_FILE, append=False)
            early_stop = EarlyStopping(monitor="val_loss", patience=3,
                                       restore_best_weights=True, verbose=1)

            logger.info("🚀 Training MLP on CPU...")
            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      callbacks=[csv_logger, early_stop],
                      verbose=1)

        # ---------------------------------------
        # 7) Evaluation
        # ---------------------------------------
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred,
                                       target_names=le.classes_, digits=4)
        acc = accuracy_score(y_test, y_pred)
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("="*60)
        logger.info("\n" + report)
        logger.info(f"✅ Final Accuracy: {acc:.4f}")

        # ---------------------------------------
        # 8) Save assets
        # ---------------------------------------
        model.save(os.path.join(BASE_DIR, "arxiv_mlp_top20_maincat_balanced.h5"))
        joblib.dump(le, os.path.join(BASE_DIR, "label_encoder.pkl"))
        logger.info("💾 All assets saved successfully.")

    except Exception:
        logger.error("❌ Experiment failed", exc_info=True)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_experiment()
