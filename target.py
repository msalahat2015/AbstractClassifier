# ============================================================
# arXiv CNN Paper Classifier (Top-20 Main Categories + Balanced)
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

# ============================================================
# Paths & Logging
# ============================================================
TARGET_DIR = "/rep/msalahat/COMP9323-ST-LLMs/COMP9319-DL/Assets/Target"
os.makedirs(TARGET_DIR, exist_ok=True)

LOG_FILE = os.path.join(TARGET_DIR, f"cnn_top20_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

SAMPLES_PER_CLASS = 10000
TOP_K = 20

VOCAB_SIZE = 15000
MAX_LEN = 256
EMBEDDING_DIM = 100
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.2

# ============================================================
# Step 1: Compute Top 20 Main Categories from the entire JSON dataset
# ============================================================
def get_top_categories(path, top_k=20):
    counts = {}
    logger.info("🔍 Counting main categories in full dataset...")

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
    logger.info("📥 Loading JSON for Top 20 categories and balancing...")

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

    df_balanced = (
        df.groupby("category", group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42))
          .reset_index(drop=True)
    )

    logger.info("📊 Balanced distribution after downsampling:")
    logger.info(df_balanced['category'].value_counts().to_string())

    # Save dataset as CSV
    csv_path = os.path.join(TARGET_DIR, "arxiv_top20_maincat_balanced.csv")
    df_balanced.to_csv(csv_path, index=False)
    logger.info(f"💾 Balanced dataset saved to CSV: {csv_path}")

    return df_balanced

# ============================================================
# Step 3: Train CNN on CPU
# ============================================================
def run_cnn_experiment():
    try:
        # ----------------------------------------------------
        # 1) Top categories
        # ----------------------------------------------------
        top_categories = get_top_categories(JSON_PATH, top_k=TOP_K)

        # ----------------------------------------------------
        # 2) Load & Balance dataset
        # ----------------------------------------------------
        df = load_balanced_top20_dataset(JSON_PATH, top_categories, SAMPLES_PER_CLASS)

        # ----------------------------------------------------
        # 3) Label Encoding
        # ----------------------------------------------------
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['category'])
        num_classes = len(le.classes_)
        logger.info(f"🧾 Number of classes: {num_classes}")

        # ----------------------------------------------------
        # 4) Tokenization
        # ----------------------------------------------------
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
        tokenizer.fit_on_texts(df['text'])
        sequences = tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        y = to_categorical(df['label'], num_classes)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=df['label'], random_state=42
        )

        # ----------------------------------------------------
        # 5) CNN Model
        # ----------------------------------------------------
        with tf.device('/CPU:0'):
            model = Sequential([
                Embedding(VOCAB_SIZE, EMBEDDING_DIM),
                Conv1D(128, 5, activation='relu'),
                GlobalMaxPooling1D(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Callbacks
            csv_logger = CSVLogger(os.path.join(TARGET_DIR, "cnn_learning_curves.csv"))
            early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)

            logger.info("🚀 Training CNN on CPU...")
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[csv_logger, early_stop],
                verbose=1
            )

        # ----------------------------------------------------
        # 6) Evaluation
        # ----------------------------------------------------
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
        acc = accuracy_score(y_true, y_pred)
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION REPORT")
        logger.info("="*60)
        logger.info("\n" + report)
        logger.info(f"✅ Final Accuracy: {acc:.4f}")

        # ----------------------------------------------------
        # 7) Save Assets
        # ----------------------------------------------------
        model.save(os.path.join(TARGET_DIR, "arxiv_cnn_top20_maincat_balanced.h5"))
        joblib.dump(tokenizer, os.path.join(TARGET_DIR, "tokenizer.pkl"))
        joblib.dump(le, os.path.join(TARGET_DIR, "label_encoder.pkl"))
        logger.info("💾 All CNN assets saved successfully.")

    except Exception:
        logger.error("❌ CNN experiment failed", exc_info=True)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_cnn_experiment()
