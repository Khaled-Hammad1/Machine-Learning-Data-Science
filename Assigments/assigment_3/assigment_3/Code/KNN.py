import os
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "good_rows_local.csv"
IMG_ROOT = "."                 # base folder for local image paths
TARGET_COL = "Season"
SEED = 42

# Split ratios (70/15/15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image feature settings
IMG_SIZE = (128, 128)
HIST_BINS = 32

# Baseline requirement
K_VALUES = [1, 3]
CLASS_ORDER = ["Fall", "Spring", "Summer", "Winter"]

# =========================
# Reproducibility
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# =========================
# Histogram Features
# =========================
def extract_color_hist_features(img, bins=32):
    """
    Extract normalized RGB histogram features (3 * bins).
    """
    arr = np.array(img)  # H,W,3
    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 255))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        feats.append(hist)
    return np.concatenate(feats, axis=0)  # (3*bins,)

def build_X_y(df):
    """
    Build feature matrix X and labels y from local images.
    Assumes ImageURL contains local file paths.
    """
    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        rel_path = str(row["ImageURL"]).strip()
        img_path = os.path.join(IMG_ROOT, rel_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        img = img.resize(IMG_SIZE)
        feat = extract_color_hist_features(img, HIST_BINS)

        X.append(feat)
        y.append(row[TARGET_COL])

    return np.array(X, dtype=np.float32), np.array(y), skipped

# =========================
# Confusion Matrix Plot
# =========================
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45)
    plt.yticks(range(len(CLASS_ORDER)), CLASS_ORDER)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# =========================
# MAIN
# =========================
def KNN():
    set_seed(SEED)

    # Load (assume clean & local)
    df = pd.read_csv(CSV_PATH, engine="python")

    # Ensure only the 4 valid season classes are used
    df = df[df[TARGET_COL].isin(CLASS_ORDER)].copy()

    print("Rows:", len(df))
    print("Class distribution:\n", df[TARGET_COL].value_counts(), "\n")

    # 70/15/15 split
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=SEED,
        stratify=df[TARGET_COL]
    )

    val_ratio_within_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_within_temp),
        random_state=SEED,
        stratify=temp_df[TARGET_COL]
    )

    print("Split sizes:")
    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df), "\n")

    # Build features (use train + test for baseline reporting)
    X_train, y_train, skipped_tr = build_X_y(train_df)
    X_test,  y_test,  skipped_te = build_X_y(test_df)

    print(f"Skipped unreadable images: train={skipped_tr}, test={skipped_te}")
    print("Feature shape:", X_train.shape, "(expected N x 96 because 3*32 bins)\n")

    # Standardize features for distance-based kNN
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Evaluate baseline for k=1 and k=3 (no tuning, as required)
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")  # distance of choice
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n===== kNN BASELINE (k={k}, distance=euclidean) =====")
        print("Test Accuracy:", acc)

        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            zero_division=0
        ))

        plot_confusion_matrix(y_test, y_pred, f"kNN Confusion Matrix (k={k})")

