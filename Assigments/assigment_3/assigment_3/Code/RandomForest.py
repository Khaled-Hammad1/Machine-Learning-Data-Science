import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CNN feature extractor
import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# CONFIG
# =========================
CSV_PATH = "good_rows_local.csv"
IMG_ROOT = "."
TARGET_COL = "Season"
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Histogram settings
HIST_IMG_SIZE = (128, 128)
HIST_BINS = 32

# CNN settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RF grid
RF_GRID = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 20, 40],
    "min_samples_leaf": [1, 2, 4],
}

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_season(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s == "autumn":
        return "Fall"
    mapping = {"winter":"Winter", "spring":"Spring", "summer":"Summer", "fall":"Fall"}
    return mapping.get(s, s)

def safe_load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

# =========================
# Histogram Features
# =========================
def extract_color_hist_features(img, bins=32):
    arr = np.array(img)  # H,W,3
    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 255))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        feats.append(hist)
    return np.concatenate(feats, axis=0)  # (bins*3,)

def build_X_y_hist(df):
    X, y = [], []
    bad = 0
    for _, row in df.iterrows():
        rel = str(row["ImageURL"]).strip()
        path = os.path.normpath(os.path.join(IMG_ROOT, rel))
        img = safe_load_image(path)
        if img is None:
            bad += 1
            continue
        img = img.resize(HIST_IMG_SIZE)
        feat = extract_color_hist_features(img, bins=HIST_BINS)
        X.append(feat)
        y.append(row[TARGET_COL])
    return np.array(X, dtype=np.float32), np.array(y), bad

# =========================
# CNN Features (ResNet18 pretrained)
# =========================
cnn_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def build_cnn_extractor():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # output: 512
    model.eval()
    model.to(DEVICE)
    return model

@torch.no_grad()
def extract_cnn_features(model, img):
    x = cnn_tf(img).unsqueeze(0).to(DEVICE)   # (1,3,224,224)
    feat = model(x).squeeze(0).cpu().numpy()  # (512,)
    return feat.astype(np.float32)

def build_X_y_cnn(df, cnn_model):
    X, y = [], []
    bad = 0
    for _, row in df.iterrows():
        rel = str(row["ImageURL"]).strip()
        path = os.path.normpath(os.path.join(IMG_ROOT, rel))
        img = safe_load_image(path)
        if img is None:
            bad += 1
            continue
        feat = extract_cnn_features(cnn_model, img)
        X.append(feat)
        y.append(row[TARGET_COL])
    return np.array(X, dtype=np.float32), np.array(y), bad

# =========================
# RF Training + Grid Search
# =========================
def rf_grid_search(X_train, y_train, X_val, y_val):
    best_acc = -1
    best_model = None
    best_cfg = None

    for n in RF_GRID["n_estimators"]:
        for d in RF_GRID["max_depth"]:
            for leaf in RF_GRID["min_samples_leaf"]:
                rf = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=d,
                    min_samples_leaf=leaf,
                    random_state=SEED,
                    n_jobs=-1,
                    class_weight="balanced"
                )
                rf.fit(X_train, y_train)
                val_pred = rf.predict(X_val)
                acc = accuracy_score(y_val, val_pred)

                print(f"  RF cfg: n={n}, depth={d}, leaf={leaf} -> val_acc={acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_model = rf
                    best_cfg = {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}

    return best_model, best_cfg, best_acc

def evaluate_and_print(name, model, X_test, y_test):
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n===== {name} | TEST RESULTS =====")
    print("Test Accuracy:", test_acc)

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)

    # ---- Plot Confusion Matrix ----
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.unique(y_test)  
    )
    disp.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return test_acc


# =========================
# MAIN
# =========================
def RandomForest():
    print("RandomForest")
    set_seed(SEED)

    # Load + clean
    df = pd.read_csv(CSV_PATH, engine="python")
    df = df[df["ImageURL"].notna()].copy()
    df["ImageURL"] = df["ImageURL"].astype(str).str.strip()
    df = df[df["ImageURL"] != ""].copy()

    df[TARGET_COL] = df[TARGET_COL].apply(clean_season)

    allowed = {"Winter", "Spring", "Summer", "Fall"}
    df = df[df[TARGET_COL].isin(allowed)].copy()

    print("Rows after cleaning:", len(df))
    print("Season distribution:\n", df[TARGET_COL].value_counts(), "\n")

    # Split 
    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - TRAIN_RATIO), random_state=SEED, stratify=df[TARGET_COL]
    )
    val_ratio_within_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_ratio_within_temp), random_state=SEED, stratify=temp_df[TARGET_COL]
    )

    print("Split sizes:", len(train_df), len(val_df), len(test_df), "\n")

    # -------------------------
    # 1) Histogram + RF
    # -------------------------
    print("=== (1) Histogram Features + RandomForest ===")
    X_train_h, y_train_h, bad_tr_h = build_X_y_hist(train_df)
    X_val_h, y_val_h, bad_va_h = build_X_y_hist(val_df)
    X_test_h, y_test_h, bad_te_h = build_X_y_hist(test_df)

    print(f"Missing/Unreadable skipped: train={bad_tr_h}, val={bad_va_h}, test={bad_te_h}")
    print("Feature shape:", X_train_h.shape, "\n")

    best_rf_h, best_cfg_h, best_val_h = rf_grid_search(X_train_h, y_train_h, X_val_h, y_val_h)
    print("\nBest Histogram config:", best_cfg_h, "Best Val Acc:", best_val_h)

    hist_test_acc = evaluate_and_print("Histogram+RF", best_rf_h, X_test_h, y_test_h)

    # -------------------------
    # 2) CNN Features + RF
    # -------------------------
    print("\n\n=== (2) CNN(ResNet18) Features + RandomForest ===")
    cnn_model = build_cnn_extractor()

    X_train_c, y_train_c, bad_tr_c = build_X_y_cnn(train_df, cnn_model)
    X_val_c, y_val_c, bad_va_c = build_X_y_cnn(val_df, cnn_model)
    X_test_c, y_test_c, bad_te_c = build_X_y_cnn(test_df, cnn_model)

    print(f"Missing/Unreadable skipped: train={bad_tr_c}, val={bad_va_c}, test={bad_te_c}")
    print("Feature shape:", X_train_c.shape, "(should be N x 512)\n")

    best_rf_c, best_cfg_c, best_val_c = rf_grid_search(X_train_c, y_train_c, X_val_c, y_val_c)
    print("\nBest CNN config:", best_cfg_c, "Best Val Acc:", best_val_c)

    cnn_test_acc = evaluate_and_print("CNNFeatures+RF", best_rf_c, X_test_c, y_test_c)

    # -------------------------
    # Summary
    # -------------------------
    print("\n\n===== FINAL COMPARISON =====")
    print(f"Histogram+RF  Test Acc: {hist_test_acc:.4f}")
    print(f"CNNFeat+RF    Test Acc: {cnn_test_acc:.4f}")


