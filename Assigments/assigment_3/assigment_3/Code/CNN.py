import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================
# CONFIG
# =========================
CSV_PATH = "good_rows_local.csv"   # <-- change if needed
IMG_ROOT = "."                      # <-- folder base for ImageURL paths
TARGET_COL = "Season"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_season(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s_low = s.lower()

    if s_low == "autumn":
        return "Fall"

    mapping = {
        "winter": "Winter",
        "spring": "Spring",
        "summer": "Summer",
        "fall": "Fall",
    }
    return mapping.get(s_low, s)

def is_not_clear(x: str) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"not clear", "notclear", "unknown", ""}

def safe_open_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None


# =========================
# Dataset
# =========================
class SeasonDataset(Dataset):
    def __init__(self, df, img_root, label2idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rel_path = str(row["ImageURL"]).strip()
        img_path = os.path.normpath(os.path.join(self.img_root, rel_path))

        img = safe_open_image(img_path)
        if img is None:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        y_str = row[TARGET_COL]
        y = self.label2idx[y_str]
        return img, y


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_true = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_true.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_true.extend(y.numpy().tolist())

    acc = accuracy_score(all_true, all_preds)
    return acc, all_true, all_preds


# =========================
# MAIN
# =========================
def CNN():
    print("CNN")
    set_seed(SEED)

    # --- Load CSV
    df = pd.read_csv(CSV_PATH, engine="python")

    # --- Clean ImageURL
    df = df[df["ImageURL"].notna()].copy()
    df["ImageURL"] = df["ImageURL"].astype(str).str.strip()
    df = df[df["ImageURL"] != ""].copy()

    # --- Clean Season
    df[TARGET_COL] = df[TARGET_COL].apply(clean_season)

    # drop Not Clear
    df = df[~df[TARGET_COL].apply(is_not_clear)].copy()

    # keep only 4 seasons
    allowed = {"Winter", "Spring", "Summer", "Fall"}
    df = df[df[TARGET_COL].isin(allowed)].copy()

    print("Rows after cleaning:", len(df))
    print("Season distribution:\n", df[TARGET_COL].value_counts(), "\n")

    # --- Label encoding
    labels = sorted(df[TARGET_COL].unique())
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    num_classes = len(labels)

    print("Classes:", labels, "\n")

    # --- Split
    temp_ratio = 1.0 - TRAIN_RATIO
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=SEED,
        stratify=df[TARGET_COL]
    )

    val_within_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_within_temp),
        random_state=SEED,
        stratify=temp_df[TARGET_COL]
    )

    print("Split sizes:")
    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df), "\n")

    # --- Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Datasets / Loaders (Windows safe: num_workers=0)
    train_ds = SeasonDataset(train_df, IMG_ROOT, label2idx, transform=train_tf)
    val_ds   = SeasonDataset(val_df,   IMG_ROOT, label2idx, transform=eval_tf)
    test_ds  = SeasonDataset(test_df,  IMG_ROOT, label2idx, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model (Transfer Learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Train loop (save best by Val Acc)
    best_val_acc = 0.0
    best_path = "best_season_resnet18.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, _, _ = evaluate(model, val_loader)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print("\nBest Validation Accuracy:", best_val_acc)

    # --- Final Test evaluation
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    test_acc, y_true, y_pred = evaluate(model, test_loader)
    print("\n===== TEST RESULTS =====")
    print("Test Accuracy:", test_acc)

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=[idx2label[i] for i in range(num_classes)]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[idx2label[i] for i in range(num_classes)])

    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

