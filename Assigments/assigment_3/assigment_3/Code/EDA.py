import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
FILE = "good_rows_unique_final.csv"
df = pd.read_csv(FILE, engine="python")

#if "ImageURL" in df.columns:
 #   df = df.drop(columns=["ImageURL"])

print("Shape:", df.shape)
print("Columns:", list(df.columns))

# =========================
# 1) Basic summaries
# =========================
print("\n========== INFO ==========")
print(df.info())


print("\n========== MISSING VALUES ==========")
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
print(pd.DataFrame({"missing": missing, "missing_%": missing_pct}))

print("\n========== DUPLICATES ==========")
dup = df.duplicated().sum()
print("Duplicate rows:", dup)

# =========================
# 3) Class distributions 
# =========================
cat_cols = [c for c in ["Season", "Weather", "TimeOfDay", "Mood", "Activity", "Country"] if c in df.columns]

for c in cat_cols:
    print(f"\n========== DISTRIBUTION: {c} ==========")
    vc = df[c].astype("string").fillna("<MISSING>").value_counts()
    pct = (vc / len(df) * 100).round(2)
    dist = pd.DataFrame({"count": vc, "pct": pct})
    print(dist.head(20))  # Top 20

    # Bar plot (Top 15)
    top15 = vc.head(15)
    plt.figure()
    plt.bar(top15.index.astype(str), top15.values)
    plt.title(f"Categories: {c}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# =========================
# 6) Categorical relationship
# =========================
if "Season" in df.columns and "Weather" in df.columns:
    ct = pd.crosstab(df["Season"], df["Weather"], normalize="index")
    print("\n========== CROSSTAB: Season vs Weather (row-normalized) ==========")
    print(ct.round(3))

pairs = [
    ("Season", "Weather"),
    ("Mood", "Season"),
    ("TimeOfDay", "Weather"),
    ("TimeOfDay", "Mood"),
    ("TimeOfDay", "Season"),
    ("Weather", "Mood"),
]

for a, b in pairs:
    if a in df.columns and b in df.columns:
        # Raw COUNTS (not normalized)
        ct = pd.crosstab(
            df[a].astype("string").fillna("<MISSING>"),
            df[b].astype("string").fillna("<MISSING>")
        ).fillna(0)

        print(f"\n========== CROSSTAB (counts): {a} vs {b} ==========")
        print(ct)

        plt.figure(figsize=(10, 6))
        plt.imshow(ct.values, aspect="auto")
        plt.xticks(range(len(ct.columns)), ct.columns, rotation=45, ha="right")
        plt.yticks(range(len(ct.index)), ct.index)
        plt.colorbar()
        plt.title(f"Heatmap (Counts): {a} vs {b}")
        plt.tight_layout()

        # Write the count inside each cell
        for i in range(ct.shape[0]):        # rows
            for j in range(ct.shape[1]):    # cols
                val = int(ct.values[i, j])
                plt.text(j, i, str(val), ha="center", va="center")

        plt.show()