import os
import re
from collections import Counter
import pandas as pd
import requests
from urllib.parse import urlparse

# =========================
# Settings
# =========================
root_dir = "data_set"   
OUTPUT_FILE = "good_rows_unique.csv"

TARGET_COLUMNS = [
    "ImageURL",
    "Description",
    "Country",
    "Weather",
    "TimeOfDay",
    "Season",
    "Activity",
    "Mood"
]

# =========================
# Collect CSV files
# =========================
csv_files = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("Total CSV files:", len(csv_files))

# =========================
# Robust CSV reader 
# =========================
def read_csv_robust(path):
    if os.path.getsize(path) == 0:
        raise ValueError("empty_file")

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", ";", "\t", "|"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip"
                )

                df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

                if df.shape[1] == 0:
                    continue

                return df
            except Exception as e:
                last_err = e

    raise last_err

# =========================
# Unify columns
# =========================
def unify_columns(df):
    new_columns = {}
    for col in df.columns:
        norm = re.sub(r'[^a-z0-9]', '', str(col).lower())

        if ("image" in norm or norm == "url" or "link" in norm or "picture" in norm):
            new_columns[col] = "ImageURL"
        elif ("description" in norm or norm == "desc" or "describtion" in norm):
            new_columns[col] = "Description"
        elif ("country" in norm or "location" in norm or "destinationname" in norm):
            new_columns[col] = "Country"
        elif "weather" in norm:
            new_columns[col] = "Weather"
        elif "timeofday" in norm or norm == "time":
            new_columns[col] = "TimeOfDay"
        elif "season" in norm:
            new_columns[col] = "Season"
        elif "activity" in norm:
            new_columns[col] = "Activity"
        elif "mood" in norm:
            new_columns[col] = "Mood"
        else:
            new_columns[col] = col

    return df.rename(columns=new_columns)

def ensure_target_columns(df):
    for c in TARGET_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df

def normalize_cell(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# URL validators 
# =========================
def is_valid_url(url):
    try:
        parsed = urlparse(str(url))
        return parsed.scheme in ("http", "https") and parsed.netloc != ""
    except:
        return False

_url_cache = {} 

def check_image_url(url, timeout=8):
    # cache
    key = "" if url is None else str(url).strip()
    if key in _url_cache:
        return _url_cache[key]

    if key == "" or key.lower() == "nan":
        _url_cache[key] = (False, "empty")
        return _url_cache[key]

    if not is_valid_url(key):
        _url_cache[key] = (False, "invalid_url")
        return _url_cache[key]

    try:
        r = requests.head(key, allow_redirects=True, timeout=timeout)
        if r.status_code in (403, 405):
            r = requests.get(key, stream=True, allow_redirects=True, timeout=timeout)

        if r.status_code != 200:
            _url_cache[key] = (False, f"http_{r.status_code}")
            return _url_cache[key]

        content_type = (r.headers.get("Content-Type") or "").lower()
        if not content_type.startswith("image/"):
            _url_cache[key] = (False, "not_image")
            return _url_cache[key]

        _url_cache[key] = (True, "ok")
        return _url_cache[key]

    except requests.exceptions.Timeout:
        _url_cache[key] = (False, "timeout")
        return _url_cache[key]
    except requests.exceptions.RequestException as e:
        _url_cache[key] = (False, type(e).__name__)
        return _url_cache[key]

# =========================
# Main process
# =========================
good_rows = []
bad_rows = []

failed_files = []

for csv_path in csv_files:
    try:
        df = read_csv_robust(csv_path)
        df = unify_columns(df)
        df = ensure_target_columns(df)

        if "ImageURL" not in df.columns:
            continue

        for _, row in df.iterrows():
            url = row.get("ImageURL", "")
            ok, reason = check_image_url(url)

            record = {c: normalize_cell(row.get(c, "")) for c in TARGET_COLUMNS}

            if ok:
                good_rows.append(record)
            else:
                bad_rows.append(record)

    except Exception as e:
        failed_files.append((csv_path, str(e)))

print("\n=========================")
print("Finished reading.")
print("Failed files count:", len(failed_files))
if failed_files:
    print("Sample failed files (up to 5):")
    for p, err in failed_files[:5]:
        print(" -", p, "|", err)

print("\nGood rows (before dedupe):", len(good_rows))
print("Bad rows:", len(bad_rows))

# =========================
# Find duplicated rows inside good_rows
# =========================
def row_key(d):
    return tuple(d.get(c, "") for c in TARGET_COLUMNS)

counts = Counter(row_key(r) for r in good_rows)
dup_count = sum(v - 1 for v in counts.values() if v > 1)

print("\nDuplicated rows inside good_rows:", dup_count)

# Remove duplicates (keep first occurrence)
seen = set()
unique_good_rows = []
for r in good_rows:
    k = row_key(r)
    if k not in seen:
        seen.add(k)
        unique_good_rows.append(r)

print("Good rows (after dedupe):", len(unique_good_rows))

# =========================
# Save to file
# =========================
out_df = pd.DataFrame(unique_good_rows, columns=TARGET_COLUMNS)
out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nSaved:", OUTPUT_FILE)
