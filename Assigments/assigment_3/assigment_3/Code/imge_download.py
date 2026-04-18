import os
import re
import hashlib
from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# Settings
# =========================
INPUT_CSV  = "good_rows_unique_final.csv"          # Input CSV that contains ImageURL
OUTPUT_CSV = "good_rows_local.csv"    # Output CSV (ImageURL replaced with local paths)
IMAGE_DIR  = "images"                # Folder to save downloaded images

os.makedirs(IMAGE_DIR, exist_ok=True)


# =========================
# Helpers
# =========================
def normalize_col_name(c: str) -> str:
    """Normalize column name by removing non-alphanumeric chars and lowercasing."""
    return re.sub(r"[^a-z0-9]", "", str(c).strip().lower())

def find_image_column(columns):
    """
    Detect the image URL column name.
    Accepts variations like: ImageURL, Image URL, image_url, image, url, link, picture, etc.
    """
    cols = list(columns)
    norm_map = {c: normalize_col_name(c) for c in cols}

    # Prefer explicit ImageURL-like columns
    for c, n in norm_map.items():
        if n in ("imageurl", "image_url", "imgurl"):
            return c

    # If it contains both "image" and "url/link"
    for c, n in norm_map.items():
        if "image" in n and ("url" in n or "link" in n):
            return c

    # Fallback to generic url/link/picture
    for c, n in norm_map.items():
        if n in ("url", "link", "picture", "image"):
            return c

    return None

def is_url(x) -> bool:
    """Return True if x looks like an http/https URL."""
    try:
        u = urlparse(str(x).strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except:
        return False

def safe_filename(url: str, idx: int) -> str:
    """Create a stable filename using row index + short hash of the URL."""
    h = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"img_{idx:06d}_{h}.jpg"

def build_session() -> requests.Session:
    """Create a requests session with retries and a browser-like User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    })
    return session

def download_and_verify_image(session: requests.Session, url: str):
    """
    Download an image and verify it is a real, decodable image.

    Returns:
      - (PIL.Image, "ok") if success
      - (None, <reason>) if failed
    """
    # Download content
    try:
        r = session.get(url, stream=True, allow_redirects=True)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"

        content = r.content

        # Verify the image integrity (PIL verify does not fully decode)
        bio = BytesIO(content)
        img = Image.open(bio)
        img.verify()

        # Re-open after verify to actually load it
        bio2 = BytesIO(content)
        img2 = Image.open(bio2).convert("RGB")

        return img2, "ok"

    except requests.RequestException as e:
        return None, type(e).__name__
    except Exception:
        # PIL failed to open/verify -> not an image or corrupted
        return None, "pil_open_failed"


# =========================
# Main
# =========================
df = pd.read_csv(INPUT_CSV, engine="python")
img_col = find_image_column(df.columns)
if img_col is None:
    raise ValueError(f"Image column not found. Columns: {list(df.columns)}")

session = build_session()

# Keep the original URL in a separate column
if "ImageURL_original" not in df.columns:
    df["ImageURL_original"] = df[img_col].astype(str)

local_paths = []
status_list = []
reason_list = []

ok_count = 0
failed_count = 0

for idx, val in tqdm(list(enumerate(df[img_col].tolist())), total=len(df), desc="Downloading & verifying"):
    url = "" if pd.isna(val) else str(val).strip()

    # Empty value
    if url == "":
        local_paths.append("")
        status_list.append("failed")
        reason_list.append("empty")
        failed_count += 1
        continue

    # Already a local path (not a URL): verify it exists and can be opened
    if not is_url(url):
        if os.path.exists(url):
            try:
                im = Image.open(url)
                im.verify()
                local_paths.append(url)
                status_list.append("ok")
                reason_list.append("already_local_ok")
                ok_count += 1
            except Exception:
                local_paths.append("")
                status_list.append("failed")
                reason_list.append("local_corrupt")
                failed_count += 1
        else:
            local_paths.append("")
            status_list.append("failed")
            reason_list.append("local_missing")
            failed_count += 1
        continue

    # Download and verify
    img, reason = download_and_verify_image(session, url)
    if img is None:
        local_paths.append("")
        status_list.append("failed")
        reason_list.append(reason)
        failed_count += 1
        continue

    # Save as JPG locally
    out_name = safe_filename(url, idx)
    out_path = os.path.join(IMAGE_DIR, out_name)

    try:
        img.save(out_path, format="JPEG", quality=95)
        local_paths.append(out_path)
        status_list.append("ok")
        reason_list.append("downloaded_ok")
        ok_count += 1
    except Exception:
        local_paths.append("")
        status_list.append("failed")
        reason_list.append("save_failed")
        failed_count += 1

# Replace ImageURL with local paths and add status columns
df[img_col] = local_paths
df["download_status"] = status_list
df["download_reason"] = reason_list

# Save failed rows separately (keep extra columns for debugging)
failed_df = df[df["download_status"] != "ok"].copy()
if len(failed_df) > 0:
    failed_df.to_csv("failed_rows.csv", index=False, encoding="utf-8-sig")
    print("Saved failed rows to: failed_rows.csv")

# Keep ONLY successful rows in the final dataset
df_ok = df[df["download_status"] == "ok"].copy()

# Drop extra columns from the final output CSV
cols_to_drop = ["ImageURL_original", "download_status", "download_reason"]
df_ok = df_ok.drop(columns=[c for c in cols_to_drop if c in df_ok.columns])

# Save final clean local-only CSV
df_ok.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\nDone ✅")
print("Input:", INPUT_CSV)
print("Output:", OUTPUT_CSV)
print("Images folder:", IMAGE_DIR)
print("OK:", len(df_ok))
print("Failed:", len(failed_df))
