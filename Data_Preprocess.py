from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd


# =========================
# Configuration
# =========================

# If you want to limit to specific capture ids, set CAPTURE_IDS to integers like [34, 43, 44]
# If CAPTURE_IDS is None, the script will auto discover folders that match the CTU IoT layout.
CAPTURE_IDS: Optional[List[int]] = [
    34, 43, 44, 49, 52, 20, 21, 42, 60, 17, 36, 33, 8, 35, 48, 39, 7, 9, 3, 1
]

BASE_PATH = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\opt\Malware-Project\BigDataset\IoTScenarios"
OUTPUT_CSV = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
# Optional Parquet output for faster reads later
OUTPUT_PARQUET: Optional[str] = None  # for example r"C:\path\to\iot23_combined.parquet"

# Chunk size per file. Increase if you have lots of RAM, decrease if memory is limited.
CHUNK_SIZE = 500_000

# Keep only these columns and in this order, if present.
COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
    "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "label",
]

# Columns to drop before modeling
DROP_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "service", "local_orig", "local_resp", "history",
]

# Numeric columns to convert
NUMERIC_COLUMNS = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]

# Label normalization map
LABEL_MAP: Dict[str, str] = {
    "-   Malicious   PartOfAHorizontalPortScan": "PartOfAHorizontalPortScan",
    "(empty)   Malicious   PartOfAHorizontalPortScan": "PartOfAHorizontalPortScan",
    "-   Malicious   Okiru": "Okiru",
    "(empty)   Malicious   Okiru": "Okiru",
    "-   Benign   -": "Benign",
    "(empty)   Benign   -": "Benign",
    "-   Malicious   DDoS": "DDoS",
    "-   Malicious   C&C": "C&C",
    "(empty)   Malicious   C&C": "C&C",
    "-   Malicious   Attack": "Attack",
    "(empty)   Malicious   Attack": "Attack",
    "-   Malicious   C&C-HeartBeat": "C&C-HeartBeat",
    "(empty)   Malicious   C&C-HeartBeat": "C&C-HeartBeat",
    "-   Malicious   C&C-FileDownload": "C&C-FileDownload",
    "-   Malicious   C&C-Torii": "C&C-Torii",
    "-   Malicious   C&C-HeartBeat-FileDownload": "C&C-HeartBeat-FileDownload",
    "-   Malicious   FileDownload": "FileDownload",
    "-   Malicious   C&C-Mirai": "C&C-Mirai",
    "-   Malicious   Okiru-Attack": "Okiru-Attack",
}


def normalize_label(raw: str) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    s = raw.strip()
    return LABEL_MAP.get(s, s)


def discover_folders(base: Path, capture_ids: Optional[List[int]]) -> List[Path]:
    """Find all Zeek conn.log.labeled files under the CTU IoT layout."""
    candidates: List[Path] = []
    if capture_ids is None:
        # Auto discover folders that look like CTU IoT captures
        for p in base.rglob("conn.log.labeled"):
            if "CTU-IoT-Malware-Capture" in str(p):
                candidates.append(p)
    else:
        for cid in capture_ids:
            folder = base / f"CTU-IoT-Malware-Capture-{cid}-1" / "bro" / "conn.log.labeled"
            candidates.append(folder)
    return candidates


def zeek_reader(path: Path, chunksize: int):
    """Stream rows from a Zeek labeled conn log, skipping meta header lines that start with '#'."""
    # Many Zeek logs have meta lines starting with '#'
    # We filter them using a custom iterator that skips such lines.
    def valid_lines():
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                yield line

    # Use python engine with whitespace separation since Zeek fields are whitespace or tab separated
    return pd.read_csv(
        filepath_or_buffer=valid_lines(),
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
        chunksize=chunksize,
        dtype=str,
        on_bad_lines="skip",
    )


def preprocess_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only expected columns that exist
    existing = [c for c in COLUMNS if c in df.columns]
    df = df[existing].copy()

    # Normalize labels
    if "label" in df.columns:
        df["label"] = df["label"].map(normalize_label)

    # Convert numeric columns robustly
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        df[col] = pd.to_numeric(df[col].replace("-", "0"), errors="coerce")

    # Drop columns not needed downstream
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    # Fill missing
    df = df.fillna(-1)

    # One hot encode a small set of categoricals
    for cat_col, prefix in [("proto", "proto"), ("conn_state", "conn")]:
        if cat_col in df.columns:
            df = pd.get_dummies(df, columns=[cat_col], prefix=[prefix], dtype="uint8")

    # Downcast numeric types to save space
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def write_incremental_csv(df: pd.DataFrame, out_path: Path, write_header: bool) -> None:
    df.to_csv(out_path, index=False, mode="w" if write_header else "a", header=write_header)


def write_incremental_parquet(df: pd.DataFrame, out_path: Path, write_header: bool) -> None:
    # For simplicity, append by concatenating in memory for Parquet is not ideal.
    # In practice, better to write one Parquet per source and later read as a dataset.
    # Here we write a single file if header is True, else append is skipped.
    if write_header:
        df.to_parquet(out_path, index=False)


def process_all():
    base = Path(BASE_PATH)
    out_csv = Path(OUTPUT_CSV)
    out_parquet = Path(OUTPUT_PARQUET) if OUTPUT_PARQUET else None

    files = discover_folders(base, CAPTURE_IDS)

    if not files:
        print("[ABORT] No input files found")
        return

    wrote_header = False
    total_rows = 0
    files_found = 0

    for file_path in files:
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        files_found += 1
        print(f"[INFO] Processing: {file_path}")

        try:
            for chunk in zeek_reader(file_path, CHUNK_SIZE):
                if chunk.empty:
                    continue
                processed = preprocess_chunk(chunk)
                total_rows += len(processed)

                write_incremental_csv(processed, out_csv, write_header=not wrote_header)
                if out_parquet is not None and not wrote_header:
                    # only write once for demo purposes
                    write_incremental_parquet(processed, out_parquet, write_header=True)

                wrote_header = True
        except Exception as e:
            print(f"[ERROR] Failed while processing {file_path}: {e}")

    print(f"[SUCCESS] Completed. Files processed: {files_found}, Rows written: {total_rows}")
    print(f"[SUCCESS] CSV saved to: {out_csv}")
    if out_parquet is not None:
        print(f"[SUCCESS] Parquet saved to: {out_parquet}")


if __name__ == "__main__":
    process_all()
