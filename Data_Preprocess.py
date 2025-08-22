from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# =========================
# Configuration
# =========================

CAPTURE_IDS: Optional[List[int]] = [
    34, 43, 44, 49, 52, 20, 21, 42, 60, 17, 36, 33, 8, 35, 48, 39, 7, 9, 3, 1
]

BASE_PATH = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\opt\Malware-Project\BigDataset\IoTScenarios"
OUTPUT_CSV = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
OUTPUT_PARQUET: Optional[str] = None  # Optional Parquet output

CHUNK_SIZE = 500_000

COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration",
    "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "label",
]

DROP_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "service", "local_orig", "local_resp", "history",
]

NUMERIC_COLUMNS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
]

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


# =========================
# Utility Functions
# =========================

def normalize_label(raw: str) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    return LABEL_MAP.get(raw.strip(), raw.strip())


def discover_folders(base: Path, capture_ids: Optional[List[int]]) -> List[Path]:
    if capture_ids is None:
        return [p for p in base.rglob("conn.log.labeled") if "CTU-IoT-Malware-Capture" in str(p)]
    else:
        return [
            base / f"CTU-IoT-Malware-Capture-{cid}-1" / "bro" / "conn.log.labeled"
            for cid in capture_ids
        ]


def zeek_reader(path: Path, chunksize: int):
    def valid_lines():
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("#"):
                    yield line

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
    df = df[[c for c in COLUMNS if c in df.columns]].copy()

    if "label" in df.columns:
        df["label"] = df["label"].map(normalize_label)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("-", "0"), errors="coerce")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    df = df.fillna(-1)

    for cat_col in [("proto", "proto"), ("conn_state", "conn")]:
        if cat_col[0] in df.columns:
            df = pd.get_dummies(df, columns=[cat_col[0]], prefix=[cat_col[1]], dtype="uint8")

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float" if pd.api.types.is_float_dtype(df[col]) else "integer")

    return df


def write_incremental_csv(df: pd.DataFrame, out_path: Path, write_header: bool) -> None:
    df.to_csv(out_path, index=False, mode="w" if write_header else "a", header=write_header)


def write_incremental_parquet(df: pd.DataFrame, out_path: Path, write_header: bool) -> None:
    if write_header:
        df.to_parquet(out_path, index=False)


# =========================
# Main Processor
# =========================

def process_all():
    base = Path(BASE_PATH)
    out_csv = Path(OUTPUT_CSV)
    out_parquet = Path(OUTPUT_PARQUET) if OUTPUT_PARQUET else None

    files = discover_folders(base, CAPTURE_IDS)
    if not files:
        print("[ABORT] No input files found.")
        return

    wrote_header = False
    total_rows = 0
    files_found = 0

    for file_path in files:
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue

        print(f"[INFO] Processing: {file_path}")
        files_found += 1

        try:
            for chunk in zeek_reader(file_path, CHUNK_SIZE):
                if chunk.empty:
                    continue

                processed = preprocess_chunk(chunk)
                total_rows += len(processed)

                write_incremental_csv(processed, out_csv, write_header=not wrote_header)

                if out_parquet and not wrote_header:
                    write_incremental_parquet(processed, out_parquet, write_header=True)

                wrote_header = True
        except Exception as e:
            print(f"[ERROR] Failed while processing {file_path}: {e}")

    print(f"[SUCCESS] Completed. Files processed: {files_found}, Rows written: {total_rows}")
    print(f"[SUCCESS] CSV saved to: {out_csv}")
    if out_parquet:
        print(f"[SUCCESS] Parquet saved to: {out_parquet}")


if __name__ == "__main__":
    process_all()
