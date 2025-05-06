import pandas as pd
import os

# === Configuration ===
FILE_IDS = [34, 43, 44, 49, 52, 20, 21, 42, 60, 17, 36, 33, 8, 35, 48, 39, 7, 9, 3, 1]
BASE_PATH = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\opt\Malware-Project\BigDataset\IoTScenarios"
OUTPUT_PATH = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"

# === Schema ===
COLUMNS = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration',
    'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'label'
]

DROP_COLUMNS = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 
    'service', 'local_orig', 'local_resp', 'history'
]

NUMERIC_COLUMNS = ['duration', 'orig_bytes', 'resp_bytes']

LABEL_MAP = {
    v: k for k, v in {
        'PartOfAHorizontalPortScan': ['-   Malicious   PartOfAHorizontalPortScan', '(empty)   Malicious   PartOfAHorizontalPortScan'],
        'Okiru': ['-   Malicious   Okiru', '(empty)   Malicious   Okiru'],
        'Benign': ['-   Benign   -', '(empty)   Benign   -'],
        'DDoS': ['-   Malicious   DDoS'],
        'C&C': ['-   Malicious   C&C', '(empty)   Malicious   C&C'],
        'Attack': ['-   Malicious   Attack', '(empty)   Malicious   Attack'],
        'C&C-HeartBeat': ['-   Malicious   C&C-HeartBeat', '(empty)   Malicious   C&C-HeartBeat'],
        'C&C-FileDownload': ['-   Malicious   C&C-FileDownload'],
        'C&C-Torii': ['-   Malicious   C&C-Torii'],
        'C&C-HeartBeat-FileDownload': ['-   Malicious   C&C-HeartBeat-FileDownload'],
        'FileDownload': ['-   Malicious   FileDownload'],
        'C&C-Mirai': ['-   Malicious   C&C-Mirai'],
        'Okiru-Attack': ['-   Malicious   Okiru-Attack']
    }.items() for v in v
}


def normalize_label(label):
    return LABEL_MAP.get(label.strip(), label.strip())


def read_conn_log(file_path):
    """Reads a conn.log.labeled file and returns a cleaned DataFrame."""
    try:
        df = pd.read_table(
            file_path,
            skiprows=10,
            nrows=100000,
            names=COLUMNS,
            dtype=str,
            engine='python'
        )
        return df.iloc[:-1]  # Drop potentially malformed last row
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return pd.DataFrame()


def preprocess_dataframe(df):
    """Preprocesses the combined DataFrame: normalize labels, convert types, one-hot encode."""
    # Label normalization
    df['label'] = df['label'].apply(normalize_label)

    # Drop irrelevant columns
    df.drop(columns=DROP_COLUMNS, errors='ignore', inplace=True)

    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col].replace('-', '0'), errors='coerce')

    # Fill missing values
    df.fillna(-1, inplace=True)

    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=['proto', 'conn_state'], prefix=['proto', 'conn'])

    return df


def main():
    all_data = []

    for file_id in FILE_IDS:
        folder = f"CTU-IoT-Malware-Capture-{file_id}-1"
        file_path = os.path.join(BASE_PATH, folder, "bro", "conn.log.labeled")

        if os.path.exists(file_path):
            df = read_conn_log(file_path)
            if not df.empty:
                all_data.append(df)
        else:
            print(f"[WARNING] File not found: {file_path}")

    if not all_data:
        print("[ABORT] No data files could be processed.")
        return

    print("[INFO] Concatenating and preprocessing data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    processed_df = preprocess_dataframe(combined_df)

    processed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Processed data saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
