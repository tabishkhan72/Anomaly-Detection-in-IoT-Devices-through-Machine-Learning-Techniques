import pandas as pd
import os

# File IDs to process
file_ids = [34, 43, 44, 49, 52, 20, 21, 42, 60, 17, 36, 33, 8, 35, 48, 39, 7, 9, 3, 1]

# Base path
base_path = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\opt\Malware-Project\BigDataset\IoTScenarios"

# Columns for the log files
columns = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration',
    'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'label'
]

# Labels normalization map
label_replacements = {
    '-   Malicious   PartOfAHorizontalPortScan': 'PartOfAHorizontalPortScan',
    '(empty)   Malicious   PartOfAHorizontalPortScan': 'PartOfAHorizontalPortScan',
    '-   Malicious   Okiru': 'Okiru',
    '(empty)   Malicious   Okiru': 'Okiru',
    '-   Benign   -': 'Benign',
    '(empty)   Benign   -': 'Benign',
    '-   Malicious   DDoS': 'DDoS',
    '-   Malicious   C&C': 'C&C',
    '(empty)   Malicious   C&C': 'C&C',
    '-   Malicious   Attack': 'Attack',
    '(empty)   Malicious   Attack': 'Attack',
    '-   Malicious   C&C-HeartBeat': 'C&C-HeartBeat',
    '(empty)   Malicious   C&C-HeartBeat': 'C&C-HeartBeat',
    '-   Malicious   C&C-FileDownload': 'C&C-FileDownload',
    '-   Malicious   C&C-Torii': 'C&C-Torii',
    '-   Malicious   C&C-HeartBeat-FileDownload': 'C&C-HeartBeat-FileDownload',
    '-   Malicious   FileDownload': 'FileDownload',
    '-   Malicious   C&C-Mirai': 'C&C-Mirai',
    '-   Malicious   Okiru-Attack': 'Okiru-Attack'
}

# Columns to drop
drop_columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'service', 'local_orig', 'local_resp', 'history']

# Columns to treat as numeric
numeric_columns = ['duration', 'orig_bytes', 'resp_bytes']

def read_and_preprocess(file_path):
    """
    Reads and preprocesses a single conn.log.labeled file.
    """
    try:
        df = pd.read_table(
            filepath_or_buffer=file_path,
            skiprows=10,
            nrows=100000,
            names=columns,
            index_col=False,
            low_memory=False
        )
        df = df[:-1]  # Remove last row if it's malformed
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

# Read and combine data
dataframes = []
for file_id in file_ids:
    file_path = os.path.join(base_path, f"CTU-IoT-Malware-Capture-{file_id}-1", "bro", "conn.log.labeled")
    df = read_and_preprocess(file_path)
    if not df.empty:
        dataframes.append(df)

df_combined = pd.concat(dataframes, ignore_index=True)

# Normalize labels
df_combined['label'] = df_combined['label'].replace(label_replacements)

# Drop unused columns
df_combined.drop(columns=drop_columns, inplace=True, errors='ignore')

# Handle missing and malformed values
df_combined[numeric_columns] = df_combined[numeric_columns].replace('-', '0')
df_combined[numeric_columns] = df_combined[numeric_columns].astype(float)
df_combined.fillna(-1, inplace=True)

# One-hot encoding for categorical features
df_combined = pd.get_dummies(df_combined, columns=['proto', 'conn_state'], prefix=['proto', 'conn'])

# Save to CSV
output_path = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
df_combined.to_csv(output_path, index=False)
print(f"Data saved to: {output_path}")
