import pandas as pd
import os

# Function to read and preprocess each file
def read_and_preprocess(file_path):
    df = pd.read_table(filepath_or_buffer=file_path, skiprows=10, nrows=100000, 
                       names=['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration',
                              'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
                              'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'label'], 
                       index_col=False)
    df = df[:-1]  # Drop the last row
    return df

# List of file paths
file_paths = ["C:/Users/tabis/OneDrive/Desktop/SU_Classes/IOT/PROJECT/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-{0}-1/bro/conn.log.labeled".format(i) for i in [34, 43, 44, 49, 52, 20, 21, 42, 60, 17, 36, 33, 8, 35, 48, 39, 7, 9, 3, 1]]

# Read and combine dataframes
frames = [read_and_preprocess(file) for file in file_paths]
df_combined = pd.concat(frames, ignore_index=True)

# Simplify label names
label_replacements = {
    '-   Malicious   PartOfAHorizontalPortScan': 'PartOfAHorizontalPortScan',
    '(empty)   Malicious   PartOfAHorizontalPortScan': 'PartOfAHorizontalPortScan',
    '-   Malicious   Okiru': 'Okiru',
    '(empty)   Malicious   Okiru': 'Okiru',
    '-   Benign   -': 'Benign',
    '(empty)   Benign   -': 'Benign',
    # Add other replacements as needed
}
df_combined['label'] = df_combined['label'].replace(label_replacements)

# Drop unnecessary columns
columns_to_drop = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'service', 'local_orig', 'local_resp', 'history']
df_combined.drop(columns=columns_to_drop, inplace=True)

# Convert '-' to '0' in numeric columns and create dummies
numeric_cols = ['duration', 'orig_bytes', 'resp_bytes']
df_combined[numeric_cols] = df_combined[numeric_cols].replace('-', '0')
df_combined[numeric_cols] = df_combined[numeric_cols].astype(float)
df_combined.fillna(-1, inplace=True)

# Create dummy variables
df_combined = pd.get_dummies(df_combined, columns=['proto', 'conn_state'])

# Save to CSV in the specified directory
output_filepath = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
df_combined.to_csv(output_filepath, index=False)
