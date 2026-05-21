import pandas as pd
import numpy as np
from scipy.io import arff

def decode_bytes(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return val

def standardize_target(val):
    if val is None or pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip().upper()
    if val_str in ['Y', 'YES', 'TRUE', '1', '1.0', 'T', 'DEFECTIVE', "B'Y'", "B'TRUE'"]:
        return 'Defective'
    elif val_str in ['N', 'NO', 'FALSE', '0', '0.0', 'F', 'CLEAN', "B'N'", "B'FALSE'"]:
        return 'Clean'
    return val_str

def load_arff_dataset(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype == 'O':
            df[col] = df[col].apply(decode_bytes)
    df = df.replace('?', np.nan)
    target_candidates = ['Defective', 'defective', 'class', 'problems', 'problem']
    target_col = None
    for c in target_candidates:
        for actual_col in df.columns:
            if actual_col.lower() == c.lower():
                target_col = actual_col
                break
        if target_col:
            break
    if not target_col:
        target_col = df.columns[-1]
    df[target_col] = df[target_col].apply(standardize_target)
    for col in df.columns:
        if col != target_col:
            if df[col].dtype == object or df[col].dtype == 'O':
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass
    return df, target_col
