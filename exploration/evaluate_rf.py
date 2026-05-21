import os
import glob
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

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

def run_rf_experiment(df, target_col='Defective', random_state=42):
    # Drop rows with NaN as RandomForest doesn't handle them implicitly in sklearn
    df = df.dropna()
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].map({'Defective': 1, 'Clean': 0}).values
    
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    
    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F-Score': [],
        'ROC-AUC': []
    }
    
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # According to the paper and rule.md: Standard scale -> SMOTE -> RF
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=random_state)),
            ('rf', RandomForestClassifier(random_state=random_state))
        ])
        
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            
            # Use weighted average as mentioned in paper
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
            rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, probabilities, average='weighted')
            except ValueError:
                auc = np.nan
                
            metrics['Accuracy'].append(acc)
            metrics['Precision'].append(prec)
            metrics['Recall'].append(rec)
            metrics['F-Score'].append(f1)
            metrics['ROC-AUC'].append(auc)
        except Exception as e:
            # e.g., if SMOTE fails due to too few samples of a class in a fold
            pass
            
    final_report = {
        'Accuracy': np.round(np.mean(metrics['Accuracy']), 2) if metrics['Accuracy'] else np.nan,
        'Precision': np.round(np.mean(metrics['Precision']), 2) if metrics['Precision'] else np.nan,
        'Recall': np.round(np.mean(metrics['Recall']), 2) if metrics['Recall'] else np.nan,
        'F-Score': np.round(np.mean(metrics['F-Score']), 2) if metrics['F-Score'] else np.nan,
        'ROC-AUC': np.round(np.nanmean(metrics['ROC-AUC']), 2) if metrics['ROC-AUC'] else np.nan
    }
    return final_report

def main():
    dataset_dir = 'datasets'
    arff_files = glob.glob(os.path.join(dataset_dir, '*.arff'))
    
    results = []
    
    for file_path in sorted(arff_files):
        ds_name = os.path.splitext(os.path.basename(file_path))[0]
        df, target_col = load_arff_dataset(file_path)
        print(f"Evaluating {ds_name}...")
        report = run_rf_experiment(df, target_col)
        report['Dataset'] = ds_name
        results.append(report)
        
    df_results = pd.DataFrame(results)
    # Reorder columns
    df_results = df_results[['Dataset', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']]
    
    print("\n=== Random Forest Results on 10 Datasets (10-fold CV + SMOTE) ===")
    print(df_results.to_string(index=False))
    
    # Save to CSV
    df_results.to_csv('rf_evaluation_results.csv', index=False)
    print("\nResults saved to rf_evaluation_results.csv")

if __name__ == '__main__':
    main()
