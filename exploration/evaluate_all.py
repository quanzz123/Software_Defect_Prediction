import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from scipy.io import arff

warnings.filterwarnings('ignore')

class SoftwareDefectPredictionEngine:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_learners = self._init_base_learners()
        self.classifiers = self._build_classifiers_suite()

    def _init_base_learners(self):
        """Khởi tạo các bộ phân loại cơ sở theo cấu hình mô tả trong bài báo gốc."""
        return {
            'RF': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'DS': DecisionTreeClassifier(random_state=self.random_state),
            'SVM': SVC(kernel='linear', probability=True, random_state=self.random_state, max_iter=2000),
            'LR': LogisticRegression(max_iter=1500, random_state=self.random_state, n_jobs=-1)
        }

    def _build_classifiers_suite(self):
        """Xây dựng 12 cấu hình mô hình từ sự kết hợp của bộ phân loại cơ sở và kỹ thuật tích hợp."""
        suite = {}
        # 1. Thêm các bộ phân loại cơ sở độc lập
        for name, clf in self.base_learners.items():
            suite[name] = clf
            
        # 2. Thêm các biến thể tích hợp tuần tự bằng AdaBoost
        for name, clf in self.base_learners.items():
            suite[f'AdaBoost_{name}'] = AdaBoostClassifier(
                estimator=clf, 
                random_state=self.random_state
            )
            
        # 3. Thêm các biến thể tích hợp song song bằng Bagging
        for name, clf in self.base_learners.items():
            suite[f'Bagging_{name}'] = BaggingClassifier(
                estimator=clf, 
                random_state=self.random_state,
                n_jobs=-1
            )
            
        return suite

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

def run_experiment_on_dataset(df, target_col='Defective', random_state=42):
    df = df.dropna()
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].map({'Defective': 1, 'Clean': 0}).values
    
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    engine = SoftwareDefectPredictionEngine(random_state=random_state)
    
    results_accumulator = {
        name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F-Score': [], 'ROC-AUC': []}
        for name in engine.classifiers.keys()
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for model_name, model in engine.classifiers.items():
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=random_state)),
                ('model', model)
            ])
            
            try:
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                if hasattr(pipeline.named_steps['model'], "predict_proba"):
                    probabilities = pipeline.predict_proba(X_test)[:, 1]
                else:
                    probabilities = pipeline.decision_function(X_test)
                
                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
                rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, probabilities, average='weighted')
                except ValueError:
                    auc = np.nan
                    
                results_accumulator[model_name]['Accuracy'].append(acc)
                results_accumulator[model_name]['Precision'].append(prec)
                results_accumulator[model_name]['Recall'].append(rec)
                results_accumulator[model_name]['F-Score'].append(f1)
                results_accumulator[model_name]['ROC-AUC'].append(auc)
            except Exception as e:
                pass
                
    final_report = {}
    for model_name, metrics in results_accumulator.items():
        final_report[model_name] = {
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
    
    all_results = []
    
    for file_path in sorted(arff_files):
        ds_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Evaluating 12 models on {ds_name}...")
        df, target_col = load_arff_dataset(file_path)
        report = run_experiment_on_dataset(df, target_col)
        
        for model_name, metrics in report.items():
            row = {'Dataset': ds_name, 'Model': model_name}
            row.update(metrics)
            all_results.append(row)
            
    df_results = pd.DataFrame(all_results)
    df_results = df_results[['Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']]
    
    # Save raw results
    df_results.to_csv('all_models_evaluation_results.csv', index=False)
    print("\nResults successfully saved to all_models_evaluation_results.csv")

if __name__ == '__main__':
    main()
