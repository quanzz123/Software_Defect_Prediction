import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from src.models import SoftwareDefectPredictionEngine

warnings.filterwarnings('ignore')

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
            
            param_grid = engine.get_param_grid(model_name)
            
            try:
                if param_grid:
                    # Rút gọn không gian tìm kiếm với RandomizedSearchCV (n_iter=10)
                    search = RandomizedSearchCV(
                        pipeline, 
                        param_distributions=param_grid, 
                        n_iter=10, 
                        cv=3, 
                        scoring='f1_weighted', 
                        random_state=random_state, 
                        n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    best_pipeline = search.best_estimator_
                else:
                    best_pipeline = pipeline
                    best_pipeline.fit(X_train, y_train)

                predictions = best_pipeline.predict(X_test)
                if hasattr(best_pipeline.named_steps['model'], "predict_proba"):
                    probabilities = best_pipeline.predict_proba(X_test)[:, 1]
                else:
                    probabilities = best_pipeline.decision_function(X_test)
                
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
