import sys
import os
import glob
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('.'))

from src.data_loader import load_arff_dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def get_models(random_state=42):
    base_estimators = {
        'RF': RandomForestClassifier(random_state=random_state, n_jobs=-1),
        'DS': DecisionTreeClassifier(random_state=random_state),
        'SVM': SVC(kernel='linear', C=0.1, probability=True, random_state=random_state, max_iter=2000),
        'LR': LogisticRegression(max_iter=1500, random_state=random_state, n_jobs=-1)
    }
    
    adaboost_models = {}
    for name, clf in base_estimators.items():
        adaboost_models[f'AdaBoost_{name}'] = AdaBoostClassifier(
            estimator=clf,
            random_state=random_state
        )
    return adaboost_models

df, target_col = load_arff_dataset('datasets/JM1.arff')
df = df.dropna()
X = df.drop(columns=[target_col]).values
y = df[target_col].map({'Defective': 1, 'Clean': 0}).values

kf = KFold(n_splits=2, shuffle=True, random_state=42)
models_dict = get_models()

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    for model_name, model in models_dict.items():
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        try:
            pipeline.fit(X_train, y_train)
            print(f"{model_name} fit OK")
        except Exception as e:
            print(f"{model_name} failed: {type(e).__name__} - {str(e)}")
    break
