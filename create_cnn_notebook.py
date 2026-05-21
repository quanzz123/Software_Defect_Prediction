import json
import os

def create_notebook(filename, title, cnn_code, report_name):
    cells = []
    
    # Markdown Cell: Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            "Notebook này triển khai Mạng Nơ-ron Tích Chập 1 Chiều (1D-CNN) sử dụng `PyTorch` để huấn luyện trên dữ liệu dạng bảng (tabular data). CNN-1D có khả năng trích xuất các đặc trưng cục bộ (local patterns) giữa các metrics lân cận trong mã nguồn.\n",
            "\n",
            "Chúng ta bọc mô hình PyTorch vào một lớp Wrapper chuẩn Scikit-Learn để tái sử dụng lại `Imblearn Pipeline` nhằm đảm bảo chuẩn 10-Fold CV + SMOTE."
        ]
    })
    
    # Code Cell: Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import os\n",
            "import glob\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "\n",
            "sys.path.append(os.path.abspath('..'))\n",
            "\n",
            "from src.data_loader import load_arff_dataset\n",
            "from sklearn.model_selection import KFold\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.base import BaseEstimator, ClassifierMixin\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
            "from imblearn.over_sampling import SMOTE\n",
            "from imblearn.pipeline import Pipeline as ImbPipeline\n",
            "import warnings\n",
            "from IPython.display import display\n",
            "\n",
            "warnings.filterwarnings('ignore')"
        ]
    })
    
    # Code Cell: Define PyTorch wrapper and models
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            cnn_code
        ]
    })
    
    # Code Cell: Evaluation Function
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def run_experiment_on_dataset(df, target_col='Defective', random_state=42):\n",
            "    df = df.dropna()\n",
            "    X = df.drop(columns=[target_col]).values\n",
            "    y = df[target_col].map({'Defective': 1, 'Clean': 0}).values\n",
            "    \n",
            "    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)\n",
            "    models_dict = get_models()\n",
            "    \n",
            "    results_accumulator = {\n",
            "        name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F-Score': [], 'ROC-AUC': []}\n",
            "        for name in models_dict.keys()\n",
            "    }\n",
            "    \n",
            "    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):\n",
            "        X_train, X_test = X[train_idx], X[test_idx]\n",
            "        y_train, y_test = y[train_idx], y[test_idx]\n",
            "        \n",
            "        for model_name, model in models_dict.items():\n",
            "            pipeline = ImbPipeline([\n",
            "                ('scaler', StandardScaler()),\n",
            "                ('smote', SMOTE(random_state=random_state)),\n",
            "                ('model', model)\n",
            "            ])\n",
            "            \n",
            "            try:\n",
            "                pipeline.fit(X_train, y_train)\n",
            "                predictions = pipeline.predict(X_test)\n",
            "                probabilities = pipeline.predict_proba(X_test)[:, 1]\n",
            "                \n",
            "                acc = accuracy_score(y_test, predictions)\n",
            "                prec = precision_score(y_test, predictions, average='weighted', zero_division=0)\n",
            "                rec = recall_score(y_test, predictions, average='weighted', zero_division=0)\n",
            "                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)\n",
            "                try:\n",
            "                    auc = roc_auc_score(y_test, probabilities, average='weighted')\n",
            "                except ValueError:\n",
            "                    auc = np.nan\n",
            "                    \n",
            "                results_accumulator[model_name]['Accuracy'].append(acc)\n",
            "                results_accumulator[model_name]['Precision'].append(prec)\n",
            "                results_accumulator[model_name]['Recall'].append(rec)\n",
            "                results_accumulator[model_name]['F-Score'].append(f1)\n",
            "                results_accumulator[model_name]['ROC-AUC'].append(auc)\n",
            "            except Exception as e:\n",
            "                pass\n",
            "                \n",
            "    final_report = {}\n",
            "    for model_name, metrics in results_accumulator.items():\n",
            "        final_report[model_name] = {\n",
            "            'Accuracy': np.round(np.mean(metrics['Accuracy']), 3) if metrics['Accuracy'] else 0.0,\n",
            "            'Precision': np.round(np.mean(metrics['Precision']), 3) if metrics['Precision'] else 0.0,\n",
            "            'Recall': np.round(np.mean(metrics['Recall']), 3) if metrics['Recall'] else 0.0,\n",
            "            'F-Score': np.round(np.mean(metrics['F-Score']), 3) if metrics['F-Score'] else 0.0,\n",
            "            'ROC-AUC': np.round(np.nanmean(metrics['ROC-AUC']), 3) if metrics['ROC-AUC'] else 0.0\n",
            "        }\n",
            "    return final_report"
        ]
    })
    
    # Code Cell: Execution Loop
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "dataset_dir = '../datasets'\n",
            "arff_files = glob.glob(os.path.join(dataset_dir, '*.arff'))\n",
            "\n",
            "all_results = []\n",
            "\n",
            "print(f\"Tìm thấy {len(arff_files)} tập dữ liệu. Bắt đầu huấn luyện CNN...\")\n",
            "\n",
            "for file_path in sorted(arff_files):\n",
            "    ds_name = os.path.splitext(os.path.basename(file_path))[0]\n",
            "    print(f\"\\n>>> Đang xử lý tập dữ liệu: {ds_name}...\")\n",
            "    df, target_col = load_arff_dataset(file_path)\n",
            "    report = run_experiment_on_dataset(df, target_col)\n",
            "    \n",
            "    for model_name, metrics in report.items():\n",
            "        row = {'Dataset': ds_name, 'Model': model_name}\n",
            "        row.update(metrics)\n",
            "        all_results.append(row)\n",
            "        \n",
            "results_df = pd.DataFrame(all_results)\n",
            "results_df = results_df[['Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']]\n",
            "\n",
            "print(\"\\n================ HOÀN TẤT THỰC NGHIỆM CNN ================\")\n",
            "display(results_df)\n",
            "\n",
            "# Lưu báo cáo\n",
            f"output_file = '{report_name}'\n",
            "results_df.to_csv(output_file, index=False)\n",
            "print(f\"\\nKết quả đã được lưu thành công vào: {output_file}\")"
        ]
    })
    
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs('notebooks', exist_ok=True)
    with open(os.path.join('notebooks', filename), 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

def main():
    cnn_code = '''
# 1. Định nghĩa các khối kiến trúc CNN 1D
class CNN_Shallow_Net(nn.Module):
    def __init__(self):
        super(CNN_Shallow_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_Medium_Net(nn.Module):
    def __init__(self):
        super(CNN_Medium_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_Deep_Net(nn.Module):
    def __init__(self):
        super(CNN_Deep_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_Wide_Net(nn.Module):
    def __init__(self):
        super(CNN_Wide_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Xây dựng Wrapper Scikit-Learn để tái sử dụng K-Fold Pipeline
class SklearnPyTorchCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, epochs=50, lr=0.001, batch_size=32):
        self.model_class = model_class
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = self.model_class()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Chuyển đổi dữ liệu về dạng Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # shape: (batch, channels=1, features)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            outputs = self.model(X_tensor)
            # Áp dụng Softmax để lấy xác suất lớp
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()

def get_models(random_state=42):
    return {
        'CNN_Shallow': SklearnPyTorchCNNClassifier(CNN_Shallow_Net, epochs=50),
        'CNN_Medium': SklearnPyTorchCNNClassifier(CNN_Medium_Net, epochs=50),
        'CNN_Deep': SklearnPyTorchCNNClassifier(CNN_Deep_Net, epochs=50),
        'CNN_Wide': SklearnPyTorchCNNClassifier(CNN_Wide_Net, epochs=50)
    }
'''
    create_notebook('05_Convolutional_Neural_Network.ipynb', 'Huấn Luyện CNN-1D Khảo Sát Kiến Trúc', cnn_code, 'report_CNN.csv')
    print("CNN Notebook created successfully.")

if __name__ == '__main__':
    main()
