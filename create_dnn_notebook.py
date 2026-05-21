import json
import os

def create_notebook(filename, title, models_code, report_name):
    cells = []
    
    # Markdown Cell: Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            "Notebook này thiết lập các kiến trúc Deep Neural Network (Mạng nơ-ron sâu) khác nhau sử dụng `MLPClassifier`. Mục tiêu là thực nghiệm xem việc tăng chiều sâu (depth) hoặc chiều rộng (width) của mạng nơ-ron có giúp phát hiện mã nguồn lỗi tốt hơn các thuật toán học máy truyền thống hay không.\n"
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
            "\n",
            "sys.path.append(os.path.abspath('..'))\n",
            "\n",
            "from src.data_loader import load_arff_dataset\n",
            "from sklearn.model_selection import KFold\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
            "from imblearn.over_sampling import SMOTE\n",
            "from imblearn.pipeline import Pipeline as ImbPipeline\n",
            "import warnings\n",
            "from IPython.display import display\n",
            "\n",
            "warnings.filterwarnings('ignore')"
        ]
    })
    
    # Code Cell: Define Models
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            models_code
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
            "                if hasattr(pipeline.named_steps['model'], \"predict_proba\"):\n",
            "                    probabilities = pipeline.predict_proba(X_test)[:, 1]\n",
            "                else:\n",
            "                    probabilities = pipeline.decision_function(X_test)\n",
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
            "print(f\"Tìm thấy {len(arff_files)} tập dữ liệu. Bắt đầu huấn luyện mạng nơ-ron...\")\n",
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
            "print(\"\\n================ HOÀN TẤT THỰC NGHIỆM DNN ================\")\n",
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
    dnn_code = (
        "from sklearn.neural_network import MLPClassifier\n"
        "\n"
        "def get_models(random_state=42):\n"
        "    # Chúng ta sẽ sử dụng tham số early_stopping=True để tránh overfitting trên các tập nhỏ\n"
        "    # max_iter=1000 đảm bảo mạng nơ ron có đủ thời gian hội tụ\n"
        "    return {\n"
        "        'DNN_Shallow': MLPClassifier(\n"
        "            hidden_layer_sizes=(64,),\n"
        "            activation='relu', solver='adam', early_stopping=True, max_iter=1000, random_state=random_state\n"
        "        ),\n"
        "        'DNN_Medium': MLPClassifier(\n"
        "            hidden_layer_sizes=(128, 64),\n"
        "            activation='relu', solver='adam', early_stopping=True, max_iter=1000, random_state=random_state\n"
        "        ),\n"
        "        'DNN_Deep': MLPClassifier(\n"
        "            hidden_layer_sizes=(128, 64, 32),\n"
        "            activation='relu', solver='adam', early_stopping=True, max_iter=1000, random_state=random_state\n"
        "        ),\n"
        "        'DNN_Wide': MLPClassifier(\n"
        "            hidden_layer_sizes=(256, 128),\n"
        "            activation='relu', solver='adam', early_stopping=True, max_iter=1000, random_state=random_state\n"
        "        )\n"
        "    }"
    )
    
    create_notebook('04_Deep_Neural_Network.ipynb', 'Huấn Luyện Deep Neural Network Khảo Sát Kiến Trúc', dnn_code, 'report_DNN.csv')
    print("DNN Notebook created successfully.")

if __name__ == '__main__':
    main()
