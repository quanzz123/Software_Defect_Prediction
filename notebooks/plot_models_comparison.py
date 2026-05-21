import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Định nghĩa đường dẫn tới các file CSV
base_file = r'D:\WORKSPACE\KPDL\Software_Defect_Prediction\notebooks\report_Base_Learners.csv'
ada_file = r'D:\WORKSPACE\KPDL\Software_Defect_Prediction\notebooks\report_AdaBoost.csv'
bag_file = r'D:\WORKSPACE\KPDL\Software_Defect_Prediction\notebooks\report_Bagging.csv'

# Đọc và kết hợp dữ liệu từ 3 file CSV
df_base = pd.read_csv(base_file)
df_ada = pd.read_csv(ada_file)
df_bag = pd.read_csv(bag_file)

df_all = pd.concat([df_base, df_ada, df_bag], ignore_index=True)

def plot_metric(df, metric):
    """
    Hàm vẽ biểu đồ cột so sánh các mô hình theo một độ đo cụ thể.
    """
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    
    # Vẽ biểu đồ cột nhóm
    # x: 10 dataset, y: giá trị độ đo, hue: 12 mô hình
    ax = sns.barplot(
        data=df, 
        x='Dataset', 
        y=metric, 
        hue='Model',
        palette='Paired' # Sử dụng palette màu phù hợp để phân biệt 12 mô hình
    )
    
    plt.title(f'So sánh {metric} của 12 mô hình trên 10 Dataset', fontsize=18, fontweight='bold')
    plt.xlabel('Dataset', fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Di chuyển legend ra ngoài để không che biểu đồ
    plt.legend(title='Mô hình', title_fontsize='13', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    
    # Tự động căn chỉnh lại layout cho vừa vặn
    plt.tight_layout()
    
    # Lưu biểu đồ thành file ảnh
    output_filename = f'comparison_{metric.replace("-", "_")}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ: {output_filename}")
    
    # Hiển thị biểu đồ (nếu chạy trong Jupyter Notebook thì sẽ hiện ra)
    plt.show()

# Danh sách các độ đo có trong dữ liệu
metrics = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']

# Vẽ biểu đồ cho từng độ đo
for metric in metrics:
    plot_metric(df_all, metric)
