import pandas as pd
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

metrics = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']

# Duyệt qua từng độ đo và tạo bảng
for metric in metrics:
    # Pivot bảng để Dataset làm hàng (index), Model làm cột, giá trị là độ đo
    df_pivot = df_all.pivot(index='Dataset', columns='Model', values=metric)
    
    # Sắp xếp lại thứ tự các cột mô hình cho đẹp (tuỳ chọn, ở đây để mặc định theo bảng chữ cái hoặc theo dữ liệu)
    
    # Lưu ra file CSV
    output_filename = f'table_{metric.replace("-", "_")}.csv'
    df_pivot.to_csv(output_filename)
    
    print(f"Đã lưu bảng tổng hợp cho {metric} vào: {output_filename}")
    print(df_pivot.head()) # In ra vài dòng để kiểm tra
    print("-" * 50)
