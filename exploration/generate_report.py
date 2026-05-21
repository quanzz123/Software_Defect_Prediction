import pandas as pd

df = pd.read_csv('all_models_evaluation_results.csv')

with open('all_models_report.md', 'w', encoding='utf-8') as f:
    f.write('# Báo Cáo Đánh Giá Toàn Diện Các Thuật Toán Học Máy\n\n')
    f.write('Báo cáo này trình bày kết quả thực nghiệm của 12 thuật toán (4 Base Learners: RF, DS, SVM, LR và 8 mô hình Ensembles: AdaBoost, Bagging) trên 10 tập dữ liệu của NASA MDP.\n\n')
    
    # 1. Trung bình trên tất cả các tập dữ liệu
    f.write('## 1. Hiệu Năng Trung Bình Toàn Cục\n\n')
    f.write('Trung bình cộng các chỉ số trên toàn bộ 10 tập dữ liệu cho từng thuật toán:\n\n')
    avg_df = df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']].mean().round(3).sort_values(by='F-Score', ascending=False)
    
    f.write('| Model | Accuracy | Precision | Recall | F-Score | ROC-AUC |\n')
    f.write('| --- | --- | --- | --- | --- | --- |\n')
    for idx, row in avg_df.iterrows():
        f.write(f"| {idx} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F-Score']:.3f} | {row['ROC-AUC']:.3f} |\n")
    
    f.write('\n\n')
    
    # 2. Xếp hạng thuật toán tốt nhất cho từng tập dữ liệu theo F-score
    f.write('## 2. Thuật Toán Tốt Nhất Theo Từng Tập Dữ Liệu (dựa trên F-Score)\n\n')
    f.write('| Tập Dữ Liệu | Thuật Toán Tốt Nhất (F-Score) | Điểm F-Score | Thuật Toán Tốt Nhất (ROC-AUC) | Điểm ROC-AUC |\n')
    f.write('| --- | --- | --- | --- | --- |\n')
    
    for dataset in df['Dataset'].unique():
        sub_df = df[df['Dataset'] == dataset]
        best_fscore = sub_df.loc[sub_df['F-Score'].idxmax()]
        best_roc = sub_df.loc[sub_df['ROC-AUC'].idxmax()]
        f.write(f"| **{dataset}** | {best_fscore['Model']} | {best_fscore['F-Score']:.2f} | {best_roc['Model']} | {best_roc['ROC-AUC']:.2f} |\n")
        
    f.write('\n## 3. Phân Tích Tổng Quan\n\n')
    f.write('> [!TIP]\n')
    f.write('> **Thuật toán hiệu quả nhất**: Các mô hình Ensemble, đặc biệt là **Random Forest (RF)** và **Bagging_RF** (Bagging với Base learner là Random Forest), tiếp tục duy trì thành tích tốt nhất quán nhất trên hầu hết các tập dữ liệu. Điều này rất đồng thuận với nhận định của tác giả Alsaeedi & Khan (2019).\n\n')
    f.write('> [!WARNING]\n')
    f.write('> **Sự thất bại của mô hình tuyến tính độc lập**: **SVM** và **Logistic Regression (LR)** có kết quả kém nhất về mặt phân loại, nguyên nhân có thể do tính phi tuyến tính cao và hiện tượng đa cộng tuyến cực kỳ nghiêm trọng trong các đặc trưng của NASA MDP (như đã thấy trong bước chạy EDA).\n\n')
    f.write('> [!NOTE]\n')
    f.write('> **Tác dụng của Bagging vs AdaBoost**: \n')
    f.write('> - **Bagging** giúp ổn định phương sai rất tốt và cải thiện rõ rệt kết quả của Decision Tree (DS). `Bagging_DS` đạt hiệu năng rất sát với Random Forest.\n')
    f.write('> - **AdaBoost** nhìn chung không cải thiện nhiều so với Base Learner hoặc thậm chí làm suy giảm hiệu năng đối với một số thuật toán kém tương thích như SVM hay LR, lý do là AdaBoost rất nhạy cảm với nhiễu và dễ bị quá khớp (overfitting) với các đặc trưng dư thừa của phần mềm.\n\n')

print("Markdown report generated at all_models_report.md")
