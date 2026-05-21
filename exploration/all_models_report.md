# Báo Cáo Đánh Giá Toàn Diện Các Thuật Toán Học Máy

Báo cáo này trình bày kết quả thực nghiệm của 12 thuật toán (4 Base Learners: RF, DS, SVM, LR và 8 mô hình Ensembles: AdaBoost, Bagging) trên 10 tập dữ liệu của NASA MDP.

## 1. Hiệu Năng Trung Bình Toàn Cục

Trung bình cộng các chỉ số trên toàn bộ 10 tập dữ liệu cho từng thuật toán:

| Model | Accuracy | Precision | Recall | F-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| AdaBoost_RF | 0.849 | 0.855 | 0.849 | 0.850 | 0.662 |
| Bagging_RF | 0.848 | 0.861 | 0.848 | 0.849 | 0.813 |
| RF | 0.848 | 0.854 | 0.848 | 0.845 | 0.803 |
| Bagging_DS | 0.842 | 0.844 | 0.842 | 0.837 | 0.759 |
| AdaBoost_DS | 0.817 | 0.842 | 0.817 | 0.824 | 0.634 |
| DS | 0.809 | 0.837 | 0.809 | 0.818 | 0.630 |
| LR | 0.758 | 0.852 | 0.758 | 0.790 | 0.776 |
| Bagging_LR | 0.758 | 0.851 | 0.758 | 0.789 | 0.776 |
| AdaBoost_SVM | 0.757 | 0.853 | 0.757 | 0.788 | 0.744 |
| AdaBoost_LR | 0.753 | 0.844 | 0.753 | 0.787 | 0.770 |
| SVM | 0.675 | 0.832 | 0.675 | 0.702 | 0.730 |
| Bagging_SVM | 0.653 | 0.803 | 0.653 | 0.677 | 0.739 |


## 2. Thuật Toán Tốt Nhất Theo Từng Tập Dữ Liệu (dựa trên F-Score)

| Tập Dữ Liệu | Thuật Toán Tốt Nhất (F-Score) | Điểm F-Score | Thuật Toán Tốt Nhất (ROC-AUC) | Điểm ROC-AUC |
| --- | --- | --- | --- | --- |
| **JM1** | RF | 0.77 | Bagging_RF | 0.72 |
| **KC3** | AdaBoost_RF | 0.80 | Bagging_RF | 0.77 |
| **MC1** | RF | 0.98 | RF | 0.88 |
| **MC2** | AdaBoost_DS | 0.73 | Bagging_RF | 0.75 |
| **MW1** | RF | 0.86 | AdaBoost_SVM | 0.76 |
| **PC1** | RF | 0.91 | RF | 0.88 |
| **PC2** | RF | 0.96 | Bagging_RF | 0.86 |
| **PC3** | AdaBoost_RF | 0.85 | RF | 0.84 |
| **PC4** | AdaBoost_RF | 0.91 | RF | 0.93 |
| **PC5** | RF | 0.77 | Bagging_RF | 0.82 |

## 3. Phân Tích Tổng Quan

> [!TIP]
> **Thuật toán hiệu quả nhất**: Các mô hình Ensemble, đặc biệt là **Random Forest (RF)** và **Bagging_RF** (Bagging với Base learner là Random Forest), tiếp tục duy trì thành tích tốt nhất quán nhất trên hầu hết các tập dữ liệu. Điều này rất đồng thuận với nhận định của tác giả Alsaeedi & Khan (2019).

> [!WARNING]
> **Sự thất bại của mô hình tuyến tính độc lập**: **SVM** và **Logistic Regression (LR)** có kết quả kém nhất về mặt phân loại, nguyên nhân có thể do tính phi tuyến tính cao và hiện tượng đa cộng tuyến cực kỳ nghiêm trọng trong các đặc trưng của NASA MDP (như đã thấy trong bước chạy EDA).

> [!NOTE]
> **Tác dụng của Bagging vs AdaBoost**: 
> - **Bagging** giúp ổn định phương sai rất tốt và cải thiện rõ rệt kết quả của Decision Tree (DS). `Bagging_DS` đạt hiệu năng rất sát với Random Forest.
> - **AdaBoost** nhìn chung không cải thiện nhiều so với Base Learner hoặc thậm chí làm suy giảm hiệu năng đối với một số thuật toán kém tương thích như SVM hay LR, lý do là AdaBoost rất nhạy cảm với nhiễu và dễ bị quá khớp (overfitting) với các đặc trưng dư thừa của phần mềm.

