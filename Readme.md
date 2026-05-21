# Software Defect Prediction

Dự án thực nghiệm dự đoán lỗi phần mềm trên các bộ dữ liệu NASA MDP bằng các mô hình học máy có giám sát, kết hợp chuẩn hóa dữ liệu, SMOTE và đánh giá chéo 10-fold.

## Mục tiêu

- Đọc và tiền xử lý các bộ dữ liệu `.arff` trong thư mục `datasets/`.
- Chuẩn hóa nhãn lỗi thành hai lớp: `Defective` và `Clean`.
- Huấn luyện và so sánh 12 cấu hình mô hình:
  - Random Forest (`RF`)
  - Decision Tree (`DS`)
  - Support Vector Machine (`SVM`)
  - Logistic Regression (`LR`)
  - AdaBoost với từng base learner
  - Bagging với từng base learner
- Đánh giá bằng các chỉ số: Accuracy, Precision, Recall, F-Score và ROC-AUC.
- Xuất kết quả tổng hợp ra file CSV và biểu đồ so sánh.

## Cấu trúc thư mục

```text
.
+-- datasets/                         # 10 bộ dữ liệu NASA MDP dạng ARFF
+-- src/
|   +-- data_loader.py                # Đọc ARFF, chuẩn hóa nhãn và kiểu dữ liệu
|   +-- evaluation.py                 # Pipeline đánh giá 10-fold CV + SMOTE
|   +-- models.py                     # Khởi tạo các mô hình và param grid
+-- notebooks/                        # Notebook thực nghiệm và báo cáo theo nhóm mô hình
+-- exploration/                      # Script/phân tích mở rộng và báo cáo chi tiết
+-- main.py                           # Chạy toàn bộ thí nghiệm trên datasets/
+-- compare_dnn_vs_ml.py              # So sánh kết quả DNN với mô hình ML truyền thống
+-- all_models_evaluation_results.csv # Kết quả tổng hợp chính
+-- table_*.csv, comparison_*.png      # Bảng và biểu đồ so sánh theo từng metric
```

## Dữ liệu

Thư mục `datasets/` hiện gồm 10 bộ dữ liệu:

- `JM1.arff`
- `KC3.arff`
- `MC1.arff`
- `MC2.arff`
- `MW1.arff`
- `PC1.arff`
- `PC2.arff`
- `PC3.arff`
- `PC4.arff`
- `PC5.arff`

Mỗi file được đọc bằng `scipy.io.arff`, sau đó module `src/data_loader.py` tự động tìm cột nhãn phổ biến như `Defective`, `class`, `problems`, `problem` hoặc dùng cột cuối cùng nếu không tìm thấy.

## Cài đặt

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài các thư viện cần thiết:

```powershell
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

Nếu chạy các notebook DNN/CNN, cài thêm:

```powershell
pip install tensorflow
```

## Cách chạy pipeline chính

Chạy toàn bộ 12 mô hình trên tất cả file `.arff` trong `datasets/`:

```powershell
python main.py
```

Kết quả sẽ được lưu vào:

```text
all_models_evaluation_results.csv
```

File kết quả gồm các cột:

```text
Dataset, Model, Accuracy, Precision, Recall, F-Score, ROC-AUC
```

## Quy trình đánh giá

Pipeline trong `src/evaluation.py` thực hiện:

1. Xóa các dòng có giá trị thiếu bằng `dropna()`.
2. Tách đặc trưng `X` và nhãn `y`.
3. Chạy `KFold(n_splits=10, shuffle=True, random_state=42)`.
4. Với mỗi fold, tạo `imblearn.pipeline.Pipeline` gồm:
   - `StandardScaler`
   - `SMOTE`
   - mô hình phân loại
5. Tối ưu tham số bằng `RandomizedSearchCV` nếu mô hình có param grid.
6. Tính trung bình các chỉ số qua 10 fold.

Việc đặt `StandardScaler` và `SMOTE` bên trong pipeline giúp tránh rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm thử của từng fold.

## Notebook và báo cáo

Các notebook chính nằm trong `notebooks/`:

- `01_Base_Learners.ipynb`
- `02_AdaBoost_Ensemble.ipynb`
- `03_Bagging_Ensemble.ipynb`
- `04_Deep_Neural_Network.ipynb`
- `05_Convolutional_Neural_Network.ipynb`
- `Tuning_*.ipynb`

Mở Jupyter Notebook:

```powershell
jupyter notebook
```

Một số kết quả và phân tích chi tiết nằm trong:

- `exploration/all_models_detailed_comparison.md`
- `exploration/all_models_report.md`
- `notebooks/report_Base_Learners.csv`
- `notebooks/report_AdaBoost.csv`
- `notebooks/report_Bagging.csv`
- `notebooks/report_DNN.csv`

## Tạo bảng và biểu đồ so sánh

Các file `table_*.csv` và `comparison_*.png` là kết quả tổng hợp theo từng metric:

- `table_Accuracy.csv`
- `table_Precision.csv`
- `table_Recall.csv`
- `table_F_Score.csv`
- `table_ROC_AUC.csv`
- `comparison_Accuracy.png`
- `comparison_Precision.png`
- `comparison_Recall.png`
- `comparison_F_Score.png`
- `comparison_ROC_AUC.png`

Có thể chạy các script hỗ trợ trong `notebooks/` để tạo lại bảng và biểu đồ:

```powershell
python notebooks\generate_metric_tables.py
python notebooks\plot_models_comparison.py
```

Lưu ý: hai script này đang dùng đường dẫn tuyệt đối theo máy hiện tại. Nếu chuyển repo sang máy khác, cần sửa lại đường dẫn file CSV đầu vào.

## So sánh DNN với ML truyền thống

Chạy:

```powershell
python compare_dnn_vs_ml.py
```

Script này đọc kết quả từ:

- `exploration/all_models_evaluation_results.csv`
- `notebooks/report_DNN.csv`

Sau đó tính trung bình Accuracy, F-Score và ROC-AUC để so sánh mô hình DNN với các mô hình học máy truyền thống.

## Ghi chú

- `.venv/`, `.antigravity/`, `.antigravitycli/` và `__pycache__/` đã được đưa vào `.gitignore`.
- Nếu thêm dataset mới, chỉ cần đặt file `.arff` vào `datasets/` rồi chạy lại `python main.py`.
- Thời gian chạy có thể lâu vì pipeline dùng 10-fold CV, SMOTE và tìm kiếm tham số cho nhiều mô hình.
