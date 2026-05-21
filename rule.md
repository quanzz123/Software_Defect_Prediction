# Nghiên Cứu Thực Nghiệm Đối Chiếu Hệ Thống Dự Đoán Lỗi Phần Mềm Sử Dụng Học Máy Có Giám Sát Và Phương Pháp Tích Hợp Trên Nền Tảng Cân Bằng Dữ Liệu SMOTE

## Bối Cảnh Và Cơ Sở Lý Thuyết Của Hệ Thống Dự Đoán Lỗi Phần Mềm

Trong kỹ nghệ phần mềm hiện đại, việc phát hiện và khắc phục các lỗi lập trình ngay từ những giai đoạn đầu của vòng đời phát triển dự án (SDLC) đóng vai trò quyết định đối với chất lượng và độ tin cậy của hệ thống. Các lỗi phát sinh trong mã nguồn thường bắt nguồn từ những sai sót trong khâu thiết kế, phân tích yêu cầu hoặc các lỗi vô ý của lập trình viên trong quá trình triển khai. Quy trình kiểm thử truyền thống trên toàn bộ các mô-đun mã nguồn tiêu tốn một lượng lớn tài nguyên và thời gian, tạo ra rào cản lớn đối với việc bàn giao sản phẩm đúng thời hạn. Để tối ưu hóa quá trình này, kỹ thuật dự đoán lỗi phần mềm (SDP) đã ra đời như một giải pháp cứu cánh, cho phép khai thác dữ liệu lịch sử thông qua các thuật toán học máy có giám sát để phân loại và định vị các mô-đun có khả năng chứa lỗi cao. Bằng cách này, đội ngũ kiểm thử có thể tập trung nguồn lực vào những phân hệ trọng yếu, giảm thiểu rủi ro vận hành sau khi bàn giao sản phẩm.

Hiệu năng của các mô hình dự đoán lỗi phần mềm phụ thuộc trực tiếp vào các đặc trưng đo lường mã nguồn (Software Metrics) được sử dụng để huấn luyện mô hình. Các số đo dòng lệnh (Lines of Code - LOC) là những chỉ số kích thước cơ bản nhất và đã được chứng minh là có tương quan chặt chẽ với mật độ phát sinh lỗi. Bên cạnh đó, các số đo độ phức tạp Cyclomatic của McCabe được tính toán dựa trên cấu trúc đồ thị dòng điều khiển của chương trình, phản ánh số lượng nhánh quyết định tuyến tính độc lập nhằm định lượng khả năng duy trì và mức độ phức tạp của mã. Một nhóm chỉ số quan trọng khác là các đặc trưng độ phức tạp phần mềm của Halstead, được xây dựng dựa trên số lượng toán tử và toán hạng độc nhất cũng như tổng số lượng của chúng, từ đó ước lượng các khía cạnh vật lý như từ vựng, độ dài, thể tích, độ khó, nỗ lực lập trình và thời gian hoàn thành của mô-đun. Cuối cùng, đối với các hệ thống hiện đại, các số đo hướng đối tượng của Chidamber và Kemerer (CK) như độ sâu cây thừa kế hay số lượng phương thức trên mỗi lớp cũng cung cấp những thông tin sâu sắc giúp phát hiện lỗi hiệu quả hơn so với các số đo kích thước truyền thống.

## Phân Tích Sự Khác Biệt Giữa Các Phiên Bản Dữ Liệu Sạch NASA MDP

Trong các nghiên cứu thực nghiệm về dự đoán lỗi phần mềm, việc sử dụng các bộ dữ liệu chuẩn hóa từ kho lưu trữ Metrics Data Program (MDP) của NASA là một tiêu chuẩn bắt buộc để đảm bảo tính khách quan và khả năng so sánh đối chiếu. Tuy nhiên, các kỹ sư dữ liệu cần lưu ý rằng dữ liệu gốc từ NASA thường chứa nhiều lỗi định dạng, dữ liệu khuyết thiếu cũng như các mẫu bị trùng lặp hoặc không nhất quán. Do đó, cộng đồng nghiên cứu đã phát triển nhiều phiên bản làm sạch khác nhau, phổ biến nhất là phiên bản $DS'$ chứa các mẫu trùng lặp và phiên bản $DS''$ (còn được gọi là phiên bản của Shepperd) đã loại bỏ hoàn toàn các mẫu trùng lặp và không nhất quán. Sự khác biệt về mặt số liệu giữa các phiên bản này có ảnh hưởng trực tiếp đến kết quả huấn luyện mô hình học máy.

Bảng dưới đây thống kê chi tiết các thông số của 10 bộ dữ liệu được sử dụng trong nghiên cứu của Alsaeedi và Khan (2019) đối chiếu với các số liệu từ các phiên bản làm sạch khác trong kho lưu trữ mã nguồn mở:

| Bộ dữ liệu | Số lượng mô-đun (Bài báo gốc) | Số mô-đun lỗi (Bài báo gốc) | Số lượng đặc trưng (Bài báo gốc) | Số lượng mô-đun (Sạch $DS''$) | Số mô-đun lỗi (Sạch $DS''$) | Số lượng đặc trưng (Sạch $DS''$) | Nguồn gốc và môi trường hệ thống |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JM1** | 7782 | 1672 | 21 | 7720 | 1612 | 22 | Phần mềm mô phỏng dự đoán mặt đất |
| **KC3** | 194 | 36 | 39 | 194 | 36 | 40 | Hệ thống phần mềm quản lý lưu trữ dữ liệu |
| **MC1** | 1988 | 46 | 38 | 1952 | 36 | 39 | Hệ thống điều khiển và xử lý thông tin |
| **MC2** | 125 | 44 | 39 | 124 | 44 | 40 | Hệ thống phần mềm điều khiển quỹ đạo |
| **MW1** | 253 | 27 | 37 | 250 | 25 | 38 | Phần mềm hệ thống viễn thông vũ trụ |
| **PC1** | 705 | 61 | 37 | 679 | 55 | 38 | Phần mềm điều khiển vệ tinh quỹ đạo Trái Đất |
| **PC2** | 745 | 16 | 36 | 722 | 16 | 37 | Phần mềm điều khiển vệ tinh quỹ đạo Trái Đất |
| **PC3** | 1077 | 134 | 37 | 1053 | 130 | 38 | Phần mềm điều khiển vệ tinh quỹ đạo Trái Đất |
| **PC4** | 1287 | 177 | 37 | 1270 | 176 | 38 | Phần mềm điều khiển vệ tinh quỹ đạo Trái Đất |
| **PC5** | 1711 | 471 | 38 | 1694 | 458 | 39 | Phần mềm điều khiển vệ tinh quỹ đạo Trái Đất |

Sự khác biệt nhỏ về số lượng mô-đun và đặc trưng giữa các phiên bản giải thích tại sao khi chạy thực nghiệm độc lập trên dữ liệu tải về từ các kho lưu trữ công khai như GitHub, kết quả thu được có thể có sai số nhỏ so với các chỉ số được công bố trong bài báo gốc. Để đảm bảo tính chính xác cao nhất cho quy trình kỹ nghệ dữ liệu, đường ống xử lý phải có khả năng tự động thích ứng với cấu trúc cột của từng phiên bản dữ liệu.

## Mô Hình Toán Học Và Các Chỉ Số Đánh Giá Chất Lượng

Quy trình đánh giá các mô hình học máy dự đoán lỗi phần mềm sử dụng ma trận nhầm lẫn nhị phân để phân loại các mô-đun thành hai lớp: Dương tính (đại diện cho mô-đun chứa lỗi) và Âm tính (đại diện cho mô-đun sạch không chứa lỗi). Các chỉ số đánh giá cốt lõi được định nghĩa thông qua các mô hình toán học nghiêm ngặt sau đây:

Độ chính xác tổng thể (Accuracy) xác định tỷ lệ phần trăm các mô-đun được dự đoán đúng trên tổng số mẫu thực nghiệm:

$$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$$

Độ chính xác dự đoán lớp lỗi (Precision) phản ánh tỷ lệ mô-đun thực sự có lỗi trong số các mô-đun được mô hình phân loại là có lỗi:

$$\text{Precision}=\frac{TP}{TP+FP}$$

Độ bao phủ lớp lỗi (Recall hay True Positive Rate - TPR) đo lường khả năng phát hiện đầy đủ các mô-đun lỗi thực tế của mô hình:

$$\text{Recall}=\text{TPR}=\frac{TP}{TP+FN}$$

Điểm F-Score là trung bình điều hòa giữa Precision và Recall, cung cấp một thước đo cân bằng giúp đánh giá hiệu năng tổng thể trên các tập dữ liệu mất cân bằng:

$$\text{F-Score}=\frac{2\times\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$$

Tỷ lệ báo động giả (False Positive Rate - FPR) phản ánh tỷ lệ các mô-đun sạch bị mô hình phân loại nhầm thành mô-đun lỗi:

$$\text{FPR}=\frac{FP}{TN+FP}$$

Độ đặc hiệu (Specificity hay True Negative Rate) thể hiện tỷ lệ mô-đun sạch được dự đoán chính xác trên tổng số mô-đun sạch thực tế:

$$\text{Specificity}=1-\text{FPR}=\frac{TN}{TN+FP}$$

Chỉ số G-measure là trung bình điều hòa của Recall và Specificity, giúp kiểm soát đồng thời khả năng phát hiện lỗi và khả năng giữ lại các mô-đun sạch:

$$\text{G-measure}=\frac{2\times\text{Recall}\times\text{Specificity}}{\text{Recall}+\text{Specificity}}$$

Bên cạnh đó, chỉ số diện tích dưới đường cong ROC (ROC-AUC) biểu diễn mối tương quan động giữa TPR và FPR tại các ngưỡng phân loại khác nhau, cho phép đo lường năng lực phân biệt của thuật toán độc lập với việc lựa chọn ngưỡng quyết định. Để triệt tiêu ảnh hưởng của sự mất cân bằng phân phối lớp, tất cả các chỉ số trên đều được tính toán bằng phương pháp trung bình cộng có trọng số theo quy mô của từng lớp dữ liệu.

## Phương Pháp Cân Bằng Dữ Liệu SMOTE Và Cơ Chế Ngăn Chặn Rò Rỉ Dữ Liệu

Một đặc trưng phổ biến của dữ liệu dự đoán lỗi phần mềm là sự mất cân bằng lớp cực hạn, trong đó số lượng các mô-đun sạch luôn chiếm đa số tuyệt đối so với các mô-đun chứa lỗi. Nếu huấn luyện các mô hình học máy trực tiếp trên các tập dữ liệu này, thuật toán sẽ bị thiên vị nặng nề về phía lớp đa số, dẫn đến độ chính xác tổng thể rất cao nhưng độ bao phủ lớp lỗi thực tế lại xấp xỉ bằng không. Để giải quyết rào cản này, thuật toán lấy mẫu dư tổng hợp cho lớp thiểu số (SMOTE) được áp dụng.

SMOTE hoạt động bằng cách phân tích không gian đặc trưng của lớp thiểu số. Với mỗi mẫu dữ liệu thiểu số $x_i$, thuật toán xác định $k$ láng giềng gần nhất (thông thường $k=5$) cũng thuộc lớp thiểu số. Một láng giềng $x_{zi}$ được chọn ngẫu nhiên, và một mẫu dữ liệu nhân tạo mới $x_{new}$ được sinh ra dọc theo đoạn thẳng nối hai mẫu này theo công thức toán học:

$$x_{new}=x_i+\lambda(x_{zi}-x_i)$$

Trong đó $\lambda$ là một biến số ngẫu nhiên tuân theo phân phối đều trong khoảng .

Trong kỹ nghệ dữ liệu, sai lầm phổ biến nhất khi áp dụng SMOTE là thực hiện lấy mẫu dư trên toàn bộ tập dữ liệu gốc trước khi chia tách dữ liệu cho quá trình kiểm định chéo (Cross-Validation). Hành vi này gây ra hiện tượng rò rỉ dữ liệu (Data Leakage) nghiêm trọng do thông tin của tập kiểm thử đã gián tiếp tham gia vào quá trình sinh mẫu tổng hợp của tập huấn luyện. Khi đó, các ước lượng hiệu năng thu được sẽ cực kỳ lạc quan nhưng mô hình sẽ thất bại hoàn toàn khi đối mặt với dữ liệu thực tế trong môi trường sản xuất.

Để ngăn chặn triệt để hiện tượng rò rỉ dữ liệu, quy trình thực nghiệm phải tuân thủ nghiêm ngặt nguyên tắc đóng gói. SMOTE chỉ được phép áp dụng trên tập huấn luyện của từng fold cục bộ sau khi đã tách biệt hoàn toàn tập kiểm thử. Tập kiểm thử của mỗi fold phải được bảo tồn nguyên bản để đóng vai trò như một tập dữ liệu hoàn toàn chưa từng được nhìn thấy. Quy trình chuẩn hóa đặc trưng (StandardScaler) cũng phải tuân thủ nguyên tắc tương tự: tính toán trung bình và phương sai trên tập huấn luyện của fold đó, sau đó dùng các tham số này để chuẩn hóa cho cả tập huấn luyện và tập kiểm thử tương ứng.

## Thiết Kế Và Cấu Hình Đường Ống Thực Nghiệm Bằng Mã Nguồn Python

Dưới đây là thiết kế chi tiết của đường ống xử lý dữ liệu tự động, sử dụng thư viện `scikit-learn` kết hợp với thư viện chuyên dụng cho dữ liệu mất cân bằng `imbalanced-learn`. Sự tích hợp của lớp `Pipeline` từ thư viện `imblearn` là chìa khóa kỹ thuật quyết định để tự động hóa việc áp dụng chuẩn hóa và lấy mẫu dư một cách an sau bên trong vòng lặp kiểm định chéo 10-fold.

```python
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

# Khóa các cảnh báo không ảnh hưởng để tối ưu hóa đầu ra của luồng xử lý
warnings.filterwarnings('ignore')

class SoftwareDefectPredictionEngine:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_learners = self._init_base_learners()
        self.classifiers = self._build_classifiers_suite()

    def _init_base_learners(self):
        """Khởi tạo các bộ phân loại cơ sở theo cấu hình mô tả trong bài báo gốc."""
        return {
            'RF': RandomForestClassifier(random_state=self.random_state),
            'DS': DecisionTreeClassifier(random_state=self.random_state),
            'SVM': SVC(kernel='linear', probability=True, random_state=self.random_state),
            'LR': LogisticRegression(max_iter=1500, random_state=self.random_state)
        }

    def _build_classifiers_suite(self):
        """Xây dựng 12 cấu hình mô hình từ sự kết hợp của bộ phân loại cơ sở và kỹ thuật tích hợp."""
        suite = {}
        # 1. Thêm các bộ phân loại cơ sở độc lập
        for name, clf in self.base_learners.items():
            suite[name] = clf
            
        # 2. Thêm các biến thể tích hợp tuần tự bằng AdaBoost
        for name, clf in self.base_learners.items():
            suite[name + '_Ada'] = AdaBoostClassifier(
                estimator=clf, 
                random_state=self.random_state
            )
            
        # 3. Thêm các biến thể tích hợp song song bằng Bagging
        for name, clf in self.base_learners.items():
            suite = BaggingClassifier(
                estimator=clf, 
                random_state=self.random_state
            )
            
        return suite

    def run_experiment(self, df, target_col='Defective'):
        """
        Thực thi quy trình thực nghiệm khép kín: 10-Fold CV -> Standard Scale -> SMOTE -> Train -> Predict.
        Đảm bảo tuyệt đối không xảy ra rò rỉ dữ liệu giữa tập huấn luyện và kiểm thử.
        """
        X = df.drop(columns=[target_col]).values
        # Chuẩn hóa nhãn mục tiêu về định dạng nhị phân nguyên (1 cho lỗi, 0 cho sạch)
        y = df[target_col].map({'Y': 1, 'N': 0, 1: 1, 0: 0, True: 1, False: 0}).values
        
        kf = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        # Khởi tạo bộ lưu trữ kết quả cho từng cấu hình mô hình
        results_accumulator = {
            name: {metric: for metric in}
            for name in self.classifiers.keys()
        }

        # Vòng lặp phân tách fold đảm bảo tính cô lập của dữ liệu kiểm thử
        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, model in self.classifiers.items():
                # Tạo đường ống động bằng imblearn để đóng gói xử lý
                pipeline = ImbPipeline()

                # Huấn luyện đường ống: StandardScaler tính tham số và chuẩn hóa X_train,
                # sau đó SMOTE sinh mẫu dư trên X_train đã chuẩn hóa, cuối cùng mô hình được huấn luyện.
                pipeline.fit(X_train, y_train)
                
                # Dự đoán trên tập kiểm thử: chỉ đi qua bước chuẩn hóa bằng tham số của tập huấn luyện.
                predictions = pipeline.predict(X_test)
                
                if hasattr(model, "predict_proba"):
                    probabilities = pipeline.predict_proba(X_test)[:, 1]
                else:
                    probabilities = pipeline.decision_function(X_test)

                # Tính toán các chỉ số hiệu năng có trọng số
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
                results_accumulator[model_name].append(rec)
                results_accumulator[model_name].append(f1)
                if not np.isnan(auc):
                    results_accumulator[model_name].append(auc)

        # Tổng hợp kết quả trung bình của 10 folds
        final_report = {}
        for model_name, metrics in results_accumulator.items():
            final_report[model_name] = {
                'Avg_Accuracy': np.round(np.mean(metrics['Accuracy']), 2),
                'Avg_Precision': np.round(np.mean(metrics['Precision']), 2),
                'Avg_Recall': np.round(np.mean(metrics), 2),
                'Avg_F-Score': np.round(np.mean(metrics), 2),
                'Avg_ROC-AUC': np.round(np.mean(metrics), 2) if metrics else np.nan
            }

        return pd.DataFrame(final_report).T

