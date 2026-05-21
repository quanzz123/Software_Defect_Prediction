from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class SoftwareDefectPredictionEngine:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_learners = self._init_base_learners()
        self.classifiers = self._build_classifiers_suite()

    def _init_base_learners(self):
        """Khởi tạo các bộ phân loại cơ sở theo cấu hình mô tả trong bài báo gốc."""
        return {
            'RF': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'DS': DecisionTreeClassifier(random_state=self.random_state),
            'SVM': SVC(kernel='linear', probability=True, random_state=self.random_state, max_iter=2000),
            'LR': LogisticRegression(max_iter=1500, random_state=self.random_state, n_jobs=-1)
        }

    def _build_classifiers_suite(self):
        """Xây dựng 12 cấu hình mô hình từ sự kết hợp của bộ phân loại cơ sở và kỹ thuật tích hợp."""
        suite = {}
        # 1. Thêm các bộ phân loại cơ sở độc lập
        for name, clf in self.base_learners.items():
            suite[name] = clf
            
        # 2. Thêm các biến thể tích hợp tuần tự bằng AdaBoost
        for name, clf in self.base_learners.items():
            suite[f'AdaBoost_{name}'] = AdaBoostClassifier(
                estimator=clf, 
                random_state=self.random_state
            )
            
        # 3. Thêm các biến thể tích hợp song song bằng Bagging
        for name, clf in self.base_learners.items():
            suite[f'Bagging_{name}'] = BaggingClassifier(
                estimator=clf, 
                random_state=self.random_state,
                n_jobs=-1
            )
            
        return suite

    def get_param_grid(self, model_name):
        """Trả về không gian tham số cần tối ưu cho mô hình trong Pipeline (tiền tố model__)."""
        grid = {}
        
        if model_name == 'RF':
            grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30]
            }
        elif model_name == 'DS':
            grid = {
                'model__max_depth': [None, 10, 20, 30, 50],
                'model__min_samples_split': [2, 5, 10]
            }
        elif model_name == 'SVM':
            grid = {
                'model__C': [0.1, 1.0, 10.0]
            }
        elif model_name == 'LR':
            grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__solver': ['lbfgs', 'liblinear']
            }
        elif model_name.startswith('AdaBoost'):
            grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 1.0]
            }
        elif model_name.startswith('Bagging'):
            grid = {
                'model__n_estimators': [10, 50, 100],
                'model__max_samples': [0.5, 0.8, 1.0]
            }
            
        return grid
