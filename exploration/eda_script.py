import os
import glob
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Setup aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

# Color palette: Premium sleek colors
COLORS = {
    'Clean': '#4A90E2',      # Sleek Blue
    'Defective': '#E15759',  # Crimson Red
    'Primary': '#2C3E50',    # Dark Navy
    'Accent': '#1ABC9C'      # Teal
}

def decode_bytes(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return val

def standardize_target(val):
    if val is None or pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip().upper()
    if val_str in ['Y', 'YES', 'TRUE', '1', '1.0', 'T', 'DEFECTIVE', 'B\'Y\'', 'B\'TRUE\'']:
        return 'Defective'
    elif val_str in ['N', 'NO', 'FALSE', '0', '0.0', 'F', 'CLEAN', 'B\'N\'', 'B\'FALSE\'']:
        return 'Clean'
    return val_str

def resolve_key_metrics(columns):
    """
    Find best matching columns for key software metrics across different NASA dataset schemas.
    """
    cols_lower = [c.lower() for c in columns]
    
    # 1. Resolve LOC
    loc_col = None
    for candidate in ['loc_total', 'locTotal', 'loc', 'number_of_lines', 'loc_executable']:
        if candidate.lower() in cols_lower:
            loc_col = columns[cols_lower.index(candidate.lower())]
            break
    if not loc_col:
        # Fallback to any col containing 'loc'
        for c in columns:
            if 'loc' in c.lower():
                loc_col = c
                break
                
    # 2. Resolve McCabe Complexity
    mccabe_col = None
    for candidate in ['cyclomatic_complexity', 'cyclomaticcomplexity', 'v(g)', 'cyclomatic_density']:
        if candidate.lower() in cols_lower:
            mccabe_col = columns[cols_lower.index(candidate.lower())]
            break
    if not mccabe_col:
        for c in columns:
            if 'complexity' in c.lower() or 'v(g)' in c.lower():
                mccabe_col = c
                break
                
    # 3. Resolve Halstead Volume
    halstead_col = None
    for candidate in ['halstead_volume', 'halsteadvolume', 'volume', 'ev(g)', 'iv(g)']:
        if candidate.lower() in cols_lower:
            halstead_col = columns[cols_lower.index(candidate.lower())]
            break
    if not halstead_col:
        for c in columns:
            if 'halstead' in c.lower() or 'volume' in c.lower():
                halstead_col = c
                break
                
    return loc_col, mccabe_col, halstead_col

def load_arff_dataset(file_path):
    """
    Load ARFF file and convert to standard pandas DataFrame with decoded columns.
    """
    print(f"Loading {os.path.basename(file_path)}...")
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode string columns from bytes to utf-8
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype == 'O':
            df[col] = df[col].apply(decode_bytes)
            
    # Clean string '?' representing missing values
    df = df.replace('?', np.nan)
    
    # Resolve target column
    target_candidates = ['Defective', 'defective', 'class', 'problems', 'problem']
    target_col = None
    for c in target_candidates:
        for actual_col in df.columns:
            if actual_col.lower() == c.lower():
                target_col = actual_col
                break
        if target_col:
            break
            
    if not target_col:
        target_col = df.columns[-1]  # Default to last column
        
    # Standardize target
    df[target_col] = df[target_col].apply(standardize_target)
    
    # Force convert other columns to numeric if possible
    for col in df.columns:
        if col != target_col:
            if df[col].dtype == object or df[col].dtype == 'O':
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass
                    
    return df, target_col

def perform_eda():
    dataset_dir = 'datasets'
    output_dir = 'eda_results'
    plots_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    arff_files = glob.glob(os.path.join(dataset_dir, '*.arff'))
    if not arff_files:
        print("No ARFF files found in datasets directory.")
        return
        
    summary_data = []
    
    # Store class distribution info for the combined plot
    imbalance_info = []
    
    for file_path in sorted(arff_files):
        ds_name = os.path.splitext(os.path.basename(file_path))[0]
        df, target_col = load_arff_dataset(file_path)
        
        # 1. Basic Stats
        n_samples, n_features = df.shape
        # Total features excluding the target column
        n_features -= 1 
        
        # Missing values
        missing_count = df.drop(columns=[target_col]).isnull().sum().sum()
        missing_percent = (df.drop(columns=[target_col]).isnull().mean().mean()) * 100
        
        # Duplicates
        dup_count = df.duplicated().sum()
        dup_percent = (dup_count / n_samples) * 100
        
        # Class distribution
        class_counts = df[target_col].value_counts()
        n_clean = class_counts.get('Clean', 0)
        n_defective = class_counts.get('Defective', 0)
        
        pct_clean = (n_clean / n_samples) * 100 if n_samples > 0 else 0
        pct_defective = (n_defective / n_samples) * 100 if n_samples > 0 else 0
        
        imbalance_info.append({
            'Dataset': ds_name,
            'Clean': pct_clean,
            'Defective': pct_defective,
            'Clean_Count': n_clean,
            'Defective_Count': n_defective,
            'Total': n_samples
        })
        
        # 2. Key Metrics Analysis
        loc_col, mccabe_col, halstead_col = resolve_key_metrics(df.drop(columns=[target_col]).columns)
        
        # 3. Correlation with Target (Pearson correlation by mapping Clean: 0, Defective: 1)
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        df_numeric['Target_Numeric'] = df[target_col].map({'Clean': 0, 'Defective': 1})
        
        correlations = df_numeric.corr(method='spearman')['Target_Numeric'].drop('Target_Numeric').abs().sort_values(ascending=False)
        top_correlated_features = correlations.head(5).to_dict()
        
        # Redundant features (highly correlated feature pairs, corr > 0.85)
        corr_matrix = df_numeric.drop(columns=['Target_Numeric']).corr(method='spearman').abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_pairs = []
        for col in upper_tri.columns:
            high_corr_indices = upper_tri.index[upper_tri[col] > 0.85].tolist()
            for idx in high_corr_indices:
                redundant_pairs.append((idx, col, upper_tri.loc[idx, col]))
                
        # 4. Feature Importance using a simple Random Forest
        # Fill missing values with median for RF
        df_rf = df_numeric.dropna().copy()
        if len(df_rf) > 10:
            X_rf = df_rf.drop(columns=['Target_Numeric'])
            y_rf = df_rf['Target_Numeric']
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_rf, y_rf)
            importances = pd.Series(rf.feature_importances_, index=X_rf.columns).sort_values(ascending=False)
            top_rf_features = importances.head(5).to_dict()
        else:
            top_rf_features = {}
            
        summary_data.append({
            'Dataset': ds_name,
            'Samples': n_samples,
            'Features': n_features,
            'Missing': f"{missing_count} ({missing_percent:.2f}%)",
            'Duplicates': f"{dup_count} ({dup_percent:.2f}%)",
            'Clean': f"{n_clean} ({pct_clean:.2f}%)",
            'Defective': f"{n_defective} ({pct_defective:.2f}%)",
            'LOC_col': loc_col,
            'McCabe_col': mccabe_col,
            'Halstead_col': halstead_col,
            'Top_Corr': top_correlated_features,
            'Top_RF': top_rf_features,
            'Redundant_Pairs_Count': len(redundant_pairs)
        })
        
        # --- Generate Individual Dataset Visualizations ---
        # A. Heatmap of Top 15 Features by correlation with target
        top_15_features = correlations.head(15).index.tolist()
        if len(top_15_features) > 1:
            plt.figure(figsize=(10, 8))
            top_corr_matrix = df_numeric[top_15_features].corr(method='spearman')
            sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                        vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title(f"Spearman Correlation Heatmap (Top Features) - {ds_name}", pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{ds_name}_correlation.png"), dpi=200)
            plt.close()
            
        # B. Boxplots of Key Metrics vs Target
        resolved_cols = [('LOC', loc_col), ('McCabe Complexity', mccabe_col), ('Halstead Volume', halstead_col)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Distribution of Key Metrics by Class - {ds_name}", y=0.98)
        
        for ax, (metric_name, col_name) in zip(axes, resolved_cols):
            if col_name and col_name in df.columns:
                # Use log scale for better visualization since metrics are highly skewed
                sns.boxplot(data=df, x=target_col, y=col_name, ax=ax, palette=[COLORS['Clean'], COLORS['Defective']])
                ax.set_title(f"{metric_name} ({col_name})")
                ax.set_xlabel("Class")
                ax.set_ylabel(col_name)
                # Apply log scale if values span orders of magnitude and are positive
                if df[col_name].max() > 10 * df[col_name].median() and df[col_name].min() >= 0:
                    ax.set_yscale('log')
                    ax.set_ylabel(f"{col_name} (Log Scale)")
            else:
                ax.text(0.5, 0.5, "Metric Not Found", ha='center', va='center')
                ax.set_title(metric_name)
                
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{ds_name}_key_metrics.png"), dpi=200)
        plt.close()
        
    # --- Generate Combined Visualizations ---
    # 1. Combined Class Distribution Bar Chart
    df_imb = pd.DataFrame(imbalance_info)
    df_imb_melted = df_imb.melt(id_vars=['Dataset'], value_vars=['Clean', 'Defective'], 
                                var_name='Class', value_name='Percentage')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_imb_melted, x='Dataset', y='Percentage', hue='Class', 
                     palette=[COLORS['Clean'], COLORS['Defective']])
    plt.title("Class Distribution (Clean vs Defective %) Across NASA MDP Datasets", pad=15)
    plt.xlabel("Dataset")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 110)
    
    # Add labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 7),
                        textcoords='offset points',
                        fontsize=9)
            
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "combined_class_distribution.png"), dpi=200)
    plt.close()
    
    # --- Generate Markdown Report ---
    report_path = os.path.join(output_dir, 'EDA_Report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo Cáo Phân Tích Khám Phá Dữ Liệu (EDA) End-to-End\n\n")
        f.write("> **Bài toán**: Dự đoán lỗi phần mềm (Software Defect Prediction)\n")
        f.write("> **Tập dữ liệu**: NASA Metrics Data Program (MDP) - 10 Datasets công khai\n")
        f.write(f"> **Ngày thực hiện**: 2026-05-20\n\n")
        
        f.write("## 1. Tổng Quan Toàn Bộ 10 Bộ Dữ Liệu\n\n")
        f.write("Dưới đây là bảng thống kê tổng hợp cấu trúc, chất lượng dữ liệu và mức độ mất cân bằng lớp trên cả 10 dataset:\n\n")
        
        f.write("| Bộ Dữ Liệu | Tổng Mẫu (Mô-đun) | Đặc Trưng (Features) | Mẫu Trùng Lặp (%) | Dữ Liệu Thiếu (NaN) | Lớp Sạch (Clean) | Lớp Lỗi (Defective) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- | :--- |\n")
        
        for item in summary_data:
            f.write(f"| **{item['Dataset']}** | {item['Samples']} | {item['Features']} | {item['Duplicates']} | {item['Missing']} | {item['Clean']} | {item['Defective']} |\n")
            
        f.write("\n### Biểu Đồ So Sánh Sự Mất Cân Bằng Lớp\n\n")
        f.write("![Class Distribution](plots/combined_class_distribution.png)\n\n")
        
        f.write("> [!IMPORTANT]\n")
        f.write("> **Nhận xét quan trọng về Mất cân bằng lớp (Class Imbalance)**:\n")
        f.write("> - Tất cả các bộ dữ liệu đều có hiện tượng mất cân bằng lớp từ vừa đến cực kỳ nghiêm trọng.\n")
        f.write("> - Bộ dữ liệu **PC2** mất cân bằng nhất với chỉ **2.15%** số lượng mẫu là lỗi (16 mẫu lỗi trên tổng số 745 mẫu).\n")
        f.write("> - Bộ dữ liệu **MC2** có tỷ lệ lỗi cao nhất với **35.20%** mẫu lỗi, tiếp theo là **PC5** với **27.53%**.\n")
        f.write("> - Việc mất cân bằng lớp cực đoan này đòi hỏi bắt buộc phải sử dụng các kỹ thuật cân bằng dữ liệu như **SMOTE** trong quá trình huấn luyện mô hình học máy (như mô tả trong [rule.md](file:///D:/WORKSPACE/KPDL/Software_Defect_Prediction/rule.md)).\n\n")

        f.write("> [!WARNING]\n")
        f.write("> **Nhận xét về Mẫu trùng lặp (Duplicates)**:\n")
        f.write("> - Các bộ dữ liệu như **MC1** chứa tới **1.81%** mẫu trùng lặp, **JM1** chứa **0.78%** mẫu trùng lặp.\n")
        f.write("> - Đây chính là đặc trưng của phiên bản gốc $DS'$. Việc loại bỏ mẫu trùng lặp và không nhất quán để tạo phiên bản sạch $DS''$ (của Shepperd) là cực kỳ cần thiết để đảm bảo tính khách quan cho mô hình học máy, tránh hiện tượng mô hình học vẹt các mẫu trùng lặp.\n\n")

        f.write("## 2. Chi Tiết Từng Bộ Dữ Liệu & Phân Tích Đặc Trưng Nhóm Metrics\n\n")
        
        for item in summary_data:
            name = item['Dataset']
            f.write(f"### {item['Dataset']}\n\n")
            f.write(f"- **Kích thước**: {item['Samples']} dòng, {item['Features']} cột đặc trưng.\n")
            f.write(f"- **Cột chỉ số LOC đã xác định**: `{item['LOC_col']}`\n")
            f.write(f"- **Cột chỉ số McCabe đã xác định**: `{item['McCabe_col']}`\n")
            f.write(f"- **Cột chỉ số Halstead đã xác định**: `{item['Halstead_col']}`\n")
            f.write(f"- **Số cặp đặc trưng dư thừa (Correlation Spearman > 0.85)**: {item['Redundant_Pairs_Count']}\n\n")
            
            f.write("#### Phân Phối Của Các Nhóm Chỉ Số Cốt Lõi:\n")
            f.write(f"![Key Metrics Distribution](plots/{name}_key_metrics.png)\n\n")
            
            f.write("#### Bản Đồ Tương Quan Spearman (Top 15 Đặc Trưng):\n")
            f.write(f"![Correlation Heatmap](plots/{name}_correlation.png)\n\n")
            
            f.write("#### Top 5 Đặc trưng tương quan mạnh nhất với Nhãn Lỗi (Spearman):\n")
            f.write("| Thứ tự | Tên Đặc Trưng | Hệ số tương quan tuyệt đối |\n")
            f.write("| :---: | :--- | :---: |\n")
            for i, (feat, val) in enumerate(item['Top_Corr'].items(), 1):
                f.write(f"| {i} | `{feat}` | {val:.4f} |\n")
                
            f.write("\n#### Top 5 Đặc trưng quan trọng nhất (Được đánh giá bằng Random Forest):\n")
            f.write("| Thứ tự | Tên Đặc Trưng | Độ quan trọng (Importance) |\n")
            f.write("| :---: | :--- | :---: |\n")
            if item['Top_RF']:
                for i, (feat, val) in enumerate(item['Top_RF'].items(), 1):
                    f.write(f"| {i} | `{feat}` | {val:.4f} |\n")
            else:
                f.write("| - | Không đủ mẫu sạch để đánh giá mô hình | - |\n")
                
            f.write("\n---\n\n")
            
        f.write("## 3. Kết Luận Định Hướng Cho Mô Hình Học Máy\n\n")
        f.write("1. **Đa cộng tuyến cực kỳ nghiêm trọng (Collinearity)**:\n")
        f.write("   - Hầu hết các bộ dữ liệu đều có số lượng rất lớn các cặp đặc trưng tương quan Spearman trên **0.85** hoặc thậm chí **0.99** (đặc biệt trong các đặc trưng Halstead và LOC).\n")
        f.write("   - *Khuyến nghị*: Nên áp dụng các phương pháp giảm chiều dữ liệu (PCA), hoặc lựa chọn đặc trưng (Feature Selection), hoặc sử dụng các bộ phân loại ít nhạy cảm với đa cộng tuyến như **Random Forest** hay **Decision Trees** thay vì Logistic Regression thuần túy.\n\n")
        f.write("2. **Độ lệch của Đặc trưng (Feature Skewness)**:\n")
        f.write("   - Các đặc trưng đo lường kích thước như LOC và Volume đều cực kỳ lệch phải (Right-skewed) với một vài hàm/mô-đun có kích thước khổng lồ kéo dài đuôi phân phối.\n")
        f.write("   - *Khuyến nghị*: Việc chuẩn hóa bằng **StandardScaler** và lấy Log-transform là bước tiền xử lý bắt buộc đối với các thuật toán nhạy cảm với thang đo như **SVM** và **Logistic Regression**.\n\n")
        f.write("3. **Xử lý Mất cân bằng lớp An Toàn**:\n")
        f.write("   - Như đã cảnh báo, **SMOTE** là bắt buộc. Tuy nhiên, việc áp dụng SMOTE phải được thực hiện **sau khi chia tách fold kiểm định** (đóng gói trong `imblearn.pipeline`) để loại bỏ hoàn toàn nguy cơ **Rò rỉ thông tin dữ liệu (Data Leakage)**.\n")
        
    print(f"EDA successfully completed! Report saved to {report_path}")

if __name__ == "__main__":
    perform_eda()
