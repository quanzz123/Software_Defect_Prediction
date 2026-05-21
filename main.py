import os
import glob
import pandas as pd
from src.data_loader import load_arff_dataset
from src.evaluation import run_experiment_on_dataset

def main():
    dataset_dir = 'datasets'
    arff_files = glob.glob(os.path.join(dataset_dir, '*.arff'))
    
    if not arff_files:
        print(f"No ARFF files found in {dataset_dir} directory.")
        return

    all_results = []
    
    for file_path in sorted(arff_files):
        ds_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Evaluating 12 models on {ds_name}...")
        df, target_col = load_arff_dataset(file_path)
        report = run_experiment_on_dataset(df, target_col)
        
        for model_name, metrics in report.items():
            row = {'Dataset': ds_name, 'Model': model_name}
            row.update(metrics)
            all_results.append(row)
            
    df_results = pd.DataFrame(all_results)
    df_results = df_results[['Dataset', 'Model', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'ROC-AUC']]
    
    output_file = 'all_models_evaluation_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults successfully saved to {output_file}")

if __name__ == '__main__':
    main()
