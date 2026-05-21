import pandas as pd

# Load previous all models results
df_all = pd.read_csv('exploration/all_models_evaluation_results.csv')
# Load DNN results
df_dnn = pd.read_csv('notebooks/report_DNN.csv')

# Compute averages across all datasets
avg_all = df_all.groupby('Model')[['Accuracy', 'F-Score', 'ROC-AUC']].mean().reset_index()
avg_dnn = df_dnn.groupby('Model')[['Accuracy', 'F-Score', 'ROC-AUC']].mean().reset_index()

# Combine them
combined = pd.concat([avg_all, avg_dnn], ignore_index=True)

# Sort by F-Score
combined = combined.sort_values(by='F-Score', ascending=False)
print("AVERAGE PERFORMANCE (Top to Bottom):")
print(combined.to_string(index=False))

# We can also find the best DNN and compare to best ML
best_dnn = avg_dnn.sort_values(by='F-Score', ascending=False).iloc[0]
best_ml = avg_all.sort_values(by='F-Score', ascending=False).iloc[0]

print("\n--- BEST DNN vs BEST ML ---")
print(f"Best ML : {best_ml['Model']} (F-Score: {best_ml['F-Score']:.3f}, AUC: {best_ml['ROC-AUC']:.3f})")
print(f"Best DNN: {best_dnn['Model']} (F-Score: {best_dnn['F-Score']:.3f}, AUC: {best_dnn['ROC-AUC']:.3f})")
