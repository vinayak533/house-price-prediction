import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

os.makedirs('portfolio-images', exist_ok=True)
df = pd.read_csv('housing.csv')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 1. EDA Distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
sns.histplot(df['median_house_value'], bins=40, kde=True, ax=axes[0], color='#2563EB')
axes[0].set_title('Target Distribution: Median House Value', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Median House Value ($USD)')
axes[0].set_ylabel('Density Frequency')
median_val = df['median_house_value'].median()
axes[0].axvline(median_val, color='#DC2626', linestyle='--', label=f'Median: ${median_val:,.0f}')
axes[0].legend()

sns.scatterplot(x='median_income', y='median_house_value', data=df.sample(2000, random_state=42), alpha=0.3, ax=axes[1], color='#0D9488')
axes[1].set_title('Income vs Median House Value (Sample n=2000)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Median Income (Tens of Thousands $USD)')
axes[1].set_ylabel('Median House Value ($USD)')
plt.tight_layout()
plt.savefig('portfolio-images/eda-distributions.png')
plt.close()

# 2. Correlation Matrix Heatmap
plt.figure(figsize=(8, 6), dpi=200)
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', cbar=True, square=True, linewidths=0.5)
plt.title('Covariate Correlation Heatmap (Pearson r)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('portfolio-images/correlation-matrix.png')
plt.close()

# 3. Model Benchmark Comparison
models = ['Linear Regression', 'Lasso', 'Ridge', 'Random Forest', 'Tuned HistGB']
cv_rmse = [68604.16, 68603.23, 68595.62, 49362.12, 47408.38]
colors = ['#94A3B8', '#94A3B8', '#94A3B8', '#38BDF8', '#10B981']

plt.figure(figsize=(8, 4.5), dpi=200)
bars = plt.barh(models, cv_rmse, color=colors, height=0.55)
plt.xlabel('5-Fold Cross-Validation RMSE ($USD) - Lower is Better', fontweight='bold')
plt.title('Model Benchmark Comparison (5-Fold CV)', fontsize=12, fontweight='bold', pad=12)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 1000, bar.get_y() + bar.get_height()/2, f'${w:,.0f}', va='center', ha='left', fontsize=9, fontweight='bold')
plt.xlim(0, 85000)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('portfolio-images/model-benchmark.png')
plt.close()

# 4. Residuals Analysis
model = joblib.load('model.pkl')
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preds = model.predict(X_test)
res = y_test - preds

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
axes[0].scatter(preds, res, alpha=0.2, s=8, color='#6366F1')
axes[0].axhline(0, color='#DC2626', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuals vs Fitted Values', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted House Value ($USD)')
axes[0].set_ylabel('Residual (Actual - Predicted)')

sns.histplot(res, bins=40, kde=True, ax=axes[1], color='#8B5CF6')
axes[1].axvline(0, color='#DC2626', linestyle='--', linewidth=1.5)
axes[1].set_title('Residual Error Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Residual ($USD)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('portfolio-images/residuals-analysis.png')
plt.close()

print('Visual assets successfully generated!')

# 5. Parity Plot: Actual vs Predicted
plt.figure(figsize=(6, 5), dpi=200)
plt.scatter(y_test, preds, alpha=0.25, s=9, color='#0284C7', edgecolors='none')
lims = [0, 520000]
plt.plot(lims, lims, color='#DC2626', linestyle='--', linewidth=1.5, label='Ideal 1:1 Parity')
plt.xlabel('Ground Truth Median House Value ($USD)', fontweight='bold')
plt.ylabel('HistGB Predicted House Value ($USD)', fontweight='bold')
plt.title('Actual vs Predicted Parity Plot (Test Set n=4,128)', fontsize=11, fontweight='bold', pad=10)
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.tight_layout()
plt.savefig('portfolio-images/prediction-parity.png')
plt.close()
print('Prediction parity plot generated!')
