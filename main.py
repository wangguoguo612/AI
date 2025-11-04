import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Set global font to avoid Chinese display issues (now unused as all text is English)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})  # Unified font size and resolution


# 1. Data Generation & Loading (Simulate IC Manufacturing Parameters and Yield)
def generate_chip_data(n_samples=1000):
    """Generate simulated IC manufacturing parameters and yield data"""
    data = {
        'diffusion_temp': np.random.normal(900, 50, n_samples),  # Diffusion Temp (°C)
        'deposition_time': np.random.normal(300, 30, n_samples),  # Deposition Time (sec)
        'implant_dose': np.random.normal(5, 1.5, n_samples),  # Ion Implant Dose (1e15 ions/cm²)
        'litho_energy': np.random.normal(50, 5, n_samples),  # Lithography Energy (mJ/cm²)
        'etch_depth': np.random.normal(500, 50, n_samples),  # Etch Depth (nm)
        'clean_time': np.random.normal(60, 10, n_samples),  # Cleaning Time (sec)
        'film_thickness': np.random.normal(200, 20, n_samples)  # Film Thickness (nm)
    }

    df = pd.DataFrame(data)

    # Simulate non-linear relationship between parameters and yield
    df['yield'] = 0.5 + 0.001 * df['diffusion_temp'] - 0.0005 * df['deposition_time'] + \
                  0.03 * df['implant_dose'] + 0.002 * df['litho_energy'] - 0.0003 * df['etch_depth'] + \
                  0.001 * df['clean_time'] + 0.0005 * df['film_thickness'] + \
                  0.000001 * df['diffusion_temp'] * df['litho_energy'] - \
                  0.00001 * df['etch_depth'] * df['film_thickness'] + \
                  np.random.normal(0, 0.05, n_samples)

    df['yield'] = df['yield'].clip(0, 1)  # Ensure yield is within 0-1 range
    return df


# Generate dataset
chip_data = generate_chip_data(1500)
print("First 5 rows of dataset:")
print(chip_data.head())
print(f"\nDataset shape: {chip_data.shape}")


# 2. Parameter Distribution Histograms (Verify data rationality)
plt.figure(figsize=(15, 10))
for i, col in enumerate(chip_data.columns[:-1]):  # Exclude yield column
    plt.subplot(3, 3, i + 1)
    sns.histplot(chip_data[col], kde=True, bins=20, color='#1f77b4')
    plt.xlabel(col, fontsize=9)
    plt.ylabel('Frequency', fontsize=9)
    plt.title(f'{col} Distribution', fontsize=10, pad=8)  # pad controls title spacing
plt.tight_layout(pad=2.0)  # Add space between subplots
plt.savefig('parameter_distribution.png', bbox_inches='tight')
plt.close()
print("\n✅ Generated parameter distribution plot: parameter_distribution.png")


# 3. Correlation Matrix & Parameter-Yield Scatter Plots
# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation = chip_data.corr()
sns.heatmap(correlation, annot=True, cmap='RdYlBu_r', fmt='.2f', linewidths=0.5, annot_kws={'fontsize': 9})
plt.title('Correlation Matrix: IC Params vs Yield', fontsize=14, pad=12)
plt.tight_layout()
plt.savefig('correlation_matrix.png', bbox_inches='tight')
plt.close()

# Parameter-Yield Scatter Plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(chip_data.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    plt.scatter(chip_data[col], chip_data['yield'], alpha=0.5, color='#ff7f0e', s=20)  # Smaller points
    plt.xlabel(col, fontsize=9)
    plt.ylabel('Yield', fontsize=9)
    plt.title(f'{col} vs Yield', fontsize=10, pad=8)
plt.tight_layout(pad=2.0)
plt.savefig('parameter_vs_yield.png', bbox_inches='tight')
plt.close()
print("✅ Generated correlation matrix: correlation_matrix.png")
print("✅ Generated param-yield scatter plots: parameter_vs_yield.png")


# 4. Data Preprocessing
X = chip_data.drop('yield', axis=1)
y = chip_data['yield']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 5. Model Training - Random Forest Regression
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"R-squared (R²): {r2:.6f}")


# 7. Prediction Error Distribution (Analyze bias)
errors = y_test - y_pred  # Calculate prediction errors
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30, color='#2ca02c')
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error', linewidth=2)
plt.xlabel('Prediction Error (Actual Yield - Predicted Yield)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Model Prediction Errors', fontsize=13, pad=10)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('error_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Generated error distribution plot: error_distribution.png")


# 8. Actual vs Predicted Yield Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='#d62728', s=30)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Ideal Prediction Line', linewidth=2)
plt.xlabel('Actual Yield', fontsize=11)
plt.ylabel('Predicted Yield', fontsize=11)
plt.title('Actual Yield vs Predicted Yield', fontsize=13, pad=10)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', bbox_inches='tight')
plt.close()
print("✅ Generated actual-predicted yield plot: prediction_vs_actual.png")


# 9. Cross-Validation & Result Plot
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"\nCross-Validation R² Scores: {cv_scores.round(4)}")
print(f"Mean CV R² Score: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

# Cross-Validation Result Plot
plt.figure(figsize=(10, 6))
cv_folds = [f'Fold {i + 1}' for i in range(len(cv_scores))]
bars = plt.bar(cv_folds, cv_scores, color='#9467bd', alpha=0.8, width=0.6)
# Add value labels on top of bars
for bar, score in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{score:.4f}', ha='center', va='bottom', fontsize=9)
plt.axhline(y=np.mean(cv_scores), color='green', linestyle='--',
            label=f'Mean R²: {np.mean(cv_scores):.4f}', linewidth=2)
plt.ylim(0.8, 1.0)  # Focus on score range for clarity
plt.xlabel('Cross-Validation Folds', fontsize=11)
plt.ylabel('R² Score', fontsize=11)
plt.title('5-Fold Cross-Validation R² Scores', fontsize=13, pad=10)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('cv_results.png', bbox_inches='tight')
plt.close()
print("✅ Generated cross-validation plot: cv_results.png")


# 10. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, color='#8c564b', alpha=0.8)
plt.xlabel('Importance Score', fontsize=11)
plt.ylabel('Manufacturing Parameters', fontsize=11)
plt.title('Feature Importance: IC Manufacturing Parameters', fontsize=13, pad=10)
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close()
print("✅ Generated feature importance plot: feature_importance.png")
print("\nFeature Importance Ranking:")
print(feature_importance)

# 11. Model Saving & Inference Example
joblib.dump(model, 'chip_yield_model.pkl')
joblib.dump(scaler, 'chip_scaler.pkl')

# Load model for inference
loaded_model = joblib.load('chip_yield_model.pkl')
loaded_scaler = joblib.load('chip_scaler.pkl')

# New IC parameter samples
new_chip_parameters = pd.DataFrame([
    [920, 290, 5.2, 51, 490, 65, 195],  # Chip 1 parameters
    [880, 310, 4.8, 49, 510, 55, 205]  # Chip 2 parameters
], columns=X.columns)

# Preprocess new data
new_data_scaled = loaded_scaler.transform(new_chip_parameters)
# Predict yield
predicted_yields = loaded_model.predict(new_data_scaled)

print("\nYield Prediction for New IC Samples:")
for i, yield_val in enumerate(predicted_yields):
    print(f"Chip {i + 1} Predicted Yield: {yield_val:.2%}")
print("\n✅ Model saved: chip_yield_model.pkl | Scaler saved: chip_scaler.pkl")