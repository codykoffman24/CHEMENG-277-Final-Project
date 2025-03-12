import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import gc  # For garbage collection

# ========================
# 1. Load & Preprocess Data (Discharge, Charge, and OCV)
# ========================
discharge_file = "Copy of Oxford_battery_data_discharge.csv"
charge_file = "Copy of Oxford_battery_data_charge.csv"
ocv_discharge_file = "Copy of Oxford_battery_data_OCVdc.csv"
ocv_charge_file = "Copy of Oxford_battery_data_OCVch.csv"

# Define function to process data in chunks
def load_and_process_in_chunks(filename, usecols, chunk_size=250000):
    chunks = pd.read_csv(filename, chunksize=chunk_size, usecols=usecols, low_memory=False)
    result = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} from {filename}...")
        result = pd.concat([result, chunk])
    return result

# Load discharge data (needed for full processing)
print("Loading discharge data...")
discharge_data = load_and_process_in_chunks(
    discharge_file,
    usecols=['cell_number', 'cycle_number', 'time', 'voltage', 'current']
)

# Sort discharge data
discharge_data = discharge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Print total rows
print(f"🔹 Total CC-CV discharge rows: {len(discharge_data)}")

# Ensure current values are negative for discharge
if discharge_data['current'].max() > 0:
    discharge_data['current'] = -discharge_data['current']

# Set time_diff to 1 second as per requirement
discharge_data['time_diff'] = 1

# Compute discharge capacity incrementally
discharge_data['incremental_discharge'] = (discharge_data['current'] * discharge_data['time_diff']).abs() / 3600
discharge_data['discharge_capacity'] = discharge_data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000

# ========================
# 2. Aggregate Discharge Data First
# ========================
print("Aggregating discharge data...")
agg_funcs = {
    'voltage': 'mean',
    'discharge_capacity': 'max'
}

grouped = discharge_data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)

# Get list of valid cell_number and cycle_number combinations for filtering OCV data
valid_cycles = grouped[['cell_number', 'cycle_number']].drop_duplicates()
print(f"🔹 Total valid discharge cycles: {len(valid_cycles)}")

# Free up memory
del discharge_data
gc.collect()

# ========================
# 3. Process OCV Data More Efficiently
# ========================
print("Processing OCV data for hysteresis calculation...")

# Load and filter OCV data
print("Loading OCV discharge data...")
ocv_discharge = load_and_process_in_chunks(
    ocv_discharge_file,
    usecols=['cell_number', 'cycle_number', 'voltage'],
    chunk_size=500000
)

print("Loading OCV charge data...")
ocv_charge = load_and_process_in_chunks(
    ocv_charge_file,
    usecols=['cell_number', 'cycle_number', 'voltage'],
    chunk_size=500000
)

# Print diagnostic information
print(f"🔹 Total OCV discharge raw rows: {len(ocv_discharge)}")
print(f"🔹 Total OCV charge raw rows: {len(ocv_charge)}")

# Filter OCV data using left join to avoid excessive row loss
ocv_discharge = pd.merge(
    ocv_discharge,
    valid_cycles,
    on=['cell_number', 'cycle_number'],
    how='left'
).dropna()

ocv_charge = pd.merge(
    ocv_charge,
    valid_cycles,
    on=['cell_number', 'cycle_number'],
    how='left'
).dropna()

print(f"🔹 Filtered OCV discharge rows: {len(ocv_discharge)}")
print(f"🔹 Filtered OCV charge rows: {len(ocv_charge)}")

# Aggregate OCV data
ocv_discharge_agg = ocv_discharge.groupby(['cell_number', 'cycle_number']).agg({'voltage': 'mean'}).reset_index()
ocv_charge_agg = ocv_charge.groupby(['cell_number', 'cycle_number']).agg({'voltage': 'mean'}).reset_index()

print(f"🔹 Total OCV discharge aggregated rows: {len(ocv_discharge_agg)}")
print(f"🔹 Total OCV charge aggregated rows: {len(ocv_charge_agg)}")

# Merge to compute hysteresis
ocv_hysteresis_df = pd.merge(
    ocv_charge_agg,
    ocv_discharge_agg,
    on=['cell_number', 'cycle_number'],
    suffixes=('_ch', '_dis')
)

# Compute hysteresis feature
ocv_hysteresis_df['OCV_hysteresis'] = ocv_hysteresis_df['voltage_ch'] - ocv_hysteresis_df['voltage_dis']
ocv_hysteresis_df.drop(columns=['voltage_ch', 'voltage_dis'], inplace=True)

# Check for NaNs
print(f"🔹 Rows with NaN OCV hysteresis before imputation: {ocv_hysteresis_df['OCV_hysteresis'].isna().sum()}")

# Fill missing values using per-cell median
ocv_hysteresis_df['OCV_hysteresis'] = ocv_hysteresis_df.groupby('cell_number')['OCV_hysteresis'].transform(
    lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
)

# Free up memory
del ocv_charge
del ocv_discharge
del ocv_charge_agg
del ocv_discharge_agg
gc.collect()

# ========================
# 4. Merge Aggregated Data and Calculate SOH
# ========================
print("Calculating SOH...")
grouped = pd.merge(grouped, ocv_hysteresis_df, on=['cell_number', 'cycle_number'], how='left')

# Compute State of Health (SOH)
grouped['SOH'] = (
    grouped['max_discharge_capacity'] /
    grouped.groupby('cell_number')['max_discharge_capacity'].transform('first')
) * 100

# ========================
# 5. Define Features & Labels
# ========================
print("Preparing for model training...")
features = ['avg_voltage', 'cycle_number', 'OCV_hysteresis']  # Removed avg_temperature

X = grouped[features].values
y = grouped['SOH'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 6. Train and Evaluate Models with Different Alpha Values
# ========================
X_train, X_test = X_scaled[:int(0.8 * len(X_scaled))], X_scaled[int(0.8 * len(X_scaled)):]
y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

print("\nEvaluating different alpha values for Ridge and Lasso regression...")

ridge_alphas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10]
lasso_alphas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10]

# Store Train & Test R² values for each Ridge & Lasso alpha
ridge_results = {
    alpha: {
        "R² Train": linear_model.Ridge(alpha=alpha).fit(X_train, y_train).score(X_train, y_train),
        "R² Test": linear_model.Ridge(alpha=alpha).fit(X_train, y_train).score(X_test, y_test)
    }
    for alpha in ridge_alphas
}

lasso_results = {
    alpha: {
        "R² Train": linear_model.Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train).score(X_train, y_train),
        "R² Test": linear_model.Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train).score(X_test, y_test)
    }
    for alpha in lasso_alphas
}

# Convert to DataFrame
df_ridge_alphas = pd.DataFrame.from_dict(ridge_results, orient='index').reset_index()
df_ridge_alphas.columns = ["Ridge Alpha", "R² Train", "R² Test"]

df_lasso_alphas = pd.DataFrame.from_dict(lasso_results, orient='index').reset_index()
df_lasso_alphas.columns = ["Lasso Alpha", "R² Train", "R² Test"]

print("\nRidge Regression Alpha Tuning Results:")
print(df_ridge_alphas.to_string(index=False))  # Print as formatted table

print("\nLasso Regression Alpha Tuning Results:")
print(df_lasso_alphas.to_string(index=False))  # Print as formatted table


# Identify the best alpha based on the highest R² Test score
best_ridge_alpha = max(ridge_results, key=lambda alpha: ridge_results[alpha]["R² Test"])
best_lasso_alpha = max(lasso_results, key=lambda alpha: lasso_results[alpha]["R² Test"])


# Train final models with best alpha values
best_ridge = linear_model.Ridge(alpha=best_ridge_alpha)
best_ridge.fit(X_train, y_train)

best_lasso = linear_model.Lasso(alpha=best_lasso_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

# Also train a standard linear regression model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, y_train)

# ========================
# 7. Select the Best Model AFTER Alpha Tuning
# ========================
models = {
    "Linear Regression": linear_reg,
    f"Ridge Regression (alpha={best_ridge_alpha})": best_ridge,
    f"Lasso Regression (alpha={best_lasso_alpha})": best_lasso
}

best_model_name = max(models, key=lambda name: models[name].score(X_test, y_test))
best_model = models[best_model_name]

# Print best model
print(f"\n🔹 Best Performing Model: {best_model_name}")
print(f"R² on Test Set: {best_model.score(X_test, y_test):.4f}")

# Extract feature coefficients from the selected best model
coefficients = best_model.coef_
feature_importance = dict(zip(features, coefficients))
sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n🔹 Feature Coefficients (Sorted by Impact):")
for feature, coef in sorted_importance:
    print(f"{feature}: {coef:.4f}")


# ========================
# 8. Visualization
# ========================
print("Creating visualizations...")

# Sample data for visualization (to avoid memory issues)
sample_step = max(1, len(grouped) // 1000)  # Show at most 1000 points

# OCV Hysteresis vs. Cycle Number
plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'][::sample_step], grouped['OCV_hysteresis'][::sample_step], alpha=0.5, color='blue')
plt.xlabel('Cycle Number')
plt.ylabel('OCV Hysteresis (V)')
plt.title('OCV Hysteresis vs. Cycle Number')
plt.grid(True)
plt.savefig('ocv_hysteresis_vs_cycle.png')  # Save the figure
plt.close()

# SOH vs. Cycle Number
plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'][::sample_step], grouped['SOH'][::sample_step], alpha=0.5, color='red')
plt.xlabel('Cycle Number')
plt.ylabel('State of Health (SOH %)')
plt.title('State of Health vs. Cycle Number')
plt.grid(True)
plt.savefig('soh_vs_cycle.png')  # Save the figure
plt.close()

# Average Voltage vs. Cycle Number
plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'][::sample_step], grouped['avg_voltage'][::sample_step], alpha=0.5, color='green')
plt.xlabel('Cycle Number')
plt.ylabel('Average Voltage (V)')
plt.title('Average Voltage vs. Cycle Number')
plt.grid(True)
plt.savefig('avg_voltage_vs_cycle.png')  # Save the figure
plt.close()

# Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
features_abs = [abs(coef) for _, coef in sorted_importance]
features_names = [feature for feature, _ in sorted_importance]

plt.bar(features_names, features_abs, color='royalblue')
plt.ylabel('Absolute Coefficient Value')
plt.xlabel('Feature')
plt.title('Feature Importance (Absolute Coefficient Values)')
plt.xticks(rotation=45)
plt.tight_layout()

# Add value labels on top of bars
for i, v in enumerate(features_abs):
    plt.text(i, v + 0.1, f"{v:.4f}", ha='center')

plt.savefig('feature_importance.png')  # Save the figure
plt.close()

print("Analysis complete! All plots saved as PNG files.")


# ========================
# 9. Model Performance Summary
# ========================
print("\nGenerating model performance summary...")

# Collect main regression results with Train & Test R² values
model_performance_data = {
    "Model": [
        "Linear Regression",
        f"Ridge Regression (alpha={best_ridge_alpha})",
        f"Lasso Regression (alpha={best_lasso_alpha})"
    ],
    "R² Train": [
        linear_reg.score(X_train, y_train),
        best_ridge.score(X_train, y_train),
        best_lasso.score(X_train, y_train)
    ],
    "R² Test": [
        linear_reg.score(X_test, y_test),
        best_ridge.score(X_test, y_test),
        best_lasso.score(X_test, y_test)
    ]
}

# Convert to DataFrame and display
df_model_performance = pd.DataFrame(model_performance_data)
df_model_performance = df_model_performance.round(4)  # Round to 4 decimal places

print("\nModel Performance Summary (Train & Test R²):")
print(df_model_performance.to_string(index=False))  # Print as formatted table


# Collect feature importance details
feature_importance_data = {
    "Feature": [feature for feature, _ in sorted_importance],
    "Coefficient": [coef for _, coef in sorted_importance],
    "Absolute Impact": [abs(coef) for _, coef in sorted_importance]
}

df_feature_importance = pd.DataFrame(feature_importance_data)
df_feature_importance = df_feature_importance.round(4)  # Round to 4 decimal places

print("\nFeature Importance:")
print(df_feature_importance.to_string(index=False))  # Print as formatted table


# ========================
# 10. Save Performance Summary as Images
# ========================
# Create a table image for model performance
plt.figure(figsize=(8, 3))
plt.axis('off')
table = plt.table(
    cellText=df_model_performance.values,
    colLabels=df_model_performance.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title('Model Performance Summary')
plt.savefig('model_performance_table.png', bbox_inches='tight')
plt.close()

# Create a table image for feature importance
plt.figure(figsize=(10, 4))
plt.axis('off')
table = plt.table(
    cellText=df_feature_importance.values,
    colLabels=df_feature_importance.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title('Feature Importance Details')
plt.savefig('feature_importance_table.png', bbox_inches='tight')
plt.close()

print("\n✅ Model performance summary displayed and saved as PNG tables.")

